"""
baselines/lsd/detector.py
=========================
LSD — Layer-wise Semantic Drift
Paper : arXiv:2510.04933 (2024)
Status: MOST SIMILAR prior work to LID — we MUST beat it on lead time

Core Idea:
    Measures how much the semantic representation of a token DRIFTS
    as it passes through successive transformer layers. High drift
    between layers = model hasn't settled on a stable representation
    = uncertain = likely to hallucinate.

    Unlike LID which perturbs hidden states, LSD measures natural
    layer-to-layer variation without any perturbation.

    Score per token t at position p:
        drift(l, t) = 1 - cosine(h_l(t), h_{l+1}(t))
        LSD_score(t) = weighted_mean( drift(l,t) for l in selected_layers )

    Higher score = more semantic instability = more likely hallucination

Key Difference from LID:
    LSD  : measures drift between CONSECUTIVE layers (natural variation)
    LID  : measures response to PERTURBATION (controlled sensitivity test)

    LID advantage: LID's perturbation isolates sensitivity from
    normal representation evolution — more targeted signal.

Target reproduction numbers (arXiv:2510.04933):
    AUROC ≈ 0.63–0.67 on TruthfulQA with Llama-7B

Author : MIT LID Research Team
Week   : 2 — Baseline Implementation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from baselines.base import BaseDetector, DetectorConfig, DetectorOutput


from dataclasses import dataclass as _dataclass

@_dataclass
class LSDConfig(DetectorConfig):
    """LSD-specific configuration."""
    layer_pairs: Optional[list] = None
    use_depth_weighting: bool = True
    aggregation: str = "mean"


class LSDDetector(BaseDetector):
    """
    LSD hallucination detector.

    Measures semantic drift between consecutive transformer layers.
    Tokens with high layer-to-layer drift are flagged as likely hallucinations.

    Key insight: before hallucination, the model's internal representation
    keeps changing across layers (hasn't converged to a stable answer).
    For factual tokens, representations stabilize in middle-to-late layers.

    Output contract: scores [B, T], higher = more likely hallucinated.
    """

    def __init__(self, config: Optional[LSDConfig] = None):
        if config is None:
            config = LSDConfig(name="lsd")
        super().__init__(config)
        self.lsd_config = config

    @property
    def name(self) -> str:
        return "lsd"

    def _cosine_drift(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute semantic drift between two hidden states.

        drift = 1 - cosine_similarity(h1, h2)

        Range: [0, 2]
            0 = no drift (identical representations)
            1 = orthogonal (complete drift)
            2 = antiparallel (opposite direction)

        Args:
            h1, h2 : Hidden states [B, T, d_model]

        Returns:
            drift : [B, T]
        """
        h1_norm = F.normalize(h1, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)
        cosine_sim = (h1_norm * h2_norm).sum(dim=-1)  # [B, T]
        return 1.0 - cosine_sim  # drift ∈ [0, 2]

    def score(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DetectorOutput:
        """
        Score each token using LSD layer-drift method.

        For each token position t:
            1. Capture hidden states at ALL layers via hooks
            2. Compute drift(l, l+1) for each consecutive layer pair
            3. Aggregate drift scores across layers
            4. score(t) = aggregated drift

        Args:
            model      : HuggingFace CausalLM
            tokenizer  : HuggingFace tokenizer
            input_ids  : [B, T]
            attention_mask : optional [B, T]

        Returns:
            DetectorOutput with scores [B, T]
        """
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        n_layers = model.config.num_hidden_layers
        all_hidden_states = {}

        # ── Register hooks on ALL layers ─────────────────────────────────
        hooks = []
        for layer_idx in range(n_layers):
            def make_hook(idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    all_hidden_states[idx] = h.detach()  # [B, T, d_model]
                return hook
            h = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            hooks.append(h)

        try:
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            for h in hooks:
                h.remove()

        # ── Compute layer-to-layer drift ──────────────────────────────────
        # Determine which layer pairs to use
        if self.lsd_config.layer_pairs is not None:
            pairs = self.lsd_config.layer_pairs
        else:
            pairs = [(l, l + 1) for l in range(n_layers - 1)]

        drift_per_pair = []  # list of [B, T] tensors

        for l1, l2 in pairs:
            h1 = all_hidden_states[l1].float()   # [B, T, d_model]
            h2 = all_hidden_states[l2].float()
            drift = self._cosine_drift(h1, h2)   # [B, T]
            drift_per_pair.append(drift)

        # ── Aggregate across layer pairs ──────────────────────────────────
        drift_stack = torch.stack(drift_per_pair, dim=0)  # [n_pairs, B, T]

        if self.lsd_config.aggregation == "mean":
            scores = drift_stack.mean(dim=0)

        elif self.lsd_config.aggregation == "max":
            scores = drift_stack.max(dim=0).values

        elif self.lsd_config.aggregation == "weighted_mean":
            # Weight later pairs more (deeper layers more semantically meaningful)
            if self.lsd_config.use_depth_weighting:
                n_pairs = len(pairs)
                # Linear depth weighting: last pair gets weight n_pairs, first gets 1
                weights = torch.linspace(1.0, float(n_pairs), n_pairs)
                weights = weights / weights.sum()
                weights = weights.to(drift_stack.device)
                scores = (drift_stack * weights[:, None, None]).sum(dim=0)
            else:
                scores = drift_stack.mean(dim=0)
        else:
            scores = drift_stack.mean(dim=0)

        # Decode tokens
        tokens = [
            [tokenizer.decode([tok_id]) for tok_id in seq]
            for seq in input_ids.cpu().tolist()
        ]

        return DetectorOutput(
            scores=scores.cpu(),
            tokens=tokens,
            metadata={
                "n_layer_pairs": len(pairs),
                "aggregation": self.lsd_config.aggregation,
                "layer_range": f"0-{n_layers-1}",
                "drift_per_pair_mean": float(drift_stack.mean().item()),
            },
        )

    def score_generated(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:
        """Generate response and score each generated token."""
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        result = self.score(model, tokenizer, gen_ids)
        prompt_len = inputs["input_ids"].shape[1]

        return {
            "generated_text": tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True
            ),
            "token_scores": result.scores[0, prompt_len:].tolist(),
            "tokens": result.tokens[0][prompt_len:],
            "prompt_len": prompt_len,
        }
