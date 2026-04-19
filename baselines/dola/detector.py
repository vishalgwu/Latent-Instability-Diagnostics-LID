"""
baselines/dola/detector.py
==========================
DoLA — Decoding by Contrasting Layers
Paper : Chuang et al., ICLR 2024
Link  : https://arxiv.org/abs/2309.03883

Core Idea:
    For each generated token, compute the Jensen-Shannon Divergence (JSD)
    between the vocabulary distribution from a "premature" (early) layer
    and the final layer. High JSD = the model changed its mind significantly
    between layers = uncertain = likely to hallucinate.

    Score per token:
        score(t) = JSD( p_premature(t) || p_final(t) )

    Where:
        p_final(t)    = softmax(logits from last layer at token t)
        p_premature(t) = softmax(logits from early layer L_pre at token t)
        JSD           = symmetric version of KL divergence, range [0, 1]

    Higher score = more disagreement between layers = more likely hallucination

Target reproduction numbers (Chuang et al., TruthfulQA, Llama-7B):
    AUROC ≈ 0.65–0.68  (we need to be within ±2pp)

Implementation notes:
    - We use the layer-wise logit projection via the model's lm_head
    - Premature layer = layer at 40% depth (tunable)
    - No per-model fine-tuning required

Author : MIT LID Research Team
Week   : 2 — Baseline Implementation
"""
"""
baselines/dola/detector.py
==========================
FIXED VERSION (NaN-safe DoLA)

Changes made:
1. Stable softmax inputs (fp32 + NaN cleaning)
2. Safe Jensen-Shannon Divergence (no inf - inf)
3. Hook safety for premature layer
4. Final NaN/Inf protection (research-safe evaluation)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from baselines.base import BaseDetector, DetectorConfig, DetectorOutput


from dataclasses import dataclass as _dataclass


@_dataclass
class DoLAConfig(DetectorConfig):
    premature_layer_ratio: float = 0.4
    premature_layer_idx: Optional[int] = None
    temperature: float = 1.0


class DoLADetector(BaseDetector):

    def __init__(self, config: Optional[DoLAConfig] = None):
        if config is None:
            config = DoLAConfig(name="dola")
        super().__init__(config)
        self.dola_config = config

    @property
    def name(self) -> str:
        return "dola"

    def _get_premature_layer(self, n_layers: int) -> int:
        if self.dola_config.premature_layer_idx is not None:
            return self.dola_config.premature_layer_idx
        return int(n_layers * self.dola_config.premature_layer_ratio)

    # =========================
    # FIXED JSD (CORE FIX)
    # =========================
    def _jsd(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
        """
        Numerically stable JSD
        """

        # Clamp BEFORE log
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)

        # Renormalize (IMPORTANT for stability)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        m = 0.5 * (p + q)
        m = m.clamp(min=eps)

        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)

        jsd = 0.5 * (kl_pm + kl_qm)

        # FINAL SAFETY (prevents NaN poisoning evaluation)
        return torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)

    def score(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DetectorOutput:

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        n_layers = model.config.num_hidden_layers
        premature_layer = self._get_premature_layer(n_layers)

        premature_hidden = {}

        def premature_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # 🔥 FIX: remove NaNs early
            premature_hidden["h"] = torch.nan_to_num(h.detach())

        # hook
        hook_handle = model.model.layers[premature_layer].register_forward_hook(
            premature_hook
        )

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            hook_handle.remove()

        # =========================
        # FINAL LOGITS (FIXED)
        # =========================
        final_logits = outputs.logits.float()
        final_logits = torch.nan_to_num(final_logits)

        final_probs = F.softmax(
            final_logits / self.dola_config.temperature,
            dim=-1
        )

        # =========================
        # PREMATURE LOGITS (FIXED)
        # =========================
        h_premature = premature_hidden["h"].float()
        h_premature = torch.nan_to_num(h_premature)

        try:
            h_normed = model.model.norm(h_premature)
        except Exception:
            h_normed = h_premature

        premature_logits = model.lm_head(h_normed)
        premature_logits = torch.nan_to_num(premature_logits)

        premature_probs = F.softmax(
            premature_logits / self.dola_config.temperature,
            dim=-1
        )

        # =========================
        # SCORE
        # =========================
        scores = self._jsd(premature_probs, final_probs)

        tokens = [
            [tokenizer.decode([tok_id]) for tok_id in seq]
            for seq in input_ids.cpu().tolist()
        ]

        return DetectorOutput(
            scores=scores.cpu(),
            tokens=tokens,
            metadata={
                "premature_layer": premature_layer,
                "n_layers": n_layers,
                "premature_layer_ratio": self.dola_config.premature_layer_ratio,
            },
        )

    def score_generated(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:

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