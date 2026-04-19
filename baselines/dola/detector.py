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

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from baselines.base import BaseDetector, DetectorConfig, DetectorOutput


from dataclasses import dataclass as _dataclass

@_dataclass
class DoLAConfig(DetectorConfig):
    """DoLA-specific configuration."""
    premature_layer_ratio: float = 0.4
    premature_layer_idx: Optional[int] = None
    temperature: float = 1.0


class DoLADetector(BaseDetector):
    """
    DoLA hallucination detector.

    Scores each token by measuring how much the vocabulary distribution
    changes between an early (premature) layer and the final layer.
    High change = high uncertainty = likely hallucination.

    Output contract: scores [B, T], higher = more likely hallucinated.
    """

    def __init__(self, config: Optional[DoLAConfig] = None):
        if config is None:
            config = DoLAConfig(name="dola")
        super().__init__(config)
        self.dola_config = config

    @property
    def name(self) -> str:
        return "dola"

    def _get_premature_layer(self, n_layers: int) -> int:
        """Determine premature layer index from config."""
        if self.dola_config.premature_layer_idx is not None:
            return self.dola_config.premature_layer_idx
        return int(n_layers * self.dola_config.premature_layer_ratio)

    def _jsd(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Jensen-Shannon Divergence between distributions p and q.

        JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)
        where m = 0.5 * (p + q)

        Range: [0, 1] (when using log base 2)
        0 = identical distributions
        1 = completely different distributions

        Args:
            p, q : Probability distributions [..., vocab_size]

        Returns:
            JSD scores [...]
        """
        m = 0.5 * (p + q)
        # KL(p||m) = sum(p * log(p/m))
        kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
        kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
        return 0.5 * kl_pm + 0.5 * kl_qm

    def score(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DetectorOutput:
        """
        Score each token using DoLA layer-contrast method.

        For each token position t:
            1. Capture hidden state at premature layer L_pre
            2. Project through lm_head → premature logits
            3. Get final layer logits (standard forward pass)
            4. Compute JSD(softmax(premature), softmax(final))
            5. score(t) = JSD value

        Args:
            model      : HuggingFace CausalLM (Llama/Mistral)
            tokenizer  : HuggingFace tokenizer
            input_ids  : [B, T] input token IDs
            attention_mask : optional [B, T]

        Returns:
            DetectorOutput with scores [B, T]
        """
        import time
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        n_layers = model.config.num_hidden_layers
        premature_layer = self._get_premature_layer(n_layers)

        # Storage for premature hidden states
        premature_hidden = {}

        def premature_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            premature_hidden["h"] = h.detach()  # [B, T, d_model]

        # Register hook on premature layer
        hook_handle = model.model.layers[premature_layer].register_forward_hook(
            premature_hook
        )

        t0 = time.perf_counter()

        try:
            with torch.no_grad():
                # Full forward pass — captures premature hidden states via hook
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,  # we use hooks instead
                )
        finally:
            hook_handle.remove()

        inference_time = time.perf_counter() - t0

        # ── Final layer logits ────────────────────────────────────────────
        final_logits = outputs.logits  # [B, T, vocab_size]
        final_probs = F.softmax(
            final_logits / self.dola_config.temperature, dim=-1
        )

        # ── Premature layer logits (project through lm_head) ─────────────
        h_premature = premature_hidden["h"]  # [B, T, d_model]

        # Apply layer norm before lm_head (important for correct logits)
        try:
            h_normed = model.model.norm(h_premature)
        except AttributeError:
            h_normed = h_premature  # fallback if norm not accessible

        premature_logits = model.lm_head(h_normed)  # [B, T, vocab_size]
        premature_probs = F.softmax(
            premature_logits / self.dola_config.temperature, dim=-1
        )

        # ── JSD score per token ───────────────────────────────────────────
        scores = self._jsd(premature_probs, final_probs)  # [B, T]

        # Decode tokens for output
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
            overhead_ratio=None,  # measured externally
        )

    def score_generated(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:
        """
        Generate a response AND score each generated token.

        This is the primary evaluation method:
        1. Generate response token by token
        2. Score each new token using DoLA

        Returns dict with:
            - generated_text : str
            - token_scores   : list[float]  per generated token
            - tokens         : list[str]    decoded generated tokens
        """
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate full sequence
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Score the full sequence (prompt + response)
        result = self.score(model, tokenizer, gen_ids)

        # Extract only the generated tokens (not the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        gen_scores = result.scores[0, prompt_len:].tolist()
        gen_tokens = result.tokens[0][prompt_len:]
        gen_text = tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True
        )

        return {
            "generated_text": gen_text,
            "token_scores": gen_scores,
            "tokens": gen_tokens,
            "prompt_len": prompt_len,
            "premature_layer": result.metadata["premature_layer"],
        }
