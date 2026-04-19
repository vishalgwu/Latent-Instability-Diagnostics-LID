"""
baselines/ssp/detector.py
=========================
SSP — Semantic Self-consistency via input Perturbation
Paper : Luo et al., June 2025
Status: CLOSEST competitor to LID

Core Idea:
    Perturbs the INPUT (token embeddings) and measures how much the
    output probability distribution changes. High sensitivity to input
    perturbation = unstable prediction = likely hallucination.

    CRITICAL DIFFERENCE from LID:
        SSP  : perturbs INPUT embeddings → measures output distribution shift
        LID  : perturbs HIDDEN STATES    → measures internal representation shift

    LID advantage:
        - More targeted: perturbation at the exact layer where instability occurs
        - Computes I(t) AND q(t) — SSP only gets magnitude shift
        - Can detect instability token-by-token during generation
        - SSP needs a full second forward pass; LID uses partial pass

    Score per token t:
        SSP_score(t) = JSD( p_clean(t) || p_perturbed(t) )

    Where:
        p_clean(t)     = output prob distribution with original embeddings
        p_perturbed(t) = output prob distribution with noisy embeddings
        JSD            = Jensen-Shannon Divergence ∈ [0, 1]

    Higher score = more sensitive to input noise = likely hallucination

Target reproduction numbers (Luo 2025):
    AUROC ≈ 0.64–0.68 on TruthfulQA
    NOTE: Paper has some ambiguous hyperparameters — document every assumption

Hyperparameter assumptions (document in fidelity report):
    - alpha = 0.05 (5% of embedding RMS — same as LID default for fair comparison)
    - Single perturbation (not ensemble)
    - Perturbation applied at embedding layer output

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
class SSPConfig(DetectorConfig):
    """SSP-specific configuration."""
    alpha: float = 0.05
    n_samples: int = 1
    perturb_seed: int = 42


class SSPDetector(BaseDetector):
    """
    SSP hallucination detector.

    Perturbs INPUT embeddings and measures how much the output
    probability distribution changes under that perturbation.

    This is the closest published method to LID:
    - Both use perturbation + distribution shift measurement
    - SSP operates on input space; LID operates on hidden state space
    - SSP is our primary comparison point for the "perturbation" novelty claim

    Output contract: scores [B, T], higher = more likely hallucinated.
    """

    def __init__(self, config: Optional[SSPConfig] = None):
        if config is None:
            config = SSPConfig(name="ssp")
        super().__init__(config)
        self.ssp_config = config

    @property
    def name(self) -> str:
        return "ssp"

    def _rms(self, h: torch.Tensor) -> torch.Tensor:
        """RMS(h) = sqrt(mean(h²)) across last dimension."""
        return h.pow(2).mean(dim=-1, keepdim=True).sqrt()

    def _inject_noise(
        self,
        embeddings: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> torch.Tensor:
        """
        Inject Gaussian noise into input embeddings.

        δ ~ N(0, σ²I),  σ = alpha * RMS(embedding)
        h_pert = h + δ

        Deterministic: same (seed, alpha, h) → same δ.
        """
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(seed)
        sigma = alpha * self._rms(embeddings)   # [B, T, 1]
        delta = torch.randn_like(embeddings, generator=generator) * sigma
        return embeddings + delta

    def _jsd(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Jensen-Shannon Divergence.
        Range [0, 1]. 0 = identical, 1 = completely different.
        """
        m = 0.5 * (p + q)
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
        Score each token using SSP input-perturbation method.

        For each token position t:
            1. Run clean forward pass → p_clean(t)
            2. Perturb input embeddings with Gaussian noise
            3. Run perturbed forward pass → p_perturbed(t)
            4. score(t) = JSD(p_clean(t), p_perturbed(t))

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

        # ── Clean forward pass ────────────────────────────────────────────
        with torch.no_grad():
            # Get clean embeddings first
            clean_embeds = model.model.embed_tokens(input_ids)  # [B, T, d_model]

            clean_outputs = model(
                inputs_embeds=clean_embeds,
                attention_mask=attention_mask,
            )
            clean_logits = clean_outputs.logits   # [B, T, vocab]
            clean_probs = F.softmax(clean_logits.float(), dim=-1)

        # ── Perturbed forward pass(es) ────────────────────────────────────
        all_pert_probs = []

        for sample_idx in range(self.ssp_config.n_samples):
            seed = self.ssp_config.perturb_seed + sample_idx

            with torch.no_grad():
                pert_embeds = self._inject_noise(
                    clean_embeds.float(),
                    alpha=self.ssp_config.alpha,
                    seed=seed,
                ).to(clean_embeds.dtype)

                pert_outputs = model(
                    inputs_embeds=pert_embeds,
                    attention_mask=attention_mask,
                )
                pert_probs = F.softmax(pert_outputs.logits.float(), dim=-1)
                all_pert_probs.append(pert_probs)

        # Average over samples if n_samples > 1
        avg_pert_probs = torch.stack(all_pert_probs, dim=0).mean(dim=0)

        # ── JSD score per token ───────────────────────────────────────────
        scores = self._jsd(clean_probs, avg_pert_probs)  # [B, T]

        # Decode tokens
        tokens = [
            [tokenizer.decode([tok_id]) for tok_id in seq]
            for seq in input_ids.cpu().tolist()
        ]

        return DetectorOutput(
            scores=scores.cpu(),
            tokens=tokens,
            metadata={
                "alpha": self.ssp_config.alpha,
                "n_samples": self.ssp_config.n_samples,
                "perturbation_level": "input_embeddings",
                "assumption": "alpha=0.05, single perturbation (Luo 2025 ambiguous)",
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
