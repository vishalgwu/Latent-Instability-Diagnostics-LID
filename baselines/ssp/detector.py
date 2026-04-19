"""
baselines/ssp/detector.py
=========================
SSP — Semantic Self-consistency via input Perturbation
Paper : Luo et al., June 2025
Status: CLOSEST competitor to LID

Core Idea:
    Perturbs the INPUT (token embeddings) and measures how much the
    output probability distribution changes.  High sensitivity to input
    perturbation = unstable prediction = likely hallucination.

    CRITICAL DIFFERENCE from LID:
        SSP  : perturbs INPUT embeddings  → measures output distribution shift
        LID  : perturbs HIDDEN STATES     → measures internal representation shift

    LID advantage:
        - More targeted: perturbation at the exact layer where instability occurs
        - Computes I(t) AND q(t) — SSP only gets magnitude shift
        - Can detect instability token-by-token during generation
        - SSP needs a full second forward pass; LID uses a partial pass

    Score per token t:
        SSP_score(t) = JSD( p_clean(t) || p_perturbed(t) )

    Where:
        p_clean(t)     = output prob distribution with original embeddings
        p_perturbed(t) = output prob distribution with noisy embeddings
        JSD            = Jensen-Shannon Divergence ∈ [0, 1]

    Higher score = more sensitive to input noise = likely hallucination

Target reproduction numbers (Luo 2025):
    AUROC ≈ 0.64–0.68 on TruthfulQA
    NOTE: Paper has ambiguous hyperparameters — all assumptions documented below

Hyperparameter assumptions (document in fidelity report):
    - alpha = 0.05  (5% of embedding RMS — same as LID default for fair comparison)
    - n_samples = 1 (single perturbation — paper is ambiguous on ensemble size)
    - Perturbation applied at embedding layer output (before first transformer block)

Author : MIT LID Research Team
Week   : 2 — Baseline Implementation
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
class SSPConfig(DetectorConfig):
    """SSP-specific configuration."""
    alpha: float = 0.05        # noise magnitude relative to embedding RMS
    n_samples: int = 1         # number of perturbation samples (paper ambiguous)
    perturb_seed: int = 42     # base seed; sample i uses seed + i


class SSPDetector(BaseDetector):
    """
    SSP hallucination detector.

    Perturbs INPUT embeddings and measures how much the output
    probability distribution changes under that perturbation.

    This is the closest published method to LID:
    - Both use perturbation + distribution shift measurement
    - SSP operates on input space; LID operates on hidden state space
    - SSP is the primary comparison point for the 'perturbation' novelty claim

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

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _rms(self, h: torch.Tensor) -> torch.Tensor:
        """
        Root-mean-square of h across the last dimension.
        RMS(h) = sqrt( mean(h²) )

        Returns shape [..., 1] for broadcasting with h.
        """
        return h.pow(2).mean(dim=-1, keepdim=True).sqrt()

    def _inject_noise(
        self,
        embeddings: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> torch.Tensor:
        """
        Add scaled Gaussian noise to embeddings.

        δ ~ N(0, σ²I),   σ = alpha × RMS(embedding)
        h_pert = h + δ

        Deterministic: the same (seed, alpha, embeddings) always produces
        the same δ, enabling exact reproducibility across runs.

        Args:
            embeddings : float32 tensor [B, T, d_model]
            alpha      : noise scale relative to embedding magnitude
            seed       : RNG seed for reproducibility

        Returns:
            Perturbed embeddings, same shape and dtype as input
        """
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(seed)
        sigma = alpha * self._rms(embeddings)            # [B, T, 1]
        delta = torch.randn_like(embeddings, generator=generator) * sigma
        return embeddings + delta

    def _jsd(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Numerically stable Jensen-Shannon Divergence.

        JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m),  m = 0.5*(p+q)
        Range: [0, ln2] ≈ [0, 0.693]

        Args:
            p, q : Probability distributions [..., vocab_size], float32
            eps  : Floor for probability values before log

        Returns:
            JSD scores [...]
        """
        # Sanitise inputs
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        # Floor → renormalize → mixture
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        m = (0.5 * (p + q)).clamp(min=eps)

        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)

        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Safe logits → probabilities.
        Handles overflow the same way as DoLA for consistent treatment.
        """
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits.clamp(-50.0, 50.0)
        probs  = F.softmax(logits, dim=-1)
        vocab  = probs.shape[-1]
        return torch.nan_to_num(probs, nan=1.0 / vocab)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

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
            (Optionally averaged over n_samples perturbations)

        Args:
            model          : HuggingFace CausalLM
            tokenizer      : HuggingFace tokenizer
            input_ids      : [B, T]
            attention_mask : optional [B, T]

        Returns:
            DetectorOutput with scores [B, T]
        """
        device    = next(model.parameters()).device
        input_ids = input_ids.to(device)
        model_dtype = next(model.parameters()).dtype

        # ── Clean forward pass ────────────────────────────────────────────
        with torch.no_grad():
            # Get embeddings in float32 for noise injection
            clean_embeds_fp32 = model.model.embed_tokens(input_ids).float()

            clean_outputs = model(
                inputs_embeds=clean_embeds_fp32.to(model_dtype),
                attention_mask=attention_mask,
            )
            clean_probs = self._logits_to_probs(clean_outputs.logits)

        # ── Perturbed forward pass(es) ────────────────────────────────────
        pert_probs_list = []

        for sample_idx in range(self.ssp_config.n_samples):
            seed = self.ssp_config.perturb_seed + sample_idx

            with torch.no_grad():
                pert_embeds_fp32 = self._inject_noise(
                    clean_embeds_fp32,
                    alpha=self.ssp_config.alpha,
                    seed=seed,
                )
                pert_outputs = model(
                    inputs_embeds=pert_embeds_fp32.to(model_dtype),
                    attention_mask=attention_mask,
                )
                pert_probs_list.append(
                    self._logits_to_probs(pert_outputs.logits)
                )

        # Average over samples if n_samples > 1
        avg_pert_probs = torch.stack(pert_probs_list, dim=0).mean(dim=0)

        # ── JSD score per token ───────────────────────────────────────────
        scores = self._jsd(clean_probs, avg_pert_probs)   # [B, T]

        # ── Decode tokens ─────────────────────────────────────────────────
        tokens = [
            [tokenizer.decode([tok_id]) for tok_id in seq]
            for seq in input_ids.cpu().tolist()
        ]

        return DetectorOutput(
            scores=scores.cpu(),
            tokens=tokens,
            metadata={
                "alpha":              self.ssp_config.alpha,
                "n_samples":          self.ssp_config.n_samples,
                "perturbation_level": "input_embeddings",
                "assumption":         (
                    "alpha=0.05, single perturbation "
                    "(Luo 2025 hyperparameters ambiguous — see fidelity report)"
                ),
            },
        )

    def score_generated(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> dict:
        """
        Generate response and score each generated token.

        Returns:
            generated_text : str
            token_scores   : list[float]  — one score per generated token
            tokens         : list[str]    — decoded generated tokens
            prompt_len     : int          — number of prompt tokens (for slicing)
        """
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        result     = self.score(model, tokenizer, gen_ids)
        prompt_len = inputs["input_ids"].shape[1]

        return {
            "generated_text": tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True
            ),
            "token_scores":  result.scores[0, prompt_len:].tolist(),
            "tokens":        result.tokens[0][prompt_len:],
            "prompt_len":    prompt_len,
        }