"""
baselines/dola/detector.py
==========================
DoLA — Decoding by Contrasting Layers
Paper : Chuang et al., ICLR 2024
Link  : https://arxiv.org/abs/2309.03883

Core Idea:
    For each generated token, compute the Jensen-Shannon Divergence (JSD)
    between the vocabulary distribution from a "premature" (early) layer
    and the final layer.

    Score per token:
        score(t) = JSD( p_premature(t) || p_final(t) )

    Where:
        p_final(t)     = softmax(logits from last layer at token t)
        p_premature(t) = softmax(logits from early layer L_pre at token t)
        JSD            = symmetric KL divergence, range [0, 1]

    Higher score = more disagreement between layers = more likely hallucination

Target reproduction numbers (Chuang et al., TruthfulQA, Llama-7B):
    AUROC ≈ 0.65–0.68  (we need to be within ±2pp)

Implementation notes:
    - We use the layer-wise logit projection via the model's lm_head
    - Premature layer = layer at 40% depth (tunable via config)
    - No per-model fine-tuning required

NaN Root Cause & Fix (documented for reproducibility):
    Intermediate hidden states at premature_layer are captured BEFORE the
    final LayerNorm normalises them.  Projecting them through lm_head
    (calibrated for post-norm representations) produces logits that can
    reach ±1e5+ in fp16/bf16.

    torch.nan_to_num() with DEFAULT posinf converts inf → 3.4e38, but
    exp(3.4e38) overflows back to inf inside softmax → NaN probabilities.

    Fix applied in four layers:
      1. Hook casts to float32 immediately (prevents fp16 inf accumulation)
      2. model.model.norm + model.lm_head temporarily promoted to float32
         for the premature projection, then restored to original dtype
      3. nan_to_num with EXPLICIT posinf=50 / neginf=-50 on logits
         (NOT the default 3.4e38 which still overflows in exp)
      4. Hard clamp to [-50, 50] before softmax
         (exp(50) ≈ 5e21 — well within float32 range)
      5. nan_to_num after softmax as final safety net (uniform fallback)

Author : MIT LID Research Team
Week   : 2 — Baseline Implementation
"""

from __future__ import annotations

import time
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

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_premature_layer(self, n_layers: int) -> int:
        """Return premature layer index from config."""
        if self.dola_config.premature_layer_idx is not None:
            return self.dola_config.premature_layer_idx
        return int(n_layers * self.dola_config.premature_layer_ratio)

    def _safe_softmax(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Logits → probabilities with full overflow protection.

        Steps:
          1. Cast to float32 (no-op if already fp32)
          2. nan_to_num with EXPLICIT bounds — default posinf=3.4e38 still
             overflows inside exp(); capping at 50 keeps exp(50) ≈ 5e21
          3. Hard clamp (redundant safety; catches any edge case from step 2)
          4. F.softmax (internally uses x-max normalisation, so no overflow)
          5. nan_to_num after softmax — if an entire row was all-zero input
             (degenerate hidden state), the output would be NaN; replace with
             uniform distribution rather than propagating NaN downstream
        """
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits.clamp(-50.0, 50.0)
        probs  = F.softmax(logits / temperature, dim=-1)
        vocab  = probs.shape[-1]
        probs  = torch.nan_to_num(probs, nan=1.0 / vocab)
        return probs

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

        Stability measures applied in order:
          1. nan_to_num on inputs — stops NaN probs from propagating
          2. clamp(min=eps) — prevents log(0) = -inf
          3. renormalize — clamp shifts probability mass; dividing by sum
             restores the valid probability simplex before the KL computation
          4. nan_to_num on output — final catch-all

        Args:
            p, q : Probability distributions [..., vocab_size], float32
            eps  : Floor for probability values before log

        Returns:
            JSD scores [...]
        """
        # 1. Sanitise inputs
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Floor (must happen before renorm, not after)
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)

        # 3. Renormalize — clamp shifts mass, KL formula requires sum == 1
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # 4. Mixture distribution
        m = (0.5 * (p + q)).clamp(min=eps)

        # 5. KL divergences in float32 (both operands already fp32)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)

        jsd = 0.5 * (kl_pm + kl_qm)

        # 6. Final safety net
        return torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)

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
        Score each token using DoLA layer-contrast method.

        For each token position t:
            1. Capture hidden state at premature layer L_pre via hook
            2. Project through lm_head in float32 → premature logits
            3. Get final layer logits (standard forward pass)
            4. score(t) = JSD(softmax(premature), softmax(final))

        Args:
            model          : HuggingFace CausalLM (Llama / Mistral / etc.)
            tokenizer      : HuggingFace tokenizer
            input_ids      : [B, T] input token IDs
            attention_mask : optional [B, T]

        Returns:
            DetectorOutput with scores [B, T]
        """
        device    = next(model.parameters()).device
        input_ids = input_ids.to(device)

        n_layers    = model.config.num_hidden_layers
        pre_layer   = self._get_premature_layer(n_layers)

        # ── Hook: capture premature hidden state in float32 ────────────────
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Cast to float32 immediately — fp16 inf must NOT enter the dict
            captured["h"] = h.detach().float()

        hook = model.model.layers[pre_layer].register_forward_hook(hook_fn)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            hook.remove()
        inference_time = time.perf_counter() - t0

        # ── Final-layer probabilities ──────────────────────────────────────
        final_probs = self._safe_softmax(
            outputs.logits, self.dola_config.temperature
        )

        # ── Premature-layer probabilities ──────────────────────────────────
        # Clean the captured hidden state
        h = torch.nan_to_num(
            captured["h"], nan=0.0, posinf=1e4, neginf=-1e4
        )

        # Apply final LayerNorm in float32.
        #
        # IMPORTANT — we must NEVER call model.model.norm.float() or
        # model.lm_head.float() to promote the module in-place.
        # If an OOM (or any exception) fires mid-computation, the restore
        # never runs and the model is permanently left in the wrong dtype,
        # causing c10::Half != float on every subsequent forward pass.
        #
        # Safe pattern: extract .weight / .bias as float32 TENSORS
        # (this creates a new tensor — the module's stored parameter is
        # untouched) and compute with F.layer_norm / F.linear directly.
        with torch.no_grad():
            try:
                norm_w = model.model.norm.weight.float()          # new fp32 tensor
                norm_b = getattr(model.model.norm, "bias", None)
                norm_b = norm_b.float() if norm_b is not None else None
                eps    = getattr(model.model.norm, "variance_epsilon",
                                 getattr(model.model.norm, "eps", 1e-5))
                # Llama / Mistral use RMSNorm (no bias, no mean subtraction)
                variance = h.pow(2).mean(-1, keepdim=True)
                h_normed = h * torch.rsqrt(variance + eps) * norm_w
                if norm_b is not None:
                    h_normed = h_normed + norm_b
            except Exception:
                # Fallback for unknown norm architectures: use raw hidden state
                h_normed = h

        # Project through lm_head using F.linear with a float32 copy of
        # the weight — no module state is mutated at any point.
        with torch.no_grad():
            lm_w      = model.lm_head.weight.float()              # new fp32 tensor
            lm_b      = model.lm_head.bias
            lm_b      = lm_b.float() if lm_b is not None else None
            pre_logits = F.linear(h_normed, lm_w, lm_b)          # [B, T, vocab]

        pre_probs = self._safe_softmax(
            pre_logits, self.dola_config.temperature
        )

        # ── JSD score per token ────────────────────────────────────────────
        scores = self._jsd(pre_probs, final_probs)   # [B, T]

        # NaN audit: after all guards there should be zero NaN.
        # If any survive, warn loudly (useful for debugging new model archs).
        n_nan = scores.isnan().sum().item()
        if n_nan > 0:
            print(
                f"[DoLA WARNING] {n_nan} NaN scores survived all sanitisation "
                f"layers.  Check: (1) model loaded with torch_dtype= not dtype=, "
                f"(2) input_ids contains no out-of-range token IDs."
            )
            scores = torch.nan_to_num(scores, nan=0.0)

        # ── Decode tokens ──────────────────────────────────────────────────
        tokens = [
            [tokenizer.decode([tok_id]) for tok_id in seq]
            for seq in input_ids.cpu().tolist()
        ]

        return DetectorOutput(
            scores=scores.cpu(),
            tokens=tokens,
            metadata={
                "premature_layer":       pre_layer,
                "n_layers":              n_layers,
                "premature_layer_ratio": self.dola_config.premature_layer_ratio,
                "inference_time_s":      inference_time,
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
        Generate a response and score each generated token.

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