"""
lid/metrics.py
==============
Core metric computation for Latent Instability Diagnostics (LID).

ALL FORMULAS from FORMULAS_STRATEGIES_REFERENCE.md are implemented here.

Formulas implemented:
  - I(l,t)  : Instability Score (normalized L2 divergence)
  - q(l,t)  : Directional Alignment (cosine similarity)
  - Z(l,t)  : Per-layer Composite Instability Score
  - Z(t)    : Aggregated Composite Score across K selected layers

Units:
  - hidden states: [batch, seq_len, d_model] or [seq_len, d_model]
  - All outputs  : [seq_len] or scalar when batch=1

Author: MIT LID Research Team
Week  : 1 (skeleton) → Full implementation Week 4
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


# ── 1. INSTABILITY SCORE ─────────────────────────────────────────────────────
def instability(
    h_clean: torch.Tensor,
    h_pert: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-token instability I(l,t).

    Formula:
        I(l,t) = ||h_clean - h_pert||_2  /  ||h_clean||_2

    Args:
        h_clean : Clean hidden states  [..., d_model]
        h_pert  : Perturbed hidden states [..., d_model]
        eps     : Numerical stability floor

    Returns:
        I : Instability scores [...] (same shape minus last dim)

    Unit-test invariants:
        - δ = 0  →  I = 0  (degenerate case)
        - I ∈ [0, ∞), typically [0, 2]
    """
    numerator = torch.norm(h_clean - h_pert, p=2, dim=-1)
    denominator = torch.norm(h_clean, p=2, dim=-1).clamp(min=eps)
    return numerator / denominator


# ── 2. DIRECTIONAL ALIGNMENT ─────────────────────────────────────────────────
def alignment(
    h_clean: torch.Tensor,
    h_pert: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-token directional alignment q(l,t).

    Formula:
        q(l,t) = (h_clean · h_pert) / (||h_clean||_2 · ||h_pert||_2)

    Args:
        h_clean : Clean hidden states  [..., d_model]
        h_pert  : Perturbed hidden states [..., d_model]
        eps     : Numerical stability floor

    Returns:
        q : Alignment scores [...] ∈ [-1, 1]

    Interpretation:
        q ≈ 1.0 → stable direction (factual)
        q ≈ 0.5 → semantic valley crossing
        q < 0.5 → hallucination likely

    Unit-test invariants:
        - δ = 0  →  q = 1.0
        - q ∈ [-1, 1]  (enforced by cosine definition)
    """
    h_clean_norm = F.normalize(h_clean, p=2, dim=-1)
    h_pert_norm = F.normalize(h_pert, p=2, dim=-1)
    return (h_clean_norm * h_pert_norm).sum(dim=-1)


# ── 3. COMPOSITE INSTABILITY SCORE (per layer) ───────────────────────────────
def composite(
    I: torch.Tensor,
    q: torch.Tensor,
    w_I: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """
    Compute per-layer composite instability Z(l,t).

    Formula:
        Z(l,t) = w_I * I(l,t)  +  (1 - w_I) * (1 - q(l,t))

    Note: (1 - q) maps cosine similarity to instability:
        q = 1.0 → instability contribution = 0
        q = 0.5 → instability contribution = 0.5
        q = 0.0 → instability contribution = 1.0

    Args:
        I   : Instability scores [...] from instability()
        q   : Alignment scores  [...] from alignment()
        w_I : Weight for I component ∈ [0, 1]
              Can be a scalar or per-layer tensor

    Returns:
        Z : Composite scores [...] ∈ [0, ~2]

    Interpretation:
        High I, High q → Creative instability (good, exploring)
        High I, Low  q → Erroneous divergence (bad, hallucination)
        Low  I, High q → Stable / correct (good)
    """
    if isinstance(w_I, float):
        w_I = torch.tensor(w_I, dtype=I.dtype, device=I.device)
    return w_I * I + (1.0 - w_I) * (1.0 - q)


# ── 4. AGGREGATE Z ACROSS LAYERS ─────────────────────────────────────────────
def aggregate_Z(
    Z_per_layer: list[torch.Tensor],
    layer_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Aggregate per-layer Z scores into final per-token Z(t).

    Formula:
        Z(t) = (1/K) * Σ_{l ∈ L_selected} Z(l,t)
        (or weighted average if layer_weights provided)

    Args:
        Z_per_layer   : List of K tensors, each shape [seq_len]
        layer_weights : Optional weight tensor [K], must sum to 1

    Returns:
        Z_t : Aggregated scores [seq_len]
    """
    stacked = torch.stack(Z_per_layer, dim=0)  # [K, seq_len]
    if layer_weights is None:
        return stacked.mean(dim=0)
    else:
        assert layer_weights.shape[0] == stacked.shape[0], (
            f"layer_weights length {layer_weights.shape[0]} != "
            f"num layers {stacked.shape[0]}"
        )
        w = layer_weights.to(stacked.dtype).unsqueeze(-1)  # [K, 1]
        return (stacked * w).sum(dim=0)


# ── 5. CONVENIENCE: FULL METRIC COMPUTATION FOR ONE LAYER ────────────────────
def compute_all(
    h_clean: torch.Tensor,
    h_pert: torch.Tensor,
    w_I: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Compute I, q, and Z for a single layer in one call.

    Returns:
        dict with keys "I", "q", "Z" each shape [...] (seq_len)
    """
    I = instability(h_clean, h_pert, eps=eps)
    q = alignment(h_clean, h_pert, eps=eps)
    Z = composite(I, q, w_I=w_I)
    return {"I": I, "q": q, "Z": Z}
