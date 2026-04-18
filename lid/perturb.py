"""
lid/perturb.py
==============
Gaussian perturbation injection for Latent Instability Diagnostics.

Core formula (from FORMULAS_STRATEGIES_REFERENCE.md):
    σ_l = α · RMS(h_l)
    δ ~ N(0, σ²·I)
    h_pert = h_clean + δ

Design principles:
    - DETERMINISTIC: fixed seed → identical δ → bitwise-identical outputs
    - SINGLE PASS: one perturbation, reused across all K layers (efficient)
    - READ-ONLY: no model modification (PyTorch forward hooks only)
    - OVERHEAD TARGET: <25% wall-clock vs clean inference

Author: MIT LID Research Team
Week  : 1 (skeleton) → Full implementation Week 4
"""

from __future__ import annotations

import torch
from typing import Optional


# ── 1. RMS COMPUTATION ───────────────────────────────────────────────────────
def rms(h: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Square of a hidden state tensor.

    Formula:
        RMS(h) = sqrt(1/d * Σ h_i²)

    Args:
        h : Hidden state [..., d_model]

    Returns:
        Scalar (or batch of scalars) representing RMS magnitude
    """
    return h.pow(2).mean(dim=-1, keepdim=True).sqrt()


# ── 2. NOISE SCALE COMPUTATION ───────────────────────────────────────────────
def noise_scale(h: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute per-position noise scale σ_l = α · RMS(h_l).

    Args:
        h     : Hidden state [..., d_model]
        alpha : Perturbation magnitude (calibrated in Phase 1)
                Expected optimal: α ≈ 0.05 (5% of layer RMS)
                Grid search range: {0.01, 0.025, 0.05, 0.1}

    Returns:
        σ : Noise scale [..., 1] (broadcastable to h shape)
    """
    return alpha * rms(h)


# ── 3. MAIN INJECTION FUNCTION ───────────────────────────────────────────────
def inject_noise(
    hidden_state: torch.Tensor,
    alpha: float,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Inject Gaussian noise into a hidden state.

    Formula:
        σ = α · RMS(h)
        δ ~ N(0, σ²·I)
        h_pert = h + δ

    CRITICAL: Uses a fixed RNG generator (not global state) so that:
        - The global torch RNG is NOT contaminated
        - Same (seed, alpha, h) → identical output across runs
        - Thread-safe for multi-GPU setups

    Args:
        hidden_state : Input hidden state [..., d_model]
        alpha        : Perturbation magnitude ∈ (0, 1)
        seed         : RNG seed for reproducibility (None = use current global state)

    Returns:
        h_pert : Perturbed hidden state, same shape as input

    Unit test invariants:
        - alpha=0   → h_pert == hidden_state  (no perturbation)
        - seed=42   → bitwise-identical across calls
        - Output shape == input shape
    """
    if alpha == 0.0:
        return hidden_state.clone()

    # Create isolated RNG generator (does NOT affect global torch.manual_seed state)
    generator = torch.Generator(device=hidden_state.device)
    if seed is not None:
        generator.manual_seed(seed)

    sigma = noise_scale(hidden_state, alpha)           # [..., 1]
    delta = torch.randn_like(hidden_state, generator=generator) * sigma
    return hidden_state + delta


# ── 4. BATCH INJECTION (single δ reused across layers) ───────────────────────
def generate_noise_vector(
    shape: tuple[int, ...],
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """
    Pre-generate a single noise vector δ to reuse across K layers.

    Strategy: Generate δ once per example, propagate through model once.
    This is Strategy 1 from FORMULAS_STRATEGIES_REFERENCE.md:
    "Reuse same δ across all layers" → 15-30% overhead vs 50-300% for multi-sampling.

    Args:
        shape  : Target tensor shape (e.g., [seq_len, d_model])
        sigma  : Noise magnitude (pre-computed from α · RMS(h))
        device : Target device
        dtype  : Target dtype
        seed   : Reproducibility seed

    Returns:
        δ : Noise tensor of given shape
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator) * sigma
