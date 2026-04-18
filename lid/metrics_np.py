"""
lid/metrics_np.py
=================
Pure-numpy implementation of all LID metrics.

PURPOSE:
    1. Runs on ANY machine (laptop, CI, GPU node) without PyTorch
    2. Used for unit-test validation of the torch version
    3. Reference implementation — if torch and numpy disagree, numpy wins

All formulas from FORMULAS_STRATEGIES_REFERENCE.md.

Author: MIT LID Research Team
Week  : 1 (Week 1 deliverable — CI-green tests)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── 1. RMS ───────────────────────────────────────────────────────────────────
def rms(h: np.ndarray) -> np.ndarray:
    """RMS(h) = sqrt(mean(h²)) across last axis."""
    return np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True))


# ── 2. PERTURBATION SCALE ────────────────────────────────────────────────────
def noise_scale(h: np.ndarray, alpha: float) -> np.ndarray:
    """σ_l = α · RMS(h_l)"""
    return alpha * rms(h)


# ── 3. INJECT NOISE ──────────────────────────────────────────────────────────
def inject_noise(h: np.ndarray, alpha: float, seed: int = 42) -> np.ndarray:
    """
    h_pert = h + δ,  δ ~ N(0, σ²I),  σ = α·RMS(h)
    Deterministic: same seed → identical δ.
    """
    if alpha == 0.0:
        return h.copy()
    rng = np.random.default_rng(seed)
    sigma = noise_scale(h, alpha)          # [..., 1]
    delta = rng.standard_normal(h.shape) * sigma
    return h + delta


# ── 4. INSTABILITY SCORE I(l,t) ──────────────────────────────────────────────
def instability(h_clean: np.ndarray, h_pert: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    I(l,t) = ||h_clean - h_pert||₂  /  ||h_clean||₂

    Invariants:
        - h_pert == h_clean  →  I = 0
        - I ≥ 0
    """
    numerator = np.linalg.norm(h_clean - h_pert, axis=-1)
    denominator = np.linalg.norm(h_clean, axis=-1).clip(min=eps)
    return numerator / denominator


# ── 5. DIRECTIONAL ALIGNMENT q(l,t) ─────────────────────────────────────────
def alignment(h_clean: np.ndarray, h_pert: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    q(l,t) = (h_clean · h_pert) / (||h_clean|| · ||h_pert||)

    Invariants:
        - h_pert == h_clean  →  q = 1.0
        - q ∈ [-1, 1]
    """
    dot = np.sum(h_clean * h_pert, axis=-1)
    norm_c = np.linalg.norm(h_clean, axis=-1).clip(min=eps)
    norm_p = np.linalg.norm(h_pert, axis=-1).clip(min=eps)
    return dot / (norm_c * norm_p)


# ── 6. COMPOSITE SCORE Z(l,t) ────────────────────────────────────────────────
def composite(I: np.ndarray, q: np.ndarray, w_I: float = 0.5) -> np.ndarray:
    """
    Z(l,t) = w_I · I(l,t)  +  (1 − w_I) · (1 − q(l,t))

    Interpretation:
        High I, High q → Creative instability (good)
        High I, Low  q → Erroneous divergence  (hallucination)
        Low  I, High q → Stable/correct         (factual)
    """
    return w_I * I + (1.0 - w_I) * (1.0 - q)


# ── 7. AGGREGATE Z ACROSS LAYERS ─────────────────────────────────────────────
def aggregate_Z(
    Z_per_layer: list[np.ndarray],
    layer_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Z(t) = (1/K) Σ_{l ∈ L_selected} Z(l,t)

    Args:
        Z_per_layer   : K arrays each shape [seq_len]
        layer_weights : Optional [K] weights summing to 1
    """
    stacked = np.stack(Z_per_layer, axis=0)   # [K, seq_len]
    if layer_weights is None:
        return stacked.mean(axis=0)
    w = layer_weights[:, np.newaxis]           # [K, 1]
    return (stacked * w).sum(axis=0)


# ── 8. FULL SINGLE-LAYER COMPUTATION ─────────────────────────────────────────
def compute_all(
    h_clean: np.ndarray,
    h_pert: np.ndarray,
    w_I: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute I, q, Z in one call. Returns dict with keys 'I', 'q', 'Z'."""
    I = instability(h_clean, h_pert)
    q = alignment(h_clean, h_pert)
    Z = composite(I, q, w_I=w_I)
    return {"I": I, "q": q, "Z": Z}


# ── 9. PEAK DETECTION ────────────────────────────────────────────────────────
def adaptive_threshold(Z: np.ndarray, multiplier: float = 1.5) -> float:
    """threshold = mean(Z) + multiplier · std(Z)"""
    return float(Z.mean() + multiplier * Z.std())


def detect_peaks(
    Z: np.ndarray,
    multiplier: float = 1.5,
    min_width: int = 1,
) -> list[dict]:
    """
    Find instability peaks in Z(t) trajectory.

    Returns list of dicts with keys:
        center, start, end, max_z, mean_z, width
    """
    threshold = adaptive_threshold(Z, multiplier)
    above = Z > threshold
    peaks = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i - 1
            width = end - start + 1
            if width >= min_width:
                cluster = Z[start:end + 1]
                positions = np.arange(start, end + 1, dtype=float)
                weights = cluster - cluster.min() + 1e-8
                center = float(np.average(positions, weights=weights))
                peaks.append(dict(
                    center=center, start=start, end=end,
                    max_z=float(cluster.max()),
                    mean_z=float(cluster.mean()),
                    width=width,
                ))
        else:
            i += 1
    return peaks


def compute_lead_time(peaks: list[dict], hallucination_start: int) -> Optional[float]:
    """lead_time = t_hallucination − t_peak_center  (positive = early detection)"""
    if not peaks:
        return None
    valid = [p for p in peaks if p["center"] < hallucination_start]
    if not valid:
        closest = min(peaks, key=lambda p: abs(p["center"] - hallucination_start))
        return float(hallucination_start - closest["center"])
    best = max(valid, key=lambda p: p["center"])
    return float(hallucination_start - best["center"])
