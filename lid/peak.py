"""
lid/peak.py
===========
Adaptive peak detection on Z(t) trajectories.

Formula (from FORMULAS_STRATEGIES_REFERENCE.md):
    threshold = mean(Z) + 1.5 · std(Z)

Algorithm:
    1. Compute Z(t) for entire sequence
    2. threshold = mean + 1.5*std
    3. Find all t where Z(t) > threshold
    4. Cluster consecutive high values into peaks
    5. Return center-of-mass of each peak cluster

Why 1.5σ:
    Balances sensitivity/specificity. Detects outliers without
    excessive false positives. (z-score 1.5 ≈ top 6.7% of normal)

Author: MIT LID Research Team
Week  : 1 (skeleton) → Full implementation Week 4
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Peak:
    """Represents a detected instability peak in a Z(t) trajectory."""
    center: float          # Center-of-mass token position
    start: int             # First token in peak cluster
    end: int               # Last token in peak cluster (inclusive)
    max_z: float           # Maximum Z value in peak
    mean_z: float          # Mean Z value in peak
    width: int             # Number of tokens in peak


# ── 1. ADAPTIVE THRESHOLD ────────────────────────────────────────────────────
def adaptive_threshold(
    Z: torch.Tensor,
    multiplier: float = 1.5,
) -> float:
    """
    Compute adaptive detection threshold.

    Formula:
        threshold = mean(Z) + multiplier · std(Z)

    Args:
        Z          : Composite score trajectory [seq_len]
        multiplier : σ multiplier (default 1.5 per spec)

    Returns:
        threshold : Scalar float
    """
    z_np = Z.float().cpu().numpy()
    return float(np.mean(z_np) + multiplier * np.std(z_np))


# ── 2. PEAK DETECTION ────────────────────────────────────────────────────────
def detect_peaks(
    Z: torch.Tensor,
    multiplier: float = 1.5,
    min_width: int = 1,
) -> list[Peak]:
    """
    Detect instability peaks in a Z(t) trajectory.

    Algorithm:
        1. Compute adaptive threshold = mean(Z) + 1.5*std(Z)
        2. Find positions where Z > threshold
        3. Cluster consecutive positions into peaks
        4. Return Peak objects with center-of-mass per peak

    Args:
        Z          : Composite score trajectory [seq_len]
        multiplier : Threshold multiplier (default 1.5)
        min_width  : Minimum tokens for a valid peak (filter noise)

    Returns:
        peaks : List of Peak objects, sorted by center position

    Unit test:
        - Synthetic spike at known t → Peak.center within ±2 tokens
        - All-zero Z → returns empty list
        - Single-spike Z → returns exactly one Peak
    """
    threshold = adaptive_threshold(Z, multiplier)
    z_np = Z.float().cpu().numpy()

    above = z_np > threshold                # boolean mask
    peaks = []
    i = 0

    while i < len(above):
        if above[i]:
            # Start of a new cluster
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i - 1  # inclusive

            width = end - start + 1
            if width >= min_width:
                cluster_z = z_np[start:end + 1]
                positions = np.arange(start, end + 1, dtype=float)
                # Center-of-mass weighted by Z values
                weights = cluster_z - cluster_z.min() + 1e-8
                center = float(np.average(positions, weights=weights))

                peaks.append(Peak(
                    center=center,
                    start=start,
                    end=end,
                    max_z=float(cluster_z.max()),
                    mean_z=float(cluster_z.mean()),
                    width=width,
                ))
        else:
            i += 1

    return peaks


# ── 3. LEAD TIME COMPUTATION ─────────────────────────────────────────────────
def compute_lead_time(
    peaks: list[Peak],
    hallucination_start: int,
) -> Optional[float]:
    """
    Compute lead time for a single example.

    Formula:
        lead_time = t_hallucination - t_peak_center

    A positive lead time means the peak was detected BEFORE the hallucination.

    Args:
        peaks               : Detected peaks from detect_peaks()
        hallucination_start : Token index where hallucination begins (from annotation)

    Returns:
        Lead time in tokens (positive = detected before error), or None if no peaks
    """
    if not peaks:
        return None

    # Use the peak closest to (but before) the hallucination
    valid_peaks = [p for p in peaks if p.center < hallucination_start]
    if not valid_peaks:
        # Use the last peak even if it's after (negative lead time)
        closest = min(peaks, key=lambda p: abs(p.center - hallucination_start))
        return float(hallucination_start - closest.center)

    best = max(valid_peaks, key=lambda p: p.center)
    return float(hallucination_start - best.center)
