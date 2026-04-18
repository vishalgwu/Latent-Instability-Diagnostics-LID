"""
evaluation/metrics.py
=====================
All evaluation metrics for lid and baselines.

Metrics implemented (from FORMULAS_STRATEGIES_REFERENCE.md):
    - AUROC   : Area Under ROC Curve   (token-level classification)
    - AUPRC   : Area Under PR Curve    (primary metric — handles imbalance)
    - Lead Time: Mean tokens before hallucination (positive = early detection)
    - FPR@TPR80: False positive rate at 80% true positive rate
    - Overhead : Wall-clock ratio vs clean inference

All metrics accept:
    scores : np.ndarray [N] — per-token scores (higher = more suspicious)
    labels : np.ndarray [N] — binary (1 = hallucinated, 0 = correct)

Author: MIT lid Research Team
Week  : 1 (skeleton) → Used from Week 5 onward
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Standardized result from evaluating one detector on one dataset.
    All values are floats or None if not computable.
    """
    method: str
    dataset: str
    model: str

    # Core metrics
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    fpr_at_tpr80: Optional[float] = None

    # Lead time (lid-specific, but computed for all methods if possible)
    lead_time_mean: Optional[float] = None
    lead_time_median: Optional[float] = None
    lead_time_positive_rate: Optional[float] = None   # fraction with positive lead

    # Efficiency
    overhead_ratio: Optional[float] = None            # wall-clock vs clean inference

    # Sample info
    n_examples: int = 0
    n_tokens: int = 0
    n_hallucinated_tokens: int = 0
    hallucination_rate: float = 0.0

    # Confidence intervals (bootstrap, filled post-computation)
    auroc_ci: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    auprc_ci: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # Flags
    below_threshold: bool = False     # True if AUROC < 0.60 (escalate to advisor)
    ood: bool = False                 # True if out-of-distribution dataset

    def summary_line(self) -> str:
        """One-line summary for weekly reports."""
        auroc_str = f"{self.auroc:.3f}" if self.auroc else "N/A"
        auprc_str = f"{self.auprc:.3f}" if self.auprc else "N/A"
        lt_str = f"{self.lead_time_mean:.2f}" if self.lead_time_mean else "N/A"
        return (
            f"[{self.method:20s}] {self.dataset:15s} | "
            f"AUROC={auroc_str}  AUPRC={auprc_str}  "
            f"LeadTime={lt_str}  Overhead={self.overhead_ratio or 'N/A'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Area Under ROC Curve — token-level classification.

    Range: [0, 1]
    0.5 = random, 1.0 = perfect, <0.5 = inverted signal

    Args:
        scores : Per-token scores [N] (higher = more suspicious)
        labels : Binary labels [N] (1 = hallucinated)

    Returns:
        AUROC scalar
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_auprc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Area Under Precision-Recall Curve.

    More meaningful than AUROC for imbalanced data.
    Baseline = hallucination_rate (random classifier).
    Target (Phase 2): AUPRC ≥ 0.70 on TruthfulQA

    Args:
        scores : Per-token scores [N]
        labels : Binary labels [N]

    Returns:
        AUPRC scalar
    """
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_fpr_at_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.80,
) -> float:
    """
    False Positive Rate at a specified True Positive Rate.

    Standard operating point: FPR@TPR=0.80
    Target: FPR < 0.10

    Args:
        scores     : Per-token scores [N]
        labels     : Binary labels [N]
        target_tpr : TPR threshold (default 0.80)

    Returns:
        FPR at the operating point closest to target_tpr
    """
    if labels.sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find the FPR where TPR is closest to target_tpr
    idx = np.argmin(np.abs(tpr - target_tpr))
    return float(fpr[idx])


def compute_lead_time_stats(
    lead_times: list[Optional[float]],
) -> dict[str, float]:
    """
    Compute lead time statistics across all examples.

    Lead time = t_hallucination - t_peak_center
    Positive = peak detected BEFORE hallucination (desired)

    Args:
        lead_times : List of per-example lead times (None if no peak detected)

    Returns:
        dict with mean, median, std, positive_rate, coverage
    """
    valid = [lt for lt in lead_times if lt is not None]
    if not valid:
        return {
            "mean": float("nan"), "median": float("nan"),
            "std": float("nan"), "positive_rate": 0.0,
            "coverage": 0.0,
        }

    arr = np.array(valid)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "positive_rate": float((arr > 0).mean()),   # fraction with early detection
        "coverage": float(len(valid) / len(lead_times)),  # fraction with any peak
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap 95% CI for any metric function.

    Args:
        scores    : Per-token scores [N]
        labels    : Binary labels [N]
        metric_fn : Callable(scores, labels) -> float
        n_bootstrap: Number of bootstrap samples
        alpha     : Significance level (0.05 = 95% CI)
        seed      : RNG seed

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    bootstrap_vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(scores[idx], labels[idx])
        if not np.isnan(val):
            bootstrap_vals.append(val)

    if not bootstrap_vals:
        return (float("nan"), float("nan"))

    arr = np.array(bootstrap_vals)
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ─────────────────────────────────────────────────────────────────────────────
# OVERHEAD MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

class OverheadTimer:
    """
    Measures wall-clock overhead ratio vs clean inference.

    Usage:
        timer = OverheadTimer()
        with timer.clean():
            run_clean_inference(...)
        with timer.instrumented():
            run_lid_inference(...)
        ratio = timer.ratio()   # e.g., 1.20 = 20% overhead
    """

    def __init__(self):
        self._clean_times: list[float] = []
        self._instrumented_times: list[float] = []
        self._mode = None

    class _Context:
        def __init__(self, timer, mode):
            self._timer = timer
            self._mode = mode
            self._start = None

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self._start
            if self._mode == "clean":
                self._timer._clean_times.append(elapsed)
            else:
                self._timer._instrumented_times.append(elapsed)

    def clean(self):
        return self._Context(self, "clean")

    def instrumented(self):
        return self._Context(self, "instrumented")

    def ratio(self) -> Optional[float]:
        """Return mean(instrumented) / mean(clean), or None if no data."""
        if not self._clean_times or not self._instrumented_times:
            return None
        return float(np.mean(self._instrumented_times) / np.mean(self._clean_times))


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    lead_times: Optional[list[Optional[float]]] = None,
    method: str = "unknown",
    dataset: str = "unknown",
    model: str = "unknown",
    overhead_ratio: Optional[float] = None,
    ood: bool = False,
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
) -> EvalResult:
    """
    Full evaluation: compute all metrics from scores + labels.

    Args:
        scores        : Flattened per-token scores [N]
        labels        : Flattened binary labels [N]
        lead_times    : Optional list of per-example lead times
        method        : Detector name (e.g., "lid", "DoLA")
        dataset       : Dataset name
        model         : Model name
        overhead_ratio: Pre-measured overhead ratio
        ood           : Whether this is OOD evaluation
        compute_ci    : Whether to compute bootstrap CIs (slower)
        n_bootstrap   : Bootstrap samples for CI

    Returns:
        EvalResult with all metrics filled in
    """
    result = EvalResult(
        method=method,
        dataset=dataset,
        model=model,
        n_tokens=len(labels),
        n_hallucinated_tokens=int(labels.sum()),
        hallucination_rate=float(labels.mean()),
        overhead_ratio=overhead_ratio,
        ood=ood,
    )

    # Core metrics
    result.auroc = compute_auroc(scores, labels)
    result.auprc = compute_auprc(scores, labels)
    result.fpr_at_tpr80 = compute_fpr_at_tpr(scores, labels, target_tpr=0.80)

    # Lead time stats
    if lead_times is not None:
        lt_stats = compute_lead_time_stats(lead_times)
        result.lead_time_mean = lt_stats["mean"]
        result.lead_time_median = lt_stats["median"]
        result.lead_time_positive_rate = lt_stats["positive_rate"]

    # Bootstrap CIs
    if compute_ci and not np.isnan(result.auroc or float("nan")):
        result.auroc_ci = bootstrap_ci(scores, labels, compute_auroc, n_bootstrap)
        result.auprc_ci = bootstrap_ci(scores, labels, compute_auprc, n_bootstrap)

    # Red flag check
    if result.auroc is not None and not np.isnan(result.auroc):
        result.below_threshold = result.auroc < 0.60

    return result
