"""
tests/test_evaluation.py
========================
Unit tests for evaluation/metrics.py

Tests:
    - Perfect classifier → AUROC=1.0, AUPRC=1.0
    - Random classifier  → AUROC≈0.5
    - Inverted scores    → AUROC < 0.5 (but we flip = > 0.5)
    - FPR@TPR80 range check
    - Lead time stats: mean, positive_rate, coverage
    - EvalResult summary line formatting
    - Bootstrap CI: lower < point estimate < upper

Run:
    pytest tests/test_evaluation.py -v --tb=short

Author: MIT LID Research Team
Week  : 1 (must be green — eval harness is used from Week 5)
"""

import numpy as np
import pytest
from evaluation.metrics import (
    compute_auroc, compute_auprc, compute_fpr_at_tpr,
    compute_lead_time_stats, evaluate, EvalResult, OverheadTimer, bootstrap_ci,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_scores():
    """Perfect classifier: high scores on hallucinated, low on correct."""
    labels = np.array([0]*50 + [1]*10, dtype=float)
    scores = np.where(labels == 1, 0.9, 0.1)
    return scores, labels

@pytest.fixture
def random_scores():
    """Random classifier: scores uncorrelated with labels."""
    rng = np.random.default_rng(0)
    labels = np.array([0]*50 + [1]*10, dtype=float)
    scores = rng.random(60)
    return scores, labels

@pytest.fixture
def realistic_scores():
    """Realistic LID-like scores with some separation but not perfect."""
    rng = np.random.default_rng(42)
    labels = np.array([0]*90 + [1]*10, dtype=float)
    # Hallucinated tokens get higher scores on average
    scores = np.concatenate([
        rng.normal(0.3, 0.15, 90).clip(0, 1),
        rng.normal(0.65, 0.15, 10).clip(0, 1),
    ])
    return scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# AUROC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAUROC:
    def test_perfect_classifier(self, perfect_scores):
        scores, labels = perfect_scores
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(1.0), f"Perfect classifier AUROC={auroc}"

    def test_random_classifier_near_half(self, random_scores):
        scores, labels = random_scores
        auroc = compute_auroc(scores, labels)
        # Random should be near 0.5, allow wide tolerance
        assert 0.2 < auroc < 0.8, f"Random AUROC expected near 0.5, got {auroc}"

    def test_all_same_label_returns_nan(self):
        scores = np.array([0.5] * 10)
        labels = np.zeros(10)      # all negative
        auroc = compute_auroc(scores, labels)
        assert np.isnan(auroc), "All-same labels should return NaN"

    def test_auroc_range(self, realistic_scores):
        scores, labels = realistic_scores
        auroc = compute_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0, f"AUROC out of [0,1]: {auroc}"


# ─────────────────────────────────────────────────────────────────────────────
# AUPRC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAUPRC:
    def test_perfect_classifier(self, perfect_scores):
        scores, labels = perfect_scores
        auprc = compute_auprc(scores, labels)
        assert auprc == pytest.approx(1.0), f"Perfect classifier AUPRC={auprc}"

    def test_auprc_above_baseline(self, realistic_scores):
        """AUPRC should beat random baseline (= hallucination_rate)."""
        scores, labels = realistic_scores
        auprc = compute_auprc(scores, labels)
        random_baseline = labels.mean()
        assert auprc > random_baseline, \
            f"AUPRC {auprc:.3f} not above random baseline {random_baseline:.3f}"

    def test_no_positives_returns_nan(self):
        scores = np.array([0.5] * 10)
        labels = np.zeros(10)
        auprc = compute_auprc(scores, labels)
        assert np.isnan(auprc)


# ─────────────────────────────────────────────────────────────────────────────
# FPR@TPR80 TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFPRAtTPR:
    def test_perfect_classifier_fpr_near_zero(self, perfect_scores):
        scores, labels = perfect_scores
        fpr = compute_fpr_at_tpr(scores, labels, target_tpr=0.80)
        assert fpr < 0.10, f"Perfect classifier FPR@TPR80 expected <0.10, got {fpr}"

    def test_fpr_in_range(self, realistic_scores):
        scores, labels = realistic_scores
        fpr = compute_fpr_at_tpr(scores, labels, target_tpr=0.80)
        assert 0.0 <= fpr <= 1.0, f"FPR out of [0,1]: {fpr}"


# ─────────────────────────────────────────────────────────────────────────────
# LEAD TIME STATS
# ─────────────────────────────────────────────────────────────────────────────

class TestLeadTimeStats:
    def test_all_positive_lead_times(self):
        lt = [2.0, 1.5, 3.0, 1.0]
        stats = compute_lead_time_stats(lt)
        assert stats["positive_rate"] == pytest.approx(1.0)
        assert stats["mean"] == pytest.approx(1.875)
        assert stats["coverage"] == pytest.approx(1.0)

    def test_mixed_positive_negative(self):
        lt = [2.0, -0.5, 1.0, None, 0.3]
        stats = compute_lead_time_stats(lt)
        assert 0 < stats["positive_rate"] < 1
        assert stats["coverage"] == pytest.approx(4/5)

    def test_all_none_returns_nan(self):
        lt = [None, None, None]
        stats = compute_lead_time_stats(lt)
        assert np.isnan(stats["mean"])
        assert stats["coverage"] == pytest.approx(0.0)

    def test_positive_rate_threshold(self):
        """Phase 1 target: positive_rate should exceed 0.5 if LID works."""
        lt = [1.5, 0.8, 2.1, 1.0, 0.5]   # all positive
        stats = compute_lead_time_stats(lt)
        assert stats["positive_rate"] > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATE() FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateFn:
    def test_returns_eval_result(self, perfect_scores):
        scores, labels = perfect_scores
        result = evaluate(scores, labels, method="LID", dataset="TruthfulQA",
                          model="llama-7b", compute_ci=False)
        assert isinstance(result, EvalResult)

    def test_perfect_scores_pass_phase1_gate(self, perfect_scores):
        """Phase 1 gate: AUROC > 0.70."""
        scores, labels = perfect_scores
        result = evaluate(scores, labels, compute_ci=False)
        assert result.auroc is not None
        assert result.auroc >= 0.70, \
            f"Perfect classifier must pass Phase 1 gate (AUROC≥0.70), got {result.auroc}"

    def test_below_threshold_flag(self):
        """AUROC < 0.60 should set below_threshold=True (red flag → escalate)."""
        labels = np.array([0]*90 + [1]*10, dtype=float)
        # Inverted scores — LID running backwards
        scores = np.where(labels == 1, 0.1, 0.9)
        result = evaluate(scores, labels, compute_ci=False)
        assert result.below_threshold is True, \
            "Poor AUROC should set below_threshold flag"

    def test_summary_line_contains_method(self, realistic_scores):
        scores, labels = realistic_scores
        result = evaluate(scores, labels, method="MyDetector", compute_ci=False)
        line = result.summary_line()
        assert "MyDetector" in line

    def test_hallucination_rate_computed(self, perfect_scores):
        scores, labels = perfect_scores
        result = evaluate(scores, labels, compute_ci=False)
        expected_rate = 10 / 60
        assert result.hallucination_rate == pytest.approx(expected_rate, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_ci_contains_point_estimate(self, realistic_scores):
        scores, labels = realistic_scores
        point = compute_auroc(scores, labels)
        lo, hi = bootstrap_ci(scores, labels, compute_auroc, n_bootstrap=200, seed=0)
        assert lo <= point <= hi, \
            f"CI [{lo:.3f}, {hi:.3f}] does not contain point estimate {point:.3f}"

    def test_ci_width_reasonable(self, realistic_scores):
        scores, labels = realistic_scores
        lo, hi = bootstrap_ci(scores, labels, compute_auroc, n_bootstrap=200, seed=0)
        width = hi - lo
        assert 0 < width < 0.5, f"CI width suspiciously large or zero: {width:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# OVERHEAD TIMER
# ─────────────────────────────────────────────────────────────────────────────

class TestOverheadTimer:
    def test_ratio_computed(self):
        import time
        timer = OverheadTimer()
        for _ in range(3):
            with timer.clean():
                time.sleep(0.001)
            with timer.instrumented():
                time.sleep(0.002)   # 2× slower
        ratio = timer.ratio()
        assert ratio is not None
        assert 1.0 < ratio < 5.0, f"Expected ratio ~2.0, got {ratio:.2f}"

    def test_no_data_returns_none(self):
        timer = OverheadTimer()
        assert timer.ratio() is None
