"""
tests/test_metrics_np.py
========================
Unit tests for lid/metrics_np.py — pure numpy, no PyTorch required.

These tests MUST pass on any machine (laptop, CI, A100) before
any GPU experiments begin.

TEST CONTRACT (from 12-Week Plan, Week 4 KPIs):
    ✅ δ = 0  →  I = 0, q = 1.0, Z = 0
    ✅ δ with known magnitude → analytically verifiable I
    ✅ q ∈ [-1, 1] always
    ✅ Z ≥ 0
    ✅ Determinism: same seed → identical output × 3 runs
    ✅ Synthetic spike → peak center within ±2 tokens
    ✅ Positive lead time when peak precedes hallucination
    ✅ Aggregate Z = mean of layer scores (uniform weights)

Run:
    pytest tests/test_metrics_np.py -v --tb=short

Author: MIT LID Research Team
Week  : 1 (must be green before Week 2 work begins)
"""

import numpy as np
import pytest
from lid.metrics_np import (
    rms, noise_scale, inject_noise,
    instability, alignment, composite, aggregate_Z, compute_all,
    adaptive_threshold, detect_peaks, compute_lead_time,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def h_clean():
    """Hidden state: seq_len=8, d_model=32 — fixed seed for reproducibility."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 32)).astype(np.float64)


@pytest.fixture
def h_zero_pert(h_clean):
    """Perturbed state identical to clean (zero perturbation)."""
    return h_clean.copy()


@pytest.fixture
def h_small_pert(h_clean):
    """Small realistic perturbation (alpha=0.05)."""
    return inject_noise(h_clean, alpha=0.05, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEGENERATE CASE: δ = 0
# ─────────────────────────────────────────────────────────────────────────────

class TestDegenerateZeroPerturbation:
    """When perturbation is zero, all instability must be zero/neutral."""

    def test_I_is_zero_when_delta_zero(self, h_clean, h_zero_pert):
        I = instability(h_clean, h_zero_pert)
        np.testing.assert_allclose(I, 0.0, atol=1e-10,
            err_msg="I must be 0 when δ=0")

    def test_q_is_one_when_delta_zero(self, h_clean, h_zero_pert):
        q = alignment(h_clean, h_zero_pert)
        np.testing.assert_allclose(q, 1.0, atol=1e-10,
            err_msg="q must be 1 when δ=0")

    def test_Z_is_zero_when_delta_zero(self, h_clean, h_zero_pert):
        I = instability(h_clean, h_zero_pert)
        q = alignment(h_clean, h_zero_pert)
        Z = composite(I, q, w_I=0.5)
        np.testing.assert_allclose(Z, 0.0, atol=1e-10,
            err_msg="Z must be 0 when δ=0")

    def test_inject_zero_alpha_is_identity(self, h_clean):
        h_out = inject_noise(h_clean, alpha=0.0, seed=42)
        np.testing.assert_array_equal(h_out, h_clean,
            err_msg="inject_noise(alpha=0) must return identical array")

    def test_compute_all_zero_delta(self, h_clean, h_zero_pert):
        out = compute_all(h_clean, h_zero_pert)
        np.testing.assert_allclose(out["I"], 0.0, atol=1e-10)
        np.testing.assert_allclose(out["q"], 1.0, atol=1e-10)
        np.testing.assert_allclose(out["Z"], 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYTIC VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticValues:
    """Verify formulas against hand-computed results."""

    def test_I_analytic_known_vectors(self):
        """
        h_clean = [1, 0],  h_pert = [0, 1]
        ||h_clean - h_pert||₂ = sqrt(2)
        ||h_clean||₂ = 1
        I = sqrt(2) / 1 = sqrt(2) ≈ 1.4142
        """
        h_c = np.array([[1.0, 0.0]])
        h_p = np.array([[0.0, 1.0]])
        I = instability(h_c, h_p)
        np.testing.assert_allclose(I, np.sqrt(2), rtol=1e-6)

    def test_q_analytic_orthogonal_vectors(self):
        """
        h_clean = [1, 0],  h_pert = [0, 1]
        cosine = 0  (orthogonal)
        """
        h_c = np.array([[1.0, 0.0]])
        h_p = np.array([[0.0, 1.0]])
        q = alignment(h_c, h_p)
        np.testing.assert_allclose(q, 0.0, atol=1e-10)

    def test_q_analytic_antiparallel_vectors(self):
        """
        h_clean = [1, 0],  h_pert = [-1, 0]
        cosine = -1  (antiparallel)
        """
        h_c = np.array([[1.0, 0.0]])
        h_p = np.array([[-1.0, 0.0]])
        q = alignment(h_c, h_p)
        np.testing.assert_allclose(q, -1.0, atol=1e-10)

    def test_Z_formula_manual(self):
        """
        I = 0.6, q = 0.4, w_I = 0.5
        Z = 0.5*0.6 + 0.5*(1-0.4) = 0.3 + 0.3 = 0.6
        """
        I = np.array([0.6])
        q = np.array([0.4])
        Z = composite(I, q, w_I=0.5)
        np.testing.assert_allclose(Z, 0.6, atol=1e-10)

    def test_Z_high_I_low_q_is_large(self):
        """High I + Low q → high Z (hallucination signal)."""
        I_hall = np.array([0.9])
        q_hall = np.array([0.1])
        Z_hall = composite(I_hall, q_hall, w_I=0.5)

        I_fact = np.array([0.05])
        q_fact = np.array([0.98])
        Z_fact = composite(I_fact, q_fact, w_I=0.5)

        assert Z_hall > Z_fact, \
            f"Hallucination Z ({Z_hall[0]:.3f}) should exceed factual Z ({Z_fact[0]:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALUE RANGE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestValueRanges:

    def test_q_in_minus1_to_1(self, h_clean, h_small_pert):
        q = alignment(h_clean, h_small_pert)
        assert np.all(q >= -1.0 - 1e-9), f"q below -1: {q.min()}"
        assert np.all(q <= 1.0 + 1e-9), f"q above 1: {q.max()}"

    def test_I_nonnegative(self, h_clean, h_small_pert):
        I = instability(h_clean, h_small_pert)
        assert np.all(I >= 0), f"I has negative values: {I.min()}"

    def test_Z_nonnegative_for_typical_perturbation(self, h_clean, h_small_pert):
        I = instability(h_clean, h_small_pert)
        q = alignment(h_clean, h_small_pert)
        Z = composite(I, q, w_I=0.5)
        assert np.all(Z >= -1e-10), f"Z has negative values: {Z.min()}"

    def test_rms_positive(self, h_clean):
        r = rms(h_clean)
        assert np.all(r > 0), "RMS must be positive for non-zero inputs"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DETERMINISM
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Same (seed, alpha, input) → bitwise-identical output, 3 runs."""

    def test_inject_noise_deterministic(self, h_clean):
        results = [inject_noise(h_clean, alpha=0.05, seed=42) for _ in range(3)]
        np.testing.assert_array_equal(results[0], results[1], err_msg="Run1 != Run2")
        np.testing.assert_array_equal(results[1], results[2], err_msg="Run2 != Run3")

    def test_different_seeds_produce_different_noise(self, h_clean):
        h1 = inject_noise(h_clean, alpha=0.05, seed=42)
        h2 = inject_noise(h_clean, alpha=0.05, seed=99)
        assert not np.array_equal(h1, h2), \
            "Different seeds must produce different perturbations"

    def test_compute_all_deterministic(self, h_clean, h_small_pert):
        results = [compute_all(h_clean, h_small_pert) for _ in range(3)]
        for key in ["I", "q", "Z"]:
            np.testing.assert_array_equal(
                results[0][key], results[1][key],
                err_msg=f"{key}: Run1 != Run2"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGGREGATE Z
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateZ:

    def test_uniform_mean(self):
        layers = [np.array([0.1, 0.2, 0.3]),
                  np.array([0.3, 0.4, 0.5]),
                  np.array([0.5, 0.6, 0.7])]
        Z = aggregate_Z(layers)
        expected = np.array([0.3, 0.4, 0.5])
        np.testing.assert_allclose(Z, expected, atol=1e-10)

    def test_custom_weights(self):
        layers = [np.ones(5) * 1.0, np.ones(5) * 0.0]
        weights = np.array([0.8, 0.2])
        Z = aggregate_Z(layers, layer_weights=weights)
        np.testing.assert_allclose(Z, np.ones(5) * 0.8, atol=1e-10)

    def test_single_layer_passthrough(self):
        z = np.array([0.1, 0.5, 0.9])
        Z = aggregate_Z([z])
        np.testing.assert_array_equal(Z, z)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PEAK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestPeakDetection:

    def test_no_peaks_flat_signal(self):
        Z = np.ones(30) * 0.5
        peaks = detect_peaks(Z)
        assert peaks == [], "Flat signal must produce zero peaks"

    def test_single_spike_detected(self):
        """Synthetic spike at token 15 → center within ±2 tokens."""
        Z = np.zeros(40)
        Z[14] = 1.5
        Z[15] = 3.0   # spike center
        Z[16] = 1.5
        peaks = detect_peaks(Z, multiplier=1.5)
        assert len(peaks) >= 1, "Must detect at least one peak"
        assert abs(peaks[0]["center"] - 15.0) <= 2.0, \
            f"Peak center {peaks[0]['center']:.2f} too far from injected spike at 15"

    def test_two_separate_peaks(self):
        Z = np.zeros(60)
        Z[10] = 3.0   # first spike
        Z[45] = 3.0   # second spike
        peaks = detect_peaks(Z, multiplier=1.5)
        assert len(peaks) == 2, f"Expected 2 peaks, got {len(peaks)}"

    def test_peak_fields_valid(self):
        Z = np.zeros(30)
        Z[10:13] = np.array([1.0, 2.0, 1.0])
        peaks = detect_peaks(Z)
        if peaks:
            p = peaks[0]
            assert p["start"] <= p["end"]
            assert p["width"] == p["end"] - p["start"] + 1
            assert p["max_z"] >= p["mean_z"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. LEAD TIME
# ─────────────────────────────────────────────────────────────────────────────

class TestLeadTime:

    def test_positive_lead_time(self):
        """Peak center=5, hallucination starts at 8 → lead_time=3."""
        peaks = [{"center": 5.0, "start": 4, "end": 6,
                  "max_z": 1.5, "mean_z": 1.2, "width": 3}]
        lt = compute_lead_time(peaks, hallucination_start=8)
        assert lt is not None
        assert lt == pytest.approx(3.0), f"Expected lead_time=3, got {lt}"

    def test_zero_lead_time(self):
        """Peak exactly at hallucination start → lead_time=0."""
        peaks = [{"center": 5.0, "start": 5, "end": 5,
                  "max_z": 1.0, "mean_z": 1.0, "width": 1}]
        lt = compute_lead_time(peaks, hallucination_start=5)
        assert lt == pytest.approx(0.0)

    def test_no_peaks_returns_none(self):
        lt = compute_lead_time([], hallucination_start=10)
        assert lt is None

    def test_uses_closest_peak_before_hallucination(self):
        """Two peaks before hallucination → use the one closest (highest center)."""
        peaks = [
            {"center": 2.0, "start": 1, "end": 3, "max_z": 1.5, "mean_z": 1.2, "width": 3},
            {"center": 7.0, "start": 6, "end": 8, "max_z": 1.8, "mean_z": 1.5, "width": 3},
        ]
        lt = compute_lead_time(peaks, hallucination_start=10)
        assert lt == pytest.approx(3.0), \
            "Should use peak at 7.0 (closest before hallucination at 10)"


# ─────────────────────────────────────────────────────────────────────────────
# 8. CREATIVE vs. ERROR INSTABILITY (core research insight)
# ─────────────────────────────────────────────────────────────────────────────

class TestCreativeVsErrorDistinction:
    """
    Core hypothesis from research:
        Creative instability: High I, HIGH q  → high Z_I, low Z_dir
        Erroneous divergence: High I, LOW  q  → high Z on both
    """

    def test_creative_signal_pattern(self):
        """High I + High q → composite Z is moderate (not extreme)."""
        I = np.array([0.8])
        q = np.array([0.85])  # high cosine similarity = creative, not error
        Z = composite(I, q, w_I=0.5)
        # Z = 0.5*0.8 + 0.5*(1-0.85) = 0.4 + 0.075 = 0.475
        np.testing.assert_allclose(Z, 0.475, atol=1e-6)

    def test_hallucination_signal_pattern(self):
        """High I + Low q → composite Z is large (error signal)."""
        I = np.array([0.8])
        q = np.array([0.15])  # low cosine = direction lost = hallucination
        Z = composite(I, q, w_I=0.5)
        # Z = 0.5*0.8 + 0.5*(1-0.15) = 0.4 + 0.425 = 0.825
        np.testing.assert_allclose(Z, 0.825, atol=1e-6)

    def test_hallucination_Z_exceeds_creative_Z(self):
        I_both = np.array([0.8])
        q_creative = np.array([0.85])
        q_hallucinates = np.array([0.15])
        Z_creative = composite(I_both, q_creative)
        Z_hallucinates = composite(I_both, q_hallucinates)
        assert Z_hallucinates[0] > Z_creative[0], \
            "Hallucination Z must exceed creative Z when I is equal"
