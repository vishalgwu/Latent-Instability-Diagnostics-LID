"""
tests/test_metrics.py
=====================
Unit tests for lid/metrics.py

TEST CONTRACT (from 12-Week Plan, Week 4 KPIs):
    - δ = 0  →  I = 0, q = 1.0, Z = 0  (degenerate case)
    - δ with known magnitude → expected I (analytically computable)
    - q ∈ [-1, 1] always
    - Z ∈ [0, ~2] range
    - Determinism: same (seed, α, tokens) → identical output × 3 runs

Run with:
    pytest tests/test_metrics.py -v

Author: MIT LID Research Team
Week  : 1 (skeleton) → Green by Week 4
"""

import pytest
import torch
from lid.metrics import instability, alignment, composite, aggregate_Z, compute_all
from lid.perturb import inject_noise


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def hidden_clean():
    """Fake hidden state: batch=1, seq_len=5, d_model=16."""
    torch.manual_seed(0)
    return torch.randn(5, 16)


@pytest.fixture
def hidden_zero_pert(hidden_clean):
    """Perturbed = clean (zero perturbation)."""
    return hidden_clean.clone()


# ── DEGENERATE CASE: δ = 0 ───────────────────────────────────────────────────

class TestDegenerateZeroPerturbation:
    """When δ = 0, all instability metrics must be zero / neutral."""

    def test_instability_is_zero(self, hidden_clean, hidden_zero_pert):
        I = instability(hidden_clean, hidden_zero_pert)
        assert torch.allclose(I, torch.zeros_like(I), atol=1e-6), \
            f"Expected I=0 when δ=0, got max={I.max().item():.6f}"

    def test_alignment_is_one(self, hidden_clean, hidden_zero_pert):
        q = alignment(hidden_clean, hidden_zero_pert)
        assert torch.allclose(q, torch.ones_like(q), atol=1e-5), \
            f"Expected q=1 when δ=0, got min={q.min().item():.6f}"

    def test_composite_is_zero(self, hidden_clean, hidden_zero_pert):
        I = instability(hidden_clean, hidden_zero_pert)
        q = alignment(hidden_clean, hidden_zero_pert)
        Z = composite(I, q, w_I=0.5)
        assert torch.allclose(Z, torch.zeros_like(Z), atol=1e-5), \
            f"Expected Z=0 when δ=0, got max={Z.max().item():.6f}"

    def test_inject_noise_zero_alpha(self, hidden_clean):
        h_pert = inject_noise(hidden_clean, alpha=0.0, seed=42)
        assert torch.allclose(hidden_clean, h_pert), \
            "inject_noise with alpha=0 must return identical tensor"


# ── RANGE CHECKS ─────────────────────────────────────────────────────────────

class TestValueRanges:
    """Validate that all metrics stay in documented ranges."""

    def test_alignment_range(self, hidden_clean):
        torch.manual_seed(1)
        h_pert = hidden_clean + torch.randn_like(hidden_clean) * 0.5
        q = alignment(hidden_clean, h_pert)
        assert q.min() >= -1.0 - 1e-6, f"q below -1: {q.min().item()}"
        assert q.max() <= 1.0 + 1e-6, f"q above 1: {q.max().item()}"

    def test_instability_nonneg(self, hidden_clean):
        torch.manual_seed(2)
        h_pert = hidden_clean + torch.randn_like(hidden_clean) * 0.2
        I = instability(hidden_clean, h_pert)
        assert (I >= 0).all(), f"I has negative values: {I.min().item()}"

    def test_composite_nonneg(self, hidden_clean):
        torch.manual_seed(3)
        h_pert = hidden_clean + torch.randn_like(hidden_clean) * 0.3
        I = instability(hidden_clean, h_pert)
        q = alignment(hidden_clean, h_pert)
        Z = composite(I, q, w_I=0.5)
        assert (Z >= 0).all(), f"Z has negative values: {Z.min().item()}"


# ── DETERMINISM ──────────────────────────────────────────────────────────────

class TestDeterminism:
    """Same (seed, alpha, input) → bitwise-identical output × 3 runs."""

    def test_inject_noise_deterministic(self, hidden_clean):
        results = [
            inject_noise(hidden_clean, alpha=0.05, seed=42)
            for _ in range(3)
        ]
        assert torch.equal(results[0], results[1]), "Run 1 != Run 2"
        assert torch.equal(results[1], results[2]), "Run 2 != Run 3"

    def test_compute_all_deterministic(self, hidden_clean):
        h_pert = inject_noise(hidden_clean, alpha=0.05, seed=42)
        results = [compute_all(hidden_clean, h_pert) for _ in range(3)]
        for key in ["I", "q", "Z"]:
            assert torch.equal(results[0][key], results[1][key]), \
                f"{key}: Run 1 != Run 2"


# ── AGGREGATE Z ──────────────────────────────────────────────────────────────

class TestAggregateZ:
    """Test layer aggregation."""

    def test_uniform_weights_equal_mean(self, hidden_clean):
        Z_layers = [
            torch.full((5,), float(v)) for v in [0.1, 0.3, 0.5]
        ]
        Z_agg = aggregate_Z(Z_layers)
        expected = torch.tensor([0.3] * 5)
        assert torch.allclose(Z_agg, expected, atol=1e-6)

    def test_custom_weights(self, hidden_clean):
        Z_layers = [torch.ones(5) * v for v in [1.0, 0.0]]
        weights = torch.tensor([0.8, 0.2])
        Z_agg = aggregate_Z(Z_layers, layer_weights=weights)
        expected = torch.ones(5) * 0.8
        assert torch.allclose(Z_agg, expected, atol=1e-6)
