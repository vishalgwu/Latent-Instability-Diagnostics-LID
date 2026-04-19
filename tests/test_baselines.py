"""
tests/test_baselines.py
=======================
Unit tests for DoLA, LSD, SSP baseline detectors.

These tests use a MOCK MODEL — no GPU needed, runs on any machine.
The mock model returns deterministic random tensors with correct shapes.

Tests validate:
    - Output shape contract: scores [B, T]
    - Score range: finite, non-negative values
    - Determinism: same inputs → same scores
    - API contract: all detectors follow BaseDetector interface
    - DoLA JSD range: [0, 1]
    - LSD drift range: [0, 2]
    - SSP: perturbed ≠ clean (noise is actually applied)

Run:
    pytest tests/test_baselines.py -v --tb=short

Author : MIT LID Research Team
Week   : 2
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


# ─────────────────────────────────────────────────────────────────────────────
# MOCK MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

class MockLMHead(nn.Module):
    """Fake lm_head: projects d_model → vocab_size."""
    def __init__(self, d_model=64, vocab_size=100):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class MockLayer(nn.Module):
    """Fake transformer layer: adds a small deterministic transform."""
    def __init__(self, d_model=64, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        # Small mixing matrix — makes each layer produce slightly different h
        self.mix = nn.Linear(d_model, d_model, bias=False)
        # Init as near-identity + small perturbation
        nn.init.eye_(self.mix.weight)
        with torch.no_grad():
            self.mix.weight += torch.randn_like(self.mix.weight) * 0.01

    def forward(self, hidden_states, **kwargs):
        out = self.mix(hidden_states)
        return (out,)  # return tuple like real transformer layers


class MockNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:])


class MockEmbedTokens(nn.Module):
    def __init__(self, vocab_size=100, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class MockTransformerModel(nn.Module):
    """Fake model.model (the inner transformer)."""
    def __init__(self, n_layers=8, d_model=64, vocab_size=100):
        super().__init__()
        self.layers = nn.ModuleList([
            MockLayer(d_model, i) for i in range(n_layers)
        ])
        self.norm = MockNorm()
        self.embed_tokens = MockEmbedTokens(vocab_size, d_model)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(input_ids)

        for layer in self.layers:
            h = layer(h)[0]

        return h


class MockCausalLM(nn.Module):
    """
    Complete mock CausalLM that mimics HuggingFace model interface.
    All forward passes are deterministic given the same input.
    """
    def __init__(self, n_layers=8, d_model=64, vocab_size=100):
        super().__init__()
        torch.manual_seed(0)
        self.model = MockTransformerModel(n_layers, d_model, vocab_size)
        self.lm_head = MockLMHead(d_model, vocab_size)

        # Config (mimics HuggingFace config)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=d_model,
            vocab_size=vocab_size,
        )

    def forward(self, input_ids=None, inputs_embeds=None,
                attention_mask=None, **kwargs):
        h = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits, hidden_states=None)

    def generate(self, input_ids, max_new_tokens=5, **kwargs):
        """Simple greedy generation mock."""
        device = input_ids.device
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=input_ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def make_mock_tokenizer(vocab_size=100):
    """Create a minimal mock tokenizer."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.decode = lambda ids, **kw: " ".join([f"tok{i}" for i in ids])
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    torch.manual_seed(42)
    model = MockCausalLM(n_layers=8, d_model=64, vocab_size=100)
    model.eval()
    return model

@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer()

@pytest.fixture
def input_ids():
    """Batch of 1 sequence, length 6."""
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 6))


# ─────────────────────────────────────────────────────────────────────────────
# DoLA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDoLA:
    def test_import(self):
        from baselines.dola.detector import DoLADetector, DoLAConfig
        assert DoLADetector is not None

    def test_name(self):
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        assert d.name == "dola"

    def test_output_shape(self, mock_model, mock_tokenizer, input_ids):
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        B, T = input_ids.shape
        assert out.scores.shape == (B, T), \
            f"Expected scores shape ({B},{T}), got {out.scores.shape}"

    def test_scores_finite(self, mock_model, mock_tokenizer, input_ids):
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.isfinite(out.scores).all(), \
            "DoLA scores contain NaN or Inf"

    def test_scores_nonnegative(self, mock_model, mock_tokenizer, input_ids):
        """JSD is always non-negative."""
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert (out.scores >= -1e-6).all(), \
            f"DoLA scores contain negative values: {out.scores.min()}"

    def test_scores_max_one(self, mock_model, mock_tokenizer, input_ids):
        """JSD is bounded by 1 (or close to it for numerical reasons)."""
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert (out.scores <= 1.0 + 1e-4).all(), \
            f"DoLA JSD score exceeds 1.0: {out.scores.max()}"

    def test_deterministic(self, mock_model, mock_tokenizer, input_ids):
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out1 = d.score(mock_model, mock_tokenizer, input_ids)
        out2 = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.allclose(out1.scores, out2.scores), \
            "DoLA is not deterministic"

    def test_metadata_contains_premature_layer(self, mock_model, mock_tokenizer, input_ids):
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert "premature_layer" in out.metadata

    def test_premature_layer_ratio(self, mock_model, mock_tokenizer, input_ids):
        """premature_layer_ratio=0.5 → layer 4 for 8-layer model."""
        from baselines.dola.detector import DoLADetector, DoLAConfig
        cfg = DoLAConfig(name="dola", premature_layer_ratio=0.5)
        d = DoLADetector(cfg)
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert out.metadata["premature_layer"] == 4

    def test_jsd_between_identical_dists_is_zero(self):
        """JSD(p, p) = 0."""
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        p = torch.softmax(torch.randn(1, 10, 100), dim=-1)
        jsd = d._jsd(p, p)
        assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-5), \
            f"JSD(p,p) should be 0, got {jsd.max()}"

    def test_jsd_between_opposite_dists_near_one(self):
        """JSD between [1,0,...] and [0,1,...] should be close to 1."""
        from baselines.dola.detector import DoLADetector
        d = DoLADetector()
        p = torch.zeros(1, 1, 100)
        p[0, 0, 0] = 1.0
        q = torch.zeros(1, 1, 100)
        q[0, 0, 1] = 1.0
        jsd = d._jsd(p, q)
        assert jsd.item() > 0.5, f"JSD between opposite dists should be >0.5, got {jsd}"


# ─────────────────────────────────────────────────────────────────────────────
# LSD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLSD:
    def test_import(self):
        from baselines.lsd.detector import LSDDetector, LSDConfig
        assert LSDDetector is not None

    def test_name(self):
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        assert d.name == "lsd"

    def test_output_shape(self, mock_model, mock_tokenizer, input_ids):
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        B, T = input_ids.shape
        assert out.scores.shape == (B, T)

    def test_scores_finite(self, mock_model, mock_tokenizer, input_ids):
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.isfinite(out.scores).all()

    def test_drift_nonnegative(self, mock_model, mock_tokenizer, input_ids):
        """Drift = 1 - cosine ≥ 0 for any two vectors."""
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert (out.scores >= -1e-5).all(), \
            f"LSD drift contains negative: {out.scores.min()}"

    def test_deterministic(self, mock_model, mock_tokenizer, input_ids):
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        out1 = d.score(mock_model, mock_tokenizer, input_ids)
        out2 = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.allclose(out1.scores, out2.scores)

    def test_cosine_drift_identical_vectors(self):
        """drift(h, h) = 0."""
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        h = torch.randn(1, 5, 64)
        drift = d._cosine_drift(h, h)
        assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-5)

    def test_cosine_drift_orthogonal_vectors(self):
        """drift between orthogonal vectors = 1.0."""
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        h1 = torch.zeros(1, 1, 4)
        h1[0, 0, 0] = 1.0
        h2 = torch.zeros(1, 1, 4)
        h2[0, 0, 1] = 1.0
        drift = d._cosine_drift(h1, h2)
        assert drift.item() == pytest.approx(1.0, abs=1e-5)

    def test_aggregation_modes(self, mock_model, mock_tokenizer, input_ids):
        from baselines.lsd.detector import LSDDetector, LSDConfig
        for agg in ["mean", "max", "weighted_mean"]:
            cfg = LSDConfig(name="lsd", aggregation=agg)
            d = LSDDetector(cfg)
            out = d.score(mock_model, mock_tokenizer, input_ids)
            assert out.scores.shape == input_ids.shape, \
                f"aggregation={agg} produced wrong shape"

    def test_metadata_has_n_pairs(self, mock_model, mock_tokenizer, input_ids):
        from baselines.lsd.detector import LSDDetector
        d = LSDDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert "n_layer_pairs" in out.metadata
        assert out.metadata["n_layer_pairs"] == 7  # 8 layers → 7 pairs


# ─────────────────────────────────────────────────────────────────────────────
# SSP TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSSP:
    def test_import(self):
        from baselines.ssp.detector import SSPDetector, SSPConfig
        assert SSPDetector is not None

    def test_name(self):
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        assert d.name == "ssp"

    def test_output_shape(self, mock_model, mock_tokenizer, input_ids):
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        B, T = input_ids.shape
        assert out.scores.shape == (B, T)

    def test_scores_finite(self, mock_model, mock_tokenizer, input_ids):
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.isfinite(out.scores).all()

    def test_scores_nonnegative(self, mock_model, mock_tokenizer, input_ids):
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert (out.scores >= -1e-6).all()

    def test_deterministic(self, mock_model, mock_tokenizer, input_ids):
        """Same seed → same perturbation → same scores."""
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        out1 = d.score(mock_model, mock_tokenizer, input_ids)
        out2 = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.allclose(out1.scores, out2.scores)

    def test_different_seeds_different_scores(self, mock_model, mock_tokenizer, input_ids):
        """Different seeds → different perturbations → different scores."""
        from baselines.ssp.detector import SSPDetector, SSPConfig
        cfg1 = SSPConfig(name="ssp", perturb_seed=42)
        cfg2 = SSPConfig(name="ssp", perturb_seed=99)
        d1 = SSPDetector(cfg1)
        d2 = SSPDetector(cfg2)
        out1 = d1.score(mock_model, mock_tokenizer, input_ids)
        out2 = d2.score(mock_model, mock_tokenizer, input_ids)
        assert not torch.allclose(out1.scores, out2.scores), \
            "Different seeds should produce different SSP scores"

    def test_zero_alpha_scores_near_zero(self, mock_model, mock_tokenizer, input_ids):
        """alpha=0 → no perturbation → JSD(p_clean, p_clean) = 0."""
        from baselines.ssp.detector import SSPDetector, SSPConfig
        cfg = SSPConfig(name="ssp", alpha=0.0)
        d = SSPDetector(cfg)
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert torch.allclose(out.scores, torch.zeros_like(out.scores), atol=1e-4), \
            f"alpha=0 should give near-zero scores, got {out.scores.max()}"

    def test_metadata_contains_alpha(self, mock_model, mock_tokenizer, input_ids):
        from baselines.ssp.detector import SSPDetector
        d = SSPDetector()
        out = d.score(mock_model, mock_tokenizer, input_ids)
        assert "alpha" in out.metadata


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED API CONTRACT TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedAPI:
    """
    All 3 baselines must satisfy the same output contract.
    This is critical for fair comparison with LID.
    """

    @pytest.mark.parametrize("detector_class,module", [
        ("DoLADetector", "baselines.dola.detector"),
        ("LSDDetector",  "baselines.lsd.detector"),
        ("SSPDetector",  "baselines.ssp.detector"),
    ])
    def test_scores_shape_contract(self, detector_class, module,
                                    mock_model, mock_tokenizer, input_ids):
        """All detectors: scores.shape == input_ids.shape."""
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, detector_class)
        detector = cls()
        out = detector.score(mock_model, mock_tokenizer, input_ids)
        assert out.scores.shape == input_ids.shape, \
            f"{detector_class} shape mismatch: {out.scores.shape} vs {input_ids.shape}"

    @pytest.mark.parametrize("detector_class,module", [
        ("DoLADetector", "baselines.dola.detector"),
        ("LSDDetector",  "baselines.lsd.detector"),
        ("SSPDetector",  "baselines.ssp.detector"),
    ])
    def test_scores_finite_contract(self, detector_class, module,
                                     mock_model, mock_tokenizer, input_ids):
        """All detectors: no NaN or Inf in scores."""
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, detector_class)
        detector = cls()
        out = detector.score(mock_model, mock_tokenizer, input_ids)
        assert torch.isfinite(out.scores).all(), \
            f"{detector_class} has NaN/Inf in scores"

    @pytest.mark.parametrize("detector_class,module", [
        ("DoLADetector", "baselines.dola.detector"),
        ("LSDDetector",  "baselines.lsd.detector"),
        ("SSPDetector",  "baselines.ssp.detector"),
    ])
    def test_has_name_property(self, detector_class, module):
        """All detectors must have a name property."""
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, detector_class)
        detector = cls()
        assert isinstance(detector.name, str)
        assert len(detector.name) > 0
