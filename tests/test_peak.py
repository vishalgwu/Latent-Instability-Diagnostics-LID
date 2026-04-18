"""
tests/test_peak.py
==================
Unit tests for lid/peak.py

TEST CONTRACT (Week 4 KPIs):
    - Peak on synthetic spike: center within ±2 tokens of injected spike
    - All-zero Z → empty peak list
    - Lead time positive when peak precedes hallucination

Run with:
    pytest tests/test_peak.py -v
"""

import pytest
import torch
from lid.peak import detect_peaks, adaptive_threshold, compute_lead_time, Peak


class TestAdaptiveThreshold:
    def test_constant_signal_threshold(self):
        Z = torch.ones(20) * 0.5
        # std=0, threshold = mean = 0.5
        t = adaptive_threshold(Z, multiplier=1.5)
        assert abs(t - 0.5) < 1e-5

    def test_threshold_above_mean(self):
        Z = torch.randn(100).abs()
        t = adaptive_threshold(Z, multiplier=1.5)
        mean_z = Z.mean().item()
        assert t > mean_z


class TestDetectPeaks:
    def test_no_peaks_flat_signal(self):
        Z = torch.ones(20) * 0.3
        peaks = detect_peaks(Z)
        assert peaks == [], "Flat signal should produce no peaks"

    def test_single_spike_detected(self):
        """Synthetic spike at token 10 → peak center within ±2."""
        Z = torch.zeros(30)
        Z[9] = 2.0
        Z[10] = 3.0   # spike center
        Z[11] = 2.0

        peaks = detect_peaks(Z, multiplier=1.5)
        assert len(peaks) >= 1, "Expected at least one peak"
        assert abs(peaks[0].center - 10.0) <= 2.0, \
            f"Peak center {peaks[0].center} too far from expected 10.0"

    def test_peak_properties_valid(self):
        Z = torch.zeros(20)
        Z[5:8] = torch.tensor([1.5, 2.0, 1.5])
        peaks = detect_peaks(Z)
        if peaks:
            p = peaks[0]
            assert p.start <= p.end
            assert p.width == p.end - p.start + 1
            assert p.max_z >= p.mean_z


class TestLeadTime:
    def test_positive_lead_time(self):
        """Peak at token 5, hallucination at token 8 → lead_time = 3."""
        peaks = [Peak(center=5.0, start=4, end=6, max_z=1.0, mean_z=0.9, width=3)]
        lead = compute_lead_time(peaks, hallucination_start=8)
        assert lead is not None
        assert lead > 0, f"Expected positive lead time, got {lead}"

    def test_no_peaks_returns_none(self):
        lead = compute_lead_time([], hallucination_start=5)
        assert lead is None
