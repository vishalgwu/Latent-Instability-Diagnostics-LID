"""
baselines/base.py
=================
Abstract base class for ALL hallucination detectors.

CRITICAL DESIGN REQUIREMENT (from 12-Week Plan, Part 1):
    Every baseline AND lid must implement the same interface:
        score(hidden_states, tokens) -> per_token_score

    This ensures fair comparison:
    - Same dataset splits
    - Same tokenization  
    - Same annotation labels
    - Identical metric computation

Output contract:
    All detectors return shape [B, T] — batch × sequence_length
    Scores are comparable across methods (higher = more likely hallucination)

Author: MIT lid Research Team
Week  : 1 (skeleton) → Baseline implementations Weeks 2-3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DetectorConfig:
    """Shared configuration for all detectors."""
    name: str                              # Human-readable name (e.g., "DoLA", "lid")
    device: str = "cuda"                   # Target device
    batch_size: int = 1                    # Inference batch size
    seed: int = 42                         # Reproducibility seed
    # Overhead tracking
    log_overhead: bool = True              # Measure wall-clock overhead vs clean inference
    extra: dict = field(default_factory=dict)  # Method-specific config


@dataclass
class DetectorOutput:
    """Standardized output from all detectors."""
    scores: torch.Tensor                   # Per-token scores [B, T] (higher = more suspicious)
    tokens: list[list[str]]                # Decoded tokens [B, T]
    metadata: dict = field(default_factory=dict)  # Method-specific (e.g., layer scores for lid)
    overhead_ratio: Optional[float] = None # Wall-clock overhead vs clean inference


class BaseDetector(ABC):
    """
    Abstract base class for hallucination detectors.

    REQUIRED: All subclasses must implement:
        - score()       : Main scoring method
        - name          : Property returning method name

    OPTIONAL overrides:
        - calibrate()   : Fit any hyperparameters on calibration data
        - reset()       : Clear any cached state between examples
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._is_calibrated = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this detector (e.g., 'dola', 'lid', 'entropy')."""
        ...

    @abstractmethod
    def score(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DetectorOutput:
        """
        Score each token for hallucination likelihood.

        Args:
            model          : HuggingFace CausalLM model
            tokenizer      : HuggingFace tokenizer
            input_ids      : Input token IDs [B, T]
            attention_mask : Optional mask [B, T]

        Returns:
            DetectorOutput with scores [B, T]
            Higher score = model more likely to hallucinate at this token.
        """
        ...

    def calibrate(self, calibration_data: list[dict]) -> None:
        """
        Optional: fit method-specific hyperparameters.
        Default: no-op (most baselines are hyperparameter-free at inference).
        """
        self._is_calibrated = True

    def reset(self) -> None:
        """Clear cached state between examples (for stateful methods)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
