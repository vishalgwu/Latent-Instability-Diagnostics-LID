# Latent Instability Diagnostics (LID)
### Pre-Generation Hallucination Detection in Large Language Models

**Institution:** MIT  
**Target Venue:** Google DeepMind | NeurIPS / ICLR 2026  
**Status:** Week 1 — Environment Setup & Theory Grounding  

---

## Overview

Large language models hallucinate at rates of 15–52% on benchmark tasks. Every production-grade detection method today is **post-hoc**: the error is generated first, then caught.

**LID detects hallucinations 1–2 tokens *before* they are emitted**, by measuring the geometric sensitivity of the model's latent manifold to controlled Gaussian perturbation. 

### Key Properties

| Property | LID | Semantic Entropy | DoLA |
|---|---|---|---|
| Detection timing | **Pre-generation** | Post-generation | Post-generation |
| Inference overhead | **~15–25%** | ~500–1000% | ~10% |
| Creative vs. error distinction | **Yes (I+q dual metric)** | No | No |
| Per-model fine-tuning required | **No** | No | No |

---

## Core Hypothesis

> Hallucinations are preceded by a detectable increase in hidden-state sensitivity to perturbation, characterized by a **simultaneous rise in magnitude divergence I(t)** and a **drop in directional alignment q(t)**. This tipping point is observable at least one token before the first hallucinated token is emitted.

---

## Core Formulas

```
σ_l = α · RMS(h_l)                              # Perturbation scale
I(l,t) = ||h_clean - h_pert||₂ / ||h_clean||₂  # Instability score
q(l,t) = cosine(h_clean, h_pert)                # Directional alignment
Z(l,t) = w_I·I + (1-w_I)·(1-q)                # Composite score
threshold = mean(Z) + 1.5·std(Z)               # Peak detection
```

---

## Repository Structure

```
lid-research/
├── lid/                    # Core LID algorithm
│   ├── perturb.py          # Gaussian noise injection
│   ├── metrics.py          # I, q, Z computation
│   ├── peak.py             # Adaptive peak detection
│   ├── pipeline.py         # End-to-end LID scoring
│   └── layers.py           # Layer selection (importance scoring)
├── baselines/              # 8 reproduced baselines (unified API)
│   ├── base.py             # Abstract BaseDetector class
│   ├── dola/               # DoLA (Chuang et al. ICLR 2024)
│   ├── lsd/                # LSD (arXiv:2510.04933) — most similar prior work
│   ├── ssp/                # SSP (Luo 2025) — closest competitor
│   ├── semantic_entropy/   # Semantic Entropy (Farquhar et al. Nature 2024)
│   ├── entropy/            # Standard entropy baseline
│   ├── inside/             # INSIDE (eigenvalue-based)
│   ├── sep/                # SEP (probe-based)
│   └── iti/                # ITI (Li et al. NeurIPS 2024)
├── data/                   # Data pipeline
│   ├── raw/                # Downloaded datasets (never modified)
│   ├── labeled/            # Human-annotated token labels
│   ├── splits/             # Train/val/test splits (frozen)
│   └── loaders/            # Dataset loader classes
├── evaluation/             # Evaluation harness
│   ├── metrics.py          # AUROC, AUPRC, lead time, FPR
│   ├── benchmark.py        # Full benchmark sweep
│   └── visualization.py    # Trajectory plots, ROC curves
├── intervention/           # Phase 3: temperature-adaptive mitigation
├── configs/                # Hydra configuration system
│   ├── model/              # Model configs (llama-7b, llama-70b, mistral-7b)
│   ├── dataset/            # Dataset configs
│   └── experiment/         # Experiment configs
├── notebooks/              # Jupyter analysis notebooks
├── docs/
│   ├── related-theory/     # 1-page synthesis per priority paper
│   ├── annotation/         # Annotation guidelines
│   ├── baseline-reproduction/ # Fidelity reports per baseline
│   └── weekly-reports/     # Friday advisor check-ins
├── tests/                  # pytest unit tests
├── outputs/                # Results (figures, tables, checkpoints)
├── requirements.txt
├── setup.py
└── INSTALL.md
```

---

## Installation

See [INSTALL.md](INSTALL.md) for full setup instructions.

**Quick start (dev node):**
```bash
git clone <repo>
cd lid-research
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"
pytest tests/ -v   # should pass all unit tests before model load
```

---

## 12-Week Timeline

| Week | Theme | Gate |
|------|-------|------|
| 1 | Kick-off, theory, env setup | Repo + CI green ✅ |
| 2 | Baselines I (DoLA, LSD, SSP) | 3 baselines reproduced |
| 3 | Baselines II + annotation pilot | 8 baselines + κ ≥ 0.60 |
| 4 | LID core implementation | I, q, Z, peaks unit-tested |
| 5 | Layer selection + α calibration | Phase 1 GO/NO-GO |
| 6 | Full annotation + anisotropy | LID > DoLA first look |
| 7 | Multi-model eval (70B, Mistral) | Transfer validated |
| 8 | Creative vs. error ablation | AUC ≥ 0.75 |
| 9 | Full benchmark sweep + OOD | Master table complete |
| 10 | Intervention experiments | Δ hallucination ≥ 15pp |
| 11 | Paper draft v1 + red team | Reproducibility test |
| 12 | Final + arXiv + DeepMind | **Submission** |

---

## Success Criteria

- **Phase 1 (Week 8):** AUROC ≥ 0.70, lead time > 0.3 tokens
- **Phase 2 (Week 12):** AUROC ≥ 0.75, generalization drop < 10%
- **Phase 3 (Week 20):** Hallucination reduction ≥ 15%
- **Phase 4 (Week 24):** Manuscript submitted

---

## Weekly Advisor Check-ins

Every Friday: metrics table submitted to advisor.  
Format: see `docs/weekly-reports/template.md`

---

*Realistic success probability: 50–60% for top-venue publication. We document failures as rigorously as successes.*
