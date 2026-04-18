# Week 1 — Friday Advisor Report
**Date:** [Fill date]  
**Submitted by:** LID Research Team  
**Advisor:** [Name]

---

## KPI Scorecard

| # | KPI | Target | Achieved | Pass/Fail |
|---|-----|--------|----------|-----------|
| 1 | Priority-1 theory papers read + 1-page synthesis each | 4/4 | [X/4] | [P/F] |
| 2 | Repo `lid-research/` created with CI scaffold | CI green | [status] | [P/F] |
| 3 | Compute env validated (Llama-7B loads + generates) | 5-example smoke test | [status] | [P/F] |
| 4 | W&B + Hydra + DVC initialized | First dummy run logged | [status] | [P/F] |

**KPIs passed this week: [X] / 4**

---

## Metrics Table

| Metric | Value | Notes |
|--------|-------|-------|
| Unit tests passing | 31 / 31 | All degenerate cases green |
| CI pipeline | ✅ Green | numpy-only, no GPU required |
| Repo structure | Complete | 35 directories, all `__init__.py` in place |
| Llama-7B smoke test | [pending] | Requires A100 access |
| W&B dummy run | [pending] | |

---

## Deliverables Completed

- [x] Repository `lid-research/` with full directory structure
- [x] `lid/metrics_np.py` — all formulas (I, q, Z) implemented + tested
- [x] `lid/perturb.py` — deterministic Gaussian injection (stub for torch)
- [x] `lid/peak.py` — adaptive threshold peak detection
- [x] `baselines/base.py` — unified `BaseDetector` abstract API
- [x] `tests/test_metrics_np.py` — 31 unit tests, all green
- [x] CI `.github/workflows/ci.yml` — runs without GPU
- [x] `INSTALL.md` — reproducible setup guide
- [x] `requirements.txt`, `setup.py`
- [ ] 4 × 1-page theory synthesis (in progress — see `docs/related-theory/`)
- [ ] Llama-7B inference smoke test (blocked on compute allocation)
- [ ] W&B + DVC first run (blocked on API keys)

---

## Red Flags This Week

*None currently. Flag immediately if:*
- Compute access not available by Day 3 → escalate to MIT SuperCloud admin
- Physics framing concerns from advisor → soften all subsequent writing NOW

---

## Next Week Priorities (Week 2)

1. Reproduce **DoLA** on TruthfulQA-dev (Llama-7B) → AUROC within ±2pp of Chuang et al.
2. Reproduce **LSD** (arXiv:2510.04933) → closest prior work, must beat on lead time
3. Reproduce **SSP** (Luo 2025) → closest competitor, match reported numbers
4. Draft annotation guidelines v1 → `docs/annotation/guidelines-v1.md`

---

## Questions for Advisor

1. [Any concerns about the physics framing in README/formulas?]
2. [Compute allocation ETA?]
3. [Preferred W&B workspace for shared visibility?]
