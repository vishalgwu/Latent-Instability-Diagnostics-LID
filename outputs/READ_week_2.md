# Outputs

## week2_baseline_results.json
- Date: 2026-04-19
- Model: Llama-3-8B, TruthfulQA-dev-100
- Status: ANNOTATION BROKEN — hallucination_rate=41% (expected 5-15%)
- DoLA AUROC: 0.516 | SSP: 0.536 | LSD: 0.309
- Root cause: automated token annotation too coarse
- Fix: human annotation pipeline (Week 3)

## tables/week2_table.csv
- Same run, CSV format for paper tables