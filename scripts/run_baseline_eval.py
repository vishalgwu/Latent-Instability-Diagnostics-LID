"""
scripts/run_baseline_eval.py
=============================
Re-run DoLA, LSD, SSP evaluation using HUMAN consensus labels.

This replaces the broken automated annotation from Week 2.
Expected results with proper labels:
    DoLA AUROC ≥ 0.60   (within ±2pp of Chuang et al.)
    LSD  AUROC ≥ 0.58
    SSP  AUROC ≥ 0.58

Usage:
    python scripts/run_baseline_eval.py --model meta-llama/Meta-Llama-3-8B

Author : MIT LID Research Team
Week   : 3 — Corrected Baseline Evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from baselines.dola.detector import DoLADetector
from baselines.lsd.detector  import LSDDetector
from baselines.ssp.detector  import SSPDetector
from evaluation.metrics      import evaluate, EvalResult

CONSENSUS_PATH = ROOT / "data" / "labeled" / "consensus" / "consensus.jsonl"
OUTPUTS_DIR    = ROOT / "outputs"


def load_consensus() -> list:
    if not CONSENSUS_PATH.exists():
        print(f"❌ Consensus labels not found: {CONSENSUS_PATH}")
        print("   Run scripts/compute_agreement.py first.")
        sys.exit(1)

    with open(CONSENSUS_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"✅ Loaded {len(records)} consensus-labeled examples")
    mean_rate = np.mean([r["hallucination_rate"] for r in records])
    print(f"   Mean hallucination rate: {mean_rate:.1%}  (expected 5-15%)")

    if mean_rate > 0.20:
        print("   ⚠️  Rate > 20% — re-check annotation quality before proceeding")
    return records


def run_evaluation(model, tokenizer, records: list) -> dict:
    """Run all 3 baselines and collect results."""
    detectors = [DoLADetector(), LSDDetector(), SSPDetector()]
    results   = {}

    for detector in detectors:
        print(f"\n{'='*55}")
        print(f"  Running: {detector.name.upper()}")
        print(f"{'='*55}")

        all_scores, all_labels = [], []
        times = []

        for i, rec in enumerate(records):
            if i % 10 == 0:
                print(f"  [{i}/{len(records)}]...")

            question = rec["question"]
            prompt   = f"Q: {question}\nA:"
            tokens   = rec["tokens"]
            labels   = np.array(rec["consensus_labels"], dtype=float)

            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            try:
                # Generate same response as annotation
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=len(tokens),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                t0  = time.perf_counter()
                out = detector.score(model, tokenizer, gen_ids)
                times.append(time.perf_counter() - t0)

                prompt_len = inputs["input_ids"].shape[1]
                scores = out.scores[0, prompt_len:].numpy()

                # Align lengths with annotated tokens
                min_len = min(len(scores), len(labels))
                all_scores.append(scores[:min_len])
                all_labels.append(labels[:min_len])

            except Exception as e:
                print(f"  ⚠️  Error on {rec['id']}: {e}")
                continue

        flat_scores = np.concatenate(all_scores)
        flat_labels = np.concatenate(all_labels)

        result = evaluate(
            scores        = flat_scores,
            labels        = flat_labels,
            method        = detector.name,
            dataset       = "TruthfulQA-dev-consensus",
            model         = "Meta-Llama-3-8B",
            overhead_ratio= None,
            compute_ci    = True,
            n_bootstrap   = 1000,
        )
        results[detector.name] = result
        print(result.summary_line())

        if result.below_threshold:
            print("  ⚠️  RED FLAG: AUROC < 0.60 after proper annotation")
            print("     Escalate to advisor — may indicate signal is weak")

    return results


def print_results_table(results: dict):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("  WEEK 3 CORRECTED BASELINE RESULTS")
    print("  (Using human consensus labels — Fleiss κ ≥ 0.60)")
    print("=" * 70)
    print(f"{'Method':<8} {'AUROC':>7} {'95% CI':>20} {'AUPRC':>7} "
          f"{'FPR@80':>8} {'Status':>8}")
    print("-" * 70)

    for name, r in results.items():
        ci = f"[{r.auroc_ci[0]:.3f}, {r.auroc_ci[1]:.3f}]"
        status = "✅ PASS" if not r.below_threshold else "⚠️  FAIL"
        print(
            f"{name.upper():<8} "
            f"{r.auroc or 0:>7.3f} "
            f"{ci:>20} "
            f"{r.auprc or 0:>7.3f} "
            f"{r.fpr_at_tpr80 or 0:>8.3f} "
            f"{status:>8}"
        )

    print("=" * 70)
    print("\nTargets (from execution plan):")
    print("  DoLA: AUROC ≥ 0.60 | LSD: AUROC ≥ 0.58 | SSP: AUROC ≥ 0.58")
    print("\nThese are the bars LID must clear in Week 5.")


def save_results(results: dict):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = {
        "week":       3,
        "annotation": "human_consensus_3_annotators",
        "results": {
            name: {
                "auroc":          r.auroc,
                "auprc":          r.auprc,
                "fpr_at_tpr80":   r.fpr_at_tpr80,
                "auroc_ci":       list(r.auroc_ci),
                "n_tokens":       r.n_tokens,
                "hallucination_rate": r.hallucination_rate,
                "below_threshold":r.below_threshold,
            }
            for name, r in results.items()
        }
    }
    path = OUTPUTS_DIR / "week3_corrected_baseline_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True, token=token,
    )
    model.eval()
    print("✅ Model loaded")

    records = load_consensus()
    results = run_evaluation(model, tokenizer, records)
    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
