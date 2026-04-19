"""
scripts/compute_agreement.py
=============================
Compute inter-annotator agreement (Fleiss' κ) across 3 annotators.

Week 3 KPI: Fleiss' κ ≥ 0.60
    If κ < 0.60 → annotation guidelines are ambiguous → DO NOT proceed
    If κ ≥ 0.60 → labels are reliable → proceed to baseline re-evaluation

Fleiss' κ formula:
    κ = (P̄ - P̄_e) / (1 - P̄_e)

    P̄   = mean proportion of agreement across all tokens
    P̄_e = expected agreement by chance (from marginal proportions)

Usage:
    python scripts/compute_agreement.py

Output:
    - Fleiss κ score
    - Per-example agreement heatmap
    - Disagreement analysis (top 20 tokens annotators disagreed on)
    - Merged consensus labels (majority vote)
    - Saves to data/labeled/consensus/consensus.jsonl

Author : MIT LID Research Team
Week   : 3 — Annotation Pipeline
"""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
LABELED    = DATA_DIR / "labeled"
CONSENSUS  = DATA_DIR / "labeled" / "consensus"
CONSENSUS.mkdir(parents=True, exist_ok=True)

N_ANNOTATORS = 3
N_CATEGORIES = 2   # 0=correct, 1=hallucinated


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_annotations() -> dict[str, dict]:
    """
    Load annotations from all 3 annotators.

    Returns:
        dict: {example_id → {annotator_id → labels_list}}
    """
    all_data = {}

    for ann_dir in sorted(LABELED.glob("annotator_*")):
        ann_id = ann_dir.name.replace("annotator_", "")
        ann_file = ann_dir / "annotations.jsonl"

        if not ann_file.exists():
            print(f"  ⚠️  No annotations file for {ann_id}")
            continue

        with open(ann_file) as f:
            for line in f:
                rec = json.loads(line.strip())
                ex_id = rec["id"]
                if ex_id not in all_data:
                    all_data[ex_id] = {
                        "question":    rec["question"],
                        "best_answer": rec["best_answer"],
                        "response":    rec["response"],
                        "tokens":      rec["tokens"],
                        "annotations": {},
                    }
                all_data[ex_id]["annotations"][ann_id] = rec["labels"]

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# FLEISS' KAPPA
# ─────────────────────────────────────────────────────────────────────────────

def fleiss_kappa(all_data: dict) -> dict:
    """
    Compute Fleiss' κ across all annotated tokens.

    Method:
        For each token position i across all examples:
            n_ij = number of annotators who assigned category j to token i
        
        P_i  = (1 / n*(n-1)) * Σ_j n_ij*(n_ij - 1)  [per-token agreement]
        P_bar = mean(P_i)                              [mean agreement]
        p_j  = (1 / N*n) * Σ_i n_ij                  [marginal probability of cat j]
        P_e  = Σ_j p_j²                               [expected agreement]
        κ    = (P_bar - P_e) / (1 - P_e)

    Reference: Fleiss (1971) — "Measuring nominal scale agreement among raters"

    Returns:
        dict with kappa, P_bar, P_e, per-category marginals, interpretation
    """
    n = N_ANNOTATORS

    # Only use examples annotated by ALL annotators
    complete = {
        ex_id: ex
        for ex_id, ex in all_data.items()
        if len(ex["annotations"]) == n
    }

    if not complete:
        return {"error": "No examples annotated by all annotators yet"}

    print(f"  Computing κ on {len(complete)} fully-annotated examples")

    # Build rating matrix: rows = tokens, cols = categories
    # Each cell = number of annotators who assigned that category
    P_i_vals  = []
    cat_counts = Counter()   # total assignments per category
    N_tokens   = 0

    for ex_id, ex in complete.items():
        ann_labels = list(ex["annotations"].values())  # list of lists
        n_tokens   = len(ann_labels[0])

        for tok_idx in range(n_tokens):
            # Ratings from all annotators for this token
            ratings = [ann_labels[a][tok_idx] for a in range(n)
                       if tok_idx < len(ann_labels[a])]
            if len(ratings) < n:
                continue

            # Count per category
            counts = Counter(ratings)
            n_ij   = [counts.get(cat, 0) for cat in range(N_CATEGORIES)]

            for cat, c in enumerate(n_ij):
                cat_counts[cat] += c

            # P_i = proportion of agreeing pairs
            P_i = sum(c * (c - 1) for c in n_ij) / (n * (n - 1))
            P_i_vals.append(P_i)
            N_tokens += 1

    if N_tokens == 0:
        return {"error": "No token-level data available"}

    P_bar = float(np.mean(P_i_vals))

    # Marginal proportions p_j
    total_ratings = sum(cat_counts.values())
    p_j = {cat: cat_counts[cat] / total_ratings for cat in range(N_CATEGORIES)}

    # Expected agreement
    P_e = sum(p ** 2 for p in p_j.values())

    # Fleiss' kappa
    if abs(1 - P_e) < 1e-10:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    # Interpretation
    if kappa < 0:
        interpretation = "POOR — less than chance agreement"
    elif kappa < 0.20:
        interpretation = "SLIGHT"
    elif kappa < 0.40:
        interpretation = "FAIR"
    elif kappa < 0.60:
        interpretation = "MODERATE"
    elif kappa < 0.80:
        interpretation = "SUBSTANTIAL ✅"
    else:
        interpretation = "ALMOST PERFECT ✅"

    return {
        "kappa":               round(kappa, 4),
        "P_bar":               round(P_bar, 4),
        "P_e":                 round(P_e, 4),
        "n_tokens_evaluated":  N_tokens,
        "n_examples":          len(complete),
        "category_marginals":  {str(k): round(v, 4) for k, v in p_j.items()},
        "hallucination_rate":  round(p_j.get(1, 0), 4),
        "interpretation":      interpretation,
        "passes_threshold":    kappa >= 0.60,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISAGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def disagreement_analysis(all_data: dict, top_n: int = 20) -> list:
    """
    Find the tokens where annotators most often disagreed.
    Used to refine annotation guidelines.

    Returns:
        List of dicts sorted by disagreement rate, most disagreed first
    """
    n      = N_ANNOTATORS
    disagr = []

    complete = {
        ex_id: ex
        for ex_id, ex in all_data.items()
        if len(ex["annotations"]) == n
    }

    for ex_id, ex in complete.items():
        ann_labels = list(ex["annotations"].values())
        n_tokens   = len(ann_labels[0])
        tokens     = ex["tokens"]

        for tok_idx in range(n_tokens):
            ratings = [ann_labels[a][tok_idx] for a in range(n)
                       if tok_idx < len(ann_labels[a])]
            if len(ratings) < n:
                continue

            # Disagreement = not unanimous
            if len(set(ratings)) > 1:
                disagr.append({
                    "example_id": ex_id,
                    "token_idx":  tok_idx,
                    "token":      tokens[tok_idx] if tok_idx < len(tokens) else "?",
                    "ratings":    ratings,
                    "context":    " ".join(tokens[max(0,tok_idx-3):tok_idx+4]),
                    "question":   ex["question"][:60],
                })

    # Sort by most disagreed-upon token text
    return disagr[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS LABELS (majority vote)
# ─────────────────────────────────────────────────────────────────────────────

def build_consensus(all_data: dict) -> list:
    """
    Build consensus labels via majority vote across all annotators.

    Majority vote: label = most common annotation
    Ties (e.g., 1 vs 1 with 2 annotators): label = 0 (conservative)

    Saves to data/labeled/consensus/consensus.jsonl
    """
    n        = N_ANNOTATORS
    complete = {
        ex_id: ex
        for ex_id, ex in all_data.items()
        if len(ex["annotations"]) == n
    }

    consensus_records = []

    for ex_id, ex in complete.items():
        ann_labels = list(ex["annotations"].values())
        n_tokens   = len(ann_labels[0])

        consensus  = []
        agreement  = []

        for tok_idx in range(n_tokens):
            ratings = [ann_labels[a][tok_idx] for a in range(n)
                       if tok_idx < len(ann_labels[a])]
            if not ratings:
                consensus.append(0)
                agreement.append(False)
                continue

            counts     = Counter(ratings)
            vote       = counts.most_common(1)[0][0]
            is_agreed  = counts.most_common(1)[0][1] >= 2  # at least 2/3 agree

            consensus.append(vote)
            agreement.append(is_agreed)

        n_hall     = sum(consensus)
        hall_rate  = n_hall / max(n_tokens, 1)
        agree_rate = sum(agreement) / max(n_tokens, 1)

        record = {
            "id":                 ex_id,
            "question":           ex["question"],
            "best_answer":        ex["best_answer"],
            "response":           ex["response"],
            "tokens":             ex["tokens"],
            "consensus_labels":   consensus,
            "agreement_mask":     agreement,
            "n_tokens":           n_tokens,
            "n_hallucinated":     n_hall,
            "hallucination_rate": hall_rate,
            "agreement_rate":     agree_rate,
            "n_annotators":       n,
        }
        consensus_records.append(record)

    # Save
    out_path = CONSENSUS / "consensus.jsonl"
    with open(out_path, "w") as f:
        for rec in consensus_records:
            f.write(json.dumps(rec) + "\n")

    print(f"  ✅ Consensus saved: {len(consensus_records)} examples → {out_path}")
    return consensus_records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  LID Research — Inter-Annotator Agreement")
    print("=" * 55)

    # Load all annotations
    print("\n📂 Loading annotations...")
    all_data = load_all_annotations()

    annotator_ids = set()
    for ex in all_data.values():
        annotator_ids.update(ex["annotations"].keys())

    print(f"  Annotators found     : {sorted(annotator_ids)}")
    print(f"  Examples with data   : {len(all_data)}")
    fully_done = sum(1 for ex in all_data.values()
                     if len(ex["annotations"]) == N_ANNOTATORS)
    print(f"  Fully annotated (3/3): {fully_done}")

    if fully_done == 0:
        print("\n  ⚠️  No examples have all 3 annotators yet.")
        print("  Complete annotation for all 3 people first.")
        return

    # Fleiss' κ
    print("\n📐 Computing Fleiss' κ...")
    kappa_result = fleiss_kappa(all_data)

    print(f"\n{'='*55}")
    print(f"  FLEISS' κ  = {kappa_result['kappa']}")
    print(f"  P̄ (observed agreement)  = {kappa_result['P_bar']}")
    print(f"  P_e (expected by chance) = {kappa_result['P_e']}")
    print(f"  Interpretation: {kappa_result['interpretation']}")
    print(f"  Hallucination rate: {kappa_result['hallucination_rate']:.1%}")
    print(f"{'='*55}")

    if kappa_result["passes_threshold"]:
        print("\n  ✅ κ ≥ 0.60 — PASS. Labels are reliable.")
        print("  Proceeding to consensus label generation...")
    else:
        print(f"\n  ❌ κ = {kappa_result['kappa']:.3f} < 0.60 — FAIL")
        print("  DO NOT proceed to baseline evaluation yet.")
        print("  Action: Review disagreement analysis and update guidelines.")

    # Disagreement analysis
    print("\n🔍 Top disagreed tokens (use to refine guidelines):")
    disagr = disagreement_analysis(all_data, top_n=10)
    for i, d in enumerate(disagr[:10]):
        print(f"  {i+1:2d}. Token: '{d['token']}'  Ratings: {d['ratings']}")
        print(f"       Context: ...{d['context']}...")
        print(f"       Q: {d['question']}")

    # Build consensus
    print("\n🗳️  Building consensus labels (majority vote)...")
    consensus = build_consensus(all_data)

    # Final hallucination rate check
    if consensus:
        rates     = [r["hallucination_rate"] for r in consensus]
        mean_rate = float(np.mean(rates))
        print(f"\n  Overall hallucination rate (consensus): {mean_rate:.1%}")
        if 0.05 <= mean_rate <= 0.15:
            print("  ✅ Within expected range (5-15%)")
        else:
            print(f"  ⚠️  Outside expected range — investigate")

    # Save kappa result
    with open(CONSENSUS / "kappa_result.json", "w") as f:
        json.dump(kappa_result, f, indent=2)
    print(f"\n  Results saved to: {CONSENSUS}")


if __name__ == "__main__":
    main()
