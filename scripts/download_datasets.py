"""
scripts/download_datasets.py
=============================
Production-grade dataset downloader for LID Research.

Improvements over v1:
    - retry_load_dataset(): exponential backoff for flaky networks
    - .env token loading for authenticated HF downloads
    - Graceful per-dataset failure (one fail doesn't stop others)
    - validate_record(): rejects empty/malformed examples
    - Unified input/target fields across all datasets
    - HaluEval deduplication by question
    - GSM8K final-answer extraction (from chain-of-thought)
    - Manifest files with split sizes for reproducibility
    - CNN/DailyMail and TruthfulQA dev splits for calibration

Usage:
    python scripts/download_datasets.py           # all datasets
    python scripts/download_datasets.py --dataset truthfulqa

Author : MIT LID Research Team
Week   : 3
"""

# ── Auth: load HF_TOKEN from .env before any HF imports ──────────────────────
from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_jsonl(records: list, path: Path) -> None:
    """Save list of dicts as JSONL with automatic parent dir creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved {len(records):>5} → {path.relative_to(ROOT)}")


def validate_record(rec: dict) -> bool:
    """Reject records with missing or empty required fields."""
    for k in ["id", "input", "target"]:
        if not rec.get(k) or not str(rec[k]).strip():
            return False
    return True


def retry_load_dataset(*args, retries: int = 5, delay: int = 2, **kwargs):
    """
    Retry wrapper for HuggingFace dataset loading.
    Uses exponential backoff: waits delay*attempt seconds between retries.
    Essential for flaky university cluster / shared network environments.
    """
    from datasets import load_dataset
    for attempt in range(retries):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            wait = delay * (attempt + 1)
            print(f"  ⚠️  Retry {attempt+1}/{retries} — {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to load dataset after {retries} attempts: {args}")


def extract_gsm8k_answer(full_answer: str) -> str:
    """
    GSM8K stores chain-of-thought + final answer: '...steps... #### 42'
    Extract only the final answer (after ####) for use as ground truth.
    """
    if "####" in full_answer:
        return full_answer.split("####")[-1].strip()
    return full_answer.strip()


def save_manifest(dataset: str, records: list, hf_id: str, notes: str = "") -> None:
    """
    Save a manifest recording dataset provenance and split sizes.
    Required for reproducibility — reviewers verify from manifest alone.
    """
    # Read split sizes from the files just saved
    splits = {}
    dataset_dir = DATA_RAW / dataset
    for sf in sorted(dataset_dir.glob("*.jsonl")):
        if sf.name != "manifest.jsonl":
            with open(sf, encoding="utf-8") as f:
                splits[sf.stem] = sum(1 for line in f if line.strip())

    manifest = {
        "dataset":    dataset,
        "total":      len(records),
        "hf_id":      hf_id,
        "downloaded": datetime.now().isoformat(),
        "notes":      notes,
        "splits":     splits,      # e.g. {"dev": 100, "test": 717, "full": 817}
    }
    save_jsonl([manifest], dataset_dir / "manifest.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: TruthfulQA
# ─────────────────────────────────────────────────────────────────────────────

def download_truthfulqa() -> list:
    """
    TruthfulQA — 817 questions designed to elicit falsehoods.
    Primary in-distribution benchmark.

    Keeps: correct_answers, incorrect_answers, category
    (needed by annotation tool and evaluation runner)
    """
    print("\n📥 TruthfulQA")
    ds = retry_load_dataset("truthful_qa", "generation", split="validation")

    records, rejected = [], 0
    for i, ex in enumerate(ds):
        rec = {
            # ── Unified fields (used by ALL downstream code) ──────────────
            "id":      f"tqa_{i:04d}",
            "dataset": "truthfulqa",
            "input":   ex["question"],
            "target":  ex["best_answer"],
            # ── Dataset-specific (needed for annotation + eval) ───────────
            "question":           ex["question"],
            "best_answer":        ex["best_answer"],
            "correct_answers":    list(ex.get("correct_answers",   []) or []),
            "incorrect_answers":  list(ex.get("incorrect_answers", []) or []),
            "category":           ex.get("category", ""),
            "source":             ex.get("source",   ""),
        }
        if validate_record(rec): records.append(rec)
        else: rejected += 1

    print(f"  Kept: {len(records)} | Rejected: {rejected}")
    save_jsonl(records[:100],  DATA_RAW / "truthfulqa/dev.jsonl")
    save_jsonl(records[100:],  DATA_RAW / "truthfulqa/test.jsonl")
    save_jsonl(records,        DATA_RAW / "truthfulqa/full.jsonl")
    save_manifest("truthfulqa", records, "truthful_qa/generation",
                  "dev=first 100 (Phase 1 annotation target), test=remaining 717")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: HaluEval
# ─────────────────────────────────────────────────────────────────────────────

def download_halueval() -> list:
    """
    HaluEval — controlled hallucination benchmark.
    Each example has a correct answer AND a hallucinated version.

    Key fixes:
        - Deduplicate by question (dataset has duplicates)
        - Break on validated record count, not raw index
        - Keep knowledge field (provides context for annotators)
    """
    print("\n📥 HaluEval")
    try:
        ds = retry_load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception as e:
        print(f"  ❌ Failed after retries: {e}")
        print("  Manual download: https://github.com/RUCAIBox/HaluEval")
        return []

    records, rejected, seen = [], 0, set()
    for ex in ds:
        q      = ex.get("question") or ex.get("query", "")
        a_true = ex.get("right_answer") or ex.get("answer", "")
        a_hall = ex.get("hallucinated_answer") or ex.get("hallucination", "")

        if not q or not a_true or not a_hall:
            rejected += 1
            continue
        if q in seen:       # skip duplicates
            continue
        seen.add(q)

        rec = {
            "id":      f"halu_{len(records):05d}",
            "dataset": "halueval",
            "input":   q,
            "target":  a_true,
            "question":            q,
            "right_answer":        a_true,
            "hallucinated_answer": a_hall,
            "knowledge":           ex.get("knowledge", ""),
        }
        if validate_record(rec): records.append(rec)
        else: rejected += 1

        if len(records) >= 5000:    # break on validated count
            break

    print(f"  Kept: {len(records)} | Rejected: {rejected}")
    save_jsonl(records[:500],  DATA_RAW / "halueval/dev.jsonl")
    save_jsonl(records[500:],  DATA_RAW / "halueval/test.jsonl")
    save_jsonl(records,        DATA_RAW / "halueval/full.jsonl")
    save_manifest("halueval", records, "pminervini/HaluEval/qa_samples",
                  "Deduplicated by question. dev=500, test=4500.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: CNN/DailyMail
# ─────────────────────────────────────────────────────────────────────────────

def download_cnn() -> list:
    """
    CNN/DailyMail — OOD: long-form summarization.
    Tests generalization from QA to summarization domain.
    """
    print("\n📥 CNN/DailyMail")
    ds = retry_load_dataset("cnn_dailymail", "3.0.0", split="test")

    records, rejected = [], 0
    for i, ex in enumerate(ds):
        if i >= 1000:
            break
        rec = {
            "id":      f"cnn_{i:04d}",
            "dataset": "cnn_dailymail",
            "input":   ex["article"][:2000],   # truncate very long articles
            "target":  ex["highlights"],
            "article": ex["article"][:2000],
            "summary": ex["highlights"],
            "ood":     True,
        }
        if validate_record(rec): records.append(rec)
        else: rejected += 1

    print(f"  Kept: {len(records)} | Rejected: {rejected}")
    save_jsonl(records[:200],  DATA_RAW / "cnn_dailymail/dev.jsonl")
    save_jsonl(records[200:],  DATA_RAW / "cnn_dailymail/test.jsonl")
    save_jsonl(records,        DATA_RAW / "cnn_dailymail/full.jsonl")
    save_manifest("cnn_dailymail", records, "cnn_dailymail/3.0.0",
                  "OOD: summarization. dev=200, test=800. Articles truncated 2000 chars.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4: GSM8K
# ─────────────────────────────────────────────────────────────────────────────

def download_gsm8k() -> list:
    """
    GSM8K — OOD: mathematical reasoning.
    Tests generalization from factual QA to math domain.

    Key fix: extract final answer from chain-of-thought (after ####).
    Graceful failure: returns [] instead of crashing if download fails.
    """
    print("\n📥 GSM8K")
    try:
        ds = retry_load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        print(f"  ❌ Failed after retries: {e}")
        print("  👉 Re-run later with: --dataset gsm8k")
        return []

    records, rejected = [], 0
    for i, ex in enumerate(ds):
        if i >= 1000:
            break
        rec = {
            "id":       f"gsm_{i:04d}",
            "dataset":  "gsm8k",
            "input":    ex["question"],
            "target":   extract_gsm8k_answer(ex["answer"]),
            "question": ex["question"],
            "answer":   extract_gsm8k_answer(ex["answer"]),
            "solution": ex["answer"],   # full chain-of-thought kept for reference
            "ood":      True,
        }
        if validate_record(rec): records.append(rec)
        else: rejected += 1

    print(f"  Kept: {len(records)} | Rejected: {rejected}")
    save_jsonl(records, DATA_RAW / "gsm8k/test.jsonl")
    save_manifest("gsm8k", records, "gsm8k/main",
                  "OOD: math reasoning. target=final answer only (extracted from ####).")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download LID research datasets (production)"
    )
    parser.add_argument(
        "--dataset",
        choices=["truthfulqa", "halueval", "cnn", "gsm8k", "all"],
        default="all",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  LID Research — Dataset Download (Production)")
    print("=" * 60)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.dataset in ("truthfulqa", "all"): results["truthfulqa"] = download_truthfulqa()
    if args.dataset in ("halueval",   "all"): results["halueval"]   = download_halueval()
    if args.dataset in ("cnn",        "all"): results["cnn"]        = download_cnn()
    if args.dataset in ("gsm8k",      "all"): results["gsm8k"]      = download_gsm8k()

    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for name, recs in results.items():
        n = len(recs) if recs else 0
        total += n
        print(f"  {'✅' if n > 0 else '❌'} {name:20s} {n:>6} examples")
    print(f"  {'TOTAL':20s} {total:>6} examples")
    print(f"\n  📁 Saved to: {DATA_RAW}")
    print("  NEXT: notebooks/LID_Week3_Annotation.ipynb")


if __name__ == "__main__":
    main()