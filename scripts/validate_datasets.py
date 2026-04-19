"""
scripts/validate_datasets.py
Validate downloaded datasets: counts, required columns, data quality.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

# Expected structure: dataset -> list of (split_name, required_fields, optional_fields)
EXPECTED = {
    "truthfulqa": {
        "splits": ["dev.jsonl", "test.jsonl", "full.jsonl", "manifest.jsonl"],
        "required": ["id", "dataset", "input", "target", "question", "best_answer"],
        "optional": ["correct_answers", "incorrect_answers", "category", "source"]
    },
    "halueval": {
        "splits": ["dev.jsonl", "test.jsonl", "full.jsonl", "manifest.jsonl"],
        "required": ["id", "dataset", "input", "target", "question", "right_answer", "hallucinated_answer"],
        "optional": ["knowledge"]
    },
    "cnn_dailymail": {
        "splits": ["dev.jsonl", "test.jsonl", "full.jsonl", "manifest.jsonl"],
        "required": ["id", "dataset", "input", "target", "article", "summary", "ood"],
        "optional": []
    },
    "gsm8k": {
        "splits": ["test.jsonl", "manifest.jsonl"],
        "required": ["id", "dataset", "input", "target", "question", "answer", "solution", "ood"],
        "optional": []
    }
}

def validate_file(path, required_fields):
    """Validate a JSONL file: count lines, check fields, non-empty strings."""
    if not path.exists():
        return {"error": "missing", "count": 0}
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                return {"error": f"JSON error at line {line_num}: {e}", "count": 0}
    
    # Check required fields
    missing_fields = set()
    empty_fields = set()
    for rec in records[:10]:  # sample first 10
        for field in required_fields:
            if field not in rec:
                missing_fields.add(field)
            elif not rec[field] or (isinstance(rec[field], str) and not rec[field].strip()):
                empty_fields.add(field)
    
    return {
        "count": len(records),
        "missing_fields": missing_fields,
        "empty_fields": empty_fields,
        "sample": records[0] if records else None
    }

def main():
    print("=" * 70)
    print("  DATASET VALIDATION")
    print("=" * 70)
    
    if not DATA_RAW.exists():
        print(f"❌ Data directory not found: {DATA_RAW}")
        print("   Run 'python scripts/download_datasets.py' first.")
        return
    
    all_ok = True
    total_records = 0
    
    for dataset, spec in EXPECTED.items():
        dataset_path = DATA_RAW / dataset
        if not dataset_path.exists():
            print(f"\n❌ {dataset}: folder missing")
            all_ok = False
            continue
        
        print(f"\n📁 {dataset.upper()}")
        for split in spec["splits"]:
            file_path = dataset_path / split
            result = validate_file(file_path, spec["required"])
            
            if "error" in result:
                print(f"  ❌ {split}: {result['error']}")
                all_ok = False
            else:
                status = "✅" if (not result["missing_fields"] and not result["empty_fields"]) else "⚠️"
                print(f"  {status} {split}: {result['count']:>6} records")
                if result["missing_fields"]:
                    print(f"       Missing fields: {result['missing_fields']}")
                    all_ok = False
                if result["empty_fields"]:
                    print(f"       Empty fields (first 10): {result['empty_fields']}")
                    # not necessarily fatal, but warn
                total_records += result["count"]
        
        # Optional: show a sample record
        sample_file = dataset_path / spec["splits"][0]  # first split
        if sample_file.exists():
            try:
                with open(sample_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        sample = json.loads(first_line)
                        print(f"  📝 Sample (first record):")
                        print(f"       id: {sample.get('id')}")
                        print(f"       input: {sample.get('input', '')[:60]}...")
                        print(f"       target: {sample.get('target', '')[:60]}...")
            except Exception as e:
                print(f"  ⚠️  Could not read sample: {e}")
    
    print("\n" + "=" * 70)
    print(f"  TOTAL VALID RECORDS: {total_records}")
    if all_ok and total_records == 7817:  # expected total from your run
        print("  ✅ ALL DATASETS VALIDATED SUCCESSFULLY")
    else:
        print("  ⚠️  Some issues found. See details above.")
    print("=" * 70)

if __name__ == "__main__":
    main()