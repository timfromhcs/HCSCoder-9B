import argparse
import json
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
parser.add_argument("--limit", type=int, default=20)
args = parser.parse_args()

print(f"Loading SWE-bench Pro dataset from {args.dataset} (limit={args.limit})...")
try:
    ds = load_dataset(args.dataset, split="test", streaming=True)
    count = 0
    samples = []
    for s in ds:
        if count >= args.limit:
            break
        samples.append({
            "instance_id": s.get("instance_id", f"task_{count}"),
            "problem_statement": s.get("problem_statement", "")[:100],
        })
        count += 1
    print(f"Successfully loaded {len(samples)} SWE-bench Pro instances.")
    with open("artifacts/reports/swe_pro_eval_sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
except Exception as e:
    print(f"SWE-bench Pro note: {e}")
