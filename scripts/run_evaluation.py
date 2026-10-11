"""python scripts/run_evaluation.py [--no-llm] [--deepeval-limit 10] [--label NAME] [--rebuild-cases]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import runner  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip every Claude-dependent step (DeepEval/judge/narrative)")
    ap.add_argument("--deepeval-limit", type=int, default=10)
    ap.add_argument("--label", default=None)
    ap.add_argument("--rebuild-cases", action="store_true")
    a = ap.parse_args()
    out = runner.run(label=a.label, deepeval_limit=a.deepeval_limit, use_llm=not a.no_llm, rebuild_cases=a.rebuild_cases,
                     progress=lambda i, n, m: print(f"[{i}/{n}] {m}", flush=True))
    print(json.dumps(out["summary"], indent=2, default=str))
    print("Saved:", out["run_id"])
