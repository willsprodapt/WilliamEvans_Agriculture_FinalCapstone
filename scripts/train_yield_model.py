"""python scripts/train_yield_model.py -> artifacts/models/yield_model.joblib + yield_metrics.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.yield_model import train  # noqa: E402

if __name__ == "__main__":
    r = train()
    print(json.dumps({"split": r["split"], "meta": {k: r["meta"][k] for k in ["model_version", "dataset_version"]}}, indent=2))
    for fs, comp in r["comparison"].items():
        print(f"\n== feature set: {fs} ==")
        for name, d in comp.items():
            if name == "selected":
                print(f"  SELECTED {d['name']}  TEST: " + json.dumps({k: round(v, 4) for k, v in d["test"].items()}))
                print("  interval coverage:", d["interval_10_90"]["empirical_test_coverage"])
            else:
                print(f"  {name:22s} val:", json.dumps({k: round(v, 4) for k, v in d["val"].items()}))
                if "test" in d:
                    print(f"  {name:22s} test:", json.dumps({k: round(v, 4) for k, v in d["test"].items()}))
