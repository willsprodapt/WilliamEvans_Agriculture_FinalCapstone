"""Reproducible experiment runner (everything seeded; every number in the report comes from these runs).

  python scripts/audit_data.py                       # dataset facts
  python scripts/run_experiments.py                  # all experiments
  python scripts/run_experiments.py --only ml        # ml | retrieval | recommend | dimensions

Outputs: artifacts/evaluation/experiments/*.json, then `python scripts/build_experiment_report.py` writes docs/experiments/Experiments_Report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

OUT = settings.eval_dir / "experiments"


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"saved {OUT / (name + '.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ml", "retrieval", "recommend", "dimensions"], default=None)
    a = ap.parse_args()
    run = lambda k: a.only in (None, k)  # noqa: E731
    t0 = time.time()
    if run("ml"):
        from src.evaluation import exp_ml
        save("regression", exp_ml.run_regression())
        save("classification", exp_ml.run_classification())
    if run("retrieval"):
        from src.evaluation import exp_retrieval
        save("retrieval", exp_retrieval.run())
    if run("recommend"):
        from src.evaluation import exp_recommend
        tune = exp_recommend.tune_popularity_weight()
        save("recommend_tuning", tune)
        from src.services import recommend
        recommend.POPULARITY_WEIGHT = tune["selected"]
        save("recommend_heldout", exp_recommend.run_heldout(pop_w=tune["selected"]))
    if run("dimensions"):
        from src.evaluation import exp_recommend
        save("recommend_dimensions", exp_recommend.run_dimensions())
    print(f"done in {time.time() - t0:.0f}s")
