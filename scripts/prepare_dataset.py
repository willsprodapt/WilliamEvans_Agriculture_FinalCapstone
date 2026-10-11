"""python scripts/prepare_dataset.py  -> data/processed/cases.csv + data_quality_report.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import settings  # noqa: E402
from src.services.data_prep import prepare  # noqa: E402

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augment", action="store_true", help="use real rows only (no labelled synthetic augmentation)")
    a = ap.parse_args()
    r = prepare(augment=not a.no_augment)
    keys = ["source_row_count", "valid_row_count", "real_row_count", "synthetic_row_count", "target_rows", "final_row_count", "target_met",
            "dataset_version", "n_crops", "n_states"]
    print(json.dumps({k: r[k] for k in keys}, indent=2))
    if not r["target_met_with_real_data_only"]:
        print(f"
NOTE: only {r['real_row_count']} real rows exist; {r['synthetic_row_count']} rows are LABELLED SYNTHETIC (is_synthetic=True).")
    df = pd.read_csv(settings.cases_path)
    settings.sample_dir.mkdir(parents=True, exist_ok=True)
    out = settings.sample_dir / "sample_cases_REAL_ROWS_subset.csv"
    df[~df["is_synthetic"]].sample(300, random_state=settings.random_seed).drop(columns=["text"]).to_csv(out, index=False)
    print("Report:", settings.quality_report_path, "\nSample (300 real rows):", out)
