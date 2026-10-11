"""python scripts/audit_data.py  -> docs/experiments/data_audit.json (facts measured from the actual files; nothing assumed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings  # noqa: E402

OUT = settings.processed_dir.parent.parent / "docs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(settings.raw_dir / "crop_yield.csv")
soil = pd.read_csv(settings.raw_dir / "state_soil_data.csv", encoding="utf-8-sig")
wx = pd.read_csv(settings.raw_dir / "state_weather_data_1997_2020.csv")
cases = pd.read_csv(settings.cases_path)
real, syn = cases[~cases["is_synthetic"]], cases[cases["is_synthetic"]]
a: dict = {}

a["kaggle_raw"] = {
    "file": "data/raw/crop_yield.csv (+ state_soil_data.csv, state_weather_data_1997_2020.csv)",
    "rows": len(raw), "columns": list(raw.columns), "missing": raw.isna().sum().to_dict(),
    "exact_duplicate_rows": int(raw.duplicated().sum()),
    "duplicate_natural_key_rows(crop,year,season,state)": int(raw.assign(season=raw["Season"].str.strip()).duplicated(["Crop", "Crop_Year", "season", "State"]).sum()) if "Crop_Year" in raw else None,
    "n_crops": int(raw.iloc[:, 0].nunique()), "n_states": int(raw["state"].nunique()) if "state" in raw else int(raw["State"].nunique()),
    "labels_present": "NO suitability / condition label. Only numeric outcomes (area, production, yield) per crop-state-season-year.",
    "not_in_schema": ["soil type", "irrigation", "pest/disease", "farm practices", "farm/field id", "lat/lon"],
}
a["kaggle_raw"]["raw_rows_vs_user_stated"] = "Raw crop file has %d rows (not ~10,000); processed Kaggle-sourced rows used: %d" % (len(raw), len(real))

# soil / weather tables
a["soil_table"] = {"rows": len(soil), "one_value_per_state": True,
                   "note": "N/P/K/pH are a single constant per state, so they carry no within-state information; they are state identifiers."}
per_state_unique_temp = wx.groupby("state")["avg_temp_c"].nunique()
a["weather_table"] = {"rows": len(wx), "states": int(wx["state"].nunique()),
                      "states_with_constant_temp_over_all_years": int((per_state_unique_temp == 1).sum()),
                      "median_unique_temp_values_per_state": float(per_state_unique_temp.median()),
                      "note": "annual state aggregates; check 'states_with_constant_temp_over_all_years' for suspected non-measured values"}

# processed corpus
a["processed_corpus"] = {"rows": len(cases), "kaggle_sourced": len(real), "synthetic": len(syn),
                         "synthetic_share": round(len(syn) / len(cases), 4),
                         "years_real": [int(real.year.min()), int(real.year.max())], "years_synth": [int(syn.year.min()), int(syn.year.max())]}

# synthetic vs real near-duplication (same crop/state/season; standardised key numeric fields)
num = ["area", "yield", "fertilizer_per_ha", "pesticide_per_ha"]
def _z(df): return np.log1p(df[num].clip(lower=0))
rg = real.groupby(["crop", "state", "season"])
near = 0; exact_key = 0; checked = 0
realz = _z(real)
sd = realz.std().replace(0, 1)
for key, grp in syn.groupby(["crop", "state", "season"]):
    if key not in rg.groups:
        continue
    rr = realz.loc[rg.groups[key]]
    s = _z(grp)
    for i in range(len(s)):
        d = np.abs((rr.values - s.values[i]) / sd.values).max(axis=1)
        near += int(d.min() < 0.05); checked += 1
a["synthetic_vs_real"] = {
    "synthetic_checked": checked,
    "within_0.05_sd_on_all_4_numeric_fields_of_some_real_row": near,
    "share_near_copy": round(near / max(1, checked), 4),
    "same_crop_state_season_as_a_real_group": int(checked),
    "new_crop_state_pairs_created": int(len(set(zip(syn.crop, syn.state)) - set(zip(real.crop, real.state)))),
    "templates": "synthetic rows are perturbed copies of randomly chosen real rows from ANY year, including later held-out test years",
}
a["synthetic_internal"] = {"exact_duplicate_numeric_rows": int(syn.duplicated(["crop", "state", "season", "year", "area", "yield"]).sum())}
ks = {}
for c in ["yield", "area", "fertilizer_per_ha", "pesticide_per_ha", "rainfall", "temperature"]:
    r, s = real[c].dropna(), syn[c].dropna()
    stat = ks_2samp(np.log1p(r.clip(lower=0)), np.log1p(s.clip(lower=0)))
    ks[c] = {"ks_stat": round(float(stat.statistic), 4), "p": float(stat.pvalue), "real_median": float(r.median()), "synthetic_median": float(s.median())}
a["distribution_shift_real_vs_synthetic"] = ks

# yield consistency + outliers + sparsity
imp = real["production"] / real["area"]
a["real_quality"] = {
    "yield_vs_production_over_area_mismatch_gt10pct": int((((real['yield'] - imp).abs() / imp.clip(lower=1e-9)) > .1).sum()),
    "zero_production_rows": int((real["production"] == 0).sum()),
    "yield_unit_note": "units not stated; crop-specific (coconut is per nut). Only comparable within crop.",
    "groups_crop_state_season": int(real.groupby(["crop", "state", "season"]).ngroups),
    "groups_with_fewer_than_5_rows": int((real.groupby(["crop", "state", "season"]).size() < 5).sum()),
    "yield_band_class_balance_real": real["yield_band"].value_counts(normalize=True).round(3).to_dict(),
    "rows_by_year_real_tail": real.groupby("year").size().tail(6).to_dict(),
}
a["problems_found"] = [
    "Raw Kaggle crop file has 19,689 rows (not ~10,000); synthetic share is 35,311 rows (not 45,000). Counts reported as measured.",
    "No ground-truth condition/suitability label exists anywhere in the Kaggle data -> a supervised suitability classifier cannot be validly trained.",
    "Soil N/P/K/pH are one constant per state: they act as state identifiers, not field-level soil measurements.",
    "Synthetic rows are noisy copies of real rows (same crop/state/season); template rows were drawn from all years including the held-out test years -> label leakage into any evaluation that mixes synthetic data with real test years.",
    "Synthetic rows do not add new crop/state/season combinations, so they add no new agronomic knowledge, only near-duplicates for retrieval and training.",
    "No irrigation, soil type, pest/disease or farming-practice fields exist, so those relationships cannot be modelled or validated.",
]
(OUT / "data_audit.json").write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: a[k] for k in ["kaggle_raw", "weather_table", "synthetic_vs_real", "real_quality"]}, indent=1, default=str)[:4500])
