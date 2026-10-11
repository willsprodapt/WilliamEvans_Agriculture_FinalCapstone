"""Reproducible dataset preparation: validate, clean, merge state soil/weather, derive, id, report.

Source files (Kaggle: anshumish/crop-yield-data-with-soil-and-weather-dataset):
  crop_yield.csv                      crop, year, season, state, area, production, fertilizer, pesticide, yield
  state_soil_data.csv                 state, N, P, K, pH        (ONE value per state - state-level constant)
  state_weather_data_1997_2020.csv    state, year, avg_temp_c, total_rainfall_mm, avg_humidity_percent
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

CASE_COLUMNS = [
    "case_id", "source_row", "crop", "year", "season", "state", "area", "production", "yield",
    "fertilizer", "pesticide", "n", "p", "k", "ph", "temperature", "rainfall", "humidity",
    "fertilizer_per_ha", "pesticide_per_ha", "yield_band", "ph_class", "rainfall_class",
    "temp_class", "humidity_class", "completeness", "flags", "is_synthetic", "data_source", "text",
]

# Documented thresholds for DERIVED descriptive classes (used in case text + display only).
PH_BINS = [(0, 5.5, "strongly acidic"), (5.5, 6.5, "acidic"), (6.5, 7.5, "neutral"), (7.5, 99, "alkaline")]
RAIN_BINS = [(0, 800, "low rainfall"), (800, 1500, "moderate rainfall"), (1500, 2500, "high rainfall"), (2500, 1e9, "very high rainfall")]
TEMP_BINS = [(-99, 20, "cool"), (20, 26, "moderate temperature"), (26, 99, "hot")]
HUM_BINS = [(0, 55, "low humidity"), (55, 72, "moderate humidity"), (72, 101, "high humidity")]


def _band(v, bins):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    for lo, hi, name in bins:
        if lo <= v < hi:
            return name
    return ""


def clean_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def make_case_id(crop: str, year: int, season: str, state: str, serial: int | None = None, synthetic: bool = False) -> str:
    key = f"{crop}|{year}|{season}|{state}" + (f"|syn{serial}" if synthetic else "")
    return ("TS-" if synthetic else "TM-") + hashlib.sha1(key.lower().encode()).hexdigest()[:10].upper()


def case_text(r: pd.Series) -> str:
    """Searchable description built ONLY from the record's own values."""
    parts = [f"{r['crop']} grown in {r['state']} during the {r['season']} season in {int(r['year'])}."]
    parts.append(f"Area {r['area']:.0f} hectares, production {r['production']:.0f}, historical yield {r['yield']:.3f}"
                 f" ({r['yield_band']} for this crop).")
    if pd.notna(r["n"]):
        parts.append(f"State soil profile: nitrogen N {r['n']:g}, phosphorus P {r['p']:g}, potassium K {r['k']:g}, "
                     f"soil pH {r['ph']:g} ({r['ph_class']} soil).")
    if pd.notna(r["rainfall"]):
        parts.append(f"Weather that year: temperature {r['temperature']:.1f} C ({r['temp_class']}), "
                     f"rainfall {r['rainfall']:.0f} mm ({r['rainfall_class']}), humidity {r['humidity']:.1f} percent ({r['humidity_class']}).")
    parts.append(f"Fertilizer {r['fertilizer']:.0f} total ({r['fertilizer_per_ha']:.1f} per hectare), "
                 f"pesticide {r['pesticide']:.0f} total ({r['pesticide_per_ha']:.2f} per hectare).")
    return " ".join(parts)


def _augment(real: pd.DataFrame, n: int, seed: int, weather: pd.DataFrame | None):
    """Generate n SYNTHETIC rows as perturbed variants of real rows (same crop/state/season; year within the group's observed range).
    Soil (state constants) and weather (state-year) are the REAL values for the chosen state/year; only the crop outcome
    columns (area, inputs, yield, production) are generated: multiplicative log-normal noise (sigma 0.20 area/inputs, 0.15 yield)."""
    rng = np.random.default_rng(seed)
    tmpl = real.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    rng_years = real.groupby(["crop", "state", "season"])["year"].agg(["min", "max"])
    key = list(zip(tmpl["crop"], tmpl["state"], tmpl["season"]))
    lo = np.array([rng_years.loc[k, "min"] for k in key]); hi = np.array([rng_years.loc[k, "max"] for k in key])
    tmpl["year"] = rng.integers(lo, hi + 1)
    area = tmpl["area"] * rng.lognormal(0, 0.20, n)
    fpha = tmpl["fertilizer"] / tmpl["area"] * rng.lognormal(0, 0.20, n)
    ppha = tmpl["pesticide"] / tmpl["area"] * rng.lognormal(0, 0.20, n)
    yld = tmpl["yield"] * rng.lognormal(0, 0.15, n)
    tmpl["area"] = area.round(1).clip(lower=0.5)
    tmpl["yield"] = yld.round(6)
    tmpl["production"] = (tmpl["yield"] * tmpl["area"]).round(0)
    tmpl["fertilizer"] = (fpha * tmpl["area"]).round(2)
    tmpl["pesticide"] = (ppha * tmpl["area"]).round(2)
    if weather is not None:
        tmpl = tmpl.drop(columns=["temperature", "rainfall", "humidity"]).merge(
            weather[["state", "year", "temperature", "rainfall", "humidity"]], on=["state", "year"], how="left")
    tmpl["is_synthetic"] = True
    tmpl["source_row"] = -(np.arange(n) + 1)     # unique negative serial: no raw row exists
    out = pd.concat([real, tmpl[real.columns]], ignore_index=True)
    method = ("Each synthetic row perturbs a randomly chosen real row of the same crop/state/season (year drawn from the group's observed range; "
              "real state soil and state-year weather re-attached); area and per-hectare inputs get log-normal noise (sigma 0.20), yield sigma 0.15, "
              "production = yield x area. No new crop-state pairs are created.")
    return out, {"enabled": True, "n_synthetic": int(n), "method": method, "seed": seed}


def prepare(raw_dir: Path | None = None, out_dir: Path | None = None, target_rows: int | None = None,
            seed: int | None = None, crop_file: Path | None = None, augment: bool = True) -> dict:
    raw_dir = raw_dir or settings.raw_dir
    out_dir = out_dir or settings.processed_dir
    target_rows = target_rows or settings.target_rows
    seed = seed if seed is not None else settings.random_seed
    crop_file = crop_file or settings.dataset_path
    decisions: list[str] = []

    if not Path(crop_file).exists():
        raise FileNotFoundError(f"Crop dataset not found at {crop_file}. See README 'Dataset' for download instructions.")
    crops = pd.read_csv(crop_file)
    crops.columns = [c.strip().lower() for c in crops.columns]
    required = ["crop", "year", "season", "state", "area", "production", "fertilizer", "pesticide", "yield"]
    missing_cols = [c for c in required if c not in crops.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    source_rows = len(crops)
    crops["source_row"] = np.arange(source_rows)

    # --- cleaning -----------------------------------------------------------------
    for c in ["crop", "season", "state"]:
        crops[c] = crops[c].map(clean_label)
    decisions.append("Trimmed/collapsed whitespace in crop, season, state labels (season values are space-padded in the source).")
    for c in ["year", "area", "production", "fertilizer", "pesticide", "yield"]:
        crops[c] = pd.to_numeric(crops[c], errors="coerce")
    n_null = int(crops[required].isna().any(axis=1).sum())
    crops = crops.dropna(subset=required)
    invalid = (crops["area"] <= 0) | (crops["production"] < 0) | (crops["yield"] < 0) | (crops["fertilizer"] < 0) | (crops["pesticide"] < 0)
    n_invalid = int(invalid.sum())
    crops = crops[~invalid]
    decisions.append(f"Dropped {n_null} rows with missing required values and {n_invalid} rows with non-positive area or negative production/yield/inputs.")
    dup_mask = crops.drop(columns="source_row").duplicated()
    n_exact = int(dup_mask.sum())
    crops = crops[~dup_mask]
    key = ["crop", "year", "season", "state"]
    n_key = int(crops.duplicated(subset=key).sum())
    crops = crops.drop_duplicates(subset=key, keep="first")
    decisions.append(f"Removed {n_exact} exact duplicate rows and {n_key} duplicate (crop, year, season, state) keys (first kept).")
    crops["year"] = crops["year"].astype(int)

    # --- soil / weather enrichment (state-level; NOT field-level) ------------------------
    soil_p, weather_p = raw_dir / "state_soil_data.csv", raw_dir / "state_weather_data_1997_2020.csv"
    if soil_p.exists():
        soil = pd.read_csv(soil_p, encoding="utf-8-sig")
        soil.columns = [c.strip().lower() for c in soil.columns]
        soil["state"] = soil["state"].map(clean_label)
        crops = crops.merge(soil[["state", "n", "p", "k", "ph"]], on="state", how="left")
        decisions.append("Joined state-level soil N, P, K, pH on state. These are ONE constant value per state, not per-farm measurements.")
    else:
        for c in ["n", "p", "k", "ph"]:
            crops[c] = np.nan
        decisions.append("state_soil_data.csv not found: soil columns left missing (not zero).")
    if weather_p.exists():
        w = pd.read_csv(weather_p)
        w.columns = [c.strip().lower() for c in w.columns]
        w["state"] = w["state"].map(clean_label)
        w = w.rename(columns={"avg_temp_c": "temperature", "total_rainfall_mm": "rainfall", "avg_humidity_percent": "humidity"})
        crops = crops.merge(w[["state", "year", "temperature", "rainfall", "humidity"]], on=["state", "year"], how="left")
        decisions.append("Joined state-year weather (annual mean temperature C, total annual rainfall mm, mean humidity %) on (state, year).")
    else:
        for c in ["temperature", "rainfall", "humidity"]:
            crops[c] = np.nan
        decisions.append("Weather file not found: weather columns left missing.")

    # --- sampling to target ---------------------------------------------------------
    valid_rows = len(crops)
    sampling = {"strategy": "all real rows used" + (" + labelled synthetic augmentation" if augment else " (no synthetic rows)"), "seed": seed}
    if valid_rows > target_rows:
        frac = target_rows / valid_rows
        strata = ["crop", "state", "season"]
        picked = crops.groupby(strata).sample(frac=frac, random_state=seed)
        # every stratum keeps at least one row so rare crop/state/season combinations stay represented
        gk = picked[strata].apply(tuple, axis=1)
        missing = crops[~crops[strata].apply(tuple, axis=1).isin(set(gk))]
        extra = missing.groupby(strata).sample(n=1, random_state=seed) if len(missing) else missing
        crops = pd.concat([picked, extra])
        if len(crops) > target_rows:
            crops = crops.sample(target_rows, random_state=seed)
        sampling = {"strategy": "proportional stratified sample by crop/state/season", "seed": seed}
    decisions.append(f"Valid rows available: {valid_rows}. Target: {target_rows}. "
                     + ("Target NOT reachable with real data" + ("; labelled synthetic rows are added (see AUGMENTATION)." if augment else "; all valid rows used, no synthetic rows created.") if valid_rows < target_rows else "Sampled."))

    # --- labelled synthetic augmentation (only when real rows < target and augment=True) -----------------
    crops["is_synthetic"] = False
    real_rows = len(crops)
    synth_info = {"enabled": False, "n_synthetic": 0}
    if augment and real_rows < target_rows:
        crops, synth_info = _augment(crops, target_rows - real_rows, seed, w if weather_p.exists() else None)
        decisions.append(f"AUGMENTATION: {synth_info['n_synthetic']} SYNTHETIC rows were generated to reach {target_rows} (is_synthetic=True, ids start 'TS-', source_row=-1). "
                         "They are not historical observations. " + synth_info["method"])

    # --- derived features ---------------------------------------------------------------
    crops["fertilizer_per_ha"] = crops["fertilizer"] / crops["area"]
    crops["pesticide_per_ha"] = crops["pesticide"] / crops["area"]
    pct = crops.groupby("crop")["yield"].rank(pct=True)
    crops["yield_band"] = np.where(pct <= 0.25, "low yield", np.where(pct > 0.75, "high yield", "typical yield"))
    crops["ph_class"] = crops["ph"].map(lambda v: _band(v, PH_BINS))
    crops["rainfall_class"] = crops["rainfall"].map(lambda v: _band(v, RAIN_BINS))
    crops["temp_class"] = crops["temperature"].map(lambda v: _band(v, TEMP_BINS))
    crops["humidity_class"] = crops["humidity"].map(lambda v: _band(v, HUM_BINS))
    feat = ["n", "p", "k", "ph", "temperature", "rainfall", "humidity"]
    crops["completeness"] = crops[feat].notna().mean(axis=1).round(3)

    implied = crops["production"] / crops["area"]
    rel = (crops["yield"] - implied).abs() / np.maximum(implied.abs(), 1e-9)
    flags = []
    for zp, mism, comp in zip(crops["production"] == 0, rel > 0.10, crops["completeness"] < 1):
        f = []
        if zp:
            f.append("zero_production")
        if mism:
            f.append("yield_differs_from_production_over_area")
        if comp:
            f.append("incomplete_conditions")
        flags.append(";".join(f))
    crops["flags"] = flags

    crops = crops.sort_values(["crop", "state", "year", "season"]).reset_index(drop=True)
    crops["case_id"] = [make_case_id(r.crop, r.year, r.season, r.state, int(r.source_row) if r.is_synthetic else None, r.is_synthetic)
                        for r in crops.itertuples()]
    crops["flags"] = [(f + ";synthetic").strip(";") if syn else f for f, syn in zip(crops["flags"], crops["is_synthetic"])]
    if crops["case_id"].duplicated().any():
        raise RuntimeError("Case ID collision detected")
    crops["text"] = crops.apply(case_text, axis=1)
    crops["data_source"] = np.where(crops["is_synthetic"], "synthetic", "kaggle")
    crops = crops[CASE_COLUMNS]

    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.csv"
    crops.to_csv(cases_path, index=False)
    version = hashlib.sha256(cases_path.read_bytes()).hexdigest()[:12]

    # outlier inspection (reported, NOT deleted)
    extremes = {}
    for c in ["yield", "area", "fertilizer", "pesticide"]:
        q1, q3 = crops[c].quantile([.25, .75])
        extremes[c] = int((crops[c] > q3 + 3 * (q3 - q1)).sum())
    ratio = float((crops["yield"] / (crops["production"] / crops["area"]).clip(lower=1e-9))[crops["production"] > 0].median())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": version,
        "source": "https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset",
        "source_row_count": source_rows,
        "dropped_missing": n_null, "dropped_invalid": n_invalid,
        "dropped_exact_duplicates": n_exact, "dropped_duplicate_keys": n_key,
        "valid_row_count": valid_rows,
        "real_row_count": int((~crops["is_synthetic"]).sum()),
        "synthetic_row_count": int(crops["is_synthetic"].sum()),
        "synthetic_method": synth_info.get("method"),
        "target_rows": target_rows,
        "final_row_count": int(len(crops)),
        "target_met": bool(len(crops) >= target_rows),
        "target_met_with_real_data_only": bool(valid_rows >= target_rows),
        "sampling": sampling,
        "random_seed": seed,
        "decisions": decisions,
        "n_crops": int(crops["crop"].nunique()), "n_states": int(crops["state"].nunique()),
        "seasons": sorted(crops["season"].unique().tolist()),
        "year_range": [int(crops["year"].min()), int(crops["year"].max())],
        "missing_by_column": {c: int(crops[c].isna().sum()) for c in crops.columns if c != "text"},
        "flag_counts": {k: int(crops["flags"].str.contains(k).sum()) for k in
                        ["zero_production", "yield_differs_from_production_over_area", "incomplete_conditions"]},
        "extreme_value_counts_3xIQR": extremes,
        "yield_definition": f"yield ~ production / area (median ratio of reported yield to production/area = {ratio:.3f}). "
                            "Units are NOT stated by the source; they are crop-specific (typically tonnes/hectare, but e.g. "
                            "coconut is reported per-nut). Yields are therefore only comparable within a crop.",
        "soil_note": "N, P, K, pH are state-level constants (one value per state); units of N/P/K are not stated by the source.",
        "weather_note": "temperature (C), rainfall (mm, annual total) and humidity (%) are state-year aggregates, not farm-level.",
        "derived_columns": {
            "fertilizer_per_ha": "fertilizer / area", "pesticide_per_ha": "pesticide / area",
            "yield_band": "per-crop percentile of yield: <=25th low, >75th high, else typical",
            "ph_class": "bins <5.5 strongly acidic, 5.5-6.5 acidic, 6.5-7.5 neutral, >7.5 alkaline",
            "rainfall_class": "bins <800, 800-1500, 1500-2500, >2500 mm",
            "temp_class": "bins <20, 20-26, >26 C", "humidity_class": "bins <55, 55-72, >72 %",
            "completeness": "share of the 7 soil/weather fields present",
        },
        "case_id": "TM- + first 10 hex chars of SHA-1(crop|year|season|state), unique per natural key; source_row = 0-based row of crop_yield.csv",
    }
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ------------------------------------------------------------------------------------------------ synthetic generator v2
GENERATOR_V2_RULES = [
    "Templates are drawn ONLY from the rows passed in (the training split): no validation/test row or label can influence a synthetic row.",
    "A synthetic row keeps its template's crop, state and season (no new crop-state pairs); its year is drawn inside the observed "
    "[min, max] year range of that crop-state-season group, so it never extrapolates beyond the training window.",
    "Yield is the template yield scaled by the per-crop log-yield time trend estimated from the training rows (slope clipped to +-3%/year) "
    "times log-normal noise (sigma 0.12); it is clipped to [0.8 x crop minimum, 1.2 x crop maximum] of the training rows.",
    "Area gets log-normal noise (sigma 0.20); fertilizer and pesticide per hectare share one log-normal factor (sigma 0.15, inputs scale "
    "together) plus individual noise (sigma 0.10); production = yield x area, so the record is internally consistent.",
    "Soil (state constants) and weather (state-year aggregate) are the REAL values for the chosen state/year, so weather is never invented.",
    "Rows whose (log area, log yield, fertilizer/ha, pesticide/ha) lie within 0.05 standard deviations of ANY real row of the same group, "
    "and rows duplicating another synthetic row, are rejected and regenerated.",
    "Target values are never an input to another feature: yield is generated from the template yield and the year trend only.",
]
_NUM4 = ["area", "yield", "fertilizer_per_ha", "pesticide_per_ha"]


def generate_synthetic(real_train: pd.DataFrame, n: int, seed: int, weather: pd.DataFrame | None = None,
                       near_copy_sd: float = 0.05) -> pd.DataFrame:
    """Leak-safe synthetic rows built from `real_train` only. See GENERATOR_V2_RULES. Returns n rows with data_source='synthetic'."""
    rng = np.random.default_rng(seed)
    base = real_train.copy().reset_index(drop=True)
    base["fertilizer_per_ha"] = base["fertilizer"] / base["area"]
    base["pesticide_per_ha"] = base["pesticide"] / base["area"]
    slope = {}
    for crop, g in base.groupby("crop"):
        ok = g["year"].nunique() >= 4 and len(g) >= 8
        slope[crop] = float(np.clip(np.polyfit(g["year"], np.log1p(g["yield"]), 1)[0], -0.03, 0.03)) if ok else 0.0
    gk = ["crop", "state", "season"]
    grp = base.groupby(gk)
    yr = grp["year"].agg(["min", "max"])
    ylim = base.groupby("crop")["yield"].agg(["min", "max"])
    amax = grp["area"].max()
    logz = np.log1p(base[_NUM4].clip(lower=0))
    sd = logz.std().replace(0, 1).to_numpy()
    real_z = {k: logz.loc[idx].to_numpy() for k, idx in grp.groups.items()}
    w = None if weather is None else weather[["state", "year", "temperature", "rainfall", "humidity"]]

    parts, have, attempt = [], 0, 0
    seen: set = set()
    while have < n and attempt < 6:
        attempt += 1
        need = int((n - have) * (1.25 if attempt == 1 else 1.6)) + 5
        t = base.sample(need, replace=True, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
        keys = list(zip(t["crop"], t["state"], t["season"]))
        lo = np.array([yr.loc[k, "min"] for k in keys])
        hi = np.array([yr.loc[k, "max"] for k in keys])
        new_year = rng.integers(lo, hi + 1)
        dy = new_year - t["year"].to_numpy()
        tr = np.array([slope[c] for c in t["crop"]])
        yld = t["yield"].to_numpy() * np.exp(tr * dy) * rng.lognormal(0, 0.12, need)
        yld = np.clip(yld, 0.8 * ylim.loc[t["crop"], "min"].to_numpy(), 1.2 * ylim.loc[t["crop"], "max"].to_numpy())
        area = np.clip(t["area"].to_numpy() * rng.lognormal(0, 0.20, need), 0.5, 1.5 * np.array([amax.loc[k] for k in keys]))
        shared = rng.lognormal(0, 0.15, need)
        fpha = t["fertilizer_per_ha"].to_numpy() * shared * rng.lognormal(0, 0.10, need)
        ppha = t["pesticide_per_ha"].to_numpy() * shared * rng.lognormal(0, 0.10, need)
        s = t[["crop", "state", "season", "n", "p", "k", "ph"]].copy()
        s["year"] = new_year
        s["area"] = area.round(1)
        s["yield"] = yld.round(6)
        s["production"] = (s["yield"] * s["area"]).round(0)
        s["fertilizer"] = (fpha * s["area"]).round(2)
        s["pesticide"] = (ppha * s["area"]).round(2)
        if w is not None:
            s = s.merge(w, on=["state", "year"], how="left")
        else:
            for c in ("temperature", "rainfall", "humidity"):
                s[c] = np.nan
        s["fertilizer_per_ha"] = s["fertilizer"] / s["area"]
        s["pesticide_per_ha"] = s["pesticide"] / s["area"]
        z = np.log1p(s[_NUM4].clip(lower=0)).to_numpy()
        keep = np.ones(len(s), bool)
        for i, k in enumerate(zip(s["crop"], s["state"], s["season"])):
            d = np.abs((real_z[k] - z[i]) / sd).max(axis=1)
            if d.min() < near_copy_sd:
                keep[i] = False
                continue
            sig = (k, int(s["year"].iat[i]), round(float(s["area"].iat[i]), 1), round(float(s["yield"].iat[i]), 4))
            if sig in seen:
                keep[i] = False
            else:
                seen.add(sig)
        s = s[keep]
        parts.append(s)
        have += len(s)
    out = pd.concat(parts, ignore_index=True).head(n)
    out["is_synthetic"] = True
    out["data_source"] = "synthetic"
    return out
