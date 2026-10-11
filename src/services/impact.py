"""Soil / weather / farming-input impact analysis (descriptive statistics + model interpretation).

Everything here is associational. Correlation and permutation importance do NOT establish causation:
soil values are state constants, weather is state-year, inputs are totals per record - state, year and crop
are strong confounders.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import settings
from ..rag.store import CaseStore

FEATURES = {"n": "N", "p": "P", "k": "K", "ph": "pH", "temperature": "Temperature", "rainfall": "Rainfall",
            "humidity": "Humidity", "fertilizer_per_ha": "Fertilizer / ha", "pesticide_per_ha": "Pesticide / ha", "area": "Area"}
CONFOUNDING = [
    "Soil N/P/K/pH are constant within a state, so soil effects cannot be separated from other state-level differences (climate, practice, reporting).",
    "Weather values are annual state aggregates; they do not describe conditions at an individual farm.",
    "Fertilizer and pesticide are record totals; per-hectare values are derived (total / area) and large farms/areas differ systematically.",
    "Spearman correlation and permutation importance show association/predictive usefulness, not causal effect.",
    "Yield units differ between crops, so cross-crop comparisons use yield relative to the crop median.",
]


def _group_table(sub: pd.DataFrame, col: str, value: str, min_n: int = 10):
    out = []
    for g, d in sub.groupby(col, observed=True):
        if g == "" or len(d) < min_n:
            continue
        out.append({"group": str(g), "n": int(len(d)), "median": float(d[value].median()),
                    "q25": float(d[value].quantile(.25)), "q75": float(d[value].quantile(.75))})
    return out


def analyze(store: CaseStore, crop: str | None = None, state: str | None = None, season: str | None = None) -> dict:
    df = store.real_df   # statistics use real records only
    sel = pd.Series(True, index=df.index)
    if crop:
        sel &= df["crop"].str.lower() == crop.lower()
    if state:
        sel &= df["state"].str.lower() == state.lower()
    if season:
        sel &= df["season"].str.lower() == season.lower()
    sub = df[sel].copy()
    scope = {"crop": crop, "state": state, "season": season, "n_records": int(len(sub))}
    if len(sub) < 30:
        return {"scope": scope, "available": False, "reason": f"Only {len(sub)} matching records (<30): too few for a reliable statistical analysis.",
                "confounding_notes": CONFOUNDING}
    med = df.groupby("crop")["yield"].transform("median")
    sub["relative_yield"] = sub["yield"] / med.loc[sub.index].replace(0, np.nan)
    value = "yield" if crop else "relative_yield"
    value_label = "Yield (source units)" if crop else "Yield relative to crop median (1.0 = typical)"

    corr = []
    for f, label in FEATURES.items():
        s = sub[[f, value]].dropna()
        if len(s) >= 30 and s[f].nunique() > 1:
            corr.append({"feature": label, "field": f, "spearman_r": float(s[f].corr(s[value], method="spearman")), "n": int(len(s))})
    corr.sort(key=lambda d: -abs(d["spearman_r"]))

    groups = {}
    for col, name in [("rainfall_class", "Rainfall class"), ("ph_class", "Soil pH class"), ("temp_class", "Temperature class"),
                      ("humidity_class", "Humidity class")]:
        groups[name] = _group_table(sub, col, value)
    for f, name in [("fertilizer_per_ha", "Fertilizer / ha tercile"), ("pesticide_per_ha", "Pesticide / ha tercile")]:
        try:
            sub[f + "_t"] = pd.qcut(sub[f].rank(method="first"), 3, labels=["low", "medium", "high"])
            groups[name] = _group_table(sub, f + "_t", value)
        except ValueError:
            pass

    by_crop = []
    if not crop:
        ref = df[df["state"].str.lower() == state.lower()] if state else df
        if season:
            ref = ref[ref["season"].str.lower() == season.lower()]
        for c, d in ref.groupby("crop"):
            if len(d) >= 10:
                by_crop.append({"crop": c, "n": int(len(d)), "median_yield": float(d["yield"].median())})
        by_crop = sorted(by_crop, key=lambda x: -x["n"])[:12]

    importance = None
    mp = settings.model_dir / "yield_metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        importance = {"method": "permutation importance on held-out test years (increase in log1p-RMSE when shuffled)",
                      "scope": "global model, not specific to the selection above",
                      "values": m.get("permutation_importance", {}).get("full", [])[:12]}
    # scatter sample for charts
    sc = sub[["rainfall", value, "crop"]].dropna().sample(min(300, len(sub)), random_state=settings.random_seed)
    return {"scope": scope, "available": True, "value_definition": value_label,
            "correlations": corr, "group_comparisons": groups, "crop_comparison": by_crop,
            "model_feature_importance": importance,
            "scatter_rainfall": [{"rainfall": float(a), "value": float(b)} for a, b in zip(sc["rainfall"], sc[value])],
            "labels": {"correlation": "Spearman rank correlation (association only)",
                       "importance": "Predictive importance (not causal)"},
            "confounding_notes": CONFOUNDING}
