"""Rule-based crop suitability / risk assessment (NO labelled suitability data exists in the dataset).

Method ("historical envelope heuristic"): for every condition the user supplied, compare it with the range in which
the crop has actually been recorded in the dataset. This is a transparent heuristic, NOT a trained classifier and
NOT a probability of crop failure.

Per-factor score: 1.0 inside the crop's historical 10th-90th percentile, 0.5 inside min-max but outside P10-P90,
0.0 outside anything recorded. Season / state use presence counts. Overall score = mean of assessed factors.
Category: >=0.75 low risk, 0.5-0.75 moderate, <0.5 high; fewer than 2 assessed factors -> 'insufficient_information'.
"""
from __future__ import annotations

import numpy as np

from ..rag.store import CaseStore

NUMERIC = {"n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)", "ph": "Soil pH",
           "rainfall": "Annual rainfall (mm)", "temperature": "Mean temperature (C)", "humidity": "Mean humidity (%)"}
METHOD = "rule_based_historical_envelope"


def assess(store: CaseStore, crop: str, cond: dict) -> dict:
    df = store.real_df[store.real_df["crop"].str.lower() == crop.lower()]
    base = {"crop": crop, "method": METHOD, "is_trained_classifier": False,
            "description": "Heuristic comparison of supplied conditions with the range in which this crop was historically recorded. "
                           "Not a probability of failure; no verified suitability labels exist in the data."}
    if df.empty:
        return {**base, "category": "insufficient_information", "score": None, "factors": [], "missing_conditions": list(NUMERIC),
                "limitations": [f"No historical records for crop '{crop}'."]}
    factors = []
    for f, label in NUMERIC.items():
        v = cond.get(f)
        if v is None:
            continue
        s = df[f].dropna()
        if len(s) < 10:
            continue
        p10, p50, p90 = np.percentile(s, [10, 50, 90])
        lo, hi = float(s.min()), float(s.max())
        v = float(v)
        if p10 <= v <= p90:
            sc, status = 1.0, "within typical historical range (P10-P90)"
        elif lo <= v <= hi:
            sc, status = 0.5, "inside recorded range but at the edge (outside P10-P90)"
        else:
            sc, status = 0.0, "outside any historically recorded value for this crop"
        factors.append({"factor": label, "field": f, "user_value": v, "historical_p10": round(float(p10), 2),
                        "historical_median": round(float(p50), 2), "historical_p90": round(float(p90), 2),
                        "historical_min": round(lo, 2), "historical_max": round(hi, 2), "score": sc, "status": status})
    if cond.get("season"):
        share = float((df["season"].str.lower() == str(cond["season"]).lower()).mean())
        sc = 1.0 if share >= 0.05 else (0.5 if share > 0 else 0.0)
        factors.append({"factor": "Season", "field": "season", "user_value": cond["season"], "score": sc,
                        "status": f"{share:.0%} of this crop's records are in this season"
                                  + ("" if share > 0 else " (never recorded in this season)")})
    if cond.get("state"):
        sub = df[df["state"].str.lower() == str(cond["state"]).lower()]
        n = len(sub)
        sc = 1.0 if n >= 10 else (0.5 if n > 0 else 0.0)
        factors.append({"factor": "Region (state)", "field": "state", "user_value": cond["state"], "score": sc,
                        "status": f"{n} historical records of this crop in this state" + ("" if n else " (none recorded)")})
        if n >= 8:
            y = sub["yield"]
            cv = float(y.std() / y.mean()) if y.mean() > 0 else float("nan")
            if not np.isnan(cv):
                sc = 1.0 if cv < 0.5 else (0.5 if cv < 1.0 else 0.0)
                factors.append({"factor": "Yield variability in this state", "field": "yield_cv", "user_value": None,
                                "score": sc, "status": f"coefficient of variation of historical yield = {cv:.2f}"})
    missing = [label for f, label in NUMERIC.items() if cond.get(f) is None]
    if len(factors) < 2:
        cat, score = "insufficient_information", None
    else:
        score = float(np.mean([f["score"] for f in factors]))
        cat = "low" if score >= 0.75 else ("moderate" if score >= 0.5 else "high")
    risky = [f["factor"] + ": " + f["status"] for f in factors if f["score"] < 1.0]
    return {**base, "category": cat, "score": None if score is None else round(score, 3), "factors": factors,
            "supporting_factors": [f["factor"] for f in factors if f["score"] == 1.0], "risk_factors": risky,
            "missing_conditions": missing,
            "limitations": ["State-level soil and state-year weather are coarse proxies for field conditions.",
                            "Historical presence does not imply agronomic suitability for a specific farm.",
                            "Irrigation, soil type and crop variety are not in the dataset and were not assessed."]}
