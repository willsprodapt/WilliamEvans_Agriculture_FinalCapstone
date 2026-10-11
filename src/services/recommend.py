"""Deterministic, evidence-based crop recommendation from prioritised historical cases.

crop_score = mean(priority of the crop's top-3 supporting cases) * (0.7 + 0.3 * min(1, n_supporting / 3))
This is a heuristic ranking score in [0, 1] - NOT a probability, and NOT based on historical yield alone
(yield units are crop-specific and a high-yield case under different conditions must not win).
"""
from __future__ import annotations

import numpy as np

from ..rag.store import CaseStore
from ..schemas.common import RetrievalItem

POPULARITY_WEIGHT = 0.5   # chosen on validation years 2015-2016 only (see artifacts/evaluation/experiments/recommend_tuning.json)

SCORE_METHOD = ("mean priority of top-3 supporting cases x support factor (0.7 + 0.3*min(1, n/3)); "
                "priority = weighted similarity to the user's conditions + hybrid relevance (see prioritizer). Heuristic, not a probability.")


def build_recommendations(store: CaseStore, items: list[RetrievalItem], profile_values: dict, excluded: list[str],
                          max_n: int = 5, crop_counts: dict | None = None, popularity_weight: float | None = None) -> list[dict]:
    excl = {e.lower() for e in excluded}
    by_crop: dict[str, list[RetrievalItem]] = {}
    for it in items:
        c = it.record["crop"]
        if c.lower() in excl:
            continue
        by_crop.setdefault(c, []).append(it)
    pw = POPULARITY_WEIGHT if popularity_weight is None else popularity_weight
    max_hist = max([int((crop_counts or {}).get(c, len(i))) for c, i in by_crop.items()] or [1]) or 1
    recs = []
    for crop, its in by_crop.items():
        its = sorted(its, key=lambda i: -(i.priority_score or 0))
        top = its[:3]
        # support = how much matching history exists for this crop (whole filtered dataset when known, else the retrieved pool)
        n_hist = int((crop_counts or {}).get(crop, len(its)))
        factor = 0.7 + 0.3 * min(1.0, max(n_hist, len(its)) / 5)
        score = float(np.mean([i.priority_score or 0 for i in top])) * factor
        if pw:   # blend in how often this crop is actually recorded for these filters (relative to the most frequent candidate)
            score = (1 - pw) * score + pw * min(1.0, max(n_hist, len(its)) / max_hist)
        ys = [i.record["yield"] for i in its if i.record.get("yield") is not None]
        years = sorted({i.record["year"] for i in its})
        states = sorted({i.record["state"] for i in its})
        seasons = sorted({i.record["season"] for i in its})
        soil_sims = [i.priority_components["soil_similarity"]["value"] for i in its if i.priority_components["soil_similarity"]["assessed"]]
        weather_sims = [i.priority_components["weather_similarity"]["value"] for i in its if i.priority_components["weather_similarity"]["assessed"]]
        reasons = [f"{len(its)} of the {len(items)} prioritised historical cases are {crop} "
                   f"(top case {its[0].case_id}, priority {its[0].priority_score:.2f})."]
        if profile_values.get("state"):
            n_state = sum(1 for i in its if i.record["state"].lower() == str(profile_values["state"]).lower())
            reasons.append(f"{n_state} supporting cases are from {profile_values['state']}.")
        else:
            reasons.append(f"Supporting cases come from {len(states)} state(s): {', '.join(states[:5])}{'...' if len(states) > 5 else ''}.")
        if profile_values.get("season"):
            n_s = sum(1 for i in its if i.record["season"].lower() == str(profile_values["season"]).lower())
            reasons.append(f"{n_s} supporting cases were grown in the {profile_values['season']} season.")
        if soil_sims:
            reasons.append(f"Average soil similarity to your values across supporting cases: {np.mean(soil_sims):.2f} (1 = identical).")
        if weather_sims:
            reasons.append(f"Average weather similarity: {np.mean(weather_sims):.2f} (1 = identical).")
        recs.append({
            "crop": crop,
            "score": round(score, 4), "score_method": SCORE_METHOD, "score_is_probability": False,
            "user_requested": bool(profile_values.get("crop") and profile_values["crop"].lower() == crop.lower()),
            "reasons": reasons,
            "supporting_case_ids": [i.case_id for i in its[:5]],
            "n_supporting_cases": len(its),
            "n_matching_records": n_hist,
            "n_synthetic_supporting": sum(1 for i in its if i.record.get("is_synthetic")),
            "historical_yield": {
                "median": float(np.median(ys)) if ys else None, "min": float(min(ys)) if ys else None,
                "max": float(max(ys)) if ys else None, "n": len(ys), "years": [years[0], years[-1]] if years else None,
                "basis": "ACTUAL historical records among the supporting cases (source units; crop-specific).",
            },
            "typical_context": {"states": states[:8], "seasons": seasons},
        })
    recs.sort(key=lambda r: -r["score"])
    # a user-requested crop stays visible even if not in the top N
    top = recs[:max_n]
    req = [r for r in recs if r["user_requested"] and r not in top]
    top = top + req[:1]
    for i, r in enumerate(top, 1):
        r["rank"] = i
        r["recommendation_id"] = None
    return top
