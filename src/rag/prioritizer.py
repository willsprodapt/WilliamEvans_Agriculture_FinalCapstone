"""Transparent case prioritisation.

priority = sum(weight_i * component_i) / sum(weight_i for components that could be ASSESSED)

Components (weight): hybrid relevance (0.30), crop match (0.15), season match (0.10), region match (0.10),
soil similarity N/P/K/pH (0.15), weather similarity rainfall/temperature/humidity (0.10), data completeness (0.10).
- Components for which the user supplied no information are 'not_assessed' and excluded (weights renormalised);
  unknown compatibility is never counted as a match.
- Similarity of a numeric value = max(0, 1 - |case - user| / (2 * dataset std)).
- Historical yield is deliberately NOT a component (a high-yield case under different conditions must not outrank
  a comparable case).
The result is a heuristic priority in [0, 1], not a probability.
"""
from __future__ import annotations

import numpy as np

from ..schemas.common import RetrievalItem
from .store import CaseStore

WEIGHTS = {"hybrid_relevance": 0.30, "crop_match": 0.15, "season_match": 0.10, "region_match": 0.10,
           "soil_similarity": 0.15, "weather_similarity": 0.10, "data_completeness": 0.10}
SOIL = ["n", "p", "k", "ph"]
WEATHER = ["rainfall", "temperature", "humidity"]


def _sim(store: CaseStore, rec: dict, user: dict, fields: list[str]):
    sims = {}
    for f in fields:
        if user.get(f) is None or rec.get(f) is None:
            continue
        sims[f] = max(0.0, 1.0 - abs(float(rec[f]) - float(user[f])) / store.scale[f])
    return sims


def prioritize(store: CaseStore, items: list[RetrievalItem], user: dict) -> list[RetrievalItem]:
    """user: dict with optional crop/state/season/n/p/k/ph/rainfall/temperature/humidity."""
    if not items:
        return items
    max_h = max(i.hybrid_score for i in items) or 1.0
    for it in items:
        rec = it.record
        comps: dict[str, dict] = {}

        def put(name, value, detail=None):
            comps[name] = {"value": None if value is None else round(float(value), 4), "weight": WEIGHTS[name],
                           "assessed": value is not None, "detail": detail}

        put("hybrid_relevance", it.hybrid_score / max_h, "RRF score relative to best candidate in this result set")
        put("crop_match", (1.0 if str(rec["crop"]).lower() == str(user["crop"]).lower() else 0.0) if user.get("crop") else None)
        put("season_match", (1.0 if str(rec["season"]).lower() == str(user["season"]).lower() else 0.0) if user.get("season") else None)
        put("region_match", (1.0 if str(rec["state"]).lower() == str(user["state"]).lower() else 0.0) if user.get("state") else None)
        s = _sim(store, rec, user, SOIL)
        put("soil_similarity", np.mean(list(s.values())) if s else None, {k: round(v, 3) for k, v in s.items()} or None)
        w = _sim(store, rec, user, WEATHER)
        put("weather_similarity", np.mean(list(w.values())) if w else None, {k: round(v, 3) for k, v in w.items()} or None)
        put("data_completeness", rec.get("completeness"))

        assessed = {k: c for k, c in comps.items() if c["assessed"]}
        tw = sum(c["weight"] for c in assessed.values())
        score = sum(c["weight"] * c["value"] for c in assessed.values()) / tw if tw else 0.0
        for c in comps.values():
            c["contribution"] = round(c["weight"] * c["value"] / tw, 4) if c["assessed"] and tw else 0.0
        it.priority_score = round(score, 4)
        it.priority_components = comps
        top = sorted(assessed.items(), key=lambda kv: -kv[1]["contribution"])[:3]
        skipped = [k for k, c in comps.items() if not c["assessed"]]
        it.priority_explanation = ("Top contributors: " + ", ".join(f"{k} ({c['value']:.2f})" for k, c in top)
                                   + (f". Not assessed (no user input): {', '.join(skipped)}." if skipped else "."))
    return sorted(items, key=lambda i: -(i.priority_score or 0))
