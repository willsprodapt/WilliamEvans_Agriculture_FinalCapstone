"""Feedback analysis. Feeds a CONTROLLED, human-reviewed improvement workflow - there is no automatic retraining."""
from __future__ import annotations

from collections import Counter

from . import db

WORKFLOW = [
    "1. Review unhelpful feedback below (review_candidates).",
    "2. Reproduce the query and verify the failure against the dataset.",
    "3. Add verified cases to data/evaluation/test_cases_extra.json (see README).",
    "4. Re-run: python scripts/run_evaluation.py and compare runs on the Evaluation page.",
    "No automatic model retraining or continuous learning is implemented.",
]


def summary() -> dict:
    fb = db.all_feedback()
    total = len(fb)
    helpful = sum(1 for f in fb if f["helpful"])
    cats = Counter(f["category"] for f in fb if not f["helpful"] and f["category"])
    by_crop: dict[str, dict] = {}
    for f in fb:
        if f["crop"]:
            d = by_crop.setdefault(f["crop"], {"crop": f["crop"], "helpful": 0, "not_helpful": 0})
            d["helpful" if f["helpful"] else "not_helpful"] += 1
    ratings = [f["rating"] for f in fb if f["rating"]]
    # candidates: unhelpful feedback with a concrete query/crop, newest first
    cand = [{"feedback_id": f["id"], "query_id": f["query_id"], "crop": f["crop"], "category": f["category"],
             "comment": f["comment"], "case_ids": f["case_ids"], "created_at": f["created_at"]}
            for f in reversed(fb) if not f["helpful"]][:20]
    daily = Counter((f["created_at"][:10], f["helpful"]) for f in fb)
    days = sorted({d for d, _ in daily})
    return {
        "total": total, "helpful": helpful, "not_helpful": total - helpful,
        "helpful_rate": (helpful / total) if total else None,
        "average_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "categories": [{"category": k, "count": v} for k, v in cats.most_common()],
        "by_crop": sorted(by_crop.values(), key=lambda d: -(d["helpful"] + d["not_helpful"]))[:15],
        "trend": [{"date": d, "helpful": daily.get((d, True), 0), "not_helpful": daily.get((d, False), 0)} for d in days],
        "recent": fb[-10:][::-1],
        "review_candidates": cand,
        "improvement_workflow": WORKFLOW,
    }
