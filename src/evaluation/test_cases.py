"""Verified evaluation case builder.

Relevance labels are derived from the ACTUAL dataset by deterministic structured filters, not by an LLM:
  constraint_retrieval  - query states crop+state+season in prose; relevant = every record whose structured fields satisfy
                          all three (verified by exact match on the processed dataset).
  conditions_recommendation - query gives state+season+real soil/weather values of an anchor record; every recommended crop must
                          have >=1 historical record in that state+season; relevant cases = records in that state+season.
  abstention            - crop/state pair with ZERO records in the dataset; the system must not recommend that crop and must say
                          there is no evidence.
  clarification         - queries without any usable agricultural context; the system must ask for clarification.
Queries are written from templates over real values; they are test INPUTS, never presented as historical records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, settings
from ..rag.store import CaseStore

EVAL_DIR = ROOT / "data" / "evaluation"
CASES_FILE = EVAL_DIR / "test_cases.json"
EXTRA_FILE = EVAL_DIR / "test_cases_extra.json"


def build(store: CaseStore, seed: int | None = None, n_retrieval: int = 16, n_reco: int = 8, n_abstain: int = 4) -> list[dict]:
    seed = settings.random_seed if seed is None else seed
    rng = np.random.default_rng(seed)
    df = store.df
    cases: list[dict] = []

    # --- A. constraint retrieval: crop+state+season groups with 3..30 records, spread over crops
    g = df.groupby(["crop", "state", "season"]).size().reset_index(name="n")
    g = g[(g["n"] >= 3) & (g["n"] <= 30)]
    crops = sorted(g["crop"].unique())
    chosen_crops = rng.choice(crops, size=min(n_retrieval, len(crops)), replace=False)
    for i, crop in enumerate(sorted(chosen_crops)):
        row = g[g["crop"] == crop].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        ids = df[(df["crop"] == row.crop) & (df["state"] == row.state) & (df["season"] == row.season)]["case_id"].tolist()
        yrs = df[df["case_id"].isin(ids)]["year"]
        ref = (f"The dataset holds {len(ids)} historical records of {row.crop} in {row.state} during the {row.season} season "
               f"({int(yrs.min())}-{int(yrs.max())}).")
        cases.append({"id": f"RET-{i + 1:02d}", "type": "constraint_retrieval",
                      "query": f"{row.crop} cultivation in {row.state} during the {row.season} season",
                      "expected": {"crop": row.crop, "state": row.state, "season": row.season},
                      "relevant_case_ids": ids, "reference_answer": ref,
                      "provenance": "relevant = all dataset records with exact crop+state+season match (deterministic filter)"})

    # --- B. conditions -> recommendation
    st = df.groupby(["state", "season"]).size().reset_index(name="n")
    st = st[st["n"] >= 40]
    for i in range(n_reco):
        r = st.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        sub = df[(df["state"] == r.state) & (df["season"] == r.season)]
        anchor = sub.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        top = sub["crop"].value_counts().head(5)
        cases.append({"id": f"REC-{i + 1:02d}", "type": "conditions_recommendation",
                      "query": (f"Which crops have historically performed well in {r.state} during the {r.season} season? "
                                f"Soil pH is {anchor.ph:g}, rainfall about {anchor.rainfall:.0f} mm and mean temperature {anchor.temperature:.1f} C."),
                      "expected": {"state": r.state, "season": r.season, "allowed_crops": sorted(sub["crop"].unique().tolist())},
                      "relevant_case_ids": sub["case_id"].tolist(),
                      "reference_answer": (f"In {r.state} during the {r.season} season the most frequently recorded crops are "
                                           + ", ".join(f"{c} ({n} records)" for c, n in top.items()) + "."),
                      "provenance": f"anchor record {anchor.case_id}; allowed crops = crops with >=1 record in state+season"})

    # --- C. abstention: crop never recorded in a state
    pairs = set(zip(df["crop"], df["state"]))
    allc, alls = sorted(df["crop"].unique()), sorted(df["state"].unique())
    cand = [(c, s) for c in allc for s in alls if (c, s) not in pairs and df[df["crop"] == c].shape[0] > 200]
    rng.shuffle(cand)
    for i, (c, s) in enumerate(cand[:n_abstain]):
        cases.append({"id": f"ABS-{i + 1:02d}", "type": "abstention",
                      "query": f"I want to grow {c} in {s}. Is it a good choice?",
                      "expected": {"crop": c, "state": s, "records_in_dataset": 0}, "relevant_case_ids": [],
                      "reference_answer": f"There are no historical records of {c} in {s}; the system should not recommend it on evidence.",
                      "provenance": f"verified: 0 records for crop={c}, state={s}"})

    # --- D. clarification
    for i, q in enumerate(["What should I plant?", "Give me a recommendation please", "Which one is best?"][:3]):
        cases.append({"id": f"CLR-{i + 1:02d}", "type": "clarification", "query": q, "expected": {"needs_clarification": True},
                      "relevant_case_ids": [], "reference_answer": "The system should ask for state/season/conditions.",
                      "provenance": "no agricultural context in query"})
    return cases


def save(cases: list[dict]):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CASES_FILE.write_text(json.dumps(cases, indent=2), encoding="utf-8")


def load(store: CaseStore | None = None, rebuild: bool = False) -> list[dict]:
    if rebuild or not CASES_FILE.exists():
        if store is None:
            raise FileNotFoundError("test_cases.json missing; run scripts/run_evaluation.py --rebuild-cases")
        save(build(store))
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if EXTRA_FILE.exists():   # human-reviewed additions from the feedback workflow
        cases += json.loads(EXTRA_FILE.read_text(encoding="utf-8"))
    return cases


def verify(store: CaseStore, cases: list[dict]) -> list[str]:
    """Re-check every label against the dataset. Returns a list of problems (empty = verified)."""
    probs = []
    df = store.df
    for c in cases:
        if c["type"] == "constraint_retrieval":
            e = c["expected"]
            true_ids = set(df[(df["crop"] == e["crop"]) & (df["state"] == e["state"]) & (df["season"] == e["season"])]["case_id"])
            if true_ids != set(c["relevant_case_ids"]):
                probs.append(f"{c['id']}: relevant ids differ from dataset filter")
        elif c["type"] == "abstention":
            e = c["expected"]
            if ((df["crop"] == e["crop"]) & (df["state"] == e["state"])).any():
                probs.append(f"{c['id']}: crop/state pair actually exists")
    return probs
