"""Tests for the evaluation methodology itself: metric maths, split integrity, synthetic-data leakage rules, safety/uncertainty checks,
exclusion handling and provenance. These use the real processed dataset (no mocks)."""
import numpy as np
import pandas as pd
import pytest

from src.agents.agents import ValidationJudge, build_facts, RecommendationNarrator
from src.evaluation import metrics
from src.evaluation.exp_ml import TRAIN_END, VAL_END, load_real, split_real
from src.schemas.common import ConditionProfile
from src.services.data_prep import generate_synthetic


# ---------------------------------------------------------------- metrics
def test_ndcg_known_values():
    rel = {"a", "b"}
    assert metrics.ndcg_at_k(["a", "b", "x"], rel, 3) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["x", "a", "b"], rel, 3) == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert metrics.ndcg_at_k(["x", "y"], rel, 2) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), 1) is None


def test_r_precision_and_recall_caps():
    rel = {"a", "b", "c"}
    assert metrics.r_precision(["a", "x", "b", "c"], rel, 10) == pytest.approx(2 / 3)   # k' = 3 -> top-3 = a, x, b
    assert metrics.r_precision(["a", "b", "c"], rel, 10) == 1.0
    assert metrics.recall_at_k(["a", "x"], rel, 2) == pytest.approx(1 / 3)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    m, lo, hi = metrics.bootstrap_ci([0.0, 1.0] * 50)
    assert lo <= m <= hi and (m, lo, hi) == metrics.bootstrap_ci([0.0, 1.0] * 50)
    assert metrics.bootstrap_ci([]) is None


# ---------------------------------------------------------------- splits & synthetic leakage
@pytest.fixture(scope="module")
def real_split():
    real, weather = load_real()
    return real, weather, split_real(real)


def test_split_is_temporal_disjoint_and_complete(real_split):
    real, _, (tr, va, te) = real_split
    assert tr["year"].max() <= TRAIN_END < va["year"].min() and va["year"].max() <= VAL_END < te["year"].min()
    assert len(tr) + len(va) + len(te) == len(real)
    assert set(tr["case_id"]).isdisjoint(te["case_id"]) and set(va["case_id"]).isdisjoint(te["case_id"])
    assert (real["data_source"] == "kaggle").all()          # splits are built from Kaggle-sourced rows only


def test_synthetic_generator_uses_training_rows_only(real_split):
    real, weather, (tr, va, te) = real_split
    s = generate_synthetic(tr, 800, seed=3, weather=weather)
    assert len(s) == 800 and (s["data_source"] == "synthetic").all() and s["is_synthetic"].all()
    # no future years, no unseen crop-state-season combination -> nothing from validation/test can leak in
    assert s["year"].max() <= TRAIN_END
    assert set(zip(s.crop, s.state, s.season)) <= set(zip(tr.crop, tr.state, tr.season))
    # internal consistency and plausibility
    assert np.allclose(s["production"], (s["yield"] * s["area"]).round(0), atol=1.0)
    assert (s[["area", "yield", "fertilizer", "pesticide"]] >= 0).all().all() and not s[["yield", "area"]].isna().any().any()
    # same crop's yield range stays within the training envelope (+-20%)
    lim = tr.groupby("crop")["yield"].agg(["min", "max"])
    assert (s["yield"] <= 1.2 * lim.loc[s["crop"], "max"].to_numpy() + 1e-9).all()


def test_synthetic_generator_is_deterministic_and_has_no_exact_duplicates(real_split):
    _, weather, (tr, _, _) = real_split
    a, b = generate_synthetic(tr, 300, seed=11, weather=weather), generate_synthetic(tr, 300, seed=11, weather=weather)
    pd.testing.assert_frame_equal(a, b)
    assert not a.duplicated(["crop", "state", "season", "year", "area", "yield"]).any()


def test_provenance_field_present_and_consistent(store):
    assert set(store.df["data_source"].unique()) == {"kaggle", "synthetic"}
    assert (store.df["data_source"].eq("synthetic") == store.df["is_synthetic"]).all()
    assert "data_source" in store.record(0)


# ---------------------------------------------------------------- safety / uncertainty checks
def _judge(store):
    return ValidationJudge(store)


def _rec(crop, n, ids):
    return {"rank": 1, "crop": crop, "score": 0.5, "supporting_case_ids": ids, "n_supporting_cases": n,
            "historical_yield": {"median": 1.0, "min": 1.0, "max": 1.0, "n": 1, "years": [2000, 2000]},
            "risk": {"category": "low", "score": 0.9, "risk_factors": [], "missing_conditions": []}}


def test_judge_flags_dose_instructions_and_missing_uncertainty(store):
    crop = store.df["crop"].iloc[0]
    ids = store.df[store.df["crop"] == crop]["case_id"].head(2).tolist()
    rec = _rec(crop, 2, ids)
    profile = ConditionProfile()
    facts = build_facts([rec], profile, [])
    bad = {"summary": f"{crop} is great.", "per_crop": [{"crop": crop, "explanation": "Apply 50 kg/ha urea fertilizer before sowing."}]}
    det = _judge(store).deterministic([rec], profile, bad, facts)
    assert det["safety_flags"] and not det["passed"]
    assert det["uncertainty_flags"] == [crop]                 # limited evidence (n=2) not hedged
    ok = {"summary": "x", "per_crop": [{"crop": crop, "explanation": "Evidence is limited, only 2 similar cases."}]}
    det2 = _judge(store).deterministic([rec], profile, ok, facts)
    assert not det2["safety_flags"] and det2["uncertainty_flags"] == []


def test_template_narrative_hedges_low_evidence_and_has_no_doses(store):
    crop = store.df["crop"].iloc[0]
    ids = store.df[store.df["crop"] == crop]["case_id"].head(1).tolist()
    facts = build_facts([_rec(crop, 1, ids)], ConditionProfile(), [])
    t = RecommendationNarrator.template(facts)
    assert "limited" in t["per_crop"][0]["explanation"].lower()
    det = _judge(store).deterministic([_rec(crop, 1, ids)], ConditionProfile(), t, facts)
    assert not det["safety_flags"] and det["uncertainty_flags"] == []


# ---------------------------------------------------------------- recommendation behaviour
def test_excluded_dominant_crop_does_not_empty_recommendations(store, retriever, predictor, no_llm, tmp_db):
    from src.agents.orchestrator import AdvisoryOrchestrator
    from src.schemas.advisory import AdvisoryRequest
    df = store.df[~store.df["is_synthetic"]]
    g = df.groupby(["state", "season"]).size().reset_index(name="n").sort_values("n", ascending=False).iloc[0]
    top = df[(df.state == g.state) & (df.season == g.season)]["crop"].value_counts().index[0]
    res = AdvisoryOrchestrator(store, retriever, predictor).run(
        AdvisoryRequest(query=f"Suggest crops for {g.state} in {g.season} but not {top}", use_llm=False))
    assert res.status in ("completed", "completed_with_warnings")
    assert res.recommendations and top not in [r["crop"] for r in res.recommendations]


def test_retrieval_year_cutoff_hides_later_evidence(store, retriever):
    from src.agents.agents import CaseRetrievalAgent
    from src.agents.classifier import ConditionClassifier
    prof, _ = ConditionClassifier(store).classify("Which crops suit Punjab in Rabi?", None, False)
    out = CaseRetrievalAgent(store, retriever).run("Which crops suit Punjab in Rabi?", prof, include_synthetic=False, year_max=2010)
    assert out["pool"] and max(i.record["year"] for i in out["pool"]) <= 2010
    assert all(not i.record["is_synthetic"] for i in out["pool"])
