import time

from src.evaluation import metrics, test_cases


def test_health_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/health").json()["status"] == "ok"


def test_system_status_never_leaks_key(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.text
    assert "sk-ant" not in body and "api_key" not in body.lower()
    j = r.json()
    assert j["dataset"]["available"] and j["index"]["valid"] and j["yield_model"]["available"]
    assert j["dataset"]["target_met"] is True and j["dataset"]["synthetic_records"] == 35311 and "labelled synthetic" in j["dataset"]["note"]


def test_stats_and_options(client):
    s = client.get("/api/stats").json()
    assert s["records"] == 55000 and s["real_records"] == 19689 and s["synthetic_records"] == 35311 and s["crops"] == 55 and s["states"] == 30
    o = client.get("/api/metadata/options").json()
    assert "Rice" in o["crops"] and "soil_type" in o["unavailable_fields"]


def test_cases_list_filter_sort_paginate(client):
    r = client.get("/api/cases", params={"crop": "Rice", "state": "Assam", "sort_by": "yield", "order": "desc", "page_size": 5})
    j = r.json()
    assert r.status_code == 200 and len(j["items"]) == 5 and j["total"] > 5
    ys = [i["yield"] for i in j["items"]]
    assert ys == sorted(ys, reverse=True)
    assert all(i["crop"] == "Rice" for i in j["items"])
    assert client.get("/api/cases", params={"page": 2, "page_size": 5, "crop": "Rice"}).json()["page"] == 2
    assert client.get("/api/cases", params={"sort_by": "nonsense"}).status_code == 400
    assert client.get("/api/cases", params={"ph_min": 9, "ph_max": 5}).status_code == 422
    assert client.get("/api/cases", params={"page_size": 5000}).status_code == 422


def test_case_detail_and_404(client):
    cid = client.get("/api/cases", params={"page_size": 1}).json()["items"][0]["case_id"]
    d = client.get(f"/api/cases/{cid}").json()
    assert d["case_id"] == cid and "text" in d
    assert client.get("/api/cases/TM-NOPE").status_code == 404


def test_search_endpoint_and_validation(client):
    r = client.post("/api/cases/search", json={"query": "wheat punjab rabi", "top_k": 5, "filters": {"state": ["Punjab"]}})
    j = r.json()
    assert r.status_code == 200 and j["items"] and all(i["record"]["state"] == "Punjab" for i in j["items"])
    assert client.post("/api/cases/search", json={"query": "", "top_k": 5}).status_code == 422
    assert client.post("/api/cases/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/cases/search", json={"query": "x", "filters": {"ph_min": 50}}).status_code == 422
    assert client.post("/api/cases/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_payload_too_large(client):
    r = client.post("/api/cases/search", content=b"x" * 70000, headers={"content-type": "application/json", "content-length": "70000"})
    assert r.status_code == 413


def test_predictions_endpoints(client):
    y = client.post("/api/predictions/yield", json={"crop": "Wheat", "state": "Punjab", "season": "Rabi", "ph": 8, "rainfall": 600})
    assert y.status_code == 200 and y.json()["is_estimate"] is True
    assert client.post("/api/predictions/yield", json={"crop": "Nope", "state": "Punjab", "season": "Rabi"}).status_code == 422
    assert client.post("/api/predictions/yield", json={"crop": "Wheat", "state": "Punjab", "season": "Rabi", "ph": 99}).status_code == 422
    s = client.post("/api/predictions/suitability", json={"crop": "Wheat", "state": "Punjab", "season": "Rabi", "ph": 8})
    assert s.status_code == 200 and s.json()["method"] == "rule_based_historical_envelope"
    i = client.post("/api/analysis/impact", json={"crop": "Wheat"})
    assert i.status_code == 200 and i.json()["available"]


def test_advisory_validation_errors(client):
    assert client.post("/api/advisory/query", json={"query": ""}).status_code == 422
    assert client.post("/api/advisory/query", json={"query": "x" * 1001}).status_code == 422
    assert client.post("/api/advisory/query", json={"query": "rice", "ph": 99}).status_code == 422
    assert client.post("/api/advisory/query", json={"query": "rice", "top_k": 0}).status_code == 422
    assert client.post("/api/advisory/query", json={"nothing": 1}).status_code == 422


def test_advisory_classify_and_query_and_job(client, no_llm):
    c = client.post("/api/advisory/classify", json={"query": "wheat in Punjab Rabi pH 7.8", "use_llm": False}).json()
    assert c["conditions"]["crop"]["value"] == "Wheat" and c["conditions"]["ph"]["provenance"] == "parsed_from_query"
    q = client.post("/api/advisory/query", json={"query": "Which crops in Kerala Kharif with pH 5.7?", "use_llm": False})
    j = q.json()
    assert q.status_code == 200 and j["recommendations"] and j["query_id"]
    assert client.get(f"/api/advisory/queries/{j['query_id']}").json()["query_id"] == j["query_id"]
    assert client.get("/api/advisory/recent").json()["total"] >= 1
    job = client.post("/api/advisory/jobs", json={"query": "rice in Assam Winter", "use_llm": False}).json()
    for _ in range(120):
        st = client.get(f"/api/advisory/jobs/{job['job_id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "completed" and st["result"]["recommendations"]
    assert [s["status"] for s in st["stages"]][:5] == ["success"] * 5
    assert client.get("/api/advisory/jobs/nope").status_code == 404


def test_feedback_persistence_and_summary(client, no_llm):
    q = client.post("/api/advisory/query", json={"query": "rice in Assam Winter", "use_llm": False}).json()
    ok = client.post("/api/feedback", json={"query_id": q["query_id"], "crop": "Rice", "helpful": False, "rating": 2, "category": "incorrect_crop",
                                           "comment": "Wrong season", "case_ids": ["TM-X"]})
    assert ok.status_code == 201 and ok.json()["helpful"] is False
    client.post("/api/feedback", json={"query_id": q["query_id"], "crop": "Rice", "helpful": True})
    s = client.get("/api/feedback/summary").json()
    assert s["total"] == 2 and s["helpful"] == 1 and s["categories"][0]["category"] == "incorrect_crop"
    assert s["review_candidates"][0]["comment"] == "Wrong season"
    assert "No automatic model retraining" in " ".join(s["improvement_workflow"])
    lst = client.get("/api/feedback").json()
    assert lst["total"] == 2 and lst["items"][0]["id"] > lst["items"][1]["id"]


def test_feedback_validation(client):
    assert client.post("/api/feedback", json={}).status_code == 422                     # helpful missing
    assert client.post("/api/feedback", json={"helpful": True, "rating": 9}).status_code == 422
    assert client.post("/api/feedback", json={"helpful": True, "category": "nope"}).status_code == 422
    assert client.post("/api/feedback", json={"helpful": True, "query_id": "q-unknown"}).status_code == 404
    assert client.post("/api/feedback", json={"helpful": True, "comment": "x" * 3000}).status_code == 422


def test_evaluation_endpoints_not_evaluated_initially(client):
    o = client.get("/api/evaluations").json()
    assert o["status"] == "not_evaluated" and o["latest_run"] is None and o["yield_model"]["comparison"]
    assert client.get("/api/evaluations/runs").json()["items"] == []
    assert client.get("/api/evaluations/runs/eval-nope").status_code == 404
    tc = client.get("/api/evaluations/test-cases").json()
    assert tc["n"] >= 20 and tc["verification_problems"] == []


# ---------------------------------------------------------------- evaluation machinery
def test_retrieval_metrics_math():
    rel = {"a", "b", "c", "d"}
    ret = ["x", "a", "y", "b"]
    assert metrics.precision_at_k(ret, rel, 4) == 0.5
    assert metrics.recall_at_k(ret, rel, 4) == 0.5
    assert metrics.reciprocal_rank(ret, rel) == 0.5
    assert metrics.hit_at_k(ret, rel, 1) == 0.0 and metrics.hit_at_k(ret, rel, 2) == 1.0
    assert metrics.precision_at_k(ret, set(), 4) is None                      # no labels -> not computed
    assert metrics.mean([None, None]) is None


def test_eval_labels_verified_against_dataset(store):
    cases = test_cases.load(store)
    assert test_cases.verify(store, cases) == []
    df = store.df
    for c in cases:
        if c["type"] == "constraint_retrieval":
            e = c["expected"]
            assert len(c["relevant_case_ids"]) == ((df.crop == e["crop"]) & (df.state == e["state"]) & (df.season == e["season"])).sum()
    bad = [dict(cases[0], relevant_case_ids=cases[0]["relevant_case_ids"][:-1])]
    assert test_cases.verify(store, bad)                                       # tampering is detected


def test_evaluation_persistence_without_llm(store, tmp_db, monkeypatch, tmp_path):
    from src.config import settings
    from src.evaluation import runner
    from src.services import db
    monkeypatch.setattr(settings, "eval_dir", tmp_path / "eval")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(test_cases, "EXTRA_FILE", tmp_path / "none.json")
    cases = test_cases.load(store)
    keep = [c for c in cases if c["type"] in ("constraint_retrieval", "clarification", "abstention")][:5]
    monkeypatch.setattr(test_cases, "load", lambda *a, **k: keep)
    out = runner.run(label="unit", use_llm=False, deepeval_limit=0)
    assert out["summary"]["deepeval_status"].startswith("unavailable")
    de = out["summary"]["deepeval"]["faithfulness"]
    assert de["mean"] is None and de["n_measured"] == 0                         # never a fake score
    stored = db.get_eval_run(out["run_id"])
    assert stored["status"] == "completed" and stored["results"][0]["test_case_id"] == keep[0]["id"]
    assert (settings.eval_dir / f"{out['run_id']}.json").exists() and (settings.eval_dir / f"{out['run_id']}.md").exists()
