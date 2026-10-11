"""All HTTP routes under /api."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from ..config import settings
from ..evaluation import runner as eval_runner
from ..evaluation import test_cases as eval_cases
from ..models.yield_model import ModelUnavailable
from ..rag.retriever import index_status
from ..rag.store import SORTABLE, DatasetUnavailable
from ..schemas.advisory import (AdvisoryRequest, ClassifyRequest, FeedbackIn, ImpactRequest, SearchRequest,
                                SuitabilityRequest, YieldRequest)
from ..schemas.common import RANGE_FIELDS, CaseFilters
from ..services import db, feedback as fb_svc
from ..services import impact as impact_svc
from ..services import suitability
from ..services.llm import llm
from .state import state

router = APIRouter(prefix="/api")
START = time.time()


# ============================================================ health / system
@router.get("/health", tags=["health"])
def api_health():
    return {"status": "ok", "uptime_s": int(time.time() - START)}


@router.get("/system/status", tags=["health"])
def system_status():
    out: dict = {"api": "ok", "llm": llm.status()}
    try:
        store = state.store()
        out["dataset"] = {"available": True, "records": len(store), "dataset_version": store.dataset_version,
                          "target_rows": settings.target_rows, "target_met": len(store) >= settings.target_rows,
                          "source_rows": store.report.get("source_row_count"),
                          "real_records": int((~store.df["is_synthetic"]).sum()), "synthetic_records": int(store.df["is_synthetic"].sum()),
                          "note": (f"{int((~store.df['is_synthetic']).sum()):,} real records + {int(store.df['is_synthetic'].sum()):,} labelled synthetic records "
                                   "(is_synthetic=true) reach the 55,000 target. Statistics, suitability envelopes and the yield model use real records only.")
                          if store.df["is_synthetic"].any() else
                          (None if len(store) >= settings.target_rows else f"Source contains only {store.report.get('source_row_count')} valid rows; no synthetic rows were created.")}
        out["index"] = index_status(store)
        out["retrieval"] = state.retriever().status
    except DatasetUnavailable as e:
        out["dataset"] = {"available": False, "reason": str(e)}
        out["index"] = index_status(None)
    try:
        p = state.predictor()
        out["yield_model"] = {"available": True, "version": p.meta["model_version"], "trained_at": p.meta["trained_at"]}
    except ModelUnavailable as e:
        out["yield_model"] = {"available": False, "reason": str(e)}
    out["database"] = {"queries": db.count_queries()}
    return out


@router.get("/stats", tags=["health"])
def stats():
    s = state.store().stats()
    s["advisory_queries_processed"] = db.count_queries()
    s["feedback_count"] = db.list_feedback(1, 1)["total"]
    s["index"] = index_status(state.store())
    return s


# ============================================================ advisory
@router.post("/advisory/query", tags=["advisory"])
def advisory_query(req: AdvisoryRequest):
    """Run the full multi-agent advisory workflow synchronously."""
    return state.orchestrator().run(req)


@router.post("/advisory/jobs", tags=["advisory"], status_code=202)
def advisory_job_start(req: AdvisoryRequest):
    """Start the advisory workflow in the background; poll GET /advisory/jobs/{job_id} for real stage progress."""
    return state.start_advisory_job(req)


@router.get("/advisory/jobs/{job_id}", tags=["advisory"])
def advisory_job_status(job_id: str):
    j = state.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in j.items() if k != "created"}


@router.post("/advisory/classify", tags=["advisory"])
def advisory_classify(req: ClassifyRequest):
    orch = state.orchestrator()
    profile, notes = orch.classifier.classify(req.query, None, req.use_llm)
    return {"conditions": profile, "notes": notes}


@router.post("/advisory/recommend", tags=["advisory"])
def advisory_recommend(req: AdvisoryRequest):
    """Same pipeline, returning only the recommendation-focused part of the response."""
    r = state.orchestrator().run(req)
    return {"query_id": r.query_id, "status": r.status, "conditions": r.conditions, "recommendations": r.recommendations,
            "narrative": r.narrative, "judge": r.judge, "warnings": r.warnings, "clarification": r.clarification}


@router.get("/advisory/recent", tags=["advisory"])
def advisory_recent(limit: int = Query(10, ge=1, le=100)):
    return {"items": db.recent_queries(limit), "total": db.count_queries()}


@router.get("/advisory/queries/{query_id}", tags=["advisory"])
def advisory_get(query_id: str):
    q = db.get_query(query_id)
    if not q:
        raise HTTPException(404, "Query not found")
    return q["response"]


@router.get("/insights/advisory-summary", tags=["advisory"])
def advisory_summary(limit: int = Query(100, ge=1, le=500)):
    """Aggregates over persisted advisory runs: judge verdicts, risk categories, agent outcomes."""
    rows = db.recent_queries(limit)
    verdicts, risk, agents, crops = Counter(), Counter(), Counter(), Counter()
    n = 0
    for r in rows:
        q = db.get_query(r["id"])
        if not q:
            continue
        resp = q["response"]
        n += 1
        v = (resp.get("judge") or {}).get("verdict")
        if v:
            verdicts[v] += 1
        for rec in resp.get("recommendations", []):
            crops[rec["crop"]] += 1
            c = (rec.get("risk") or {}).get("category")
            if c:
                risk[c] += 1
        for a in resp.get("agents", []):
            agents[(a["name"], a["status"])] += 1
    return {"n_queries": n,
            "judge_verdicts": [{"verdict": k, "count": v} for k, v in verdicts.items()],
            "risk_categories": [{"category": k, "count": v} for k, v in risk.items()],
            "agent_outcomes": [{"agent": a, "status": s, "count": c} for (a, s), c in sorted(agents.items())],
            "top_recommended_crops": [{"crop": k, "count": v} for k, v in crops.most_common(10)]}


# ============================================================ cases
def _filters_from(request: Request) -> CaseFilters:
    qp = request.query_params
    d: dict = {}
    for k in ("crop", "season", "state"):
        v = [x for x in qp.getlist(k) if x]
        if v:
            d[k] = v
    for k in ("year_min", "year_max"):
        if qp.get(k):
            d[k] = int(qp[k])
    for name in RANGE_FIELDS:
        for suf in ("_min", "_max"):
            if qp.get(name + suf) not in (None, ""):
                d[name + suf] = float(qp[name + suf])
    try:
        return CaseFilters(**d)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))


@router.get("/cases", tags=["cases"])
def list_cases(request: Request, q: str | None = Query(None, max_length=200, description="substring match on crop/state/season or exact case id"),
               sort_by: str = Query("year", description=f"one of {SORTABLE}"), order: str = Query("desc", pattern="^(asc|desc)$"),
               page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200)):
    """Filter via query params: crop, state, season (repeatable), year_min/max, and <field>_min/_max for
    n,p,k,ph,rainfall,temperature,humidity,fertilizer,pesticide,fertilizer_per_ha,pesticide_per_ha,area,yield."""
    try:
        f = _filters_from(request)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return state.store().list_cases(f, sort_by, order, page, page_size, q)


@router.get("/cases/{case_id}", tags=["cases"])
def get_case(case_id: str):
    c = state.store().get(case_id)
    if not c:
        raise HTTPException(404, f"Case {case_id} not found")
    return c


@router.post("/cases/search", tags=["cases"])
def search_cases(req: SearchRequest):
    """Hybrid (BM25 + semantic, RRF) search with optional metadata filters."""
    return state.retriever().search(req.query, req.filters, top_k=req.top_k, mode=req.mode)


@router.get("/metadata/options", tags=["cases"])
def metadata_options():
    return state.store().options()


# ============================================================ predictions / analysis
@router.post("/predictions/yield", tags=["predictions"])
def predict_yield(req: YieldRequest):
    store = state.store()
    for name, val, pool in (("crop", req.crop, store.df["crop"]), ("state", req.state, store.df["state"]), ("season", req.season, store.df["season"])):
        if val not in set(pool):
            raise HTTPException(422, f"Unknown {name} '{val}'. See /api/metadata/options")
    d = req.model_dump()
    return state.predictor().predict(d)


@router.post("/predictions/suitability", tags=["predictions"])
def predict_suitability(req: SuitabilityRequest):
    d = req.model_dump()
    return suitability.assess(state.store(), req.crop, {k: v for k, v in d.items() if v is not None and k != "crop"})


@router.post("/analysis/impact", tags=["predictions"])
def analysis_impact(req: ImpactRequest):
    return impact_svc.analyze(state.store(), req.crop, req.state, req.season)


# ============================================================ evaluation
_eval_jobs: dict[str, dict] = {}


@router.get("/evaluations", tags=["evaluation"])
def evaluations_overview():
    """Latest evaluation summary. Returns status 'not_evaluated' until a run exists."""
    runs = db.list_eval_runs()
    ym = settings.model_dir / "yield_metrics.json"
    yield_metrics = None
    if ym.exists():
        y = json.loads(ym.read_text())
        yield_metrics = {"meta": y["meta"], "split": y["split"], "comparison": y["comparison"],
                         "permutation_importance": y.get("permutation_importance")}
    return {"status": "evaluated" if runs else "not_evaluated", "latest_run": runs[0] if runs else None, "n_runs": len(runs),
            "yield_model": yield_metrics, "active_jobs": [j for j in _eval_jobs.values() if j["status"] == "running"]}


@router.get("/evaluations/runs", tags=["evaluation"])
def evaluation_runs():
    return {"items": db.list_eval_runs()}


@router.get("/evaluations/runs/{run_id}", tags=["evaluation"])
def evaluation_run(run_id: str):
    r = db.get_eval_run(run_id)
    if r:
        return r
    if run_id in _eval_jobs:
        return {"run_id": run_id, **_eval_jobs[run_id]}
    raise HTTPException(404, "Evaluation run not found")


@router.get("/evaluations/test-cases", tags=["evaluation"])
def evaluation_test_cases():
    cases = eval_cases.load(state.store())
    return {"items": cases, "n": len(cases), "verification_problems": eval_cases.verify(state.store(), cases)}


@router.post("/evaluations/run", tags=["evaluation"], status_code=202)
def evaluation_run_start(label: str | None = Query(None, max_length=60), deepeval_limit: int = Query(10, ge=0, le=40),
                         use_llm: bool = True):
    if any(j["status"] == "running" for j in _eval_jobs.values()):
        raise HTTPException(409, "An evaluation is already running")
    state.orchestrator()  # fail fast with 503 if artifacts are missing
    run_id = "eval-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    job = {"status": "running", "done": 0, "total": 0, "message": "starting", "error": None, "started": time.time()}
    _eval_jobs[run_id] = job

    def prog(i, n, msg):
        job.update(done=i, total=n, message=msg)

    def work():
        try:
            eval_runner.run(label=label, deepeval_limit=deepeval_limit, use_llm=use_llm, run_id=run_id, progress=prog)
            job["status"] = "completed"
        except Exception as e:  # noqa: BLE001
            job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id, "status": "running"}


# ============================================================ feedback
@router.post("/feedback", tags=["feedback"], status_code=201)
def post_feedback(fb: FeedbackIn):
    if fb.query_id and not db.get_query(fb.query_id):
        raise HTTPException(404, f"query_id {fb.query_id} not found")
    return db.add_feedback(fb.model_dump())


@router.get("/feedback", tags=["feedback"])
def get_feedback(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100)):
    return db.list_feedback(page, page_size)


@router.get("/feedback/summary", tags=["feedback"])
def feedback_summary():
    return fb_svc.summary()
