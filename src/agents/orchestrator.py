"""Central orchestrator: validation -> classification -> retrieval -> prioritisation -> recommendation ->
(risk | yield | impact in parallel) -> explanation -> judge -> consolidated response.

Each stage records REAL status/timing; a stage is only marked 'success' after its operation completed. Optional
agents fail independently without discarding other agents' results.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from typing import Callable

from ..config import settings
from ..models.yield_model import ModelUnavailable, YieldPredictor
from ..rag.retriever import HybridRetriever, RetrievalError
from ..rag.store import CaseStore
from ..schemas.advisory import AdvisoryRequest, AdvisoryResponse, AgentStatus
from ..schemas.common import ConditionProfile
from ..services import db
from ..services.recommend import build_recommendations
from ..utils.guardrails import injection_suspected, sanitize_query
from .agents import (CaseRetrievalAgent, ImpactAnalysisAgent, RecommendationNarrator, SuitabilityRiskAgent,
                     ValidationJudge, YieldPredictionAgent, build_facts)
from .classifier import ConditionClassifier

log = logging.getLogger("terramind.orchestrator")

STAGES = [("validation", "Input validation"), ("classification", "Condition classification"),
          ("retrieval", "Hybrid retrieval"), ("prioritization", "Case prioritization"),
          ("recommendation", "Crop recommendation"), ("analysis", "Risk & yield analysis"), ("judge", "Judge validation")]
AGENT_TIMEOUT_S = 90


class AdvisoryOrchestrator:
    def __init__(self, store: CaseStore, retriever: HybridRetriever, predictor: YieldPredictor | None):
        self.store = store
        self.classifier = ConditionClassifier(store)
        self.retrieval = CaseRetrievalAgent(store, retriever)
        self.risk = SuitabilityRiskAgent(store)
        self.yield_agent = YieldPredictionAgent(store, predictor)
        self.impact = ImpactAnalysisAgent(store)
        self.narrator = RecommendationNarrator()
        self.judge = ValidationJudge(store)
        self.pool = ThreadPoolExecutor(max_workers=6)

    def run(self, req: AdvisoryRequest, query_id: str | None = None,
            on_progress: Callable[[list[dict], list[dict]], None] | None = None) -> AdvisoryResponse:
        t_start = time.time()
        qid = query_id or db.new_id("q")
        stages = [{"key": k, "label": l, "status": "pending", "duration_ms": None, "message": None} for k, l in STAGES]
        agents: dict[str, AgentStatus] = {}
        warnings: list[str] = []

        def emit():
            if on_progress:
                on_progress([dict(s) for s in stages], [a.model_dump() for a in agents.values()])

        def stage(key: str, status: str, msg: str | None = None, t0: float | None = None):
            s = next(x for x in stages if x["key"] == key)
            s["status"] = status
            s["message"] = msg
            if t0 is not None:
                s["duration_ms"] = int((time.time() - t0) * 1000)
            emit()

        def agent(name: str, status: str, t0: float | None = None, detail: str | None = None):
            agents[name] = AgentStatus(name=name, status=status, detail=detail,
                                       duration_ms=None if t0 is None else int((time.time() - t0) * 1000))
            emit()

        def finish(status: str, **kw) -> AdvisoryResponse:
            for s in stages:
                if s["status"] in ("pending", "running"):
                    s["status"] = "skipped"
            md = {"duration_ms": int((time.time() - t_start) * 1000), "llm_model": settings.llm_model if settings.llm_configured else None,
                  "dataset_version": self.store.dataset_version, "n_dataset_records": len(self.store),
                  "retrieval_backends": self.retrieval.retriever.status}
            resp = AdvisoryResponse(query_id=qid, status=status, query=req.query, validated_input=req.model_dump(exclude_none=True),
                                    agents=list(agents.values()), stages=stages, warnings=warnings, metadata=md, **kw)
            try:
                db.save_query(qid, req.query, status, md["duration_ms"], resp.model_dump())
            except Exception as e:  # noqa: BLE001
                log.error("could not persist query: %s", e)
            emit()
            return resp

        # 1. validation -------------------------------------------------------------------
        t0 = time.time()
        stage("validation", "running")
        query = sanitize_query(req.query)
        if injection_suspected(query):
            warnings.append("The query contains text resembling instructions to the AI system. It is treated purely as data.")
        stage("validation", "success", "Request schema, ranges and length validated", t0)

        # 2. classification ---------------------------------------------------------------
        t0 = time.time()
        stage("classification", "running")
        agent("Condition Classification Agent", "running")
        try:
            profile, notes = self.classifier.classify(query, req.explicit(), req.use_llm)
        except Exception as e:  # noqa: BLE001
            log.exception("classification failed")
            agent("Condition Classification Agent", "failed", t0, str(e))
            stage("classification", "failed", str(e), t0)
            return finish("failed", conditions=None, clarification="Condition classification failed.")
        agent("Condition Classification Agent", "success", t0, f"LLM: {notes['llm']}")
        stage("classification", "success", f"LLM: {notes['llm']}", t0)
        if str(notes["llm"]).startswith(("failed", "unavailable")):
            warnings.append(f"LLM classification unavailable ({notes['llm']}); used deterministic parsing only.")
        warnings += profile.unusual_values
        if profile.needs_clarification:
            for k in ("retrieval", "prioritization", "recommendation", "analysis", "judge"):
                stage(k, "skipped", "Waiting for clarification")
            return finish("needs_clarification", conditions=profile, clarification=profile.clarification_question)

        # 3-4. retrieval + prioritisation ----------------------------------------------------
        t0 = time.time()
        stage("retrieval", "running")
        agent("Historical Case Retrieval Agent", "running")
        try:
            rr = self.retrieval.run(query, profile, req.top_k, req.include_synthetic)
        except RetrievalError as e:
            agent("Historical Case Retrieval Agent", "failed", t0, str(e))
            stage("retrieval", "failed", str(e), t0)
            return finish("failed", conditions=profile, warnings=warnings)
        resp = rr["response"]
        if resp.fallback:
            warnings.append(resp.fallback)
        warnings += rr.get("notes", [])
        agent("Historical Case Retrieval Agent", "success", t0, f"{len(rr['pool'])} candidate cases")
        stage("retrieval", "success", f"{len(rr['pool'])} candidates; {resp.filtered_out} filtered out by metadata", t0)
        t1 = time.time()
        stage("prioritization", "running")
        pool = rr["pool"]
        stage("prioritization", "success", f"Ranked {len(pool)} cases by transparent priority score", t1)
        retrieval_info = {"mode": resp.mode, "filters_applied": rr["filters"], "backends": resp.backends, "fallback": resp.fallback,
                          "total_candidates_after_filter": resp.total_candidates_after_filter, "filtered_out": resp.filtered_out,
                          "scoring": resp.scoring, "filter_diagnostics": rr["filter_diagnostics"],
                          "priority_formula": "see src/rag/prioritizer.py"}
        cases = [c.model_dump() for c in rr["shown"]]
        if not pool:
            stage("recommendation", "skipped", "No historical evidence matched the filters")
            stage("analysis", "skipped")
            stage("judge", "skipped")
            warnings.append("No historical cases match the stated filters. No recommendation is made (no evidence).")
            return finish("no_evidence", conditions=profile, retrieval=retrieval_info, cases=[],
                          clarification="No historical cases matched. Try removing the state or season constraint.")

        # 5. recommendation ----------------------------------------------------------------
        t0 = time.time()
        stage("recommendation", "running")
        agent("Crop Recommendation Agent", "running")
        excl = {e.lower() for e in profile.excluded_crops}
        # exclusions are applied BEFORE truncating the pool: otherwise an excluded crop that dominates the top results empties it
        rec_pool = [c for c in pool if c.record["crop"].lower() not in excl][:60]
        recs = build_recommendations(self.store, rec_pool, profile.values(), profile.excluded_crops, req.max_recommendations,
                                   crop_counts=rr.get("crop_counts"))
        for r in recs:
            r["recommendation_id"] = f"{qid}:{r['crop']}"
        if not recs:
            agent("Crop Recommendation Agent", "failed", t0, "No recommendable crops after exclusions")
            stage("recommendation", "failed", "No recommendable crops after exclusions", t0)
            return finish("no_evidence", conditions=profile, retrieval=retrieval_info, cases=cases)
        req_crop = profile.get("crop", True)
        if req_crop and not any(r["user_requested"] for r in recs):
            warnings.append(f"No historical records of {req_crop} match your constraints "
                            f"({', '.join(f'{k}={v}' for k, v in rr['filters'].items()) or 'none'}); "
                            f"{req_crop} is not recommended on evidence. Alternatives below are supported by data.")
        agent("Crop Recommendation Agent", "success", t0, f"{len(recs)} crops")
        stage("recommendation", "success", f"{len(recs)} crops ranked from evidence", t0)

        # 6. risk | yield | impact (parallel, isolated) ---------------------------------------
        t0 = time.time()
        stage("analysis", "running")
        cond = profile.values()
        by_id = {c.case_id: c for c in pool}
        for n in ("Suitability & Risk Agent", "Yield Prediction Agent", "Soil, Weather & Input Impact Agent"):
            agent(n, "running")
        t_a = {n: time.time() for n in agents}
        extra = {"area": req.area, "fertilizer": req.fertilizer, "pesticide": req.pesticide} if req.area else {}
        f_risk = {r["crop"]: self.pool.submit(self.risk.run, r["crop"], cond) for r in recs}
        f_yield = {r["crop"]: self.pool.submit(self.yield_agent.run, r["crop"], cond,
                                               [by_id[i].record for i in r["supporting_case_ids"] if i in by_id], extra) for r in recs}
        focus = profile.get("crop") or recs[0]["crop"]
        f_imp = self.pool.submit(self.impact.run, focus, profile.get("state", True), profile.get("season", True))
        failed = {}

        def collect(futs, key, label):
            ok = True
            for crop, f in futs.items():
                r = next(x for x in recs if x["crop"] == crop)
                try:
                    r[key] = f.result(timeout=AGENT_TIMEOUT_S)
                except (FutTimeout, Exception) as e:  # noqa: BLE001
                    ok = False
                    r[key] = {"available": False, "reason": f"{label} failed: {type(e).__name__}: {e}"}
                    failed[label] = str(e)
            return ok

        ok_r = collect(f_risk, "risk", "Suitability & Risk Agent")
        agent("Suitability & Risk Agent", "success" if ok_r else "failed", t_a["Suitability & Risk Agent"], failed.get("Suitability & Risk Agent"))
        ok_y = collect(f_yield, "yield_prediction", "Yield Prediction Agent")
        y_unavail = [r["yield_prediction"].get("reason") for r in recs if not r["yield_prediction"].get("available")]
        agent("Yield Prediction Agent", "success" if ok_y and not y_unavail else ("failed" if not ok_y else "skipped"),
              t_a["Yield Prediction Agent"], failed.get("Yield Prediction Agent") or (y_unavail[0] if y_unavail else None))
        impact = None
        try:
            impact = f_imp.result(timeout=AGENT_TIMEOUT_S)
            agent("Soil, Weather & Input Impact Agent", "success", t_a["Soil, Weather & Input Impact Agent"],
                  None if impact.get("available") else impact.get("reason"))
        except Exception as e:  # noqa: BLE001
            agent("Soil, Weather & Input Impact Agent", "failed", t_a["Soil, Weather & Input Impact Agent"], str(e))
            failed["Impact"] = str(e)
        if failed:
            warnings.append("Some analyses failed and were omitted: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
        if y_unavail and ok_y:
            warnings.append(f"Yield prediction unavailable: {y_unavail[0]}")
        stage("analysis", "failed" if len(failed) >= 3 else "success",
              ("Partial: " + ", ".join(failed)) if failed else "Risk, yield and impact analyses completed", t0)

        # explanation + 7. judge -------------------------------------------------------------
        facts = build_facts(recs, profile, cases)
        t0 = time.time()
        agent("Crop Recommendation Agent", "running")
        narrative = self.narrator.run(facts, req.use_llm)
        if narrative.get("llm_error"):
            warnings.append(f"LLM explanation unavailable ({narrative['llm_error']}); showing a template explanation.")
        agent("Crop Recommendation Agent", "success", t0, f"explanation source: {narrative.get('source')}")
        t0 = time.time()
        stage("judge", "running")
        agent("Recommendation Validation Agent", "running")
        try:
            judge = self.judge.run(recs, profile, narrative, facts, req.use_llm)
            st = judge.get("llm_judge_status", "")
            agent("Recommendation Validation Agent", "failed" if st.startswith("failed") else ("success" if st == "success" else "skipped"),
                  t0, st or None)
            stage("judge", "failed" if st.startswith("failed") else ("success" if st == "success" else "skipped"),
                  f"verdict: {judge['verdict']}" + (f" ({st})" if st != "success" else ""), t0)
            if st.startswith(("failed", "not_run")):
                warnings.append(f"LLM-as-judge did not run ({st}); only deterministic validation applies.")
            pr = {x["crop"]: x for x in (judge.get("llm_judge") or {}).get("per_recommendation", [])}
            for r in recs:
                r["judge"] = pr.get(r["crop"]) or {"verdict": "not_run", "notes": "No per-recommendation LLM verdict."}
        except Exception as e:  # noqa: BLE001
            log.exception("judge failed")
            judge = {"verdict": "not_run", "error": str(e)}
            agent("Recommendation Validation Agent", "failed", t0, str(e))
            stage("judge", "failed", str(e), t0)

        for r in recs:
            r["warnings"] = list(profile.unusual_values)
            r["missing_information"] = profile.missing_information
            r["confidence_note"] = ("Evidence-based heuristic ranking; supported by "
                                    f"{r['n_supporting_cases']} similar historical cases. It is not a probability of success.")
        status = "completed_with_warnings" if warnings else "completed"
        return finish(status, conditions=profile, retrieval=retrieval_info, cases=cases, recommendations=recs,
                      narrative=narrative, impact_analysis=impact, judge=judge)
