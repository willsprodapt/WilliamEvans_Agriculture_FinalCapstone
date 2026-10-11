"""Evaluation run: retrieval metrics, recommendation checks, DeepEval metrics, yield-model metrics. All persisted."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable

from ..agents.orchestrator import AdvisoryOrchestrator
from ..config import settings
from ..models.yield_model import ModelUnavailable, YieldPredictor
from ..rag.retriever import HybridRetriever, RetrievalError, get_retriever, index_status
from ..rag.store import get_store
from ..schemas.advisory import AdvisoryRequest
from ..services import db
from . import deepeval_runner, metrics, test_cases

log = logging.getLogger("terramind.eval")
K = 10


def _answer_text(res: dict) -> str:
    n = res.get("narrative") or {}
    parts = [n.get("summary", "")] + [f"{p['crop']}: {p['explanation']}" for p in n.get("per_crop", [])]
    return "\n".join(x for x in parts if x).strip()


def run(label: str | None = None, deepeval_limit: int = 10, use_llm: bool = True, run_id: str | None = None,
        progress: Callable[[int, int, str], None] | None = None, rebuild_cases: bool = False) -> dict:
    t0 = time.time()
    store = get_store()
    run_id = run_id or "eval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cases = test_cases.load(store, rebuild=rebuild_cases)
    problems = test_cases.verify(store, cases)
    retriever: HybridRetriever = get_retriever(store)
    try:
        predictor = YieldPredictor()
    except ModelUnavailable:
        predictor = None
    orch = AdvisoryOrchestrator(store, retriever, predictor)
    de_ok, de_why = deepeval_runner.availability()
    if not use_llm:
        de_ok, de_why = False, "LLM disabled for this run"

    results, n_de = [], 0
    for i, tc in enumerate(cases):
        if progress:
            progress(i, len(cases), f"{tc['id']}: {tc['query'][:60]}")
        rel = set(tc.get("relevant_case_ids", []))
        r: dict = {"test_case_id": tc["id"], "type": tc["type"], "input": tc["query"], "expected": tc["expected"],
                   "expected_relevant_case_ids": tc.get("relevant_case_ids", [])[:100], "n_expected_relevant": len(rel),
                   "reference_answer": tc["reference_answer"], "provenance": tc["provenance"], "failure_reasons": [],
                   "retrieved_case_ids": [], "generated_answer": None, "retrieval_metrics": None, "deepeval": None,
                   "recommendation_checks": None, "judge_verdict": None, "pipeline_status": None}
        pipeline = None
        try:
            if tc["type"] == "constraint_retrieval":
                resp = retriever.search(tc["query"], None, top_k=K)
                ids = [it.case_id for it in resp.items]
                r["retrieved_case_ids"] = ids
                r["retrieval_metrics"] = {"precision_at_k": metrics.precision_at_k(ids, rel, K), "recall_at_k": metrics.recall_at_k(ids, rel, K),
                                          "mrr": metrics.reciprocal_rank(ids, rel), "hit_rate_at_k": metrics.hit_at_k(ids, rel, K), "k": K,
                                          "recall_note": f"{len(rel)} relevant cases exist; max achievable Recall@{K} = {min(1, K / max(1, len(rel))):.2f}"}
                r["backends_used"] = resp.backends.get("used")
                rk = min(K, len(rel))
                rprec = metrics.precision_at_k(ids, rel, rk)
                r["retrieval_metrics"]["r_precision"] = rprec
                r["retrieval_metrics"]["r_precision_note"] = f"precision at k=min(K,|relevant|)={rk}; used for pass/fail because Precision@{K} is capped at {rk}/{K} when only {len(rel)} relevant cases exist"
                r["passed"] = (rprec or 0) >= 0.5
                if not r["passed"]:
                    r["failure_reasons"].append(f"R-precision@{rk} = {rprec:.2f} < 0.5")
            run_pipeline = (tc["type"] != "constraint_retrieval") or (n_de < deepeval_limit)
            if run_pipeline:
                req = AdvisoryRequest(query=tc["query"], top_k=K, use_llm=use_llm)
                pipeline = orch.run(req).model_dump()
                r["pipeline_status"] = pipeline["status"]
                r["generated_answer"] = _answer_text(pipeline)
                j = pipeline.get("judge") or {}
                r["judge_verdict"] = {"verdict": j.get("verdict"), "llm_status": j.get("llm_judge_status"),
                                      "deterministic_passed": (j.get("deterministic") or {}).get("passed"),
                                      "violations": (j.get("deterministic") or {}).get("constraint_violations", []),
                                      "unsupported_claims": (j.get("deterministic") or {}).get("unsupported_claims", [])}
                recs = pipeline["recommendations"]
                if tc["type"] != "constraint_retrieval":
                    r["retrieved_case_ids"] = [c["case_id"] for c in pipeline["cases"]]
        except RetrievalError as e:
            r["passed"] = False
            r["failure_reasons"].append(f"Retrieval error: {e}")
            results.append(r)
            continue
        except Exception as e:  # noqa: BLE001
            log.exception("test case failed")
            r["passed"] = False
            r["failure_reasons"].append(f"{type(e).__name__}: {e}")
            results.append(r)
            continue

        if tc["type"] == "conditions_recommendation" and pipeline:
            ids = r["retrieved_case_ids"]
            r["retrieval_metrics"] = {"precision_at_k": metrics.precision_at_k(ids, rel, K), "recall_at_k": None,
                                      "mrr": metrics.reciprocal_rank(ids, rel), "hit_rate_at_k": metrics.hit_at_k(ids, rel, K), "k": K,
                                      "recall_note": "Recall not computed: relevance = any record in state+season (hundreds of relevant cases)"}
            allowed = set(tc["expected"]["allowed_crops"])
            rec_crops = [x["crop"] for x in pipeline["recommendations"]]
            outside = [c for c in rec_crops if c not in allowed]
            checks = {"n_recommendations": len(rec_crops), "all_recommended_crops_recorded_in_state_season": not outside,
                      "recommended_crops": rec_crops,
                      "has_supporting_case_ids": all(x["supporting_case_ids"] for x in pipeline["recommendations"]) and bool(rec_crops),
                      "missing_info_disclosed": bool(pipeline["recommendations"]) and all(x.get("missing_information") is not None for x in pipeline["recommendations"]),
                      "unsupported_numeric_claims": len((r["judge_verdict"] or {}).get("unsupported_claims", [])),
                      "judge_deterministic_passed": (r["judge_verdict"] or {}).get("deterministic_passed")}
            r["recommendation_checks"] = checks
            r["passed"] = bool(rec_crops) and not outside and checks["has_supporting_case_ids"] and (checks["judge_deterministic_passed"] is not False)
            if not rec_crops:
                r["failure_reasons"].append("No recommendations produced")
            if outside:
                r["failure_reasons"].append(f"Recommended crops with no record in {tc['expected']['state']}/{tc['expected']['season']}: {outside}")
            if checks["judge_deterministic_passed"] is False:
                r["failure_reasons"] += (r["judge_verdict"]["violations"] + r["judge_verdict"]["unsupported_claims"])
        elif tc["type"] == "abstention" and pipeline:
            c = tc["expected"]["crop"]
            recommended = [x["crop"] for x in pipeline["recommendations"]]
            says_no_evidence = any(c.lower() in w.lower() and "no historical" in w.lower() for w in pipeline["warnings"])
            r["recommendation_checks"] = {"requested_crop": c, "recommended_crops": recommended,
                                          "requested_crop_not_recommended": c not in recommended, "disclosed_no_evidence": says_no_evidence}
            r["passed"] = c not in recommended and says_no_evidence
            if c in recommended:
                r["failure_reasons"].append(f"Recommended {c} despite 0 historical records in the state")
            if not says_no_evidence:
                r["failure_reasons"].append("Did not disclose lack of evidence for the requested crop")
        elif tc["type"] == "clarification" and pipeline:
            r["recommendation_checks"] = {"status": pipeline["status"]}
            r["passed"] = pipeline["status"] == "needs_clarification"
            if not r["passed"]:
                r["failure_reasons"].append(f"Expected needs_clarification, got {pipeline['status']}")
        r.setdefault("passed", False)

        # DeepEval on pipeline cases that produced a narrative and have retrieved evidence
        if pipeline and r["generated_answer"] and tc["type"] in ("constraint_retrieval", "conditions_recommendation") and pipeline["cases"]:
            ctx = [(store.get(c["case_id"]) or {}).get("text", "") for c in pipeline["cases"][:K]]
            if de_ok:
                r["deepeval"] = deepeval_runner.evaluate_case(tc["query"], r["generated_answer"], tc["reference_answer"], ctx)
                n_de += 1
            else:
                r["deepeval"] = {m: {"available": False, "score": None, "passed": None, "reason": de_why} for m in deepeval_runner.METRICS}
        results.append(r)
        # persist progress so a crash keeps partial evidence
    if progress:
        progress(len(cases), len(cases), "aggregating")

    # ---- aggregates (only from values that exist)
    def col(f):
        return [x["retrieval_metrics"][f] for x in results if x.get("retrieval_metrics") and x["retrieval_metrics"].get(f) is not None]
    ret_cases = [x for x in results if x["type"] == "constraint_retrieval"]
    de = {}
    for m in deepeval_runner.METRICS:
        vals = [x["deepeval"][m]["score"] for x in results if x.get("deepeval") and x["deepeval"][m]["available"]]
        unav = [x["deepeval"][m]["reason"] for x in results if x.get("deepeval") and not x["deepeval"][m]["available"]]
        de[m] = {"mean": metrics.mean(vals), "n_measured": len(vals), "n_unavailable": len(unav),
                 "unavailable_reason": unav[0] if unav else None,
                 "pass_rate": (sum(1 for x in results if x.get("deepeval") and x["deepeval"][m]["available"] and x["deepeval"][m]["passed"]) / len(vals)) if vals else None}
    by_type = {}
    for x in results:
        d = by_type.setdefault(x["type"], {"total": 0, "passed": 0})
        d["total"] += 1
        d["passed"] += 1 if x.get("passed") else 0
    verdicts = {}
    for x in results:
        v = (x.get("judge_verdict") or {}).get("verdict")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1
    ym = settings.model_dir / "yield_metrics.json"
    yield_eval = None
    if ym.exists():
        y = json.loads(ym.read_text())
        yield_eval = {"model_version": y["meta"]["model_version"], "split": y["split"],
                      "selected_test_metrics": {fs: y["comparison"][fs]["selected"]["test"] for fs in y["comparison"]},
                      "baseline_test_metrics": {fs: y["comparison"][fs]["baseline_crop_median"]["test"] for fs in y["comparison"]},
                      "selected_models": {fs: y["comparison"][fs]["selected"]["name"] for fs in y["comparison"]}}
    summary = {
        "n_test_cases": len(results), "n_passed": sum(1 for x in results if x.get("passed")),
        "n_failed": sum(1 for x in results if not x.get("passed")),
        "by_type": by_type,
        "retrieval": {"k": K, "n_cases": len(ret_cases), "precision_at_k": metrics.mean([x["retrieval_metrics"]["precision_at_k"] for x in ret_cases if x.get("retrieval_metrics")]),
                      "recall_at_k": metrics.mean([x["retrieval_metrics"]["recall_at_k"] for x in ret_cases if x.get("retrieval_metrics")]),
                      "mrr": metrics.mean([x["retrieval_metrics"]["mrr"] for x in ret_cases if x.get("retrieval_metrics")]),
                      "hit_rate_at_k": metrics.mean([x["retrieval_metrics"]["hit_rate_at_k"] for x in ret_cases if x.get("retrieval_metrics")]),
                      "recommendation_case_precision_at_k": metrics.mean([x["retrieval_metrics"]["precision_at_k"] for x in results
                                                                          if x["type"] == "conditions_recommendation" and x.get("retrieval_metrics")])},
        "deepeval": de,
        "recommendation_validation": {"judge_verdicts": verdicts,
                                      "unsupported_numeric_claims_total": sum((x.get("recommendation_checks") or {}).get("unsupported_numeric_claims", 0) for x in results),
                                      "constraint_satisfaction_rate": metrics.mean([1.0 if (x.get("recommendation_checks") or {}).get("all_recommended_crops_recorded_in_state_season") else 0.0
                                                                                    for x in results if x["type"] == "conditions_recommendation"]),
                                      "abstention_pass_rate": (by_type.get("abstention", {}).get("passed", 0) / by_type["abstention"]["total"]) if "abstention" in by_type else None,
                                      "clarification_pass_rate": (by_type.get("clarification", {}).get("passed", 0) / by_type["clarification"]["total"]) if "clarification" in by_type else None},
        "yield_model": yield_eval,
        "deepeval_status": "measured" if de_ok else f"unavailable: {de_why}",
    }
    ist = index_status(store)
    out = {
        "run_id": run_id, "label": label, "created_at": db.now(), "duration_s": round(time.time() - t0, 1), "status": "completed",
        "versions": {"dataset_version": store.dataset_version, "n_records": len(store),
                     "yield_model_version": predictor.meta["model_version"] if predictor else None,
                     "llm_model": settings.llm_model if use_llm and settings.llm_configured else None,
                     "embedding_model": ist.get("embedding_model"), "retrieval": "hybrid RRF k=60 (BM25 + MiniLM FAISS)",
                     "top_k": K, "deepeval_limit": deepeval_limit},
        "label_verification": {"problems": problems, "n_cases": len(cases), "verified": not problems},
        "summary": summary, "results": results,
    }
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    (settings.eval_dir / f"{run_id}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (settings.eval_dir / f"{run_id}.md").write_text(markdown_report(out), encoding="utf-8")
    db.save_eval_run(run_id, "completed", label, out)
    return out


def _f(v, d=3):
    return "Not evaluated" if v is None else f"{v:.{d}f}"


def markdown_report(o: dict) -> str:
    s = o["summary"]
    L = [f"# TerraMind AI evaluation report - {o['run_id']}", "", f"- Created: {o['created_at']}  (duration {o['duration_s']} s)",
         f"- Dataset version: `{o['versions']['dataset_version']}` ({o['versions']['n_records']} records)",
         f"- Yield model: `{o['versions']['yield_model_version']}`; LLM: `{o['versions']['llm_model']}`; embeddings: `{o['versions']['embedding_model']}`",
         f"- Test cases: {s['n_test_cases']} - passed {s['n_passed']}, failed {s['n_failed']}; labels verified: {o['label_verification']['verified']}", "",
         "## Retrieval metrics (constraint-retrieval cases, verified relevance labels)", "",
         f"| Metric | Value |\n|---|---|\n| Precision@{s['retrieval']['k']} | {_f(s['retrieval']['precision_at_k'])} |\n| Recall@{s['retrieval']['k']} | {_f(s['retrieval']['recall_at_k'])} |\n"
         f"| MRR | {_f(s['retrieval']['mrr'])} |\n| Hit Rate@{s['retrieval']['k']} | {_f(s['retrieval']['hit_rate_at_k'])} |", "",
         "## DeepEval (Claude judge)", "", f"Status: {s['deepeval_status']}", "", "| Metric | Mean | Measured | Unavailable |\n|---|---|---|---|"]
    for m, d in s["deepeval"].items():
        L.append(f"| {m} | {_f(d['mean'])} | {d['n_measured']} | {d['n_unavailable']} |")
    rv = s["recommendation_validation"]
    L += ["", "## Recommendation validation", "", f"- Judge verdicts: {rv['judge_verdicts'] or 'Not evaluated'}",
          f"- Constraint satisfaction rate: {_f(rv['constraint_satisfaction_rate'])}", f"- Unsupported numeric claims: {rv['unsupported_numeric_claims_total']}",
          f"- Abstention pass rate: {_f(rv['abstention_pass_rate'])}; clarification pass rate: {_f(rv['clarification_pass_rate'])}", ""]
    y = s.get("yield_model")
    L += ["## Yield model (held-out test years, separate from LLM metrics)", ""]
    if y:
        L += ["| Feature set | Model | MAE | RMSE | R2 | R2 (log1p) |\n|---|---|---|---|---|---|"]
        for fs, t in y["selected_test_metrics"].items():
            L.append(f"| {fs} | {y['selected_models'][fs]} | {t['mae']:.2f} | {t['rmse']:.2f} | {t['r2']:.3f} | {t['r2_log1p']:.3f} |")
    else:
        L.append("Not evaluated")
    L += ["", "## Failures", ""]
    fails = [r for r in o["results"] if not r.get("passed")]
    L += [f"- **{r['test_case_id']}** ({r['type']}): {'; '.join(r['failure_reasons']) or 'unspecified'}" for r in fails] or ["None"]
    return "\n".join(L) + "\n"
