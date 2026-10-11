"""Specialist agents. Each has a narrow responsibility and returns plain structured data; the orchestrator
(orchestrator.py) owns sequencing, timeouts, status reporting and failure isolation."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..config import settings
from ..models.yield_model import ModelUnavailable, YieldPredictor
from ..rag.prioritizer import prioritize
from ..rag.retriever import HybridRetriever
from ..rag.store import CaseStore
from ..schemas.advisory import JudgeVerdict, Narrative
from ..schemas.common import CaseFilters, ConditionProfile, RetrievalItem
from ..services import impact as impact_svc
from ..services import suitability
from ..services.llm import LLMError, llm, wrap_untrusted


# ----------------------------------------------------------------------------- Agent 2
def profile_to_query_text(p: ConditionProfile, store: CaseStore) -> str:
    """Query-side description phrased like case text so lexical + semantic matching can use soil/weather classes."""
    from ..services.data_prep import PH_BINS, RAIN_BINS, TEMP_BINS, HUM_BINS, _band
    v = p.values()
    bits = []
    if v.get("crop"):
        bits.append(str(v["crop"]))
    if v.get("state"):
        bits.append(f"in {v['state']}")
    if v.get("season"):
        bits.append(f"{v['season']} season")
    if "ph" in v:
        bits.append(f"soil pH {v['ph']:g} {_band(v['ph'], PH_BINS)} soil")
    if "rainfall" in v:
        bits.append(f"rainfall {v['rainfall']:g} mm {_band(v['rainfall'], RAIN_BINS)}")
    if "temperature" in v:
        bits.append(f"temperature {v['temperature']:g} C {_band(v['temperature'], TEMP_BINS)}")
    if "humidity" in v:
        bits.append(f"humidity {v['humidity']:g} percent {_band(v['humidity'], HUM_BINS)}")
    for k, lab in (("n", "nitrogen N"), ("p", "phosphorus P"), ("k", "potassium K")):
        if k in v:
            bits.append(f"{lab} {v[k]:g}")
    return " ".join(bits)


class CaseRetrievalAgent:
    name = "Historical Case Retrieval Agent"

    def __init__(self, store: CaseStore, retriever: HybridRetriever):
        self.store, self.retriever = store, retriever

    def _filters(self, p: ConditionProfile) -> CaseFilters | None:
        d = {}
        if not self.include_synthetic:
            d["origin"] = ["real"]
        if self.year_max is not None:
            d["year_max"] = int(self.year_max)
        if p.get("state", True):
            d["state"] = [p.get("state", True)]
        if p.get("season", True):
            d["season"] = [p.get("season", True)]
        return CaseFilters(**d) if d else None

    def _structured_candidates(self, p: ConditionProfile, mask: np.ndarray, per_crop: int = 4) -> list[int]:
        """Numeric condition-similarity candidates (embeddings are weak on numbers). Best `per_crop` cases per crop."""
        v = {k: p.get(k) for k in ("n", "p", "k", "ph", "rainfall", "temperature", "humidity") if p.get(k) is not None}
        if not v:
            return []
        df = self.store.df
        sims = []
        for f, val in v.items():
            sims.append(np.clip(1 - np.abs(df[f].to_numpy() - float(val)) / self.store.scale[f], 0, 1))
        s = np.nanmean(np.vstack(sims), axis=0)
        s = np.where(mask & ~np.isnan(s), s, -1)
        tmp = pd.DataFrame({"i": np.arange(len(df)), "s": s, "crop": df["crop"].values, "year": df["year"].values})
        tmp = tmp[tmp["s"] >= 0].sort_values(["s", "year"], ascending=False)
        return tmp.groupby("crop").head(per_crop)["i"].tolist()

    include_synthetic = True
    year_max: int | None = None     # evidence cut-off (used to evaluate recommendations against later, held-out years)

    def run(self, query: str, p: ConditionProfile, top_k: int = 15, include_synthetic: bool = True, year_max: int | None = None) -> dict:
        self.include_synthetic = include_synthetic
        self.year_max = year_max
        filters = self._filters(p)
        excluded = {e.lower() for e in p.excluded_crops}
        q_clean = query
        for e in p.excluded_crops:   # a crop the user rejects must not steer retrieval toward itself ("not Gram" would retrieve Gram)
            for name in {e, e.split("(")[0].strip()}:
                q_clean = re.sub(rf"(?i)(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", " ", q_clean)
        qtext = (q_clean + " " + profile_to_query_text(p, self.store)).strip()
        resp = self.retriever.search(qtext, filters, top_k=60, candidate_k=150)
        items: dict[str, RetrievalItem] = {i.case_id: i for i in resp.items}
        notes = []
        crop = p.get("crop", True)
        if crop:
            cf = CaseFilters(crop=[crop], **({"origin": ["real"]} if not include_synthetic else {}), **({"year_max": year_max} if year_max is not None else {}), **({"state": filters.state} if filters and filters.state else {}),
                             **({"season": filters.season} if filters and filters.season else {}))
            for i in self.retriever.search(qtext, cf, top_k=15).items:
                items.setdefault(i.case_id, i)
        mask = self.store.mask(filters)
        for idx in self._structured_candidates(p, mask):
            rec = self.store.record(idx)
            if rec["case_id"] not in items:
                items[rec["case_id"]] = RetrievalItem(
                    case_id=rec["case_id"], hybrid_rank=0, hybrid_score=0.0, record=rec,
                    relevance_explanation="Added by structured condition match (soil/weather closeness); no keyword/semantic rank.")
        for cid in [cid for cid, it in items.items() if it.record["crop"].lower() in excluded]:
            del items[cid]
        pool = prioritize(self.store, list(items.values()), p.values())
        # Constraints are never silently relaxed: with thin evidence the (few) crops recorded for the requested state+season are returned
        # as they are, and the recommendation flags the low support.
        crop_counts = self.store.df.loc[mask, "crop"].value_counts().to_dict()
        diag = {}
        if not pool:
            for name, val in (("state", p.get("state", True)), ("season", p.get("season", True))):
                if val:
                    f = CaseFilters(**{name: [val]})
                    diag[f"{name}={val}"] = int(self.store.mask(f).sum())
        return {"pool": pool, "shown": pool[:top_k], "response": resp, "notes": notes, "crop_counts": crop_counts, "filter_diagnostics": diag,
                "filters": filters.active() if filters else {}}


# ----------------------------------------------------------------------------- Agent 4
class SuitabilityRiskAgent:
    name = "Suitability & Risk Agent"

    def __init__(self, store: CaseStore):
        self.store = store

    def run(self, crop: str, cond: dict) -> dict:
        return suitability.assess(self.store, crop, cond)


# ----------------------------------------------------------------------------- Agent 5
class YieldPredictionAgent:
    name = "Yield Prediction Agent"

    def __init__(self, store: CaseStore, predictor: YieldPredictor | None):
        self.store, self.predictor = store, predictor

    def run(self, crop: str, cond: dict, supporting: list[dict], extra: dict | None = None) -> dict:
        if self.predictor is None:
            return {"available": False, "reason": "Yield model artifact not found (run scripts/train_yield_model.py)."}
        df = self.store.df
        assumptions = []
        state = cond.get("state")
        season = cond.get("season")
        if not state and supporting:
            state = pd.Series([s["state"] for s in supporting]).mode().iat[0]
            assumptions.append(f"state not given: assumed '{state}' (most common among supporting cases)")
        if not season and supporting:
            season = pd.Series([s["season"] for s in supporting]).mode().iat[0]
            assumptions.append(f"season not given: assumed '{season}' (most common among supporting cases)")
        if not state or not season:
            return {"available": False, "reason": "State and season could not be determined."}
        inp = {"crop": crop, "state": state, "season": season, **{k: cond.get(k) for k in ("n", "p", "k", "ph", "rainfall", "temperature", "humidity")}}
        inp.update(extra or {})
        st = df[df["state"] == state]
        for f in ("n", "p", "k", "ph"):
            if inp.get(f) is None and not st.empty:
                inp[f] = float(st[f].iloc[0])
                assumptions.append(f"{f.upper() if f != 'ph' else 'pH'} not given: used {state}'s state-level value {inp[f]:g}")
        recent = st[st["year"] >= st["year"].max() - 4] if not st.empty else st
        for f in ("rainfall", "temperature", "humidity"):
            if inp.get(f) is None and not recent.empty and recent[f].notna().any():
                inp[f] = float(recent[f].median())
                assumptions.append(f"{f} not given: used {state}'s median over its last 5 recorded years ({inp[f]:.1f})")
        try:
            out = self.predictor.predict(inp)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"Prediction failed: {e}"}
        out["available"] = True
        out["assumptions"] = assumptions + out.get("assumptions", [])
        out["label"] = "MODEL ESTIMATE - not an observed historical yield"
        return out


# ----------------------------------------------------------------------------- Agent 6
class ImpactAnalysisAgent:
    name = "Soil, Weather & Input Impact Agent"

    def __init__(self, store: CaseStore):
        self.store = store

    def run(self, crop: str | None, state: str | None, season: str | None) -> dict:
        res = impact_svc.analyze(self.store, crop=crop, state=state, season=season)
        if not res.get("available") and state:   # too narrow: widen to crop-only, flagged
            wide = impact_svc.analyze(self.store, crop=crop)
            wide["note"] = f"Scope widened to crop only because the narrower selection had too few records ({res.get('reason')})"
            return wide
        return res


# ----------------------------------------------------------------------------- facts, narrative, judge
def build_facts(recs: list[dict], profile: ConditionProfile, cases: list[dict]) -> dict:
    return {
        "user_conditions": profile.values(),
        "excluded_crops": profile.excluded_crops,
        "recommendations": [{
            "rank": r["rank"], "crop": r["crop"], "heuristic_score": r["score"], "supporting_case_ids": r["supporting_case_ids"],
            "n_supporting_cases": r["n_supporting_cases"],
            "evidence_level": "limited" if r["n_supporting_cases"] < 3 else "moderate" if r["n_supporting_cases"] < 8 else "strong",
            "historical_yield_actual": {k: r["historical_yield"][k] for k in ("median", "min", "max", "n", "years")},
            "predicted_yield_model_estimate": (None if not (r.get("yield_prediction") or {}).get("available") else {
                "value": r["yield_prediction"]["predicted_yield"], "interval_10_90": r["yield_prediction"]["interval_10_90"]["low"],
                "interval_high": r["yield_prediction"]["interval_10_90"]["high"]}),
            "risk": {"category": (r.get("risk") or {}).get("category"), "score": (r.get("risk") or {}).get("score"),
                     "risk_factors": (r.get("risk") or {}).get("risk_factors", [])[:4],
                     "missing_conditions": (r.get("risk") or {}).get("missing_conditions", [])},
        } for r in recs],
        "retrieved_cases": [{"case_id": c["case_id"], "crop": c["record"]["crop"], "state": c["record"]["state"],
                             "season": c["record"]["season"], "year": c["record"]["year"], "yield": c["record"]["yield"]} for c in cases[:12]],
    }


def _numbers_in(obj) -> list[float]:
    out = []
    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            out.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for m in re.findall(r"\d+(?:\.\d+)?", o):
                out.append(float(m))
    walk(obj)
    return out


def numeric_claims_check(text: str, facts: dict) -> list[str]:
    """Flag numbers in generated text that do not appear (within 1.5%) in the supplied facts."""
    allowed = _numbers_in(facts)
    bad = []
    stripped = re.sub(r"TM-[0-9A-F]{10}", " ", text)
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", stripped):
        v = float(m.group(1))
        if v <= 12 and float(v).is_integer():       # ranks, counts, 'top 3'
            continue
        if any(abs(v - a) <= max(0.015 * abs(a), 0.006) for a in allowed):
            continue
        if any(abs(round(v, 1) - round(a, 1)) < 0.051 for a in allowed):
            continue
        bad.append(m.group(1))
    return sorted(set(bad))


SAFETY_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?(?:kg|g|l|ml|litres?|liters?|tonnes?|lbs?)\s?(?:/|per)\s?(?:ha|hectare|acre)\b",
    r"\b(?:spray|apply|mix|dilute|dose|treat)\b[^.]{0,50}\b(?:pesticides?|insecticides?|fungicides?|herbicides?|fertili[sz]ers?|urea|dap|npk|chemicals?)\b",
    r"\bapply\s+\d",
]
HEDGE_WORDS = ("limited", "few", "only", "small", "caution", "uncertain", "weak")


class RecommendationNarrator:
    """LLM explanation grounded in deterministic facts; falls back to a template when the LLM is unavailable."""
    name = "Crop Recommendation Agent (explanation)"

    def run(self, facts: dict, use_llm: bool = True) -> dict:
        if use_llm and llm.configured:
            try:
                n = llm.complete_json(
                    "You are an agricultural decision-support writer. Explain the crop recommendations below to a farmer in plain "
                    "language. Use ONLY the facts in the supplied JSON; quote numbers exactly as given; cite supporting case ids "
                    "(TM-XXXXXXXXXX) that appear in the JSON. Keep historical (actual) yield and predicted (model estimate) yield "
                    "clearly separate. Never promise yields. Mention missing information and risks. Where a crop's evidence_level is 'limited', say "
                    "the evidence is limited. Do NOT give fertilizer, pesticide, irrigation or other treatment instructions or doses; those are "
                    "not in the evidence. Do not recommend a crop that is not in the JSON. summary <= 120 words; "
                    "each per_crop explanation <= 70 words.",
                    wrap_untrusted("facts_json", json.dumps(facts, default=str)), Narrative, max_tokens=2500, effort="low")
                return {**n.model_dump(), "source": "llm", "model": settings.llm_model}
            except LLMError as e:
                fb = self.template(facts)
                fb["llm_error"] = f"{e.kind}: {e}"
                return fb
        fb = self.template(facts)
        if use_llm:
            fb["llm_error"] = "not_configured: ANTHROPIC_API_KEY missing"
        return fb

    @staticmethod
    def template(facts: dict) -> dict:
        recs = facts["recommendations"]
        if not recs:
            return {"summary": "No recommendation could be produced from the retrieved evidence.", "per_crop": [], "caveats": [], "source": "template"}
        top = ", ".join(r["crop"] for r in recs[:3])
        per = []
        for r in recs:
            h = r["historical_yield_actual"]
            t = f"{r['n_supporting_cases']} similar historical cases support {r['crop']} (e.g. {', '.join(r['supporting_case_ids'][:2])})."
            if h.get("median") is not None:
                t += f" Median historical yield among them: {h['median']:.2f} (source units)."
            if r["risk"]["category"]:
                t += f" Rule-based risk: {r['risk']['category']}."
            if r["n_supporting_cases"] < 3:
                t += f" Evidence is limited (only {r['n_supporting_cases']} similar case{'s' if r['n_supporting_cases'] != 1 else ''}), so treat this suggestion with caution."
            per.append({"crop": r["crop"], "explanation": t})
        return {"summary": f"Based on similar historical cases, the best-supported crops are {top}. "
                           "Ranking is a heuristic over case similarity, not a guarantee of yield.",
                "per_crop": per, "caveats": ["Template explanation (no LLM)."], "source": "template"}


class ValidationJudge:
    """Agent 7 - deterministic checks (always) + Claude LLM-as-judge (when configured). A judge verdict is an extra
    quality check, never ground truth, and never overrides the deterministic findings."""
    name = "Recommendation Validation Agent"

    def __init__(self, store: CaseStore):
        self.store = store

    def deterministic(self, recs: list[dict], profile: ConditionProfile, narrative: dict, facts: dict) -> dict:
        violations, unsupported = [], []
        excl = {c.lower() for c in profile.excluded_crops}
        state, season = profile.get("state", True), profile.get("season", True)
        for r in recs:
            ids = r["supporting_case_ids"]
            if not ids:
                violations.append(f"{r['crop']}: no supporting case ids")
            for cid in ids:
                rec = self.store.get(cid, include_text=False)
                if rec is None:
                    violations.append(f"{r['crop']}: supporting case {cid} does not exist")
                elif rec["crop"] != r["crop"]:
                    violations.append(f"{r['crop']}: supporting case {cid} is a {rec['crop']} record")
            if r["crop"].lower() in excl:
                violations.append(f"{r['crop']} was excluded by the user")
            sc = [self.store.get(c, False) for c in ids]
            if state and sc and not any(x and x["state"].lower() == state.lower() for x in sc):
                violations.append(f"{r['crop']}: no supporting case from requested state {state}")
            if season and sc and not any(x and x["season"].lower() == season.lower() for x in sc):
                violations.append(f"{r['crop']}: no supporting case from requested season {season}")
        text = " ".join([narrative.get("summary", "")] + [p["explanation"] for p in narrative.get("per_crop", [])])
        safety = [m.group(0) for pat in SAFETY_PATTERNS for m in [re.search(pat, text, flags=re.I)] if m]
        if safety:
            violations.append("Explanation contains treatment/dose instructions that are not supported by the evidence: " + "; ".join(safety[:3]))
        per = {p["crop"]: p["explanation"].lower() for p in narrative.get("per_crop", [])}
        uncertainty_flags = [r["crop"] for r in recs if r["n_supporting_cases"] < 3
                             and not any(w in per.get(r["crop"], "") + " " + narrative.get("summary", "").lower() for w in HEDGE_WORDS)]
        bad_nums = numeric_claims_check(text, facts)
        if bad_nums:
            unsupported.append("Numbers in the explanation not found in the evidence: " + ", ".join(bad_nums))
        ids_in_text = set(re.findall(r"TM-[0-9A-F]{10}", text))
        known = {c["case_id"] for c in facts["retrieved_cases"]} | {i for r in facts["recommendations"] for i in r["supporting_case_ids"]}
        ghost = ids_in_text - known
        if ghost:
            unsupported.append("Explanation cites case ids that were not retrieved: " + ", ".join(sorted(ghost)))
        return {"constraint_violations": violations, "unsupported_claims": unsupported,
                "checks_run": ["supporting case ids exist and match crop", "user exclusions respected",
                               "state/season constraints evidenced", "numeric claims traceable to evidence",
                               "no fabricated case ids", "no treatment/dose instructions", "limited evidence is flagged as uncertain"],
                "safety_flags": safety, "uncertainty_flags": uncertainty_flags,
                "passed": not violations and not unsupported}

    def run(self, recs: list[dict], profile: ConditionProfile, narrative: dict, facts: dict, use_llm: bool = True) -> dict:
        det = self.deterministic(recs, profile, narrative, facts)
        out = {"deterministic": det, "llm_judge": None, "judge_note": "A judge verdict is an additional quality check, not ground truth."}
        if not (use_llm and llm.configured):
            out["verdict"] = "not_run" if det["passed"] else "unsupported"
            out["llm_judge_status"] = "not_run: ANTHROPIC_API_KEY not configured" if use_llm else "not_run: disabled"
            if not det["passed"]:
                out["verdict"] = "partially_supported"
            return out
        try:
            retrieved_ids_text = json.dumps({"facts": facts, "narrative": narrative}, default=str)
            j = llm.complete_json(
                "You are a strict validator of crop recommendations (LLM-as-judge). Judge ONLY against the evidence JSON: "
                "evidence_support, relevance to the user's conditions, crop/season/state constraint satisfaction, consistency with "
                "retrieved cases, numeric claim consistency, unsupported assumptions, hallucinations, risk disclosure, missing "
                "information, and whether historical (actual) and predicted (model estimate) yield are kept distinct. Do not invent "
                "evidence. 'supported' only if every recommendation is backed by its supporting cases and nothing is overclaimed.",
                wrap_untrusted("evidence_and_narrative", retrieved_ids_text), JudgeVerdict, max_tokens=2500, effort="medium")
            out["llm_judge"] = j.model_dump()
            v = j.verdict
            if not det["passed"] and v == "supported":
                v = "partially_supported"      # LLM may not override deterministic findings
            out["verdict"] = v
            out["llm_judge_status"] = "success"
        except LLMError as e:
            out["verdict"] = "partially_supported" if not det["passed"] else "not_run"
            out["llm_judge_status"] = f"failed: {e.kind}: {e}"
        return out
