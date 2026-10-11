"""Agent 1 - Agricultural Condition Classification Agent.

Deterministic parsing (regex + dataset vocabulary) first; Claude (structured output) is used only to fill what the
parser cannot (colloquial seasons/crops, objectives, constraints). Every value carries provenance and is validated.
LLM-inferred categorical values are checked against the dataset vocabulary and are NEVER used as hard filters.
"""
from __future__ import annotations

import re

from ..rag.store import CaseStore
from ..schemas.advisory import AdvisoryRequest, LLMExtraction
from ..schemas.common import RANGE_FIELDS, ConditionField, ConditionProfile
from ..services.llm import LLMError, llm, wrap_untrusted
from ..utils.guardrails import unusual_values

CROP_ALIASES = {
    "paddy": "Rice", "rice": "Rice", "corn": "Maize", "maize": "Maize", "cotton": "Cotton(lint)", "soybean": "Soyabean",
    "soya": "Soyabean", "soybeans": "Soyabean", "peanut": "Groundnut", "peanuts": "Groundnut", "groundnut": "Groundnut",
    "mustard": "Rapeseed &Mustard", "rapeseed": "Rapeseed &Mustard", "pigeon pea": "Arhar/Tur", "pigeonpea": "Arhar/Tur",
    "arhar": "Arhar/Tur", "tur": "Arhar/Tur", "red gram": "Arhar/Tur", "green gram": "Moong(Green Gram)", "moong": "Moong(Green Gram)",
    "mung": "Moong(Green Gram)", "black gram": "Urad", "urad": "Urad", "chickpea": "Gram", "chana": "Gram", "bengal gram": "Gram",
    "lentil": "Masoor", "masoor": "Masoor", "chilli": "Dry chillies", "chillies": "Dry chillies", "chili": "Dry chillies",
    "sesame": "Sesamum", "til": "Sesamum", "pearl millet": "Bajra", "sorghum": "Jowar", "finger millet": "Ragi",
    "cassava": "Tapioca", "areca": "Arecanut", "cowpea": "Cowpea(Lobia)", "lobia": "Cowpea(Lobia)", "castor": "Castor seed",
    "cashew": "Cashewnut", "peas": "Peas & beans (Pulses)", "beans": "Peas & beans (Pulses)",
}
STATE_ALIASES = {"orissa": "Odisha", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand", "j&k": "Jammu and Kashmir",
                 "jammu & kashmir": "Jammu and Kashmir", "tamilnadu": "Tamil Nadu", "up": None}
SEASON_ALIASES = {"kharif": "Kharif", "monsoon": "Kharif", "rabi": "Rabi", "winter": "Winter", "summer": "Summer",
                  "autumn": "Autumn", "whole year": "Whole Year", "year-round": "Whole Year", "year round": "Whole Year",
                  "perennial": "Whole Year"}
_NEG = r"(?:avoid|exclude|excluding|without|except|not|no|other than|instead of|alternative to|alternatives to)"
NUM = r"(\d+(?:\.\d+)?)"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9& ]+", " ", s.lower()).strip()


class ConditionClassifier:
    def __init__(self, store: CaseStore):
        self.store = store
        self.crops = sorted(store.df["crop"].unique(), key=len, reverse=True)
        self.states = sorted(store.df["state"].unique(), key=len, reverse=True)
        self.seasons = sorted(store.df["season"].unique())
        self.crop_lookup = {_norm(c): c for c in self.crops}

    # ------------------------------------------------------------------ deterministic parse
    def _find_crops(self, q: str) -> list[tuple[str, int]]:
        """Return (crop, start_index) for every crop mention; longest names / aliases first, spans consumed."""
        low = " " + _norm(q) + " "
        found, taken = [], []
        cands = [(k, v) for k, v in self.crop_lookup.items()] + list(CROP_ALIASES.items())
        for key, canon in sorted(cands, key=lambda kv: -len(kv[0])):
            if canon not in self.crops:
                continue
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(_norm(key))}s?(?![a-z0-9])", low):
                if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((canon, m.start()))
        return sorted(found, key=lambda x: x[1])

    def _parse_text(self, q: str) -> dict:
        out: dict = {"numeric": {}, "excluded": []}
        ql = q.lower()
        # state
        for s in self.states:
            if re.search(rf"(?<![a-z]){re.escape(s.lower())}(?![a-z])", ql):
                out["state"] = s
                break
        else:
            for a, s in STATE_ALIASES.items():
                if s and re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", ql):
                    out["state"] = s
                    break
        # season
        for a, s in SEASON_ALIASES.items():
            if re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", ql):
                if s in self.seasons:
                    out["season"] = s
                    out["season_alias"] = a if a != s.lower() else None
                break
        # crops: negated vs positive
        mentions = self._find_crops(q)
        low = _norm(q)
        pos, neg = [], []
        for crop, _ in mentions:
            idx = low.find(_norm(crop)) if _norm(crop) in low else -1
            pre = low[max(0, idx - 28): idx] if idx >= 0 else ""
            (neg if re.search(rf"\b{_NEG}\b[^.]*$", pre) and "alternative" not in pre else pos).append(crop)
        # "alternative to X": X is the current crop, recommend others -> treat as excluded from recommendations
        if re.search(r"alternatives? to|instead of|replace|switch from|other than", ql):
            for c in list(pos):
                neg.append(c)
            pos = []
        out["crop"] = pos[0] if pos else None
        out["excluded"] = sorted(set(neg))
        # numerics
        n = out["numeric"]
        m = re.search(r"\bN\s*[-:/ ]\s*P\s*[-:/ ]\s*K\b[^0-9]{0,15}" + NUM + r"\s*[-:/,]\s*" + NUM + r"\s*[-:/,]\s*" + NUM, q) or \
            re.search(r"\bNPK\b[^0-9]{0,15}" + NUM + r"\s*[-:/,]\s*" + NUM + r"\s*[-:/,]\s*" + NUM, q, re.I)
        if m:
            n["n"], n["p"], n["k"] = (float(m.group(i)) for i in (1, 2, 3))
        for f, words, single in (("n", "nitrogen", "N"), ("p", "phosph(?:orus|orous)", "P"), ("k", "potassium|potash", "K")):
            if f in n:
                continue
            m = re.search(rf"(?:{words})\s*(?:level|content|value|\(\w\))?\s*(?:of|is|at|=|:)?\s*{NUM}", q, re.I) or \
                re.search(rf"(?<![A-Za-z0-9]){single}(?![A-Za-z])\s*(?:=|:|of|is)?\s*{NUM}(?![\d.]*\s*(?:mm|%|c\b))", q)
            if m:
                n[f] = float(m.group(1))
        m = re.search(rf"\bph\s*(?:level|value)?\s*(?:of|is|at|=|:)?\s*{NUM}", q, re.I)
        if m:
            n["ph"] = float(m.group(1))
        m = re.search(rf"{NUM}\s*mm", q, re.I) or re.search(rf"rain(?:fall)?\s*(?:of|is|at|=|:|around|about)?\s*{NUM}", q, re.I)
        if m:
            n["rainfall"] = float(m.group(1))
        m = re.search(rf"{NUM}\s*(?:°|º|degrees?|deg)\s*c?\b", q, re.I) or re.search(rf"temp(?:erature)?\s*(?:of|is|at|=|:|around|about)?\s*{NUM}", q, re.I)
        if m:
            n["temperature"] = float(m.group(1))
        m = re.search(rf"{NUM}\s*%\s*(?:relative\s*)?humidity", q, re.I) or re.search(rf"humidity\s*(?:of|is|at|=|:|around|about)?\s*{NUM}", q, re.I)
        if m:
            n["humidity"] = float(m.group(1))
        return out

    # ------------------------------------------------------------------ main
    def classify(self, query: str, explicit: dict | None = None, use_llm: bool = True) -> tuple[ConditionProfile, dict]:
        explicit = explicit or {}
        parsed = self._parse_text(query)
        prof = ConditionProfile()
        notes: dict = {"llm": "not_used"}

        def setf(name, value, prov, note=None):
            setattr(prof, name, ConditionField(value=value, provenance=prov, note=note))

        for name in ["crop", "state", "season", "n", "p", "k", "ph", "rainfall", "temperature", "humidity", "fertilizer", "pesticide"]:
            if explicit.get(name) is not None:
                v = explicit[name]
                if name in ("crop", "state", "season"):
                    v = self._canon(name, v) or v
                setf(name, v, "user_provided")
        for name in ("crop", "state", "season"):
            if getattr(prof, name).provenance == "missing" and parsed.get(name):
                note = f"from '{parsed['season_alias']}'" if name == "season" and parsed.get("season_alias") else None
                setf(name, parsed[name], "parsed_from_query", note)
        for name, v in parsed["numeric"].items():
            lo, hi = RANGE_FIELDS[name]
            if getattr(prof, name).provenance == "missing":
                if lo <= v <= hi:
                    setf(name, v, "parsed_from_query")
                else:
                    prof.unusual_values.append(f"{name}={v} in the query is outside the valid range {lo}-{hi} and was ignored")
        prof.excluded_crops = parsed["excluded"]

        # LLM augmentation (structured, validated, provenance model_inferred)
        if use_llm and llm.configured:
            try:
                ext = llm.complete_json(
                    "You extract agricultural context from a farmer's question. Use ONLY the question text. "
                    "crop/state/season must be copied from these allowed values or be null. "
                    f"Allowed seasons: {self.seasons}. Allowed states: {self.states}. Allowed crops: {self.crops}. "
                    "excluded_crops: crops the user wants to avoid or replace. objectives: short goals (e.g. 'low water use'). "
                    "constraints: explicit user constraints. Do not invent numbers.",
                    wrap_untrusted("user_query", query), LLMExtraction, max_tokens=800)
                notes["llm"] = "used"
                for name, allowed in (("crop", self.crops), ("state", self.states), ("season", self.seasons)):
                    v = self._canon(name, getattr(ext, name))
                    if v and getattr(prof, name).provenance == "missing":
                        setf(name, v, "model_inferred", "Inferred by the LLM from wording; not applied as a hard filter")
                prof.excluded_crops = sorted(set(prof.excluded_crops) | {c for c in (self._canon("crop", x) for x in ext.excluded_crops) if c})
                prof.objectives = [o[:120] for o in ext.objectives[:5]]
                prof.constraints = [c[:160] for c in ext.constraints[:5]]
            except LLMError as e:
                notes["llm"] = f"failed: {e.kind}"
        elif use_llm:
            notes["llm"] = "unavailable: ANTHROPIC_API_KEY not configured (deterministic parsing only)"

        if prof.get("crop") and prof.get("crop") in prof.excluded_crops:
            prof.crop = ConditionField()
        # missing information
        labels = {"n": "nitrogen (N)", "p": "phosphorus (P)", "k": "potassium (K)", "ph": "soil pH", "rainfall": "rainfall",
                  "temperature": "temperature", "humidity": "humidity", "state": "state/region", "season": "season"}
        prof.missing_information = [lab for f, lab in labels.items() if getattr(prof, f).provenance == "missing"]
        prof.unusual_values += unusual_values(prof.values())
        essential = [prof.get(n, True) for n in ("crop", "state", "season", "n", "p", "k", "ph", "rainfall", "temperature", "humidity")]
        if not any(v is not None for v in essential):
            prof.needs_clarification = True
            prof.clarification_question = ("I could not find a crop, state, season or any soil/weather values in your question. "
                                           "Please add at least a state or season (e.g. 'Kharif in Assam') or some conditions "
                                           "such as pH, rainfall or N/P/K.")
        return prof, notes

    def _canon(self, name: str, value) -> str | None:
        if not value:
            return None
        pool = {"crop": self.crops, "state": self.states, "season": self.seasons}[name]
        v = str(value).strip()
        for c in pool:
            if c.lower() == v.lower():
                return c
        if name == "crop":
            return CROP_ALIASES.get(v.lower()) or self.crop_lookup.get(_norm(v))
        if name == "season":
            return SEASON_ALIASES.get(v.lower())
        if name == "state":
            return STATE_ALIASES.get(v.lower())
        return None
