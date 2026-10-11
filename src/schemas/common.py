"""Shared Pydantic schemas: filters, condition profile, retrieval results."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, model_validator

Provenance = Literal["user_provided", "parsed_from_query", "model_inferred", "missing"]

# numeric fields that support inclusive [min, max] range filtering, with plausible validation ranges
RANGE_FIELDS: dict[str, tuple[float, float]] = {
    "n": (0, 500), "p": (0, 300), "k": (0, 300), "ph": (3, 10),
    "rainfall": (0, 12000), "temperature": (-10, 50), "humidity": (0, 100),
    "fertilizer": (0, 1e11), "pesticide": (0, 1e9),
    "fertilizer_per_ha": (0, 1e5), "pesticide_per_ha": (0, 1e4), "area": (0, 1e9), "yield": (0, 1e6),
}


def _filter_fields():
    f = {
        "crop": (Optional[list[str]], Field(None, description="Exact crop names (case-insensitive)")),
        "season": (Optional[list[str]], Field(None)),
        "state": (Optional[list[str]], Field(None)),
        "origin": (Optional[list[str]], Field(None, description="'real' and/or 'synthetic'")),
        "year_min": (Optional[int], Field(None, ge=1900, le=2100)),
        "year_max": (Optional[int], Field(None, ge=1900, le=2100)),
    }
    for name, (lo, hi) in RANGE_FIELDS.items():
        f[f"{name}_min"] = (Optional[float], Field(None, ge=lo, le=hi))
        f[f"{name}_max"] = (Optional[float], Field(None, ge=lo, le=hi))
    return f


_BaseFilters = create_model("_BaseFilters", **_filter_fields())


class CaseFilters(_BaseFilters):  # type: ignore[valid-type,misc]
    """Metadata filters applied against structured case fields. Numeric ranges are inclusive."""

    @model_validator(mode="after")
    def _check_ranges(self):
        pairs = [("year_min", "year_max")] + [(f"{n}_min", f"{n}_max") for n in RANGE_FIELDS]
        for lo, hi in pairs:
            a, b = getattr(self, lo), getattr(self, hi)
            if a is not None and b is not None and a > b:
                raise ValueError(f"{lo} ({a}) must be <= {hi} ({b})")
        return self

    def active(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v not in (None, [], "")}


class ConditionField(BaseModel):
    value: Optional[float | str] = None
    provenance: Provenance = "missing"
    note: Optional[str] = None


class ConditionProfile(BaseModel):
    """Structured agricultural conditions with per-field provenance."""
    crop: ConditionField = Field(default_factory=ConditionField)
    state: ConditionField = Field(default_factory=ConditionField)
    season: ConditionField = Field(default_factory=ConditionField)
    n: ConditionField = Field(default_factory=ConditionField)
    p: ConditionField = Field(default_factory=ConditionField)
    k: ConditionField = Field(default_factory=ConditionField)
    ph: ConditionField = Field(default_factory=ConditionField)
    rainfall: ConditionField = Field(default_factory=ConditionField)
    temperature: ConditionField = Field(default_factory=ConditionField)
    humidity: ConditionField = Field(default_factory=ConditionField)
    fertilizer: ConditionField = Field(default_factory=ConditionField)
    pesticide: ConditionField = Field(default_factory=ConditionField)
    excluded_crops: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    unusual_values: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    def get(self, name: str, explicit_only: bool = False):
        """Value of a field. explicit_only=True ignores model_inferred values (used for hard filters)."""
        f: ConditionField = getattr(self, name)
        if f.provenance == "missing" or (explicit_only and f.provenance == "model_inferred"):
            return None
        return f.value

    def values(self, explicit_only: bool = False) -> dict:
        names = ["crop", "state", "season", "n", "p", "k", "ph", "rainfall", "temperature", "humidity", "fertilizer", "pesticide"]
        return {n: self.get(n, explicit_only) for n in names if self.get(n, explicit_only) is not None}


class RetrievalItem(BaseModel):
    case_id: str
    hybrid_rank: int
    hybrid_score: float
    keyword_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    matched_terms: list[str] = Field(default_factory=list)
    relevance_explanation: str = ""
    record: dict = Field(default_factory=dict)
    priority_score: Optional[float] = None
    priority_components: Optional[dict] = None
    priority_explanation: Optional[str] = None


class RetrievalResponse(BaseModel):
    query: str
    mode: str
    items: list[RetrievalItem]
    total_candidates_after_filter: int
    filtered_out: int
    filters_applied: dict = Field(default_factory=dict)
    backends: dict = Field(default_factory=dict)
    fallback: Optional[str] = None
    scoring: str = ("Hybrid score = Reciprocal Rank Fusion: sum(1/(60+rank)) over BM25 and semantic rankings. "
                    "It is a rank-based ordering score, NOT a probability or calibrated similarity.")
