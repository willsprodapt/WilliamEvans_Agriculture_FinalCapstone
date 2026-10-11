"""Request / response schemas for the advisory workflow and other API surfaces."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .common import CaseFilters, ConditionProfile


class AdvisoryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Natural-language question about crop choice")
    crop: Optional[str] = Field(None, max_length=60)
    state: Optional[str] = Field(None, max_length=60)
    season: Optional[str] = Field(None, max_length=30)
    n: Optional[float] = Field(None, ge=0, le=500, description="Nitrogen (state-level units of the dataset)")
    p: Optional[float] = Field(None, ge=0, le=300)
    k: Optional[float] = Field(None, ge=0, le=300)
    ph: Optional[float] = Field(None, ge=3, le=10)
    rainfall: Optional[float] = Field(None, ge=0, le=12000, description="Annual rainfall, mm")
    temperature: Optional[float] = Field(None, ge=-10, le=50, description="Mean temperature, C")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Mean humidity, %")
    fertilizer: Optional[float] = Field(None, ge=0, description="Total fertilizer (dataset units)")
    pesticide: Optional[float] = Field(None, ge=0, description="Total pesticide (dataset units)")
    area: Optional[float] = Field(None, gt=0, description="Cultivated area (hectares) for yield model 'full' feature set")
    top_k: int = Field(15, ge=1, le=50, description="Retrieved cases shown")
    max_recommendations: int = Field(5, ge=1, le=10)
    use_llm: bool = True
    include_synthetic: bool = Field(True, description="Include labelled synthetic cases in retrieval")

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query must contain at least 3 characters")
        return v

    def explicit(self) -> dict:
        names = ["crop", "state", "season", "n", "p", "k", "ph", "rainfall", "temperature", "humidity", "fertilizer", "pesticide"]
        return {n: getattr(self, n) for n in names if getattr(self, n) not in (None, "")}


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    use_llm: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[CaseFilters] = None
    top_k: int = Field(10, ge=1, le=100)
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid"


class YieldRequest(BaseModel):
    crop: str = Field(..., min_length=1, max_length=60)
    state: str = Field(..., min_length=1, max_length=60)
    season: str = Field(..., min_length=1, max_length=30)
    year: Optional[int] = Field(None, ge=1990, le=2100)
    n: Optional[float] = Field(None, ge=0, le=500)
    p: Optional[float] = Field(None, ge=0, le=300)
    k: Optional[float] = Field(None, ge=0, le=300)
    ph: Optional[float] = Field(None, ge=3, le=10)
    rainfall: Optional[float] = Field(None, ge=0, le=12000)
    temperature: Optional[float] = Field(None, ge=-10, le=50)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    area: Optional[float] = Field(None, gt=0)
    fertilizer: Optional[float] = Field(None, ge=0)
    pesticide: Optional[float] = Field(None, ge=0)


class SuitabilityRequest(BaseModel):
    crop: str = Field(..., min_length=1, max_length=60)
    state: Optional[str] = None
    season: Optional[str] = None
    n: Optional[float] = Field(None, ge=0, le=500)
    p: Optional[float] = Field(None, ge=0, le=300)
    k: Optional[float] = Field(None, ge=0, le=300)
    ph: Optional[float] = Field(None, ge=3, le=10)
    rainfall: Optional[float] = Field(None, ge=0, le=12000)
    temperature: Optional[float] = Field(None, ge=-10, le=50)
    humidity: Optional[float] = Field(None, ge=0, le=100)


class ImpactRequest(BaseModel):
    crop: Optional[str] = None
    state: Optional[str] = None
    season: Optional[str] = None


class FeedbackIn(BaseModel):
    query_id: Optional[str] = Field(None, max_length=60)
    recommendation_id: Optional[str] = Field(None, max_length=80)
    crop: Optional[str] = Field(None, max_length=60)
    helpful: bool
    rating: Optional[int] = Field(None, ge=1, le=5)
    category: Optional[Literal["incorrect_crop", "missing_evidence", "wrong_conditions", "unclear_explanation",
                               "yield_prediction", "risk_assessment", "other"]] = None
    comment: Optional[str] = Field(None, max_length=2000)
    case_ids: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("comment")
    @classmethod
    def _c(cls, v):
        return v.strip() or None if v else v


# ---- LLM structured outputs ---------------------------------------------------------
class LLMExtraction(BaseModel):
    """What Claude may extract from a farmer's query. Every value is validated and tagged model_inferred."""
    crop: Optional[str] = None
    state: Optional[str] = None
    season: Optional[str] = None
    excluded_crops: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    rationale: str = ""


class PerCropNarrative(BaseModel):
    crop: str
    explanation: str


class Narrative(BaseModel):
    summary: str
    per_crop: list[PerCropNarrative] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class JudgeRecVerdict(BaseModel):
    crop: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    notes: str = ""


class JudgeVerdict(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported"]
    evidence_support: Literal["strong", "moderate", "weak", "none"]
    condition_match: Literal["good", "partial", "poor", "unknown"]
    constraint_violations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    risk_disclosure: Literal["adequate", "insufficient", "not_applicable"]
    historical_vs_predicted_distinguished: bool = True
    per_recommendation: list[JudgeRecVerdict] = Field(default_factory=list)
    explanation: str


class AgentStatus(BaseModel):
    name: str
    status: Literal["success", "failed", "skipped", "not_run", "running", "pending"] = "pending"
    duration_ms: Optional[int] = None
    detail: Optional[str] = None


class AdvisoryResponse(BaseModel):
    query_id: str
    status: Literal["completed", "completed_with_warnings", "needs_clarification", "no_evidence", "failed"]
    query: str
    validated_input: dict
    conditions: Optional[ConditionProfile] = None
    retrieval: Optional[dict] = None
    cases: list[dict] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)
    narrative: Optional[dict] = None
    impact_analysis: Optional[dict] = None
    agents: list[AgentStatus] = Field(default_factory=list)
    stages: list[dict] = Field(default_factory=list)
    judge: Optional[dict] = None
    warnings: list[str] = Field(default_factory=list)
    clarification: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
