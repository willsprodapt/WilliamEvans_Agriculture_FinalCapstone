"""DeepEval integration (faithfulness, answer relevancy, contextual precision, contextual recall).

The judge model is Claude, wired through a small DeepEvalBaseLLM adapter around our LLMService (so key handling, timeouts and
sampling-parameter rules stay in one place). A metric that cannot run is recorded as unavailable with the reason - it is never
silently skipped or reported as passed.
"""
from __future__ import annotations

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

from ..config import settings  # noqa: E402
from ..services.llm import LLMError, llm  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]


def _adapter():
    from deepeval.models import DeepEvalBaseLLM

    class ClaudeJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = settings.llm_model

        def load_model(self):
            return None

        def generate(self, prompt: str, schema=None):
            if schema is not None:
                return llm.complete_json("You are a careful evaluation assistant.", prompt, schema, max_tokens=3000, effort="low")
            return llm.complete_text("You are a careful evaluation assistant.", prompt, max_tokens=3000, effort="low")

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return self.model_name

    return ClaudeJudge()


def availability() -> tuple[bool, str | None]:
    if not llm.configured:
        return False, "ANTHROPIC_API_KEY not configured: DeepEval needs an LLM judge"
    try:
        import deepeval  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"deepeval not importable: {e}"
    return True, None


def evaluate_case(query: str, actual_output: str, expected_output: str, retrieval_context: list[str]) -> dict:
    """Returns {metric: {score, passed, reason, available, error}} for one test case."""
    ok, why = availability()
    out = {m: {"available": False, "score": None, "reason": why, "passed": None} for m in METRICS}
    if not ok:
        return out
    from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric)
    from deepeval.test_case import LLMTestCase

    judge = _adapter()
    tc = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output, retrieval_context=retrieval_context)
    mets = {
        "faithfulness": FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False),
        "contextual_precision": ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
        "contextual_recall": ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
    }
    fatal: str | None = None     # billing/auth/not-configured failures affect every metric: stop calling the API
    for name, m in mets.items():
        if fatal:
            out[name] = {"available": False, "score": None, "passed": None, "reason": fatal}
            continue
        try:
            m.measure(tc)
            out[name] = {"available": True, "score": float(m.score), "passed": bool(m.is_successful()),
                         "reason": (m.reason or "")[:600], "threshold": m.threshold}
        except LLMError as e:
            out[name] = {"available": False, "score": None, "passed": None, "reason": f"LLM error ({e.kind}): {e}"}
            if e.kind in ("billing", "auth", "not_configured", "model_not_found"):
                fatal = out[name]["reason"]
        except Exception as e:  # noqa: BLE001
            out[name] = {"available": False, "score": None, "passed": None, "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    return out
