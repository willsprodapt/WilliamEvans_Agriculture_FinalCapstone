"""python scripts/check_llm.py - one real Claude call + one tiny DeepEval metric, to confirm the judge works end to end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import deepeval_runner  # noqa: E402
from src.services.llm import LLMError, llm  # noqa: E402

print("LLM status:", llm.status()["provider"], llm.status()["model"], "configured =", llm.configured)
try:
    print("Claude reply:", llm.complete_text("Reply with one word.", "Say OK", max_tokens=20))
except LLMError as e:
    print(f"Claude call FAILED [{e.kind}]: {e}")
    sys.exit(1)
res = deepeval_runner.evaluate_case("Which crop suits Punjab in Rabi?", "Wheat suits Punjab in Rabi (4.5 t/ha in 23 cases).",
                                    "Wheat is the best-supported Rabi crop in Punjab.", ["Wheat, Punjab, Rabi, 1999, yield 4.5"])
for k, v in res.items():
    print(f"{k:22s} available={v['available']} score={v.get('score')} {('- ' + str(v.get('reason'))[:90]) if not v['available'] else ''}")
sys.exit(0 if all(v["available"] for v in res.values()) else 1)
