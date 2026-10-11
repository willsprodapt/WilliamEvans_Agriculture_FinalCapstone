"""Input validation helpers and prompt-injection screening."""
from __future__ import annotations

import re

MAX_QUERY_CHARS = 1000

# Soft "unusual value" bounds: flagged for confirmation, not rejected (hard bounds live in the Pydantic schemas).
UNUSUAL = {
    "ph": (4.0, 9.0, "pH outside 4-9 is uncommon in agricultural soils"),
    "rainfall": (150, 5000, "annual rainfall outside 150-5000 mm is unusual for the dataset's regions"),
    "temperature": (5, 35, "mean annual temperature outside 5-35 C is unusual"),
    "humidity": (20, 95, "mean humidity outside 20-95 % is unusual"),
    "n": (10, 250, "nitrogen outside 10-250 is outside the range seen in the data"),
    "p": (5, 100, "phosphorus outside 5-100 is outside the range seen in the data"),
    "k": (5, 100, "potassium outside 5-100 is outside the range seen in the data"),
}

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now\b", r"reveal (your|the) (system )?(prompt|instructions|api key|key)",
    r"system prompt", r"api[_ -]?key", r"</?(system|assistant|untrusted_data)>",
    r"act as (an? )?(different|unrestricted)", r"jailbreak", r"developer mode",
]
_inj_re = re.compile("|".join(_INJECTION), re.I)


def injection_suspected(text: str) -> bool:
    return bool(_inj_re.search(text or ""))


def unusual_values(values: dict) -> list[str]:
    out = []
    for k, (lo, hi, msg) in UNUSUAL.items():
        v = values.get(k)
        if v is not None and not (lo <= float(v) <= hi):
            out.append(f"{k}={v}: {msg}. Please confirm this value.")
    return out


def sanitize_query(q: str) -> str:
    q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", q or "")
    return re.sub(r"\s+", " ", q).strip()
