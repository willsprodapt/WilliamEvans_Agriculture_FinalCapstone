"""python scripts/build_index.py  -> artifacts/indexes/{cases.faiss,bm25.pkl,manifest.json}"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import build_index  # noqa: E402
from src.rag.store import CaseStore  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    store = CaseStore()
    print(f"Embedding {len(store)} cases ...")
    info = build_index(store)
    info["seconds"] = round(time.time() - t, 1)
    print(json.dumps(info, indent=2))
