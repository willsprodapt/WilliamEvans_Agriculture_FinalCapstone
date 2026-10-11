"""Hybrid retrieval: BM25 (lexical) + FAISS/SentenceTransformers (semantic) fused with Reciprocal Rank Fusion.

Filtering: structured metadata filters are applied against the case store BEFORE ranking. Each backend
produces a full ranking, candidates outside the filter mask are discarded, and the first `candidate_k`
survivors of each ranking are fused. Filtered-out counts are reported.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

from ..config import settings
from ..schemas.common import CaseFilters, RetrievalItem, RetrievalResponse
from .store import CaseStore

log = logging.getLogger("terramind.retriever")

RRF_K = 60
STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "with", "my", "i", "is", "are", "what", "which",
        "me", "that", "this", "at", "be", "can", "should", "do", "have", "has", "it", "as", "by", "from", "or",
        "under", "grown", "grow", "farm", "crops", "crop", "season", "during"}
_tok = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _tok.findall(text.lower())


class RetrievalError(RuntimeError):
    pass


_NEGATORS = {"not", "except", "exclude", "excluding", "without", "avoid", "no", "never", "alternative", "alternatives", "instead", "than"}
ENTITY_TIER_BONUS = 0.05      # every detected entity matched: ranks above partial matches
ENTITY_PARTIAL_BONUS = 0.03   # scaled by the fraction of detected entities matched


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _osa(a: str, b: str) -> int:
    """Optimal-string-alignment distance (insert/delete/substitute/adjacent transposition)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _variants(store: CaseStore, col: str) -> dict[str, list[str]]:
    """normalised surface form -> canonical names, cached on the store. Crops also expose their first word when it is unique to one crop."""
    cache = getattr(store, "_entity_variants", None)
    if cache is None:
        cache = store._entity_variants = {}
    if col not in cache:
        names = list(store.df[col].unique())
        v: dict[str, list[str]] = {}
        for n in names:
            v.setdefault(_norm(n), []).append(n)
            head = _norm(n.split("(")[0])
            if col == "crop" and head:
                v.setdefault(head, []).append(n)
        if col == "crop":
            first: dict[str, list[str]] = {}
            for n in names:
                w = _norm(n.split("(")[0]).split()
                if len(w) > 1:
                    first.setdefault(w[0], []).append(n)
            for w, ns in first.items():
                if len(ns) == 1 and w not in v:
                    v[w] = ns
        cache[col] = v
    return cache[col]


def detect_entities(store: CaseStore, query: str) -> dict[str, list[str]]:
    """Crop / state / season names mentioned in the query. Exact word-boundary matches first; then near-misses (one typo or
    transposition) for tokens that match no vocabulary word. Negated mentions ('not rice') are ignored."""
    q = " " + _norm(query) + " "
    toks = q.split()
    known_words = {w for col in ("crop", "state", "season") for k in _variants(store, col) for w in k.split()}
    out: dict[str, list[str]] = {}
    for col in ("crop", "state", "season"):
        found = []
        for v, names in _variants(store, col).items():
            if not v or (col == "season" and v == "year"):
                continue
            m = re.search(rf" {re.escape(v)} ", q)
            if not m:
                continue
            before = q[:m.start()].split()[-2:]
            if any(w in _NEGATORS for w in before):
                continue
            found += names
        if not found and col in ("crop", "state"):
            # fuzzy: unigram and bigram windows of unknown tokens
            for n_gram in (1, 2):
                for i in range(len(toks) - n_gram + 1):
                    win = toks[i:i + n_gram]
                    if any(w in known_words or w in STOP or len(w) < 4 for w in win) and n_gram == 1:
                        continue
                    if n_gram == 2 and all(w in known_words for w in win):
                        continue
                    phrase = " ".join(win)
                    if i > 0 and toks[i - 1] in _NEGATORS:
                        continue
                    tol = 1 if len(phrase) < 8 else 2
                    best = [(d, names) for v, names in _variants(store, col).items()
                            if abs(len(v) - len(phrase)) <= tol and len(v.split()) == n_gram and (d := _osa(phrase, v)) <= tol]
                    if best:
                        dmin = min(d for d, _ in best)
                        found += [n for d, names in best if d == dmin for n in names]
                if found:
                    break
        found = list(dict.fromkeys(found))
        found = [f for f in found if not any(f != g and _norm(f) in _norm(g) for g in found)]
        if found:
            out[col] = found
    return out


class Embedder:
    """Lazy SentenceTransformer wrapper (loaded once per process)."""
    _model = None

    def __init__(self, name: str | None = None):
        self.name = name or settings.embedding_model

    def _load(self):
        if Embedder._model is None or getattr(Embedder._model, "_tm_name", None) != self.name:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.name)
            m._tm_name = self.name
            Embedder._model = m
        return Embedder._model

    def encode(self, texts: list[str], batch_size: int = 128, progress: bool = False) -> np.ndarray:
        v = self._load().encode(texts, batch_size=batch_size, show_progress_bar=progress,
                                convert_to_numpy=True, normalize_embeddings=True)
        return v.astype("float32")


# --------------------------------------------------------------------------- index build
def build_index(store: CaseStore, index_dir: Path | None = None, embedder: Embedder | None = None,
                progress: bool = True) -> dict:
    import faiss
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or Embedder()
    texts = store.df["text"].tolist()
    emb = embedder.encode(texts, progress=progress)
    index = faiss.IndexFlatIP(emb.shape[1])   # normalized vectors -> inner product == cosine similarity
    index.add(emb)
    faiss.write_index(index, str(index_dir / "cases.faiss"))
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    manifest = {
        "dataset_version": store.dataset_version, "n_cases": len(store), "embedding_model": embedder.name,
        "dim": int(emb.shape[1]), "index_type": "faiss.IndexFlatIP (cosine on L2-normalised vectors)",
        "bm25": "rank_bm25.BM25Okapi over lowercase alphanumeric tokens",
        "case_ids": store.df["case_id"].tolist(),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {k: v for k, v in manifest.items() if k != "case_ids"}


def index_status(store: CaseStore | None = None, index_dir: Path | None = None) -> dict:
    index_dir = Path(index_dir or settings.index_dir)
    mp = index_dir / "manifest.json"
    st = {"built": False, "valid": False, "faiss": (index_dir / "cases.faiss").exists(),
          "bm25": (index_dir / "bm25.pkl").exists(), "reason": None}
    if not mp.exists():
        st["reason"] = "Index not built. Run: python scripts/build_index.py"
        return st
    m = json.loads(mp.read_text(encoding="utf-8"))
    st.update(built=True, n_cases=m["n_cases"], embedding_model=m["embedding_model"], dataset_version=m["dataset_version"])
    if store is not None:
        if m["dataset_version"] != store.dataset_version or m["n_cases"] != len(store):
            st["reason"] = "Index is stale (dataset changed). Rebuild: python scripts/build_index.py"
            return st
    st["valid"] = st["faiss"] and st["bm25"]
    return st


# --------------------------------------------------------------------------- retriever
class HybridRetriever:
    def __init__(self, store: CaseStore, index_dir: Path | None = None, embedder: Embedder | None = None):
        self.store = store
        self.index_dir = Path(index_dir or settings.index_dir)
        self.embedder = embedder or Embedder()
        self.bm25 = None
        self.faiss = None
        self.errors: dict[str, str] = {}
        self._load()

    def _load(self):
        st = index_status(self.store, self.index_dir)
        if not st["built"] or st["reason"]:
            self.errors["index"] = st["reason"] or "index missing"
        try:
            with open(self.index_dir / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            self.errors["keyword"] = f"BM25 unavailable: {e}"
        try:
            import faiss
            self.faiss = faiss.read_index(str(self.index_dir / "cases.faiss"))
            if self.faiss.ntotal != len(self.store):
                self.faiss = None
                self.errors["semantic"] = "FAISS index size does not match dataset; rebuild index"
        except Exception as e:  # noqa: BLE001
            self.errors["semantic"] = f"FAISS unavailable: {e}"
        if self.errors.get("index") and "stale" in self.errors["index"]:
            # a stale index would map ranks to wrong records: refuse to use it
            self.bm25 = None
            self.faiss = None

    @property
    def status(self) -> dict:
        return {"keyword": self.bm25 is not None, "semantic": self.faiss is not None, "errors": self.errors}

    # ----- single backends ------------------------------------------------------------
    def _kw_scores(self, query: str) -> np.ndarray | None:
        toks = [t for t in tokenize(query) if t not in STOP]
        return np.asarray(self.bm25.get_scores(toks)) if toks else None

    def _sem_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])
        D, I = self.faiss.search(q, self.faiss.ntotal)
        out = np.full(self.faiss.ntotal, -1.0, dtype="float32")
        out[I[0]] = D[0]
        return out

    @staticmethod
    def _top(scores: np.ndarray | None, mask: np.ndarray, k: int, positive_only: bool):
        if scores is None:
            return []
        sc = np.where(mask, scores, -np.inf)
        order = np.argsort(-sc)[:k]
        return [(int(i), float(sc[i])) for i in order if np.isfinite(sc[i]) and (sc[i] > 0 or not positive_only)]

    # ----- public ---------------------------------------------------------------------
    def search(self, query: str, filters: CaseFilters | None = None, top_k: int = 10, candidate_k: int = 100,
               mode: str = "hybrid", w_keyword: float = 1.0, w_semantic: float = 1.0,
               entity_boost: bool = True) -> RetrievalResponse:
        if mode not in ("hybrid", "keyword", "semantic"):
            raise ValueError("mode must be hybrid, keyword or semantic")
        mask = self.store.mask(filters)
        n_ok = int(mask.sum())
        resp_kwargs = dict(query=query, total_candidates_after_filter=n_ok, filtered_out=int(len(mask) - n_ok),
                           filters_applied=filters.active() if filters else {})
        if n_ok == 0:
            return RetrievalResponse(items=[], mode=mode, backends=self.status, **resp_kwargs)

        kw, sem, notes = [], [], []
        kw_all = sem_all = None
        used = {"keyword": False, "semantic": False}
        if mode in ("hybrid", "keyword"):
            if self.bm25 is None:
                notes.append(f"keyword backend unavailable ({self.errors.get('keyword', self.errors.get('index', 'not loaded'))})")
            else:
                try:
                    kw_all = self._kw_scores(query)
                    kw = self._top(kw_all, mask, candidate_k, True)
                    used["keyword"] = True
                except Exception as e:  # noqa: BLE001
                    log.warning("keyword search failed: %s", e)
                    notes.append(f"keyword search failed: {type(e).__name__}")
        if mode in ("hybrid", "semantic"):
            if self.faiss is None:
                notes.append(f"semantic backend unavailable ({self.errors.get('semantic', self.errors.get('index', 'not loaded'))})")
            else:
                try:
                    sem_all = self._sem_scores(query)
                    sem = self._top(sem_all, mask, candidate_k, False)
                    used["semantic"] = True
                except Exception as e:  # noqa: BLE001
                    log.warning("semantic search failed: %s", e)
                    notes.append(f"semantic search failed: {type(e).__name__}")
        if not used["keyword"] and not used["semantic"]:
            raise RetrievalError("No retrieval backend available: " + "; ".join(notes or ["index not loaded"]))
        fallback = None
        if mode == "hybrid" and (used["keyword"] != used["semantic"]):
            fallback = ("Semantic search unavailable; returning keyword (BM25) results only. " if used["keyword"]
                        else "Keyword search unavailable; returning semantic results only. ") + "; ".join(notes)

        kw_rank = {i: (r + 1, s) for r, (i, s) in enumerate(kw)}
        sem_rank = {i: (r + 1, s) for r, (i, s) in enumerate(sem)}
        fused: dict[int, float] = {}
        for i, (r, _) in kw_rank.items():
            fused[i] = fused.get(i, 0) + w_keyword / (RRF_K + r)
        for i, (r, _) in sem_rank.items():
            fused[i] = fused.get(i, 0) + w_semantic / (RRF_K + r)
        # Entity-aware stage: crop / state / season named in the query (and not already a hard filter) get a soft boost,
        # and the records matching ALL of them are re-ranked among themselves by BM25 + cosine so none is lost to candidate_k.
        entities = detect_entities(self.store, query) if (entity_boost and mode == "hybrid") else {}
        for col in ("crop", "state", "season"):
            if filters is not None and getattr(filters, col, None):
                entities.pop(col, None)
        if entities:
            hit = np.zeros((len(self.store), len(entities)), dtype=bool)
            for j, (col, names) in enumerate(entities.items()):
                hit[:, j] = self.store._lower[col].isin([n.lower() for n in names]).to_numpy()
            frac = hit.mean(axis=1)
            full = mask & (frac >= 1.0 - 1e-9)
            for i in np.flatnonzero(mask & (frac > 0)):
                fused[int(i)] = fused.get(int(i), 0.0) + ENTITY_PARTIAL_BONUS * float(frac[i])
            if full.any():
                idx = np.flatnonzero(full)
                local: dict[int, float] = {int(i): ENTITY_TIER_BONUS for i in idx}
                for scores in (kw_all, sem_all):
                    if scores is None:
                        continue
                    order = idx[np.argsort(-scores[idx], kind="stable")]
                    for r, i in enumerate(order, 1):
                        local[int(i)] += 1.0 / (RRF_K + r)
                for i, v in local.items():
                    fused[i] = max(fused.get(i, 0.0), ENTITY_PARTIAL_BONUS + v)
            notes.append("entity-aware boost: " + ", ".join(f"{c}={'/'.join(n)}" for c, n in entities.items()))
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

        q_terms = {t for t in tokenize(query) if t not in STOP}
        items = []
        for pos, (i, score) in enumerate(ranked, 1):
            rec = self.store.record(i)
            doc_terms = set(tokenize(self.store.df["text"].iat[i]))
            matched = sorted(q_terms & doc_terms)[:10]
            kr, ks = kw_rank.get(i, (None, None))
            sr, ss = sem_rank.get(i, (None, None))
            parts = []
            if kr:
                parts.append(f"keyword rank {kr} (BM25 {ks:.2f}; matched: {', '.join(matched) or 'none'})")
            if sr:
                parts.append(f"semantic rank {sr} (cosine {ss:.3f})")
            if entities:
                ok = [f"{c}={rec[c]}" for c, names in entities.items() if str(rec[c]).lower() in {n.lower() for n in names}]
                if ok:
                    parts.append("matches query " + ", ".join(ok))
            items.append(RetrievalItem(
                case_id=rec["case_id"], hybrid_rank=pos, hybrid_score=round(score, 5),
                keyword_rank=kr, keyword_score=None if ks is None else round(ks, 4),
                semantic_rank=sr, semantic_score=None if ss is None else round(ss, 4),
                matched_terms=matched, relevance_explanation="; ".join(parts) or "n/a", record=rec))
        return RetrievalResponse(items=items, mode=mode, backends={**self.status, "used": used}, fallback=fallback, **resp_kwargs)


_retriever: HybridRetriever | None = None


def get_retriever(store: CaseStore) -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(store)
    return _retriever


def reset_retriever():
    global _retriever
    _retriever = None
