"""Retrieval experiments: keyword-only vs semantic-only vs hybrid (RRF) vs hybrid + entity-aware boost.

Relevance labels are deterministic structured filters over the dataset (crop / state / season equality, numeric predicates); they are never
produced by an LLM or by the retriever. Two corpora are evaluated separately: `kaggle` (observed rows only; synthetic rows filtered out) and
`full` (Kaggle + synthetic). Query types:

  full_constraint   "<crop> cultivation in <state> during the <season> season"          (all three constraints stated)
  partial           "<crop> <state>" and a natural paraphrase                              (one constraint implied by omission)
  typo              crop name with two adjacent letters swapped                           (robustness of lexical vs semantic)
  numeric           "<state> records with rainfall above X mm" / pH + rainfall ranges    (numbers; embeddings are weak on these)

LIMITATION: labels come from the same metadata the query is built from, so these queries test whether the retriever honours structured
constraints written in prose. They do NOT measure agronomic usefulness of the retrieved cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..rag.retriever import get_retriever
from ..rag.store import get_store
from ..schemas.common import CaseFilters
from . import metrics

K = 10
SEED = settings.random_seed
MODES = {"keyword_only": dict(mode="keyword"), "semantic_only": dict(mode="semantic"),
         "hybrid_rrf": dict(mode="hybrid", entity_boost=False), "hybrid_rrf+entity_boost": dict(mode="hybrid", entity_boost=True)}


def _typo(word: str, rng) -> str:
    w = list(word)
    idx = [i for i in range(len(w) - 1) if w[i].isalpha() and w[i + 1].isalpha() and w[i] != w[i + 1]]
    if not idx:
        return word
    i = int(rng.choice(idx))
    w[i], w[i + 1] = w[i + 1], w[i]
    return "".join(w)


def build_queries(df: pd.DataFrame, n_each: int = 40, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    g = df.groupby(["crop", "state", "season"]).size().reset_index(name="n")
    g = g[(g["n"] >= 3) & (g["n"] <= 60)]
    pick = g.sample(min(3 * n_each, len(g)), random_state=seed).reset_index(drop=True)
    qs = []
    for i, r in pick.iterrows():
        rel = (df["crop"] == r.crop) & (df["state"] == r.state) & (df["season"] == r.season)
        if i < n_each:
            qs.append({"type": "full_constraint", "query": f"{r.crop} cultivation in {r.state} during the {r.season} season", "pred": ("crop", "state", "season"), "vals": (r.crop, r.state, r.season)})
        elif i < 2 * n_each:
            relcs = (df["crop"] == r.crop) & (df["state"] == r.state)
            qs.append({"type": "partial", "query": f"how is {r.crop.lower()} doing in {r.state}" if i % 2 else f"{r.crop} {r.state}", "pred": ("crop", "state"), "vals": (r.crop, r.state)})
        else:
            first = r.crop.split()[0].split("(")[0]
            if len(first) >= 4:
                qs.append({"type": "typo", "query": f"{_typo(first, rng)} cultivation in {r.state} during the {r.season} season", "pred": ("crop", "state", "season"), "vals": (r.crop, r.state, r.season)})
    # numeric predicates
    states = df["state"].value_counts().head(25).index.tolist()
    for st in rng.choice(states, size=min(n_each // 2, len(states)), replace=False):
        sub = df[df["state"] == st]
        thr = float(sub["rainfall"].quantile(.75))
        qs.append({"type": "numeric", "query": f"{st} records with annual rainfall above {thr:.0f} mm", "numeric": ("state", st, "rainfall_gt", thr)})
    for ph, rain in [(7.5, 800), (6.0, 1500), (7.0, 1000), (5.5, 2000), (6.5, 600), (7.8, 700), (6.2, 1200), (5.8, 1800)]:
        qs.append({"type": "numeric", "query": f"soil pH above {ph} and annual rainfall below {rain} mm", "numeric": ("ph_rain", ph, rain)})
    return qs


def _relevant(df: pd.DataFrame, q: dict) -> set[str]:
    if "pred" in q:
        m = np.ones(len(df), bool)
        for col, v in zip(q["pred"], q["vals"]):
            m &= (df[col] == v).to_numpy()
    else:
        kind = q["numeric"]
        if kind[0] == "state":
            m = ((df["state"] == kind[1]) & (df["rainfall"] > kind[3])).to_numpy()
        else:
            m = ((df["ph"] > kind[1]) & (df["rainfall"] < kind[2])).to_numpy()
    return set(df.loc[m, "case_id"])


def run() -> dict:
    store = get_store()
    ret = get_retriever(store)
    out = {"k": K, "label_note": __doc__.split("LIMITATION:")[1].strip(), "corpora": {}, "failures": {}}
    for corpus in ("kaggle", "full"):
        df = store.df[~store.df["is_synthetic"]] if corpus == "kaggle" else store.df
        filt = CaseFilters(origin=["real"]) if corpus == "kaggle" else None
        qs = [q for q in build_queries(df) if (rel := _relevant(df, q)) and len(rel) >= 3]
        res: dict = {m: {} for m in MODES}
        per_q: dict = {m: [] for m in MODES}
        for q in qs:
            rel = _relevant(df, q)
            for m, kw in MODES.items():
                ids = [i.case_id for i in ret.search(q["query"], filt, top_k=K, **kw).items]
                row = {"type": q["type"], "query": q["query"], "n_relevant": len(rel), "retrieved": ids,
                       "p@k": metrics.precision_at_k(ids, rel, K), "r_precision": metrics.r_precision(ids, rel, K),
                       "recall@k": metrics.recall_at_k(ids, rel, K), "mrr": metrics.reciprocal_rank(ids, rel),
                       "ndcg@k": metrics.ndcg_at_k(ids, rel, K), "hit@k": metrics.hit_at_k(ids, rel, K)}
                per_q[m].append(row)
        types = sorted({q["type"] for q in qs})
        for m in MODES:
            for t in types + ["ALL"]:
                rows = [r for r in per_q[m] if t == "ALL" or r["type"] == t]
                if not rows:
                    continue
                agg = {k: metrics.mean([r[k] for r in rows]) for k in ("p@k", "r_precision", "recall@k", "mrr", "ndcg@k", "hit@k")}
                ci = metrics.bootstrap_ci([r["ndcg@k"] for r in rows])
                agg["ndcg@k_ci95"] = None if ci is None else [round(ci[1], 4), round(ci[2], 4)]
                agg["n_queries"] = len(rows)
                res[m][t] = agg
        out["corpora"][corpus] = {"n_documents": int(len(df)), "n_queries": len(qs), "query_types": {t: sum(q["type"] == t for q in qs) for t in types}, "results": res}
        # failures: worst hybrid+entity queries and where they differ from hybrid_rrf
        worst = sorted(per_q["hybrid_rrf+entity_boost"], key=lambda r: (r["ndcg@k"], r["p@k"]))[:5]
        out["failures"][corpus] = [{"query": r["query"], "type": r["type"], "n_relevant": r["n_relevant"], "ndcg@k": round(r["ndcg@k"], 3),
                                   "top3_retrieved": [{k: store.record(store.id_to_idx[c])[k] for k in ("crop", "state", "season", "year", "rainfall", "ph")} for c in r["retrieved"][:3]]} for r in worst]
    return out
