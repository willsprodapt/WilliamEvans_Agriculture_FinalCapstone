import pytest

from src.rag.prioritizer import prioritize
from src.rag.retriever import HybridRetriever, RetrievalError, index_status
from src.schemas.common import CaseFilters


def test_index_valid(store):
    st = index_status(store)
    assert st["built"] and st["valid"] and st["n_cases"] == len(store)


def test_hybrid_returns_both_signals(retriever):
    r = retriever.search("wheat Punjab rabi", None, top_k=10)
    assert r.items and r.fallback is None
    assert all(i.record["crop"] for i in r.items)
    assert any(i.keyword_rank for i in r.items) and any(i.semantic_rank for i in r.items)
    scores = [i.hybrid_score for i in r.items]
    assert scores == sorted(scores, reverse=True)
    assert [i.hybrid_rank for i in r.items] == list(range(1, len(r.items) + 1))
    assert "not a probability" in r.scoring.lower() or "NOT a probability" in r.scoring


def test_keyword_finds_exact_crop(retriever):
    r = retriever.search("Sugarcane Uttar Pradesh", None, top_k=5, mode="keyword")
    assert {i.record["crop"] for i in r.items} == {"Sugarcane"}
    assert all(i.semantic_rank is None for i in r.items)


def test_filters_applied_before_ranking(retriever):
    f = CaseFilters(state=["Kerala"], season=["Kharif"])
    r = retriever.search("acidic soil high rainfall", f, top_k=10)
    assert r.items and all(i.record["state"] == "Kerala" and i.record["season"] == "Kharif" for i in r.items)
    assert r.filtered_out > 0 and r.total_candidates_after_filter + r.filtered_out == len(retriever.store)


def test_no_matching_cases_returns_empty_not_fabricated(retriever):
    r = retriever.search("coconut", CaseFilters(crop=["Coconut"], state=["Delhi"]), top_k=5)
    assert r.items == [] and r.total_candidates_after_filter == 0


def test_fallback_semantic_down(store):
    rt = HybridRetriever(store)
    rt.faiss = None
    r = rt.search("rice assam", None, top_k=3)
    assert r.items and "keyword" in r.fallback.lower() and all(i.semantic_rank is None for i in r.items)


def test_fallback_keyword_down(store, retriever):
    rt = HybridRetriever(store, embedder=retriever.embedder)
    rt.bm25 = None
    r = rt.search("rice assam", None, top_k=3)
    assert r.items and "semantic" in r.fallback.lower() and all(i.keyword_rank is None for i in r.items)


def test_both_backends_down_is_error(store):
    rt = HybridRetriever(store)
    rt.bm25 = None
    rt.faiss = None
    with pytest.raises(RetrievalError):
        rt.search("rice", None)


def test_stale_index_refused(store, tmp_path):
    rt = HybridRetriever(store, index_dir=tmp_path)       # empty dir: nothing built
    assert rt.bm25 is None and rt.faiss is None and "index" in rt.errors
    with pytest.raises(RetrievalError):
        rt.search("rice")


def test_prioritization_transparent_and_not_yield_based(store, retriever):
    r = retriever.search("rice kharif", CaseFilters(crop=["Rice"]), top_k=20)
    user = {"crop": "Rice", "state": "Kerala", "ph": 5.7}
    out = prioritize(store, r.items, user)
    for it in out:
        comps = it.priority_components
        assert "yield" not in "".join(comps)               # historical yield is not a component
        assert comps["season_match"]["assessed"] is False   # unknown != match
        assert abs(sum(c["contribution"] for c in comps.values()) - it.priority_score) < 1e-3
    assert [i.priority_score for i in out] == sorted([i.priority_score for i in out], reverse=True)
    kerala = [i for i in out if i.record["state"] == "Kerala"]
    other = [i for i in out if i.record["state"] != "Kerala"]
    if kerala and other:
        assert max(i.priority_score for i in kerala) >= min(i.priority_score for i in other)
