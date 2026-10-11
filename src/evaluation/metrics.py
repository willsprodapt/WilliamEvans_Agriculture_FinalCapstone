"""Retrieval metrics (computed only when verified relevance labels exist)."""
from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / k if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Recall@K against all relevant ids (can be < 1 even for a perfect ranker when |relevant| > K)."""
    if not relevant:
        return None
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def mean(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Binary-relevance nDCG@K."""
    import math
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def r_precision(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Precision at k' = min(K, |relevant|): the fair precision when fewer than K relevant items exist."""
    if not relevant:
        return None
    kk = min(k, len(relevant))
    return sum(1 for r in retrieved[:kk] if r in relevant) / kk


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float] | None:
    """(mean, low, high) percentile-bootstrap CI of the mean."""
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
