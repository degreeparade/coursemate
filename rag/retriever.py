"""Module 5 - Semantic retrieval (+ a light lexical re-rank)."""
from __future__ import annotations

from typing import Optional, Sequence

from .guard import coverage
from .types import Hit

LEXICAL_WEIGHT = 0.10
POOL_FACTOR = 3


def rerank(query: str, candidates: list[Hit], k: int) -> list[Hit]:
    """Re-order candidates by cosine + small bonus for query-term overlap; keep raw cosine in .score."""
    scored = sorted(candidates, key=lambda h: h.score + LEXICAL_WEIGHT * coverage(query, [h]), reverse=True)
    return scored[:k]


class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int, sources: Optional[Sequence[str]] = None) -> list[Hit]:
        q = self.embedder.encode([query], is_query=True)[0]
        pool = self.store.query(q, k * POOL_FACTOR, sources)
        return rerank(query, pool, k)
