"""Module 9 - Chunk size & overlap experiment.

For every (chunk_size, overlap) pair the corpus is re-chunked, re-embedded and queried with the eval set.
Retrieval is brute-force cosine in NumPy (mathematically identical to Chroma's cosine space, but without
building a database per configuration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chunker import chunk_pages
from .config import Settings
from .evaluation import first_relevant_rank
from .guard import assess
from .retriever import POOL_FACTOR, rerank
from .types import Hit


def run_chunk_experiment(pages, items, embedder, settings: Settings, sizes, overlaps, k: int = 4, progress=None) -> pd.DataFrame:
    q_vecs = embedder.encode([it["question"] for it in items], is_query=True)
    cache: dict[str, np.ndarray] = {}
    combos = [(s, o) for s in sizes for o in overlaps if o < s]
    rows = []
    for ci, (size, ov) in enumerate(combos):
        chunks = chunk_pages(pages, size, ov)
        texts = [c.embed_text for c in chunks]
        missing = sorted({t for t in texts if t not in cache})
        if missing:
            for t, v in zip(missing, embedder.encode(missing)):
                cache[t] = v
        mat = np.stack([cache[t] for t in texts])
        hits_rr, ranks, behaviour = [], [], []
        for it, qv in zip(items, q_vecs):
            sims = mat @ qv
            top = np.argsort(-sims)[: k * POOL_FACTOR]
            pool = [Hit(chunks[i].id, chunks[i].text, chunks[i].source, chunks[i].page, chunks[i].section, float(sims[i])) for i in top]
            hits = rerank(it["question"], pool, k)
            refused = not assess(it["question"], hits, settings).in_scope
            if it["type"] == "in_scope":
                r = first_relevant_rank(hits, it.get("evidence", []))
                ranks.append(r)
                behaviour.append(not refused)
            else:
                behaviour.append(refused)
        in_ranks = [r for r in ranks]
        rows.append({
            "chunk_size": size, "overlap": ov, "chunks": len(chunks),
            "avg_chars": round(float(np.mean([len(c.text) for c in chunks])), 0),
            "hit_rate": float(np.mean([r is not None for r in in_ranks])) if in_ranks else float("nan"),
            "mrr": float(np.mean([1 / r if r else 0 for r in in_ranks])) if in_ranks else float("nan"),
            "guard_accuracy": float(np.mean(behaviour)),
        })
        if progress:
            progress((ci + 1) / len(combos))
    df = pd.DataFrame(rows)
    return df.sort_values(["hit_rate", "mrr", "guard_accuracy"], ascending=False).reset_index(drop=True)
