"""Module 4 - Vector store (ChromaDB, cosine space, persisted on disk)."""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from .types import Chunk, Hit


class VectorStore:
    def __init__(self, path: Optional[str], collection: str):
        import chromadb

        self.collection_name = collection
        self.client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self._open()

    def _open(self) -> None:
        self.col = self.client.get_or_create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._open()

    def count(self) -> int:
        return self.col.count()

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        for i in range(0, len(chunks), 128):
            part = chunks[i : i + 128]
            self.col.upsert(
                ids=[c.id for c in part],
                embeddings=embeddings[i : i + 128].tolist(),
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})

    def sources(self) -> dict[str, int]:
        if self.count() == 0:
            return {}
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metas))

    def query(self, embedding: np.ndarray, k: int, sources: Optional[Sequence[str]] = None) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        where = None
        if sources:
            where = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
        res = self.col.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, n),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(
                id=cid, text=doc, source=meta["source"],
                page=None if meta["page"] < 0 else int(meta["page"]),
                section=meta["section"] or None,
                score=float(1.0 - dist),       # cosine distance -> similarity
            ))
        return hits
