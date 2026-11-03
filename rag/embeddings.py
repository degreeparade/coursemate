"""Module 3 - Embedding generation (Sentence Transformers)."""
from __future__ import annotations

import zlib

import numpy as np

from .config import Settings
from .text import tokens


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        low = model_name.lower()
        self._q_prefix = "Represent this sentence for searching relevant passages: " if "bge" in low else ("query: " if "e5" in low else "")
        self._d_prefix = "passage: " if "e5" in low else ""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        prefix = self._q_prefix if is_query else self._d_prefix
        vecs = self.model.encode([prefix + t for t in texts], normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=32)
        return np.asarray(vecs, dtype="float32")


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. NOT for production - it exists so the test-suite
    and CI can run the whole pipeline offline without downloading a transformer model."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        toks = tokens(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = zlib.crc32(g.encode())
            v[h % self.dim] += 1.0
        v = np.sqrt(v)
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._vec(t) for t in texts])


def get_embedder(settings: Settings):
    if settings.embed_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embed_model)
