"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT / "data"))))

    # --- embeddings -------------------------------------------------------
    # "sentence-transformers" (default, real model) or "hash" (tiny offline fallback for tests)
    embed_backend: str = field(default_factory=lambda: os.getenv("EMBED_BACKEND", "sentence-transformers"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # --- generation -------------------------------------------------------
    # "auto" (Ollama if reachable, else extractive), "ollama", "hf", "extractive"
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "auto"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    hf_model: str = field(default_factory=lambda: os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))

    # --- chunking / retrieval --------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    # --- out-of-syllabus guard -------------------------------------------
    min_similarity: Optional[float] = None     # hard floor on best-chunk cosine similarity
    strong_similarity: Optional[float] = None  # above this, trust the embedding alone
    min_coverage: float = field(default_factory=lambda: _env_float("MIN_COVERAGE", 0.34))

    def __post_init__(self) -> None:
        hashing = self.embed_backend == "hash"
        if self.min_similarity is None:
            self.min_similarity = _env_float("MIN_SIMILARITY", 0.10 if hashing else 0.28)
        if self.strong_similarity is None:
            self.strong_similarity = _env_float("STRONG_SIMILARITY", 0.30 if hashing else 0.50)
        self.data_dir = Path(self.data_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"
