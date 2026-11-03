"""Module 7 - the end-to-end pipeline: ingest -> retrieve -> guard -> generate -> cite."""
from __future__ import annotations

import hashlib
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .chunker import chunk_pages
from .config import Settings
from .embeddings import get_embedder
from .generator import NOT_FOUND, get_generator
from .guard import assess
from .loader import SUPPORTED, load_file
from .retriever import Retriever
from .types import Answer, Citation
from .vectorstore import VectorStore

REFUSAL_TEXT = (
    "I couldn't find this in your uploaded course material, so I won't answer it from general knowledge. "
    "Try rephrasing the question, check that the right chapter/notes are uploaded, or ask your teacher."
)


@dataclass
class IngestReport:
    source: str
    pages: int
    chunks: int
    warnings: list[str]


def parse_citations(text: str, n_hits: int) -> list[int]:
    found: list[int] = []
    for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        for num in re.split(r"\s*,\s*", grp):
            i = int(num)
            if 1 <= i <= n_hits and i not in found:
                found.append(i)
    return found


class StudyRAG:
    def __init__(self, settings: Optional[Settings] = None, persist: bool = True):
        self.settings = settings or Settings()
        self.embedder = get_embedder(self.settings)
        coll = "course_" + hashlib.md5(self.embedder.name.encode()).hexdigest()[:10]
        self.store = VectorStore(str(self.settings.chroma_dir) if persist else None, coll)
        self.retriever = Retriever(self.embedder, self.store)
        self.generator = get_generator(self.settings, self.embedder)

    # ---------------------------------------------------------------- ingestion
    def set_backend(self, backend: str) -> None:
        self.settings.llm_backend = backend
        self.generator = get_generator(self.settings, self.embedder, backend)

    def ingest_file(self, path: str | Path, display_name: Optional[str] = None) -> IngestReport:
        name = display_name or Path(path).name
        pages, warns = load_file(path, name)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        self.store.delete_source(name)
        if chunks:
            self.store.add_chunks(chunks, self.embedder.encode([c.embed_text for c in chunks]))
        return IngestReport(name, len(pages), len(chunks), warns)

    def add_file(self, path: str | Path) -> IngestReport:
        """Copy a local file into the uploads folder (so it can be re-indexed later) and ingest it."""
        src = Path(path)
        dest = self.settings.uploads_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)
        return self.ingest_file(dest)

    def ingest_bytes(self, name: str, data: bytes) -> IngestReport:
        name = Path(name).name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported file type: {name}")
        dest = self.settings.uploads_dir / name
        dest.write_bytes(data)
        return self.ingest_file(dest)

    def reindex(self) -> list[IngestReport]:
        """Rebuild the whole index from stored uploads (use after changing chunk size / overlap)."""
        self.store.reset()
        return [self.ingest_file(p) for p in sorted(self.settings.uploads_dir.iterdir())
                if p.suffix.lower() in SUPPORTED]

    def remove(self, source: str) -> None:
        self.store.delete_source(source)
        f = self.settings.uploads_dir / source
        if f.exists():
            f.unlink()

    def documents(self) -> dict[str, int]:
        return self.store.sources()

    # ---------------------------------------------------------------- question answering
    def retrieve(self, question: str, sources: Optional[Sequence[str]] = None):
        return self.retriever.retrieve(question, self.settings.top_k, sources)

    def ask(self, question: str, sources: Optional[Sequence[str]] = None) -> Answer:
        t0 = time.time()
        hits = self.retrieve(question, sources)
        verdict = assess(question, hits, self.settings)
        base = dict(question=question, hits=hits, top_score=verdict.top_score, coverage=verdict.coverage,
                    backend=self.generator.name)
        if not verdict.in_scope:
            return Answer(text=REFUSAL_TEXT, refused=True, reason=verdict.reason, latency=time.time() - t0, **base)

        raw = self.generator.generate(question, hits)
        if NOT_FOUND in raw.upper().replace(" ", "_"):
            return Answer(text=REFUSAL_TEXT, refused=True, reason="generator_declined (excerpts do not answer it)",
                          latency=time.time() - t0, **base)

        used = parse_citations(raw, len(hits))
        verified = bool(used)
        if not used:                       # model forgot markers -> show the top sources but flag it
            used = list(range(1, min(2, len(hits)) + 1))
        cites = [Citation(n=i, source=hits[i - 1].source, page=hits[i - 1].page, section=hits[i - 1].section,
                          snippet=_snippet(hits[i - 1].text), text=hits[i - 1].text, score=hits[i - 1].score)
                 for i in used]
        return Answer(text=raw, refused=False, reason="ok", citations=cites, citations_verified=verified,
                      latency=time.time() - t0, **base)


def _snippet(text: str, n: int = 400) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"
