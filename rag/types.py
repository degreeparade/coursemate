"""Plain dataclasses shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def format_location(source: str, page: Optional[int], section: Optional[str]) -> str:
    parts = [source]
    if page is not None and page >= 0:
        parts.append(f"p. {page}")
    if section:
        parts.append(f"§ {section}")
    return " · ".join(parts)


@dataclass
class Page:
    """One loaded unit of text: a PDF page (or a section of it) or a section of a text file."""
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class Chunk:
    text: str
    source: str
    page: Optional[int]
    section: Optional[str]
    idx: int
    id: str

    @property
    def embed_text(self) -> str:
        # Prefixing the section title gives the embedding model extra topical context.
        return f"{self.section}\n{self.text}" if self.section else self.text

    def metadata(self) -> dict:
        # ChromaDB metadata cannot hold None.
        return {
            "source": self.source,
            "page": -1 if self.page is None else int(self.page),
            "section": self.section or "",
            "idx": self.idx,
        }


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: Optional[int]
    section: Optional[str]
    score: float  # cosine similarity in [-1, 1]

    @property
    def label(self) -> str:
        return format_location(self.source, self.page, self.section)


@dataclass
class Citation:
    n: int
    source: str
    page: Optional[int]
    section: Optional[str]
    snippet: str
    text: str
    score: float

    @property
    def label(self) -> str:
        return format_location(self.source, self.page, self.section)


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    reason: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    top_score: float = 0.0
    coverage: float = 0.0
    citations_verified: bool = True
    latency: float = 0.0
    backend: str = ""
