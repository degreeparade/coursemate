"""Module 2 - Chunking strategy.

Recursive character splitting (paragraph -> line -> sentence -> word) applied *inside* each page/section
segment, so a chunk can never straddle two pages and the cited page number is always exact.
"""
from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .types import Chunk, Page

MIN_CHARS = 20


def chunk_pages(pages: list[Page], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for p in pages:
        for piece in splitter.split_text(p.text):
            piece = piece.strip()
            if len(piece) < MIN_CHARS:
                continue
            idx = counters.get(p.source, 0)
            counters[p.source] = idx + 1
            cid = hashlib.sha1(f"{p.source}|{p.page}|{idx}|{piece}".encode()).hexdigest()[:20]
            chunks.append(Chunk(text=piece, source=p.source, page=p.page, section=p.section, idx=idx, id=cid))
    return chunks
