"""Module 1 - Document upload & parsing.

PDFs are read page-by-page (PyPDF) so every chunk keeps its page number. Headings are detected with
light heuristics so that citations can also name the section (e.g. "p. 3 · § 5.3 Embeddings...").
Markdown / plain-text files are split on their headings and carry the section name only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .types import Page

SUPPORTED = (".pdf", ".txt", ".md")

_MD = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_NAMED = re.compile(r"^(?i:chapter|unit|module|section|lab|experiment|topic|lecture|part)\s+[\w.]+.*$")
_NUMBERED = re.compile(r"^\d+(?:\.\d+){1,3}\s+[A-Z][^\n]*$")


def is_heading(line: str) -> Optional[str]:
    s = line.strip()
    if not s or len(s) > 90:
        return None
    m = _MD.match(s)
    if m:
        return m.group(1).strip()
    if s[-1] in ".,;!":
        return None
    if _NAMED.match(s) or _NUMBERED.match(s):
        return s
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 6 and s.isupper() and len(s.split()) <= 8:
        return s
    return None


def split_sections(text: str, current: Optional[str]) -> tuple[list[tuple[Optional[str], str]], Optional[str]]:
    """Split text at headings. Returns [(section, text)], and the section active at the end."""
    segments: list[tuple[Optional[str], str]] = []
    buf: list[str] = []
    sec = current
    for line in text.splitlines():
        h = is_heading(line)
        if h:
            if len([b for b in buf if b.strip()]) > 1 or (buf and not is_heading(buf[0])):
                segments.append((sec, "\n".join(buf).strip()))
            buf = [line.strip().lstrip("#").strip()]
            sec = h
        else:
            buf.append(line)
    if "".join(buf).strip():
        segments.append((sec, "\n".join(buf).strip()))
    return segments, sec


def _clean(text: str) -> str:
    text = text.replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)      # re-join words hyphenated at line end
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_pdf(path: Path, name: str) -> tuple[list[Page], list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    warnings: list[str] = []
    section: Optional[str] = None
    empty = 0
    for i, pg in enumerate(reader.pages, start=1):
        text = _clean(pg.extract_text() or "")
        if len(text) < 20:
            empty += 1
            continue
        segments, section = split_sections(text, section)
        for sec, body in segments:
            pages.append(Page(text=body, source=name, page=i, section=sec))
    if empty:
        warnings.append(f"{name}: {empty} page(s) had no extractable text (scanned images?). OCR is needed for those.")
    return pages, warnings


def _load_text(path: Path, name: str) -> tuple[list[Page], list[str]]:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    segments, _ = split_sections(text, None)
    return [Page(text=b, source=name, page=None, section=s) for s, b in segments], []


def load_file(path: str | Path, display_name: Optional[str] = None) -> tuple[list[Page], list[str]]:
    path = Path(path)
    name = display_name or path.name
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages, warns = _load_pdf(path, name)
    elif ext in (".txt", ".md"):
        pages, warns = _load_text(path, name)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED)}")
    if not pages:
        warns.append(f"{name}: no text could be extracted.")
    return pages, warns
