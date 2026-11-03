"""Module 8 - Out-of-syllabus detection.

Layer 1 (here):   best-chunk similarity floor + lexical coverage of the question's key terms.
Layer 2 (generator): the LLM is told to reply NOT_IN_MATERIAL if the excerpts do not answer the question.
Either layer can trigger a refusal - the assistant never falls back to general knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .text import content_terms
from .types import Hit


def coverage(query: str, hits: list[Hit]) -> float:
    """Fraction of the question's content terms that appear somewhere in the retrieved text."""
    q = content_terms(query)
    if not q:
        return 1.0
    ctx: set[str] = set()
    for h in hits:
        ctx |= content_terms(h.text)
        if h.section:
            ctx |= content_terms(h.section)
    return len(q & ctx) / len(q)


@dataclass
class Verdict:
    in_scope: bool
    reason: str
    top_score: float
    coverage: float


def assess(query: str, hits: list[Hit], s: Settings) -> Verdict:
    if not hits:
        return Verdict(False, "no_material", 0.0, 0.0)
    top = max(h.score for h in hits)
    cov = coverage(query, hits[:3])
    if top < s.min_similarity:
        return Verdict(False, f"low_similarity (best {top:.2f} < {s.min_similarity:.2f})", top, cov)
    if top < s.strong_similarity and cov < s.min_coverage:
        return Verdict(False, f"low_term_coverage (only {cov:.0%} of the question's key terms appear in the material)", top, cov)
    return Verdict(True, "ok", top, cov)
