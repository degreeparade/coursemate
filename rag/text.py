"""Tiny text utilities shared by the guard, the extractive generator and the evaluator."""
from __future__ import annotations

import re

STOPWORDS = set("""a an the and or but if then else of in on at to for from by with without about as is are was were be been being
do does did done can could should would will shall may might must have has had having this that these those it its
what which who whom whose when where why how i me my we our you your he she they them their there here than so such
not no nor too very just also into over under between among per via explain define describe tell give write please
mention state discuss list""".split())


def stem(w: str) -> str:
    for suf in ("ingly", "edly", "ations", "ation", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 2 or t.isdigit():
            out.append(stem(t))
    return out


def content_terms(text: str) -> set[str]:
    return set(tokens(text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
