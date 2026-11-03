"""Module 10 - Answer-quality evaluation.

Eval-set format (JSON list):
  {"id": "q1", "type": "in_scope" | "out_of_scope", "question": "...",
   "evidence": ["phrase that must appear in a relevant chunk", ...],   # in_scope only
   "keywords": ["terms a good answer should mention", ...]}           # in_scope only (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .text import content_terms, normalize
from .types import Hit


def load_eval_set(path: str | Path) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in items:
        it.setdefault("type", "in_scope")
    return items


def is_relevant(text: str, evidence: list[str]) -> bool:
    t = normalize(text)
    return any(normalize(e) in t for e in evidence)


def first_relevant_rank(hits: list[Hit], evidence: list[str]) -> Optional[int]:
    for i, h in enumerate(hits, start=1):
        if is_relevant(h.text, evidence):
            return i
    return None


def _faithfulness(answer: str, context: str) -> float:
    import re
    a = content_terms(re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", answer))
    return len(a & content_terms(context)) / len(a) if a else 1.0


def evaluate(rag, items: list[dict], progress=None) -> tuple[pd.DataFrame, dict]:
    rows = []
    for n, it in enumerate(items):
        ans = rag.ask(it["question"])
        in_scope = it["type"] == "in_scope"
        ev = it.get("evidence", [])
        rank = first_relevant_rank(ans.hits, ev) if in_scope and ev else None
        row = {
            "id": it.get("id", f"q{n+1}"), "type": it["type"], "question": it["question"],
            "refused": ans.refused, "correct_behaviour": (not ans.refused) if in_scope else ans.refused,
            "retrieval_rank": rank, "hit": rank is not None if in_scope else None,
            "rr": (1 / rank if rank else 0.0) if in_scope else None,
            "cited_correct": None, "keyword_recall": None, "faithfulness": None,
            "best_similarity": round(ans.top_score, 3), "reason": ans.reason, "latency_s": round(ans.latency, 2),
            "answer": ans.text,
        }
        if in_scope and not ans.refused:
            row["cited_correct"] = any(is_relevant(c.text, ev) for c in ans.citations) if ev else None
            kws = it.get("keywords", [])
            if kws:
                al = normalize(ans.text)
                row["keyword_recall"] = sum(normalize(k) in al for k in kws) / len(kws)
            row["faithfulness"] = _faithfulness(ans.text, " ".join(c.text for c in ans.citations) or " ".join(h.text for h in ans.hits))
        rows.append(row)
        if progress:
            progress((n + 1) / len(items))
    df = pd.DataFrame(rows)
    return df, summarise(df)


def summarise(df: pd.DataFrame) -> dict:
    ins, oos = df[df["type"] == "in_scope"], df[df["type"] == "out_of_scope"]
    answered = ins[~ins["refused"]]
    mean = lambda s: float(s.dropna().astype(float).mean()) if s.notna().any() else float("nan")
    return {
        "questions": len(df), "in_scope": len(ins), "out_of_scope": len(oos),
        "hit_rate@k": mean(ins["hit"]), "mrr": mean(ins["rr"]),
        "answer_rate (in-scope)": float((~ins["refused"]).mean()) if len(ins) else float("nan"),
        "refusal_rate (out-of-scope)": float(oos["refused"].mean()) if len(oos) else float("nan"),
        "behaviour_accuracy": float(df["correct_behaviour"].mean()),
        "citation_accuracy": mean(answered["cited_correct"]),
        "keyword_recall": mean(answered["keyword_recall"]),
        "faithfulness": mean(answered["faithfulness"]),
    }
