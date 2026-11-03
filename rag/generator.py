"""Module 6 - Grounded answer generation.

Backends: Ollama (local LLM), Hugging Face Transformers (local model) or an *extractive* fallback that needs
no LLM at all and can only quote sentences from the retrieved chunks (grounded by construction).
"""
from __future__ import annotations

import re

import numpy as np
import requests

from .config import Settings
from .types import Hit

NOT_FOUND = "NOT_IN_MATERIAL"

SYSTEM_PROMPT = f"""You are a study assistant. You answer a student's doubt STRICTLY from the numbered course-material excerpts provided.
Rules:
1. Use ONLY information found in the excerpts. Never use outside or general knowledge, even if you know the answer.
2. After every sentence or claim, cite the supporting excerpt(s) like [1] or [2][3].
3. Use the terminology and notation of the excerpts.
4. If the excerpts do not contain enough information to answer the question, reply with exactly: {NOT_FOUND}
5. Be concise and clear. Use short bullet points or numbered steps when helpful."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.label})\n{h.text}" for i, h in enumerate(hits, start=1))


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    return f"EXCERPTS:\n{build_context(hits)}\n\nSTUDENT QUESTION: {question}\n\nANSWER (cite with [n]; or {NOT_FOUND}):"


class OllamaGenerator:
    def __init__(self, s: Settings):
        self.url, self.model = s.ollama_url.rstrip("/"), s.ollama_model
        self.name = f"ollama:{self.model}"

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=2)
            names = [m["name"] for m in r.json().get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] and ":" not in self.model for n in names)
        except Exception:
            return False

    def generate(self, question: str, hits: list[Hit]) -> str:
        r = requests.post(f"{self.url}/api/chat", timeout=300, json={
            "model": self.model, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_user_prompt(question, hits)}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class HFGenerator:
    def __init__(self, s: Settings):
        self.model_name = s.hf_model
        self.name = f"hf:{self.model_name}"
        self._pipe = None

    def generate(self, question: str, hits: list[Hit]) -> str:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-generation", model=self.model_name, device_map="auto")
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, hits)}]
        out = self._pipe(msgs, max_new_tokens=400, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


class ExtractiveGenerator:
    """No-LLM fallback: returns the sentences of the retrieved chunks that best match the question."""
    name = "extractive (no LLM)"

    def __init__(self, s: Settings, embedder):
        self.s, self.embedder = s, embedder

    def generate(self, question: str, hits: list[Hit]) -> str:
        cands = []  # (hit_no, pos, sentence)
        for n, h in enumerate(hits[:4], start=1):
            sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", h.text) if len(x.strip()) >= 25]
            sents = [x for x in sents if not x.rstrip().endswith("?")] or sents   # prefer statements over questions
            for pos, sent in enumerate(sents):
                cands.append((n, pos, sent))
        if not cands:
            return NOT_FOUND
        q = self.embedder.encode([question], is_query=True)[0]
        m = self.embedder.encode([c[2] for c in cands])
        sims = m @ q
        order = np.argsort(-sims)[:3]
        best = float(sims[order[0]])
        if best < self.s.min_similarity:
            return NOT_FOUND
        keep = [i for i in order if sims[i] >= max(self.s.min_similarity, best - 0.15)]
        keep.sort(key=lambda i: (cands[i][0], cands[i][1]))
        lines = [f"- {cands[i][2]} [{cands[i][0]}]" for i in keep]
        return "Here is what your course material says:\n\n" + "\n".join(lines)


def get_generator(s: Settings, embedder, backend: str | None = None):
    backend = backend or s.llm_backend
    if backend == "ollama":
        return OllamaGenerator(s)
    if backend == "hf":
        return HFGenerator(s)
    if backend == "extractive":
        return ExtractiveGenerator(s, embedder)
    o = OllamaGenerator(s)  # auto
    return o if o.available() else ExtractiveGenerator(s, embedder)
