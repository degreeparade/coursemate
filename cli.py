"""Command-line interface: python cli.py {ingest,ask,eval,experiment} ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import ROOT, Settings
from rag.evaluation import evaluate, load_eval_set
from rag.experiment import run_chunk_experiment
from rag.loader import SUPPORTED, load_file
from rag.pipeline import StudyRAG

DEFAULT_EVAL = ROOT / "evaluation" / "eval_set.json"
DEFAULT_DOCS = ROOT / "sample_material"


def _docs(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="index files or a folder"); p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask", help="ask one question"); p.add_argument("question")
    p = sub.add_parser("eval", help="answer-quality evaluation"); p.add_argument("--eval-set", default=str(DEFAULT_EVAL)); p.add_argument("--out", default="evaluation/results.csv")
    p = sub.add_parser("experiment", help="chunk size/overlap grid")
    p.add_argument("--docs", default=str(DEFAULT_DOCS)); p.add_argument("--eval-set", default=str(DEFAULT_EVAL))
    p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200])
    p.add_argument("--overlaps", type=int, nargs="+", default=[0, 50, 100, 200])
    p.add_argument("--k", type=int, default=4); p.add_argument("--out", default="evaluation/chunk_experiment.csv")
    a = ap.parse_args()

    if a.cmd == "ingest":
        rag = StudyRAG()
        for raw in a.paths:
            path = Path(raw)
            for f in (_docs(path) if path.is_dir() else [path]):
                r = rag.add_file(f)
                print(f"[OK] {r.source}: {r.pages} segments -> {r.chunks} chunks", *r.warnings)
    elif a.cmd == "ask":
        rag = StudyRAG(); ans = rag.ask(a.question)
        print(("[REFUSED] " if ans.refused else "") + ans.text)
        print(f"\n[{ans.reason} | best sim {ans.top_score:.2f} | coverage {ans.coverage:.0%} | {ans.backend}]")
        for c in ans.citations:
            print(f"  [{c.n}] {c.label}")
    elif a.cmd == "eval":
        rag = StudyRAG(); df, summ = evaluate(rag, load_eval_set(a.eval_set))
        df.to_csv(a.out, index=False)
        for k, v in summ.items():
            print(f"{k:30s} {v:.3f}" if isinstance(v, float) else f"{k:30s} {v}")
        print(f"\nPer-question results -> {a.out}")
    elif a.cmd == "experiment":
        rag = StudyRAG(persist=False)
        pages = [pg for f in _docs(Path(a.docs)) for pg in load_file(f)[0]]
        df = run_chunk_experiment(pages, load_eval_set(a.eval_set), rag.embedder, rag.settings, a.sizes, a.overlaps, a.k)
        df.to_csv(a.out, index=False); print(df.to_string(index=False)); print(f"\n-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
