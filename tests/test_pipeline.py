from pathlib import Path

import pytest

from rag.chunker import chunk_pages
from rag.config import Settings
from rag.evaluation import evaluate, load_eval_set
from rag.loader import load_file
from rag.pipeline import StudyRAG, parse_citations

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_material"


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    r = StudyRAG(Settings(data_dir=tmp_path_factory.mktemp("data")))
    for f in sorted(SAMPLE.iterdir()):
        r.add_file(f)
    return r


def test_pdf_pages_and_sections():
    pages, _ = load_file(SAMPLE / "genai_unit5_notes.pdf")
    assert {p.page for p in pages} == {1, 2, 3, 4, 5}
    assert any(p.section and p.section.startswith("5.3") for p in pages)


def test_chunker_validates_and_keeps_page_numbers():
    pages, _ = load_file(SAMPLE / "genai_unit5_notes.pdf")
    chunks = chunk_pages(pages, 300, 50)
    assert all(c.page in {1, 2, 3, 4, 5} for c in chunks)
    assert max(len(c.text) for c in chunks) <= 300
    with pytest.raises(ValueError):
        chunk_pages(pages, 100, 100)


def test_parse_citations():
    assert parse_citations("A [1]. B [2, 3][5].", 3) == [1, 2, 3]


def test_in_scope_answer_is_cited_with_page(rag):
    a = rag.ask("How many dimensions are in all-MiniLM-L6-v2 embeddings?")
    assert not a.refused and "384" in a.text
    assert a.citations and a.citations[0].page == 3


def test_out_of_scope_is_refused(rag):
    for q in ["Who won the FIFA World Cup in 2018?", "Give me a recipe for butter chicken.", "What is the capital of Australia?"]:
        a = rag.ask(q)
        assert a.refused and not a.citations, q


def test_document_filter_and_remove(rag):
    a = rag.ask("What is chunk overlap?", sources=["neural_networks_lab_manual.md"])
    assert a.refused or all(c.source == "neural_networks_lab_manual.md" for c in a.citations)


def test_full_eval_quality(rag):
    df, s = evaluate(rag, load_eval_set(ROOT / "evaluation" / "eval_set.json"))
    assert s["hit_rate@k"] >= 0.85
    assert s["refusal_rate (out-of-scope)"] >= 0.85
    assert s["behaviour_accuracy"] >= 0.85
