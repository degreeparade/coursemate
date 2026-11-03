"""CourseMate - Streamlit interface (Module 11: Student interface)."""
from __future__ import annotations

import html
import json

import altair as alt
import pandas as pd
import streamlit as st

from rag import evaluation, experiment
from rag.config import ROOT, Settings
from rag.loader import SUPPORTED, load_file
from rag.pipeline import StudyRAG

SAMPLE_DIR = ROOT / "sample_material"
SAMPLE_EVAL = ROOT / "evaluation" / "eval_set.json"

st.set_page_config(page_title="CourseMate · Doubt Solver", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 100%); color:#fff; padding:1.3rem 1.6rem; border-radius:16px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:1.8rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.92;}
.refusal {border-left:5px solid #F59E0B; background:#FFFBEB; padding:.9rem 1rem; border-radius:8px; color:#78350F;}
.pill {display:inline-block; background:#EEF2FF; color:#3730A3; border-radius:999px; padding:.15rem .7rem; font-size:.8rem; margin-right:.3rem;}
[data-testid="stMetric"] {background:#F4F5FB; padding:.7rem 1rem; border-radius:12px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model and vector store…")
def get_rag() -> StudyRAG:
    return StudyRAG(Settings())


rag = get_rag()
s = rag.settings
docs = rag.documents()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📚 CourseMate")
    st.caption("Answers **only** from your uploaded material — with page-level citations.")

    st.subheader("1 · Upload material")
    files = st.file_uploader("Notes, textbook chapters, lab manuals, past papers",
                             type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    if st.button("➕ Add to knowledge base", disabled=not files, width="stretch", type="primary"):
        with st.spinner("Parsing, chunking and embedding…"):
            for f in files:
                try:
                    r = rag.ingest_bytes(f.name, f.getvalue())
                    st.success(f"{r.source}: {r.chunks} chunks")
                    for w in r.warnings:
                        st.warning(w)
                except Exception as e:  # noqa: BLE001
                    st.error(f"{f.name}: {e}")
        st.rerun()
    if st.button("🎓 Load sample course material", width="stretch"):
        with st.spinner("Indexing sample files…"):
            for p in sorted(SAMPLE_DIR.iterdir()):
                if p.suffix.lower() in SUPPORTED:
                    rag.add_file(p)
        st.rerun()

    st.subheader("2 · Knowledge base")
    if not docs:
        st.info("Nothing uploaded yet.")
    for name, n in docs.items():
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{name}**  \n<span class='pill'>{n} chunks</span>", unsafe_allow_html=True)
        if c2.button("🗑", key=f"rm_{name}", help=f"Remove {name}"):
            rag.remove(name)
            st.rerun()
    scope = st.multiselect("Search only in…", list(docs), placeholder="All documents") if docs else []

    with st.expander("⚙️ Settings"):
        s.top_k = st.slider("Passages retrieved (top-k)", 1, 8, s.top_k)
        s.min_similarity = st.slider("Relevance threshold", 0.0, 0.8, float(s.min_similarity), 0.01,
                                     help="Higher = stricter. Questions whose best match scores below this are refused.")
        backend = st.selectbox("Answer generator", ["auto", "ollama", "extractive", "hf"],
                               index=["auto", "ollama", "extractive", "hf"].index(s.llm_backend))
        if backend != s.llm_backend:
            rag.set_backend(backend)
        st.caption("**Chunking** (needs re-index)")
        s.chunk_size = st.slider("Chunk size (chars)", 200, 2000, s.chunk_size, 50)
        s.chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, min(s.chunk_overlap, s.chunk_size - 50), 10)
        if st.button("🔄 Re-index with these settings", width="stretch"):
            with st.spinner("Re-indexing…"):
                rag.reindex()
            st.rerun()

# ------------------------------------------------------------------ header
st.markdown("""<div class="hero"><h1>📚 CourseMate — your syllabus-only doubt solver</h1>
<p>Ask anything from your notes. If it isn't in your uploaded material, CourseMate says so instead of guessing.</p></div>""",
            unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", len(docs))
m2.metric("Chunks indexed", sum(docs.values()))
m3.metric("Embedding model", rag.embedder.name.split("/")[-1])
m4.metric("Generator", rag.generator.name.split(":")[0].replace(" (no LLM)", ""))

tab_chat, tab_exp, tab_eval, tab_about = st.tabs(["💬 Ask a doubt", "🧪 Chunking experiment", "📊 Evaluation", "ℹ️ How it works"])


# ------------------------------------------------------------------ chat
def render_answer(a) -> None:
    if a.refused:
        st.markdown(f'<div class="refusal">🚫 <b>Not in your course material</b><br>{html.escape(a.text)}</div>',
                    unsafe_allow_html=True)
        with st.expander("Why was this refused?"):
            st.caption(f"Reason: {a.reason}  \nBest similarity **{a.top_score:.2f}** · key-term coverage **{a.coverage:.0%}**")
            if a.hits:
                st.markdown("**Closest passages found (not used for answering):**")
                for h in a.hits[:3]:
                    st.markdown(f"- {h.label} — similarity {h.score:.2f}")
    else:
        st.markdown(a.text)
        if not a.citations_verified:
            st.caption("⚠️ The model did not mark citations explicitly; showing the most relevant sources instead.")
        st.markdown("**📎 Sources**")
        for c in a.citations:
            with st.expander(f"[{c.n}] {c.label}   ·   match {c.score:.2f}"):
                st.write(c.snippet)
    st.caption(f"⏱ {a.latency:.1f}s · {a.backend}")


with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not docs:
        st.info("👈 Upload your course material (or click **Load sample course material**) to start asking questions.")
    else:
        if not st.session_state.messages:
            st.markdown("**Try:** *What is chunk overlap?* · *Why can a perceptron not learn XOR?* · "
                        "and a question outside the notes, like *Who won the 2018 World Cup?*")
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "📚"):
                st.markdown(m["content"]) if m["role"] == "user" else render_answer(m["answer"])
        q = st.chat_input("Ask a doubt from your course material…")
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="📚"):
                with st.spinner("Searching your material…"):
                    ans = rag.ask(q, sources=scope or None)
                render_answer(ans)
            st.session_state.messages.append({"role": "assistant", "answer": ans})
        if st.session_state.messages and st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ------------------------------------------------------------------ helpers for eval-set selection
def pick_eval_set(key: str):
    choice = st.radio("Evaluation set", ["Bundled sample set", "Upload my own (JSON)"], horizontal=True, key=key)
    if choice.startswith("Bundled"):
        return evaluation.load_eval_set(SAMPLE_EVAL)
    up = st.file_uploader("Eval set JSON", type=["json"], key=key + "_up")
    if up:
        items = json.loads(up.getvalue())
        for it in items:
            it.setdefault("type", "in_scope")
        return items
    return None


# ------------------------------------------------------------------ experiment
with tab_exp:
    st.markdown("Re-chunk the stored documents with many **chunk size × overlap** settings and measure how well "
                "each retrieves the right passage (hit-rate@k, MRR) and how well the out-of-syllabus guard behaves.")
    items = pick_eval_set("exp_set")
    c1, c2, c3 = st.columns(3)
    sizes = c1.multiselect("Chunk sizes", [200, 300, 400, 600, 800, 1000, 1200, 1500], [200, 400, 800, 1200])
    overlaps = c2.multiselect("Overlaps", [0, 25, 50, 100, 150, 200, 300], [0, 50, 100, 200])
    k = c3.number_input("k", 1, 10, s.top_k)
    files_on_disk = [p for p in sorted(s.uploads_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if st.button("▶ Run experiment", type="primary", disabled=not (items and files_on_disk and sizes and overlaps)):
        bar = st.progress(0.0, text="Running…")
        pages = [pg for f in files_on_disk for pg in load_file(f)[0]]
        st.session_state.exp_df = experiment.run_chunk_experiment(pages, items, rag.embedder, s, sizes, overlaps, int(k), lambda x: bar.progress(x))
        bar.empty()
    if not files_on_disk:
        st.info("Upload documents first.")
    df = st.session_state.get("exp_df")
    if df is not None and len(df):
        best = df.iloc[0]
        st.success(f"Best configuration: **chunk size {int(best.chunk_size)}, overlap {int(best.overlap)}** — "
                   f"hit-rate {best.hit_rate:.0%}, MRR {best.mrr:.2f}")
        for metric, title in [("hit_rate", "Hit-rate@k"), ("mrr", "Mean reciprocal rank")]:
            base = alt.Chart(df).encode(x=alt.X("overlap:O", title="Overlap"), y=alt.Y("chunk_size:O", title="Chunk size"))
            heat = base.mark_rect().encode(color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title=title))
            txt = base.mark_text(fontSize=13).encode(text=alt.Text(f"{metric}:Q", format=".2f"),
                                                     color=alt.condition(alt.datum[metric] > 0.6, alt.value("white"), alt.value("black")))
            st.altair_chart((heat + txt).properties(title=title, height=230), width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", df.to_csv(index=False), "chunk_experiment.csv", "text/csv")
        if d2.button("✅ Apply best settings and re-index"):
            s.chunk_size, s.chunk_overlap = int(best.chunk_size), int(best.overlap)
            with st.spinner("Re-indexing…"):
                rag.reindex()
            st.rerun()


# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.markdown("Runs the **full pipeline** on a labelled question set: retrieval quality, citation correctness, "
                "keyword recall, faithfulness and — most importantly — **refusal behaviour** on out-of-syllabus questions.")
    items2 = pick_eval_set("eval_set")
    if st.button("▶ Run evaluation", type="primary", disabled=not (items2 and docs)):
        bar = st.progress(0.0, text="Evaluating…")
        st.session_state.eval = evaluation.evaluate(rag, items2, lambda x: bar.progress(x))
        bar.empty()
    if st.session_state.get("eval"):
        res, summ = st.session_state.eval
        cols = st.columns(4)
        shown = [("Hit-rate@k", "hit_rate@k"), ("MRR", "mrr"), ("Answered in-scope", "answer_rate (in-scope)"),
                 ("Refused out-of-scope", "refusal_rate (out-of-scope)"), ("Behaviour accuracy", "behaviour_accuracy"),
                 ("Citation accuracy", "citation_accuracy"), ("Keyword recall", "keyword_recall"), ("Faithfulness", "faithfulness")]
        for i, (label, key) in enumerate(shown):
            v = summ[key]
            cols[i % 4].metric(label, "–" if v != v else (f"{v:.2f}" if key == "mrr" else f"{v:.0%}"))
        st.dataframe(res.drop(columns=["answer"]), width="stretch", hide_index=True)
        st.download_button("⬇ Download results CSV", res.to_csv(index=False), "evaluation_results.csv", "text/csv")
        with st.expander("Show generated answers"):
            for _, r in res.iterrows():
                st.markdown(f"**{r['id']}** — {r['question']}  \n{r['answer']}")
                st.divider()

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
### Pipeline
1. **Load** — PyPDF reads PDFs page-by-page; Markdown/TXT are split by headings. Page numbers and section titles are kept as metadata.
2. **Chunk** — recursive character splitting *within* a page/section, so a citation's page is always exact.
3. **Embed** — Sentence-Transformers (`all-MiniLM-L6-v2`), normalised vectors.
4. **Store** — ChromaDB persistent collection, cosine space.
5. **Retrieve** — top-k semantic search with a light lexical re-rank.
6. **Guard (restraint)** — refuse if the best similarity is below the threshold, *or* the question's key terms are
   missing from the retrieved text, *or* the LLM answers `NOT_IN_MATERIAL`.
7. **Generate** — Ollama / Hugging Face LLM with a strict “excerpts only + cite every claim” prompt, or an extractive no-LLM fallback.
8. **Cite** — every answer lists the exact file, page and section, with the passage text.
""")
