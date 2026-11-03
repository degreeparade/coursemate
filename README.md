# 📚 CourseMate — Doubt-Solving Study Chatbot (RAG)

An assistant that answers a student's doubts **strictly from uploaded course material** (lecture notes, textbook chapters,
lab manuals, previous papers), **cites the exact file · page · section** for every answer, and — the key behaviour —
**refuses** when the question is outside the material instead of falling back on the LLM's general knowledge.

```mermaid
flowchart LR
  A[PDF / TXT / MD upload] --> B[PyPDF loader<br/>page + section metadata]
  B --> C[Recursive chunker<br/>size / overlap]
  C --> D[Sentence-Transformers<br/>embeddings]
  D --> E[(ChromaDB<br/>cosine, persistent)]
  Q[Student question] --> F[Embed + top-k retrieve<br/>+ lexical re-rank]
  E --> F
  F --> G{Out-of-syllabus guard<br/>similarity + term coverage}
  G -- fails --> R[🚫 Refuse + show closest passages]
  G -- passes --> H[Grounded LLM<br/>Ollama / HF / extractive]
  H -- NOT_IN_MATERIAL --> R
  H --> S[Answer + [n] citations<br/>file · page · section]
```

## Module map

| Module | Where |
|---|---|
| Document upload & parsing | `rag/loader.py` (PyPDF, heading detection) |
| Chunking strategy | `rag/chunker.py` (LangChain recursive splitter, page-safe) |
| Embedding generation | `rag/embeddings.py` (Sentence Transformers) |
| Vector store setup | `rag/vectorstore.py` (ChromaDB, cosine, persisted in `data/chroma`) |
| Semantic retrieval | `rag/retriever.py` |
| Grounded answer generation | `rag/generator.py` (Ollama · Hugging Face · extractive) |
| Source citation | `rag/pipeline.py` (`[n]` parsing → file/page/section + snippet) |
| Out-of-syllabus detection & refusal | `rag/guard.py` + `NOT_IN_MATERIAL` protocol in `generator.py` |
| Chunk size & overlap experiment | `rag/experiment.py`, `cli.py experiment`, “Chunking experiment” tab |
| Answer quality evaluation | `rag/evaluation.py`, `evaluation/eval_set.json`, “Evaluation” tab |
| Student interface | `app.py` (Streamlit) |
| Deployment | `Dockerfile`, `docker-compose.yml`, steps below |

## How restraint works (3 layers)
1. **Similarity floor** – best retrieved chunk below `MIN_SIMILARITY` → refuse before the LLM is even called.
2. **Term coverage** – if similarity is only moderate, at least ~⅓ of the question's key terms must occur in the retrieved text.
3. **LLM veto** – the prompt forces the model to answer `NOT_IN_MATERIAL` when the excerpts don't suffice; the pipeline turns that into a refusal.

Refusals also show the *closest passages found (not used)* so the student can see why.

---

## Run locally

```bash
git clone <your-repo-url> coursemate && cd coursemate
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                      # opens http://localhost:8501
```
In the app click **🎓 Load sample course material**, then ask *“What is chunk overlap?”* and *“Who won the 2018 World Cup?”*.

**Better answers with a local LLM (recommended):**
```bash
# install Ollama from https://ollama.com, then
ollama pull llama3.2:3b            # ~2 GB; use qwen2.5:3b or llama3.1:8b if you have more RAM
streamlit run app.py               # Generator "auto" now picks Ollama automatically
```
Without Ollama the app still works in **extractive mode** (quotes the best sentences from your material).

**Command line**
```bash
python cli.py ingest sample_material/
python cli.py ask "What index does ChromaDB use?"
python cli.py eval                                   # answer-quality report
python cli.py experiment --sizes 200 400 800 1200 --overlaps 0 50 100 200
pytest -q                                            # unit tests (offline)
```

## Use your own material & evaluation set
Upload your PDFs in the sidebar. To evaluate on *your* syllabus, write an eval JSON (see `evaluation/eval_set.json`):
```json
{"id":"q1","type":"in_scope","question":"...","evidence":["exact phrase from the notes"],"keywords":["term1","term2"]}
{"id":"o1","type":"out_of_scope","question":"a question the notes cannot answer"}
```
Use 15–30 questions (about ⅓ out-of-scope) and tune **chunk size / overlap / relevance threshold** with the experiment and evaluation tabs.

---

## Deployment

### Option A — Docker (full stack with Ollama) ✅ recommended
```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b     # one-time model download
# open http://localhost:8501
```
Data (uploads + Chroma index) persists in `./data`. Deploy the same compose file on any VM
(AWS EC2 / Azure / GCP / a college server): open port 8501, run the three commands above. 8 GB RAM minimum for the 3B model.

### Option B — Streamlit Community Cloud (free, public URL)
Ollama can't run there, so the app runs in **extractive mode** (or call a hosted LLM — see note).
1. Push the project to a **public GitHub repo** (`git init && git add . && git commit -m "CourseMate" && git push`).
2. Go to <https://share.streamlit.io> → **New app** → select repo, branch `main`, main file `app.py` → **Deploy**.
3. In **Advanced settings → Secrets/Environment** add `LLM_BACKEND=extractive`.
4. First boot takes a few minutes (downloads the embedding model). Share the URL.
*Note:* the free tier has ~1 GB RAM and an ephemeral disk — uploads vanish on restart; re-upload or click “Load sample material”.
To use a hosted LLM, point `OLLAMA_URL` at an Ollama server you run elsewhere (Option A on a VM) and set `LLM_BACKEND=ollama`.

### Option C — Hugging Face Spaces (Docker)
1. Create a new Space → SDK **Docker** → push this repo.
2. Add `app_port: 8501` to the Space README front-matter. Set `LLM_BACKEND=extractive` (or `hf` on GPU hardware).

### Pushing to GitHub
```bash
git init && git add . && git commit -m "Initial commit: CourseMate RAG study assistant"
git branch -M main && git remote add origin https://github.com/<you>/coursemate.git && git push -u origin main
```
CI (`.github/workflows/ci.yml`) runs the tests on every push.

## Configuration (environment variables)
See `.env.example` — embedding model, generator backend, Ollama URL/model, chunk size/overlap, top-k, thresholds.

## Limitations & ideas
* Scanned PDFs need OCR (the loader warns when a page has no text).
* The deployed app shares one knowledge base across all visitors; add per-session collections for multi-user use.
* Thresholds depend on the embedding model — recalibrate with the Evaluation tab after changing models.
* Extensions: hybrid BM25 + dense retrieval, cross-encoder re-ranking, query rewriting for follow-ups, OCR, per-student collections.
