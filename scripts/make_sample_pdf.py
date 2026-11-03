"""Generates sample_material/genai_unit5_notes.pdf (5 pages) so the demo has a real, paginated PDF."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

PAGES = [
 [("h1", "Unit V: Retrieval-Augmented Generation"),
  ("h2", "5.1 Why Retrieval-Augmented Generation?"),
  ("p", "Large language models (LLMs) store knowledge in their parameters during training. This creates three problems: the knowledge has a cutoff date, the model may hallucinate plausible but false statements, and it has no access to private documents such as a student's own lecture notes. Retrieval-Augmented Generation (RAG) solves this by retrieving relevant passages from an external knowledge base at question time and supplying them to the LLM as context. The model is instructed to answer only from that context."),
  ("p", "A RAG system has two phases. The indexing phase loads documents, splits them into chunks, converts each chunk into an embedding and stores it in a vector database. The query phase embeds the question, retrieves the top-k most similar chunks and passes them to the LLM to generate a grounded answer."),
  ("p", "Advantages of RAG: answers can be updated by changing documents instead of retraining the model, every answer can cite its source, and hallucination is reduced.")],
 [("h2", "5.2 Document Loading and Chunking"),
  ("p", "Document loaders convert PDFs, text files and web pages into plain text while keeping metadata such as the file name and page number. In Python, the PyPDF library reads a PDF page by page. Keeping the page number is essential for citations."),
  ("p", "Chunking splits long text into smaller pieces because embedding models have limited input length and because small, focused chunks give more precise retrieval. The chunk size is the maximum number of characters or tokens in a chunk. The chunk overlap is the number of characters shared between consecutive chunks so that a sentence cut at a boundary is still found in full in one of the chunks."),
  ("h2", "5.2.1 Choosing chunk size and overlap"),
  ("p", "Very small chunks (under 200 characters) lose context, while very large chunks (over 1500 characters) mix several topics and dilute the embedding. A common starting point is a chunk size of 500 to 1000 characters with an overlap of 10 to 20 percent. The best values must be found experimentally by measuring hit rate on a set of test questions."),
  ("p", "Recursive character splitting tries to split on paragraphs first, then lines, then sentences, and finally words, which keeps related text together.")],
 [("h2", "5.3 Embeddings and Vector Stores"),
  ("p", "An embedding model maps a piece of text to a dense vector so that texts with similar meaning lie close together. Sentence Transformers models such as all-MiniLM-L6-v2 produce 384-dimensional vectors. Similarity between two vectors is usually measured with cosine similarity, which is the cosine of the angle between them and ranges from -1 to 1."),
  ("p", "A vector store such as ChromaDB stores the embeddings together with their metadata and supports fast nearest-neighbour search. ChromaDB uses the HNSW (Hierarchical Navigable Small World) graph index, which finds approximate nearest neighbours much faster than comparing the query with every vector. ChromaDB can persist the collection on disk so the index survives restarts."),
  ("p", "The same embedding model must be used for indexing and querying. Using different models gives vectors in different spaces and the similarity scores become meaningless.")],
 [("h2", "5.4 Retrieval and Grounded Generation"),
  ("p", "Semantic retrieval embeds the user's question and returns the top-k chunks with the highest cosine similarity. Maximal Marginal Relevance (MMR) can be used to reduce redundancy by balancing relevance with diversity among the retrieved chunks."),
  ("p", "In grounded generation the retrieved chunks are placed in the prompt together with a strict instruction: answer only from the context, cite the source of every claim, and say that the answer is not available if the context does not contain it. A low temperature such as 0 to 0.2 keeps the model factual."),
  ("h2", "5.4.1 Out-of-syllabus detection"),
  ("p", "A trustworthy study assistant must refuse questions that are outside the course material. Three checks are used: first, a minimum similarity threshold on the best retrieved chunk; second, a lexical coverage check that verifies the question's key terms appear in the retrieved text; third, an instruction to the LLM to reply NOT_IN_MATERIAL when the context is insufficient. Refusing is the correct behaviour; answering from general knowledge defeats the purpose of RAG.")],
 [("h2", "5.5 Evaluating RAG Systems"),
  ("p", "Retrieval quality is measured with hit rate at k (the fraction of questions for which a relevant chunk appears in the top k results) and Mean Reciprocal Rank (MRR), which is the average of 1 divided by the rank of the first relevant chunk. Generation quality is measured with faithfulness, meaning every claim in the answer is supported by the retrieved context, and with answer relevance, meaning the answer addresses the question."),
  ("p", "For a restraint-focused assistant, refusal accuracy is also measured: the fraction of out-of-syllabus questions that are correctly refused and of in-syllabus questions that are correctly answered. Experiments that vary chunk size and overlap show the trade-off between retrieval precision and context completeness.")],
]

out = Path(__file__).resolve().parent.parent / "sample_material" / "genai_unit5_notes.pdf"
st = getSampleStyleSheet()
els = []
for i, page in enumerate(PAGES):
    for kind, text in page:
        els.append(Paragraph(text, {"h1": st["Title"], "h2": st["Heading2"], "p": st["BodyText"]}[kind]))
        els.append(Spacer(1, 8))
    if i < len(PAGES) - 1:
        els.append(PageBreak())
SimpleDocTemplate(str(out), pagesize=A4, title="Generative AI - Unit V Notes").build(els)
print("wrote", out)
