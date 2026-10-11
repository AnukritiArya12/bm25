# Document Question Answering with BM25

Upload a PDF or DOCX, ask a question, and get the most relevant passages **retrieved directly from the
document** with their source, page and BM25 score.

**There is no RAG, no LLM, no embeddings, no vector database and no semantic search anywhere in this project.**
The "answer" is the best-matching passage, shown exactly as written in your document.

---

## 1. Project overview

| Step | What happens |
|---|---|
| 1 | You upload a PDF/DOCX and click **Build / Rebuild index** |
| 2 | Text is extracted page by page (PDF) or section by section (DOCX) |
| 3 | Text is cleaned and split into overlapping, sentence-aware chunks (each keeps `source`, `page`, `chunk_id`) |
| 4 | Chunks are tokenized and a BM25 index is built (`rank-bm25`) |
| 5 | You ask a question; it is tokenized the same way |
| 6 | BM25 scores every chunk; the top-K chunks above the relevance threshold are shown |
| 7 | If nothing scores high enough, the app says so instead of guessing |

## 2. Architecture

```
Uploaded file
   │
   ▼
document_loader.py  ── validate → extract (PyMuPDF / python-docx) → pages with metadata
   │
   ▼
utils.clean_text    ── NFKC, fix hyphenated line breaks, collapse whitespace
   │
   ▼
chunker.py          ── sentence-aware chunks (size + overlap in words), one page per chunk
   │
   ▼
tokenizer.py        ── lowercase, split into words, drop a small stopword list
   │
   ▼
bm25_engine.py      ── BM25Okapi index (rank-bm25)
   │
   ▼
retriever.search    ── tokenize question → BM25 scores → top-K → relevance threshold
   │
   ▼
app.py (Streamlit)  ── ranked passages + source + page + score, or "no relevant information"
```

## 3. How BM25 works

**Simple version.** BM25 ranks passages by how well their words match the words in your question. A passage
scores higher when it contains your question's words often, when those words are rare across the whole document
(rare words are more informative than common ones), and it is not simply long (long passages contain more words by
chance, so they are slightly penalised).

**Technical version.**

```
score(D, Q) = Σ over terms t in Q of   IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
```

* `f(t,D)`: occurrences of term `t` in chunk `D`; the `k1` term makes repeated occurrences saturate (the 10th
  "python" adds much less than the 1st). Default `k1 = 1.5`.
* `|D|/avgdl`: chunk length relative to the average chunk length; `b` controls how strongly this is applied.
  Default `b = 0.75`.
* `IDF(t)`: inverse document frequency, `log(N − n(t) + 0.5) − log(n(t) + 0.5)`, where `n(t)` is the number of
  chunks containing `t`. Rare terms weigh more. (`rank-bm25` replaces negative IDF values with a small positive
  floor.)

## 4. Why BM25?

It is fast, needs no training or model downloads, works offline, runs on a laptop, and is **fully explainable**: you
can always see which keywords matched and why a passage ranked first. For questions whose wording overlaps with the
document ("Who created Python?"), it is very effective and remains a strong baseline in information retrieval.

## 5. Why embeddings are not used

Embeddings turn text into vectors and match by meaning. That would be *semantic* search, which is explicitly out of
scope. It would also need a model download and more compute. BM25 matches exact words only.

## 6. Why vector databases are not used

Vector databases (FAISS, Chroma, Pinecone...) store and search embeddings. BM25 needs no vectors, so the whole
index is a plain in-memory Python structure.

## 7. Why RAG is not used

RAG = retrieve passages, then have an LLM write an answer. Here there is **no generation step**. The app returns
the retrieved text as it appears in the document, so nothing can be invented, and no API keys are needed.

## 8. Technology stack

Python 3.11+, Streamlit, PyMuPDF (PDF), python-docx (DOCX), rank-bm25 (BM25), pytest (tests). Everything else is
the standard library.

## 9. Project structure

```
document_qa_bm25/
├── app.py                 Streamlit interface
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── utils.py           exceptions, validation, text cleaning, highlighting, shared types
│   ├── tokenizer.py       tokenization + stopwords (used for chunks AND questions)
│   ├── document_loader.py PDF/DOCX extraction with metadata (LOADERS registry for new formats)
│   ├── chunker.py         sentence-aware overlapping chunks
│   ├── bm25_engine.py     thin wrapper around rank-bm25
│   └── retriever.py       build index + search + relevance threshold
└── tests/
    ├── conftest.py        sample document + PDF/DOCX builders
    ├── test_document_loader.py
    ├── test_chunker.py
    └── test_bm25.py       tokenization, indexing, ranking, threshold, errors, metadata
```

## 10. Installation

```bash
cd document_qa_bm25
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 11. Run

```bash
streamlit run app.py     # opens http://localhost:8501
pytest                   # run the tests
```

## 12. Example questions

For a document that contains the sentence "Python was created by Guido van Rossum and first released in 1991.":

| Question | Result |
|---|---|
| `Who created Python?` | Page 1 passage, matched keywords: `created`, `python`, BM25 ≈ 3.1 (small demo document) |
| `When was Python first released?` | same passage |
| `quantum chromodynamics gluon` | "No sufficiently relevant information was found in the uploaded document." |

## 13. Limitations

BM25 is lexical, so it can struggle with:

* **Synonyms**: "automobile" will not match a passage that only says "car".
* **Paraphrasing**: the question's words must appear in the document.
* **Semantic similarity**: related concepts with different words are not connected.
* **Reasoning questions**: "Why did X happen?" needs understanding; BM25 only finds passages sharing the words.
* **Answers spread across distant sections**: each chunk is scored on its own.
* Scanned/image-only PDFs need OCR (not included). DOCX files have no page numbers, so the nearest heading is shown.
* The relevance threshold is an absolute BM25 score. Scores depend on document size (small documents produce lower
  scores), so tune the sidebar slider. The "no result" message shows the best score found to help you tune it.
* Tables, headers/footers and multi-column PDFs may extract in an odd order.

## 14. Future improvements

* Optional OCR module for scanned PDFs (as a separate, clearly-labelled add-on).
* Query expansion with a hand-made synonym list (still lexical).
* Support more formats (TXT, HTML, EPUB): write a loader and add it to `LOADERS`.
* Multiple documents in one index, with per-document filtering.
* Show the highest-scoring sentence inside each passage.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fitz'` or `'pymupdf'` | `pip install PyMuPDF` |
| `fitz` imports but `fitz.open` is missing / weird errors | An unrelated package named `fitz` is installed: `pip uninstall fitz` then `pip install --force-reinstall PyMuPDF` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root (`document_qa_bm25/`) |
| "looks scanned or image-only" | The PDF has no text layer. Use an OCR tool first (e.g. export a searchable PDF) |
| "No sufficiently relevant information..." | Use words that appear in the document; lower the threshold slider; use a smaller chunk size |
| Old results after changing chunk size | Click **Build / Rebuild index** (chunk settings apply at build time) |
| `streamlit: command not found` | Activate the virtual environment, or use `python -m streamlit run app.py` |
| Large PDF is slow | Increase chunk size; the limit is 50 MB per file |
