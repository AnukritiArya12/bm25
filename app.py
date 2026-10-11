"""
Document Q&A with BM25 -- SINGLE FILE VERSION.

Upload a PDF or DOCX, ask a question, get the most relevant passages using
BM25 keyword search. No RAG, no LLM, no embeddings, no vector database.

Install:  pip install streamlit PyMuPDF python-docx rank-bm25
Run:      python -m streamlit run bm25_app.py
"""
import html
import io
import re
import unicodedata

from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or but if then than that this these those
    it its as is are was were be been being am do does did has have had not no i me my we our you
    your he she they them their there here also so such what which who whom whose when where why
    how can could will would should may might must about into over tell explain describe define
    give please""".split()
)
TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)+|\w+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


# ----------------------------- text processing ----------------------------- #
def tokenize(text):
    """Lowercase, split into words, drop stopwords. Used for chunks AND questions."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [t for t in TOKEN_PATTERN.findall(text) if t not in STOPWORDS]


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def highlight(text, terms):
    """HTML-safe text with matched query words wrapped in <mark>."""
    out, last = [], 0
    for m in TOKEN_PATTERN.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        word = html.escape(m.group(0))
        matched = unicodedata.normalize("NFKC", m.group(0)).lower() in terms
        out.append(f"<mark>{word}</mark>" if matched else word)
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


# --------------------------------- loading --------------------------------- #
def load_document(file_name, data):
    """Return (pages, total_pages). pages = [{'page': int|None, 'section': str|None, 'text': str}]."""
    name = file_name.lower()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if name.endswith(".pdf"):
        return _load_pdf(data)
    if name.endswith(".docx"):
        return _load_docx(data)
    raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")


def _load_pdf(data):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    pages = []
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("This PDF is password-protected.")
            total = pdf.page_count
            for i in range(total):
                text = pdf.load_page(i).get_text("text") or ""
                if text.strip():
                    pages.append({"page": i + 1, "section": None, "text": text})
    except ValueError:
        raise
    except Exception:
        raise ValueError("Could not read this PDF. It may be corrupted.")
    if total == 0:
        raise ValueError("This PDF has no pages.")
    if sum(len(p["text"].strip()) for p in pages) / total < 20:
        raise ValueError(
            "This PDF looks scanned or image-only, so no text could be extracted. "
            "OCR would be required (not included in this project)."
        )
    return pages, total


def _load_docx(data):
    from docx import Document

    try:
        document = Document(io.BytesIO(data))
    except Exception:
        raise ValueError("Could not read this DOCX file. It may be corrupted.")
    units, heading, buffer = [], None, []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            units.append({"page": None, "section": heading, "text": text})
        buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = getattr(paragraph.style, "name", "") or ""
        if style.startswith("Heading") or style == "Title":
            flush()
            heading = text
        else:
            buffer.append(text)
    flush()
    if not units:
        raise ValueError("This DOCX file contains no readable text.")
    return units, None


# --------------------------------- chunking -------------------------------- #
def chunk_text(text, chunk_size=120, overlap=30):
    """Sentence-aware chunks of at most chunk_size words with `overlap` words carried over."""
    if not 20 <= chunk_size <= 2000:
        raise ValueError("Chunk size must be between 20 and 2000 words.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be at least 0 and smaller than the chunk size.")
    units = []
    for sentence in (s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()):
        words = sentence.split()
        if len(words) > chunk_size:
            units += [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        else:
            units.append(sentence)
    chunks, current, count = [], [], 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > chunk_size:
            chunks.append(" ".join(current))
            carry, carried = [], 0
            for prev in reversed(current):
                m = len(prev.split())
                if carried + m > overlap:
                    break
                carry.insert(0, prev)
                carried += m
            if carried + n > chunk_size:
                carry, carried = [], 0
            current, count = carry, carried
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_index(pages, source, chunk_size, overlap):
    """Clean -> chunk (never across pages) -> tokenize -> BM25. Returns a dict."""
    chunks, tokenized = [], []
    for page in pages:
        for text in chunk_text(clean_text(page["text"]), chunk_size, overlap):
            tokens = tokenize(text)
            if tokens:
                chunks.append({"chunk_id": len(chunks), "text": text, "page": page["page"],
                               "section": page["section"], "source": source})
                tokenized.append(tokens)
    if not chunks:
        raise ValueError("The document has no searchable text.")
    return {"bm25": BM25Okapi(tokenized), "chunks": chunks, "tokens": tokenized}


# --------------------------------- searching ------------------------------- #
def search(index, question, top_k=3, min_score=1.0):
    """BM25 ranking. Returns (results, best_score, query_terms)."""
    if index is None:
        raise ValueError("Please upload a document and build the index first.")
    if not question or not question.strip():
        raise ValueError("Please type a question.")
    terms = tokenize(question)
    if not terms:
        raise ValueError("Your question has no searchable keywords. Include specific words.")
    scores = [float(s) for s in index["bm25"].get_scores(terms)]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    results = []
    for i in order[:top_k]:
        if scores[i] <= 0 or scores[i] < min_score:
            break
        chunk = index["chunks"][i]
        results.append({**chunk, "rank": len(results) + 1, "score": scores[i],
                        "matched": sorted(set(terms) & set(index["tokens"][i]))})
    return results, (scores[order[0]] if order else 0.0), terms


# ------------------------------------ UI ----------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="Document Q&A with BM25", page_icon="🔎", layout="wide")

    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload a document", type=["pdf", "docx"])
        chunk_size = st.number_input("Chunk size (words)", 20, 2000, 120, step=10)
        overlap = st.number_input("Chunk overlap (words)", 0, 1999, 30, step=5)
        top_k = st.slider("Top-K passages", 1, 10, 3)
        min_score = st.slider("Relevance threshold (min BM25 score)", 0.0, 15.0, 1.0, 0.1)
        if st.button("Build / Rebuild index", type="primary", use_container_width=True):
            if uploaded is None:
                st.error("Please upload a PDF or DOCX file first.")
            else:
                try:
                    with st.spinner("Building BM25 index..."):
                        pages, total = load_document(uploaded.name, uploaded.getvalue())
                        st.session_state["index"] = build_index(pages, uploaded.name, int(chunk_size), int(overlap))
                        st.session_state["info"] = {"name": uploaded.name, "pages": len(pages), "total": total}
                    st.success("Index built.")
                except ValueError as error:
                    st.session_state.pop("index", None)
                    st.session_state.pop("info", None)
                    st.error(str(error))
                except Exception:
                    st.error("Something went wrong while processing the file. Please try another document.")

    st.title("🔎 Document Question Answering with BM25")
    st.markdown(
        "Answers are **retrieved directly from your document** using **BM25 keyword search**. "
        "No LLM, no embeddings, no RAG."
    )
    info, index = st.session_state.get("info"), st.session_state.get("index")
    if index is None:
        st.info("Index status: **not built**. Upload a document and click **Build / Rebuild index**.")
    else:
        st.success(f"Index status: **ready**. {info['name']} - {len(index['chunks'])} chunks.")

    with st.form("ask"):
        question = st.text_input("Ask a question about the document")
        submitted = st.form_submit_button("Search")
    if submitted:
        try:
            results, best, terms = search(index, question, int(top_k), float(min_score))
        except ValueError as error:
            st.warning(str(error))
        else:
            if not results:
                st.warning("No sufficiently relevant information was found in the uploaded document.")
                st.caption(f"Best BM25 score: {best:.2f} (threshold {min_score:.1f}). BM25 matches exact words, "
                           "so it can miss answers worded differently. Try other keywords or lower the threshold.")
            for r in results:
                where = f"Page {r['page']}" if r["page"] is not None else f"Section: {r['section'] or 'Document body'}"
                with st.expander(f"#{r['rank']} · {where} · BM25 score {r['score']:.2f}", expanded=r["rank"] == 1):
                    st.markdown(highlight(r["text"], set(r["matched"])), unsafe_allow_html=True)
                    st.markdown(f"**Source:** {r['source']}  \n**Location:** {where}  \n"
                                f"**BM25 score:** {r['score']:.2f}  \n**Matched keywords:** {', '.join(r['matched'])}")


if __name__ == "__main__":
    main()
