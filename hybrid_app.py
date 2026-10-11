"""
Document Q&A -- HYBRID VERSION: BM25 + semantic search + local LLM (Ollama).

    PDF/DOCX -> Chunking -> [BM25]  ---\\
                         -> [Semantic] --> Hybrid ranking -> Relevant chunks -> LLM -> Answer

Everything here is free and runs on your own computer:
  - BM25:      rank-bm25 (keyword search, same as before)
  - Semantic:  sentence-transformers, model "all-MiniLM-L6-v2" (downloaded once, then offline)
  - LLM:       Ollama (a free local program) -- if it is not running, the app still works and
               shows the retrieved passages, it just skips the generated-answer step.

Install:
    pip install streamlit PyMuPDF python-docx rank-bm25 sentence-transformers numpy requests

Install Ollama (one-time, free): https://ollama.com/download
Then in a terminal:
    ollama pull llama3.2:1b        # small (~1.3GB) and fast; use "llama3.2" for a bigger, better model

Run:
    python -m streamlit run hybrid_app.py
"""
import html
import io
import math
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or but if then than that this these those
    it its as is are was were be been being am do does did has have had i me my we our you
    your he she they them their what which who whom whose when where why how about into
    tell explain describe define give please""".split()
)
TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)+|\w+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
OLLAMA_URL = "http://localhost:11434"


# ----------------------------- text processing ----------------------------- #
def stem(word):
    """Very light suffix stripping so 'networks'/'network', 'created'/'creates' match.
    Words containing digits (bm25, k1, 3.11) and very short words are left unchanged."""
    if len(word) <= 2 or any(c.isdigit() for c in word):
        return word
    for suffix, replacement, min_stem in (("ies", "y", 2), ("ing", "", 2), ("ed", "", 2),
                                          ("es", "", 3), ("s", "", 3)):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= min_stem:
            word = word[: -len(suffix)] + replacement
            break
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    return word


def tokenize(text):
    """Lowercase, split into words, drop stopwords, light stemming. Used for BM25 only."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [stem(t) for t in TOKEN_PATTERN.findall(text) if t not in STOPWORDS]


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
        matched = stem(unicodedata.normalize("NFKC", m.group(0)).lower()) in terms
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
def chunk_text(text, chunk_size=100, overlap=20):
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


# -------------------------------- embeddings -------------------------------- #
_EMBED_MODEL = None


def get_embed_model():
    """Load the sentence-transformers model once and re-use it (it is slow to load)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def embed_texts(texts):
    """Return an (N, D) float32 array of L2-normalized embeddings (so dot product = cosine similarity)."""
    model = get_embed_model()
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


# ---------------------------------- index ----------------------------------- #
def build_index(pages, source, chunk_size, overlap, use_semantic=True):
    """Clean -> chunk (never across pages) -> BM25 tokens -> (optional) embeddings."""
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

    bm25 = BM25Okapi(tokenized)
    # rank-bm25's default IDF goes NEGATIVE for words found in more than half the chunks, which makes
    # exact keywords look irrelevant in a small document. Use the always-positive (BM25+) form instead.
    doc_freq = {}
    for doc in tokenized:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    n_docs = len(tokenized)
    bm25.idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}
    bigrams = [set(zip(t, t[1:])) for t in tokenized]

    embeddings = embed_texts([c["text"] for c in chunks]) if use_semantic else None

    return {"bm25": bm25, "chunks": chunks, "tokens": tokenized, "bigrams": bigrams,
            "vocab": set(doc_freq), "embeddings": embeddings}


# --------------------------------- searching -------------------------------- #
def best_sentence(text, terms):
    sentences = [x.strip() for x in SENTENCE_SPLIT.split(text) if x.strip()] or [text]
    wanted = set(terms)
    return max(sentences, key=lambda x: len(wanted & set(tokenize(x))))


def explain_terms(index, question):
    words = TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", question).lower())
    ignored = [w for w in words if w in STOPWORDS]
    kept = [w for w in words if w not in STOPWORDS]
    found = [w for w in kept if stem(w) in index["vocab"]]
    missing = [w for w in kept if stem(w) not in index["vocab"]]
    return found, missing, ignored


def _normalize(values):
    values = np.asarray(values, dtype="float32")
    top = values.max() if len(values) else 0.0
    return values / top if top > 0 else values


def hybrid_search(index, question, top_k=3, min_score=0.0, semantic_weight=0.5, phrase_boost=True):
    """Combine BM25 (keyword) and cosine similarity (semantic, meaning-based) scores.

    bm25_norm      = BM25 score / best BM25 score in this search   (0..1, 0 if no keyword overlap at all)
    semantic_norm  = cosine similarity, already roughly 0..1 since embeddings are normalized
    combined       = (1 - semantic_weight) * bm25_norm + semantic_weight * semantic_norm

    semantic_weight = 0   -> behaves like pure BM25 (exact words only)
    semantic_weight = 1   -> behaves like pure semantic search (meaning only, e.g. related terms/synonyms)
    A passage still needs a positive combined score, and phrase_boost still applies only to the BM25 side.
    """
    if index is None:
        raise ValueError("Please upload a document and build the index first.")
    if not question or not question.strip():
        raise ValueError("Please type a question.")
    terms = tokenize(question)
    if not terms:
        raise ValueError("Your question has no searchable keywords. Include specific words.")

    bm25_scores = np.asarray([float(x) for x in index["bm25"].get_scores(terms)])
    best_bm25 = float(bm25_scores.max()) if len(bm25_scores) else 0.0
    bm25_norm = _normalize(bm25_scores)

    if index.get("embeddings") is not None:
        query_vec = embed_texts([question])[0]
        semantic_scores = index["embeddings"] @ query_vec  # cosine similarity, already normalized vectors
    else:
        semantic_scores = np.zeros(len(bm25_scores))
    semantic_norm = np.clip((semantic_scores + 1) / 2, 0, 1)  # cosine is roughly -1..1 -> 0..1

    query_bigrams = set(zip(terms, terms[1:])) if phrase_boost else set()
    bonuses = np.zeros(len(bm25_scores))
    for i in range(len(bm25_scores)):
        if bm25_scores[i] > 0 and query_bigrams:
            bonuses[i] = 0.1 * min(len(query_bigrams & index["bigrams"][i]), 3)

    w = semantic_weight
    combined = (1 - w) * bm25_norm + w * semantic_norm + bonuses
    order = np.argsort(-combined, kind="stable")

    results, picked = [], []
    for i in order:
        i = int(i)
        if len(results) >= top_k or combined[i] <= 0:
            break
        if combined[i] < min_score:
            continue
        tokens = set(index["tokens"][i])
        if any(len(tokens & other) / max(len(tokens | other), 1) > 0.6 for other in picked):
            continue  # near-duplicate of a better passage
        picked.append(tokens)
        chunk = index["chunks"][i]
        results.append({**chunk, "rank": len(results) + 1, "combined": float(combined[i]),
                        "bm25_score": float(bm25_scores[i]), "semantic_score": float(semantic_scores[i]),
                        "matched": sorted(set(terms) & tokens),
                        "best_sentence": best_sentence(chunk["text"], terms)})
    return results, best_bm25, terms


def exact_match(index, question, top_k=3):
    """Fallback: passages that literally contain the typed text (case-insensitive)."""
    needle = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question)).strip().lower().rstrip("?.! ")
    if not needle:
        return []
    results = []
    for chunk in index["chunks"]:
        if needle in chunk["text"].lower():
            sentences = [x.strip() for x in SENTENCE_SPLIT.split(chunk["text"]) if x.strip()] or [chunk["text"]]
            sentence = next((x for x in sentences if needle in x.lower()), chunk["text"])
            results.append({**chunk, "rank": len(results) + 1, "phrase": needle, "best_sentence": sentence})
            if len(results) >= top_k:
                break
    return results


def highlight_phrase(text, phrase):
    parts = re.split("(" + re.escape(phrase) + ")", text, flags=re.IGNORECASE)
    return "".join(f"<mark>{html.escape(p)}</mark>" if i % 2 else html.escape(p) for i, p in enumerate(parts))


# ----------------------------------- LLM ------------------------------------ #
def ollama_is_running():
    import requests
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).ok
    except Exception:
        return False


def ollama_models():
    import requests
    try:
        data = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def generate_answer(question, results, model):
    """Ask a local Ollama model to write a natural-language answer using ONLY the retrieved passages.

    Returns the answer text, or None if Ollama could not be reached (the caller should then
    fall back to showing the retrieved passages, which is the previous behavior).
    """
    import requests

    context = "\n\n".join(
        f"[Passage {r['rank']} - {'page ' + str(r['page']) if r['page'] is not None else r['source']}]\n{r['text']}"
        for r in results
    )
    prompt = (
        "Answer the question using ONLY the information in the context below. "
        "If the context does not contain the answer, say so clearly instead of guessing. "
        "Keep the answer concise and mention which passage number(s) you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip() or None
    except Exception:
        return None


# ------------------------------------ UI ------------------------------------ #
def render_passages(st, results, exact=False):
    def show(text, r):
        if r.get("phrase"):
            return highlight_phrase(text, r["phrase"])
        return highlight(text, set(r["matched"]))

    for r in results:
        where = f"Page {r['page']}" if r["page"] is not None else f"Section: {r['section'] or 'Document body'}"
        if exact:
            title = f"#{r['rank']} · {where} · exact text match"
        else:
            title = (f"#{r['rank']} · {where} · combined {r['combined']:.2f} "
                     f"(BM25 {r['bm25_score']:.2f}, semantic {r['semantic_score']:.2f})")
        with st.expander(title, expanded=r["rank"] == 1):
            st.markdown(show(r["text"], r), unsafe_allow_html=True)
            details = f"**Source:** {r['source']}  \n**Location:** {where}"
            if not exact:
                details += f"  \n**Matched keywords:** {', '.join(r['matched']) or 'none (semantic match only)'}"
            st.markdown(details)


def main():
    import streamlit as st

    st.set_page_config(page_title="Hybrid Document Q&A", page_icon="🔎", layout="wide")

    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload a document", type=["pdf", "docx"])
        chunk_size = st.number_input("Chunk size (words)", 20, 2000, 100, step=10)
        overlap = st.number_input("Chunk overlap (words)", 0, 1999, 20, step=5)
        top_k = st.slider("Top-K passages", 1, 10, 3)
        semantic_weight = st.slider(
            "Semantic weight", 0.0, 1.0, 0.5, 0.05,
            help="0 = pure BM25 (exact words only). 1 = pure semantic search (meaning/related terms only). "
                 "0.5 blends both, which is usually the best of both worlds.",
        )
        phrase_boost = st.checkbox("Boost exact phrase matches", value=True)

        st.divider()
        st.subheader("Local LLM (optional)")
        running = ollama_is_running()
        if running:
            models = ollama_models()
            st.success("Ollama detected.")
            llm_model = st.selectbox("Model", models) if models else None
            if not models:
                st.warning("No models pulled yet. Run e.g. `ollama pull llama3.2:1b` in a terminal.")
        else:
            st.info("Ollama not detected at localhost:11434. Generated answers will be skipped; "
                    "retrieved passages will still be shown. Install: https://ollama.com/download")
            llm_model = None
        use_llm = st.checkbox("Generate a natural-language answer with the local LLM", value=running and bool(llm_model := llm_model))

        if st.button("Build / Rebuild index", type="primary", use_container_width=True):
            if uploaded is None:
                st.error("Please upload a PDF or DOCX file first.")
            else:
                try:
                    with st.spinner("Reading document, chunking, and computing embeddings (first run downloads a small model)..."):
                        pages, total = load_document(uploaded.name, uploaded.getvalue())
                        st.session_state["index"] = build_index(pages, uploaded.name, int(chunk_size), int(overlap))
                        st.session_state["info"] = {"name": uploaded.name, "pages": len(pages), "total": total}
                    st.success("Index built.")
                except ValueError as error:
                    st.session_state.pop("index", None)
                    st.session_state.pop("info", None)
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Something went wrong while processing the file: {error}")

    st.title("🔎 Hybrid Document Q&A")
    st.markdown(
        "PDF/DOCX → chunks → **BM25** (exact keywords) + **semantic search** (meaning/related terms), "
        "blended into one ranking → optionally sent to a **local LLM** for a written answer."
    )
    info, index = st.session_state.get("info"), st.session_state.get("index")
    if index is None:
        st.info("Index status: **not built**. Upload a document and click **Build / Rebuild index**.")
    else:
        st.success(f"Index status: **ready**. {info['name']} - {len(index['chunks'])} chunks, "
                   f"{len(index['vocab'])} distinct keywords.")

    with st.form("ask"):
        question = st.text_input("Ask a question about the document")
        submitted = st.form_submit_button("Search")

    if submitted:
        try:
            results, best_bm25, _ = hybrid_search(index, question, int(top_k), 0.0, semantic_weight, phrase_boost)
        except ValueError as error:
            st.warning(str(error))
            return

        if not results:
            matches = exact_match(index, question, int(top_k))
            if matches:
                st.info("No ranked match, but your exact text appears in the document:")
                render_passages(st, matches, exact=True)
            else:
                st.warning("No relevant information was found in the uploaded document, "
                           "even with semantic search. Try different wording.")
            return

        if use_llm and llm_model:
            with st.spinner("Asking the local LLM..."):
                answer = generate_answer(question, results, llm_model)
            if answer:
                st.markdown("### Answer")
                st.markdown(answer)
                st.caption("Generated by your local Ollama model, using ONLY the retrieved passages below as context.")
            else:
                st.warning("Could not reach Ollama to generate an answer. Showing retrieved passages instead.")

        st.markdown("### Retrieved passages")
        render_passages(st, results)

        with st.expander("Why these results? (search details)"):
            found, missing, ignored = explain_terms(index, question)
            st.markdown(f"- **Keywords found in document:** {', '.join(found) or 'none'}")
            st.markdown(f"- **Keywords NOT found (semantic search may still catch these):** {', '.join(missing) or 'none'}")
            st.markdown(f"- **Common words ignored:** {', '.join(ignored) or 'none'}")
            st.markdown(f"- **Semantic weight used:** {semantic_weight:.2f}")


if __name__ == "__main__":
    main()
