"""
Document Q&A -- LITE HYBRID VERSION: BM25 + WordNet synonym expansion + local LLM (Ollama).

Same idea as the PyTorch version, but with NO PyTorch, NO sentence-transformers, and NO
compiled ML library at all -- just plain Python. This avoids Windows "Application Control"
policies blocking PyTorch's DLLs.

    PDF/DOCX -> Chunking -> [BM25 on your exact words]        ---\
                         -> [BM25 on WordNet synonyms of your words] --> Hybrid ranking -> LLM -> Answer

"Semantic search" here means: for each word in your question, look up its synonyms in
WordNet (a free English thesaurus) and also search for those. It is still keyword matching
under the hood (not true meaning-based embeddings), but it directly solves "related words
should also match" without any heavy/blocked dependency.

Install (light, no PyTorch):
    pip install streamlit PyMuPDF python-docx rank-bm25 nltk requests numpy

First-time setup (downloads a small dictionary file, a few MB, run once):
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Install Ollama (free, optional, for the written answer): https://ollama.com/download
    ollama pull llama3.2:1b

Run:
    python -m streamlit run hybrid_app_lite.py
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


# --------------------------------- synonyms --------------------------------- #
_WORDNET_READY = None  # None = not checked yet, True/False after the first check


def wordnet_available():
    """Check once whether nltk + the WordNet data are usable. Never raises."""
    global _WORDNET_READY
    if _WORDNET_READY is None:
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")  # touches the data files; raises if not downloaded
            _WORDNET_READY = True
        except Exception:
            _WORDNET_READY = False
    return _WORDNET_READY


def get_synonyms(word):
    """Single-word synonyms of `word` from WordNet, stemmed, excluding the word itself."""
    if not wordnet_available():
        return set()
    from nltk.corpus import wordnet as wn
    found = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if " " not in name and name.lower() != word:
                found.add(stem(name.lower()))
    return found


def expand_terms(terms):
    """Return the set of synonym stems for a list of already-tokenized query terms."""
    expansion = set()
    for term in terms:
        expansion |= get_synonyms(term)
    return expansion - set(terms)


# ---------------------------------- index ----------------------------------- #
def build_index(pages, source, chunk_size, overlap, use_semantic=True):
    """Clean -> chunk (never across pages) -> BM25 tokens. `use_semantic` is unused here
    (kept for the same call signature as the embeddings version); synonym expansion happens
    per-question in hybrid_search instead, since it depends on the words actually asked."""
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

    return {"bm25": bm25, "chunks": chunks, "tokens": tokenized, "bigrams": bigrams, "vocab": set(doc_freq)}



# ------------------------------ definition finder ---------------------------- #
_DEFINITION_PREFIX = re.compile(
    r"^(what\s+(is|are)|define|explain|describe|what\s+does)\s+", re.IGNORECASE
)
_DEFINITION_SUFFIX = re.compile(r"\s+mean\??$", re.IGNORECASE)


def split_sentences(text):
    return [x.strip() for x in SENTENCE_SPLIT.split(text) if x.strip()] or [text]


def question_subject(question):
    """'What is deep learning?' -> 'deep learning'. Empty string if the question isn't this shape."""
    q = question.strip().rstrip("?.! ")
    stripped = _DEFINITION_PREFIX.sub("", q)
    if stripped == q:  # no "what is / define / explain" prefix found
        return ""
    stripped = _DEFINITION_SUFFIX.sub("", stripped).strip()
    return stripped if 1 <= len(stripped.split()) <= 6 else ""


_COPULA_STEMS = {"is", "are", stem("refers"), stem("means"), stem("defined")}
_NEGATOR_STEMS = {"not", stem("hardly"), stem("rarely"), "n't"}


def definition_bonus_and_sentence(chunk_text, subject_terms):
    """If the stemmed subject words appear together, immediately followed by a (non-negated)
    defining verb, in some sentence of this chunk, return (bonus, that_sentence).

    Matching is done on TOKENS (stemmed, whitespace-insensitive), not on the raw text, because
    real-world PDF extraction often inserts irregular spacing or line breaks that would break a
    literal substring search even though the sentence is really there.
    """
    if not subject_terms:
        return 0.0, None
    n = len(subject_terms)
    for sentence in split_sentences(chunk_text):
        words = TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", sentence).lower())
        tokens = [stem(w) for w in words]
        for i in range(len(tokens) - n):
            if tokens[i:i + n] != subject_terms:
                continue
            after = tokens[i + n]
            if after not in _COPULA_STEMS:
                continue
            following = tokens[i + n + 1] if i + n + 1 < len(tokens) else ""
            if following in _NEGATOR_STEMS:
                continue
            return 3.0, sentence  # 3.0 comfortably outranks a normal 0..~1.5 combined score
    return 0.0, None


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
    """Combine two BM25 passes: your exact words, and their WordNet synonyms.

    exact_norm     = BM25 score using your exact words / best such score   (0..1)
    related_norm   = BM25 score using ONLY the synonym words / best such score   (0..1)
    combined       = (1 - semantic_weight) * exact_norm + semantic_weight * related_norm

    semantic_weight = 0   -> exact words only (plain BM25)
    semantic_weight = 1   -> related words only
    1 is rarely useful alone; the point of blending is that a passage using different but
    related words can still surface even when your exact words are missing from it.
    """
    if index is None:
        raise ValueError("Please upload a document and build the index first.")
    if not question or not question.strip():
        raise ValueError("Please type a question.")
    terms = tokenize(question)
    if not terms:
        raise ValueError("Your question has no searchable keywords. Include specific words.")

    exact_scores = np.asarray([float(x) for x in index["bm25"].get_scores(terms)])
    best_bm25 = float(exact_scores.max()) if len(exact_scores) else 0.0
    exact_norm = _normalize(exact_scores)

    related_terms = sorted(expand_terms(terms)) if semantic_weight > 0 else []
    if related_terms:
        related_scores = np.asarray([float(x) for x in index["bm25"].get_scores(related_terms)])
    else:
        related_scores = np.zeros(len(exact_scores))
    related_norm = _normalize(related_scores)

    query_bigrams = set(zip(terms, terms[1:])) if phrase_boost else set()
    bonuses = np.zeros(len(exact_scores))
    for i in range(len(exact_scores)):
        if exact_scores[i] > 0 and query_bigrams:
            bonuses[i] = 0.1 * min(len(query_bigrams & index["bigrams"][i]), 3)

    subject_terms = tokenize(question_subject(question))
    def_bonuses, def_sentences = [], []
    for chunk in index["chunks"]:
        bonus, sentence = definition_bonus_and_sentence(chunk["text"], subject_terms)
        def_bonuses.append(bonus)
        def_sentences.append(sentence)
    def_bonuses = np.asarray(def_bonuses)

    w = semantic_weight
    combined = (1 - w) * exact_norm + w * related_norm + bonuses + def_bonuses
    order = np.argsort(-combined, kind="stable")

    results, picked = [], []
    for i in order:
        i = int(i)
        if len(results) >= top_k or combined[i] <= 0:
            break
        if combined[i] < min_score and def_bonuses[i] == 0:
            continue  # a definition match always counts, even below the usual threshold
        tokens = set(index["tokens"][i])
        if def_bonuses[i] == 0 and any(
            len(tokens & other) / max(len(tokens | other), 1) > 0.6 for other in picked
        ):
            continue  # near-duplicate of a better passage (definition hits are never skipped this way)
        picked.append(tokens)
        chunk = index["chunks"][i]
        matched_exact = sorted(set(terms) & tokens)
        matched_related = sorted(set(related_terms) & tokens)
        is_definition = def_bonuses[i] > 0
        results.append({**chunk, "rank": len(results) + 1, "combined": float(combined[i]),
                        "bm25_score": float(exact_scores[i]), "semantic_score": float(related_scores[i]),
                        "matched": matched_exact, "matched_related": matched_related,
                        "is_definition": is_definition,
                        "best_sentence": def_sentences[i] if is_definition else best_sentence(chunk["text"], terms)})
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


def generate_general_knowledge_answer(question, model):
    """Ask the local LLM to answer from its OWN knowledge (not the document).

    Used only as a fallback when nothing relevant was found in the document. The prompt
    explicitly tells the model this, and the caller must label the result to the user as
    general knowledge -- never present it as if it came from the uploaded document.
    """
    import requests

    prompt = (
        "The user's document does not contain information to answer this question. "
        "Answer it using your own general knowledge instead. Be concise, and if you are "
        "not confident in the answer, say so.\n\n"
        f"Question: {question}\nAnswer:"
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
            related_note = f", related-word score {r['semantic_score']:.2f}" if r.get("matched_related") else ""
            tag = "📖 Definition found · " if r.get("is_definition") else ""
            title = f"{tag}#{r['rank']} · {where} · combined {r['combined']:.2f} (exact-word BM25 {r['bm25_score']:.2f}{related_note})"
        with st.expander(title, expanded=r["rank"] == 1):
            st.markdown(show(r["text"], r), unsafe_allow_html=True)
            details = f"**Source:** {r['source']}  \n**Location:** {where}"
            if not exact:
                details += f"  \n**Matched your words:** {', '.join(r['matched']) or 'none'}"
                if r.get("matched_related"):
                    details += f"  \n**Matched related words (via WordNet):** {', '.join(r['matched_related'])}"
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
        wn_ready = wordnet_available()
        if wn_ready:
            st.caption("✔ WordNet synonym lookup ready.")
        else:
            st.warning("WordNet data not found. Related-word matching is off; run the first-time "
                       "setup command from the top of hybrid_app_lite.py, then restart the app.")
        semantic_weight = st.slider(
            "Related-word weight", 0.0, 1.0, 0.5 if wn_ready else 0.0, 0.05,
            disabled=not wn_ready,
            help="0 = your exact words only. 1 = only words related via WordNet. "
                 "0.5 blends both, so a passage using different-but-related wording can still surface.",
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
        allow_general_knowledge = st.checkbox(
            "If nothing relevant is found, answer from the LLM's general knowledge instead",
            value=True,
            help="When ON, a question the document can't answer still gets an answer, clearly "
                 "labeled as general knowledge rather than from your document. When OFF, the app "
                 "only ever answers from the document and says so when it can't.",
        )

        if st.button("Build / Rebuild index", type="primary", use_container_width=True):
            if uploaded is None:
                st.error("Please upload a PDF or DOCX file first.")
            else:
                try:
                    with st.spinner("Reading document and building the BM25 index..."):
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

    st.title("🔎 Hybrid Document Q&A (lite, no PyTorch)")
    st.markdown(
        "PDF/DOCX → chunks → **BM25** (your exact words) + **WordNet synonyms** (related words), "
        "blended into one ranking → optionally sent to a **local LLM** for a written answer. "
        "No PyTorch, no GPU, no big downloads."
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
                return
            if use_llm and llm_model and allow_general_knowledge:
                with st.spinner("Not in the document -- asking the LLM's general knowledge..."):
                    answer = generate_general_knowledge_answer(question, llm_model)
                st.warning("⚠️ Not found in your document. This answer is from the LLM's "
                          "general knowledge instead, and has NOT been checked against your document:")
                if answer:
                    st.markdown(answer)
                else:
                    st.error("Could not reach Ollama to generate a general-knowledge answer either.")
            else:
                st.warning("No relevant information was found in the uploaded document, "
                           "even with related-word matching. Try different wording, or turn on "
                           "the general-knowledge fallback in the sidebar.")
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
            st.markdown(f"- **Keywords NOT found (a related word may still match):** {', '.join(missing) or 'none'}")
            st.markdown(f"- **Common words ignored:** {', '.join(ignored) or 'none'}")
            st.markdown(f"- **Related-word weight used:** {semantic_weight:.2f}")


if __name__ == "__main__":
    main()
