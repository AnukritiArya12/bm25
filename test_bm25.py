import pytest

from src.retriever import build_search_index, index_document, search
from src.chunker import chunk_pages
from src.tokenizer import tokenize
from src.utils import (
    ConfigError,
    EmptyDocumentError,
    EmptyQuestionError,
    IndexNotBuiltError,
    clean_text,
    highlight_terms,
)


@pytest.fixture
def index(sample_document):
    return index_document(sample_document, chunk_size=40, overlap=10)


# ----------------------------- tokenization ----------------------------- #
def test_tokenize_lowercases_and_removes_stopwords():
    assert tokenize("Who created Python 3.11 and BM25?") == ["created", "python", "3.11", "bm25"]


def test_tokenize_splits_hyphens_and_normalizes_ligatures():
    assert tokenize("Machine-Learning") == ["machine", "learning"]
    assert tokenize("ﬁnd") == ["find"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ?!  ") == []


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("exam-\nple   text\nhere") == "example text here"


# ------------------------------- indexing ------------------------------- #
def test_index_has_one_bm25_entry_per_chunk(index):
    assert len(index.chunks) == len(index.tokenized_chunks) == 6
    assert index.bm25.corpus_size == 6


def test_empty_corpus_raises():
    with pytest.raises(EmptyDocumentError):
        build_search_index([])


def test_document_with_only_stopwords_raises():
    chunks = chunk_pages([{"page": 1, "section": None, "text": "The of and to a in."}], "x.pdf", 40, 5)
    with pytest.raises(EmptyDocumentError):
        build_search_index(chunks)


# -------------------------------- ranking -------------------------------- #
def test_who_created_python_returns_the_right_passage(index):
    response = search(index, "Who created Python?")
    top = response["results"][0]
    assert "Guido van Rossum" in top["text"]
    assert top["page"] == 1
    assert top["source"] == "sample.pdf"
    assert top["rank"] == 1
    assert top["score"] > 0
    assert set(top["matched_terms"]) == {"created", "python"}


@pytest.mark.parametrize(
    "question,expected_page",
    [
        ("Where is the Eiffel Tower?", 2),
        ("How do plants convert sunlight into glucose?", 3),
        ("What is BM25 used for?", 4),
        ("Which river flows through Egypt?", 6),
    ],
)
def test_ranking_and_page_metadata(index, question, expected_page):
    top = search(index, question)["results"][0]
    assert top["page"] == expected_page


def test_results_are_sorted_by_score_and_limited_by_top_k(index):
    response = search(index, "tower Paris Eiffel light energy gravity", top_k=3, min_score=0.0)
    scores = [r["score"] for r in response["results"]]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) <= 3
    assert [r["rank"] for r in response["results"]] == list(range(1, len(scores) + 1))


# --------------------------- no answer / errors --------------------------- #
def test_no_relevant_result(index):
    response = search(index, "quantum chromodynamics gluon")
    assert response["results"] == []
    assert response["best_score"] == 0.0


def test_threshold_filters_weak_matches(index):
    assert search(index, "Who created Python?", min_score=0.0)["results"]
    strict = search(index, "Who created Python?", min_score=100.0)
    assert strict["results"] == []
    assert strict["best_score"] > 0  # a match existed, it just was not strong enough


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question(index, question):
    with pytest.raises(EmptyQuestionError):
        search(index, question)


def test_question_with_only_stopwords(index):
    with pytest.raises(EmptyQuestionError):
        search(index, "what is the")


def test_search_before_index_exists():
    with pytest.raises(IndexNotBuiltError):
        search(None, "anything")


@pytest.mark.parametrize("top_k,min_score", [(0, 1.0), (100, 1.0), (3, -1.0)])
def test_invalid_search_settings(index, top_k, min_score):
    with pytest.raises(ConfigError):
        search(index, "python", top_k=top_k, min_score=min_score)


# ------------------------------ highlighting ------------------------------ #
def test_highlight_is_html_safe():
    html_text = highlight_terms("Python <b>rocks</b> & Python.", {"python"})
    assert html_text.count("<mark>Python</mark>") == 2
    assert "&lt;b&gt;" in html_text and "<b>" not in html_text
