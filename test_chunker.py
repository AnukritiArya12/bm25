import pytest

from src.chunker import chunk_pages, chunk_text, split_sentences
from src.utils import ConfigError

TEXT = " ".join(f"Sentence number {i} talks about topic {i} in some detail." for i in range(1, 13))


def test_split_sentences():
    assert split_sentences("First one. Second one? Third one!") == ["First one.", "Second one?", "Third one!"]


def test_chunks_respect_size_limit():
    chunks = chunk_text(TEXT, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 30 for chunk in chunks)


def test_overlap_repeats_sentences_between_chunks():
    chunks = chunk_text(TEXT, chunk_size=30, overlap=10)
    first = set(split_sentences(chunks[0]))
    second = set(split_sentences(chunks[1]))
    assert first & second  # at least one shared sentence


def test_no_overlap_means_no_shared_sentences():
    chunks = chunk_text(TEXT, chunk_size=30, overlap=0)
    assert not set(split_sentences(chunks[0])) & set(split_sentences(chunks[1]))


def test_very_long_sentence_is_split():
    long_sentence = " ".join(["word"] * 100) + "."
    chunks = chunk_text(long_sentence, chunk_size=40, overlap=0)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 40 for c in chunks)


def test_chunk_pages_preserve_metadata(sample_pages):
    chunks = chunk_pages(sample_pages, source="book.pdf", chunk_size=40, overlap=10)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c["source"] == "book.pdf" for c in chunks)
    assert {c["page"] for c in chunks} == {1, 2, 3, 4, 5, 6}
    guido_chunk = next(c for c in chunks if "Guido" in c["text"])
    assert guido_chunk["page"] == 1


def test_chunks_never_cross_pages():
    pages = [
        {"page": 1, "section": None, "text": "Alpha beta gamma. Delta epsilon zeta."},
        {"page": 2, "section": None, "text": "Eta theta iota. Kappa lambda mu."},
    ]
    chunks = chunk_pages(pages, "x.pdf", chunk_size=100, overlap=10)
    assert len(chunks) == 2
    assert "Eta" not in chunks[0]["text"]


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 100), (100, -1), (5000, 0)])
def test_invalid_settings_rejected(size, overlap):
    with pytest.raises(ConfigError):
        chunk_text("Some text here.", size, overlap)
