"""Split page text into sentence-aware, overlapping chunks that keep their metadata.

Why this strategy suits BM25
----------------------------
* BM25 scores whole "documents". If the whole book were one document, every
  query would match it and ranking would be useless; if chunks are tiny, a
  query's words are scattered across neighbours. A few sentences (about 100-150
  words) is the sweet spot: long enough to contain the query words together,
  short enough that a high score means "this passage is about it".
* Chunks are built from whole sentences, so a returned passage reads naturally.
* Chunks never cross page boundaries, so each chunk has exactly one page number.
* Overlap repeats the last sentence(s) of a chunk at the start of the next one,
  so an answer that straddles a boundary still appears intact in one chunk.
* Chunk length is roughly uniform, which works well with BM25's length
  normalisation (parameter b).
"""
from __future__ import annotations

import re
from typing import TypedDict

from src.utils import PageText, validate_chunk_settings


class Chunk(TypedDict):
    chunk_id: int
    text: str
    page: int | None
    section: str | None
    source: str


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (no external NLP library)."""
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _break_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Hard-split a sentence longer than chunk_size into word windows."""
    words = sentence.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Group sentences into chunks of at most `chunk_size` words.

    The next chunk starts with up to `overlap` words taken from the end of the
    previous chunk (whole sentences only).
    """
    validate_chunk_settings(chunk_size, overlap)

    units: list[str] = []
    for sentence in split_sentences(text):
        if _word_count(sentence) > chunk_size:
            units.extend(_break_long_sentence(sentence, chunk_size))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = _word_count(unit)
        if current and current_words + unit_words > chunk_size:
            chunks.append(" ".join(current))
            carried: list[str] = []
            carried_words = 0
            for previous in reversed(current):
                previous_words = _word_count(previous)
                if carried_words + previous_words > overlap:
                    break
                carried.insert(0, previous)
                carried_words += previous_words
            if carried_words + unit_words > chunk_size:  # keep chunks within the limit
                carried, carried_words = [], 0
            current, current_words = carried, carried_words
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[PageText], source: str, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Chunk every page separately so each chunk keeps its page/section/source."""
    validate_chunk_settings(chunk_size, overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page["text"], chunk_size, overlap):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": text,
                    "page": page["page"],
                    "section": page["section"],
                    "source": source,
                }
            )
    return chunks
