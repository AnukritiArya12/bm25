"""Shared types, exceptions, validation and small text helpers.

Every exception derived from DocumentQAError has a message that is safe and
useful to show directly to an end user (no stack traces).
"""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from src.tokenizer import TOKEN_PATTERN, normalize_text

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")
MAX_FILE_SIZE_MB = 50


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DocumentQAError(Exception):
    """Base class for errors whose message can be shown to the user."""


class UnsupportedFileError(DocumentQAError):
    """The uploaded file type is not supported."""


class EmptyDocumentError(DocumentQAError):
    """The document contains no usable text."""


class ScannedPdfError(DocumentQAError):
    """The PDF looks scanned/image-only, so text extraction found (almost) nothing."""


class ExtractionError(DocumentQAError):
    """The file could not be read or parsed."""


class ConfigError(DocumentQAError):
    """A configuration value (chunk size, top-k, threshold...) is invalid."""


class EmptyQuestionError(DocumentQAError):
    """The question is empty or contains no searchable keywords."""


class IndexNotBuiltError(DocumentQAError):
    """A search was attempted before a BM25 index was built."""


# --------------------------------------------------------------------------- #
# Shared data shapes
# --------------------------------------------------------------------------- #
class PageText(TypedDict):
    """Text of one page (PDF) or one section (DOCX) with its location metadata."""

    page: int | None  # 1-based page number; None for DOCX (no fixed pages)
    section: str | None  # nearest heading (DOCX) or None
    text: str


class LoadedDocument(TypedDict):
    name: str
    extension: str
    total_pages: int | None  # None for DOCX
    pages: list[PageText]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_file(file_name: str, size_bytes: int) -> str:
    """Check name/size and return the lower-case extension (e.g. '.pdf')."""
    if not file_name:
        raise UnsupportedFileError("No file was provided.")
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise UnsupportedFileError(
            f"Unsupported file type '{extension or 'unknown'}'. Supported types: {supported}."
        )
    if size_bytes <= 0:
        raise EmptyDocumentError("The uploaded file is empty.")
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ExtractionError(f"The file is larger than the {MAX_FILE_SIZE_MB} MB limit.")
    return extension


def validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    """Chunk size and overlap are measured in words."""
    if not isinstance(chunk_size, int) or not isinstance(overlap, int):
        raise ConfigError("Chunk size and overlap must be whole numbers.")
    if not 20 <= chunk_size <= 2000:
        raise ConfigError("Chunk size must be between 20 and 2000 words.")
    if overlap < 0:
        raise ConfigError("Chunk overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ConfigError("Chunk overlap must be smaller than the chunk size.")


def validate_search_settings(top_k: int, min_score: float) -> None:
    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ConfigError("Top-K must be a whole number between 1 and 50.")
    if not isinstance(min_score, (int, float)) or min_score < 0:
        raise ConfigError("The relevance threshold cannot be negative.")


def validate_config(chunk_size: int, overlap: int, top_k: int, min_score: float) -> None:
    validate_chunk_settings(chunk_size, overlap)
    validate_search_settings(top_k, min_score)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalise extracted text without touching the actual wording.

    - Unicode NFKC (turns ligatures such as 'ﬁ' into 'fi')
    - re-joins words hyphenated across line breaks ('exam-\\nple' -> 'example')
    - removes control characters and collapses all whitespace to single spaces
    """
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)
    text = _CONTROL_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_location(page: int | None, section: str | None) -> str:
    """Human-readable location of a passage."""
    if page is not None:
        return f"Page {page}"
    if section:
        return f"Section: {section}"
    return "Document body"


def highlight_terms(text: str, terms: list[str] | set[str]) -> str:
    """Return HTML-safe text with query terms wrapped in <mark> tags."""
    wanted = set(terms)
    pieces: list[str] = []
    last_end = 0
    for match in TOKEN_PATTERN.finditer(text):
        pieces.append(html.escape(text[last_end:match.start()]))
        word = match.group(0)
        escaped = html.escape(word)
        pieces.append(f"<mark>{escaped}</mark>" if normalize_text(word) in wanted else escaped)
        last_end = match.end()
    pieces.append(html.escape(text[last_end:]))
    return "".join(pieces)
