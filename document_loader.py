"""Read PDF and DOCX files into pages/sections with metadata.

To support a new format later, write a `load_xxx(file_name, data)` function that
returns a LoadedDocument and register it in LOADERS at the bottom of this file.
"""
from __future__ import annotations

import io
import logging
from typing import Callable

try:  # newer PyMuPDF versions expose the 'pymupdf' name (avoids clashes with the unrelated 'fitz' package)
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from docx import Document

from src.utils import (
    DocumentQAError,
    EmptyDocumentError,
    ExtractionError,
    LoadedDocument,
    PageText,
    ScannedPdfError,
    validate_file,
)

logger = logging.getLogger(__name__)

# If a PDF yields fewer characters than this per page on average, we treat it as scanned.
MIN_AVG_CHARS_PER_PAGE = 20


def load_pdf(file_name: str, data: bytes) -> LoadedDocument:
    """Extract text page by page, keeping the original 1-based page numbers."""
    pages: list[PageText] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ExtractionError("This PDF is password-protected. Please remove the password first.")
            total_pages = pdf.page_count
            for index in range(total_pages):
                try:
                    text = pdf.load_page(index).get_text("text") or ""
                except Exception:  # one bad page must not break the whole document
                    logger.warning("Could not extract page %d of %s", index + 1, file_name, exc_info=True)
                    text = ""
                if text.strip():
                    pages.append({"page": index + 1, "section": None, "text": text})
    except DocumentQAError:
        raise
    except Exception as exc:
        logger.exception("PDF parsing failed for %s", file_name)
        raise ExtractionError("Could not read this PDF. The file may be corrupted or not a valid PDF.") from exc

    if total_pages == 0:
        raise EmptyDocumentError("This PDF has no pages.")

    total_chars = sum(len(page["text"].strip()) for page in pages)
    if total_chars / total_pages < MIN_AVG_CHARS_PER_PAGE:
        raise ScannedPdfError(
            "Almost no text could be extracted. This PDF looks scanned or image-only. "
            "OCR (optical character recognition) would be required to read it, and this "
            "project intentionally does not include OCR."
        )
    return {"name": file_name, "extension": ".pdf", "total_pages": total_pages, "pages": pages}


def load_docx(file_name: str, data: bytes) -> LoadedDocument:
    """Extract paragraphs and group them into sections using Heading styles.

    DOCX files have no fixed pages, so each unit carries the nearest heading
    (`section`) instead of a page number.
    """
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        logger.exception("DOCX parsing failed for %s", file_name)
        raise ExtractionError("Could not read this DOCX file. It may be corrupted or not a valid DOCX.") from exc

    units: list[PageText] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            units.append({"page": None, "section": current_heading, "text": text})
        buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = getattr(paragraph.style, "name", "") or ""
        if style_name.startswith("Heading") or style_name == "Title":
            flush()
            current_heading = text
        else:
            buffer.append(text)
    flush()

    if not units:
        raise EmptyDocumentError("This DOCX file contains no readable text.")
    return {"name": file_name, "extension": ".docx", "total_pages": None, "pages": units}


LOADERS: dict[str, Callable[[str, bytes], LoadedDocument]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(file_name: str, data: bytes) -> LoadedDocument:
    """Validate the file and dispatch to the loader for its extension."""
    extension = validate_file(file_name, len(data))
    return LOADERS[extension](file_name, data)
