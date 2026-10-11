import pytest

from src.document_loader import load_document
from src.utils import (
    EmptyDocumentError,
    ExtractionError,
    ScannedPdfError,
    UnsupportedFileError,
)

LINE_1 = "Python was created by Guido van Rossum in 1991."
LINE_3 = "The Eiffel Tower is located in Paris and opened in 1889."


def test_pdf_extraction_keeps_page_numbers(make_pdf):
    data = make_pdf([LINE_1, "", LINE_3])  # page 2 is blank
    doc = load_document("book.pdf", data)
    assert doc["name"] == "book.pdf"
    assert doc["total_pages"] == 3
    assert [p["page"] for p in doc["pages"]] == [1, 3]  # blank page skipped, numbers preserved
    assert "Guido" in doc["pages"][0]["text"]
    assert "Eiffel" in doc["pages"][1]["text"]


def test_scanned_pdf_raises_clear_error(make_pdf):
    data = make_pdf(["", "", ""])  # no text layer at all
    with pytest.raises(ScannedPdfError) as error:
        load_document("scan.pdf", data)
    assert "OCR" in str(error.value)


def test_corrupted_pdf_is_handled(make_pdf):
    with pytest.raises(ExtractionError):
        load_document("broken.pdf", b"this is definitely not a pdf")


def test_docx_extraction_preserves_sections(make_docx):
    data = make_docx([("Introduction", "Python is popular."), ("History", "Guido created it.")])
    doc = load_document("notes.docx", data)
    assert doc["name"] == "notes.docx"
    assert doc["total_pages"] is None
    assert [p["section"] for p in doc["pages"]] == ["Introduction", "History"]
    assert doc["pages"][1]["text"] == "Guido created it."


def test_empty_docx_raises(make_docx):
    with pytest.raises(EmptyDocumentError):
        load_document("empty.docx", make_docx([]))


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_document("notes.txt", b"hello")


def test_empty_file_bytes():
    with pytest.raises(EmptyDocumentError):
        load_document("empty.pdf", b"")
