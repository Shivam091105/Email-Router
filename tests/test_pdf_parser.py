"""
PDF parser tests.

Uses pypdf itself (PdfWriter) to build minimal real PDFs in memory, so
these tests exercise the actual pypdf read path rather than mocking it —
no network, no test fixture files needed on disk.
"""

import io

import pytest
from pypdf import PdfWriter

from app.integrations.pdf_parser import PDFExtractionError, extract_text_from_pdf


def _build_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_text_from_blank_pdf_returns_empty_string():
    """A structurally valid PDF with no text content (e.g. a scanned image)
    should return an empty string, not raise — this is the documented
    no-OCR limitation, not an error condition."""
    pdf_bytes = _build_blank_pdf()
    text = extract_text_from_pdf(pdf_bytes)
    assert text == ""


def test_extract_text_from_garbage_bytes_raises():
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(b"this is not a pdf at all")


def test_extract_text_handles_per_page_failure_gracefully(monkeypatch):
    """
    If one page's text extraction throws, we should still return whatever
    the other pages produced, not blow up the whole request.
    """
    pdf_bytes = _build_blank_pdf()

    original_extract = None

    def broken_extract_text(self, *args, **kwargs):
        raise RuntimeError("simulated extraction failure")

    from pypdf._page import PageObject

    monkeypatch.setattr(PageObject, "extract_text", broken_extract_text)

    text = extract_text_from_pdf(pdf_bytes)
    assert text == ""  # failed page contributes nothing, but no exception propagates
