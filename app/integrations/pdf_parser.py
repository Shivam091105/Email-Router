"""
PDF attachment text extraction.

Deliberately simple: pypdf's text extraction, page by page, joined with
blank lines. No OCR (scanned/image-only PDFs will yield empty or near-empty
text) — that's an explicit, documented future improvement, not a bug.
"""

import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    pass


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts and concatenates text from every page of a PDF.

    Returns an empty string (not an exception) for a PDF that parses fine
    but simply contains no extractable text (e.g. a scanned image) — that's
    a normal, expected outcome. Raises PDFExtractionError only when the
    bytes aren't a readable PDF at all, since that's a genuine failure the
    caller needs to know about rather than silently treat as "no text."
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as exc:
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from PDF page %d: %s", i, exc)
            pages_text.append("")

    return "\n\n".join(t for t in pages_text if t).strip()
