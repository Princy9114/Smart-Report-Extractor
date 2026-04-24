"""
pdf_utils.py
~~~~~~~~~~~~
Utilities for extracting text and table data from PDF files using pdfplumber.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import pdfplumber

# pdfplumber delegates PDF parsing to pdfminer; encrypted PDFs raise this.
from pdfminer.pdfdocument import PDFPasswordIncorrect

logger = logging.getLogger(__name__)


def extract_pdf_content(
    pdf_bytes: bytes,
) -> Optional[tuple[str, list[list], int]]:
    """Extract text, tables, and page count from a PDF supplied as raw bytes.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of a PDF file (e.g. from an uploaded file or read from disk).

    Returns
    -------
    tuple[str, list[list], int]
        A 3-tuple of:
        - ``full_text``  – all page text joined with newlines.
        - ``tables``     – flat list of tables found across all pages; each
                           table is a ``list[list]`` where inner lists are rows.
        - ``page_count`` – total number of pages in the document.

    Returns ``None`` if the PDF is encrypted / password-protected.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count: int = len(pdf.pages)

            text_parts: list[str] = []
            tables: list[list] = []

            for page in pdf.pages:
                # --- text ---------------------------------------------------
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

                # --- tables -------------------------------------------------
                for table in page.extract_tables():
                    if table:  # skip empty table objects
                        tables.append(table)

            full_text = "\n".join(text_parts)
            return full_text, tables, page_count

    except PDFPasswordIncorrect:
        logger.warning("extract_pdf_content: PDF is encrypted – returning None.")
        return None
    except Exception:
        logger.exception("extract_pdf_content: unexpected error while parsing PDF.")
        raise
