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
) -> Optional[tuple[str, list[list], int, list[dict[str, Any]]]]:
    """Extract text, tables, page count, and word-level bounding boxes from a PDF.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of a PDF file (e.g. from an uploaded file or read from disk).

    Returns
    -------
    tuple[str, list[list], int, list[dict[str, Any]]]
        A 4-tuple of:
        - ``full_text``     – all page text joined with newlines.
        - ``tables``        – flat list of tables found across all pages; each
                              table is a ``list[list]`` where inner lists are rows.
        - ``page_count``    – total number of pages in the document.
        - ``spatial_words`` – list of word bounding boxes with keys:
                              ``text``, ``x0``, ``x1``, ``top``, ``bottom``, ``page_idx``.

    Returns ``None`` if the PDF is encrypted / password-protected.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count: int = len(pdf.pages)

            text_parts: list[str] = []
            tables: list[list] = []
            spatial_words: list[dict[str, Any]] = []

            for page_idx, page in enumerate(pdf.pages):
                # --- text ---------------------------------------------------
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

                # --- tables -------------------------------------------------
                for table in page.extract_tables():
                    if table:  # skip empty table objects
                        tables.append(table)

                # --- spatial words with 2D coordinates ----------------------
                try:
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True,
                    )
                    for w in words:
                        spatial_words.append({
                            "text": w["text"],
                            "x0": float(w.get("x0", 0.0)),
                            "x1": float(w.get("x1", 0.0)),
                            "top": float(w.get("top", 0.0)),
                            "bottom": float(w.get("bottom", 0.0)),
                            "page_idx": page_idx,
                        })
                except Exception:
                    logger.debug("extract_pdf_content: could not extract spatial words on page %d", page_idx)

            full_text = "\n".join(text_parts)
            return full_text, tables, page_count, spatial_words

    except PDFPasswordIncorrect:
        logger.warning("extract_pdf_content: PDF is encrypted – returning None.")
        return None
    except Exception:
        logger.exception("extract_pdf_content: unexpected error while parsing PDF.")
        raise

