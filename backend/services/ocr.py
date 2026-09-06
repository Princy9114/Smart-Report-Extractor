"""
ocr.py
~~~~~~
OCR (Optical Character Recognition) service for extracting textual content
from image files (PNG, JPG, TIFF, WEBP, etc.) and scanned PDF documents.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global lazy OCR instance
_ocr_engine: Any = None


def _get_ocr_engine():
    """Lazily load the RapidOCR engine to avoid slow startup."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            logger.info("RapidOCR engine initialized successfully.")
        except Exception as exc:
            logger.warning("Failed to initialize RapidOCR: %s", exc)
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def extract_text_from_image(image: Image.Image | bytes | np.ndarray) -> str:
    """Perform OCR on a PIL Image, raw image bytes, or numpy array.

    Parameters
    ----------
    image:
        The input image to extract text from.

    Returns
    -------
    str:
        Extracted text formatted with preserved line breaks.
    """
    if isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    engine = _get_ocr_engine()
    if engine is None:
        logger.warning("OCR engine not available; returning empty text.")
        return ""

    try:
        results, _ = engine(img_np)
        if not results:
            return ""

        # results is a list of [dt_boxes, text, score]
        lines = [item[1] for item in results if len(item) > 1 and item[1]]
        return "\n".join(lines).strip()
    except Exception as exc:
        logger.error("OCR execution error on image: %s", exc)
        return ""


def extract_text_from_pdf_pages(pdf_bytes: bytes, scale: float = 2.0) -> str:
    """Render each page of a scanned PDF as high-resolution image and run OCR.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of the PDF file.
    scale:
        Resolution multiplier for page rendering (default: 2.0 for high fidelity).

    Returns
    -------
    str:
        Aggregated extracted text from all rendered pages.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; cannot render PDF pages for OCR.")
        return ""

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        page_texts: list[str] = []

        for idx, page in enumerate(doc):
            # Render page to high-res PIL image
            rendered_image = page.render(scale=scale).to_pil()
            page_text = extract_text_from_image(rendered_image)
            if page_text:
                page_texts.append(page_text)
            logger.debug("Scanned PDF OCR page %d extracted (%d chars)", idx + 1, len(page_text))

        return "\n\n".join(page_texts).strip()
    except Exception as exc:
        logger.error("Failed to render and OCR PDF pages: %s", exc)
        return ""


def is_scanned_pdf(extracted_text: str, page_count: int = 1) -> bool:
    """Determine whether a PDF is scanned based on extracted character density.

    If a PDF has less than an average of ~30 characters per page, it is very likely
    a scanned document containing raster images rather than selectable text.
    """
    cleaned = extracted_text.strip()
    threshold = max(30, page_count * 25)
    return len(cleaned) < threshold
