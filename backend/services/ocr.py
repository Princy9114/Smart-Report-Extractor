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
from PIL import Image, ImageFilter, ImageOps

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


def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Preprocess image to maximize OCR recognition accuracy.

    1. Upscales low-resolution images so character strokes and small fonts are clear.
    2. Auto-contrast normalization to remove background shading/shadows.
    3. Subtle unsharp masking to enhance character edge definition.
    """
    try:
        w, h = pil_img.size
        # Upscale small scans/images if minimum dimension is low
        min_dim = min(w, h)
        if min_dim < 1000:
            scale_factor = max(1.5, 1200.0 / max(w, h))
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to grayscale for contrast enhancement
        gray = pil_img.convert("L")
        contrasted = ImageOps.autocontrast(gray, cutoff=1)

        # Apply subtle edge sharpening
        sharpened = contrasted.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        return sharpened.convert("RGB")
    except Exception as exc:
        logger.debug("Image preprocessing failed, using original: %s", exc)
        return pil_img.convert("RGB")


def extract_text_and_spatial_words_from_image(
    image: Image.Image | bytes | np.ndarray,
    page_idx: int = 0,
    coord_scale: float = 1.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Perform OCR on an image and extract both formatted text and 2D spatial bounding boxes.

    Parameters
    ----------
    image:
        Input image as PIL Image, raw image bytes, or numpy array.
    page_idx:
        Page index to attach to spatial words.
    coord_scale:
        Scale factor to convert image pixel coordinates back to target coordinate space.

    Returns
    -------
    tuple[str, list[dict[str, Any]]]
        - ``text``: Extracted text formatted with preserved line breaks.
        - ``spatial_words``: List of word bounding box dictionaries with keys:
          ``text``, ``x0``, ``x1``, ``top``, ``bottom``, ``page_idx``, ``confidence``.
    """
    if isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    orig_w, orig_h = pil_img.size
    preprocessed_img = preprocess_image_for_ocr(pil_img)
    prep_w, prep_h = preprocessed_img.size

    # Scaling ratios to map coordinates back to original coordinate system
    resize_ratio_x = prep_w / orig_w if orig_w else 1.0
    resize_ratio_y = prep_h / orig_h if orig_h else 1.0
    total_scale_x = resize_ratio_x * coord_scale
    total_scale_y = resize_ratio_y * coord_scale

    engine = _get_ocr_engine()
    if engine is None:
        logger.warning("OCR engine not available; returning empty result.")
        return "", []

    try:
        results, _ = engine(np.array(preprocessed_img))
        if not results:
            return "", []

        lines: list[str] = []
        spatial_words: list[dict[str, Any]] = []

        for item in results:
            if not item or len(item) < 2 or not item[1]:
                continue

            dt_boxes = item[0]  # [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            block_text = str(item[1]).strip()
            score = float(item[2]) if len(item) > 2 else 0.85

            if not block_text:
                continue

            lines.append(block_text)

            # Map bounding box coordinates back to target coordinate space
            raw_x0 = min(pt[0] for pt in dt_boxes) / total_scale_x
            raw_x1 = max(pt[0] for pt in dt_boxes) / total_scale_x
            raw_top = min(pt[1] for pt in dt_boxes) / total_scale_y
            raw_bottom = max(pt[1] for pt in dt_boxes) / total_scale_y

            # Subdivide multi-word blocks into word-level bounding boxes
            words_in_block = block_text.split()
            if len(words_in_block) <= 1:
                spatial_words.append({
                    "text": block_text,
                    "x0": round(raw_x0, 1),
                    "x1": round(raw_x1, 1),
                    "top": round(raw_top, 1),
                    "bottom": round(raw_bottom, 1),
                    "page_idx": page_idx,
                    "confidence": score,
                })
            else:
                total_chars = sum(len(w) for w in words_in_block) + (len(words_in_block) - 1)
                total_width = raw_x1 - raw_x0
                curr_x = raw_x0
                for w in words_in_block:
                    w_len = len(w)
                    w_width = (w_len / total_chars) * total_width if total_chars else (total_width / len(words_in_block))
                    spatial_words.append({
                        "text": w,
                        "x0": round(curr_x, 1),
                        "x1": round(curr_x + w_width, 1),
                        "top": round(raw_top, 1),
                        "bottom": round(raw_bottom, 1),
                        "page_idx": page_idx,
                        "confidence": score,
                    })
                    curr_x += w_width + (1.0 / total_chars) * total_width

        full_text = "\n".join(lines).strip()
        return full_text, spatial_words

    except Exception as exc:
        logger.error("OCR execution error on image: %s", exc)
        return "", []


def extract_text_from_image(image: Image.Image | bytes | np.ndarray) -> str:
    """Perform OCR on a PIL Image, raw image bytes, or numpy array.

    Returns extracted text formatted with preserved line breaks.
    """
    text, _ = extract_text_and_spatial_words_from_image(image)
    return text


def extract_text_and_spatial_words_from_pdf_pages(
    pdf_bytes: bytes,
    scale: float = 2.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Render each page of a scanned PDF as high-resolution image and run OCR with 2D coordinates.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of the PDF file.
    scale:
        Resolution multiplier for page rendering (default: 2.0 for high fidelity).

    Returns
    -------
    tuple[str, list[dict[str, Any]]]
        Aggregated extracted text and word-level bounding boxes across all rendered pages.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; cannot render PDF pages for OCR.")
        return "", []

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        page_texts: list[str] = []
        all_spatial_words: list[dict[str, Any]] = []

        for idx, page in enumerate(doc):
            rendered_image = page.render(scale=scale).to_pil()
            page_text, page_words = extract_text_and_spatial_words_from_image(
                rendered_image,
                page_idx=idx,
                coord_scale=scale,
            )
            if page_text:
                page_texts.append(page_text)
            if page_words:
                all_spatial_words.extend(page_words)
            logger.debug("Scanned PDF OCR page %d extracted (%d chars, %d word boxes)", idx + 1, len(page_text), len(page_words))

        full_text = "\n\n".join(page_texts).strip()
        return full_text, all_spatial_words
    except Exception as exc:
        logger.error("Failed to render and OCR PDF pages: %s", exc)
        return "", []


def extract_text_from_pdf_pages(pdf_bytes: bytes, scale: float = 2.0) -> str:
    """Render each page of a scanned PDF as high-resolution image and run OCR.

    Returns aggregated extracted text from all rendered pages.
    """
    text, _ = extract_text_and_spatial_words_from_pdf_pages(pdf_bytes, scale=scale)
    return text


def is_scanned_pdf(extracted_text: str, page_count: int = 1) -> bool:
    """Determine whether a PDF is scanned based on extracted character density.

    If a PDF has less than an average of ~30 characters per page, it is very likely
    a scanned document containing raster images rather than selectable text.
    """
    cleaned = extracted_text.strip()
    threshold = max(30, page_count * 25)
    return len(cleaned) < threshold

