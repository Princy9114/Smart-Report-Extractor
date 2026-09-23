"""
pipeline.py
~~~~~~~~~~~
Central orchestration service for extracting structured data from documents.

Workflow:
1. Parse PDF bytes or image bytes to text and 2D spatial word bounding boxes.
2. Detect document type using heuristic keyword scoring.
3. Run extraction layers (Layer 4 LLM first if available, followed by Layer 1 Spatial, Layer 2 NER, Layer 3 Regex).
4. Merge results using consensus reconciliation and compute confidence scoring.
5. Generate contextual document summary.
6. Export the final result to the requested format (JSON/CSV) as a streaming response.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi.responses import Response

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType
from backend.services import detector, exporter, merger, ocr, summarizer
from backend.services.layers import layer1_pdfplumber, layer2_spacy, layer3_regex, layer4_llm
from backend.utils import pdf_utils

logger = logging.getLogger(__name__)

ExtractionResult = dict[str, FieldResult]

_LLM_CONFIDENCE_THRESHOLD = 0.85
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}


def _is_image(filename: str, file_bytes: bytes) -> bool:
    """Check if the provided file is an image by extension or header."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in _IMAGE_EXTENSIONS:
        return True
    # Fallback to checking byte signatures
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n") or file_bytes.startswith(b"\xff\xd8\xff"):
        return True
    return False


async def run_pipeline(
    file_bytes: bytes,
    filename: str = "",
    output_format: str = "json",
) -> Response | dict[str, Any]:
    """Execute the full document processing pipeline with OCR and multi-layer extraction.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded PDF or image file.
    filename:
        Original filename to determine format / handling.
    output_format:
        Desired output format (e.g. "json", "csv").

    Returns
    -------
    Response
        The merged ExtractionResult exported in the requested format.
    """
    logger.info("==================== STARTING EXTRACTION PIPELINE ====================")
    logger.info("Document: '%s' | Requested Output Format: %s", filename or "unnamed", output_format)

    # 1. Extract raw text and tables (handling images, digital PDFs, and scanned PDFs)
    text: str = ""
    tables: list[list] = []
    spatial_words: list[dict[str, Any]] = []
    ocr_used: bool = False

    if _is_image(filename, file_bytes):
        logger.info("[OCR TRIGGERED] 📸 File '%s' is an image. Invoking RapidOCR engine with 2D spatial layout...", filename)
        ocr_result = ocr.extract_text_and_spatial_words_from_image(file_bytes)
        if isinstance(ocr_result, tuple):
            text, spatial_words = ocr_result
        else:
            text, spatial_words = ocr_result, []

        text = text or ""
        spatial_words = spatial_words or []
        page_count = 1
        ocr_used = True
        logger.info("[OCR COMPLETED] Extracted %d character(s) and %d spatial bounding box(es) from image.", len(text), len(spatial_words))
    else:
        pdf_data = pdf_utils.extract_pdf_content(file_bytes)
        if pdf_data is None:
            logger.warning("[PIPELINE ABORTED] PDF '%s' is encrypted or unreadable.", filename)
            return exporter.export({}, output_format)

        text, tables, page_count, spatial_words = pdf_data
        text = text or ""
        tables = tables or []
        spatial_words = spatial_words or []

        # Check if the PDF is a scanned document (image-only with little/no digital text)
        if ocr.is_scanned_pdf(text, page_count):
            logger.info(
                "[OCR TRIGGERED] 📄 PDF '%s' has %d page(s) with minimal digital text (%d chars). Running page-by-page OCR rendering with spatial extraction...",
                filename,
                page_count,
                len(text),
            )
            ocr_result = ocr.extract_text_and_spatial_words_from_pdf_pages(file_bytes)
            if isinstance(ocr_result, tuple):
                ocr_text, ocr_spatial_words = ocr_result
            else:
                ocr_text, ocr_spatial_words = ocr_result, []

            if ocr_text:
                text = ocr_text or ""
                spatial_words = ocr_spatial_words or []
                ocr_used = True
                logger.info("[OCR COMPLETED] Extracted %d character(s) and %d spatial bounding box(es) across %d scanned page(s).", len(text), len(spatial_words), page_count)
        else:
            logger.info(
                "[DIGITAL EXTRACTION] 📑 PDF '%s' contains digital text (%d chars, %d table(s), %d page(s)). (OCR skipped).",
                filename,
                len(text),
                len(tables),
                page_count,
            )

    clean_text = (text or "").strip()

    # 2. Document type detection
    report_type, detect_conf = detector.detect_report_type(clean_text)
    logger.info("[DETECTOR] 🏷️ Document classified as ReportType.%s (Detection Confidence: %.2f)", report_type.name, detect_conf)

    if report_type is ReportType.UNKNOWN:
        logger.warning("[PIPELINE ABORTED] Document type is UNKNOWN. Returning empty result.")
        return exporter.export({}, output_format)

    # 3. Extraction layers (Layer 4 LLM called first if configured)
    layer_results: dict[str, ExtractionResult] = {}

    if layer4_llm.is_available():
        active_prov = (layer4_llm.get_active_provider() or "NONE").upper()
        logger.info("[LAYER 4: %s LLM (PRIMARY)] 🤖 Invoking %s LLM extraction first...", active_prov, active_prov)
        try:
            l4_result = await layer4_llm.extract(clean_text, report_type)
            if l4_result:
                layer_results["layer4_llm"] = l4_result
                logger.info("[LAYER 4: %s LLM] Extracted %d field(s): %s", active_prov, len(l4_result), list(l4_result.keys()))
            else:
                logger.info("[LAYER 4: %s LLM] No fields returned or extraction fallback triggered.", active_prov)
        except Exception as exc:
            logger.warning("[LAYER 4: %s LLM] LLM extraction error: %s. Continuing with offline layers.", active_prov, exc)
    else:
        logger.info("[LAYER 4: LLM] Skipped (No local Ollama / Gemini provider configured). Running offline layers.")

    # Offline extraction layers
    try:
        l1 = layer1_pdfplumber.extract(clean_text, tables or [], report_type, spatial_words=spatial_words or []) or {}
    except Exception as exc:
        logger.warning("Layer 1 extraction error: %s", exc)
        l1 = {}

    try:
        l2 = layer2_spacy.extract(clean_text, report_type) or {}
    except Exception as exc:
        logger.warning("Layer 2 extraction error: %s", exc)
        l2 = {}

    try:
        l3 = layer3_regex.extract(clean_text, report_type) or {}
    except Exception as exc:
        logger.warning("Layer 3 extraction error: %s", exc)
        l3 = {}

    layer_results["layer1_pdfplumber"] = l1
    layer_results["layer2_spacy"] = l2
    layer_results["layer3_regex"] = l3

    logger.info("[LAYER 1: pdfplumber/layout] Extracted %d field(s): %s", len(l1), list(l1.keys()))
    logger.info("[LAYER 2: spaCy NER]         Extracted %d field(s): %s", len(l2), list(l2.keys()))
    logger.info("[LAYER 3: Regex Patterns]   Extracted %d field(s): %s", len(l3), list(l3.keys()))

    # 4. Consensus merge of all active layers
    final_result = merger.merge(layer_results or {})

    # 5. Generate Document Summary
    try:
        doc_summary_text = await summarizer.generate_summary(clean_text, report_type, final_result)
    except Exception as exc:
        logger.warning("Summarizer failed: %s. Using fallback empty summary.", exc)
        doc_summary_text = ""

    doc_summary_text = (doc_summary_text or "").strip()
    final_result["document_summary"] = FieldResult(
        value=doc_summary_text,
        confidence=1.0,
        source="summarizer"
    )

    # Log Layer Attribution Table in Terminal
    meta_info = final_result.get("__meta__", FieldResult(value={})).value or {}
    final_conf = meta_info.get("overall_confidence", 0.0) if isinstance(meta_info, dict) else 0.0

    logger.info("-------------------- FIELD SOURCE ATTRIBUTION BREAKDOWN --------------------")
    for field_name, field_res in final_result.items():
        if field_name == "__meta__":
            continue
        val_str = str(field_res.value)
        if len(val_str) > 35:
            val_str = val_str[:32] + "..."
        logger.info("  • %-20s -> %-35s [Conf: %.2f | Source: %s]", field_name, repr(val_str), field_res.confidence, field_res.source)
    logger.info("  • %-20s -> Overall Conf: %.2f | Total Fields: %d | OCR Used: %s", "SUMMARY STATS", final_conf, len(final_result) - 1, ocr_used)
    logger.info("============================================================================")

    # 6. Export pipeline result
    return exporter.export(final_result, output_format)
