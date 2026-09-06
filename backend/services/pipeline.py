"""
pipeline.py
~~~~~~~~~~~
Central orchestration service for extracting structured data from documents.

Workflow:
1. Parse PDF bytes to text and tables (returns early if encrypted).
2. Detect document type using heuristic keyword scoring.
3. Run synchronous extraction layers (1: pdfplumber, 2: spaCy, 3: regex).
4. Merge results to compute preliminary overall confidence.
5. If confidence falls below the threshold and the LLM API is enabled,
   invoke Layer 4 (Anthropic) and re-merge.
6. Export the final result to the requested format (JSON/CSV) as a
   streaming response.
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
    """Execute the full document processing pipeline with OCR support.

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
    tables: list[list] = []
    ocr_used: bool = False

    if _is_image(filename, file_bytes):
        logger.info("[OCR TRIGGERED] 📸 File '%s' is an image. Invoking RapidOCR engine...", filename)
        text = ocr.extract_text_from_image(file_bytes)
        page_count = 1
        ocr_used = True
        logger.info("[OCR COMPLETED] Extracted %d character(s) from image via OCR.", len(text))
    else:
        pdf_data = pdf_utils.extract_pdf_content(file_bytes)
        if pdf_data is None:
            logger.warning("[PIPELINE ABORTED] PDF '%s' is encrypted or unreadable.", filename)
            return exporter.export({}, output_format)

        text, tables, page_count = pdf_data

        # Check if the PDF is a scanned document (image-only with little/no digital text)
        if ocr.is_scanned_pdf(text, page_count):
            logger.info(
                "[OCR TRIGGERED] 📄 PDF '%s' has %d page(s) with minimal digital text (%d chars). Running page-by-page OCR rendering...",
                filename,
                page_count,
                len(text),
            )
            ocr_text = ocr.extract_text_from_pdf_pages(file_bytes)
            if ocr_text:
                text = ocr_text
                ocr_used = True
                logger.info("[OCR COMPLETED] Extracted %d character(s) across %d scanned page(s).", len(text), page_count)
        else:
            logger.info(
                "[DIGITAL EXTRACTION] 📑 PDF '%s' contains digital text (%d chars, %d table(s), %d page(s)). (OCR skipped).",
                filename,
                len(text),
                len(tables),
                page_count,
            )

    # 2. Document type detection
    report_type, detect_conf = detector.detect_report_type(text)
    logger.info("[DETECTOR] 🏷️ Document classified as ReportType.%s (Detection Confidence: %.2f)", report_type.name, detect_conf)

    if report_type is ReportType.UNKNOWN:
        logger.warning("[PIPELINE ABORTED] Document type is UNKNOWN. Returning empty result.")
        return exporter.export({}, output_format)

    # 3. Fast offline layers
    layer_results: dict[str, ExtractionResult] = {}

    l1 = layer1_pdfplumber.extract(text, tables, report_type)
    l2 = layer2_spacy.extract(text, report_type)
    l3 = layer3_regex.extract(text, report_type)

    layer_results["layer1_pdfplumber"] = l1
    layer_results["layer2_spacy"] = l2
    layer_results["layer3_regex"] = l3

    logger.info("[LAYER 1: pdfplumber/layout] Extracted %d field(s): %s", len(l1), list(l1.keys()))
    logger.info("[LAYER 2: spaCy NER]         Extracted %d field(s): %s", len(l2), list(l2.keys()))
    logger.info("[LAYER 3: Regex Patterns]   Extracted %d field(s): %s", len(l3), list(l3.keys()))

    # 4. Preliminary merge to check confidence
    merged_prelim = merger.merge(layer_results)
    prelim_meta = merged_prelim.get("__meta__", FieldResult(value={})).value
    overall_conf = prelim_meta.get("overall_confidence", 0.0)

    logger.info("[CONSENSUS MERGE] Offline confidence: %.3f (Threshold for LLM: %.2f)", overall_conf, _LLM_CONFIDENCE_THRESHOLD)

    # 5. Conditional LLM fallback
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    
    if api_key and overall_conf < _LLM_CONFIDENCE_THRESHOLD:
        logger.info("[LAYER 4: Gemini LLM] 🤖 Confidence %.3f < %.2f. Invoking LLM fallback...", overall_conf, _LLM_CONFIDENCE_THRESHOLD)
        l4_result = await layer4_llm.extract(text, report_type)
        if l4_result:
            layer_results["layer4_llm"] = l4_result
            logger.info("[LAYER 4: Gemini LLM] Extracted %d field(s): %s", len(l4_result), list(l4_result.keys()))
            final_result = merger.merge(layer_results)
        else:
            logger.info("[LAYER 4: Gemini LLM] No additional fields returned from LLM.")
            final_result = merged_prelim
    else:
        if not api_key:
            logger.info("[LAYER 4: Gemini LLM] Skipped (No GOOGLE_API_KEY configured).")
        else:
            logger.info("[LAYER 4: Gemini LLM] Skipped (Offline confidence %.3f is sufficient).", overall_conf)
        final_result = merged_prelim

    # 6. Generate Document Summary
    doc_summary_text = await summarizer.generate_summary(text, report_type, final_result)
    final_result["document_summary"] = FieldResult(
        value=doc_summary_text,
        confidence=1.0,
        source="summarizer"
    )

    # Log Layer Attribution Table in Terminal
    meta_info = final_result.get("__meta__", FieldResult(value={})).value
    final_conf = meta_info.get("overall_confidence", overall_conf)

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

    # 7. Export pipeline result
    return exporter.export(final_result, output_format)
