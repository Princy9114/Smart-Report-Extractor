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

from fastapi.responses import StreamingResponse

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType
from backend.services import detector, exporter, merger, summarizer
from backend.services.layers import layer1_pdfplumber, layer2_spacy, layer3_regex, layer4_llm
from backend.utils import pdf_utils

logger = logging.getLogger(__name__)

ExtractionResult = dict[str, FieldResult]

_LLM_CONFIDENCE_THRESHOLD = 0.85


async def run_pipeline(
    pdf_bytes: bytes,
    output_format: str = "json",
) -> StreamingResponse | dict[str, Any]:
    """Execute the full document processing pipeline.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of the uploaded PDF file.
    output_format:
        Desired output format (e.g. "json", "csv").

    Returns
    -------
    StreamingResponse
        The merged ExtractionResult streamed in the requested format.
    """
    logger.info("Pipeline started (format: %s)", output_format)

    # 1. Base PDF extraction (text + tables)
    pdf_data = pdf_utils.extract_pdf_content(pdf_bytes)
    if pdf_data is None:
        logger.warning("Pipeline aborted: PDF is encrypted.")
        # Return an empty result structure
        return exporter.export({}, output_format)

    text, tables, page_count = pdf_data
    logger.debug("PDF extracted: %d pages", page_count)

    # 2. Document type detection
    report_type, detect_conf = detector.detect_report_type(text)
    logger.info("Detected type: %s (confidence: %.2f)", report_type.value, detect_conf)

    if report_type is ReportType.UNKNOWN:
        logger.warning("Pipeline aborted: document type is UNKNOWN.")
        return exporter.export({}, output_format)

    # 3. Fast offline layers
    layer_results: dict[str, ExtractionResult] = {}

    layer_results["layer1_pdfplumber"] = layer1_pdfplumber.extract(text, tables, report_type)
    layer_results["layer2_spacy"] = layer2_spacy.extract(text, report_type)
    layer_results["layer3_regex"] = layer3_regex.extract(text, report_type)

    # 4. Preliminary merge to check confidence
    merged_prelim = merger.merge(layer_results)
    prelim_meta = merged_prelim.get("__meta__", FieldResult(value={})).value
    overall_conf = prelim_meta.get("overall_confidence", 0.0)

    logger.info("Offline layers complete. Preliminary confidence: %.3f", overall_conf)

    # 5. Conditional LLM fallback
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    
    if api_key and overall_conf < _LLM_CONFIDENCE_THRESHOLD:
        logger.info(
            "Confidence %.3f < %.3f. Falling back to LLM layer...",
            overall_conf,
            _LLM_CONFIDENCE_THRESHOLD,
        )
        l4_result = await layer4_llm.extract(text, report_type)
        if l4_result:
            layer_results["layer4_llm"] = l4_result
            # Re-merge with the new LLM data included
            final_result = merger.merge(layer_results)
        else:
            final_result = merged_prelim
    else:
        if not api_key:
            logger.debug("Skipping LLM fallback: no API key configured.")
        else:
            logger.debug("Skipping LLM fallback: confidence %f is sufficient.", overall_conf)
        final_result = merged_prelim

    # 6. Generate Document Summary
    doc_summary_text = await summarizer.generate_summary(text, report_type, final_result)
    final_result["document_summary"] = FieldResult(
        value=doc_summary_text,
        confidence=1.0,
        source="summarizer"
    )

    # 7. Export pipeline result
    return exporter.export(final_result, output_format)
