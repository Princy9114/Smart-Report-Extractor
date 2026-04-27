import os
import logging
from typing import Dict, Any
from anthropic import AsyncAnthropic

from backend.models.report_type import ReportType
from backend.models.field_result import FieldResult

logger = logging.getLogger(__name__)

# Module-level client load
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if _ANTHROPIC_API_KEY:
    try:
        _client = AsyncAnthropic(api_key=_ANTHROPIC_API_KEY)
    except Exception as e:
        logger.warning(f"Summarizer failed to init Anthropic client: {e}")
        _client = None
else:
    _client = None

_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


async def generate_summary(text: str, report_type: ReportType, extracted_fields: Dict[str, FieldResult]) -> str:
    """
    Generate a concise summary of the document.
    Prioritizes an LLM (Claude) abstraction if API credentials are provided.
    Otherwise, falls back to a deterministic offline heuristic summary.
    """
    if _client:
        logger.info("Generating summary via Anthropic API (LLM)...")
        # --- LLM Based Summarization (API) ---
        prompt = (
            f"You are an AI document summarizer. Give a concise 1-2 sentence paragraph summarizing the following raw "
            f"{report_type.value} text. Focus on the core purpose, timeline, amounts, or primary entity/individual. "
            f"Do not format as a list. Do not introduce with 'Here is a summary...'\n\n"
            f"<document_text>\n{text[:5000]}\n</document_text>" # Strip to 5000 chars to save tokens on massive docs
        )
        
        try:
            response = await _client.messages.create(
                model=_MODEL,
                max_tokens=300,
                temperature=0.3, # Keep variations low
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Summarizer LLM fallback failed: {e}. Falling back to heuristic summarizer.")

    # --- Heuristic Based Summarization (Non-API) ---
    logger.info("Generating summary via Heuristics (Offline)...")
    
    # Try to intelligently construct sentences based on fields actually grabbed
    fields = {k: v.value for k, v in extracted_fields.items() if k != "__meta__"}
    if not fields:
        return f"This document appears to be a {report_type.value.replace('_', ' ')} but no specific fields could be parsed."

    if report_type == ReportType.INVOICE:
        amount = fields.get("total", "an unknown amount")
        invoice_num = fields.get("invoice_number", "unknown invoice number")
        return f"This is an invoice (Ref: {invoice_num}) for a total amount of {amount}. Parsed seamlessly using offline rules."
        
    elif report_type == ReportType.RESUME:
        name = fields.get("PERSON", "an unknown candidate")
        return f"This document is a professional resume for {name}. It was processed automatically identifying {len(fields)} core data points."
        
    elif report_type == ReportType.BANK_STATEMENT:
        acc = fields.get("account_number", "an unknown account")
        return f"This is a bank statement mapping to account ending in {str(acc)[-4:]}. Extracted offline."
        
    else:
        return f"Document structured layout parsed locally finding {len(fields)} items."
