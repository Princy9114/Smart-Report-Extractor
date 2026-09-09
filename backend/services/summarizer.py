import logging
import os
from typing import Any

from google import genai
from google.genai import types

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)


def _get_gemini_client() -> genai.Client | None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        try:
            return genai.Client(api_key=api_key.strip())
        except Exception as exc:
            logger.warning("Summarizer: Failed to init Gemini client: %s", exc)
    return None


def _get_anthropic_client() -> Any | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and AsyncAnthropic:
        try:
            return AsyncAnthropic(api_key=api_key.strip())
        except Exception as exc:
            logger.warning("Summarizer: Failed to init Anthropic client: %s", exc)
    return None


async def generate_summary(text: str, report_type: ReportType, extracted_fields: dict[str, FieldResult]) -> str:
    """Generate a concise summary of the document.

    Prioritizes Gemini (or Anthropic) if API credentials are provided.
    Otherwise, falls back to a deterministic offline heuristic summary.
    """
    summary_prompt = (
        f"You are an AI document summarizer. Give a concise 1-2 sentence paragraph summarizing the following raw "
        f"{report_type.value} text. Focus on the core purpose, timeline, amounts, or primary entity/individual. "
        f"Do not format as a list. Do not introduce with 'Here is a summary...'\n\n"
        f"<document_text>\n{text[:5000]}\n</document_text>"
    )

    # 1. Try Google Gemini LLM
    gemini_client = _get_gemini_client()
    if gemini_client:
        logger.info("Generating summary via Google Gemini API...")
        try:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            response = await gemini_client.aio.models.generate_content(
                model=model,
                contents=summary_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=250,
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as exc:
            logger.warning("Summarizer: Gemini LLM summary failed: %s. Trying fallback...", exc)

    # 2. Try Anthropic Claude LLM
    anthropic_client = _get_anthropic_client()
    if anthropic_client:
        logger.info("Generating summary via Anthropic API...")
        try:
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
            response = await anthropic_client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Summarizer: Anthropic fallback failed: %s.", exc)

    # 3. Deterministic Heuristic-Based Summarization (Offline)
    logger.info("Generating summary via Heuristics (Offline)...")
    fields = {k: v.value for k, v in extracted_fields.items() if k != "__meta__"}
    if not fields:
        return f"This document appears to be a {report_type.value.replace('_', ' ')} but no specific fields could be parsed."

    if report_type == ReportType.INVOICE:
        amount = fields.get("total_amount") or fields.get("total") or "an unknown amount"
        invoice_num = fields.get("invoice_number", "unknown invoice number")
        vendor = fields.get("vendor", "")
        vendor_part = f" from {vendor}" if vendor else ""
        return f"This is an invoice (Ref: {invoice_num}){vendor_part} for a total amount of {amount}. Parsed seamlessly using multi-layer extraction."

    elif report_type == ReportType.RESUME:
        name = fields.get("name") or fields.get("PERSON") or "a candidate"
        return f"This document is a professional resume for {name}. It was processed automatically identifying {len(fields)} core data points."

    elif report_type == ReportType.BANK_STATEMENT:
        acc = fields.get("account_number", "an account")
        acc_str = str(acc)[-4:] if len(str(acc)) >= 4 else str(acc)
        bank = fields.get("bank_name", "Bank")
        return f"This is a {bank} statement mapping to account ending in {acc_str}. Extracted offline."

    else:
        return f"Document structured layout parsed locally finding {len(fields)} items."

