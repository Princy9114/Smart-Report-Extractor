import logging
import os
from typing import Any

import httpx
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
    if api_key and str(api_key).strip():
        try:
            return genai.Client(api_key=str(api_key).strip())
        except Exception as exc:
            logger.warning("Summarizer: Failed to init Gemini client: %s", exc)
    return None


def _get_anthropic_client() -> Any | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and str(api_key).strip() and AsyncAnthropic:
        try:
            return AsyncAnthropic(api_key=str(api_key).strip())
        except Exception as exc:
            logger.warning("Summarizer: Failed to init Anthropic client: %s", exc)
    return None


async def _summarize_with_ollama(prompt: str) -> str | None:
    """Summarize document text using local Ollama model."""
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "llama3.2"
    endpoint = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code == 200:
                res_data = resp.json()
                summary = res_data.get("response")
                if summary and str(summary).strip():
                    return str(summary).strip()
    except Exception as exc:
        logger.debug("Summarizer: Ollama local summarization unavailable: %s", exc)
    return None


async def generate_summary(text: str, report_type: ReportType, extracted_fields: dict[str, FieldResult] | None) -> str:
    """Generate a concise summary of the document.

    Checks:
    1. Local Ollama (if configured or enabled)
    2. Google Gemini (if API key configured)
    3. Anthropic Claude (if API key configured)
    4. Deterministic offline heuristics fallback
    """
    clean_text = (text or "").strip()
    fields_dict = extracted_fields or {}

    summary_prompt = (
        f"You are an AI document summarizer. Give a concise 1-2 sentence paragraph summarizing the following raw "
        f"{report_type.value} text. Focus on the core purpose, timeline, amounts, or primary entity/individual. "
        f"Do not format as a list. Do not introduce with 'Here is a summary...'\n\n"
        f"<document_text>\n{clean_text[:5000]}\n</document_text>"
    )

    # 1. Try Local Ollama if enabled or configured
    if os.getenv("LLM_PROVIDER") == "ollama" or (os.getenv("OLLAMA_ENABLED") or "").lower() in ("true", "1", "yes"):
        logger.info("Generating summary via Local Ollama...")
        ollama_res = await _summarize_with_ollama(summary_prompt)
        if ollama_res and str(ollama_res).strip():
            return str(ollama_res).strip()

    # 2. Try Google Gemini LLM
    gemini_client = _get_gemini_client()
    if gemini_client:
        logger.info("Generating summary via Google Gemini API...")
        try:
            model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
            response = await gemini_client.aio.models.generate_content(
                model=model,
                contents=summary_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=250,
                ),
            )
            resp_text = getattr(response, "text", "")
            if resp_text and str(resp_text).strip():
                return str(resp_text).strip()
        except Exception as exc:
            logger.warning("Summarizer: Gemini LLM summary failed: %s. Trying fallback...", exc)

    # 3. Try Local Ollama if not already tried
    ollama_res = await _summarize_with_ollama(summary_prompt)
    if ollama_res and str(ollama_res).strip():
        return str(ollama_res).strip()

    # 4. Try Anthropic Claude LLM
    anthropic_client = _get_anthropic_client()
    if anthropic_client:
        logger.info("Generating summary via Anthropic API...")
        try:
            model = os.getenv("ANTHROPIC_MODEL") or "claude-3-5-haiku-20241022"
            response = await anthropic_client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            if response and getattr(response, "content", None):
                content_text = getattr(response.content[0], "text", "")
                if content_text and str(content_text).strip():
                    return str(content_text).strip()
        except Exception as exc:
            logger.warning("Summarizer: Anthropic fallback failed: %s.", exc)

    # 5. Deterministic Heuristic-Based Summarization (Offline)
    logger.info("Generating summary via Heuristics (Offline)...")
    fields = {k: v.value for k, v in fields_dict.items() if k != "__meta__" and v and v.value is not None}
    if not fields:
        return f"This document appears to be a {report_type.value.replace('_', ' ')} but no specific fields could be parsed."

    if report_type == ReportType.INVOICE:
        amount = fields.get("total_amount") or fields.get("total") or "an unknown amount"
        invoice_num = fields.get("invoice_number") or "unknown invoice number"
        vendor = fields.get("vendor") or ""
        vendor_part = f" from {vendor}" if vendor else ""
        return f"This is an invoice (Ref: {invoice_num}){vendor_part} for a total amount of {amount}. Parsed seamlessly using multi-layer extraction."

    elif report_type == ReportType.RESUME:
        name = fields.get("name") or fields.get("PERSON") or "a candidate"
        return f"This document is a professional resume for {name}. It was processed automatically identifying {len(fields)} core data points."

    elif report_type == ReportType.BANK_STATEMENT:
        acc = fields.get("account_number") or "an account"
        acc_str = str(acc)[-4:] if len(str(acc)) >= 4 else str(acc)
        bank = fields.get("bank_name") or "Bank"
        return f"This is a {bank} statement mapping to account ending in {acc_str}. Extracted offline."

    else:
        return f"Document structured layout parsed locally finding {len(fields)} items."
