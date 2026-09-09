"""
layer4_llm.py
~~~~~~~~~~~~~
Extraction Layer 4: Google Gemini structured extraction using the official Google GenAI SDK.

Sends the document text plus a structured field schema to Gemini and parses the
JSON response into ``dict[str, FieldResult]``. All results carry
``confidence=0.85`` and ``source="llm_gemini"``.

On any failure (API error, invalid key, JSON parse error, etc.) the function logs a
warning and returns an empty dict so the pipeline continues gracefully with offline layers.

Environment variables
---------------------
GOOGLE_API_KEY / GEMINI_API_KEY   Required for live LLM extraction.
GEMINI_MODEL                      Optional. Defaults to ``gemini-2.5-flash``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-3.8-flash"
_CONFIDENCE = 0.85

# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: str | None = None) -> genai.Client | None:
    """Instantiate a Google GenAI client if an API key is available."""
    key = api_key or os.getenv("Google_API_Key") or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        return None
    try:
        return genai.Client(api_key=key.strip())
    except Exception as exc:
        logger.warning("layer4_llm: Failed to initialize Google GenAI Client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise document data-extraction assistant.
You will be given the text of a {report_type} document and a JSON schema
that lists the fields to extract.

Rules:
1. Return ONLY a valid JSON object matching the requested schema.
2. Keys must exactly match the field names defined in the schema.
3. If a field cannot be found in the text, omit its key entirely.
4. For list fields (e.g. line_items, transactions, skills) return a JSON array.
5. Do not invent or hallucinate values; extract only what is explicitly present.
"""

_USER_PROMPT = """\
FIELD SCHEMA (extract these fields):
{field_schema}

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"

Respond with a JSON object only.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompts(
    text: str,
    report_type: ReportType,
    field_schema: dict[str, str],
) -> tuple[str, str]:
    """Build the system instruction and user prompt for Gemini API."""
    system = _SYSTEM_PROMPT.format(report_type=report_type.value.replace("_", " "))
    user_content = _USER_PROMPT.format(
        field_schema=json.dumps(field_schema, indent=2),
        text=text[:30_000],  # limit text to keep under reasonable bounds
    )
    return system, user_content


def _parse_response(raw: str) -> dict[str, Any]:
    """Extract the JSON object from the response text, stripping markdown code fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json")
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```")
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def _to_field_results(data: dict[str, Any]) -> dict[str, FieldResult]:
    """Convert raw extracted dictionary to FieldResult mapping."""
    return {
        key: FieldResult(value=val, confidence=_CONFIDENCE, source="llm_gemini")
        for key, val in data.items()
        if val is not None and val != "" and val != []
    }


# ---------------------------------------------------------------------------
# Default field schemas per report type
# ---------------------------------------------------------------------------

_DEFAULT_SCHEMAS: dict[ReportType, dict[str, str]] = {
    ReportType.INVOICE: {
        "invoice_number": "string",
        "date": "string (ISO 8601 or original format)",
        "vendor": "string (name of issuing company/vendor)",
        "client": "string (name of customer/recipient)",
        "gstin": "string",
        "pan": "string",
        "email": "string",
        "phone": "string",
        "subtotal": "string",
        "tax": "string",
        "total": "string (final total payable amount)",
        "line_items": "array of {description, quantity, unit_price, amount}",
    },
    ReportType.BANK_STATEMENT: {
        "bank_name": "string",
        "account_number": "string",
        "account_name": "string (account holder name)",
        "ifsc": "string",
        "statement_period": "string",
        "opening_balance": "string",
        "closing_balance": "string",
        "email": "string",
        "phone": "string",
        "transactions": "array of {date, description, debit, credit, balance}",
    },
    ReportType.RESUME: {
        "name": "string (candidate full name)",
        "email": "string",
        "phone": "string",
        "profile_url": "string (LinkedIn/GitHub/Portfolio)",
        "summary": "string (professional summary)",
        "skills": "array of strings",
        "organizations": "array of strings (universities, previous employers)",
        "education": "array of {institution, degree, year}",
        "experience": "array of {company, title, start_date, end_date, description}",
        "projects": "array of {name, description}",
        "dates": "array of date ranges",
    },
}


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def extract(
    text: str,
    report_type: ReportType,
    field_schema: dict[str, str] | None = None,
    *,
    client: genai.Client | None = None,
    model_name: str | None = None,
) -> dict[str, FieldResult]:
    """Call the Google Gemini API to extract structured fields from *text*.

    Parameters
    ----------
    text:
        Plain text of the document.
    report_type:
        Detected document type; used to select the default schema and to
        contextualise the prompt.
    field_schema:
        Optional override. A ``{field_name: description}`` dict that tells
        Gemini which fields to extract and in what format.
    client:
        Optional pre-configured ``genai.Client`` instance (useful for testing or custom auth).
    model_name:
        Optional Gemini model name override (defaults to GEMINI_MODEL env var or ``gemini-2.5-flash``).

    Returns
    -------
    dict[str, FieldResult]
        Extracted fields with ``confidence=0.85`` and ``source="llm_gemini"``.
        Returns ``{}`` on any failure or if no API key is set.
    """
    if report_type is ReportType.UNKNOWN:
        logger.debug("layer4_llm: skipping LLM call for UNKNOWN report type.")
        return {}

    schema = field_schema or _DEFAULT_SCHEMAS.get(report_type, {})
    if not schema:
        logger.warning("layer4_llm: no field schema available for %s.", report_type)
        return {}

    gemini_client = client or get_gemini_client()
    if gemini_client is None:
        logger.debug("layer4_llm: Google GenAI client not configured (missing GOOGLE_API_KEY/GEMINI_API_KEY).")
        return {}

    model = model_name or os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)

    try:
        system_instruction, user_content = _build_prompts(text, report_type, schema)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.1,
        )

        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=user_content,
            config=config,
        )

        raw_text: str = response.text or ""
        data = _parse_response(raw_text)
        result = _to_field_results(data)
        logger.info(
            "layer4_llm: extracted %d field(s) for %s using model %s.",
            len(result),
            report_type.value,
            model,
        )
        return result

    except json.JSONDecodeError as exc:
        logger.warning("layer4_llm: failed to parse JSON response — %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("layer4_llm: Google Gemini API error — %s", exc)

    return {}

