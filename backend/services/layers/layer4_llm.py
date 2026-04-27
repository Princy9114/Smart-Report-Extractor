"""
layer4_llm.py
~~~~~~~~~~~~~
Extraction Layer 4: Google Gemini structured extraction.

Sends the document text plus a field schema to Gemini and parses the
JSON response into ``Dict[str, FieldResult]``.  All results carry
``confidence=0.85`` and ``source="llm_gemini"``.

On any failure (API error, JSON parse error, etc.) the function logs a
warning and returns an empty dict so the pipeline can fall back gracefully.

Environment variables
---------------------
GOOGLE_API_KEY      Required. Set in .env or the environment.
GEMINI_MODEL        Optional. Defaults to ``gemini-1.5-flash``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import google.generativeai as genai

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client — instantiated once at module level
# ---------------------------------------------------------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

_MODEL       = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_CONFIDENCE  = 0.85

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise document data-extraction assistant.
You will be given the text of a {report_type} document and a JSON schema
that lists the fields to extract.

Rules:
1. Return ONLY a valid JSON object — no markdown, no explanation.
2. Keys must exactly match the field names defined in the schema.
3. If a field cannot be found in the text, omit its key entirely.
4. For list fields (e.g. line_items, transactions) return a JSON array.
5. Do not invent values; extract only what is explicitly present.
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
        text=text[:30_000],   # limit text to keep under reasonable bounds
    )
    return system, user_content


def _parse_response(raw: str) -> dict[str, Any]:
    """Extract the JSON object from the response text."""
    # Strip potential markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


def _to_field_results(data: dict[str, Any]) -> dict[str, FieldResult]:
    return {
        key: FieldResult(value=val, confidence=_CONFIDENCE, source="llm_gemini")
        for key, val in data.items()
        if val is not None
    }


# ---------------------------------------------------------------------------
# Default field schemas per report type
# ---------------------------------------------------------------------------

_DEFAULT_SCHEMAS: dict[ReportType, dict[str, str]] = {
    ReportType.INVOICE: {
        "invoice_number": "string",
        "date":           "string (ISO 8601 if possible)",
        "vendor":         "string",
        "client":         "string",
        "gstin":          "string",
        "pan":            "string",
        "subtotal":       "string",
        "tax":            "string",
        "total":          "string",
        "line_items":     "array of {description, quantity, unit_price, amount}",
    },
    ReportType.BANK_STATEMENT: {
        "bank_name":        "string",
        "account_number":   "string",
        "account_holder":   "string",
        "ifsc":             "string",
        "statement_period": "string",
        "opening_balance":  "string",
        "closing_balance":  "string",
        "transactions":     "array of {date, description, debit, credit, balance}",
    },
    ReportType.RESUME: {
        "name":             "string",
        "email":            "string",
        "phone":            "string",
        "profile_url":      "string",
        "summary":          "string (2-4 sentences)",
        "skills":           "array of strings",
        "education":        "array of {institution, degree, year}",
        "experience":       "array of {company, title, start_date, end_date, description}",
        "certifications":   "array of strings",
    },
}


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def extract(
    text: str,
    report_type: ReportType,
    field_schema: dict[str, str] | None = None,
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
        Optional override.  A ``{field_name: description}`` dict that tells
        Gemini which fields to extract and in what format.  Falls back to
        ``_DEFAULT_SCHEMAS[report_type]`` when not supplied.

    Returns
    -------
    dict[str, FieldResult]
        Extracted fields with ``confidence=0.85`` and
        ``source="llm_gemini"``.  Returns ``{}`` on any failure.
    """
    if report_type is ReportType.UNKNOWN:
        logger.debug("layer4_llm: skipping LLM call for UNKNOWN report type.")
        return {}

    schema = field_schema or _DEFAULT_SCHEMAS.get(report_type, {})
    if not schema:
        logger.warning("layer4_llm: no field schema available for %s.", report_type)
        return {}

    try:
        system, user_content = _build_prompts(text, report_type, schema)

        model = genai.GenerativeModel(
            model_name=_MODEL,
            system_instruction=system
        )

        response = await model.generate_content_async(
            user_content,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )

        raw_text: str = response.text
        data = _parse_response(raw_text)
        result = _to_field_results(data)
        logger.info(
            "layer4_llm: extracted %d field(s) for %s.", len(result), report_type.value
        )
        return result

    except json.JSONDecodeError as exc:
        logger.warning("layer4_llm: failed to parse JSON response — %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("layer4_llm: API or unexpected error — %s", exc)

    return {}
