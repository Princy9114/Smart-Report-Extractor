"""
layer4_llm.py
~~~~~~~~~~~~~
Extraction Layer 4: Multi-Provider LLM Structured Extraction.

Supports:
1. **Local Quantized LLMs (Ollama / Local HTTP endpoint)**:
   - Zero API token cost, 100% data privacy.
   - Runs models like Llama 3.2, Llama 3.1 8B, Mistral 7B, Phi-3.
   - Native JSON mode with strict schema guidance.
2. **Google Gemini (Official Google GenAI SDK)**:
   - High-throughput multimodal/flash extraction via `gemini-2.5-flash` or `gemini-3.8-flash`.

Environment Variables
---------------------
LLM_PROVIDER        "auto" (default), "ollama", "local", "gemini", or "none".
OLLAMA_BASE_URL     Defaults to "http://localhost:11434".
OLLAMA_MODEL        Defaults to "llama3.2" (or "llama3.1:8b", "mistral:7b", "phi3").
LOCAL_LLM_URL       Custom OpenAI-compatible URL (e.g. "http://localhost:1234/v1/chat/completions").
LOCAL_LLM_MODEL     Defaults to "local-model".
GOOGLE_API_KEY      Google GenAI API Key.
GEMINI_API_KEY      Alternative alias for Google GenAI API Key.
GEMINI_MODEL        Defaults to "gemini-2.5-flash".
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)

# Default model identifiers
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_CONFIDENCE = 0.85

# ---------------------------------------------------------------------------
# Prompt Templates
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
# Helpers
# ---------------------------------------------------------------------------

def _build_prompts(
    text: str,
    report_type: ReportType,
    field_schema: dict[str, str],
) -> tuple[str, str]:
    """Build the system instruction and user prompt."""
    system = _SYSTEM_PROMPT.format(report_type=report_type.value.replace("_", " "))
    user_content = _USER_PROMPT.format(
        field_schema=json.dumps(field_schema, indent=2),
        text=text[:30_000],
    )
    return system, user_content


def _parse_response(raw: Any) -> dict[str, Any]:
    """Extract JSON object from raw LLM output, stripping markdown fences."""
    if not raw:
        return {}
    cleaned = str(raw).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json")
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```")
    cleaned = cleaned.strip()
    if not cleaned:
        return {}
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _to_field_results(data: dict[str, Any], source: str = "llm_gemini") -> dict[str, FieldResult]:
    """Convert raw dictionary to FieldResult mapping."""
    return {
        key: FieldResult(value=val, confidence=_CONFIDENCE, source=source)
        for key, val in data.items()
        if val is not None and val != "" and val != []
    }


# ---------------------------------------------------------------------------
# Provider Detection
# ---------------------------------------------------------------------------

def get_active_provider() -> str:
    """Determine which LLM provider should be used based on configuration.

    Returns one of: 'ollama', 'local', 'gemini', or 'none'.
    """
    prov = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    if prov in ("ollama", "local", "gemini"):
        return prov

    if prov == "none":
        return "none"

    # In 'auto' mode, prefer local Ollama if explicitly configured, else Gemini if key present, else local Ollama
    if os.getenv("OLLAMA_ENABLED", "").lower() in ("true", "1", "yes"):
        return "ollama"

    if os.getenv("LOCAL_LLM_URL"):
        return "local"

    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        return "gemini"

    # Default to ollama if configured or available
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"

    return "none"


def is_available() -> bool:
    """Check if any LLM extraction engine is currently configured."""
    return get_active_provider() != "none"


# ---------------------------------------------------------------------------
# Provider 1: Local Ollama Model
# ---------------------------------------------------------------------------

async def _extract_with_ollama(
    system_instruction: str,
    user_content: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute structured JSON inference using a local Ollama instance."""
    url = (base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip("/")
    model_name = model or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    eff_timeout = timeout if timeout is not None else float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

    endpoint = f"{url}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    logger.debug("layer4_llm: Querying local Ollama endpoint %s with model '%s' (timeout=%.1fs)...", endpoint, model_name, eff_timeout)

    try:
        async with httpx.AsyncClient(timeout=eff_timeout) as client:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code == 404:
                err_msg = resp.text
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", err_msg)
                except Exception:
                    pass
                raise ValueError(f"Model '{model_name}' not found in Ollama ({err_msg}). Run 'ollama pull {model_name}' first.")
            resp.raise_for_status()
            res_json = resp.json()

            # Check response structure
            content = ""
            if "message" in res_json and "content" in res_json["message"]:
                content = res_json["message"]["content"]
            elif "response" in res_json:
                content = res_json["response"]

            return _parse_response(content)
    except httpx.ConnectError:
        raise ConnectionError(f"Could not connect to Ollama at {url}. Make sure Ollama daemon is running.")
    except httpx.TimeoutException:
        raise TimeoutError(f"Ollama inference timed out after {eff_timeout:.0f}s (CPU/GPU took too long to generate).")


# ---------------------------------------------------------------------------
# Provider 2: Local OpenAI-Compatible Server (vLLM / LM Studio / LocalAI)
# ---------------------------------------------------------------------------

async def _extract_with_local_endpoint(
    system_instruction: str,
    user_content: str,
    *,
    endpoint_url: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute inference against a local OpenAI-compatible endpoint."""
    url = endpoint_url or os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1/chat/completions")
    model_name = model or os.getenv("LOCAL_LLM_MODEL", "local-model")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    logger.debug("layer4_llm: Querying local OpenAI-compatible endpoint %s...", url)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        res_json = resp.json()
        raw_text = res_json["choices"][0]["message"]["content"]
        return _parse_response(raw_text)


# ---------------------------------------------------------------------------
# Provider 3: Google Gemini API
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: str | None = None) -> Any | None:
    """Instantiate a Google GenAI client if an API key is available."""
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        return None
    try:
        from google import genai
        return genai.Client(api_key=key.strip())
    except Exception as exc:
        logger.warning("layer4_llm: Failed to initialize Google GenAI Client: %s", exc)
        return None


async def _extract_with_gemini(
    system_instruction: str,
    user_content: str,
    *,
    client: Any | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Execute structured JSON inference using the Google GenAI SDK."""
    from google.genai import types

    gemini_client = client or get_gemini_client()
    if gemini_client is None:
        raise ValueError("Google GenAI client not configured (missing GOOGLE_API_KEY).")

    model = model_name or os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)

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
    return _parse_response(raw_text)


# ---------------------------------------------------------------------------
# Public Unified Entry-Point
# ---------------------------------------------------------------------------

async def extract(
    text: str,
    report_type: ReportType,
    field_schema: dict[str, str] | None = None,
    *,
    provider: str | None = None,
    client: Any | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> dict[str, FieldResult]:
    """Extract structured fields from *text* using local or remote LLMs.

    Parameters
    ----------
    text:
        Plain text of the document.
    report_type:
        Detected document type.
    field_schema:
        Optional schema dictionary.
    provider:
        Explicit provider override: 'ollama', 'local', 'gemini', or 'auto'.
    client:
        Optional client override for Gemini.
    model_name:
        Optional model override.
    base_url:
        Optional base URL override for Ollama / local endpoints.

    Returns
    -------
    dict[str, FieldResult]
        Extracted fields with confidence and source tags.
    """
    if report_type is ReportType.UNKNOWN:
        logger.debug("layer4_llm: skipping LLM call for UNKNOWN report type.")
        return {}

    schema = field_schema or _DEFAULT_SCHEMAS.get(report_type, {})
    if not schema:
        logger.warning("layer4_llm: no field schema available for %s.", report_type)
        return {}

    chosen_provider = (provider or get_active_provider()).lower()
    if chosen_provider == "none":
        logger.debug("layer4_llm: No LLM provider configured. Skipping Layer 4.")
        return {}

    system_instruction, user_content = _build_prompts(text, report_type, schema)

    # 1. Ollama Local Execution
    if chosen_provider == "ollama":
        try:
            data = await _extract_with_ollama(
                system_instruction,
                user_content,
                base_url=base_url,
                model=model_name,
            )
            res = _to_field_results(data, source="llm_local_ollama")
            logger.info("layer4_llm (Ollama): Extracted %d field(s) for %s.", len(res), report_type.value)
            return res
        except Exception as exc:
            logger.warning("layer4_llm: Ollama local extraction failed: %s. Continuing...", exc)
            return {}

    # 2. Custom OpenAI-compatible Local Execution
    if chosen_provider == "local":
        try:
            data = await _extract_with_local_endpoint(
                system_instruction,
                user_content,
                endpoint_url=base_url,
                model=model_name,
            )
            res = _to_field_results(data, source="llm_local")
            logger.info("layer4_llm (Local Endpoint): Extracted %d field(s) for %s.", len(res), report_type.value)
            return res
        except Exception as exc:
            logger.warning("layer4_llm: Local LLM extraction failed: %s. Continuing...", exc)
            return {}

    # 3. Google Gemini API Execution
    if chosen_provider == "gemini":
        try:
            data = await _extract_with_gemini(
                system_instruction,
                user_content,
                client=client,
                model_name=model_name,
            )
            res = _to_field_results(data, source="llm_gemini")
            logger.info("layer4_llm (Gemini): Extracted %d field(s) for %s.", len(res), report_type.value)
            return res
        except json.JSONDecodeError as exc:
            logger.warning("layer4_llm: failed to parse JSON response from Gemini — %s", exc)
        except Exception as exc:
            logger.warning("layer4_llm: Google Gemini API error — %s", exc)
        return {}

    return {}
