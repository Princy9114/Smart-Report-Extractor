import json
import os
import sys
import traceback
import httpx
from app.schemas.base import DocumentType, ExtractionMethod
from app.schemas import InvoiceExtraction, ResumeExtraction, BankStatementExtraction
from app.utils.pdf_parser import ParsedPDF, extract_tables_as_text

# ── Provider Configuration ─────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Debug logging
import sys
print(f"[LLM Config] Provider: {LLM_PROVIDER}", file=sys.stderr)
print(f"[LLM Config] Anthropic API Key Set: {bool(ANTHROPIC_API_KEY)}", file=sys.stderr)
print(f"[LLM Config] Gemini API Key Set: {bool(GEMINI_API_KEY)}", file=sys.stderr)
print(f"[LLM Config] OpenRouter API Key Set: {bool(OPENROUTER_API_KEY)}", file=sys.stderr)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL = "gemini-2.0-flash"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Tool schemas (enforce structured output) ──────────────────────────────────

INVOICE_TOOL = {
    "name": "extract_invoice",
    "description": "Extract all structured fields from an invoice or purchase order PDF.",
    "input_schema": {
        "type": "object",
        "properties": {
            "vendor_name": {"type": ["string", "null"]},
            "vendor_address": {"type": ["string", "null"]},
            "client_name": {"type": ["string", "null"]},
            "client_address": {"type": ["string", "null"]},
            "invoice_number": {"type": ["string", "null"]},
            "invoice_date": {"type": ["string", "null"]},
            "due_date": {"type": ["string", "null"]},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "quantity": {"type": ["number", "null"]},
                        "unit_price": {"type": ["number", "null"]},
                        "amount": {"type": ["number", "null"]},
                    },
                },
            },
            "subtotal": {"type": ["number", "null"]},
            "tax": {"type": ["number", "null"]},
            "discount": {"type": ["number", "null"]},
            "total_amount": {"type": ["number", "null"]},
            "currency": {"type": ["string", "null"]},
            "payment_terms": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
            "summary": {"type": ["string", "null"]},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
    },
}

RESUME_TOOL = {
    "name": "extract_resume",
    "description": "Extract all structured fields from a resume or CV PDF.",
    "input_schema": {
        "type": "object",
        "properties": {
            "full_name": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "location": {"type": ["string", "null"]},
            "linkedin": {"type": ["string", "null"]},
            "github": {"type": ["string", "null"]},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": ["string", "null"]},
                        "title": {"type": ["string", "null"]},
                        "start_date": {"type": ["string", "null"]},
                        "end_date": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                    },
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution": {"type": ["string", "null"]},
                        "degree": {"type": ["string", "null"]},
                        "field_of_study": {"type": ["string", "null"]},
                        "year": {"type": ["string", "null"]},
                    },
                },
            },
            "certifications": {"type": "array", "items": {"type": "string"}},
            "languages": {"type": "array", "items": {"type": "string"}},
            "total_years_experience": {"type": ["number", "null"]},
            "summary": {"type": ["string", "null"]},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
    },
}

BANK_STATEMENT_TOOL = {
    "name": "extract_bank_statement",
    "description": "Extract all structured fields from a bank or financial statement PDF.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bank_name": {"type": ["string", "null"]},
            "account_holder": {"type": ["string", "null"]},
            "account_number_masked": {"type": ["string", "null"], "description": "Last 4 digits only for security"},
            "account_type": {"type": ["string", "null"]},
            "statement_period_start": {"type": ["string", "null"]},
            "statement_period_end": {"type": ["string", "null"]},
            "opening_balance": {"type": ["number", "null"]},
            "closing_balance": {"type": ["number", "null"]},
            "total_credits": {"type": ["number", "null"]},
            "total_debits": {"type": ["number", "null"]},
            "currency": {"type": ["string", "null"]},
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "debit": {"type": ["number", "null"]},
                        "credit": {"type": ["number", "null"]},
                        "balance": {"type": ["number", "null"]},
                    },
                },
            },
            "summary": {"type": ["string", "null"]},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
    },
}

DOC_TYPE_TO_TOOL = {
    DocumentType.INVOICE: INVOICE_TOOL,
    DocumentType.RESUME: RESUME_TOOL,
    DocumentType.BANK_STATEMENT: BANK_STATEMENT_TOOL,
}


# ── Gemini Schema Conversion ───────────────────────────────────────────────────
def _convert_to_gemini_schema(tool: dict) -> dict:
    """Convert Anthropic tool schema to Google Gemini function calling format."""
    properties = tool["input_schema"]["properties"]
    
    def convert_property(value):
        """Convert a single property definition from Anthropic to Gemini format."""
        # Handle arrays
        if value.get("type") == "array":
            items = value.get("items", {})
            item_type = items.get("type")
            if isinstance(item_type, list):
                # Get first non-null type
                item_type = next((t for t in item_type if t != "null"), "string").upper()
            else:
                item_type = (item_type or "string").upper()
            
            item_props = items.get("properties", {})
            if item_props:
                # Array of objects
                converted_props = {}
                for key, val in item_props.items():
                    converted_props[key] = convert_property(val)
                return {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": converted_props
                    }
                }
            else:
                # Array of simple types
                return {
                    "type": "ARRAY",
                    "items": {"type": item_type}
                }
        
        # Handle type arrays (like ["string", "null"])
        type_val = value.get("type")
        if isinstance(type_val, list):
            # Filter out "null" and get the actual type
            actual_type = next((t for t in type_val if t != "null"), "string")
        else:
            actual_type = type_val or "string"
        
        # Map JSON types to Gemini types
        type_map = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
        }
        
        gemini_type = type_map.get(actual_type, "STRING")
        
        # Handle nested objects
        if actual_type == "object" and "properties" in value:
            converted_props = {}
            for key, val in value.get("properties", {}).items():
                converted_props[key] = convert_property(val)
            return {
                "type": "OBJECT",
                "properties": converted_props
            }
        
        return {"type": gemini_type}
    
    # Convert all properties
    gemini_properties = {}
    for key, val in properties.items():
        gemini_properties[key] = convert_property(val)
    
    # Build required list (fields without "null" in their type array)
    required = []
    for key, value in properties.items():
        type_val = value.get("type")
        if isinstance(type_val, list) and "null" not in type_val:
            required.append(key)
        elif isinstance(type_val, str) and type_val != "null":
            required.append(key)
    
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": {
            "type": "OBJECT",
            "properties": gemini_properties,
            "required": required if required else [],
        },
    }


# ── OpenRouter Schema Conversion ───────────────────────────────────────────────
def _convert_to_openrouter_schema(tool: dict) -> dict:
    """Convert Anthropic tool schema to OpenAI/OpenRouter function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
    }


def _build_prompt(parsed: ParsedPDF, doc_type: DocumentType) -> str:
    tables_text = extract_tables_as_text(parsed.tables)
    content = parsed.full_text[:12000]  # cap to avoid huge payloads
    if tables_text:
        content += f"\n\n{tables_text[:4000]}"

    instructions = {
        DocumentType.INVOICE: (
            "This is an invoice or purchase order. Extract every field you can find. "
            "For numbers, return numeric values only (no currency symbols). "
            "If a field is not present, return null — never guess or hallucinate."
        ),
        DocumentType.RESUME: (
            "This is a resume or CV. Extract all personal info, work history, education, "
            "and skills. Estimate total_years_experience from the date ranges. "
            "If a field is absent, return null."
        ),
        DocumentType.BANK_STATEMENT: (
            "This is a bank statement. Extract account details, balance info, and "
            "all transactions. For account_number_masked return last 4 digits only. "
            "Amounts should be positive numbers — use the debit/credit fields to indicate direction. "
            "If a field is absent, return null."
        ),
    }

    return (
        f"{instructions[doc_type]}\n\n"
        f"Also write a plain-English summary (2–3 sentences) of what this document is about.\n\n"
        f"Document text:\n{content}"
    )


async def extract_with_llm(
    parsed: ParsedPDF,
    doc_type: DocumentType,
    confidence: float,
) -> InvoiceExtraction | ResumeExtraction | BankStatementExtraction | None:
    """
    Route to the appropriate LLM provider (Anthropic, Gemini, or OpenRouter).
    Falls back gracefully if API is unavailable or returns an error.
    """
    print(f"[Extract] Using LLM provider: {LLM_PROVIDER}", file=sys.stderr)
    
    if LLM_PROVIDER == "openrouter":
        print("[Extract] Routing to OpenRouter", file=sys.stderr)
        return await _extract_with_openrouter(parsed, doc_type, confidence)
    elif LLM_PROVIDER == "gemini":
        print("[Extract] Routing to Gemini", file=sys.stderr)
        return await _extract_with_gemini(parsed, doc_type, confidence)
    else:
        print("[Extract] Routing to Anthropic", file=sys.stderr)
        return await _extract_with_anthropic(parsed, doc_type, confidence)


async def _extract_with_anthropic(
    parsed: ParsedPDF,
    doc_type: DocumentType,
    confidence: float,
) -> InvoiceExtraction | ResumeExtraction | BankStatementExtraction | None:
    """
    Call Claude with tool_use to enforce schema — the model MUST populate
    the tool's input_schema, making JSON parsing failures structurally impossible.
    Falls back gracefully if API is unavailable or returns an error.
    """
    if not ANTHROPIC_API_KEY:
        print("[Anthropic] No API key found", file=sys.stderr)
        return None

    tool = DOC_TYPE_TO_TOOL.get(doc_type)
    if not tool:
        print(f"[Anthropic] No tool found for {doc_type}", file=sys.stderr)
        return None

    prompt = _build_prompt(parsed, doc_type)

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2048,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"[Anthropic] Calling API", file=sys.stderr)
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )

        print(f"[Anthropic] Response status: {response.status_code}", file=sys.stderr)
        
        if response.status_code != 200:
            error_text = response.text
            print(f"[Anthropic] API Error: {error_text}", file=sys.stderr)
            return None

        data = response.json()

        # Find the tool_use block — this is guaranteed to be valid JSON
        tool_input = None
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                tool_input = block.get("input", {})
                print(f"[Anthropic] Found tool_use with {len(tool_input)} fields", file=sys.stderr)
                break

        if not tool_input:
            print("[Anthropic] No tool_use found in response", file=sys.stderr)
            return None

        # Build typed schema from LLM output
        base_fields = {
            "document_type": doc_type,
            "confidence": confidence,
            "extraction_method": ExtractionMethod.LLM,
            "raw_text_length": len(parsed.full_text),
            "summary": tool_input.pop("summary", None),
            "warnings": tool_input.pop("warnings", []),
        }

        print(f"[Anthropic] Extraction successful!", file=sys.stderr)

        print(f"[Anthropic] Extraction successful!", file=sys.stderr)

        if doc_type == DocumentType.INVOICE:
            return InvoiceExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.RESUME:
            return ResumeExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.BANK_STATEMENT:
            return BankStatementExtraction(**base_fields, **tool_input)

    except Exception as e:
        print(f"[Anthropic] Exception: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None


async def _extract_with_gemini(
    parsed: ParsedPDF,
    doc_type: DocumentType,
    confidence: float,
) -> InvoiceExtraction | ResumeExtraction | BankStatementExtraction | None:
    """
    Call Google Gemini with function calling to enforce schema.
    Falls back gracefully if API is unavailable or returns an error.
    """
    if not GEMINI_API_KEY:
        print("[Gemini] No API key found", file=sys.stderr)
        return None

    tool = DOC_TYPE_TO_TOOL.get(doc_type)
    if not tool:
        print(f"[Gemini] No tool found for {doc_type}", file=sys.stderr)
        return None

    prompt = _build_prompt(parsed, doc_type)
    gemini_tool = _convert_to_gemini_schema(tool)
    
    print(f"[Gemini] Tool schema (first 500 chars): {str(gemini_tool)[:500]}", file=sys.stderr)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "tools": [
            {
                "function_declarations": [gemini_tool],
            }
        ],
        "tool_config": {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tool["name"]],
            }
        },
        "generationConfig": {
            "maxOutputTokens": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = GEMINI_API_URL.format(model=GEMINI_MODEL)
            print(f"[Gemini] Calling API: {url}", file=sys.stderr)
            response = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        print(f"[Gemini] Response status: {response.status_code}", file=sys.stderr)
        
        if response.status_code != 200:
            error_text = response.text
            print(f"[Gemini] API Error: {error_text}", file=sys.stderr)
            return None

        data = response.json()
        print(f"[Gemini] Response preview: {str(data)[:200]}...", file=sys.stderr)

        # Extract function call from Gemini response
        tool_input = None
        for content in data.get("candidates", []):
            for part in content.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    tool_input = part["functionCall"].get("args", {})
                    print(f"[Gemini] Found function call with {len(tool_input)} fields", file=sys.stderr)
                    break
            if tool_input:
                break

        if not tool_input:
            print("[Gemini] No function call found in response", file=sys.stderr)
            return None

        # Build typed schema from LLM output
        base_fields = {
            "document_type": doc_type,
            "confidence": confidence,
            "extraction_method": ExtractionMethod.LLM,
            "raw_text_length": len(parsed.full_text),
            "summary": tool_input.pop("summary", None),
            "warnings": tool_input.pop("warnings", []),
        }
        
        print(f"[Gemini] Extraction successful!", file=sys.stderr)

        if doc_type == DocumentType.INVOICE:
            return InvoiceExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.RESUME:
            return ResumeExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.BANK_STATEMENT:
            return BankStatementExtraction(**base_fields, **tool_input)

    except Exception as e:
        print(f"[Gemini] Exception: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return None


async def _extract_with_openrouter(
    parsed: ParsedPDF,
    doc_type: DocumentType,
    confidence: float,
) -> InvoiceExtraction | ResumeExtraction | BankStatementExtraction | None:
    """
    Call OpenRouter API with function calling to enforce schema.
    Falls back gracefully if API is unavailable or returns an error.
    """
    if not OPENROUTER_API_KEY:
        print("[OpenRouter] No API key found", file=sys.stderr)
        return None

    tool = DOC_TYPE_TO_TOOL.get(doc_type)
    if not tool:
        print(f"[OpenRouter] No tool found for {doc_type}", file=sys.stderr)
        return None

    prompt = _build_prompt(parsed, doc_type)
    openrouter_tool = _convert_to_openrouter_schema(tool)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [openrouter_tool],
        "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"[OpenRouter] Calling API", file=sys.stderr)
            response = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-username/smart-report-extractor",
                    "X-Title": "Smart Report Extractor",
                },
                json=payload,
            )

        print(f"[OpenRouter] Response status: {response.status_code}", file=sys.stderr)
        
        if response.status_code != 200:
            error_text = response.text
            print(f"[OpenRouter] API Error: {error_text}", file=sys.stderr)
            return None

        data = response.json()
        
        choices = data.get("choices", [])
        if not choices:
            print("[OpenRouter] No choices returned", file=sys.stderr)
            return None
            
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        tool_input = None
        if tool_calls:
            # We enforce the specific tool
            for call in tool_calls:
                func = call.get("function", {})
                if func and func.get("name") == tool.get("name"):
                    args_str = func.get("arguments", "{}")
                    try:
                        tool_input = json.loads(args_str)
                    except json.JSONDecodeError:
                        print(f"[OpenRouter] Failed to parse JSON arguments: {args_str}", file=sys.stderr)
                        return None
                    break

        if not tool_input:
            print("[OpenRouter] No valid function call arguments found", file=sys.stderr)
            return None

        # Build typed schema from LLM output
        base_fields = {
            "document_type": doc_type,
            "confidence": confidence,
            "extraction_method": ExtractionMethod.LLM,
            "raw_text_length": len(parsed.full_text),
            "summary": tool_input.pop("summary", None),
            "warnings": tool_input.pop("warnings", []),
        }

        print(f"[OpenRouter] Extraction successful!", file=sys.stderr)

        if doc_type == DocumentType.INVOICE:
            return InvoiceExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.RESUME:
            return ResumeExtraction(**base_fields, **tool_input)
        elif doc_type == DocumentType.BANK_STATEMENT:
            return BankStatementExtraction(**base_fields, **tool_input)

    except Exception as e:
        print(f"[OpenRouter] Exception: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return None
