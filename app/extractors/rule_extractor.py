"""
Rule-based extractor — the fallback when Claude API is unavailable or fails.

Design philosophy:
- Regex handles predictable, templated documents well (most invoices and statements are).
- For resumes (freeform), extraction is best-effort with section detection.
- Every extraction returns partial results rather than raising — the caller
  marks warnings for missing fields.
- This runs synchronously (no I/O), so it's always fast.
"""
import re
from app.schemas.base import DocumentType, ExtractionMethod
from app.schemas import InvoiceExtraction, ResumeExtraction, BankStatementExtraction
from app.utils.pdf_parser import ParsedPDF


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _find(pattern: str, text: str, group: int = 1, flags=re.IGNORECASE) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _find_amount(pattern: str, text: str) -> float | None:
    raw = _find(pattern, text)
    if raw:
        cleaned = re.sub(r"[,$£€\s]", "", raw)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _summarize_fallback(doc_type: DocumentType, data: dict) -> str:
    """Generate a minimal plain-English summary from extracted fields."""
    if doc_type == DocumentType.INVOICE:
        vendor = data.get("vendor_name") or "Unknown vendor"
        total = data.get("total_amount")
        inv_num = data.get("invoice_number") or "unknown"
        total_str = f"${total:,.2f}" if total else "an unspecified amount"
        return (
            f"Invoice #{inv_num} from {vendor} for {total_str}. "
            f"Extracted using rule-based parsing — LLM was unavailable."
        )
    elif doc_type == DocumentType.RESUME:
        name = data.get("full_name") or "Unknown candidate"
        skills = data.get("skills", [])
        skill_str = ", ".join(skills[:4]) if skills else "not detected"
        return (
            f"Resume for {name}. Key skills include: {skill_str}. "
            f"Extracted using rule-based parsing — LLM was unavailable."
        )
    elif doc_type == DocumentType.BANK_STATEMENT:
        holder = data.get("account_holder") or "Unknown holder"
        closing = data.get("closing_balance")
        balance_str = f"${closing:,.2f}" if closing else "unspecified"
        return (
            f"Bank statement for {holder} with closing balance {balance_str}. "
            f"Extracted using rule-based parsing — LLM was unavailable."
        )
    return "Document extracted using rule-based parsing."


# ── Invoice ────────────────────────────────────────────────────────────────────

def _extract_invoice_rules(text: str, tables: list) -> dict:
    data: dict = {}
    warnings: list[str] = []

    # ── Invoice Number ──────────────────────────────────────────────────────
    data["invoice_number"] = (
        _find(r"invoice\s*(?:#|no\.?|number)\s*:?\s*([A-Z0-9\-]+)", text)
        or _find(r"inv(?:oice)?\s*(?:#|no\.?)\s*:?\s*([A-Z0-9\-]+)", text)
        or _find(r"^invoice\s*#?\s*([A-Z0-9\-]+)", text, flags=re.IGNORECASE | re.MULTILINE)
    )
    if not data["invoice_number"]:
        warnings.append("invoice_number not found")

    # ── Invoice Date ───────────────────────────────────────────────────────
    data["invoice_date"] = (
        _find(r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text)
        or _find(r"date(?:\s*of\s*invoice)?\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text)
        or _find(r"^date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text, flags=re.MULTILINE)
    )

    # ── Due Date ────────────────────────────────────────────────────────────
    data["due_date"] = (
        _find(r"due\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text)
        or _find(r"payment\s*due\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text)
        or _find(r"(?:due\s*)?by\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text)
    )

    # ── Vendor Name ─────────────────────────────────────────────────────────
    data["vendor_name"] = (
        _find(r"(?:from|bill(?:ed)?\s+from|vendor|company|seller)\s*:?\s*\n?\s*([^\n]+?)(?:\n|$)", text)
        or _find(r"^([A-Z][A-Za-z0-9\s,&.]+?)(?:\n|$)", text, flags=re.MULTILINE)  # First capitalized line
    )

    # ── Vendor Address ──────────────────────────────────────────────────────
    data["vendor_address"] = _find(
        r"(?:from|bill(?:ed)?\s+from|vendor|company)[^:]*:?\s*\n+(.+?)\n+(?:to|bill\s+to|invoice\s+to|client|customer)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # ── Client Name ─────────────────────────────────────────────────────────
    data["client_name"] = (
        _find(r"(?:to|bill\s+to|bill(?:ed)?\s+to|invoice\s+to|customer|client)\s*:?\s*\n?\s*([^\n]+?)(?:\n|$)", text)
        or _find(r"(?:for|sold to)\s*:?\s*\n?\s*([^\n]+?)(?:\n|$)", text)
    )

    # ── Client Address ──────────────────────────────────────────────────────
    data["client_address"] = _find(
        r"(?:to|bill\s+to|invoice\s+to|customer|client)[^:]*:?\s*\n+(?:[^\n]+)\n+(.+?)\n",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # ── Total Amount ────────────────────────────────────────────────────────
    data["total_amount"] = (
        _find_amount(r"(?:total\s*(?:amount)?\s*(?:due|owed)?|grand\s*total|amount\s*due)\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text)
        or _find_amount(r"(?:^|\n)\s*(?:total|total\s*amount)\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text, flags=re.MULTILINE)
        or _find_amount(r"total\s+\$?\s*([\d,]+\.?\d{0,2})(?:\n|$)", text)
    )
    if not data["total_amount"]:
        warnings.append("total_amount not found")

    # ── Subtotal ────────────────────────────────────────────────────────────
    data["subtotal"] = (
        _find_amount(r"sub\s*total\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text)
        or _find_amount(r"(?:^|\n)\s*subtotal\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text, flags=re.MULTILINE)
    )

    # ── Tax ──────────────────────────────────────────────────────────────────
    data["tax"] = (
        _find_amount(r"(?:tax|vat|gst|sales\s*tax)\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text)
        or _find_amount(r"(?:^|\n)\s*tax\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text, flags=re.MULTILINE)
    )

    # ── Discount ─────────────────────────────────────────────────────────────
    data["discount"] = _find_amount(r"discount\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})", text)

    # ── Currency ─────────────────────────────────────────────────────────────
    currency_match = re.search(r"(USD|EUR|GBP|INR|AUD|CAD|JPY|\$|€|£|₹|¥)", text)
    if currency_match:
        sym_map = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
        raw = currency_match.group(1)
        data["currency"] = sym_map.get(raw, raw)

    # ── Payment Terms ────────────────────────────────────────────────────────
    data["payment_terms"] = (
        _find(r"(?:payment\s*terms?|terms?)\s*:?\s*(.+?)(?:\n|$)", text)
        or _find(r"(?:net|due)\s+(\d+)\s*(?:days?)", text)
    )

    # ── Notes ────────────────────────────────────────────────────────────────
    data["notes"] = _find(r"(?:notes?|memo|comments?)\s*:?\s*(.+?)(?:\n\n|$)", text, flags=re.DOTALL)

    # ── Line Items from Tables ──────────────────────────────────────────────
    line_items = []
    for table in tables:
        for row in table:
            if len(row) < 2:
                continue
            desc = row[0].strip() if row[0] else None
            
            # Skip header rows
            if desc and desc.lower() in ("description", "item", "qty", "quantity", "unit price", "price", "amount", "total", ""):
                continue
            
            if not desc or len(desc) < 2:
                continue

            # Extract amounts
            amounts = []
            for cell in row[1:]:
                if not cell:
                    continue
                try:
                    val = float(re.sub(r"[,$£€\s]", "", str(cell)))
                    amounts.append(val)
                except (ValueError, TypeError):
                    pass

            item = {"description": desc}
            
            # Try to parse qty, unit_price, amount from amounts array
            if len(amounts) >= 3:
                item["quantity"] = amounts[0]
                item["unit_price"] = amounts[1]
                item["amount"] = amounts[2]
            elif len(amounts) == 2:
                item["quantity"] = amounts[0]
                item["amount"] = amounts[1]
            elif len(amounts) == 1:
                item["amount"] = amounts[0]
            
            line_items.append(item)

    data["line_items"] = line_items[:20]

    data["warnings"] = warnings
    return data


# ── Resume ─────────────────────────────────────────────────────────────────────

def _extract_resume_rules(text: str) -> dict:
    data: dict = {}
    warnings: list[str] = []

    email = _find(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", text, group=0, flags=0)
    data["email"] = email

    phone = _find(
        r"(\+?\d[\d\s\-().]{7,}\d)", text, group=1
    )
    data["phone"] = phone

    linkedin = _find(r"(linkedin\.com/in/[\w\-]+)", text, group=0, flags=re.IGNORECASE)
    data["linkedin"] = f"https://{linkedin}" if linkedin else None

    github = _find(r"(github\.com/[\w\-]+)", text, group=0, flags=re.IGNORECASE)
    data["github"] = f"https://{github}" if github else None

    # Skills: look for a skills section and grab keywords
    skills_block = _find(
        r"(?:skills?|technical\s*skills?|core\s*competencies?)[:\n\s]+([^\n]+(?:\n[^\n]+){0,8})",
        text,
        group=1,
    )
    skills = []
    if skills_block:
        raw_skills = re.split(r"[,•|·/\n\t]+", skills_block)
        skills = [s.strip() for s in raw_skills if 2 < len(s.strip()) < 40][:20]
    data["skills"] = skills
    if not skills:
        warnings.append("skills section not detected clearly")

    # Education
    education = []
    edu_pattern = re.finditer(
        r"(bachelor|master|b\.?s\.?|m\.?s\.?|ph\.?d|b\.?e|b\.?tech|m\.?tech)[^,\n]*[,\s]+([^\n,]+)?[,\s]*(\d{4})?",
        text,
        re.IGNORECASE,
    )
    for m in edu_pattern:
        education.append({
            "degree": m.group(0).strip(),
            "institution": None,
            "field_of_study": None,
            "year": m.group(3),
        })
    data["education"] = education[:4]

    data["warnings"] = warnings
    return data


# ── Bank Statement ─────────────────────────────────────────────────────────────

def _extract_bank_rules(text: str, tables: list) -> dict:
    data: dict = {}
    warnings: list[str] = []

    data["account_number_masked"] = _find(
        r"(?:account\s*(?:no|number|num))[:\s#]*[\*xX]*(\d{4})", text
    )
    data["opening_balance"] = _find_amount(
        r"(?:opening|beginning)\s*balance\s*:?\s*\$?([\d,]+\.?\d{0,2})", text
    )
    data["closing_balance"] = _find_amount(
        r"(?:closing|ending)\s*balance\s*:?\s*\$?([\d,]+\.?\d{0,2})", text
    )
    data["total_credits"] = _find_amount(
        r"(?:total\s*credits?|total\s*deposits?)\s*:?\s*\$?([\d,]+\.?\d{0,2})", text
    )
    data["total_debits"] = _find_amount(
        r"(?:total\s*debits?|total\s*withdrawals?)\s*:?\s*\$?([\d,]+\.?\d{0,2})", text
    )

    # Statement period
    period = _find(
        r"(?:statement\s*(?:period|date)|for\s*the\s*period)\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        text,
    )
    data["statement_period_start"] = period

    # Currency
    currency_match = re.search(r"(USD|EUR|GBP|INR|AUD|CAD)", text)
    data["currency"] = currency_match.group(1) if currency_match else None

    # Transactions from tables
    transactions = []
    date_re = re.compile(r"^\d{1,2}[\/\-\.]\d{1,2}([\/\-\.]\d{2,4})?$")
    for table in tables:
        for row in table:
            if not row:
                continue
            date_cell = row[0].strip() if row[0] else ""
            if date_re.match(date_cell):
                desc = row[1].strip() if len(row) > 1 and row[1] else None
                amounts = []
                for cell in row[2:]:
                    try:
                        val = float(re.sub(r"[,$£€\s]", "", cell))
                        amounts.append(val)
                    except (ValueError, TypeError):
                        amounts.append(None)

                txn: dict = {"date": date_cell, "description": desc}
                if len(amounts) >= 3:
                    txn["debit"] = amounts[0]
                    txn["credit"] = amounts[1]
                    txn["balance"] = amounts[2]
                elif len(amounts) == 2:
                    txn["debit"] = amounts[0]
                    txn["balance"] = amounts[1]
                elif len(amounts) == 1:
                    txn["balance"] = amounts[0]
                transactions.append(txn)

    data["transactions"] = transactions[:50]
    if not transactions:
        warnings.append("transaction table not detected — PDF may be scanned or image-based")

    if not data["opening_balance"]:
        warnings.append("opening_balance not found")
    if not data["closing_balance"]:
        warnings.append("closing_balance not found")

    data["warnings"] = warnings
    return data


# ── Public entry point ─────────────────────────────────────────────────────────

def extract_with_rules(
    parsed: ParsedPDF,
    doc_type: DocumentType,
    confidence: float,
) -> InvoiceExtraction | ResumeExtraction | BankStatementExtraction:
    text = parsed.full_text
    tables = parsed.tables

    if doc_type == DocumentType.INVOICE:
        data = _extract_invoice_rules(text, tables)
    elif doc_type == DocumentType.RESUME:
        data = _extract_resume_rules(text)
    elif doc_type == DocumentType.BANK_STATEMENT:
        data = _extract_bank_rules(text, tables)
    else:
        data = {"warnings": ["Unknown document type — no rules applied"]}

    warnings = data.pop("warnings", [])
    summary = _summarize_fallback(doc_type, data)

    base = {
        "document_type": doc_type,
        "confidence": confidence,
        "extraction_method": ExtractionMethod.RULE_BASED,
        "raw_text_length": len(text),
        "summary": summary,
        "warnings": warnings,
    }

    if doc_type == DocumentType.INVOICE:
        return InvoiceExtraction(**base, **data)
    elif doc_type == DocumentType.RESUME:
        return ResumeExtraction(**base, **data)
    elif doc_type == DocumentType.BANK_STATEMENT:
        return BankStatementExtraction(**base, **data)
    else:
        return InvoiceExtraction(**base)
