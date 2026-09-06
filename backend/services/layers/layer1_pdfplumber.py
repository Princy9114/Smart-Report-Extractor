"""
layer1_pdfplumber.py
~~~~~~~~~~~~~~~~~~~~
Extraction Layer 1: pdfplumber-derived signals.

Uses the structured outputs already obtained from pdfplumber (plain text and
tables) to pull field values via:
  - Label-adjacent regex scanning (invoice / bank-statement scalar fields)
  - Table parsing (line items, transactions)
  - Heading-heuristic on text layout (resumes)

Font-size signals are a pdfplumber concept that lives on ``page.chars``.
Because this layer receives only text and tables (already serialised), it
approximates font-size prominence via ALL-CAPS / title-case line detection,
which closely mirrors what a true font-size filter would select as headings.
If the caller has access to raw ``pdfplumber.Page`` objects, pass them via the
``pages`` keyword to unlock real font-size-based heading detection.
"""

from __future__ import annotations

import re
from typing import Any

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ExtractionResult = dict[str, FieldResult]

# ---------------------------------------------------------------------------
# Shared regex helpers
# ---------------------------------------------------------------------------

def _first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    """Return the first capturing group of *pattern* in *text*, or None."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _label_value(label: str, text: str) -> str | None:
    """
    Find the value that appears immediately after a label on the same line.

    Accepts separators: ``:``, ``#``, ``-``, or plain whitespace after the label.
    Example:   "Invoice Number: INV-2024-001"  →  "INV-2024-001"
    """
    pattern = rf"(?i){re.escape(label)}\s*[:#\-]?\s*(.+)"
    return _first_match(pattern, text)


# ---------------------------------------------------------------------------
# ── INVOICE ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Labels tried in order; first match wins.
_INV_NUMBER_LABELS = [
    "invoice number", "invoice no", "invoice #", "invoice id", "inv no", "inv #",
]
_INV_DATE_LABELS = [
    "invoice date", "date of invoice", "issue date", "billing date", "date",
]
_INV_TOTAL_LABELS = [
    "total due", "amount due", "total amount", "grand total", "balance due", "total",
]


_NON_VENDOR_HEADERS = {
    "tax invoice", "invoice", "bill of supply", "commercial invoice", "original",
    "original for recipient", "duplicate", "triplicate", "proforma invoice", "receipt",
    "bill", "statement", "credit note", "debit note",
}


def _extract_invoice_vendor(text: str) -> tuple[str, float] | None:
    """Extract vendor name from top header lines of invoice."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:6]:
        low = line.lower()
        if low in _NON_VENDOR_HEADERS:
            continue
        if any(kw in low for kw in ("gstin", "invoice no", "inv no", "date:", "bill to", "ship to", "phone:", "email:")):
            continue
        if re.search(r"\b(?:pvt|ltd|inc|llc|corp|enterprises|solutions|services|technologies|company|co\.|industries)\b", low):
            return line, 0.88
        words = line.split()
        if 2 <= len(words) <= 6 and not re.search(r"\d", line):
            return line, 0.82
    return None


def _extract_invoice(
    text: str,
    tables: list[list[list[str | None]]],
) -> ExtractionResult:
    result: ExtractionResult = {}

    # ── vendor from top header block ─────────────────────────────────────────
    vendor_info = _extract_invoice_vendor(text)
    if vendor_info:
        vendor_val, vendor_conf = vendor_info
        result["vendor"] = FieldResult(
            value=vendor_val,
            confidence=vendor_conf,
            source="header_positional",
        )

    # ── scalar fields via label-adjacent search ──────────────────────────────
    for field_name, labels in [
        ("invoice_number", _INV_NUMBER_LABELS),
        ("date",           _INV_DATE_LABELS),
        ("total",          _INV_TOTAL_LABELS),
    ]:
        for label in labels:
            raw = _label_value(label, text)
            if raw:
                # Strip trailing noise (e.g. extra columns bled into the line)
                value = re.split(r"\s{2,}|\t", raw)[0].strip()

                if field_name == "total":
                    # Check if value is a unit-of-measure / quantity like '2 NOS', '5 PCS'
                    if re.search(r"\b(?:nos|no|pcs|pieces|qty|quantity|items|units|kg|meters)\b", value, re.IGNORECASE):
                        # Search if an actual monetary amount exists elsewhere on the line
                        amt_match = re.search(r"(?:[₹$€£¥]|INR|Rs\.?)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})|[0-9]{2,}(?:\.[0-9]{2})?)", raw)
                        if amt_match:
                            value = amt_match.group(0).strip()
                        else:
                            # Skip this non-monetary match and try next label
                            continue

                result[field_name] = FieldResult(
                    value=value,
                    confidence=0.85,
                    source="label_adjacent",
                    raw=raw,
                )
                break

    # ── line items from tables ───────────────────────────────────────────────
    line_items: list[dict[str, str]] = []
    for table in tables:
        if not table or len(table) < 2:
            continue

        # Detect header row: look for a row containing "description"/"item"
        header_row: list[str | None] | None = None
        data_start = 0
        for idx, row in enumerate(table):
            cells = [str(c).lower().strip() if c else "" for c in row]
            if any(kw in cells for kw in ("description", "item", "product", "service")):
                header_row = row
                data_start = idx + 1
                break

        if header_row is None:
            # Use the first row as a fallback header
            header_row = table[0]
            data_start = 1

        headers = [str(h).strip().lower() if h else f"col{i}"
                   for i, h in enumerate(header_row)]

        for row in table[data_start:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            item: dict[str, str] = {}
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers):
                    item[headers[col_idx]] = str(cell).strip() if cell else ""
            if any(v for v in item.values()):
                line_items.append(item)

    if line_items:
        result["line_items"] = FieldResult(
            value=line_items,
            confidence=0.90,
            source="table",
        )

    return result


# ---------------------------------------------------------------------------
# ── BANK STATEMENT ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_BANK_SCALAR_LABELS: list[tuple[str, list[str]]] = [
    ("account_number",   ["account number", "account no", "acc no", "account #"]),
    ("account_name",     ["account name", "account holder"]),
    ("opening_balance",  ["opening balance", "beginning balance", "balance brought forward"]),
    ("closing_balance",  ["closing balance", "ending balance", "balance carried forward"]),
    ("statement_period", ["statement period", "period", "statement date", "from"]),
    ("bank_name",        ["bank name", "bank"]),
]

# Column keywords that signal a transaction table
_TXN_COL_SIGNALS = {"date", "description", "debit", "credit", "balance", "amount", "particulars"}


def _extract_bank_statement(
    text: str,
    tables: list[list[list[str | None]]],
) -> ExtractionResult:
    result: ExtractionResult = {}

    # ── scalar fields ────────────────────────────────────────────────────────
    for field_name, labels in _BANK_SCALAR_LABELS:
        for label in labels:
            raw = _label_value(label, text)
            if raw:
                value = re.split(r"\s{2,}|\t", raw)[0].strip()
                result[field_name] = FieldResult(
                    value=value,
                    confidence=0.85,
                    source="label_adjacent",
                    raw=raw,
                )
                break

    # ── transaction table ────────────────────────────────────────────────────
    best_table: list[list[str | None]] | None = None
    best_signal_count = 0

    for table in tables:
        if not table:
            continue
        # Score the first row as a header candidate
        first_row = [str(c).lower().strip() if c else "" for c in table[0]]
        hits = sum(1 for cell in first_row if cell in _TXN_COL_SIGNALS)
        if hits > best_signal_count:
            best_signal_count = hits
            best_table = table

    if best_table and best_signal_count >= 2:
        header_row = best_table[0]
        headers = [str(h).strip().lower() if h else f"col{i}"
                   for i, h in enumerate(header_row)]

        transactions: list[dict[str, str]] = []
        for row in best_table[1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            txn: dict[str, str] = {}
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers):
                    txn[headers[col_idx]] = str(cell).strip() if cell else ""
            if any(v for v in txn.values()):
                transactions.append(txn)

        if transactions:
            result["transactions"] = FieldResult(
                value=transactions,
                confidence=0.90,
                source="table",
            )

    return result


# ---------------------------------------------------------------------------
# ── RESUME ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Well-known resume section heading names (case-insensitive)
_RESUME_SECTIONS = [
    "education", "work experience", "experience", "employment history",
    "professional experience", "skills", "technical skills", "certifications",
    "awards", "achievements", "projects", "publications", "volunteer",
    "languages", "interests", "hobbies", "references", "summary",
    "professional summary", "objective", "profile", "contact",
]

# Pre-compiled patterns
_SECTION_PATTERN = re.compile(
    r"^("
    + "|".join(re.escape(s) for s in _RESUME_SECTIONS)
    + r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ALL-CAPS line heuristic (≥ 3 alpha chars, no lowercase) — high font-size proxy
_ALLCAPS_PATTERN = re.compile(r"^[A-Z][A-Z &/\-]{2,}$")

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_URL_PATTERN   = re.compile(r"(https?://[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+)", re.IGNORECASE)


def _detect_headings_from_text(text: str) -> list[str]:
    """
    Approximate font-size-based heading detection from plain text.

    Strategy (mirrors what pdfplumber ``page.chars`` font-size filtering does):
    1. Named section headings matched against a known vocabulary.
    2. ALL-CAPS short lines (≤ 6 words) — typically rendered in a larger/bold font.

    Parameters
    ----------
    text:
        Full extracted text of the resume.

    Returns
    -------
    list[str]
        Unique ordered list of detected section headings.
    """
    headings: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.upper()
        if key in seen:
            continue
        # Priority 1: vocabulary match
        if _SECTION_PATTERN.match(stripped):
            headings.append(stripped)
            seen.add(key)
            continue
        # Priority 2: ALL-CAPS heuristic (short lines only)
        words = stripped.split()
        if 1 <= len(words) <= 6 and _ALLCAPS_PATTERN.match(stripped):
            headings.append(stripped)
            seen.add(key)

    return headings


_NON_NAME_HEADERS = {
    "resume", "curriculum vitae", "cv", "biodata", "bio-data", "bio data",
    "portfolio", "profile", "contact", "contact info", "personal details",
    "personal information", "about me", "summary", "professional summary",
    "objective", "career objective",
}

_ROLE_OR_TECH_WORDS = {
    "developer", "engineer", "scientist", "analyst", "architect", "designer",
    "manager", "consultant", "intern", "student", "specialist", "administrator",
    "lead", "full stack", "frontend", "backend", "data", "software", "machine learning",
    "artificial intelligence", "deep learning", "python", "java", "react", "c++",
    "fastapi", "docker", "aws", "cloud", "devops",
}


def _extract_header_name(lines: list[str]) -> tuple[str, float] | None:
    """Extract candidate full name from top header lines of a resume."""
    for line in lines[:6]:
        s = line.strip()
        if not s:
            continue
        # Split on common header delimiters (e.g. "Princy Patel | Software Engineer")
        primary = re.split(r"\s*[|•·/]\s*", s)[0].strip()
        low = primary.lower()
        if low in _NON_NAME_HEADERS:
            continue
        if any(kw in low for kw in ("@", "http", "www.", "+91", "phone:", "email:", "linkedin", "github")):
            continue
        if re.search(r"\d", primary):
            continue
        words = primary.split()
        if not (1 <= len(words) <= 5):
            continue
        # Ensure all words look like name tokens (letters, dots, hyphens, apostrophes)
        if not all(re.match(r"^[A-Za-z]+[.\-']?$", w) for w in words):
            continue
        # Reject pure role/tech descriptor lines (e.g. "Full Stack Developer")
        if low in _ROLE_OR_TECH_WORDS or all(w.lower() in _ROLE_OR_TECH_WORDS for w in words):
            continue

        conf = 0.90 if 2 <= len(words) <= 3 else 0.80
        return primary, conf

    return None


def _extract_resume(
    text: str,
    tables: list[list[list[str | None]]],
    pages: list[Any] | None = None,
) -> ExtractionResult:
    """
    Parameters
    ----------
    pages:
        Optional list of raw ``pdfplumber.Page`` objects.  When supplied,
        real font-size signals are used to detect headings instead of the
        text-only heuristic.
    """
    result: ExtractionResult = {}
    lines = [ln.strip() for ln in text.splitlines()]
    non_empty = [ln for ln in lines if ln]

    # ── candidate name: validated top header block
    header_name_info = _extract_header_name(non_empty)
    if header_name_info:
        name_val, name_conf = header_name_info
        result["name"] = FieldResult(
            value=name_val,
            confidence=name_conf,
            source="header_positional",
        )
    elif non_empty:
        result["name"] = FieldResult(
            value=non_empty[0],
            confidence=0.60,
            source="first_line_heuristic",
        )

    # ── contact fields via regex ─────────────────────────────────────────────
    email_m = _EMAIL_PATTERN.search(text)
    if email_m:
        result["email"] = FieldResult(value=email_m.group(), confidence=0.95, source="regex")

    phone_m = _PHONE_PATTERN.search(text)
    if phone_m:
        result["phone"] = FieldResult(value=phone_m.group(1).strip(), confidence=0.90, source="regex")

    url_m = _URL_PATTERN.search(text)
    if url_m:
        result["profile_url"] = FieldResult(value=url_m.group(), confidence=0.95, source="regex")

    # ── section headings ─────────────────────────────────────────────────────
    if pages:
        # Real font-size detection: collect chars with above-median font size
        try:
            heading_lines: list[str] = []
            for page in pages:
                chars = page.chars  # list of char dicts from pdfplumber
                if not chars:
                    continue
                sizes = [c.get("size", 0) for c in chars]
                if not sizes:
                    continue
                median_size = sorted(sizes)[len(sizes) // 2]
                threshold = median_size * 1.15  # 15 % above median → heading
                # Group large chars into words/lines (simplified)
                large_text = "".join(
                    c["text"] for c in chars if c.get("size", 0) >= threshold
                )
                for token in re.split(r"\n+", large_text):
                    token = token.strip()
                    if token and token not in heading_lines:
                        heading_lines.append(token)
            headings = heading_lines
            source = "font_size"
        except Exception:
            headings = _detect_headings_from_text(text)
            source = "text_heuristic_fallback"
    else:
        headings = _detect_headings_from_text(text)
        source = "text_heuristic"

    if headings:
        result["sections"] = FieldResult(
            value=headings,
            confidence=0.80,
            source=source,
        )

    # ── section content: slice text between consecutive headings ─────────────
    if headings:
        heading_set = {h.upper() for h in headings}
        sections_content: dict[str, str] = {}
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line.upper() in heading_set:
                if current_heading:
                    sections_content[current_heading] = "\n".join(current_lines).strip()
                current_heading = line
                current_lines = []
            else:
                current_lines.append(line)

        if current_heading:
            sections_content[current_heading] = "\n".join(current_lines).strip()

        for heading, content in sections_content.items():
            key = re.sub(r"\s+", "_", heading.lower().strip(": -"))
            result[f"section_{key}"] = FieldResult(
                value=content,
                confidence=0.80,
                source=source,
            )

        # ── extract organizations from education & experience sections ─────────
        _DEGREE_WORDS = {
            "bachelor", "master", "b.tech", "btech", "m.tech", "mtech", "b.e.", "b.sc",
            "m.sc", "ph.d", "doctor", "diploma", "associate", "degree", "bba", "mba",
            "bca", "mca",
        }
        org_candidates: list[str] = []
        for heading, content in sections_content.items():
            head_low = heading.lower()
            if any(k in head_low for k in ("education", "academic", "university", "college", "experience", "employment", "work")):
                for sec_line in content.splitlines():
                    sl = sec_line.strip()
                    if not sl or len(sl) > 60:
                        continue
                    if sl.startswith(("-", "•", "*", "–", "+")):
                        continue
                    low_sl = sl.lower()
                    # Skip pure degree lines
                    if any(deg in low_sl for deg in _DEGREE_WORDS) and not any(u in low_sl for u in ("university", "college", "institute", "academy")):
                        continue
                    # Match known institutional suffixes
                    if re.search(r"\b(?:university|college|institute|academy|systems|technologies|corporation|corp|inc|ltd|pvt|solutions|consulting|labs|llc|hospital)\b", sl, re.IGNORECASE):
                        clean_org = re.split(r"\s*[|•·,]\s*(?:bachelor|master|b\.tech|m\.tech|b\.e|m\.e|b\.sc|m\.sc|ph\.d|20\d\d|19\d\d)", sl, flags=re.IGNORECASE)[0].strip()
                        if clean_org and len(clean_org) >= 3 and clean_org not in org_candidates:
                            org_candidates.append(clean_org)
                    # Standalone Title-Case company line under experience
                    elif any(k in head_low for k in ("experience", "work", "employment")):
                        words = sl.split()
                        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()) and not re.search(r"\d", sl):
                            if not any(w.lower() in _ROLE_OR_TECH_WORDS for w in words):
                                cand_n = header_name_info[0] if header_name_info else ""
                                if sl not in org_candidates and sl != cand_n:
                                    org_candidates.append(sl)

        if org_candidates:
            result["organizations"] = FieldResult(
                value=org_candidates,
                confidence=0.85,
                source="section_layout",
            )

    return result


# ---------------------------------------------------------------------------
# ── Public entry-point ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def extract(
    text: str,
    tables: list[list[list[str | None]]],
    report_type: ReportType,
    *,
    pages: list[Any] | None = None,
) -> ExtractionResult:
    """Layer 1 extraction using pdfplumber-derived text and table signals.

    Parameters
    ----------
    text:
        Full extracted text of the document (from ``pdfplumber``).
    tables:
        All tables extracted from the document; each table is a
        ``list[list[str | None]]``.
    report_type:
        The detected document type (from ``detector.detect_report_type``).
    pages:
        Optional list of raw ``pdfplumber.Page`` objects.  Only used for
        resume extraction; enables real font-size-based heading detection.

    Returns
    -------
    dict[str, FieldResult]
        Field name → ``FieldResult`` mapping.  Returns an empty dict for
        ``ReportType.UNKNOWN``.
    """
    match report_type:
        case ReportType.INVOICE:
            return _extract_invoice(text, tables)
        case ReportType.BANK_STATEMENT:
            return _extract_bank_statement(text, tables)
        case ReportType.RESUME:
            return _extract_resume(text, tables, pages=pages)
        case _:
            return {}
