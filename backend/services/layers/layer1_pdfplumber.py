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
# Shared regex & 2D spatial coordinate helpers
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


def _normalize_token(t: str) -> str:
    """Strip colons, hashes, hyphens, periods, and whitespace, lowercase."""
    return re.sub(r"[:#\-\.\s]+", "", t).lower()


def _find_label_in_words(
    target_clean: str,
    words: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Find all occurrences of a label in spatial words using a sliding window.

    Returns a list of (label_bbox, matched_words).
    """
    matches = []
    if not target_clean or not words:
        return matches

    for w_size in range(1, 5):
        for i in range(len(words) - w_size + 1):
            window = words[i : i + w_size]
            # All words must be on same page
            if any(w.get("page_idx", 0) != window[0].get("page_idx", 0) for w in window):
                continue
            # All words must be on same horizontal line (tops within 5pt)
            if any(abs(w.get("top", 0) - window[0].get("top", 0)) > 5.0 for w in window):
                continue

            joined = "".join(_normalize_token(w["text"]) for w in window)
            if joined == target_clean:
                bbox = {
                    "page_idx": window[0].get("page_idx", 0),
                    "x0": min(w.get("x0", 0.0) for w in window),
                    "x1": max(w.get("x1", 0.0) for w in window),
                    "top": min(w.get("top", 0.0) for w in window),
                    "bottom": max(w.get("bottom", 0.0) for w in window),
                }
                matches.append((bbox, window))

    return matches


_KNOWN_LABEL_PHRASES = {
    "account no", "account number", "acc no", "a/c no", "ac no",
    "account holder", "account name", "customer name", "cust name",
    "open balance", "opening balance", "beginning balance",
    "close balance", "closing balance", "ending balance",
    "statement period", "statement date", "ifsc code", "ifsc",
    "cif number", "cif no", "account type", "branch", "branch code",
    "invoice no", "invoice number", "inv no", "invoice date", "issue date",
    "total amount", "grand total", "sub total", "subtotal",
    "total due", "amount due", "balance due",
}


def _spatial_label_value(
    labels: list[str],
    spatial_words: list[dict[str, Any]] | None,
    text: str,
    *,
    field_name: str = "",
) -> tuple[str, float, str, str] | None:
    """Extract key-value pair using 2D spatial coordinate proximity with fallback to text regex.

    Returns:
        tuple[value, confidence, source, raw] or None
    """
    if spatial_words:
        sorted_words = sorted(
            spatial_words,
            key=lambda w: (w.get("page_idx", 0), round(w.get("top", 0.0), 1), round(w.get("x0", 0.0), 1))
        )

        for label in labels:
            clean_label = _normalize_token(label)
            if not clean_label:
                continue

            matches = _find_label_in_words(clean_label, sorted_words)
            for bbox, label_words in matches:
                p_idx = bbox["page_idx"]
                page_words = [w for w in sorted_words if w.get("page_idx", 0) == p_idx]

                # -------------------------------------------------------------
                # Strategy 1: Horizontal Right Scan (Inline Key-Value)
                # -------------------------------------------------------------
                right_candidates = []
                for w in page_words:
                    is_same_line = (
                        abs(w.get("top", 0.0) - bbox["top"]) <= 5.0
                        or (w.get("top", 0.0) >= bbox["top"] - 3.0 and w.get("bottom", 0.0) <= bbox["bottom"] + 3.0)
                    )
                    if is_same_line and w.get("x0", 0.0) >= bbox["x1"] - 2.0 and w.get("x0", 0.0) <= bbox["x1"] + 350.0:
                        if w not in label_words:
                            right_candidates.append(w)

                right_candidates.sort(key=lambda w: w.get("x0", 0.0))

                collected_h: list[str] = []
                last_x1 = bbox["x1"]

                for w in right_candidates:
                    w_text = w["text"].strip()
                    if not collected_h and w_text in {":", "-", "#", "—", "|"}:
                        last_x1 = w.get("x1", last_x1)
                        continue

                    low_w = _normalize_token(w_text)
                    if low_w in {"ifsc", "cif", "branch", "pan", "gstin"} or w_text.endswith(":"):
                        if collected_h:
                            break
                        else:
                            collected_h = []
                            break

                    # Stop if a large gap indicates column boundary (> 45pt)
                    if collected_h and (w.get("x0", 0.0) - last_x1 > 45.0):
                        break

                    collected_h.append(w_text)
                    last_x1 = w.get("x1", last_x1)

                if collected_h:
                    raw_val = " ".join(collected_h).strip()
                    val = re.sub(r"^[:#\-\s]+|[:#\-\s]+$", "", raw_val).strip()
                    norm_val = _normalize_token(val)
                    # Check if extracted string is another known label phrase (e.g. in grid/table headers)
                    is_label_phrase = any(
                        norm_val == _normalize_token(lp) or norm_val.startswith(_normalize_token(lp))
                        for lp in _KNOWN_LABEL_PHRASES
                    )
                    if not is_label_phrase and val and re.search(r"[A-Za-z0-9]", val):
                        return val, 0.90, "spatial_horizontal", raw_val

                # -------------------------------------------------------------
                # Strategy 2: Vertical Downward Scan (Stacked / Boxed Grid Cell)
                # -------------------------------------------------------------
                line_h = max(bbox["bottom"] - bbox["top"], 10.0)
                min_top = bbox["bottom"] - 2.0
                max_top = bbox["bottom"] + max(28.0, line_h * 2.2)

                down_candidates = []
                for w in page_words:
                    if w in label_words:
                        continue
                    w_top = w.get("top", 0.0)
                    w_x0 = w.get("x0", 0.0)
                    if min_top <= w_top <= max_top:
                        if (w_x0 >= bbox["x0"] - 25.0) and (w_x0 <= bbox["x1"] + 150.0):
                            down_candidates.append(w)

                down_candidates.sort(key=lambda w: (round(w.get("top", 0.0), 1), w.get("x0", 0.0)))

                if down_candidates:
                    first_line_top = down_candidates[0].get("top", 0.0)
                    first_line_words = [
                        w for w in down_candidates
                        if abs(w.get("top", 0.0) - first_line_top) <= 4.0
                    ]
                    first_line_words.sort(key=lambda w: w.get("x0", 0.0))

                    collected_v: list[str] = []
                    last_x1_v = bbox["x0"]
                    for w in first_line_words:
                        w_text = w["text"].strip()
                        if not collected_v and w_text in {":", "-", "#", "—", "|"}:
                            last_x1_v = w.get("x1", last_x1_v)
                            continue
                        if collected_v and (w.get("x0", 0.0) - last_x1_v > 45.0):
                            break
                        collected_v.append(w_text)
                        last_x1_v = w.get("x1", last_x1_v)

                    if collected_v:
                        raw_val = " ".join(collected_v).strip()
                        val = re.sub(r"^[:#\-\s]+|[:#\-\s]+$", "", raw_val).strip()
                        if val and re.search(r"[A-Za-z0-9]", val):
                            return val, 0.88, "spatial_vertical", raw_val

    # -------------------------------------------------------------------------
    # Fallback: Serialized Text Regex Scan
    # -------------------------------------------------------------------------
    for label in labels:
        raw = _label_value(label, text)
        if raw:
            value = re.split(r"\s{2,}|\t", raw)[0].strip()
            val = re.sub(r"^[:#\-\s]+|[:#\-\s]+$", "", value).strip()
            if val and re.search(r"[A-Za-z0-9]", val):
                return val, 0.85, "label_adjacent", raw

    return None


# ---------------------------------------------------------------------------
# ── INVOICE ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

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
    spatial_words: list[dict[str, Any]] | None = None,
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

    # ── scalar fields via spatial & label-adjacent search ────────────────────
    for field_name, labels in [
        ("invoice_number", _INV_NUMBER_LABELS),
        ("date",           _INV_DATE_LABELS),
        ("total",          _INV_TOTAL_LABELS),
    ]:
        spatial_res = _spatial_label_value(labels, spatial_words, text, field_name=field_name)
        if spatial_res:
            value, conf, source, raw = spatial_res
            if field_name == "total":
                if re.search(r"\b(?:nos|no|pcs|pieces|qty|quantity|items|units|kg|meters)\b", value, re.IGNORECASE):
                    amt_match = re.search(r"(?:[₹$€£¥]|INR|Rs\.?)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})|[0-9]{2,}(?:\.[0-9]{2})?)", raw)
                    if amt_match:
                        value = amt_match.group(0).strip()
                    else:
                        continue
            result[field_name] = FieldResult(
                value=value,
                confidence=conf,
                source=source,
                raw=raw,
            )

    # ── line items from tables ───────────────────────────────────────────────
    line_items: list[dict[str, str]] = []
    for table in tables:
        if not table or len(table) < 2:
            continue

        header_row: list[str | None] | None = None
        data_start = 0
        for idx, row in enumerate(table):
            cells = [str(c).lower().strip() if c else "" for c in row]
            if any(kw in cells for kw in ("description", "item", "product", "service")):
                header_row = row
                data_start = idx + 1
                break

        if header_row is None:
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
    ("account_number",   ["account number", "account no", "acc no", "account #", "a/c no", "a/c number", "ac no"]),
    ("account_name",     ["account name", "account holder", "customer name", "name", "account holder name", "cust name"]),
    ("opening_balance",  ["opening balance", "beginning balance", "balance brought forward", "open balance", "b/f balance", "op bal"]),
    ("closing_balance",  ["closing balance", "ending balance", "balance carried forward", "close balance", "c/f balance", "cl bal"]),
    ("statement_period", ["statement period", "period", "statement date", "from", "duration"]),
    ("bank_name",        ["bank name", "bank"]),
    ("ifsc",             ["ifsc code", "ifsc", "rtgs/neft ifsc"]),
]

_TXN_COL_SIGNALS = {"date", "description", "debit", "credit", "balance", "amount", "particulars"}


def _extract_bank_statement(
    text: str,
    tables: list[list[list[str | None]]],
    spatial_words: list[dict[str, Any]] | None = None,
) -> ExtractionResult:
    result: ExtractionResult = {}

    # ── bank_name from top lines heuristic
    top_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:4]
    for line in top_lines:
        low = line.lower()
        if any(b in low for b in ("bank", "hdfc", "icici", "sbi", "axis", "kotak", "pnb", "bob", "citi", "standard chartered", "hsbc")):
            result["bank_name"] = FieldResult(
                value=line,
                confidence=0.88,
                source="header_positional",
            )
            break

    # ── scalar fields ────────────────────────────────────────────────────────
    for field_name, labels in _BANK_SCALAR_LABELS:
        if field_name == "bank_name" and "bank_name" in result:
            continue
        spatial_res = _spatial_label_value(labels, spatial_words, text, field_name=field_name)
        if spatial_res:
            value, conf, source, raw = spatial_res
            result[field_name] = FieldResult(
                value=value,
                confidence=conf,
                source=source,
                raw=raw,
            )

    # ── transaction table ────────────────────────────────────────────────────
    best_table: list[list[str | None]] | None = None
    best_signal_count = 0

    for table in tables:
        if not table:
            continue
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
    *,
    pages: list[Any] | None = None,
    spatial_words: list[dict[str, Any]] | None = None,
) -> ExtractionResult:
    """
    Parameters
    ----------
    pages:
        Optional list of raw ``pdfplumber.Page`` objects.  When supplied,
        real font-size signals are used to detect headings instead of the
        text-only heuristic.
    spatial_words:
        Optional list of word bounding boxes with 2D coordinates.
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
    spatial_words: list[dict[str, Any]] | None = None,
) -> ExtractionResult:
    """Layer 1 extraction using pdfplumber-derived text, table, and 2D spatial word signals.

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
    spatial_words:
        Optional list of word bounding boxes with 2D spatial coordinates
        (``text``, ``x0``, ``x1``, ``top``, ``bottom``, ``page_idx``).

    Returns
    -------
    dict[str, FieldResult]
        Field name → ``FieldResult`` mapping.  Returns an empty dict for
        ``ReportType.UNKNOWN``.
    """
    match report_type:
        case ReportType.INVOICE:
            return _extract_invoice(text, tables, spatial_words=spatial_words)
        case ReportType.BANK_STATEMENT:
            return _extract_bank_statement(text, tables, spatial_words=spatial_words)
        case ReportType.RESUME:
            return _extract_resume(text, tables, pages=pages, spatial_words=spatial_words)
        case _:
            return {}

