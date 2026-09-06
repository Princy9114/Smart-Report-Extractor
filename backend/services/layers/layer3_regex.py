"""
layer3_regex.py
~~~~~~~~~~~~~~~
Extraction Layer 3: high-confidence regex patterns.

All patterns are compiled once at module level in the ``PATTERNS`` dict.
``extract(text, report_type)`` applies the subset of patterns relevant to
the given document type and returns a ``Dict[str, FieldResult]`` with
``confidence=0.9`` for every match.

Pattern coverage
----------------
Field            Format / notes
-----------      -------------------------------------------------------
invoice_number   INV-, bill-, or purely numeric IDs (6-12 digits)
gstin            Indian GST Identification Number (15-char alphanum)
pan              Indian Permanent Account Number (10-char: AAAAA9999A)
email            RFC-5321-compatible local@domain.tld
phone            Indian mobile (+91 / 0) and generic international
total_amount     Currency symbols / keywords before a decimal number
date             DD/MM/YYYY, MM-DD-YYYY, YYYY-MM-DD, "15 Jan 2024"
account_number   Label-anchored 9-18 digit bank account numbers
ifsc             Indian Financial System Code (4-alpha 0 5-alphanum)
"""

from __future__ import annotations

import re
from typing import Any

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

# ---------------------------------------------------------------------------
# Compiled pattern registry
# ---------------------------------------------------------------------------

PATTERNS: dict[str, re.Pattern[str]] = {

    # Invoice / document reference numbers
    # Strongly prefers labelled numbers or prefixed identifiers (INV-2024-001, Invoice No: 987654)
    # Excludes bare 6-digit zipcodes or 10-digit phone numbers
    "invoice_number": re.compile(
        r"(?:invoice\s*(?:number|no\.?|#|id)|inv\s*no\.?|bill\s*no\.?|bill\s*#)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,30})"
        r"|\b(?:INV|INVOICE|BILL|REC)[-_/][A-Za-z0-9\-_/]{3,25}\b",
        re.IGNORECASE,
    ),

    # Indian GSTIN — 15 characters: 2-digit state code + PAN + 1 digit + Z + checksum
    "gstin": re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b",
        re.IGNORECASE,
    ),

    # Indian PAN — 5 alpha + 4 digit + 1 alpha (always uppercase in practice)
    "pan": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b",
    ),

    # Email address
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    ),

    # Phone: Indian mobile (+91 / 0), landlines with STD code (022-...), or clean international
    "phone": re.compile(
        r"(?:(?<=\s)|(?<=^)|(?<=[|•·,;:]))(?:\+91[\s\-]?)?[6-9]\d{9}\b"
        r"|\b0\d{2,4}[\s\-]?\d{6,8}\b"
        r"|(?:(?<=\s)|(?<=^)|(?<=[|•·,;:]))(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b",
        re.IGNORECASE,
    ),

    # Monetary total — strictly captures numeric/currency totals, ignoring unit-of-measure (NOS/PCS)
    "total_amount": re.compile(
        r"(?:total(?:\s+(?:amount|due|payable|value))?|grand\s+total|net\s+(?:amount|payable)|balance\s+due|amount\s+payable)"
        r"\s*[:\-]?\s*"
        r"(?:(?:INR|Rs\.?|USD|EUR|GBP|₹|\$|€|£)\s*)?"
        r"([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})|[0-9]{2,}(?:\.[0-9]{2})?)",
        re.IGNORECASE,
    ),

    # Date in common formats
    # DD/MM/YYYY · YYYY-MM-DD · "15 Jan 2024" · "May 2023 - Present" · "2021 - 2025"
    "date": re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}|present)\b"
        r"|\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present)\b"
        r"|\b\d{1,2}[/\-]\d{1,2}[/\-](?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}[/\-]\d{1,2}[/\-]\d{1,2}\b"
        r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
    "dates": re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}|present)\b"
        r"|\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present)\b"
        r"|\b\d{1,2}[/\-]\d{1,2}[/\-](?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}[/\-]\d{1,2}[/\-]\d{1,2}\b"
        r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),

    # Bank account number — label-anchored, 9-18 digits
    "account_number": re.compile(
        r"(?:account\s*(?:number|no\.?|#)|a/?c\s*(?:no\.?|#)?)"
        r"\s*[:\-]?\s*(\d[\d\s]{7,17}\d)",
        re.IGNORECASE,
    ),

    # Indian IFSC code — 4 alpha letters + '0' + 6 alphanumeric chars
    "ifsc": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    ),
}

# ---------------------------------------------------------------------------
# Field sets per document type
# (controls which patterns are applied to avoid meaningless hits)
# ---------------------------------------------------------------------------

_TYPE_FIELDS: dict[ReportType, set[str]] = {
    ReportType.INVOICE: {
        "invoice_number", "gstin", "pan", "email", "phone",
        "total_amount", "date",
    },
    ReportType.BANK_STATEMENT: {
        "account_number", "ifsc", "date", "total_amount",
        "email", "phone", "pan",
    },
    ReportType.RESUME: {
        "email", "phone", "pan", "dates",
    },
}

_CONFIDENCE = 0.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return all non-overlapping matches / group(1) values for *pattern*."""
    results: list[str] = []
    for m in pattern.finditer(text):
        # Prefer capturing group 1 when present (e.g. total_amount, account_number)
        try:
            val = m.group(1)
        except IndexError:
            val = m.group(0)
        val = val.strip()
        if val and val not in results:
            results.append(val)
    return results


def _field(value: Any) -> FieldResult:
    return FieldResult(value=value, confidence=_CONFIDENCE, source="regex")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def extract(text: str, report_type: ReportType) -> dict[str, FieldResult]:
    """Apply document-type-specific regex patterns to *text*.

    Parameters
    ----------
    text:
        Plain text of the document.
    report_type:
        Detected document type; governs which patterns are applied.

    Returns
    -------
    dict[str, FieldResult]
        One entry per field with at least one match.  Multi-match fields store
        a ``list[str]``; single-match fields store a plain ``str``.
        All entries have ``confidence=0.9`` and ``source="regex"``.
        Returns ``{}`` for ``ReportType.UNKNOWN``.
    """
    active_fields = _TYPE_FIELDS.get(report_type)
    if not active_fields:
        return {}

    result: dict[str, FieldResult] = {}

    for field_name in active_fields:
        pattern = PATTERNS[field_name]
        matches = _apply(pattern, text)
        if not matches:
            continue
        # Single value for unambiguous fields; list for multi-occurrence fields
        value: Any = matches[0] if len(matches) == 1 else matches
        result[field_name] = _field(value)

    return result
