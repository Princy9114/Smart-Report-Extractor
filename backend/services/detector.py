"""
detector.py
~~~~~~~~~~~
Keyword-scoring heuristic to identify the type of a text document.
"""

from __future__ import annotations

import re
from backend.models.report_type import ReportType

# ---------------------------------------------------------------------------
# Keyword tables
# Each keyword is paired with a weight (0 < w <= 1.0).
# Rarer / more discriminating terms get a higher weight.
# ---------------------------------------------------------------------------
_KEYWORDS: dict[ReportType, dict[str, float]] = {
    ReportType.INVOICE: {
        # core billing terms
        "invoice":          1.0,
        "invoice number":   1.0,
        "invoice date":     0.9,
        "bill to":          0.9,
        "subtotal":         0.8,
        "total due":        0.9,
        "amount due":       0.9,
        "payment due":      0.8,
        "tax":              0.4,
        "vat":              0.7,
        "gst":              0.7,
        "purchase order":   0.7,
        "po number":        0.8,
        "vendor":           0.6,
        "remit to":         0.8,
        "due date":         0.5,
    },
    ReportType.BANK_STATEMENT: {
        # core banking terms
        "bank statement":   1.0,
        "account number":   0.9,
        "account balance":  0.9,
        "opening balance":  1.0,
        "closing balance":  1.0,
        "statement period": 1.0,
        "transaction date": 0.9,
        "debit":            0.7,
        "credit":           0.5,   # also appears in invoices — lower weight
        "withdrawal":       0.9,
        "deposit":          0.7,
        "available balance":0.9,
        "sort code":        0.8,
        "iban":             0.8,
        "swift":            0.7,
        "bic":              0.7,
    },
    ReportType.RESUME: {
        # core CV / résumé terms
        "curriculum vitae": 1.0,
        "resume":           1.0,
        "work experience":  1.0,
        "professional summary": 0.9,
        "objective":        0.5,
        "education":        0.7,
        "skills":           0.6,
        "certifications":   0.8,
        "references":       0.6,
        "employment history":0.9,
        "projects":         0.5,
        "languages":        0.5,
        "volunteer":        0.6,
        "linkedin":         0.7,
        "github":           0.6,
        "gpa":              0.7,
    },
}

# Precompile a regex for each keyword (word-boundary aware, case-insensitive)
_PATTERNS: dict[ReportType, list[tuple[re.Pattern[str], float]]] = {
    report_type: [
        (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
        for kw, weight in keywords.items()
    ]
    for report_type, keywords in _KEYWORDS.items()
}

# Maximum achievable raw score per type (used to normalise to [0, 1])
_MAX_SCORES: dict[ReportType, float] = {
    rt: sum(w for _, w in patterns)
    for rt, patterns in _PATTERNS.items()
}

# Confidence threshold below which we call the document UNKNOWN
_THRESHOLD: float = 0.1


def detect_report_type(text: str) -> tuple[ReportType, float]:
    """Infer the document type from plain text using keyword scoring.

    The algorithm:
    1. For each candidate ``ReportType``, find every keyword that appears in
       *text* and accumulate its weight.
    2. Normalise the raw score against the maximum possible score for that type
       to get a confidence in **[0, 1]**.
    3. Pick the type with the highest normalised confidence.
    4. If that confidence is below ``_THRESHOLD`` (0.4), return
       ``(ReportType.UNKNOWN, 0.0)``.

    Parameters
    ----------
    text:
        The full extracted text of the document.

    Returns
    -------
    tuple[ReportType, float]
        ``(report_type, confidence)`` where *confidence* is in **[0.0, 1.0]**.
        Returns ``(ReportType.UNKNOWN, 0.0)`` when no type reaches the
        confidence threshold.
    """
    if not text or not text.strip():
        return ReportType.UNKNOWN, 0.0

    scores: dict[ReportType, float] = {}

    for report_type, patterns in _PATTERNS.items():
        raw = sum(
            weight
            for pattern, weight in patterns
            if pattern.search(text)
        )
        max_score = _MAX_SCORES[report_type]
        scores[report_type] = raw / max_score if max_score > 0 else 0.0

    best_type = max(scores, key=lambda rt: scores[rt])
    best_conf = round(scores[best_type], 4)

    if best_conf < _THRESHOLD:
        return ReportType.UNKNOWN, 0.0

    return best_type, best_conf
