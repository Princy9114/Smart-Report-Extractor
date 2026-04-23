import re
from app.schemas.base import DocumentType


# Keyword sets with weights — heavier weight = more discriminative signal
INVOICE_KEYWORDS = {
    "invoice": 10, "bill to": 8, "ship to": 6, "invoice #": 10,
    "invoice number": 10, "invoice date": 8, "due date": 7,
    "purchase order": 7, "po number": 7, "subtotal": 6,
    "amount due": 8, "payment terms": 6, "tax": 3, "total": 3,
    "vendor": 5, "billing address": 6, "remit to": 7,
    "line item": 5, "qty": 4, "unit price": 6, "description": 2,
}

RESUME_KEYWORDS = {
    "resume": 10, "curriculum vitae": 10, "cv": 6,
    "work experience": 8, "professional experience": 8,
    "employment history": 8, "education": 6, "skills": 5,
    "objective": 5, "summary": 3, "references": 5,
    "certifications": 6, "projects": 4, "achievements": 4,
    "linkedin": 7, "github": 6, "portfolio": 5,
    "bachelor": 6, "master": 6, "university": 5, "college": 4,
    "internship": 6, "job title": 5, "position": 3,
}

BANK_STATEMENT_KEYWORDS = {
    "statement": 7, "bank statement": 10, "account statement": 10,
    "account number": 8, "account balance": 8,
    "opening balance": 10, "closing balance": 10,
    "transaction": 6, "debit": 7, "credit": 5,
    "deposit": 6, "withdrawal": 7, "transfer": 5,
    "available balance": 8, "statement period": 9,
    "sort code": 7, "routing number": 7, "swift": 6,
    "iban": 7, "bank": 4, "branch": 4,
}

ALL_KEYWORDS = {
    DocumentType.INVOICE: INVOICE_KEYWORDS,
    DocumentType.RESUME: RESUME_KEYWORDS,
    DocumentType.BANK_STATEMENT: BANK_STATEMENT_KEYWORDS,
}


def classify_document(text: str) -> tuple[DocumentType, float]:
    """
    Score each document type by counting weighted keyword hits in the text.
    Returns (document_type, confidence_score 0.0–1.0).

    This is intentionally simple and fast — it runs before any LLM call.
    The LLM is not needed for classification; even a template-less PDF has
    enough vocabulary to discriminate between these three types reliably.
    """
    lower_text = text.lower()
    scores: dict[DocumentType, float] = {}

    for doc_type, keywords in ALL_KEYWORDS.items():
        score = 0.0
        for keyword, weight in keywords.items():
            if keyword in lower_text:
                # Count occurrences but cap contribution to avoid single-word dominance
                count = len(re.findall(re.escape(keyword), lower_text))
                score += weight * min(count, 3)
        scores[doc_type] = score

    if not any(scores.values()):
        return DocumentType.UNKNOWN, 0.0

    total = sum(scores.values())
    best_type = max(scores, key=lambda t: scores[t])
    confidence = scores[best_type] / total if total > 0 else 0.0

    # Require at least 40% confidence AND a minimum raw score to claim a type.
    # The raw score guard prevents a single weak keyword hit on gibberish text
    # from producing a confident-looking classification.
    if confidence < 0.40 or scores[best_type] < 8:
        return DocumentType.UNKNOWN, confidence

    return best_type, round(confidence, 3)
