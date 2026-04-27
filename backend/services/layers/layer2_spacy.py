"""
layer2_spacy.py
~~~~~~~~~~~~~~~
Extraction Layer 2: spaCy Named Entity Recognition.

Loads ``en_core_web_sm`` **once** at module import and exposes a single
``extract(text, report_type)`` function that maps spaCy entity labels to
document-type-specific field names.

Entity-to-field mapping
-----------------------
INVOICE
  DATE   → date          (first occurrence)
  ORG    → vendor        (first occurrence)
  MONEY  → amounts       (all occurrences; highest becomes total_amount)

BANK_STATEMENT
  DATE   → statement_date  (first occurrence)
  ORG    → bank_name       (first occurrence)
  MONEY  → amounts         (all occurrences)

RESUME
  PERSON → name            (first occurrence)
  ORG    → organizations   (all occurrences — employers / universities)
  DATE   → dates           (all occurrences — employment / graduation dates)
"""

from __future__ import annotations

import logging
from typing import Any

import spacy
from spacy.language import Language

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model load — done once, shared across all requests.
# ---------------------------------------------------------------------------
_MODEL_NAME = "en_core_web_md"
_NLP: Language | None = None

try:
    _NLP = spacy.load(_MODEL_NAME)
    logger.info("spaCy model '%s' loaded successfully.", _MODEL_NAME)
except OSError:
    logger.error(
        "spaCy model '%s' not found. Run: python -m spacy download %s",
        _MODEL_NAME,
        _MODEL_NAME,
    )

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ExtractionResult = dict[str, FieldResult]

_CONFIDENCE = 0.75          # fixed confidence for all NER-derived fields
_ENTITY_LABELS = {"DATE", "ORG", "PERSON", "MONEY"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ner(text: str) -> list[tuple[str, str]]:
    """Return ``(entity_text, label)`` pairs for the four target labels."""
    if _NLP is None:
        return []
    doc = _NLP(text)
    return [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in _ENTITY_LABELS and ent.text.strip()
    ]


def _field(value: Any, *, many: bool = False) -> FieldResult:
    source = "spacy_ner"
    return FieldResult(value=value, confidence=_CONFIDENCE, source=source)


# ---------------------------------------------------------------------------
# Per-type extractor functions
# ---------------------------------------------------------------------------

def _extract_invoice(entities: list[tuple[str, str]]) -> ExtractionResult:
    result: ExtractionResult = {}
    amounts: list[str] = []

    for text, label in entities:
        match label:
            case "DATE" if "date" not in result:
                result["date"] = _field(text)
            case "ORG" if "vendor" not in result:
                result["vendor"] = _field(text)
            case "MONEY":
                amounts.append(text)

    if amounts:
        result["amounts"] = _field(amounts)
        # Heuristic: the last MONEY entity is usually the grand total
        result["total_amount"] = FieldResult(
            value=amounts[-1],
            confidence=_CONFIDENCE,
            source="spacy_ner_last_money",
        )

    return result


def _extract_bank_statement(entities: list[tuple[str, str]]) -> ExtractionResult:
    result: ExtractionResult = {}
    amounts: list[str] = []

    for text, label in entities:
        match label:
            case "DATE" if "statement_date" not in result:
                result["statement_date"] = _field(text)
            case "ORG" if "bank_name" not in result:
                result["bank_name"] = _field(text)
            case "MONEY":
                amounts.append(text)

    if amounts:
        result["amounts"] = _field(amounts)

    return result


def _extract_resume(entities: list[tuple[str, str]]) -> ExtractionResult:
    result: ExtractionResult = {}
    orgs: list[str] = []
    dates: list[str] = []

    for text, label in entities:
        match label:
            case "PERSON" if "name" not in result:
                result["name"] = _field(text)
            case "ORG":
                if text not in orgs:
                    orgs.append(text)
            case "DATE":
                if text not in dates:
                    dates.append(text)

    if orgs:
        result["organizations"] = _field(orgs)
    if dates:
        result["dates"] = _field(dates)

    return result


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def extract(text: str, report_type: ReportType) -> ExtractionResult:
    """Run spaCy NER on *text* and map entities to field names.

    Parameters
    ----------
    text:
        Plain text of the document.
    report_type:
        Detected document type; governs how entity labels are mapped to fields.

    Returns
    -------
    dict[str, FieldResult]
        All detected fields, each with ``confidence=0.75`` and
        ``source="spacy_ner"``.  Returns an empty dict when the model is not
        loaded or the type is ``UNKNOWN``.
    """
    if _NLP is None:
        logger.warning("extract() called but spaCy model is not loaded — returning {}.")
        return {}

    if report_type is ReportType.UNKNOWN:
        return {}

    entities = _run_ner(text)

    match report_type:
        case ReportType.INVOICE:
            return _extract_invoice(entities)
        case ReportType.BANK_STATEMENT:
            return _extract_bank_statement(entities)
        case ReportType.RESUME:
            return _extract_resume(entities)
        case _:
            return {}
