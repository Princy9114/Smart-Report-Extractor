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
import re
from typing import Any

import spacy
from spacy.language import Language

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model load — done once, shared across all requests.
# ---------------------------------------------------------------------------
_NLP: Language | None = None

for _model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"):
    try:
        _NLP = spacy.load(_model_name)
        logger.info("spaCy model '%s' loaded successfully.", _model_name)
        break
    except OSError:
        continue

if _NLP is None:
    logger.warning("No spaCy English model found. Run: python -m spacy download en_core_web_sm")

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

def _clean_vendor(name: str) -> str:
    """Clean OCR or NER artifacts from extracted vendor name."""
    s = re.sub(r"\s+", " ", name).strip()
    s = re.sub(r"(?i)^(?:m/s\.?|to:?|from:?)\s*", "", s)
    # Strip trailing conjunctions/prepositions like '&Supplyof', 'of', '&'
    s = re.sub(r"(?i)(?:\s*(?:&|\band\b|\bof\b|\bfor\b|\bto\b))+$", "", s).strip()
    return s


def _extract_invoice(entities: list[tuple[str, str]]) -> ExtractionResult:
    result: ExtractionResult = {}
    amounts: list[str] = []

    for text, label in entities:
        match label:
            case "DATE" if "date" not in result:
                result["date"] = _field(text)
            case "ORG" if "vendor" not in result:
                cleaned = _clean_vendor(text)
                if cleaned and len(cleaned) > 2:
                    result["vendor"] = _field(cleaned)
            case "MONEY":
                # Filter out pure small integers or non-monetary strings
                cleaned_amt = text.strip()
                if not re.match(r"^\d{1,2}$", cleaned_amt):
                    amounts.append(cleaned_amt)

    if amounts:
        result["amounts"] = _field(amounts)
        # Select the last valid currency / decimal money entity
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


# ---------------------------------------------------------------------------
# Entity validation & domain blacklists
# ---------------------------------------------------------------------------

_TECH_AND_TOOL_TERMS: set[str] = {
    # Models & AI/ML
    "yolo", "yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11",
    "bert", "roberta", "gpt", "llm", "rag", "transformer", "resnet", "vgg",
    "cnn", "rnn", "lstm", "gan", "diffusion", "opencv", "spacy", "nltk",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "huggingface",
    "nlp", "computer vision", "machine learning", "deep learning", "artificial intelligence",
    "ai", "ml", "cv",
    # Languages & Frameworks
    "python", "javascript", "typescript", "java", "c++", "c#", "c", "rust",
    "golang", "go", "php", "ruby", "swift", "kotlin", "html", "css", "sql",
    "react", "angular", "vue", "nextjs", "nodejs", "node", "express",
    "fastapi", "flask", "django", "spring", "springboot", "flutter", "dart",
    # Databases & DevOps
    "mongodb", "postgresql", "postgres", "mysql", "redis", "sqlite", "oracle",
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "git", "github",
    "gitlab", "linux", "ubuntu", "nginx", "apache", "jenkins",
    # Data & Analytics
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "tableau", "powerbi",
    # Generic section / role non-names
    "developer", "engineer", "scientist", "intern", "manager", "lead",
    "projects", "experience", "education", "skills", "certifications",
    "achievements", "summary", "objective", "bachelor", "master", "btech",
    "curriculum vitae", "resume", "portfolio", "profile",
}

_PLATFORM_AND_META_ORGS: set[str] = {
    "linkedin", "github", "gitlab", "leetcode", "kaggle", "hackerrank",
    "codechef", "codeforces", "medium", "twitter", "x", "geeksforgeeks",
    "stackoverflow", "portfolio", "curriculum vitae", "resume", "cv",
    "b.tech", "btech", "m.tech", "mtech", "b.e.", "b.sc", "m.sc", "ph.d",
    "computer science", "information technology", "data science",
    "artificial intelligence", "machine learning", "nlp", "computer vision",
}


def _is_valid_person_name(name: str) -> bool:
    """Validate whether an extracted PERSON entity is a plausible human name."""
    s = name.strip()
    if not s or len(s) < 2 or len(s) > 40:
        return False
    low = s.lower()
    if low in _TECH_AND_TOOL_TERMS:
        return False
    # Reject strings containing digits (e.g. YOLOv8, Python3)
    if re.search(r"\d", s):
        return False
    # Reject strings with URLs or symbols
    if any(c in s for c in ("@", "/", "\\", "http", ".com", ".in", "_", "+", "=", "*")):
        return False
    words = s.split()
    if not (1 <= len(words) <= 4):
        return False
    # If single word, must not be purely lowercase
    if len(words) == 1 and s.islower():
        return False
    # Reject if any token is a known tech term
    if any(w.lower() in _TECH_AND_TOOL_TERMS for w in words):
        return False
    # Must consist of alphabetic letters, hyphens, periods, or apostrophes
    if not all(re.match(r"^[A-Za-z]+[.\-']?$", w) for w in words):
        return False
    return True


def _clean_org(org: str, candidate_name: str | None = None, project_titles: set[str] | None = None) -> str | None:
    """Filter non-organization noise (platforms, candidate name, tech terms, project titles)."""
    s = re.sub(r"\s+", " ", org).strip()
    if not s or len(s) < 2 or len(s) > 60:
        return None
    low = s.lower()
    if low in _PLATFORM_AND_META_ORGS or low in _TECH_AND_TOOL_TERMS:
        return None
    if project_titles and low in project_titles:
        return None
    # Filter candidate name if matched (e.g. 'PRINCY PATEL' vs 'Princy Patel')
    if candidate_name:
        cand_low = candidate_name.lower().strip()
        cand_tokens = set(cand_low.split())
        if low == cand_low or low in cand_tokens:
            return None
    # Reject URLs, emails, or phone patterns
    if "@" in s or "http" in low or ".com" in low or re.match(r"^\+?\d", s):
        return None
    return s


def _is_valid_resume_date(date_str: str) -> bool:
    """Validate whether an extracted DATE entity is a valid calendar date/range."""
    s = date_str.strip()
    if not s:
        return False
    # If purely digits, must be a 4-digit calendar year (1950–2035)
    if re.fullmatch(r"\d+", s):
        if len(s) == 4 and 1950 <= int(s) <= 2035:
            return True
        return False  # Rejects "93282", "1", "2", "400604"
    # Reject phone-like digit sequences
    if len(re.sub(r"\D", "", s)) >= 7 and not re.search(r"[a-zA-Z]", s):
        return False
    # Valid if it contains a month abbreviation/name
    if re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present)\b", s, re.IGNORECASE):
        return True
    # Valid if formatted date or year range (e.g. 2021-2025, 05/2023)
    if re.search(r"\b\d{4}\s*[-–—/]\s*(?:\d{4}|present)\b", s, re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b", s):
        return True
    return False


def _extract_resume(entities: list[tuple[str, str]], doc_text: str = "") -> ExtractionResult:
    result: ExtractionResult = {}
    orgs: list[str] = []
    dates: list[str] = []
    candidate_name: str | None = None
    project_titles: set[str] = set()

    if doc_text:
        # 1. Derive candidate name from top lines
        first_lines = [ln.strip() for ln in doc_text.splitlines() if ln.strip()][:3]
        for line in first_lines:
            cand = re.split(r"\s*[|•·/]\s*", line)[0].strip()
            if _is_valid_person_name(cand):
                candidate_name = cand
                break

        # 2. Collect project titles to exclude from organizations
        in_projects = False
        for line in doc_text.splitlines():
            s = line.strip()
            if not s:
                continue
            if re.match(r"^(?:projects?|personal projects?|academic projects?)\b", s, re.IGNORECASE):
                in_projects = True
                continue
            if in_projects and re.match(r"^[A-Z][A-Z\s]{2,}$", s) and len(s.split()) <= 4:
                in_projects = False
            elif in_projects and not s.startswith(("-", "•", "*", "–")) and not re.search(r"\b(?:20\d\d|19\d\d|present)\b", s, re.IGNORECASE):
                project_titles.add(s.lower())

    for text, label in entities:
        match label:
            case "PERSON" if "name" not in result:
                if _is_valid_person_name(text):
                    result["name"] = _field(text)
                    if not candidate_name:
                        candidate_name = text
            case "ORG":
                cleaned_org = _clean_org(text, candidate_name, project_titles=project_titles)
                if cleaned_org and cleaned_org not in orgs:
                    orgs.append(cleaned_org)
            case "DATE":
                if _is_valid_resume_date(text):
                    cleaned_date = text.strip()
                    if cleaned_date not in dates:
                        dates.append(cleaned_date)

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
            return _extract_resume(entities, doc_text=text)
        case _:
            return {}
