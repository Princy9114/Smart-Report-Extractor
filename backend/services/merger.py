"""
merger.py
~~~~~~~~~
Merges field results from multiple extraction layers into a single
ExtractionResult with strict single-value resolution, field canonicalization,
and domain-aware tie-breaking.
"""

from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Any

from backend.models.field_result import FieldResult

logger = logging.getLogger(__name__)

ExtractionResult = dict[str, FieldResult]

_CONSENSUS_BONUS = 0.10

# Canonical field name mappings to unify different layer names
_CANONICAL_FIELD_MAP: dict[str, str] = {
    "total": "total_amount",
    "grand_total": "total_amount",
    "amount_due": "total_amount",
    "invoice_no": "invoice_number",
    "inv_number": "invoice_number",
    "invoice_num": "invoice_number",
    "account_no": "account_number",
    "acc_number": "account_number",
    "bank": "bank_name",
}

# Fields that are explicitly allowed to be multi-valued lists
_LIST_FIELDS: set[str] = {
    "line_items",
    "transactions",
    "skills",
    "organizations",
    "dates",
    "amounts",
}

_UNIT_OF_MEASURE_PATTERN = re.compile(
    r"\b(?:nos|no|pcs|pieces|qty|quantity|items|units|kg|kgs|g|gms|mtr|meters|hrs|hours|box|pkts|pkt)\b",
    re.IGNORECASE,
)


_TECH_BLACKLIST = {
    "yolo", "yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11",
    "bert", "roberta", "gpt", "llm", "rag", "transformer", "resnet", "vgg",
    "cnn", "rnn", "lstm", "gan", "diffusion", "opencv", "spacy", "nltk",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "huggingface",
    "nlp", "computer vision", "machine learning", "deep learning", "artificial intelligence",
    "ai", "ml", "cv", "ocr", "etl", "sql etl",
    "python", "javascript", "typescript", "java", "c++", "c#", "c", "rust",
    "golang", "go", "php", "ruby", "swift", "kotlin", "html", "css", "sql",
    "react", "angular", "vue", "nextjs", "nodejs", "node", "express",
    "fastapi", "flask", "django", "spring", "springboot", "flutter", "dart",
    "mongodb", "postgresql", "postgres", "mysql", "redis", "sqlite", "oracle",
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "git", "github",
    "gitlab", "linux", "ubuntu", "nginx", "apache", "jenkins",
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "tableau", "powerbi",
    "developer", "engineer", "scientist", "intern", "manager", "lead",
    "projects", "experience", "education", "skills", "technical skills", "certifications",
    "tools & libraries", "tools and libraries", "tools", "libraries", "frameworks",
    "achievements", "summary", "objective", "bachelor", "master", "btech",
    "curriculum vitae", "resume", "portfolio", "profile",
}

_PLATFORM_ORGS = {
    "linkedin", "github", "gitlab", "leetcode", "kaggle", "hackerrank",
    "codechef", "codeforces", "medium", "twitter", "x", "geeksforgeeks",
    "stackoverflow", "portfolio", "curriculum vitae", "resume", "cv",
    "b.tech", "btech", "m.tech", "mtech", "b.e.", "b.sc", "m.sc", "ph.d",
    "computer science", "information technology", "data science",
    "computer engineering", "electrical engineering", "mechanical engineering",
    "artificial intelligence", "machine learning", "nlp", "computer vision",
}


def _normalise(value: Any) -> str:
    """Stable string key used for consensus comparison."""
    if isinstance(value, list):
        return str(sorted(str(v).strip().lower() for v in value))
    return str(value).strip().lower()


def _score_candidate(field_name: str, val_str: str) -> float:
    """Domain-aware validation bonus/penalty for a scalar candidate value."""
    s = val_str.strip()
    if not s:
        return -1.0
    low = s.lower()

    match field_name:
        case "name":
            # Severe penalty for digits (e.g. YOLOv8, Python3)
            if re.search(r"\d", s):
                return -0.95
            # Severe penalty for known tech terms or role names
            if low in _TECH_BLACKLIST or any(w.lower() in _TECH_BLACKLIST for w in s.split()):
                return -0.90
            # Severe penalty for URLs, emails, phone prefixes
            if any(c in s for c in ("@", "/", "\\", "http", ".com", ".in", "_", "+", "=", "*")):
                return -0.90
            # Penalty for single lowercase word or short strings
            words = s.split()
            if len(words) == 1 and (s.islower() or len(s) < 3):
                return -0.50
            if len(words) > 4:
                return -0.40
            # Strong bonus for clean 2-3 word full names in Title Case or UPPERCASE
            if 2 <= len(words) <= 3 and all(re.match(r"^[A-Za-z]+[.\-']?$", w) for w in words):
                if s.isupper() or all(w[0].isupper() for w in words):
                    return 0.35
            return 0.10

        case "invoice_number":
            # Penalize phone numbers (e.g. 10 digits starting with 6-9, or 11 digits starting with 0)
            if re.match(r"^(?:\+91|0)?[6-9]\d{9}$", s) or re.match(r"^0\d{9,11}$", s):
                return -0.60
            # Penalize pure 6-digit zip/pin codes
            if re.match(r"^\d{6}$", s):
                return -0.40
            # Penalize generic words
            if s.upper() in {"INVOICE", "TAX INVOICE", "BILL", "ORIGINAL", "DUPLICATE", "REF"}:
                return -0.80
            # Strong bonus for alphanumeric or labelled prefixes (INV-, BILL-, 2024/, etc.)
            if re.search(r"\b(?:INV|BILL|REC|INVOICE)[-_/]", s, re.IGNORECASE) or ("/" in s and any(c.isdigit() for c in s)):
                return 0.30
            # Bonus for containing both letters and numbers
            if any(c.isalpha() for c in s) and any(c.isdigit() for c in s):
                return 0.20
            return 0.0

        case "total_amount":
            # Reject unit-of-measure quantities like '2 NOS', '5 PCS'
            if _UNIT_OF_MEASURE_PATTERN.search(s) and not re.search(r"\d+\.\d{2}", s):
                return -0.90
            # Penalize pure small integers (< 10) without decimals
            if re.match(r"^\d{1,2}$", s):
                return -0.60
            # Bonus for standard 2-decimal monetary amount
            if re.search(r"\d+\.\d{2}\b", s):
                return 0.25
            # Bonus for currency symbol prefix
            if re.match(r"^[₹$€£¥]|(?:INR|USD|EUR|Rs\.?)\s*\d", s, re.IGNORECASE):
                return 0.20
            return 0.0

        case "phone":
            # Penalize embedded newlines or garbage length
            if "\n" in s or len(s) < 7 or len(s) > 16:
                return -0.50
            clean_digits = re.sub(r"\D", "", s)
            if len(clean_digits) in {10, 11, 12}:
                return 0.30
            return 0.0

        case "vendor":
            # Penalize garbled fragments like 'Manufacturing &Supplyof'
            if re.search(r"(?i)\b(?:of|and|the|for|to|&)\s*$", s):
                return -0.40
            if len(s.split()) >= 2 and all(w[0].isupper() for w in s.split() if w.isalpha()):
                return 0.20
            return 0.0

        case "gstin":
            if re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}$", s, re.IGNORECASE):
                return 0.30
            return -0.50

        case "pan":
            if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", s, re.IGNORECASE):
                return 0.30
            return -0.50

        case "email":
            if re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", s):
                return 0.30
            return -0.50

        case _:
            return 0.0


def _clean_list_field(field_name: str, items: Any, resolved_name: str | None = None) -> list[Any]:
    """Sanitize and deduplicate multi-valued list items (e.g. organizations, dates)."""
    raw_list = items if isinstance(items, list) else [items]
    cleaned: list[Any] = []
    seen: set[str] = set()

    for item in raw_list:
        if isinstance(item, dict):
            cleaned.append(item)
            continue

        s = str(item).strip()
        if not s:
            continue
        low = s.lower()

        if field_name == "organizations":
            # Strip trailing degrees or department descriptors
            s = re.split(r"\s*[|•·,]\s*(?:bachelor|master|m\.s\.|b\.s\.|b\.tech|m\.tech|ph\.d)", s, flags=re.IGNORECASE)[0].strip()
            s = re.sub(r"\s+(?:m\.s\.|b\.s\.|ph\.d|btech|mtech)\b", "", s, flags=re.IGNORECASE).strip()
            low = s.lower()
            if low in _PLATFORM_ORGS or low in _TECH_BLACKLIST:
                continue
            if resolved_name:
                res_low = resolved_name.lower().strip()
                res_tokens = set(res_low.split())
                if low == res_low or low in res_tokens:
                    continue
            if "@" in s or "http" in low or ".com" in low or re.match(r"^\+?\d+$", s):
                continue
            if len(s) < 3:
                continue

        elif field_name == "dates":
            # Exclude isolated non-year numbers like "93282", "1", "2"
            if re.fullmatch(r"\d+", s):
                if not (len(s) == 4 and 1950 <= int(s) <= 2035):
                    continue
            # Exclude phone fragments (7+ digits without letters that aren't a 4-digit year range like 2021-2025)
            clean_digits = re.sub(r"\D", "", s)
            if len(clean_digits) >= 7 and not re.search(r"[a-zA-Z]", s):
                if not re.search(r"\b(?:19|20)\d{2}\s*[-–—/]\s*(?:19|20)\d{2}\b", s):
                    continue

        if low not in seen:
            seen.add(low)
            cleaned.append(s)

    # Filter out standalone year fragments when a full date range containing that year exists
    if field_name == "dates" and len(cleaned) > 1:
        final_dates = []
        for d in cleaned:
            if any(len(other) > len(d) and d.lower() in other.lower() for other in cleaned):
                continue
            final_dates.append(d)
        return final_dates

    return cleaned


def _resolve_scalar_field(
    field_name: str,
    raw_entries: list[tuple[str, FieldResult]],
) -> FieldResult:
    """Resolve multiple layer outputs for a scalar field into exactly ONE single value."""
    # 1. Unpack list values into individual candidate entries
    unpacked_candidates: list[tuple[str, str, float, Any]] = []
    for layer_name, fr in raw_entries:
        if isinstance(fr.value, list):
            for item in fr.value:
                str_item = str(item).strip()
                if str_item:
                    unpacked_candidates.append((layer_name, str_item, fr.confidence, fr.raw))
        else:
            str_val = str(fr.value).strip()
            if str_val:
                unpacked_candidates.append((layer_name, str_val, fr.confidence, fr.raw))

    if not unpacked_candidates:
        return FieldResult(value="", confidence=0.0, source="merger")

    # 2. Score each candidate with domain validation
    scored_candidates: list[dict[str, Any]] = []
    for layer_name, cand_val, base_conf, raw in unpacked_candidates:
        val_score = _score_candidate(field_name, cand_val)
        scored_candidates.append({
            "layer": layer_name,
            "value": cand_val,
            "base_conf": base_conf,
            "val_score": val_score,
            "adjusted_score": base_conf + val_score,
            "raw": raw,
        })

    # 3. Check consensus across layers for identical normalised values
    norm_groups: dict[str, list[dict[str, Any]]] = {}
    for cand in scored_candidates:
        norm_k = _normalise(cand["value"])
        norm_groups.setdefault(norm_k, []).append(cand)

    for norm_k, group in norm_groups.items():
        distinct_layers = {c["layer"] for c in group}
        if len(distinct_layers) >= 2:
            for c in group:
                c["adjusted_score"] += _CONSENSUS_BONUS
                c["consensus_layers"] = sorted(distinct_layers)

    # 4. Sort descending by adjusted score
    scored_candidates.sort(key=lambda c: c["adjusted_score"], reverse=True)
    winner = scored_candidates[0]

    # Calculate clean final confidence
    final_conf = min(1.0, max(0.10, winner["base_conf"] + (0.10 if "consensus_layers" in winner else 0.0)))

    source = winner["layer"]
    if "consensus_layers" in winner:
        source += f"+consensus({','.join(winner['consensus_layers'])})"

    return FieldResult(
        value=winner["value"],
        confidence=round(final_conf, 4),
        source=source,
        raw=winner["raw"],
    )


def merge(layer_results: dict[str, ExtractionResult]) -> ExtractionResult:
    """Merge ``FieldResult`` dicts from multiple extraction layers with strict single-value resolution.

    Parameters
    ----------
    layer_results:
        Mapping of ``layer_name → {field_name: FieldResult}``.

    Returns
    -------
    dict[str, FieldResult]
        One resolved ``FieldResult`` per field, plus a special ``"__meta__"`` entry.
    """
    if not layer_results:
        return {}

    # ── Canonicalise and group candidates per field ─────────────────────────
    canonical_candidates: dict[str, list[tuple[str, FieldResult]]] = {}

    for layer_name, fields in layer_results.items():
        if not fields:
            continue
        for raw_field_name, field_result in fields.items():
            if raw_field_name.startswith("__"):
                continue  # skip meta keys
            # Map aliases to canonical field name (e.g. 'total' -> 'total_amount')
            canon_name = _CANONICAL_FIELD_MAP.get(raw_field_name, raw_field_name)
            canonical_candidates.setdefault(canon_name, []).append((layer_name, field_result))

    # ── Merge and resolve scalar fields first ────────────────────────────────
    merged: ExtractionResult = {}

    for field_name, entries in canonical_candidates.items():
        if field_name not in _LIST_FIELDS:
            merged[field_name] = _resolve_scalar_field(field_name, entries)

    resolved_name = str(merged["name"].value) if "name" in merged and merged["name"].value else None

    # ── Merge and sanitize multi-valued list fields ──────────────────────────
    for field_name, entries in canonical_candidates.items():
        if field_name in _LIST_FIELDS:
            combined_items: list[Any] = []
            max_conf = 0.0
            sources: list[str] = []
            for layer_name, fr in entries:
                if fr.confidence > max_conf:
                    max_conf = fr.confidence
                sources.append(layer_name)
                if isinstance(fr.value, list):
                    combined_items.extend(fr.value)
                elif fr.value:
                    combined_items.append(fr.value)

            sanitized_list = _clean_list_field(field_name, combined_items, resolved_name=resolved_name)
            if sanitized_list:
                merged[field_name] = FieldResult(
                    value=sanitized_list,
                    confidence=round(max_conf, 4),
                    source="+".join(sorted(set(sources))),
                )

    # ── Overall confidence summary ──────────────────────────────────────────
    field_confidences = [fr.confidence for fr in merged.values()]
    overall = round(mean(field_confidences), 4) if field_confidences else 0.0

    merged["__meta__"] = FieldResult(
        value={
            "overall_confidence": overall,
            "field_count": len(merged),
            "layers_used": list(layer_results.keys()),
        },
        confidence=overall,
        source="merger",
    )

    logger.info(
        "merger: %d field(s) merged from %d layer(s); overall_confidence=%.4f",
        len(merged) - 1,
        len(layer_results),
        overall,
    )

    return merged

