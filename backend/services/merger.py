"""
merger.py
~~~~~~~~~
Merges field results from multiple extraction layers into a single
``ExtractionResult``.

Merge logic (per field)
-----------------------
1. Collect all ``FieldResult`` values for the field from every layer.
2. Pick the candidate with the **highest confidence** as the base.
3. Apply a **consensus bonus of +0.10** when ≥ 2 layers agree on the same
   normalised string value (capped at 1.0).
4. Record which layers contributed and the source of the winning value.

Overall confidence
------------------
``overall_confidence`` = arithmetic mean of every merged field's final
confidence.  Stored in the returned ``ExtractionResult`` under the special
key ``"__meta__"``.
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import Any

from backend.models.field_result import FieldResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ExtractionResult = dict[str, FieldResult]

# Consensus bonus added when ≥ 2 layers agree on the same value.
_CONSENSUS_BONUS = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(value: Any) -> str:
    """Stable string key used for consensus comparison."""
    if isinstance(value, list):
        return str(sorted(str(v) for v in value))
    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def merge(layer_results: dict[str, ExtractionResult]) -> ExtractionResult:
    """Merge ``FieldResult`` dicts from multiple extraction layers.

    Parameters
    ----------
    layer_results:
        Mapping of ``layer_name → {field_name: FieldResult}``.
        Example keys: ``"layer1"``, ``"layer2_spacy"``, ``"layer3_regex"``,
        ``"layer4_llm"``.

    Returns
    -------
    dict[str, FieldResult]
        One merged ``FieldResult`` per field, plus a special
        ``"__meta__"`` entry whose ``value`` is a summary dict::

            {
                "overall_confidence": float,   # mean of all field confidences
                "field_count": int,
                "layers_used": list[str],
            }
    """
    if not layer_results:
        return {}

    # ── Collect all candidates per field ────────────────────────────────────
    # field_name → list of (layer_name, FieldResult)
    candidates: dict[str, list[tuple[str, FieldResult]]] = {}

    for layer_name, fields in layer_results.items():
        if not fields:
            continue
        for field_name, field_result in fields.items():
            if field_name.startswith("__"):
                continue  # skip any meta keys passed in
            candidates.setdefault(field_name, []).append((layer_name, field_result))

    # ── Merge each field ─────────────────────────────────────────────────────
    merged: ExtractionResult = {}

    for field_name, entries in candidates.items():
        # Sort by confidence descending; highest becomes the winner
        entries_sorted = sorted(entries, key=lambda t: t[1].confidence, reverse=True)
        winning_layer, winner = entries_sorted[0]

        # Consensus check: do ≥ 2 layers share the same normalised value?
        norm_winner = _normalise(winner.value)
        agreeing_layers = [
            ln for ln, fr in entries if _normalise(fr.value) == norm_winner
        ]
        consensus = len(agreeing_layers) >= 2
        bonus = _CONSENSUS_BONUS if consensus else 0.0
        final_confidence = min(1.0, winner.confidence + bonus)

        contributing_layers = [ln for ln, _ in entries]

        merged[field_name] = FieldResult(
            value=winner.value,
            confidence=round(final_confidence, 4),
            source=(
                f"{winning_layer}"
                + (f"+consensus({','.join(agreeing_layers)})" if consensus else "")
            ),
            raw=winner.raw,
        )

        if consensus:
            logger.debug(
                "merger: '%s' consensus across %s → confidence %.3f",
                field_name,
                agreeing_layers,
                final_confidence,
            )

    # ── Overall confidence ────────────────────────────────────────────────────
    field_confidences = [fr.confidence for fr in merged.values()]
    overall = round(mean(field_confidences), 4) if field_confidences else 0.0

    merged["__meta__"] = FieldResult(
        value={
            "overall_confidence": overall,
            "field_count": len(merged),   # excludes __meta__ itself
            "layers_used": list(layer_results.keys()),
        },
        confidence=overall,
        source="merger",
    )

    logger.info(
        "merger: %d field(s) merged from %d layer(s); overall_confidence=%.4f",
        len(merged) - 1,  # minus __meta__
        len(layer_results),
        overall,
    )

    return merged
