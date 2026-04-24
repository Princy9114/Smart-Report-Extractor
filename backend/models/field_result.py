"""
field_result.py
~~~~~~~~~~~~~~~
Container for a single extracted field value with provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldResult:
    """Holds the extracted value and metadata for a single document field.

    Attributes
    ----------
    value:
        The extracted value (str, list, dict, etc.).
    confidence:
        Extraction confidence in **[0.0, 1.0]**.
    source:
        Human-readable label for the extraction strategy used
        (e.g. ``"label_adjacent"``, ``"table"``, ``"regex"``).
    raw:
        Optional raw string / data captured before post-processing.
    """

    value: Any
    confidence: float = 1.0
    source: str = "unknown"
    raw: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
