"""
exporter.py
~~~~~~~~~~~
Utility to format and stream an ExtractionResult as either JSON or CSV.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncGenerator

from fastapi.responses import StreamingResponse

from backend.models.field_result import FieldResult

ExtractionResult = dict[str, FieldResult]

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _ResultEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle `FieldResult` dataclasses."""
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "value") and hasattr(obj, "confidence"):
            # It's a FieldResult (since it's a dataclass, isinstance checks can be tricky)
            # Just extract the pure value and strip out confidence/source metadata
            return obj.value
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


async def _stream_json(result: ExtractionResult) -> AsyncGenerator[str, None]:
    """Yield pretty-printed JSON chunks."""
    # We dump it all to a string then yield it, as the dict is memory-bound
    # and not exceedingly large.
    out = json.dumps(result, cls=_ResultEncoder, indent=2)
    yield out


async def _stream_csv(result: ExtractionResult) -> AsyncGenerator[str, None]:
    """
    Yield CSV lines. Flattens list fields by repeating the parent field name.
    Includes a trailing comment row with the __meta__ summary.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["field", "value"])
    yield output.getvalue()
    output.seek(0)
    output.truncate(0)

    meta = result.get("__meta__")

    for field_name, field_result in result.items():
        if field_name == "__meta__":
            continue
            
        value = field_result.value
        
        # Flatten lists: one row per item
        if isinstance(value, list) and value:
            for item in value:
                # If item is a dict (like a line_item), dump as JSON string
                if isinstance(item, dict):
                    str_val = json.dumps(item)
                else:
                    str_val = str(item)
                
                writer.writerow([field_name, str_val])
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)
        else:
            # Scalar value
            writer.writerow([field_name, str(value)])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    # Write summary footer 
    if meta and isinstance(meta.value, dict):
        overall = meta.value.get("overall_confidence", 0.0)
        count = meta.value.get("field_count", 0)
        writer.writerow([]) # empty row separator
        writer.writerow([f"# SUMMARY: {count} fields extracted. Overall confidence: {overall:.2f}"])
        yield output.getvalue()

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def export(result: ExtractionResult, output_format: str = "json") -> StreamingResponse:
    """Format an ExtractionResult dict into a downloadable FastAPI response.

    Parameters
    ----------
    result:
        The merged output from ``merger.merge()``.
    output_format:
        ``"json"`` or ``"csv"``. Defaults to ``"json"``.

    Returns
    -------
    StreamingResponse
        A FastAPI response with correct media type and disposition headers.
    """
    fmt = output_format.lower()
    
    if fmt == "csv":
        generator = _stream_csv(result)
        media_type = "text/csv"
        ext = "csv"
    else:
        generator = _stream_json(result)
        media_type = "application/json"
        ext = "json"

    headers = {
        "Content-Disposition": f"attachment; filename=extraction_result.{ext}"
    }

    return StreamingResponse(
        content=generator, 
        media_type=media_type, 
        headers=headers
    )
