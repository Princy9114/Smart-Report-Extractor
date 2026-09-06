"""
exporter.py
~~~~~~~~~~~
Utility to format and stream an ExtractionResult as either JSON or CSV.
"""

from __future__ import annotations

import csv
import io
import json

from fastapi.responses import JSONResponse, Response

from backend.models.field_result import FieldResult

ExtractionResult = dict[str, FieldResult]


def _format_csv(result: ExtractionResult) -> str:
    """Format extraction result into CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["field", "value"])

    for field_name, field_result in result.items():
        if field_name == "__meta__":
            continue
        value = field_result.value
        if isinstance(value, list) and value:
            for item in value:
                str_val = json.dumps(item) if isinstance(item, dict) else str(item)
                writer.writerow([field_name, str_val])
        else:
            writer.writerow([field_name, str(value)])

    meta = result.get("__meta__")
    if meta and isinstance(meta.value, dict):
        overall = meta.value.get("overall_confidence", 0.0)
        count = meta.value.get("field_count", 0)
        writer.writerow([])
        writer.writerow([f"# SUMMARY: {count} fields extracted. Overall confidence: {overall:.2f}"])

    return output.getvalue()


def export(result: ExtractionResult, output_format: str = "json") -> Response:
    """Format an ExtractionResult dict into a downloadable FastAPI response."""
    if output_format.lower() == "csv":
        return Response(
            content=_format_csv(result),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=extraction_result.csv"},
        )

    return JSONResponse(
        content={k: v.value for k, v in result.items()},
        headers={"Content-Disposition": "attachment; filename=extraction_result.json"},
    )
