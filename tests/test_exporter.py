import csv
import io
import json
import pytest

from backend.models.field_result import FieldResult
from backend.services.exporter import _stream_json, _stream_csv

@pytest.mark.asyncio
async def test_exporter_json():
    # Prepare dummy data
    result = {
        "invoice_number": FieldResult(value="INV-123", confidence=0.9),
        "line_items": FieldResult(value=[{"desc": "apple", "price": "10"}], confidence=0.8),
        "__meta__": FieldResult(value={"overall_confidence": 0.85, "field_count": 2}, confidence=0.85)
    }

    # Generate output
    generator = _stream_json(result)
    parts = [chunk async for chunk in generator]
    final_output = "".join(parts)
    
    # Validate it parses as JSON
    parsed = json.loads(final_output)
    assert "invoice_number" in parsed
    assert parsed["invoice_number"] == "INV-123"
    assert parsed["line_items"][0]["desc"] == "apple"

@pytest.mark.asyncio
async def test_exporter_csv():
    # Prepare dummy data
    result = {
        "invoice_number": FieldResult(value="INV-001", confidence=0.9),
        "total": FieldResult(value="100.00", confidence=0.95),
        "__meta__": FieldResult(value={"overall_confidence": 0.925, "field_count": 2}, confidence=1.0)
    }
    
    # Generate output
    generator = _stream_csv(result)
    parts = [chunk async for chunk in generator]
    final_output = "".join(parts)
    
    # Parse as CSV
    reader = csv.reader(io.StringIO(final_output))
    rows = list(reader)
    
    # Ensure header is present
    assert rows[0] == ["field", "value"]
    
    # Ensure values were exported
    fields = [r[0] for r in rows if r]
    assert "invoice_number" in fields
    assert "total" in fields
    
    # Ensure footer summary is there
    # The last non-empty row should be the summary
    non_empty_rows = [r for r in rows if r]
    assert non_empty_rows[-1][0].startswith("# SUMMARY")
