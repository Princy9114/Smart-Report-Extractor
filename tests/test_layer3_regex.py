import pytest

from backend.models.report_type import ReportType
from backend.services.layers import layer3_regex

def test_extract_gstin_and_email():
    # Provide a sample that should trigger the regexes.
    # Note: INVOICE applies both gstin and email patterns.
    text = (
        "Vendor details:\n"
        "Contact us at hello@example.com for queries.\n"
        "Our GSTIN is 22AAAAA0000A1Z5.\n"
        "Thank you."
    )
    
    result = layer3_regex.extract(text, ReportType.INVOICE)
    
    # Assert email
    assert "email" in result
    assert result["email"].value == "hello@example.com"
    assert result["email"].confidence == 0.9
    
    # Assert GSTIN
    assert "gstin" in result
    assert result["gstin"].value == "22AAAAA0000A1Z5"
    assert result["gstin"].confidence == 0.9

def test_extract_disabled_fields():
    # RESUME type does not attempt to extract gstin
    text = "GSTIN 22AAAAA0000A1Z5 mail: user@test.com"
    result = layer3_regex.extract(text, ReportType.RESUME)
    
    assert "email" in result
    assert "gstin" not in result
