"""
test_layer4_llm.py
~~~~~~~~~~~~~~~~~~
Unit and integration tests for Layer 4 Google Gemini LLM extraction service.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType
from backend.services.layers import layer4_llm


def test_build_prompts():
    text = "TAX INVOICE\nInvoice Number: INV-12345\nTotal: $500.00"
    schema = {"invoice_number": "string", "total": "string"}
    system, user_content = layer4_llm._build_prompts(text, ReportType.INVOICE, schema)

    assert "invoice" in system.lower()
    assert "invoice_number" in user_content
    assert "INV-12345" in user_content


def test_parse_response_clean_json():
    raw = '{"invoice_number": "INV-12345", "total": "$500.00"}'
    data = layer4_llm._parse_response(raw)
    assert data["invoice_number"] == "INV-12345"
    assert data["total"] == "$500.00"


def test_parse_response_markdown_fences():
    raw = '```json\n{"invoice_number": "INV-9999", "vendor": "Acme Corp"}\n```'
    data = layer4_llm._parse_response(raw)
    assert data["invoice_number"] == "INV-9999"
    assert data["vendor"] == "Acme Corp"

    raw_generic = '```\n{"name": "Alice Smith"}\n```'
    data_generic = layer4_llm._parse_response(raw_generic)
    assert data_generic["name"] == "Alice Smith"


def test_to_field_results():
    data = {
        "invoice_number": "INV-001",
        "total": "100.00",
        "empty_str": "",
        "none_val": None,
        "empty_list": [],
    }
    results = layer4_llm._to_field_results(data)
    assert len(results) == 2
    assert "invoice_number" in results
    assert results["invoice_number"].value == "INV-001"
    assert results["invoice_number"].confidence == 0.85
    assert results["invoice_number"].source == "llm_gemini"
    assert "empty_str" not in results
    assert "none_val" not in results
    assert "empty_list" not in results


@pytest.mark.asyncio
async def test_extract_unknown_report_type():
    res = await layer4_llm.extract("Sample text", ReportType.UNKNOWN)
    assert res == {}


@pytest.mark.asyncio
async def test_extract_no_client(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = await layer4_llm.extract("Sample invoice text", ReportType.INVOICE)
    assert res == {}


@pytest.mark.asyncio
async def test_extract_with_mock_gemini_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "invoice_number": "INV-2026-X",
        "date": "2026-09-09",
        "total": "4,500.00",
        "vendor": "Apex Cloud Inc",
    })

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    text = "TAX INVOICE\nApex Cloud Inc\nInvoice No: INV-2026-X\nDate: 2026-09-09\nTotal: 4,500.00"
    res = await layer4_llm.extract(text, ReportType.INVOICE, client=mock_client)

    assert len(res) == 4
    assert res["invoice_number"].value == "INV-2026-X"
    assert res["invoice_number"].confidence == 0.85
    assert res["invoice_number"].source == "llm_gemini"
    assert res["vendor"].value == "Apex Cloud Inc"
    assert res["total"].value == "4,500.00"
