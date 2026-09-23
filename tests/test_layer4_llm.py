"""
test_layer4_llm.py
~~~~~~~~~~~~~~~~~~
Unit and integration tests for Layer 4 Multi-Provider LLM extraction service
(Local Ollama, Local OpenAI-compatible, Google Gemini).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType
from backend.services.layers import layer4_llm
from backend.services import summarizer


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
    results = layer4_llm._to_field_results(data, source="llm_local_ollama")
    assert len(results) == 2
    assert "invoice_number" in results
    assert results["invoice_number"].value == "INV-001"
    assert results["invoice_number"].confidence == 0.85
    assert results["invoice_number"].source == "llm_local_ollama"
    assert "empty_str" not in results
    assert "none_val" not in results
    assert "empty_list" not in results


def test_provider_detection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert layer4_llm.get_active_provider() == "ollama"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert layer4_llm.get_active_provider() == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert layer4_llm.get_active_provider() == "none"
    assert not layer4_llm.is_available()

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    # In clean environment without config, provider is none
    assert layer4_llm.get_active_provider() == "none"

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert layer4_llm.get_active_provider() == "gemini"

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    assert layer4_llm.get_active_provider() == "ollama"


@pytest.mark.asyncio
async def test_extract_unknown_report_type():
    res = await layer4_llm.extract("Sample text", ReportType.UNKNOWN)
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
    res = await layer4_llm.extract(text, ReportType.INVOICE, provider="gemini", client=mock_client)

    assert len(res) == 4
    assert res["invoice_number"].value == "INV-2026-X"
    assert res["invoice_number"].confidence == 0.85
    assert res["invoice_number"].source == "llm_gemini"
    assert res["vendor"].value == "Apex Cloud Inc"
    assert res["total"].value == "4,500.00"


@pytest.mark.asyncio
async def test_extract_with_mock_ollama():
    mock_response = {
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "bank_name": "Chase Bank",
                "account_number": "9876543210",
                "closing_balance": "12,450.00",
            }),
        }
    }

    async def mock_post(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = mock_response
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        text = "Chase Bank Account Statement\nAccount Number: 9876543210\nClosing Balance: 12,450.00"
        res = await layer4_llm.extract(text, ReportType.BANK_STATEMENT, provider="ollama")

        assert len(res) == 3
        assert res["bank_name"].value == "Chase Bank"
        assert res["account_number"].value == "9876543210"
        assert res["closing_balance"].value == "12,450.00"
        assert res["bank_name"].source == "llm_local_ollama"
        assert res["bank_name"].confidence == 0.85


@pytest.mark.asyncio
async def test_extract_with_mock_local_endpoint():
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "name": "Jane Doe",
                        "skills": ["Python", "FastAPI", "Docker"],
                    }),
                }
            }
        ]
    }

    async def mock_post(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = mock_response
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        text = "Resume\nJane Doe\nSkills: Python, FastAPI, Docker"
        res = await layer4_llm.extract(text, ReportType.RESUME, provider="local")

        assert len(res) == 2
        assert res["name"].value == "Jane Doe"
        assert res["skills"].value == ["Python", "FastAPI", "Docker"]
        assert res["name"].source == "llm_local"


@pytest.mark.asyncio
async def test_extract_graceful_fallback_on_network_error():
    async def mock_post_fail(*args, **kwargs):
        raise httpx.ConnectError("Ollama daemon not running")

    with patch("httpx.AsyncClient.post", side_effect=mock_post_fail):
        res = await layer4_llm.extract("Sample text", ReportType.INVOICE, provider="ollama")
        assert res == {}
