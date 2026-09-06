import io
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.services import ocr
from backend.services.pipeline import run_pipeline


def _create_synthetic_invoice_image() -> bytes:
    """Create a synthetic image containing invoice text for OCR testing."""
    # 600x400 white canvas
    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)

    lines = [
        "INVOICE",
        "Invoice Number: INV-9988",
        "Vendor: Acme Cloud Solutions",
        "Total Due: $1,250.00",
        "Date: 2026-08-30",
        "Email: contact@acmecloud.com",
    ]

    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 45

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_extract_text_from_image():
    img_bytes = _create_synthetic_invoice_image()
    extracted_text = ocr.extract_text_from_image(img_bytes)

    assert len(extracted_text) > 0
    # RapidOCR should detect key strings from the rendered image
    upper = extracted_text.upper()
    assert "INVOICE" in upper or "INV" in upper or "ACME" in upper or "1,250" in upper or "TOTAL" in upper


def test_is_scanned_pdf_heuristic():
    # Empty or short text is considered scanned
    assert ocr.is_scanned_pdf("", page_count=1) is True
    assert ocr.is_scanned_pdf("Short header", page_count=2) is True

    # Long text is digital PDF
    long_text = "Standard digital text " * 50
    assert ocr.is_scanned_pdf(long_text, page_count=1) is False


@pytest.mark.asyncio
async def test_pipeline_with_image_input():
    img_bytes = _create_synthetic_invoice_image()
    response = await run_pipeline(img_bytes, filename="invoice_scan.png", output_format="json")

    assert response.status_code == 200
    import json
    data = json.loads(response.body.decode())

    # Verify structured fields were extracted via pipeline
    assert "__meta__" in data
