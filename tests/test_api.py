import io
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from main import app

client = TestClient(app)


def test_extract_endpoint_rejects_unsupported_file():
    # Send a plain text file upload
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"Hello, this is just text", "text/plain")},
        data={"format": "json"},
    )

    # Assert we get a 400 because only supported formats are allowed
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_extract_endpoint_accepts_image():
    # Generate a realistic synthetic invoice image
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)
    text = (
        "INVOICE\n"
        "Invoice Number: INV-001\n"
        "Bill To: Acme Customer\n"
        "Total Due: $100.00\n"
        "Amount Due: $100.00"
    )
    draw.text((20, 20), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    response = client.post(
        "/extract",
        files={"file": ("receipt.png", buf.getvalue(), "image/png")},
        data={"format": "json"},
    )

    assert response.status_code == 200
    assert "__meta__" in response.json()


def test_health_check_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
