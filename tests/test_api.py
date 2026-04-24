from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_extract_endpoint_rejects_non_pdf():
    # Send a plain text file upload
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"Hello, this is just text", "text/plain")},
        data={"format": "json"},
    )
    
    # Assert we get a 400 because only PDFs are supported
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

def test_health_check_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
