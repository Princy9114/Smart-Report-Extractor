from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import (
    PDFParseError,
    UnsupportedDocumentError,
    ExtractionError,
    LLMUnavailableError,
    FileTooLargeError,
)
from app.models.response import ExtractionResponse, DocumentType
from app.services.pdf_parser import PDFParser
from app.services.classifier import DocumentClassifier
from app.services.extractor import ExtractorService
from app.services.llm_service import LLMService

app = FastAPI(
    title="Smart Report Extractor",
    description="Extracts structured data from PDF invoices, resumes, and bank statements.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service singletons ---
pdf_parser = PDFParser()
classifier = DocumentClassifier()
extractor_service = ExtractorService()
llm_service = LLMService()
settings = get_settings()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend UI."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text())
    return HTMLResponse("<h2>UI not found. Use POST /extract directly.</h2>")


@app.post("/extract", response_model=ExtractionResponse)
async def extract(file: UploadFile = File(...)):
    """
    Main extraction endpoint.
    Accepts a PDF file, classifies it, extracts fields, and returns a summary.
    """
    # 1. Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 2. Read and validate file size
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB.",
        )

    warnings: list[str] = []

    # 3. Parse PDF → raw text
    try:
        parsed = pdf_parser.extract(file_bytes)
    except PDFParseError as e:
        raise HTTPException(status_code=422, detail=str(e))

    text: str = parsed["text"]
    page_count: int = parsed["page_count"]

    # 4. Classify document type
    doc_type, confidence = classifier.classify(text)

    if doc_type == DocumentType.UNKNOWN:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine document type. "
                "Supported formats: Invoice, Resume, Bank Statement."
            ),
        )

    # 5. Extract structured fields
    try:
        fields = extractor_service.extract(doc_type, text)
    except (UnsupportedDocumentError, ExtractionError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 6. Generate LLM summary (graceful degradation)
    summary: str | None = None
    summary_available = True

    try:
        summary = llm_service.summarize(text, doc_type)
    except LLMUnavailableError as e:
        summary_available = False
        warnings.append(f"Summary unavailable: {str(e)}")

    return ExtractionResponse(
        document_type=doc_type,
        confidence=confidence,
        extracted_fields=fields,
        summary=summary,
        summary_available=summary_available,
        page_count=page_count,
        filename=file.filename,
        warnings=warnings,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# --- Global exception handlers ---
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
