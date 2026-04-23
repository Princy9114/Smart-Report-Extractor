import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas.base import DocumentType, ExtractionMethod
from app.schemas import InvoiceExtraction, ResumeExtraction, BankStatementExtraction
from app.utils.pdf_parser import parse_pdf
from app.utils.classifier import classify_document
from app.extractors.llm_extractor import extract_with_llm, LLM_PROVIDER, ANTHROPIC_API_KEY, GEMINI_API_KEY
from app.extractors.rule_extractor import extract_with_rules

MAX_FILE_SIZE_MB = 20

app = FastAPI(
    title="Smart Report Extractor",
    description="Extract structured data and plain-English summaries from PDF documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI
ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Smart Report Extractor API — see /docs"}


@app.get("/health")
async def health():
    llm_available = False
    llm_provider_name = "none"
    
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        llm_available = True
        llm_provider_name = "Google Gemini"
    elif LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        llm_available = True
        llm_provider_name = "Anthropic Claude"
    
    return {
        "status": "ok",
        "llm_available": llm_available,
        "llm_provider": llm_provider_name,
        "supported_types": ["invoice", "resume", "bank_statement"],
    }


@app.post(
    "/extract",
    summary="Extract structured data from a PDF",
    response_model=None,
)
async def extract(file: UploadFile = File(...)):
    # ── Validate input ──────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    # ── Parse PDF ───────────────────────────────────────────────────────────
    try:
        parsed = parse_pdf(content)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse PDF — it may be encrypted or corrupted. ({e})",
        )

    if not parsed.full_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image — try an OCR tool first."
            ),
        )

    # ── Classify document type ──────────────────────────────────────────────
    doc_type, confidence = classify_document(parsed.full_text)

    if doc_type == DocumentType.UNKNOWN:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine document type (invoice, resume, or bank statement). "
                f"Classifier confidence was too low ({confidence:.0%}). "
                "Ensure the document is one of the supported types."
            ),
        )

    # ── Extract: try LLM first, fall back to rules ──────────────────────────
    result = await extract_with_llm(parsed, doc_type, confidence)

    if result is None:
        # LLM unavailable, key missing, or returned an error
        result = extract_with_rules(parsed, doc_type, confidence)
        if not result.warnings:
            result.warnings = []
        result.warnings.append(
            "LLM extraction was unavailable — rule-based fallback was used."
        )

    return result


@app.get("/docs-ui", include_in_schema=False)
async def docs_redirect():
    return FileResponse("ui/index.html")
