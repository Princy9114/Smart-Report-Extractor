from fastapi import APIRouter, Form, HTTPException
from fastapi import UploadFile, File

from backend.services.pipeline import run_pipeline

router = APIRouter(tags=["Extract"])


@router.post("/extract", summary="Extract structured data from a document")
async def extract_document(
    file: UploadFile = File(...),
    format: str = Form("json"),
):
    """
    Accepts a document file and returns extracted structured data.
    `format` can be 'json' or 'csv'.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read bytes from upload
    pdf_bytes = await file.read()

    # Pass to orchestrator pipeline
    return await run_pipeline(pdf_bytes, output_format=format)
