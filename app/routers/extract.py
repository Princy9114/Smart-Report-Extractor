import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.pipeline import run_pipeline

router = APIRouter(tags=["Extract"])

_ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}


@router.post("/extract", summary="Extract structured data from a document or image")
async def extract_document(
    file: UploadFile = File(...),
    format: str = Form("json"),
):
    """Accepts a PDF document or image file and returns extracted structured data.
    `format` can be 'json' or 'csv'.
    """
    ext = os.path.splitext(file.filename.lower() if file.filename else "")[1]
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP.",
        )

    # Read bytes from upload
    file_bytes = await file.read()

    # Pass to orchestrator pipeline
    return await run_pipeline(file_bytes, filename=file.filename or "", output_format=format)
