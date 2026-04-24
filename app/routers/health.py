from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check")
async def health_check():
    """Returns 200 OK when the service is up."""
    return {"status": "ok"}
