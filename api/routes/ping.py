from fastapi import APIRouter, Request
from models.response_models import PingResponse, ModelsLoadedStatus

router = APIRouter()

@router.get("/ping", response_model=PingResponse)
def ping(request: Request):
    """
    Health check. Returns model loading status.
    Frontend should disable mic until all models_loaded are true.
    """
    # Fetch status from app.state, with fallback to all-False dictionary
    status = getattr(request.app.state, "models_status", {})
    
    return PingResponse(
        status="ok",
        version="1.0.0",
        models_loaded=ModelsLoadedStatus(
            whisper=status.get("whisper", False),
            nllb=status.get("nllb", False),
            tts=status.get("tts", False),
            rag=status.get("rag", False)
        )
    )