from fastapi import APIRouter
from models.response_models import PingResponse, ModelsLoadedStatus

router = APIRouter()

@router.get("/ping", response_model=PingResponse)
def ping():
    """
    Health check. Returns model loading status.
    Frontend should disable mic until all models_loaded are true.
    """
    return PingResponse(
        status="ok",
        version="1.0.0",
        models_loaded=ModelsLoadedStatus(
            whisper=False,
            nllb=False,
            tts=False,
            rag=False
        )
    )