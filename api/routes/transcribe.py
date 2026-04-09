from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os

router = APIRouter()


class TranscribeResponse(BaseModel):
    transcription: str


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(request: Request, audio: UploadFile = File(...)):
    """
    Transcribe audio file using Whisper.
    Input: audio file (wav, mp3, m4a, etc.)
    Output: raw transcription text
    """
    # Get whisper model from app state
    whisper_model = getattr(request.app.state, "ml_models", {}).get("whisper")

    if not whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not loaded. Please wait and try again."
        )

    # Save uploaded file to temp location
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.file.read())
        tmp_path = tmp.name

    try:
        # Transcribe using Whisper
        result = whisper_model.transcribe(tmp_path)
        transcription = result.get("text", "").strip()

        return TranscribeResponse(transcription=transcription)

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)
