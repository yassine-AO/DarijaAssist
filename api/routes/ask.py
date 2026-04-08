from fastapi import APIRouter, UploadFile, File, HTTPException
from models.response_models import AskResponse, SourceInfo, PipelineMeta
import uuid

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask(audio: UploadFile = File(...)):
    """
    Core pipeline endpoint.
    Receives: multipart/form-data with audio file (.wav/.webm)
    Returns: Audio answer + metadata (contract-compliant structure)
    """
    # Setup phase: Validate input exists (contract requires audio field)
    if not audio or not audio.filename:
        raise HTTPException(
            status_code=422,
            detail="audio_invalid: Audio file required"
        )

    # Return dummy data matching contract structure exactly
    # In implementation phase, this will run Whisper -> NLLB -> RAG -> GPT -> TTS
    return AskResponse(
        request_id=str(uuid.uuid4()),
        answer_audio_b64="base64encoded_placeholder",  # Empty string in setup
        answer_text_darija="غادين شرحلك...",  # "I will explain to you..."
        source=SourceInfo(
            document_name="Guide CNSS 2023",
            chunk_preview="Pour bénéficier des prestations...",
            relevance_score=0.91
        ),
        pipeline_meta=PipelineMeta(
            whisper_transcript="كيفاش نسجل فالكنس؟",
            english_translation="How do I register with CNSS?",
            processing_time_ms=4200
        )
    )