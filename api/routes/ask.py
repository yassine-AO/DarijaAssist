from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from models.response_models import AskResponse, SourceInfo, PipelineMeta
from services.text_rag_pipeline import TextRAGPipeline
import uuid
import time
import os
import base64
import tempfile
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, audio: UploadFile = File(...)):
    """
    Core pipeline endpoint.
    Receives: multipart/form-data with audio file (.wav/.webm)
    Returns: Audio answer (base64) + metadata (contract-compliant structure)
    """
    # 1. Validate inputs and load status
    if not audio or not audio.filename:
        raise HTTPException(
            status_code=422,
            detail="audio_invalid: Audio file required"
        )

    models = getattr(request.app.state, "ml_models", {})
    whisper_model = models.get("whisper")
    translation_svc = models.get("nllb")
    rag_svc = models.get("rag")
    answer_svc = models.get("answer")
    tts_svc = models.get("tts")

    if not all([whisper_model, translation_svc, rag_svc, answer_svc, tts_svc]):
        raise HTTPException(
            status_code=503,
            detail="models_not_ready: Core AI models/services are still loading. Please wait."
        )

    start_time = time.time()

    # 2. Save uploaded audio to a temporary file
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.file.read())
        tmp_path = tmp.name

    try:
        # Step 1: Transcribe audio to Darija using Whisper
        logger.info("Starting Whisper transcription on temporary file...")
        transcribe_result = whisper_model.transcribe(tmp_path)
        darija_query = transcribe_result.get("text", "").strip()
        
        if not darija_query:
            raise HTTPException(
                status_code=400,
                detail="audio_empty: No speech could be transcribed from the audio input."
            )
        logger.info("Whisper transcript: '%s'", darija_query)

        # Step 2: Run Text RAG Pipeline Orchestrator (Translation -> FAISS -> LLaMA -> Translation)
        pipeline = TextRAGPipeline(translation_svc, rag_svc, answer_svc)
        pipeline_result = pipeline.process(darija_query)

        query_english = pipeline_result["query_english"]
        answer_english = pipeline_result["answer_english"]
        answer_darija = pipeline_result["answer_darija"]
        retrieved_chunks = pipeline_result["retrieved_chunks"]
        top_source = retrieved_chunks[0] if retrieved_chunks else None

        # Step 3: Synthesize Darija response text back to Speech using TTS Service
        logger.info("Synthesizing final Darija response back to audio...")
        audio_bytes, _ = tts_svc.synthesize(answer_darija)

        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Compile metadata and elapsed time
        processing_time_ms = int((time.time() - start_time) * 1000)

        if top_source:
            source_info = SourceInfo(
                document_name=top_source["document_name"],
                chunk_preview=top_source["text"][:150] + "...",
                relevance_score=top_source["relevance_score"]
            )
        else:
            source_info = SourceInfo(
                document_name="General Knowledge",
                chunk_preview="No direct administrative document matched the search.",
                relevance_score=0.5
            )

        return AskResponse(
            request_id=str(uuid.uuid4()),
            answer_audio_b64=audio_b64,
            answer_text_darija=answer_darija,
            source=source_info,
            pipeline_meta=PipelineMeta(
                whisper_transcript=darija_query,
                english_translation=query_english,
                processing_time_ms=processing_time_ms
            )
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline processing crash:")
        raise HTTPException(
            status_code=500,
            detail=f"pipeline_failed: Internal processing error: {str(exc)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)