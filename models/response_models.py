from pydantic import BaseModel
from typing import List, Optional

class ModelsLoadedStatus(BaseModel):
    whisper: bool
    nllb: bool
    tts: bool
    rag: bool

class PingResponse(BaseModel):
    status: str
    version: str
    models_loaded: ModelsLoadedStatus

class ServiceItem(BaseModel):
    id: str
    label_darija: str
    label_latin: str

class ServicesResponse(BaseModel):
    services: List[ServiceItem]

class SourceInfo(BaseModel):
    document_name: str
    chunk_preview: str
    relevance_score: float

class PipelineMeta(BaseModel):
    whisper_transcript: str
    english_translation: str
    processing_time_ms: int

class AskResponse(BaseModel):
    request_id: str
    answer_audio_b64: str
    answer_text_darija: str
    source: SourceInfo
    pipeline_meta: PipelineMeta

class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None