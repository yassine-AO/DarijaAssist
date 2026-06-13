from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from api.routes import ping, ask, services, transcribe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store the models
ml_models = {}

# Keep track of loading status
models_loaded_status = {
    "whisper": False,
    "nllb": False,
    "tts": False,
    "rag": False
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup actions ---
    logger.info("Starting up DarijaAssist API...")
    
    # 1. Load Whisper Model
    logger.info("Loading Whisper model...")
    try:
        import whisper
        from pathlib import Path
        
        # Check for custom fine-tuned Whisper model checkpoint
        custom_whisper_path = Path(__file__).resolve().parent / "models" / "whisper" / "darija_whisper.pt"
        
        if custom_whisper_path.exists():
            logger.info("✅ Custom fine-tuned Whisper model found at '%s'. Loading...", custom_whisper_path)
            ml_models["whisper"] = whisper.load_model(str(custom_whisper_path))
        else:
            logger.info("No custom model found at '%s'. Loading standard 'base' Whisper model...", custom_whisper_path)
            ml_models["whisper"] = whisper.load_model("base")
            
        models_loaded_status["whisper"] = True
        logger.info("Whisper model loaded successfully.")
    except ImportError:
        logger.warning("Whisper library not installed. Please run: pip install openai-whisper")
    
    # 2. Load NLLB Translation Model (AtlasIA Terjman-Ultra)
    logger.info("Loading NLLB translation model...")
    try:
        from services.translation_service import TranslationService
        ml_models["nllb"] = TranslationService()
        models_loaded_status["nllb"] = True
        logger.info("NLLB translation model loaded successfully.")
    except Exception as e:
        logger.warning("NLLB model failed to load: %s", e)

    # 3. Load RAG Service (FAISS)
    logger.info("Loading RAG service...")
    try:
        from services.rag_service import RAGService
        ml_models["rag"] = RAGService()
        models_loaded_status["rag"] = True
        logger.info("RAG service loaded successfully.")
    except Exception as e:
        logger.warning("RAG service failed to load: %s", e)

    # 4. Load Answer Service (Groq)
    logger.info("Loading Answer service...")
    try:
        from services.answer_service import AnswerService
        ml_models["answer"] = AnswerService()
        logger.info("Answer service loaded successfully.")
    except Exception as e:
        logger.warning("Answer service failed to load: %s", e)

    # 5. Load TTS Service (ElevenLabs + Piper Fallback)
    logger.info("Loading TTS service...")
    try:
        from services.tts_service import TTSService
        ml_models["tts"] = TTSService()
        models_loaded_status["tts"] = True
        logger.info("TTS service loaded successfully.")
    except Exception as e:
        logger.warning("TTS service failed to load: %s", e)
    
    # Attach our status and models to app.state so routes can access them
    app.state.models_status = models_loaded_status
    app.state.ml_models = ml_models

    yield
    
    # --- Shutdown actions ---
    logger.info("Shutting down DarijaAssist API...")
    ml_models.clear()
    logger.info("Models cleared from memory.")

app = FastAPI(
    title="DarijaAssist API",
    version="1.0.0",
    description="Voice AI agent for Moroccan government services - API Contract v1.0",
    lifespan=lifespan
)

app.include_router(ping.router)
app.include_router(ask.router)
app.include_router(services.router)
app.include_router(transcribe.router)