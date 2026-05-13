"""
TTS Service — Gemini (primary) + Piper (fallback)
=====================================================
Converts Darija text into audio bytes (WAV from Gemini, WAV from Piper).

Strategy:
    1. Try Gemini (gemini-2.5-flash-tts).
    2. If Gemini fails for *any* reason (quota, network, auth …),
       fall back to the local Piper TTS engine (ar_JO model).

The service always returns raw audio bytes and the corresponding format
string ("wav") so the caller can save / stream as needed.

Usage:
    from services.tts_service import TTSService

    svc = TTSService()
    audio_bytes, fmt = svc.synthesize("كيفاش نسجل فالكنس؟")
    with open(f"output.{fmt}", "wb") as f:
        f.write(audio_bytes)
"""

import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Gemini
GEMINI_DEFAULT_VOICE = "Kore"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

# Piper (local fallback)
_PIPER_MODEL_DIR   = Path(__file__).resolve().parent.parent / "models" / "piper"
_PIPER_MODEL_NAME  = "ar_JO-kareem-medium"
_PIPER_MODEL_URL   = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx"
)
_PIPER_CONFIG_URL  = _PIPER_MODEL_URL + ".json"


class TTSService:
    """
    Text-to-Speech service with automatic Gemini → Piper fallback.

    Initialisation:
        • Checks for ``GEMINI_API_KEY`` in env / .env.
        • If missing, logs a warning and disables the Gemini path.
        • Pre-checks (lazy) for the Piper model files.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_voice: str = GEMINI_DEFAULT_VOICE,
        gemini_model_id: str = GEMINI_DEFAULT_MODEL,
    ):
        load_dotenv()

        # ---------- Gemini ----------
        self._gemini_key   = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._gemini_voice = gemini_voice
        self._gemini_model = gemini_model_id
        self._gemini_client = None

        if self._gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self._gemini_key)
                logger.info("✅ Gemini client ready (voice=%s).", self._gemini_voice)
            except Exception as exc:
                logger.warning("⚠️  Gemini init failed, will rely on Piper: %s", exc)
        else:
            logger.warning("⚠️  GEMINI_API_KEY not set — Gemini disabled.")

        # ---------- Piper ----------
        self._piper_voice = None   # lazy-loaded on first fallback call

        logger.info("TTSService initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> tuple[bytes, str]:
        """
        Convert Darija (or Arabic) text to audio.

        Args:
            text: Arabic-script text to speak.

        Returns:
            (audio_bytes, format_string)
            format_string is ``"wav"``.

        Raises:
            ValueError:  If the text is empty.
            RuntimeError: If both Gemini *and* Piper fail.
        """
        if not text or not text.strip():
            raise ValueError("TTS input text cannot be empty.")

        # --- Try Gemini first ---
        if self._gemini_client:
            try:
                audio = self._synthesize_gemini(text)
                logger.info("🔊 Synthesised via Gemini (%d bytes, wav).", len(audio))
                return audio, "wav"
            except Exception as exc:
                logger.warning(
                    "⚠️  Gemini synthesis failed, falling back to Piper: %s", exc
                )

        # --- Fallback: Piper ---
        try:
            audio = self._synthesize_piper(text)
            logger.info("🔊 Synthesised via Piper (%d bytes, wav).", len(audio))
            return audio, "wav"
        except Exception as exc:
            logger.error("❌ Piper fallback also failed: %s", exc)
            raise RuntimeError(
                "Both ElevenLabs and Piper TTS failed. "
                f"Last error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _synthesize_gemini(self, text: str) -> bytes:
        """Call Gemini TTS and return WAV bytes."""
        from google.genai import types
        
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._gemini_voice
                        )
                    )
                )
            )
        )

        audio_bytes = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    raw_pcm = part.inline_data.data
                    
                    # Gemini returns raw PCM audio (audio/L16;codec=pcm;rate=24000)
                    # We must wrap it in a proper WAV header so players can read it.
                    import wave
                    import io
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wav_file:
                        wav_file.setnchannels(1)      # Mono
                        wav_file.setsampwidth(2)      # 16-bit
                        wav_file.setframerate(24000)  # 24 kHz
                        wav_file.writeframes(raw_pcm)
                    
                    audio_bytes = buf.getvalue()
                    break

        if not audio_bytes:
            raise RuntimeError("Gemini returned empty audio.")

        return audio_bytes

    # ------------------------------------------------------------------
    # Piper backend (local fallback)
    # ------------------------------------------------------------------

    def _ensure_piper_model(self) -> Path:
        """
        Download the Piper ONNX model + config if not already present.

        Returns:
            Path to the .onnx model file.
        """
        _PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        onnx_path   = _PIPER_MODEL_DIR / f"{_PIPER_MODEL_NAME}.onnx"
        config_path = _PIPER_MODEL_DIR / f"{_PIPER_MODEL_NAME}.onnx.json"

        if onnx_path.exists() and config_path.exists():
            return onnx_path

        import requests

        for url, dest in [(_PIPER_MODEL_URL, onnx_path), (_PIPER_CONFIG_URL, config_path)]:
            logger.info("Downloading Piper asset: %s …", dest.name)
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("✅ Saved %s (%d bytes).", dest.name, dest.stat().st_size)

        return onnx_path

    def _load_piper_voice(self):
        """Lazy-load the Piper voice model."""
        if self._piper_voice is not None:
            return

        onnx_path = self._ensure_piper_model()

        # pyrefly: ignore [missing-import]
        from piper import PiperVoice

        logger.info("Loading Piper voice from %s …", onnx_path)
        self._piper_voice = PiperVoice.load(str(onnx_path))
        logger.info("✅ Piper voice loaded.")

    def _synthesize_piper(self, text: str) -> bytes:
        """Synthesise text with Piper and return WAV bytes."""
        self._load_piper_voice()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._piper_voice.synthesize_wav(text, wav_file)

        audio_bytes = buf.getvalue()

        if not audio_bytes:
            raise RuntimeError("Piper returned empty audio.")

        return audio_bytes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def gemini_available(self) -> bool:
        """True if Gemini client is initialised."""
        return self._gemini_client is not None

    @property
    def piper_model_downloaded(self) -> bool:
        """True if the Piper ONNX model exists locally."""
        return (_PIPER_MODEL_DIR / f"{_PIPER_MODEL_NAME}.onnx").exists()

    def __repr__(self) -> str:
        gm = "ready" if self.gemini_available else "disabled"
        pp = "downloaded" if self.piper_model_downloaded else "not yet"
        return f"<TTSService gemini={gm} piper_model={pp}>"
