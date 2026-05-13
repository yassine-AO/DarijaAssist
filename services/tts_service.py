"""
TTS Service — ElevenLabs (primary) + Piper (fallback)
=====================================================
Converts Darija text into audio bytes (MP3 from ElevenLabs, WAV from Piper).

Strategy:
    1. Try ElevenLabs (eleven_multilingual_v2, Arabic-capable voice).
    2. If ElevenLabs fails for *any* reason (quota, network, auth …),
       fall back to the local Piper TTS engine (ar_JO model).

The service always returns raw audio bytes and the corresponding format
string ("mp3" or "wav") so the caller can save / stream as needed.

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

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ElevenLabs
ELEVENLABS_DEFAULT_VOICE  = "PmGnwGtnBs40iau7JfoF"   # jawad — multilingual
ELEVENLABS_DEFAULT_MODEL  = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT  = "mp3_44100_128"

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
    Text-to-Speech service with automatic ElevenLabs → Piper fallback.

    Initialisation:
        • Checks for ``ELEVENLABS_API_KEY`` in env / .env.
        • If missing, logs a warning and disables the ElevenLabs path.
        • Pre-checks (lazy) for the Piper model files.
    """

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: str = ELEVENLABS_DEFAULT_VOICE,
        elevenlabs_model_id: str = ELEVENLABS_DEFAULT_MODEL,
    ):
        load_dotenv()

        # ---------- ElevenLabs ----------
        self._el_key      = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self._el_voice_id = elevenlabs_voice_id
        self._el_model_id = elevenlabs_model_id
        self._el_client   = None

        if self._el_key:
            try:
                from elevenlabs.client import ElevenLabs
                self._el_client = ElevenLabs(api_key=self._el_key)
                logger.info("✅ ElevenLabs client ready (voice=%s).", self._el_voice_id)
            except Exception as exc:
                logger.warning("⚠️  ElevenLabs init failed, will rely on Piper: %s", exc)
        else:
            logger.warning("⚠️  ELEVENLABS_API_KEY not set — ElevenLabs disabled.")

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
            format_string is ``"mp3"`` (ElevenLabs) or ``"wav"`` (Piper).

        Raises:
            ValueError:  If the text is empty.
            RuntimeError: If both ElevenLabs *and* Piper fail.
        """
        if not text or not text.strip():
            raise ValueError("TTS input text cannot be empty.")

        # --- Try ElevenLabs first ---
        if self._el_client:
            try:
                audio = self._synthesize_elevenlabs(text)
                logger.info("🔊 Synthesised via ElevenLabs (%d bytes, mp3).", len(audio))
                return audio, "mp3"
            except Exception as exc:
                logger.warning(
                    "⚠️  ElevenLabs synthesis failed, falling back to Piper: %s", exc
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
    # ElevenLabs backend
    # ------------------------------------------------------------------

    def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Call ElevenLabs TTS and return MP3 bytes."""
        response = self._el_client.text_to_speech.convert(
            text=text,
            voice_id=self._el_voice_id,
            model_id=self._el_model_id,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
        )

        # The SDK returns an iterator of chunks
        chunks = []
        for chunk in response:
            chunks.append(chunk)

        audio_bytes = b"".join(chunks)

        if not audio_bytes:
            raise RuntimeError("ElevenLabs returned empty audio.")

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
    def elevenlabs_available(self) -> bool:
        """True if ElevenLabs client is initialised."""
        return self._el_client is not None

    @property
    def piper_model_downloaded(self) -> bool:
        """True if the Piper ONNX model exists locally."""
        return (_PIPER_MODEL_DIR / f"{_PIPER_MODEL_NAME}.onnx").exists()

    def __repr__(self) -> str:
        el = "ready" if self.elevenlabs_available else "disabled"
        pp = "downloaded" if self.piper_model_downloaded else "not yet"
        return f"<TTSService elevenlabs={el} piper_model={pp}>"
