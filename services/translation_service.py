"""
Translation Service — AtlasIA Terjman-Ultra (NLLB-200 fine-tuned for Darija)
=============================================================================
Handles bidirectional translation between English and Moroccan Darija (ary).

Model : atlasia/Terjman-Ultra-v2.0  (1.3B params, NLLB-200 backbone)
Codes : eng_Latn  = English
        ary_Arab  = Moroccan Darija (Arabic script)

Usage:
    from services.translation_service import TranslationService

    svc = TranslationService()          # loads model + tokenizer once
    en  = svc.darija_to_english("كيفاش نسجل فالكنس؟")
    dar = svc.english_to_darija("How do I register with CNSS?")
"""

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "BounharAbdelaziz/Terjman-Ultra-v2.0"

# NLLB language codes used by the tokenizer
LANG_ENGLISH = "eng_Latn"
LANG_DARIJA  = "ary_Arab"

# Generation defaults (conservative — good for short government-domain text)
DEFAULT_MAX_LENGTH = 256
DEFAULT_NUM_BEAMS  = 4


class TranslationService:
    """
    Singleton-style wrapper around the Terjman-Ultra NLLB translation model.

    Loads the model + tokenizer once, then exposes two clean methods:
        • darija_to_english(text)
        • english_to_darija(text)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Load the tokenizer and model into memory.

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to atlasia/Terjman-Ultra-v2.0.

        Raises:
            RuntimeError: If model files cannot be downloaded / loaded.
        """
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

        # Determine the best available device (GPU if possible, else CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Translation device: %s", self._device)

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download (if needed) and load the tokenizer + model weights."""
        try:
            logger.info("Loading tokenizer from '%s' …", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            logger.info("Loading model from '%s' …", self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()  # inference mode — no dropout, etc.

            logger.info("✅ Translation model loaded successfully on %s.", self._device)

        except Exception as exc:
            logger.error("❌ Failed to load translation model: %s", exc)
            raise RuntimeError(
                f"Could not load translation model '{self._model_name}'. "
                f"Make sure you have accepted the model conditions on HuggingFace "
                f"and that 'transformers', 'torch', and 'sentencepiece' are installed."
            ) from exc

    def _translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> str:
        """
        Core translation method used by both public helpers.

        Args:
            text:       Source text to translate.
            src_lang:   NLLB source language code (e.g. 'eng_Latn').
            tgt_lang:   NLLB target language code (e.g. 'ary_Arab').
            max_length: Maximum tokens in the generated output.
            num_beams:  Beam search width (higher = slower but better quality).

        Returns:
            Translated text as a plain string.

        Raises:
            ValueError:  If the input text is empty or None.
            RuntimeError: If the model is not loaded.
        """
        # --- Input validation ---
        if not text or not text.strip():
            raise ValueError("Translation input text cannot be empty.")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Translation model is not loaded. Cannot translate.")

        # --- Tokenize with source / target language hints ---
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self._device)

        # --- Generate translation ---
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length,
                num_beams=num_beams,
            )

        # --- Decode back to text ---
        translated_text = self._tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return translated_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def darija_to_english(
        self,
        text: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> str:
        """
        Translate Moroccan Darija (Arabic script) → English.

        Args:
            text: Darija text to translate.

        Returns:
            English translation string.

        Raises:
            ValueError: If text is empty.
        """
        logger.debug("Darija → English | input: %s", text[:80])
        result = self._translate(
            text=text,
            src_lang=LANG_DARIJA,
            tgt_lang=LANG_ENGLISH,
            max_length=max_length,
            num_beams=num_beams,
        )
        logger.debug("Darija → English | output: %s", result[:80])
        return result

    def english_to_darija(
        self,
        text: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> str:
        """
        Translate English → Moroccan Darija (Arabic script).

        Args:
            text: English text to translate.

        Returns:
            Darija translation string (Arabic script).

        Raises:
            ValueError: If text is empty.
        """
        logger.debug("English → Darija | input: %s", text[:80])
        result = self._translate(
            text=text,
            src_lang=LANG_ENGLISH,
            tgt_lang=LANG_DARIJA,
            max_length=max_length,
            num_beams=num_beams,
        )
        logger.debug("English → Darija | output: %s", result[:80])
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if both the model and tokenizer are ready."""
        return self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> str:
        """Return the device the model is running on."""
        return str(self._device)

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "NOT loaded"
        return f"<TranslationService model='{self._model_name}' status={status} device={self._device}>"
