"""
Test Script — Translation Service (Darija ↔ English)
=====================================================
Quick smoke test to verify the AtlasIA Terjman-Ultra NLLB model
works in both directions:

    1. Darija  → English
    2. English → Darija

Run:
    python test_translation.py
"""

import sys
import logging

# Set up visible logging so we can watch the model load
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # Step 0: Import the service
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  DarijaAssist — Translation Service Test")
    logger.info("=" * 60)

    try:
        from services.translation_service import TranslationService
    except ImportError as e:
        logger.error("Import failed: %s", e)
        logger.error("Make sure you run this from the project root directory.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Load model (first run downloads ~2.5 GB)
    # ------------------------------------------------------------------
    logger.info("Loading translation model (this may take a while on first run)…")
    try:
        translator = TranslationService()
    except RuntimeError as e:
        logger.error("Model loading failed: %s", e)
        sys.exit(1)

    logger.info("Model info: %s", translator)

    # ------------------------------------------------------------------
    # Step 2: Test Darija → English
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("  TEST 1: Darija → English")
    logger.info("-" * 60)

    darija_samples = [
        "كيفاش نسجل فالكنس؟",                          # How do I register with CNSS?
        "بغيت نعرف شحال خصني نخلص ضريبة",              # I want to know how much tax I need to pay
        "فين نمشي باش ندير البطاقة الوطنية؟",           # Where do I go to get the national ID card?
    ]

    for text in darija_samples:
        try:
            result = translator.darija_to_english(text)
            logger.info("  [AR] %s", text)
            logger.info("  [EN] %s", result)
            logger.info("")
        except Exception as e:
            logger.error("  ❌ Failed on '%s': %s", text, e)

    # ------------------------------------------------------------------
    # Step 3: Test English → Darija
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("  TEST 2: English → Darija")
    logger.info("-" * 60)

    english_samples = [
        "How do I register with CNSS?",
        "I need to renew my passport, what documents are required?",
        "Where is the nearest government office?",
    ]

    for text in english_samples:
        try:
            result = translator.english_to_darija(text)
            logger.info("  [EN] %s", text)
            logger.info("  [AR] %s", result)
            logger.info("")
        except Exception as e:
            logger.error("  ❌ Failed on '%s': %s", text, e)

    # ------------------------------------------------------------------
    # Step 4: Edge-case tests (should raise ValueError)
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("  TEST 3: Edge Cases (should raise ValueError)")
    logger.info("-" * 60)

    for bad_input in ["", "   ", None]:
        try:
            translator.darija_to_english(bad_input)
            logger.error("  ❌ Should have raised ValueError for input: %r", bad_input)
        except ValueError:
            logger.info("  ✅ Correctly rejected empty/None input: %r", bad_input)
        except Exception as e:
            logger.error("  ❌ Unexpected error for %r: %s", bad_input, e)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  ✅ All translation tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
