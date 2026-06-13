"""
test_full_pipeline_api.py
==========================
Integration test for the FastAPI /ask endpoint.

Uses FastAPI's TestClient to spin up the API (which triggers lifespan model loading) 
and submits a real spoken Darija WAV file.

Run from project root:
    python test_full_pipeline_api.py
"""

import logging
import sys
import base64
from pathlib import Path

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_full_pipeline_api")

# -- Imports ------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from main import app
except ImportError as exc:
    logger.error("Failed to import test client dependencies: %s", exc)
    logger.error("Run: pip install httpx")
    sys.exit(1)


def main():
    # Force stdout/stderr to use UTF-8 to avoid encoding crashes on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("  DarijaAssist - End-to-End Voice API Integration Test")
    print("=" * 60)

    # Path to test spoken audio file (generated from test_tts_service.py)
    audio_path = Path("tests_output/test_default.wav")
    if not audio_path.exists():
        logger.error(
            "Test audio file not found at %s.\n"
            "Please run 'python test_tts_service.py' first to generate test files.",
            audio_path
        )
        sys.exit(1)

    logger.info("Initializing FastAPI TestClient (loading models)...")
    try:
        with TestClient(app) as client:
            logger.info("TestClient loaded. Sending POST request to /ask ...")
            
            with open(audio_path, "rb") as audio_file:
                files = {"audio": (audio_path.name, audio_file, "audio/wav")}
                response = client.post("/ask", files=files)
                
            print("\n" + "-" * 60)
            print(f"  Response Status: {response.status_code}")
            print("-" * 60)

            if response.status_code != 200:
                logger.error("API returned error: %s", response.text)
                sys.exit(1)

            resp_data = response.json()
            
            print("\n=== API PIPELINE OUTPUTS ===")
            print(f"Request ID         : {resp_data.get('request_id')}")
            print(f"Whisper Transcript : {resp_data.get('pipeline_meta', {}).get('whisper_transcript')}")
            print(f"English Query      : {resp_data.get('pipeline_meta', {}).get('english_translation')}")
            print(f"Processing Time    : {resp_data.get('pipeline_meta', {}).get('processing_time_ms')} ms")
            
            print("\nRetrieved Source Info:")
            source = resp_data.get("source", {})
            print(f"  - Document Name  : {source.get('document_name')}")
            print(f"  - Relevance Score: {source.get('relevance_score')}")
            print(f"  - Preview        : {source.get('chunk_preview')}")
            
            print(f"\nFinal Darija Answer Text:\n{resp_data.get('answer_text_darija')}")
            
            audio_b64 = resp_data.get("answer_audio_b64")
            audio_len_bytes = len(base64.b64decode(audio_b64)) if audio_b64 else 0
            print(f"\nAnswer Audio Output: {audio_len_bytes:,} bytes of decoded audio (Base64 length: {len(audio_b64):,})")
            print("============================================\n")

            assert resp_data.get("request_id") is not None, "Missing request_id"
            assert audio_len_bytes > 0, "Audio response is empty"
            assert len(resp_data.get("answer_text_darija", "")) > 0, "Darija answer text is empty"
            
            print("  [PASS] Full End-to-End Voice API Integration Test completed successfully!")
            sys.exit(0)

    except Exception as exc:
        logger.exception("Integration test failed:")
        print("  [FAIL] Test run encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
