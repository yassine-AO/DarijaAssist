"""
test_tts_service.py
===================
Smoke-tests for the TTSService (ElevenLabs + Piper fallback).

Run from the project root:
    python test_tts_service.py

What is tested
--------------
1. Happy path   — Darija text → audio bytes saved to disk.
2. Piper only   — Force the Piper fallback (no ElevenLabs key).
3. Empty input  — must raise ValueError immediately.

Output files are written to tests_output/ for manual listening.

Requirements
------------
• ELEVENLABS_API_KEY in .env  (optional — Piper still works without it)
• pip install elevenlabs piper-tts requests python-dotenv
"""

import logging
import os
import sys
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_tts_service")

# ── Import service ───────────────────────────────────────────────────────────
try:
    from services.tts_service import TTSService
except ImportError as exc:
    logger.error("Could not import TTSService: %s", exc)
    sys.exit(1)

# ── Output dir ───────────────────────────────────────────────────────────────
OUT_DIR = Path("tests_output")
OUT_DIR.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
SEPARATOR = "─" * 60
DARIJA_TEXT = "كيفاش نقدر نسجل فالكنس؟ بغيت نعرف الوثائق اللي خاصني."


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def pass_label(label: str) -> None:
    print(f"  ✅  PASS — {label}")


def fail_label(label: str, reason: str) -> None:
    print(f"  ❌  FAIL — {label}: {reason}")


# ── Test cases ───────────────────────────────────────────────────────────────

def test_synthesize_default(svc: TTSService) -> bool:
    """Full synthesis — uses whichever backend is available."""
    section("Test 1 — Default synthesis (ElevenLabs → Piper fallback)")

    try:
        audio, fmt = svc.synthesize(DARIJA_TEXT)
        out_path = OUT_DIR / f"test_default.{fmt}"
        out_path.write_bytes(audio)
        print(f"\n  Input  : {DARIJA_TEXT}")
        print(f"  Format : {fmt}")
        print(f"  Size   : {len(audio):,} bytes")
        print(f"  Saved  : {out_path}\n")
        pass_label(f"Audio saved ({fmt})")
        return True
    except Exception as exc:
        fail_label("Default synthesis", str(exc))
        return False


def test_piper_only() -> bool:
    """Force Piper by creating a service with no ElevenLabs key."""
    section("Test 2 — Piper-only (no ElevenLabs key)")

    try:
        svc_piper = TTSService(elevenlabs_api_key="__disabled__")
        # The invalid key will cause ElevenLabs init to fail → only Piper
        # But actually we should pass a clearly invalid key so it errors
        # OR just pass nothing. Let's create one with explicitly no key.
        svc_piper = TTSService.__new__(TTSService)
        # Manually init with no ElevenLabs
        svc_piper._el_client = None
        svc_piper._el_key = None
        svc_piper._el_voice_id = None
        svc_piper._el_model_id = None
        svc_piper._piper_voice = None

        audio, fmt = svc_piper.synthesize(DARIJA_TEXT)
        assert fmt == "wav", f"Expected wav but got {fmt}"

        out_path = OUT_DIR / f"test_piper.{fmt}"
        out_path.write_bytes(audio)
        print(f"\n  Input  : {DARIJA_TEXT}")
        print(f"  Format : {fmt}")
        print(f"  Size   : {len(audio):,} bytes")
        print(f"  Saved  : {out_path}\n")
        pass_label("Piper fallback produced WAV audio")
        return True
    except Exception as exc:
        fail_label("Piper-only", str(exc))
        return False


def test_empty_text_raises(svc: TTSService) -> bool:
    """Empty text must raise ValueError immediately."""
    section("Test 3 — Empty text → ValueError")

    try:
        svc.synthesize("   ")
        fail_label("Empty text guard", "No exception raised!")
        return False
    except ValueError as exc:
        print(f"\n  Caught expected ValueError: {exc}\n")
        pass_label("ValueError raised for empty text")
        return True
    except Exception as exc:
        fail_label("Empty text guard", f"Wrong type: {type(exc).__name__}: {exc}")
        return False


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  DarijaAssist — TTSService Test Suite")
    print("═" * 60)

    logger.info("Initialising TTSService …")
    try:
        svc = TTSService()
    except Exception as exc:
        logger.error("Setup failed: %s", exc)
        sys.exit(1)

    print(f"  Service state: {svc}")

    results = [
        test_synthesize_default(svc),
        test_piper_only(),
        test_empty_text_raises(svc),
    ]

    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'═' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"  Audio files → {OUT_DIR.resolve()}")
    print("═" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
