"""
test_answer_service.py
======================
Quick smoke-tests for the AnswerService (Groq Cloud).

Run from the project root (with the .venv active):
    python test_answer_service.py

What is tested
--------------
1. Happy path   — question + real context chunks → English answer printed.
2. No chunks    — service must still return a valid answer (fallback to model knowledge).
3. Empty input  — service must raise ValueError immediately, no API call made.

Requirements
------------
• GROQ_API_KEY must be set in .env (or the environment).
• pip install groq python-dotenv
"""

import logging
import sys

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_answer_service")

# ── Import service ───────────────────────────────────────────────────────────
try:
    from services.answer_service import AnswerService
except ImportError as exc:
    logger.error("Could not import AnswerService: %s", exc)
    sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────────────────────

SEPARATOR = "─" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def pass_label(label: str) -> None:
    print(f"  ✅  PASS — {label}")


def fail_label(label: str, reason: str) -> None:
    print(f"  ❌  FAIL — {label}: {reason}")


# ── Test cases ───────────────────────────────────────────────────────────────

def test_happy_path(svc: AnswerService) -> bool:
    """Question + context chunks → coherent English answer."""
    section("Test 1 — Happy path (question + chunks)")

    question = "What documents do I need to register with CNSS?"
    chunks = [
        "To register with CNSS (Caisse Nationale de Sécurité Sociale) in Morocco, "
        "an employee must submit a CIN (Carte d'Identité Nationale) copy, a work "
        "contract or employer letter, and a completed CNSS registration form.",
        "Self-employed individuals need a trade register extract (registre de commerce) "
        "and proof of professional address in addition to the standard CIN copy.",
    ]

    try:
        answer = svc.answer(question=question, chunks=chunks)
        print(f"\n  Question : {question}")
        print(f"\n  Answer   :\n  {answer}\n")
        pass_label("Received a non-empty answer")
        return True
    except Exception as exc:
        fail_label("Happy path", str(exc))
        return False


def test_no_chunks(svc: AnswerService) -> bool:
    """Service must answer even when no chunks are supplied."""
    section("Test 2 — No chunks (general knowledge fallback)")

    question = "What is the main purpose of Morocco's ANAPEC agency?"

    try:
        answer = svc.answer(question=question, chunks=[])
        print(f"\n  Question : {question}")
        print(f"\n  Answer   :\n  {answer}\n")
        pass_label("Got answer with empty chunk list")
        return True
    except Exception as exc:
        fail_label("No chunks", str(exc))
        return False


def test_empty_question_raises(svc: AnswerService) -> bool:
    """Passing an empty question must raise ValueError immediately."""
    section("Test 3 — Empty question → ValueError")

    try:
        svc.answer(question="   ", chunks=["some context"])
        fail_label("Empty question guard", "No exception raised — guard is missing!")
        return False
    except ValueError as exc:
        print(f"\n  Caught expected ValueError: {exc}\n")
        pass_label("ValueError raised for empty question")
        return True
    except Exception as exc:
        fail_label("Empty question guard", f"Wrong exception type: {type(exc).__name__}: {exc}")
        return False


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  DarijaAssist — AnswerService Test Suite (Groq)")
    print("═" * 60)

    # Initialise the service once (raises early if the API key is missing)
    logger.info("Initialising AnswerService …")
    try:
        svc = AnswerService()
    except EnvironmentError as exc:
        logger.error("Setup failed: %s", exc)
        print(
            "\n  ❌  Cannot run tests — API key not configured.\n"
            "  → Add GROQ_API_KEY=<your-key> to your .env file.\n"
        )
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Setup failed: %s", exc)
        sys.exit(1)

    results = [
        test_happy_path(svc),
        test_no_chunks(svc),
        test_empty_question_raises(svc),
    ]

    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'═' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("═" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
