"""
test_text_rag_pipeline.py
==========================
Smoke test for the TextRAGPipeline orchestrator.

Run from project root:
    python test_text_rag_pipeline.py

Verifies:
1. End-to-end integration of Translation, RAG, and Groq Answer services.
2. A Darija query is successfully translated, matched against index documents,
   answered in English, and translated back to Darija.
"""

import logging
import sys

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_text_rag_pipeline")

# -- Imports ------------------------------------------------------------------
try:
    from services.translation_service import TranslationService
    from services.rag_service import RAGService
    from services.answer_service import AnswerService
    from services.text_rag_pipeline import TextRAGPipeline
except ImportError as exc:
    logger.error("Failed to import core services: %s", exc)
    logger.error("Ensure you run this from the project root directory.")
    sys.exit(1)


def main():
    # Force stdout and stderr to use UTF-8 to avoid encoding errors on Windows terminal
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("  DarijaAssist - Text RAG Pipeline End-to-End Test")
    print("=" * 60)

    # 1. Initialize core services
    logger.info("Initializing Translation Service...")
    try:
        translation_svc = TranslationService()
    except Exception as e:
        logger.error("Failed to load TranslationService: %s", e)
        sys.exit(1)

    logger.info("Initializing RAG Service (FAISS)...")
    try:
        rag_svc = RAGService()
    except Exception as e:
        logger.error("Failed to load RAGService: %s", e)
        sys.exit(1)

    logger.info("Initializing Answer Service (Groq)...")
    try:
        answer_svc = AnswerService()
    except Exception as e:
        logger.error("Failed to load AnswerService: %s. Check GROQ_API_KEY.", e)
        sys.exit(1)

    # 2. Instantiate pipeline
    logger.info("Initializing Text RAG Pipeline Orchestrator...")
    pipeline = TextRAGPipeline(
        translation_service=translation_svc,
        rag_service=rag_svc,
        answer_service=answer_svc
    )

    # 3. Process test query in Darija:
    # "كيفاش نسجل فالكنس؟" (How do I register with CNSS?)
    test_query = "كيفاش نسجل فالكنس؟"

    print("\n" + "-" * 60)
    print(f"  Query (Darija): {test_query}")
    print("-" * 60)

    try:
        result = pipeline.process(test_query)

        print("\n=== INTERMEDIATE PIPELINE OUTPUTS ===")
        print(f"1. English Query   : {result['query_english']}")
        
        print("\n2. Retrieved Chunks:")
        for idx, chunk in enumerate(result['retrieved_chunks'], start=1):
            print(f"   [{idx}] Source: '{chunk['document_name']}' (Score: {chunk['relevance_score']:.4f})")
            print(f"       Text  : {chunk['text'][:120]}...")
            
        print(f"\n3. English Answer  :\n{result['answer_english']}")
        
        print("\n=== FINAL PIPELINE OUTPUT ===")
        print(f"Darija Answer      :\n{result['answer_darija']}")
        print("=====================================\n")

        print("  [PASS] End-to-end Text RAG Pipeline executed successfully!")
        sys.exit(0)

    except Exception as exc:
        logger.exception("Pipeline execution failed:")
        print("  [FAIL] Test execution encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
