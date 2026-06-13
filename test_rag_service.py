"""
test_rag_service.py
===================
Smoke-tests for the RAGService (FAISS Index + sentence-transformers).

Run from the project root:
    python test_rag_service.py

What is tested
--------------
1. Setup & Load  — checks if RAGService loads the sentence-transformers model and initialises.
2. Dummy Index   — checks if index files index.faiss and index.pkl are created on disk.
3. Semantic Search — retrieves items using queries related to CNSS, AMO, CIN, and Moqawala.
4. Edge Case     — checks empty query returns gracefully.
"""

import logging
import os
import sys
from pathlib import Path

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_rag_service")

# ── Import service ───────────────────────────────────────────────────────────
try:
    from services.rag_service import RAGService
except ImportError as exc:
    logger.error("Could not import RAGService: %s", exc)
    sys.exit(1)

# -- Helpers ------------------------------------------------------------------
SEPARATOR = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def pass_label(label: str) -> None:
    print(f"  [PASS] {label}")


def fail_label(label: str, reason: str) -> None:
    print(f"  [FAIL] {label}: {reason}")


# ── Test cases ───────────────────────────────────────────────────────────────

def test_initialization_and_files() -> RAGService:
    section("Test 1 — Initialization & Index persistence")
    
    svc = RAGService()
    print(f"  Service status: {svc}")
    
    # Check loaded status
    assert svc.is_loaded, "RAGService failed to mark is_loaded as True"
    assert svc.document_count == 6, f"Expected 6 documents indexed, got {svc.document_count}"
    pass_label("RAGService initialized with 6 documents")
    
    # Check index directory files
    index_path = Path("data/faiss_index/index.faiss")
    pkl_path = Path("data/faiss_index/index.pkl")
    
    assert index_path.exists(), f"Index file does not exist at {index_path}"
    assert pkl_path.exists(), f"Metadata pickle does not exist at {pkl_path}"
    
    print(f"  Saved index size: {index_path.stat().st_size:,} bytes")
    print(f"  Saved metadata size: {pkl_path.stat().st_size:,} bytes")
    pass_label("Index and metadata files persisted to data/faiss_index/")
    
    return svc


def test_semantic_retrieval(svc: RAGService) -> bool:
    section("Test 2 — Semantic Retrieval Queries")
    
    test_queries = [
        ("How do I register with CNSS?", "CNSS"),
        ("What is needed for health insurance AMO reimbursement?", "AMO"),
        ("Step to renew national identity card CIN", "CIN"),
        ("Auto-entrepreneur registration and tax rate", "Auto-Entrepreneur"),
    ]
    
    all_passed = True
    for query, expected_keyword in test_queries:
        print(f"\n  Query: '{query}'")
        results = svc.retrieve(query, top_k=2)
        
        if len(results) == 0:
            fail_label(f"Query: '{query}'", "Returned 0 results")
            all_passed = False
            continue
            
        print("  Top retrieved chunk:")
        top_hit = results[0]
        print(f"    - Doc: {top_hit['document_name']}")
        print(f"    - Score: {top_hit['relevance_score']:.4f}")
        print(f"    - Text preview: {top_hit['text'][:120]}...")
        
        # Verify if the context returned matches the expected domain
        doc_name_lower = top_hit['document_name'].lower()
        if expected_keyword.lower() in doc_name_lower or expected_keyword.lower() in top_hit['text'].lower():
            pass_label(f"Retrieved relevant document containing keyword '{expected_keyword}'")
        else:
            fail_label(f"Query: '{query}'", f"Top hit did not match expected topic '{expected_keyword}'. Got: {top_hit['document_name']}")
            all_passed = False
            
    return all_passed


def test_empty_query(svc: RAGService) -> bool:
    section("Test 3 — Empty Query Handling")
    
    results = svc.retrieve("   ")
    assert len(results) == 0, f"Expected 0 results for empty query, got {len(results)}"
    pass_label("Gracefully returned empty list for empty query")
    return True


# -- Entry point --------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  DarijaAssist - RAGService Test Suite")
    print("=" * 60)

    try:
        svc = test_initialization_and_files()
        
        results = [
            test_semantic_retrieval(svc),
            test_empty_query(svc),
        ]
        
        total  = len(results) + 1  # Including initialization test
        passed = sum(results) + 1
        failed = total - passed
        
        print(f"\n{'=' * 60}")
        print(f"  Results: {passed}/{total} passed, {failed} failed")
        print("=" * 60 + "\n")
        
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as exc:
        logger.exception("Test execution failed:")
        sys.exit(1)


if __name__ == "__main__":
    main()
