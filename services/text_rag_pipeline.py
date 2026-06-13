"""
Text RAG Pipeline — Orchestrator Service
========================================
Chains together translation, vector store retrieval, and LLM grounded answering.

Usage:
    from services.text_rag_pipeline import TextRAGPipeline
    
    pipeline = TextRAGPipeline(translation_svc, rag_svc, answer_svc)
    result = pipeline.process("كيفاش نسجل فالكنس؟")
    print(result["answer_darija"])
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TextRAGPipeline:
    """
    Orchestrator class coordinating NLLB Translation, FAISS Retrieval, and Groq LLM
    answering into a single text-to-text pipeline.
    """

    def __init__(self, translation_service, rag_service, answer_service):
        """
        Initialize the orchestrator with pre-loaded service instances.

        Args:
            translation_service: Pre-loaded TranslationService instance.
            rag_service:         Pre-loaded RAGService instance.
            answer_service:      Pre-loaded AnswerService instance.
        """
        self.translation_svc = translation_service
        self.rag_svc = rag_service
        self.answer_svc = answer_service

        logger.info("✅ TextRAGPipeline orchestrator initialized.")

    def process(self, query_darija: str) -> Dict[str, Any]:
        """
        Process a Darija question text-to-text.

        Workflow:
            1. Translate Darija question -> English.
            2. Retrieve relevant administrative context chunks from FAISS index.
            3. Call Groq LLM (LLaMA) to generate an English response grounded in the context.
            4. Translate the English answer back -> Moroccan Darija.

        Args:
            query_darija: The user's question in Moroccan Darija (Arabic script).

        Returns:
            Dict containing query_english, answer_english, answer_darija, and retrieved_chunks.
        """
        if not query_darija or not query_darija.strip():
            raise ValueError("Query text cannot be empty.")

        logger.info("Starting pipeline execution for query: '%s'", query_darija[:50])

        # Step 1: Translate Darija input -> English
        logger.info("[Step 1/4] Translating Darija query to English...")
        query_english = self.translation_svc.darija_to_english(query_darija)
        logger.info("  -> English Query: '%s'", query_english)

        # Step 2: Retrieve context chunks using RAG Service
        logger.info("[Step 2/4] Retrieving relevant chunks from FAISS...")
        retrieved_chunks = self.rag_svc.retrieve(query_english, top_k=2)
        logger.info("  -> Retrieved %d chunks.", len(retrieved_chunks))

        # Extract raw text chunks for the Answer Service
        chunks_text = [chunk["text"] for chunk in retrieved_chunks]

        # Step 3: Generate English grounded answer using Answer Service (Groq)
        logger.info("[Step 3/4] Generating grounded answer from Groq...")
        answer_english = self.answer_svc.answer(question=query_english, chunks=chunks_text)
        logger.info("  -> English Answer: '%s...'", answer_english[:80])

        # Step 4: Translate English answer -> Darija
        logger.info("[Step 4/4] Translating English answer back to Darija...")
        answer_darija = self.translation_svc.english_to_darija(answer_english)
        logger.info("  -> Darija Answer: '%s...'", answer_darija[:80])

        logger.info("✅ Pipeline execution completed successfully.")

        return {
            "query_darija": query_darija.strip(),
            "query_english": query_english,
            "answer_english": answer_english,
            "answer_darija": answer_darija,
            "retrieved_chunks": retrieved_chunks
        }
