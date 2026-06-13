"""
RAG Service — FAISS Vector Store + SentenceTransformers
======================================================
Handles document embedding generation, indexing, local index persistence, 
and context chunk retrieval.

Usage:
    from services.rag_service import RAGService
    
    svc = RAGService()
    results = svc.retrieve("How do I register with CNSS?")
    for res in results:
        print(res["text"], res["document_name"], res["relevance_score"])
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional

# RAG dependencies (will be imported after verification of installation)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "index.pkl"


class RAGService:
    """
    RAGService handles local FAISS indexing and retrieval.
    It reads pre-indexed documents from disk if available; otherwise, it builds
    a dummy index of Moroccan administration procedures (CNSS, AMO, CIN, Moqawala)
    so the system is functional until the real index files are provided.
    """

    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = EMBEDDER_MODEL):
        self._index_dir = index_dir
        self._index_path = index_dir / "index.faiss"
        self._metadata_path = index_dir / "index.pkl"
        
        self._model_name = model_name
        self._embedder = None
        self._index = None
        self._chunks: List[Dict[str, str]] = []

        logger.info("Initializing RAG Service...")
        
        # Load embedding model
        try:
            logger.info("Loading SentenceTransformer model '%s' ...", self._model_name)
            self._embedder = SentenceTransformer(self._model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer model: %s", exc)
            raise RuntimeError(
                f"Could not load SentenceTransformer '{self._model_name}'. "
                "Ensure sentence-transformers is installed."
            ) from exc

        # Initialize/Load FAISS Index
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        """Loads index from disk if present, otherwise builds a dummy index."""
        if self._index_path.exists() and self._metadata_path.exists():
            try:
                logger.info("Loading existing FAISS index from '%s' ...", self._index_path)
                self._index = faiss.read_index(str(self._index_path))
                
                with open(self._metadata_path, "rb") as f:
                    self._chunks = pickle.load(f)
                    
                logger.info("✅ FAISS index loaded successfully with %d documents.", len(self._chunks))
            except Exception as exc:
                logger.error("Failed to load FAISS index from disk: %s. Rebuilding...", exc)
                self._build_dummy_index()
        else:
            logger.info("FAISS index not found on disk. Building dummy index...")
            self._build_dummy_index()

    def _build_dummy_index(self) -> None:
        """Create a set of dummy documents for Moroccan admin procedures and index them."""
        # 6 dummy documents covering Moroccan public services in English
        dummy_docs = [
            {
                "text": "To register with CNSS as an employee in Morocco, you must submit a copy of your National Identity Card (CIN), a copy of your employment contract or a certificate of employment signed by your employer, and the completed CNSS registration form. This registration allows employees to claim family allowances, health benefits, and retirement pensions.",
                "document_name": "CNSS Employee Registration Guide"
            },
            {
                "text": "For self-employed individuals and independent workers registering with CNSS, the required documents are: a copy of the National Identity Card (CIN), a trade register extract (registre de commerce), and proof of professional address. Registration must be done at the nearest local CNSS office.",
                "document_name": "CNSS Self-Employed Registration Guide"
            },
            {
                "text": "Morocco's AMO (Assurance Maladie Obligatoire) is a compulsory basic health insurance program. Eligible groups include public and private sector employees, pensioners, and since recently, all citizens under the AMO Tadamon program. To benefit, one must keep their CNSS registration up-to-date and pay monthly contributions.",
                "document_name": "AMO Eligibility Guide"
            },
            {
                "text": "Reimbursement under the AMO program requires submitting a medical claim form (feuille de soins) stamped by the prescribing doctor and the pharmacy, along with original medical prescriptions, pharmacy receipts with barcode stickers (PPV), and supporting medical reports or scans.",
                "document_name": "AMO Reimbursement Guide"
            },
            {
                "text": "To obtain or renew the Moroccan National Electronic Identity Card (CNIE or CIN), you must first pre-register online at the official CNIE portal. Then, book an appointment and go to the local police station (DGSN center) with: a birth certificate (extrait d'acte de naissance) in Arabic and French, an administrative certificate of residence, 4 recent passport photos, and the administrative fee stamp (75 dirhams).",
                "document_name": "CIN CNIE Obtain/Renew Guide"
            },
            {
                "text": "Morocco's Auto-Entrepreneur (Statut de l'Auto-entrepreneur) scheme simplifies registration for small businesses. To sign up, apply online on the auto-entrepreneur portal, print the application form, and submit it to an approved bank partner along with a copy of your CIN and a passport photo. Auto-entrepreneurs enjoy reduced tax rates (0.5% for commercial activities, 1% for services) and simplified accounting.",
                "document_name": "Auto-Entrepreneur Moqawala Guide"
            }
        ]

        try:
            texts = [doc["text"] for doc in dummy_docs]
            logger.info("Computing embeddings for %d dummy documents...", len(texts))
            embeddings = self._embedder.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype("float32")

            # Initialize index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Persist to disk
            self._index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self._index_path))
            
            with open(self._metadata_path, "wb") as f:
                pickle.dump(dummy_docs, f)

            self._index = index
            self._chunks = dummy_docs
            logger.info("✅ Dummy FAISS index built and saved to '%s'.", self._index_dir)
        except Exception as exc:
            logger.error("Failed to build/save dummy FAISS index: %s", exc)
            raise RuntimeError("Could not initialize dummy FAISS index.") from exc

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the top_k relevant documents for a given query.

        Args:
            query: The search query in English.
            top_k: Number of relevant documents to retrieve.

        Returns:
            List of dicts: [
                {
                    "text": "...",
                    "document_name": "...",
                    "relevance_score": 0.85
                }
            ]
        """
        if not query or not query.strip():
            logger.warning("Empty search query received in RAG retrieve.")
            return []

        if self._index is None or not self._chunks:
            logger.warning("RAG index is not loaded. Cannot retrieve.")
            return []

        try:
            # Generate query embedding
            query_vector = self._embedder.encode([query], convert_to_numpy=True).astype("float32")

            # Search FAISS index
            distances, indices = self._index.search(query_vector, top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx >= len(self._chunks):
                    continue

                chunk = self._chunks[idx]
                distance = float(distances[0][i])
                
                # Convert L2 distance to a pseudo similarity score: 1 / (1 + distance)
                relevance_score = 1.0 / (1.0 + distance)

                results.append({
                    "text": chunk["text"],
                    "document_name": chunk["document_name"],
                    "relevance_score": round(relevance_score, 4)
                })

            logger.info("Retrieved %d matches for query: '%s'", len(results), query[:50])
            return results

        except Exception as exc:
            logger.error("Error during retrieval: %s", exc)
            return []

    @property
    def is_loaded(self) -> bool:
        """Returns True if the FAISS index and chunks are loaded and ready."""
        return self._index is not None and len(self._chunks) > 0

    @property
    def document_count(self) -> int:
        """Returns the total number of documents indexed."""
        return len(self._chunks)

    def __repr__(self) -> str:
        status = f"loaded with {self.document_count} docs" if self.is_loaded else "NOT loaded"
        return f"<RAGService model='{self._model_name}' status='{status}'>"
