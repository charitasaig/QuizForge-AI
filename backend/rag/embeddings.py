"""
embeddings.py — Free, local embedding engine for QuizForge AI RAG.

Prefers SentenceTransformers if available and initialized.
Falls back to normalized Scikit-Learn TF-IDF embeddings with zero network requirements.
"""
from __future__ import annotations

import logging
from typing import Literal
import numpy as np

from backend.nlp.semantic_similarity import SemanticSimilarityDetector

logger = logging.getLogger(__name__)


class LocalEmbeddingEngine:
    """Provides local embedding vectors for document chunks and user queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", prefer_transformer: bool = True):
        self._detector = SemanticSimilarityDetector(
            model_name=model_name,
            prefer_transformer=prefer_transformer,
        )

    @property
    def mode(self) -> Literal["sentence-transformers", "tfidf"]:
        return self._detector.get_active_mode()

    def get_active_mode(self) -> str:
        """Return the active embedding engine name."""
        return self.mode

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors."""
        return self._detector.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1D or 2D vector."""
        embs = self._detector.encode([query])
        return embs[0] if len(embs) > 0 else np.zeros((1,))
