"""
rag_service.py — Unified RAG service for QuizForge AI.

Orchestrates document chunking, in-memory vector storage, diverse topic passage retrieval,
and factual grounding verification.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from backend.rag.chunker import TextChunk, split_text_into_chunks
from backend.rag.embeddings import LocalEmbeddingEngine
from backend.rag.grounding_verifier import verify_grounding
from backend.rag.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """End-to-end RAG orchestrator for QuizForge AI."""

    def __init__(
        self,
        chunk_size: int = RAG_CHUNK_SIZE,
        chunk_overlap: int = RAG_CHUNK_OVERLAP,
        embedder: LocalEmbeddingEngine | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder or LocalEmbeddingEngine()
        self.vector_store = LocalVectorStore(embedder=self.embedder)
        self.chunks: list[TextChunk] = []

    def get_active_embedding_mode(self) -> str:
        """Return the active embedding engine name ('sentence-transformers' or 'tfidf')."""
        return self.embedder.get_active_mode()

    def ingest_notes(self, notes: str, source_name: str = "notes") -> list[TextChunk]:
        """Chunk and index lecture notes into the vector store."""
        self.vector_store.clear()
        self.chunks = split_text_into_chunks(
            notes,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            source_name=source_name,
        )
        self.vector_store.add_chunks(self.chunks)
        return self.chunks

    def retrieve_chunks_for_quiz(self, num_questions: int) -> list[TextChunk]:
        """
        Retrieve a set of diverse chunks spanning the notes to ground questions.
        If the document is short, returns all available chunks.
        """
        if not self.chunks:
            return []
        # Sample chunks across the document to ensure balanced coverage
        return self.vector_store.get_diverse_chunks(num_questions)

    def search_relevant_chunks(self, query: str, top_k: int = 3) -> list[tuple[TextChunk, float]]:
        """Search indexed chunks by semantic similarity."""
        return self.vector_store.similarity_search(query, top_k=top_k)

    def verify_grounding(
        self,
        question: dict[str, Any],
        chunk: TextChunk,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Verify that a question is factually grounded in a specific chunk."""
        return verify_grounding(question, chunk)
