"""
backend.rag — Document chunking, vector storage, and evidence grounding for QuizForge AI.
"""
from backend.rag.chunker import TextChunk, split_text_into_chunks
from backend.rag.embeddings import LocalEmbeddingEngine
from backend.rag.grounding_verifier import verify_grounding
from backend.rag.vector_store import LocalVectorStore
from backend.rag.rag_service import RAGService

__all__ = [
    "TextChunk",
    "split_text_into_chunks",
    "LocalEmbeddingEngine",
    "LocalVectorStore",
    "verify_grounding",
    "RAGService",
]
