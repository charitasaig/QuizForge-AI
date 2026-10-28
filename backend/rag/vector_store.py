"""
vector_store.py — Lightweight in-memory vector store for QuizForge AI.

Performs fast cosine similarity search and diverse passage sampling over document chunks
without requiring external servers, remote APIs, or paid vector databases.
"""
from __future__ import annotations

import numpy as np
from backend.rag.chunker import TextChunk
from backend.rag.embeddings import LocalEmbeddingEngine


class LocalVectorStore:
    """In-memory vector store for indexing and retrieving TextChunk passages."""

    def __init__(self, embedder: LocalEmbeddingEngine | None = None):
        self.embedder = embedder or LocalEmbeddingEngine()
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray = np.empty((0, 0))

    @property
    def embedding_mode(self) -> str:
        """Name of the active embedding engine."""
        return self.embedder.get_active_mode()

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Add and embed a list of TextChunk objects."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vecs = self.embedder.encode(texts)

        if len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.vectors = new_vecs
        else:
            # Check dimension compatibility (relevant if TF-IDF re-fits vocabulary)
            if self.vectors.shape[1] != new_vecs.shape[1]:
                all_chunks = self.chunks + list(chunks)
                all_texts = [c.text for c in all_chunks]
                self.chunks = all_chunks
                self.vectors = self.embedder.encode(all_texts)
            else:
                self.chunks.extend(chunks)
                self.vectors = np.vstack([self.vectors, new_vecs])

    def similarity_search(self, query: str, top_k: int = 3) -> list[tuple[TextChunk, float]]:
        """
        Search for the top_k most relevant chunks matching the query string.
        Returns a list of tuples: (chunk, cosine_similarity_score).
        """
        if not self.chunks or self.vectors.size == 0 or not query.strip():
            return []

        # When using TF-IDF, re-encode query against the current chunk texts for optimal alignment
        if self.embedder.get_active_mode() == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            all_texts = [c.text for c in self.chunks] + [query]
            try:
                vec = TfidfVectorizer(ngram_range=(1, 1), use_idf=False, stop_words="english")
                mat = normalize(vec.fit_transform(all_texts).toarray())
                chunk_mat = mat[:-1]
                query_vec = mat[-1]
                sims = np.dot(chunk_mat, query_vec)
            except Exception:
                # Fallback to standard dot product
                q_vec = self.embedder.encode_query(query)
                sims = np.dot(self.vectors, q_vec)
        else:
            q_vec = self.embedder.encode_query(query)
            sims = np.dot(self.vectors, q_vec)

        sims = np.clip(sims, 0.0, 1.0)
        ranked_indices = np.argsort(sims)[::-1][:top_k]

        results = [(self.chunks[i], float(sims[i])) for i in ranked_indices]
        return results

    def get_diverse_chunks(self, n: int) -> list[TextChunk]:
        """
        Sample n representative chunks spanning the entire document sequence.
        Ensures questions are not concentrated on only one part of the notes.
        """
        if not self.chunks:
            return []
        if len(self.chunks) <= n:
            return list(self.chunks)

        # Evenly spaced sampling across document chunks
        indices = np.linspace(0, len(self.chunks) - 1, n, dtype=int)
        seen = set()
        selected: list[TextChunk] = []
        for idx in indices:
            if idx not in seen:
                selected.append(self.chunks[idx])
                seen.add(idx)

        # Pad if unique indices were slightly fewer than n
        current_idx = 0
        while len(selected) < n and current_idx < len(self.chunks):
            if current_idx not in seen:
                selected.append(self.chunks[current_idx])
                seen.add(current_idx)
            current_idx += 1

        return selected

    def get_chunk_by_id(self, chunk_id: str) -> TextChunk | None:
        """Find a chunk by its chunk_id."""
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def clear(self) -> None:
        """Reset the vector store."""
        self.chunks.clear()
        self.vectors = np.empty((0, 0))
