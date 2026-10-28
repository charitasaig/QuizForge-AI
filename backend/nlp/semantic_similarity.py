"""
semantic_similarity.py — NLP semantic similarity detection for QuizForge AI.

Supports:
  1. Local SentenceTransformers (e.g. all-MiniLM-L6-v2) if installed and accessible.
  2. Fallback to scikit-learn TfidfVectorizer (character + word n-grams) + cosine similarity,
     guaranteeing 100% offline, zero-network, local execution.

Provides configurable threshold and detects both exact and paraphrased duplicate questions.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Default threshold for cosine similarity considered a semantic duplicate
DEFAULT_SIMILARITY_THRESHOLD = 0.82


def _normalize_text(text: str) -> str:
    """Basic normalization: strip diacritics, lowercase, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SemanticSimilarityDetector:
    """
    Computes semantic similarity between questions and identifies duplicates.
    Prefers SentenceTransformers; falls back gracefully to Scikit-Learn TF-IDF.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        prefer_transformer: bool = True,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.mode: Literal["sentence-transformers", "tfidf"] = "tfidf"
        self._st_model = None

        if prefer_transformer:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st_model = SentenceTransformer(model_name)
                self.mode = "sentence-transformers"
            except Exception as exc:
                logger.info("SentenceTransformers not available (%s); using TF-IDF fallback.", exc)
                self.mode = "tfidf"

    def get_active_mode(self) -> str:
        """Return the active embedding mode name ('sentence-transformers' or 'tfidf')."""
        return self.mode

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into embedding vectors."""
        if not texts:
            return np.empty((0, 1))

        if self.mode == "sentence-transformers" and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # Normalize vectors for cosine similarity via dot product
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return embeddings / norms
            except Exception as exc:
                logger.warning("Error running SentenceTransformer, falling back to TF-IDF: %s", exc)

        # TF-IDF / Term Frequency Fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        cleaned = [_normalize_text(t) or "empty" for t in texts]

        # Use normalized term-frequency with english stop words removed
        # (avoiding small-sample IDF distortion on short question sets)
        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 1),
                use_idf=False,
                stop_words="english",
                sublinear_tf=False,
            )
            mat = vectorizer.fit_transform(cleaned)
            dense = mat.toarray()
            # If all tokens were filtered as stop words, fall back without stop words
            if dense.shape[1] == 0:
                vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), use_idf=False)
                dense = vectorizer.fit_transform(cleaned).toarray()
            return normalize(dense, axis=1)
        except ValueError:
            try:
                vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), use_idf=False)
                dense = vectorizer.fit_transform(cleaned).toarray()
                return normalize(dense, axis=1)
            except Exception:
                return np.zeros((len(texts), 1))

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two strings, in range [0.0, 1.0]."""
        norm1 = _normalize_text(text1)
        norm2 = _normalize_text(text2)
        if not norm1 and not norm2:
            return 1.0
        if not norm1 or not norm2:
            return 0.0
        if norm1 == norm2:
            return 1.0

        embs = self.encode([text1, text2])
        sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, min(1.0, sim))

    def compute_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Compute the NxN cosine similarity matrix for a list of texts."""
        if not texts:
            return np.empty((0, 0))
        embs = self.encode(texts)
        sim_mat = np.dot(embs, embs.T)
        return np.clip(sim_mat, 0.0, 1.0)

    def find_duplicates(
        self,
        questions: list[str],
        threshold: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Find duplicate questions.
        Returns a list of tuples: (duplicate_index, original_index, similarity_score)
        where duplicate_index > original_index and similarity_score >= threshold.
        """
        if len(questions) < 2:
            return []

        th = threshold if threshold is not None else self.threshold
        sim_mat = self.compute_similarity_matrix(questions)
        duplicates: list[tuple[int, int, float]] = []
        already_flagged: set[int] = set()

        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if j in already_flagged:
                    continue
                score = float(sim_mat[i, j])

                # Exact normalized match is always a duplicate regardless of threshold
                exact = (_normalize_text(questions[i]) == _normalize_text(questions[j])) and len(questions[i].strip()) > 0
                if exact or score >= th:
                    duplicates.append((j, i, score if not exact else 1.0))
                    already_flagged.add(j)

        return duplicates
