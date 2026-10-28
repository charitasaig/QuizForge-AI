"""
backend.nlp — NLP utilities for semantic similarity, duplicate detection, and quality validation.
"""
from backend.nlp.semantic_similarity import (
    SemanticSimilarityDetector,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from backend.nlp.quality_checker import (
    check_all_quality,
    check_distractor_quality,
    check_answer_leakage,
    check_linguistic_clarity,
)

__all__ = [
    "SemanticSimilarityDetector",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "check_all_quality",
    "check_distractor_quality",
    "check_answer_leakage",
    "check_linguistic_clarity",
]
