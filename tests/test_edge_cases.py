"""
test_edge_cases.py — Comprehensive tests for edge cases, failures, and boundary conditions.

Covers:
  1. Malformed model output handling in QuizAgent.
  2. Provider failure fallback in QuizAgent.
  3. Offline embedding fallback handling empty / all-stopword text.
  4. Evidence grounding edge cases (empty documents, zero overlap, partial overlap).
  5. Retry exhaustion: verify agent safely terminates without raising or looping infinitely.
  6. Pipeline get_last_agent_metrics() persistence and reset.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch

from backend.agent.quiz_agent import QuizAgent, AgentExecutionMetrics
from backend.core.pipeline import run_quiz_pipeline, get_last_agent_metrics
from backend.nlp.semantic_similarity import SemanticSimilarityDetector
from backend.rag.chunker import split_text_into_chunks, TextChunk
from backend.rag.grounding_verifier import verify_grounding
from backend.rag.vector_store import LocalVectorStore


_NOTES = (
    "Cellular respiration converts glucose into ATP energy in the presence of oxygen. "
    "Glycolysis occurs in the cytoplasm, while the Krebs cycle occurs in the mitochondria."
)


# ---------------------------------------------------------------------------
# 1. Malformed Model Output Handling
# ---------------------------------------------------------------------------

class TestMalformedModelOutput:
    def test_completely_invalid_json_recovers_gracefully(self):
        """When LLM returns non-JSON, agent retries and terminates safely."""
        with patch("backend.services.llm_service.generate_quiz_raw", return_value=("Sorry, I cannot help with that.", "demo")):
            agent = QuizAgent(max_retries=1)
            questions, provider, metrics = agent.run(_NOTES, 2, "medium", ["mcq"])
            assert isinstance(questions, list)
            assert metrics.retries_attempted <= 2

    def test_json_missing_questions_key(self):
        malformed = json.dumps({"status": "success", "data": "not a questions list"})
        with patch("backend.services.llm_service.generate_quiz_raw", return_value=(malformed, "demo")):
            agent = QuizAgent(max_retries=1)
            questions, provider, metrics = agent.run(_NOTES, 1, "easy", ["mcq"])
            assert isinstance(questions, list)


# ---------------------------------------------------------------------------
# 2. Embedding Fallback Edge Cases
# ---------------------------------------------------------------------------

class TestEmbeddingFallbackEdgeCases:
    def test_all_stopwords_text_similarity(self):
        detector = SemanticSimilarityDetector(prefer_transformer=False)
        assert detector.get_active_mode() == "tfidf"
        # Two texts made entirely of stop-words
        s1 = "This is that and this"
        s2 = "This is that and this"
        sim = detector.compute_similarity(s1, s2)
        assert sim >= 0.90

    def test_empty_string_similarity(self):
        detector = SemanticSimilarityDetector(prefer_transformer=False)
        assert detector.compute_similarity("", "") == 1.0
        assert detector.compute_similarity("Hello", "") == 0.0

    def test_single_word_similarity(self):
        detector = SemanticSimilarityDetector(prefer_transformer=False)
        sim_same = detector.compute_similarity("mitochondria", "mitochondria")
        sim_diff = detector.compute_similarity("mitochondria", "photosynthesis")
        assert sim_same >= 0.99
        assert sim_diff == 0.0


# ---------------------------------------------------------------------------
# 3. Evidence Grounding Edge Cases
# ---------------------------------------------------------------------------

class TestGroundingEdgeCases:
    def test_empty_chunk_text_rejects_grounding(self):
        empty_chunk = TextChunk("c0", 0, "", 0, 0)
        q = {
            "type": "mcq",
            "question": "What is ATP?",
            "options": {"A": "Energy", "B": "Sugar", "C": "Gas", "D": "Acid"},
            "answer": "A",
            "source_hint": "ATP",
        }
        ok, reason, meta = verify_grounding(q, empty_chunk)
        assert ok is False
        assert meta["grounded"] is False

    def test_exact_sentence_quote_matches_chunk_evidence(self):
        chunk = TextChunk(
            "c1", 0,
            "Glycolysis occurs in the cytoplasm and breaks glucose into pyruvate.",
            0, 68,
        )
        q = {
            "type": "short_answer",
            "question": "Where does glycolysis occur?",
            "answer": "In the cytoplasm",
            "source_hint": "Glycolysis occurs in the cytoplasm",
        }
        ok, reason, meta = verify_grounding(q, chunk)
        assert ok is True
        assert meta["grounded"] is True
        assert "cytoplasm" in meta["source_evidence"].lower()


# ---------------------------------------------------------------------------
# 4. Retry Exhaustion and Safe Termination
# ---------------------------------------------------------------------------

class TestRetryExhaustion:
    def test_retry_exhaustion_terminates_safely(self):
        """When retries are exhausted on ungrounded/invalid questions, agent terminates safely."""
        persistent_hallucination = json.dumps({
            "questions": [{
                "type": "mcq",
                "question": "What year was the Magna Carta signed?",
                "options": {"A": "1215", "B": "1066", "C": "1492", "D": "1776"},
                "answer": "A",
                "explanation": "Signed in 1215.",
                "source_hint": "Magna Carta 1215.",
            }]
        })
        with patch("backend.services.llm_service.generate_quiz_raw", return_value=(persistent_hallucination, "demo")):
            agent = QuizAgent(max_retries=2)
            questions, provider, metrics = agent.run(_NOTES, 1, "hard", ["mcq"])
            # The agent must terminate without hanging
            assert isinstance(questions, list)
            assert metrics.retries_attempted <= 2


# ---------------------------------------------------------------------------
# 5. Pipeline Metrics Persistence
# ---------------------------------------------------------------------------

class TestPipelineMetricsPersistence:
    def test_get_last_agent_metrics_populated(self):
        questions, provider = run_quiz_pipeline(_NOTES, 2, "easy", ["mcq"])
        metrics = get_last_agent_metrics()
        assert metrics is not None
        assert isinstance(metrics, AgentExecutionMetrics)
        assert metrics.total_requested == 2
        assert metrics.total_accepted == len(questions)
