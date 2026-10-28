"""
test_agentic_workflow.py — Unit tests for the bounded agentic workflow in QuizForge AI.

Covers:
  1. Planning and bounded execution limits (maximum 2 retries).
  2. Targeted self-correction on structural or duplicate rejections.
  3. Execution metrics auditing and failure category tracking.
  4. Non-breaking integration with run_quiz_pipeline().
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from backend.agent.quiz_agent import QuizAgent, AgentExecutionMetrics
from backend.core.pipeline import run_quiz_pipeline


_SAMPLE_NOTES = (
    "The mitochondria is the powerhouse of the cell. It generates most of the chemical "
    "energy needed to power the cell's biochemical reactions through cellular respiration. "
    "Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP)."
)


# ---------------------------------------------------------------------------
# 1. Bounded Execution & Metrics Tests
# ---------------------------------------------------------------------------

class TestQuizAgentBoundsAndMetrics:
    def test_agent_initialization_defaults(self):
        agent = QuizAgent()
        assert agent.max_retries == 2
        assert agent.metrics.max_retries_per_question == 2
        assert "structural" in agent.metrics.rejections_by_category
        assert "duplicate" in agent.metrics.rejections_by_category

    def test_agent_run_returns_metrics(self):
        agent = QuizAgent(max_retries=2)
        questions, provider, metrics = agent.run(
            _SAMPLE_NOTES,
            num_questions=2,
            difficulty="easy",
            question_types=["mcq"],
        )
        assert len(questions) >= 1
        assert isinstance(metrics, AgentExecutionMetrics)
        assert metrics.total_requested == 2
        assert metrics.max_retries_per_question == 2
        assert metrics.total_accepted == len(questions)


# ---------------------------------------------------------------------------
# 2. Rejection & Bounded Retry Self-Correction Tests
# ---------------------------------------------------------------------------

class TestAgentRetryAndSelfCorrection:
    def test_retry_on_structural_failure_succeeds(self):
        """
        Simulate an initial generation returning an invalid MCQ (only 2 options),
        followed by a valid retry question.
        """
        invalid_mcq = json.dumps({
            "questions": [{
                "type": "mcq",
                "question": "What does mitochondria produce?",
                "options": {"A": "ATP", "B": "Water"},  # invalid: missing C, D
                "answer": "A",
                "explanation": "ATP is produced.",
                "source_hint": "stored in a small molecule called adenosine triphosphate (ATP)",
            }]
        })

        valid_mcq = json.dumps({
            "questions": [{
                "type": "mcq",
                "question": "What energy molecule is produced by the mitochondria?",
                "options": {"A": "ATP", "B": "Glucose", "C": "Lipid", "D": "Protein"},
                "answer": "A",
                "explanation": "Mitochondria produces ATP.",
                "source_hint": "chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP)",
            }]
        })

        call_count = 0
        def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return invalid_mcq, "demo"
            return valid_mcq, "demo"

        with patch("backend.services.llm_service.generate_quiz_raw", side_effect=fake_generate):
            agent = QuizAgent(max_retries=2)
            questions, provider, metrics = agent.run(
                _SAMPLE_NOTES,
                num_questions=1,
                difficulty="medium",
                question_types=["mcq"],
            )

        assert len(questions) == 1
        assert metrics.retries_attempted >= 1
        assert metrics.retries_succeeded >= 1
        assert metrics.rejections_by_category["structural"] >= 1

    def test_retry_never_exceeds_max_retries(self):
        """
        Simulate persistent failures to verify the agent respects the bound of 2 retries.
        """
        persistent_invalid = json.dumps({
            "questions": [{
                "type": "mcq",
                "question": "Bad question",
                "options": {"A": "Only A"},
                "answer": "A",
                "source_hint": "",
            }]
        })

        call_count = 0
        def fake_generate_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return persistent_invalid, "demo"

        with patch("backend.services.llm_service.generate_quiz_raw", side_effect=fake_generate_fail):
            agent = QuizAgent(max_retries=2)
            questions, provider, metrics = agent.run(
                _SAMPLE_NOTES,
                num_questions=1,
                difficulty="easy",
                question_types=["mcq"],
            )

        # Max retries per question is 2 (initial call + at most 2 retries = max 3 calls)
        assert metrics.retries_attempted <= 2


# ---------------------------------------------------------------------------
# 3. Pipeline Integration
# ---------------------------------------------------------------------------

class TestPipelineIntegrationWithAgent:
    def test_run_quiz_pipeline_signature_preserved(self):
        questions, provider = run_quiz_pipeline(
            _SAMPLE_NOTES,
            num_questions=2,
            difficulty="easy",
            question_types=["mcq", "true_false"],
        )
        assert isinstance(questions, list)
        assert isinstance(provider, str)
        assert len(questions) >= 1
        for q in questions:
            assert "id" in q
            assert "question" in q
            assert "answer" in q
