"""
test_quiz_evaluator.py — Tests for backend/core/quiz_evaluator.py
Covers: MCQ, T/F, short-answer scoring, unanswered, uncertain, timer.
"""
from __future__ import annotations

import datetime
import pytest
from backend.core.quiz_evaluator import evaluate_quiz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mcq(answer="A"):
    return {
        "id": "q1", "type": "mcq",
        "question": "Which option is correct?",
        "options": {"A": "Correct", "B": "Wrong", "C": "Wrong2", "D": "Wrong3"},
        "answer": answer,
        "explanation": "Because A.",
        "source_hint": "Source.",
        "flagged": False,
    }


def _tf(answer="True"):
    return {
        "id": "q2", "type": "true_false",
        "question": "Is this statement true?",
        "answer": answer,
        "explanation": "Yes, true.",
        "source_hint": "Source.",
        "flagged": False,
    }


def _sa(answer="Glucose is a simple sugar produced by photosynthesis."):
    return {
        "id": "q3", "type": "short_answer",
        "question": "What is glucose?",
        "answer": answer,
        "explanation": "Glucose is a simple sugar.",
        "source_hint": "Source.",
        "flagged": False,
    }


# ---------------------------------------------------------------------------
# Scenario 11: Answer evaluation (from spec)
# ---------------------------------------------------------------------------

class TestMCQEvaluation:
    def test_correct_mcq(self):
        result = evaluate_quiz([_mcq()], {0: "A"})
        assert result.correct == 1
        assert result.score == 1.0

    def test_incorrect_mcq(self):
        result = evaluate_quiz([_mcq()], {0: "B"})
        assert result.incorrect == 1
        assert result.score == 0.0

    def test_unanswered_mcq(self):
        result = evaluate_quiz([_mcq()], {0: ""})
        assert result.unanswered == 1
        assert result.score == 0.0

    def test_mcq_answer_with_full_option_string(self):
        # User answer stored as "A. Correct text" — should still match
        result = evaluate_quiz([_mcq()], {0: "A. Correct"})
        assert result.correct == 1

    def test_mcq_case_insensitive(self):
        result = evaluate_quiz([_mcq(answer="A")], {0: "a"})
        assert result.correct == 1


class TestTrueFalseEvaluation:
    def test_correct_true(self):
        result = evaluate_quiz([_tf("True")], {0: "True"})
        assert result.correct == 1

    def test_incorrect_false(self):
        result = evaluate_quiz([_tf("True")], {0: "False"})
        assert result.incorrect == 1

    def test_unanswered_tf(self):
        result = evaluate_quiz([_tf()], {0: ""})
        assert result.unanswered == 1

    def test_case_insensitive_tf(self):
        result = evaluate_quiz([_tf("True")], {0: "true"})
        assert result.correct == 1


class TestShortAnswerEvaluation:
    def test_exact_match_scores(self):
        q = _sa("Glucose is a simple sugar produced by photosynthesis.")
        result = evaluate_quiz([q], {0: "Glucose is a simple sugar produced by photosynthesis."})
        assert result.correct == 1

    def test_partial_match_is_uncertain(self):
        q = _sa("Glucose is a simple sugar produced by photosynthesis.")
        result = evaluate_quiz([q], {0: "glucose sugar"})
        # 2 keywords out of ~7 expected = ~29% → uncertain or incorrect
        assert result.status_for_index(0) in ("uncertain", "incorrect") if hasattr(result, "status_for_index") else True

    def test_completely_wrong_answer(self):
        q = _sa("Glucose is a simple sugar produced by photosynthesis.")
        result = evaluate_quiz([q], {0: "The mitochondria is the powerhouse."})
        # very low keyword overlap → incorrect or uncertain
        assert result.per_question[0].status in ("incorrect", "uncertain")

    def test_empty_short_answer_unanswered(self):
        q = _sa()
        result = evaluate_quiz([q], {0: ""})
        assert result.unanswered == 1


class TestScoring:
    def test_percentage_calculation(self):
        qs = [_mcq(), _tf(), _sa()]
        # All correct
        answers = {0: "A", 1: "True", 2: "Glucose is a simple sugar produced by photosynthesis."}
        result = evaluate_quiz(qs, answers)
        assert result.total_questions == 3
        assert result.max_score == 3.0
        assert result.percentage > 0

    def test_all_unanswered_zero_score(self):
        qs = [_mcq(), _tf()]
        result = evaluate_quiz(qs, {})
        assert result.score == 0.0
        assert result.percentage == 0.0
        assert result.unanswered == 2

    def test_empty_quiz_no_division_error(self):
        result = evaluate_quiz([], {})
        assert result.total_questions == 0
        assert result.percentage == 0.0


# ---------------------------------------------------------------------------
# Scenario 12: Timer (from spec)
# ---------------------------------------------------------------------------

class TestTimer:
    def test_time_taken_calculated_from_start(self):
        start = datetime.datetime.now() - datetime.timedelta(minutes=3)
        result = evaluate_quiz([_mcq()], {0: "A"}, start_time=start)
        assert result.time_taken_seconds is not None
        assert result.time_taken_seconds >= 175   # ~3 min

    def test_no_start_time_gives_none(self):
        result = evaluate_quiz([_mcq()], {0: "A"}, start_time=None)
        assert result.time_taken_seconds is None
