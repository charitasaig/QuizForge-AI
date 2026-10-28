"""
test_quiz_validator.py — Tests for backend/core/quiz_validator.py
Covers: invalid MCQ, duplicates, flagged questions, ambiguity heuristics.
"""
from __future__ import annotations

import pytest
from backend.core.quiz_validator import validate_questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcq(question="What is X?", answer="A", options=None, flagged=False, idx=0):
    return {
        "id": f"q{idx}",
        "type": "mcq",
        "question": question,
        "options": options or {"A": "Correct", "B": "Wrong1", "C": "Wrong2", "D": "Wrong3"},
        "answer": answer,
        "explanation": "Because A.",
        "source_hint": "Source text.",
        "flagged": flagged,
    }


def _tf(question="Is X true?", answer="True", flagged=False, idx=0):
    return {
        "id": f"q{idx}",
        "type": "true_false",
        "question": question,
        "answer": answer,
        "explanation": "Because true.",
        "source_hint": "Source.",
        "flagged": flagged,
    }


def _sa(question="Explain X?", answer="X is Y.", flagged=False, idx=0):
    return {
        "id": f"q{idx}",
        "type": "short_answer",
        "question": question,
        "answer": answer,
        "explanation": "Because Y.",
        "source_hint": "Source.",
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Valid questions pass through
# ---------------------------------------------------------------------------

class TestValidQuestions:
    def test_single_valid_mcq_passes(self):
        report = validate_questions([_mcq()])
        assert report.valid_count == 1
        assert report.invalid_count == 0
        assert report.duplicate_count == 0

    def test_mixed_valid_types_pass(self):
        qs = [_mcq(idx=0), _tf(idx=1), _sa(idx=2)]
        report = validate_questions(qs)
        assert report.valid_count == 3

    def test_empty_list_returns_zero_valid(self):
        report = validate_questions([])
        assert report.valid_count == 0
        assert report.total_generated == 0


# ---------------------------------------------------------------------------
# Invalid MCQ structure
# ---------------------------------------------------------------------------

class TestInvalidMCQ:
    def test_missing_options_excluded(self):
        q = _mcq()
        q["options"] = {"A": "Only A"}   # only 1 option
        report = validate_questions([q])
        assert report.invalid_count == 1
        assert report.valid_count == 0

    def test_wrong_answer_key_excluded(self):
        q = _mcq(answer="E")   # E is not a valid option
        report = validate_questions([q])
        assert report.invalid_count == 1
        assert report.valid_count == 0

    def test_empty_option_value_flagged(self):
        q = _mcq(options={"A": "", "B": "Option B", "C": "Option C", "D": "Option D"})
        report = validate_questions([q])
        # empty option is an issue but not necessarily excluded from valid set
        issue_types = [i.issue_type for i in report.issues]
        assert "invalid_mcq" in issue_types

    def test_duplicate_option_values_flagged(self):
        q = _mcq(options={"A": "Same", "B": "Same", "C": "Different", "D": "Also Different"})
        report = validate_questions([q])
        issue_types = [i.issue_type for i in report.issues]
        assert "invalid_mcq" in issue_types

    def test_valid_tf_wrong_answer_excluded(self):
        q = _tf(answer="Maybe")
        report = validate_questions([q])
        assert report.invalid_count == 1


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_exact_duplicate_removed(self):
        q1 = _mcq(question="What is photosynthesis?", idx=0)
        q2 = _mcq(question="What is photosynthesis?", idx=1)
        report = validate_questions([q1, q2])
        assert report.duplicate_count == 1
        assert report.valid_count == 1   # only the first survives

    def test_near_duplicate_with_punctuation_removed(self):
        q1 = _mcq(question="What is photosynthesis?", idx=0)
        q2 = _mcq(question="What is photosynthesis!", idx=1)
        report = validate_questions([q1, q2])
        assert report.duplicate_count == 1

    def test_different_questions_both_kept(self):
        q1 = _mcq(question="What is photosynthesis?", idx=0)
        q2 = _mcq(question="What is cellular respiration?", idx=1)
        report = validate_questions([q1, q2])
        assert report.duplicate_count == 0
        assert report.valid_count == 2


# ---------------------------------------------------------------------------
# Flagged questions
# ---------------------------------------------------------------------------

class TestFlaggedQuestions:
    def test_flagged_question_counted(self):
        q = _mcq(flagged=True)
        report = validate_questions([q])
        assert report.flagged_count == 1

    def test_flagged_question_still_included_in_valid(self):
        # Flagged ≠ excluded — flagging is informational
        q = _mcq(flagged=True)
        report = validate_questions([q])
        assert report.valid_count == 1

    def test_unflagged_question_not_counted(self):
        q = _mcq(flagged=False)
        report = validate_questions([q])
        assert report.flagged_count == 0


# ---------------------------------------------------------------------------
# Scenario 9: Invalid MCQ (from spec)
# ---------------------------------------------------------------------------

class TestScenario9InvalidMCQ:
    def test_missing_options_key(self):
        q = {"id": "q1", "type": "mcq", "question": "Q?", "answer": "A",
             "explanation": "E", "source_hint": "", "flagged": False}
        # options key absent — should fail structural check
        report = validate_questions([q])
        assert report.invalid_count == 1

    def test_five_options_excluded(self):
        q = _mcq(options={"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"})
        report = validate_questions([q])
        assert report.invalid_count == 1

    def test_correct_answer_not_in_options(self):
        q = _mcq(answer="Z")
        report = validate_questions([q])
        assert report.invalid_count == 1


# ---------------------------------------------------------------------------
# Scenario 10: Duplicate detection (from spec)
# ---------------------------------------------------------------------------

class TestScenario10Duplicates:
    def test_three_identical_questions_keep_one(self):
        qs = [_mcq(question="Same question here?", idx=i) for i in range(3)]
        report = validate_questions(qs)
        assert report.duplicate_count == 2
        assert report.valid_count == 1
