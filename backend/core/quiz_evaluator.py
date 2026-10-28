"""
quiz_evaluator.py — Score calculation and answer evaluation for QuizForge AI.

Deterministic baseline evaluator:
  - MCQ / True-False: exact match on normalised answer key
  - Short Answer: keyword-presence check against expected answer
    Marked as "uncertain" when the match is partial.

Returns an EvaluationResult per question and a QuizResult summary.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str, min_len: int = 4) -> set[str]:
    """Return meaningful words (length >= min_len) from text."""
    return {w for w in _normalise(text).split() if len(w) >= min_len}


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_index: int
    question_id: str
    question_text: str
    question_type: str
    correct_answer: str
    student_answer: str
    is_correct: bool | None      # None = "needs manual review"
    score: float                 # 0 or 1 (or 0.5 for partial)
    status: str                  # "correct" | "incorrect" | "unanswered" | "uncertain"
    explanation: str
    source_hint: str


# ---------------------------------------------------------------------------
# Quiz-level result
# ---------------------------------------------------------------------------

@dataclass
class QuizResult:
    total_questions: int = 0
    correct: int = 0
    incorrect: int = 0
    unanswered: int = 0
    uncertain: int = 0
    score: float = 0.0
    max_score: float = 0.0
    percentage: float = 0.0
    time_taken_seconds: float | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
    per_question: list[QuestionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_mcq(q: dict, user_ans: str) -> tuple[bool | None, str]:
    """Return (is_correct, status)."""
    if not user_ans or not user_ans.strip():
        return None, "unanswered"
    correct = str(q.get("answer", "")).strip().upper()
    # user_ans may be "A" or "A. Some option text" — grab first non-space char
    given = user_ans.strip()[:1].upper()
    return (given == correct), ("correct" if given == correct else "incorrect")


def _evaluate_true_false(q: dict, user_ans: str) -> tuple[bool | None, str]:
    """Return (is_correct, status)."""
    if not user_ans or not user_ans.strip():
        return None, "unanswered"
    correct = str(q.get("answer", "")).strip().lower()
    given = user_ans.strip().lower()
    return (given == correct), ("correct" if given == correct else "incorrect")


def _evaluate_short_answer(q: dict, user_ans: str) -> tuple[bool | None, str]:
    """
    Keyword-overlap baseline for short answers.

    Rules:
    - Unanswered if empty.
    - "correct" if student answer shares ≥ 60 % of expected-answer keywords.
    - "uncertain" if overlap is 30–60 %.
    - "incorrect" if overlap < 30 %.
    """
    if not user_ans or not user_ans.strip():
        return None, "unanswered"

    expected_keywords = _extract_keywords(str(q.get("answer", "")))
    if not expected_keywords:
        return None, "uncertain"   # can't evaluate with no keywords

    student_keywords = _extract_keywords(user_ans)
    overlap = expected_keywords & student_keywords
    ratio = len(overlap) / len(expected_keywords)

    if ratio >= 0.60:
        return True, "correct"
    elif ratio >= 0.30:
        return None, "uncertain"
    else:
        return False, "incorrect"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_quiz(
    questions: list[dict[str, Any]],
    user_answers: dict[int, str],
    start_time: datetime | None = None,
) -> QuizResult:
    """
    Evaluate all student answers and return a QuizResult.

    Args:
        questions:    List of question dicts (from pipeline / session_state).
        user_answers: {question_index: student_answer_string}
        start_time:   When the quiz was started (for time-taken calculation).
    """
    result = QuizResult(
        total_questions=len(questions),
        max_score=float(len(questions)),
    )

    if start_time is not None:
        result.time_taken_seconds = (datetime.now() - start_time).total_seconds()

    for i, q in enumerate(questions):
        q_type = q.get("type", "mcq")
        user_ans = str(user_answers.get(i, "")).strip()
        correct_answer = str(q.get("answer", "")).strip()

        if q_type == "mcq":
            is_correct, status = _evaluate_mcq(q, user_ans)
        elif q_type == "true_false":
            is_correct, status = _evaluate_true_false(q, user_ans)
        elif q_type == "short_answer":
            is_correct, status = _evaluate_short_answer(q, user_ans)
        else:
            is_correct, status = None, "uncertain"

        score = 1.0 if is_correct is True else 0.0

        qr = QuestionResult(
            question_index=i,
            question_id=q.get("id", f"q{i+1}"),
            question_text=q.get("question", ""),
            question_type=q_type,
            correct_answer=correct_answer,
            student_answer=user_ans,
            is_correct=is_correct,
            score=score,
            status=status,
            explanation=q.get("explanation", ""),
            source_hint=q.get("source_hint", ""),
        )
        result.per_question.append(qr)

        # Tally
        if status == "correct":
            result.correct += 1
        elif status == "incorrect":
            result.incorrect += 1
        elif status == "unanswered":
            result.unanswered += 1
        elif status == "uncertain":
            result.uncertain += 1

        result.score += score

    if result.max_score > 0:
        result.percentage = (result.score / result.max_score) * 100

    return result
