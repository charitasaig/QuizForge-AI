"""
quiz_exporter.py — TXT and CSV export for QuizForge AI.

Exports:
  1. Questions-only TXT  (no answers)
  2. Answer-key TXT      (answers + explanations)
  3. Full CSV            (all fields, UTF-8, properly escaped)
  4. Results CSV         (student answers + grading outcome)
"""
from __future__ import annotations

import csv
import io
import datetime
from typing import Any


# ---------------------------------------------------------------------------
# TXT — questions only (safe for students)
# ---------------------------------------------------------------------------

def export_questions_txt(
    questions: list[dict[str, Any]],
    difficulty: str,
    quiz_id: str = "",
) -> bytes:
    """Return UTF-8 encoded TXT with questions and options but NO answers."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("QuizForge AI — Quiz Questions")
    if quiz_id:
        lines.append(f"Quiz ID : {quiz_id}")
    lines.append(f"Difficulty : {difficulty.capitalize()}")
    lines.append(f"Questions  : {len(questions)}")
    lines.append(f"Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)
    lines.append("Note: Answers and explanations are in the separate answer-key file.")
    lines.append("")

    for i, q in enumerate(questions):
        q_type = q.get("type", "mcq").upper().replace("_", "/")
        flag = " [FLAGGED]" if q.get("flagged") else ""
        lines.append(f"Q{i + 1} [{q_type}]{flag}")
        lines.append(q.get("question", ""))
        lines.append("")

        if q.get("type") == "mcq":
            for k, v in (q.get("options") or {}).items():
                lines.append(f"  {k}. {v}")
            lines.append("")

        elif q.get("type") == "true_false":
            lines.append("  True / False")
            lines.append("")

        elif q.get("type") == "short_answer":
            lines.append("  [Write your answer here]")
            lines.append("")

        lines.append("-" * 40)

    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# TXT — answer key
# ---------------------------------------------------------------------------

def export_answer_key_txt(
    questions: list[dict[str, Any]],
    difficulty: str,
    quiz_id: str = "",
) -> bytes:
    """Return UTF-8 encoded answer-key TXT."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("QuizForge AI — Answer Key & Explanations")
    if quiz_id:
        lines.append(f"Quiz ID : {quiz_id}")
    lines.append(f"Difficulty : {difficulty.capitalize()}")
    lines.append(f"Questions  : {len(questions)}")
    lines.append(f"Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)
    lines.append("")

    for i, q in enumerate(questions):
        q_type = q.get("type", "mcq")
        lines.append(f"Q{i + 1}. {q.get('question', '')}")
        lines.append(f"Type       : {q_type.upper().replace('_', '/')}")
        lines.append(f"Answer     : {q.get('answer', '')}")
        if q_type == "short_answer":
            lines.append(f"Expected   : {q.get('answer', '')}")
        lines.append(f"Explanation: {q.get('explanation', '')}")
        if q.get("source_hint"):
            lines.append(f"Source     : {q.get('source_hint', '')}")
        if q.get("flagged"):
            lines.append("** FLAGGED: This question was marked as potentially vague or weakly supported. **")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# CSV — full quiz data
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "quiz_id", "question_id", "question_type", "difficulty",
    "question", "option_a", "option_b", "option_c", "option_d",
    "correct_answer", "expected_answer", "explanation", "source_evidence", "flagged",
]


def export_quiz_csv(
    questions: list[dict[str, Any]],
    difficulty: str,
    quiz_id: str = "",
) -> bytes:
    """Return UTF-8 encoded CSV of the full quiz."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    for i, q in enumerate(questions):
        q_type = q.get("type", "mcq")
        opts = q.get("options") or {}
        row = {
            "quiz_id": quiz_id,
            "question_id": q.get("id", f"q{i + 1}"),
            "question_type": q_type,
            "difficulty": difficulty,
            "question": q.get("question", ""),
            "option_a": opts.get("A", ""),
            "option_b": opts.get("B", ""),
            "option_c": opts.get("C", ""),
            "option_d": opts.get("D", ""),
            "correct_answer": q.get("answer", ""),
            "expected_answer": q.get("answer", "") if q_type == "short_answer" else "",
            "explanation": q.get("explanation", ""),
            "source_evidence": q.get("source_hint", ""),
            "flagged": str(q.get("flagged", False)),
        }
        writer.writerow(row)

    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# CSV — results (student answers + grading)
# ---------------------------------------------------------------------------

_RESULTS_COLUMNS = [
    "quiz_id", "question_id", "question_type", "question",
    "correct_answer", "student_answer", "status", "score",
]


def export_results_csv(
    quiz_result,   # QuizResult from quiz_evaluator
    quiz_id: str = "",
) -> bytes:
    """Return UTF-8 encoded CSV of the student's results."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_RESULTS_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    for qr in quiz_result.per_question:
        row = {
            "quiz_id": quiz_id,
            "question_id": qr.question_id,
            "question_type": qr.question_type,
            "question": qr.question_text,
            "correct_answer": qr.correct_answer,
            "student_answer": qr.student_answer,
            "status": qr.status,
            "score": qr.score,
        }
        writer.writerow(row)

    return buf.getvalue().encode("utf-8")
