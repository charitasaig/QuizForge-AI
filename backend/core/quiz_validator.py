"""
quiz_validator.py — Question quality validation for QuizForge AI.

Checks for:
  - Invalid MCQ structure (wrong option count, missing answer, duplicate options)
  - Duplicate questions (exact + near-exact normalised text)
  - Flagged/unsupported questions (from LLM's own flag)
  - Ambiguous questions (basic heuristics)

Returns a ValidationReport dataclass with counts and per-question details.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuestionIssue:
    question_index: int
    question_text: str
    issue_type: str          # "invalid_mcq" | "duplicate" | "flagged" | "ambiguous"
    detail: str
    excluded: bool = False   # True = removed from final quiz


@dataclass
class ValidationReport:
    total_generated: int = 0
    valid_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    flagged_count: int = 0
    issues: list[QuestionIssue] = field(default_factory=list)
    valid_questions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def _validate_mcq_structure(q: dict, index: int) -> list[QuestionIssue]:
    issues: list[QuestionIssue] = []
    options: dict = q.get("options") or {}
    answer: str = str(q.get("answer", "")).strip().upper()

    # Must have exactly 4 options with keys A-D
    expected_keys = {"A", "B", "C", "D"}
    actual_keys = {k.upper() for k in options}
    if actual_keys != expected_keys:
        issues.append(QuestionIssue(
            index, q.get("question", ""), "invalid_mcq",
            f"Expected options A–D, got: {sorted(actual_keys) or 'none'}",
            excluded=True,
        ))
        return issues  # further checks would be misleading

    # Check for empty option values
    for k, v in options.items():
        if not str(v).strip():
            issues.append(QuestionIssue(
                index, q.get("question", ""), "invalid_mcq",
                f"Option {k} is empty.", excluded=True,
            ))

    # Check for duplicate option values
    normalised_values = [_normalise(str(v)) for v in options.values()]
    if len(normalised_values) != len(set(normalised_values)):
        issues.append(QuestionIssue(
            index, q.get("question", ""), "invalid_mcq",
            "Duplicate option values detected.", excluded=False,
        ))

    # Correct answer must be one of A–D
    if answer not in expected_keys:
        issues.append(QuestionIssue(
            index, q.get("question", ""), "invalid_mcq",
            f"Correct answer '{answer}' is not one of A, B, C, D.", excluded=True,
        ))

    return issues


def _validate_true_false_structure(q: dict, index: int) -> list[QuestionIssue]:
    issues: list[QuestionIssue] = []
    answer = str(q.get("answer", "")).strip().lower()
    if answer not in ("true", "false"):
        issues.append(QuestionIssue(
            index, q.get("question", ""), "invalid_mcq",
            f"True/False answer must be 'True' or 'False', got: '{answer}'",
            excluded=True,
        ))
    return issues


def _check_ambiguous(q: dict, index: int) -> list[QuestionIssue]:
    """Very lightweight heuristic ambiguity checks."""
    issues: list[QuestionIssue] = []
    text = q.get("question", "")

    vague_refs = re.findall(r"\b(this|it|they|these|those|the method|the process)\b", text, re.I)
    if len(vague_refs) >= 2:
        issues.append(QuestionIssue(
            index, text, "ambiguous",
            f"Question contains vague references: {set(vague_refs)}",
            excluded=False,
        ))

    if len(text.split()) < 5:
        issues.append(QuestionIssue(
            index, text, "ambiguous",
            "Question text is very short and may lack context.",
            excluded=False,
        ))

    return issues


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_questions(questions: list[dict[str, Any]]) -> ValidationReport:
    """
    Run all quality checks on a list of question dicts.

    Returns a ValidationReport with:
      - valid_questions: the questions that passed all structural checks
      - issues: list of per-question problems
      - counts for the summary UI
    """
    report = ValidationReport(total_generated=len(questions))
    seen_normalised: dict[str, int] = {}   # normalised_text → first index
    valid: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()

    for i, q in enumerate(questions):
        q_type = q.get("type", "mcq")
        q_issues: list[QuestionIssue] = []

        # --- Structural checks ---
        if q_type == "mcq":
            q_issues.extend(_validate_mcq_structure(q, i))
        elif q_type == "true_false":
            q_issues.extend(_validate_true_false_structure(q, i))
        # short_answer has no strict structural requirement

        # --- Flagged by LLM ---
        if q.get("flagged"):
            q_issues.append(QuestionIssue(
                i, q.get("question", ""), "flagged",
                "Flagged by the AI as vague or weakly supported by the notes.",
                excluded=False,
            ))
            report.flagged_count += 1

        # --- Ambiguity heuristics ---
        q_issues.extend(_check_ambiguous(q, i))

        # --- Duplicate detection ---
        norm_q = _normalise(q.get("question", ""))
        if norm_q in seen_normalised:
            q_issues.append(QuestionIssue(
                i, q.get("question", ""), "duplicate",
                f"Near-identical to Q{seen_normalised[norm_q] + 1}.",
                excluded=True,
            ))
            report.duplicate_count += 1
            excluded_indices.add(i)
        else:
            seen_normalised[norm_q] = i

        # Aggregate
        report.issues.extend(q_issues)

        # Mark as excluded if any hard-fail issue was raised
        hard_fail = any(iss.excluded for iss in q_issues if iss.question_index == i)
        if hard_fail and i not in excluded_indices:
            excluded_indices.add(i)
            report.invalid_count += 1

    # Build valid list
    for i, q in enumerate(questions):
        if i not in excluded_indices:
            q_with_id = dict(q)
            q_with_id.setdefault("id", f"q{i + 1}")
            valid.append(q_with_id)

    report.valid_questions = valid
    report.valid_count = len(valid)
    return report
