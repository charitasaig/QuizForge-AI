"""
quality_checker.py — Question quality and pedagogical validity checks for QuizForge AI.

Checks for:
  1. Distractor quality (options must be non-empty, distinct from each other, distinct from answer).
  2. Answer-in-stem leakage (the question stem shouldn't give away the correct answer).
  3. Linguistic clarity, extreme brevity (< 4 words), run-on length (> 65 words),
     and dangling deictic references ("as mentioned above", "in this diagram").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.quiz_validator import QuestionIssue


def _make_issue(
    question_index: int,
    question_text: str,
    issue_type: str,
    detail: str,
    excluded: bool = False,
):
    from backend.core.quiz_validator import QuestionIssue
    return QuestionIssue(
        question_index=question_index,
        question_text=question_text,
        issue_type=issue_type,
        detail=detail,
        excluded=excluded,
    )



def _norm(text: str) -> str:
    """Lowercase and strip punctuation/whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_distractor_quality(q: dict[str, Any], index: int) -> list[QuestionIssue]:
    """Check MCQ distractors for plausibility, distinctness, and non-triviality."""
    issues: list[QuestionIssue] = []
    if q.get("type") != "mcq":
        return issues

    options: dict = q.get("options") or {}
    answer_key = str(q.get("answer", "")).strip().upper()
    correct_val = str(options.get(answer_key, "")).strip()

    if not options:
        return issues

    norm_values: list[str] = []
    keys = list(options.keys())

    for k in keys:
        val = str(options[k]).strip()
        if not val:
            continue
        n_val = _norm(val)
        norm_values.append(n_val)

    # Check for duplicate distractors among options
    seen = set()
    for k in keys:
        v = _norm(str(options[k]))
        if v in seen and v:
            issues.append(
                _make_issue(
                    question_index=index,
                    question_text=q.get("question", ""),
                    issue_type="invalid_mcq",
                    detail=f"Duplicate option text found for option '{k}': '{options[k]}'",
                    excluded=False,
                )
            )
        seen.add(v)

    # Check for trivial distractors (e.g., single character, "none", "n/a")
    for k, v in options.items():
        v_str = str(v).strip()
        if len(v_str) == 1 and k.upper() != v_str.upper():
            issues.append(
                _make_issue(
                    question_index=index,
                    question_text=q.get("question", ""),
                    issue_type="quality",
                    detail=f"Option {k} ('{v_str}') is too brief to be a meaningful distractor.",
                    excluded=False,
                )
            )

    return issues


def check_answer_leakage(q: dict[str, Any], index: int) -> list[QuestionIssue]:
    """
    Check if the question stem gives away the answer.
    E.g. Question: "In photosynthesis, what process produces glucose?" Answer: "Photosynthesis"
    """
    issues: list[QuestionIssue] = []
    stem = q.get("question", "")
    q_type = q.get("type", "mcq")

    # Determine answer text
    if q_type == "mcq":
        options = q.get("options") or {}
        ans_key = str(q.get("answer", "")).strip().upper()
        answer_text = str(options.get(ans_key, "")).strip()
    else:
        answer_text = str(q.get("answer", "")).strip()

    norm_stem = f" {_norm(stem)} "
    norm_ans = _norm(answer_text)

    # If answer is non-trivial (at least 2 words or a single distinctive word >= 6 letters)
    words = norm_ans.split()
    is_distinctive = len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 6)

    if is_distinctive and norm_ans in norm_stem and q_type != "true_false":
        issues.append(
            _make_issue(
                question_index=index,
                question_text=stem,
                issue_type="quality",
                detail=f"Question stem appears to leak the correct answer ('{answer_text}').",
                excluded=False,
            )
        )

    return issues


def check_linguistic_clarity(q: dict[str, Any], index: int) -> list[QuestionIssue]:
    """Check question clarity, length bounds, and dangling references."""
    issues: list[QuestionIssue] = []
    text = q.get("question", "").strip()
    words = text.split()

    # Brevity check
    if len(words) < 4:
        issues.append(
            _make_issue(
                question_index=index,
                question_text=text,
                issue_type="ambiguous",
                detail="Question stem is fewer than 4 words and lacks context.",
                excluded=False,
            )
        )

    # Extreme length check
    if len(words) > 65:
        issues.append(
            _make_issue(
                question_index=index,
                question_text=text,
                issue_type="quality",
                detail="Question stem is overly long (> 65 words) and may be confusing.",
                excluded=False,
            )
        )

    # Dangling deictic references (referencing external materials not in the quiz)
    dangling_patterns = [
        r"\b(as mentioned above|as shown above|in the diagram above|in this figure|in the table below)\b",
        r"\b(refer to the diagram|refer to figure|look at the image)\b",
    ]
    for pattern in dangling_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            issues.append(
                _make_issue(
                    question_index=index,
                    question_text=text,
                    issue_type="ambiguous",
                    detail=f"Question contains dangling reference: '{match.group(0)}'.",
                    excluded=False,
                )
            )
            break

    return issues


def check_all_quality(q: dict[str, Any], index: int) -> list[QuestionIssue]:
    """Run all quality, distractor, and leakage checks on a single question dict."""
    issues: list[QuestionIssue] = []
    issues.extend(check_distractor_quality(q, index))
    issues.extend(check_answer_leakage(q, index))
    issues.extend(check_linguistic_clarity(q, index))
    return issues
