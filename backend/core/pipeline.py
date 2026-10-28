"""
pipeline.py — Core quiz generation pipeline for QuizForge AI.

Responsibilities:
  1. Extract text from TXT / PDF uploads.
  2. Truncate / chunk text to fit context window.
  3. Build the LLM prompt.
  4. Parse and validate the JSON response.
  5. Run quality validation.
  6. Return structured quiz data.
"""
from __future__ import annotations

import io
import json
import re
import textwrap
from typing import Any

from backend.config import MAX_TEXT_CHARS
from backend.services.llm_service import generate_quiz_raw


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_txt(file) -> str:
    """Decode a TXT UploadedFile to a plain string."""
    file.seek(0)
    raw_bytes = file.read()
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF UploadedFile using pypdf.
    Returns an empty string and raises ValueError for scanned/image PDFs.
    """
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError("pypdf is not installed. Run: pip install pypdf")

    file.seek(0)
    reader = pypdf.PdfReader(io.BytesIO(file.read()))
    pages_text: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "Could not extract text from this PDF. "
            "It may be a scanned image. Please use a text-based PDF or paste the content manually."
        )

    return full_text


def extract_text(file) -> str:
    """Dispatch to the right extractor based on file extension."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    return extract_text_from_txt(file)


# ---------------------------------------------------------------------------
# 2. Text preparation
# ---------------------------------------------------------------------------

def prepare_notes(raw_text: str) -> str:
    """Clean and truncate notes to fit the context window."""
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        # Try to cut at a sentence boundary
        last_period = text.rfind(". ")
        if last_period > MAX_TEXT_CHARS * 0.8:
            text = text[: last_period + 1]
        text += "\n\n[… content truncated to fit model context limit …]"

    return text


# ---------------------------------------------------------------------------
# 3. Prompt builder
# ---------------------------------------------------------------------------

_TYPE_INSTRUCTIONS = {
    "mcq": textwrap.dedent("""\
        - MCQ: provide "question", "options" (object with keys A–D), "answer" (the correct key letter), "explanation", "source_hint".
    """),
    "true_false": textwrap.dedent("""\
        - True/False: provide "question", "answer" ("True" or "False"), "explanation", "source_hint".
    """),
    "short_answer": textwrap.dedent("""\
        - Short Answer: provide "question", "answer" (1–2 sentences), "explanation", "source_hint".
    """),
}

def build_prompt(
    notes: str,
    num_questions: int,
    difficulty: str,
    question_types: list[str],
) -> str:
    type_instructions = "".join(_TYPE_INSTRUCTIONS[t] for t in question_types)
    types_str = ", ".join(question_types)

    prompt = textwrap.dedent(f"""\
        You are QuizForge AI, an expert educational assistant.
        Your task is to generate a quiz based ONLY on the notes provided below.

        ## Instructions
        - Generate exactly {num_questions} questions.
        - Question types to include (distribute evenly): {types_str}.
        - Difficulty level: {difficulty}.
        - Every question MUST be answerable from the notes. Do NOT invent facts.
        - For each question include a "source_hint": a short phrase (≤ 15 words) quoting or paraphrasing the part of the notes that supports the answer.
        - Flag any question you consider vague or weakly supported by adding "flagged": true.

        ## Question format rules
        {type_instructions}
        - Set "type" to exactly one of: "mcq", "true_false", "short_answer".

        ## Output format
        Return ONLY valid JSON — no markdown fences, no commentary — matching this schema:
        {{
          "questions": [
            {{
              "type": "...",
              "question": "...",
              "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},  // MCQ only
              "answer": "...",
              "explanation": "...",
              "source_hint": "...",
              "flagged": false   // set true if vague or unsupported
            }}
          ]
        }}

        ## Notes
        {notes}
    """)
    return prompt


# ---------------------------------------------------------------------------
# 4. JSON parser
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown code fences or prose wrapping if the model added them."""
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first { … } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def parse_quiz_response(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse the LLM JSON response into a list of question dicts.
    Raises ValueError with a human-readable message on failure.
    """
    try:
        clean = _extract_json(raw_text)
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The AI returned a response that could not be parsed as JSON. "
            f"Detail: {exc}\n\nRaw response (first 500 chars):\n{raw_text[:500]}"
        )

    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(
            "The AI response did not contain a 'questions' list. "
            "Try again — the model may have been overloaded."
        )

    # Basic per-question normalisation
    normalised: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        q.setdefault("id", f"q{i+1}")
        q.setdefault("type", "mcq")
        q.setdefault("question", f"Question {i+1}")
        q.setdefault("answer", "N/A")
        q.setdefault("explanation", "No explanation provided.")
        q.setdefault("source_hint", "")
        q.setdefault("flagged", False)
        normalised.append(q)

    return normalised


# ---------------------------------------------------------------------------
# 5. Public pipeline entry point
# ---------------------------------------------------------------------------

def run_quiz_pipeline(
    notes: str,
    num_questions: int,
    difficulty: str,
    question_types: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """
    Full pipeline: prepare notes → build prompt → call LLM → parse → validate.

    Returns:
        (questions, provider_used)
    Raises:
        ValueError on parse failure (caller should display to user).
    """
    from backend.core.quiz_validator import validate_questions

    prepared = prepare_notes(notes)
    prompt = build_prompt(prepared, num_questions, difficulty, question_types)
    raw, provider = generate_quiz_raw(prompt, num_questions, difficulty, question_types)
    questions = parse_quiz_response(raw)

    # Run quality validation — returns only structurally valid, non-duplicate questions
    report = validate_questions(questions)
    valid_qs = report.valid_questions if report.valid_questions else questions

    return valid_qs, provider
