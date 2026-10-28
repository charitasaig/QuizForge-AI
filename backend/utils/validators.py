"""
validators.py — Input and file validation for QuizForge AI.
All functions return (is_valid: bool, message: str).
"""
from __future__ import annotations

import io
from backend.config import MAX_FILE_SIZE_MB, MAX_TEXT_CHARS, ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Text validation
# ---------------------------------------------------------------------------

def validate_text_input(text: str) -> tuple[bool, str]:
    """Validate pasted text notes."""
    if not text or not text.strip():
        return False, "Notes cannot be empty. Please paste some text or upload a file."

    stripped = text.strip()

    if len(stripped) < 50:
        return False, (
            "Notes are too short (fewer than 50 characters). "
            "Please provide more content for meaningful quiz generation."
        )

    if len(stripped) > MAX_TEXT_CHARS:
        return False, (
            f"Notes are too long ({len(stripped):,} characters). "
            f"Please trim to {MAX_TEXT_CHARS:,} characters or fewer "
            "to stay within the model's context limit."
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_uploaded_file(file) -> tuple[bool, str]:
    """
    Validate a Streamlit UploadedFile object.
    Checks extension, size and that the buffer is readable.
    """
    if file is None:
        return False, "No file was uploaded."

    filename: str = file.name.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}."
        )

    # Check size
    file.seek(0, io.SEEK_END)
    size_bytes: int = file.tell()
    file.seek(0)

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes == 0:
        return False, "The uploaded file is empty. Please upload a file with content."

    if size_bytes > max_bytes:
        return False, (
            f"File is too large ({size_bytes / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# Quiz-parameter validation
# ---------------------------------------------------------------------------

def validate_quiz_params(
    num_questions: int,
    difficulty: str,
    question_types: list[str],
) -> tuple[bool, str]:
    """Validate user-selected quiz generation parameters."""
    if num_questions < 1 or num_questions > 20:
        return False, "Number of questions must be between 1 and 20."

    valid_difficulties = {"easy", "medium", "hard"}
    if difficulty.lower() not in valid_difficulties:
        return False, f"Difficulty must be one of: {', '.join(valid_difficulties)}."

    valid_types = {"mcq", "true_false", "short_answer"}
    for qt in question_types:
        if qt not in valid_types:
            return False, f"Unknown question type '{qt}'."

    if not question_types:
        return False, "Please select at least one question type."

    return True, "OK"
