"""
test_validators.py — pytest tests for backend/utils/validators.py
Run with: pytest tests/
"""
import io
import pytest
from backend.utils.validators import (
    validate_text_input,
    validate_uploaded_file,
    validate_quiz_params,
)


# ---------------------------------------------------------------------------
# validate_text_input
# ---------------------------------------------------------------------------

class TestValidateTextInput:
    def test_empty_string_fails(self):
        ok, msg = validate_text_input("")
        assert not ok
        assert "empty" in msg.lower()

    def test_whitespace_only_fails(self):
        ok, msg = validate_text_input("   \n\t  ")
        assert not ok

    def test_too_short_fails(self):
        ok, msg = validate_text_input("Hi")
        assert not ok
        assert "short" in msg.lower()

    def test_valid_text_passes(self):
        notes = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesise nutrients."
        ok, msg = validate_text_input(notes)
        assert ok
        assert msg == "OK"

    def test_too_long_fails(self):
        long_text = "a" * 16_000
        ok, msg = validate_text_input(long_text)
        assert not ok
        assert "long" in msg.lower()

    def test_exactly_max_passes(self):
        from backend.config import MAX_TEXT_CHARS
        text = "word " * (MAX_TEXT_CHARS // 5)
        ok, _ = validate_text_input(text[:MAX_TEXT_CHARS])
        assert ok


# ---------------------------------------------------------------------------
# validate_uploaded_file
# ---------------------------------------------------------------------------

class _FakeFile:
    """Minimal mock of a Streamlit UploadedFile."""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._buf = io.BytesIO(content)

    def seek(self, *args):
        self._buf.seek(*args)

    def tell(self):
        return self._buf.tell()

    def read(self, *args):
        return self._buf.read(*args)


class TestValidateUploadedFile:
    def test_none_fails(self):
        ok, msg = validate_uploaded_file(None)
        assert not ok

    def test_unsupported_extension_fails(self):
        f = _FakeFile("notes.docx", b"some content here")
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "unsupported" in msg.lower()

    def test_empty_file_fails(self):
        f = _FakeFile("notes.txt", b"")
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "empty" in msg.lower()

    def test_valid_txt_passes(self):
        f = _FakeFile("notes.txt", b"These are my class notes about biology.")
        ok, msg = validate_uploaded_file(f)
        assert ok

    def test_valid_pdf_extension_passes(self):
        # We just test the extension + size logic; actual PDF parsing is tested elsewhere
        f = _FakeFile("lecture.pdf", b"%PDF-1.4 fake content here for size check")
        ok, msg = validate_uploaded_file(f)
        assert ok

    def test_oversized_file_fails(self):
        big_content = b"x" * (6 * 1024 * 1024)  # 6 MB
        f = _FakeFile("big.txt", big_content)
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "large" in msg.lower()


# ---------------------------------------------------------------------------
# validate_quiz_params
# ---------------------------------------------------------------------------

class TestValidateQuizParams:
    def test_valid_params_pass(self):
        ok, msg = validate_quiz_params(5, "medium", ["mcq", "true_false"])
        assert ok

    def test_zero_questions_fails(self):
        ok, msg = validate_quiz_params(0, "easy", ["mcq"])
        assert not ok

    def test_too_many_questions_fails(self):
        ok, msg = validate_quiz_params(21, "hard", ["mcq"])
        assert not ok

    def test_invalid_difficulty_fails(self):
        ok, msg = validate_quiz_params(5, "extreme", ["mcq"])
        assert not ok
        assert "difficulty" in msg.lower()

    def test_empty_types_fails(self):
        ok, msg = validate_quiz_params(5, "easy", [])
        assert not ok

    def test_invalid_type_fails(self):
        ok, msg = validate_quiz_params(5, "easy", ["essay"])
        assert not ok
