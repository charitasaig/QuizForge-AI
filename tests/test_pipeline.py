"""
test_pipeline.py — pytest tests for backend/core/pipeline.py
Run with: pytest tests/
"""
import json
import io
import pytest

from backend.core.pipeline import (
    prepare_notes,
    build_prompt,
    parse_quiz_response,
    extract_text_from_txt,
)


# ---------------------------------------------------------------------------
# prepare_notes
# ---------------------------------------------------------------------------

class TestPrepareNotes:
    def test_short_text_unchanged(self):
        text = "Cell biology is the study of cell structure and function."
        result = prepare_notes(text)
        assert result == text

    def test_long_text_truncated(self):
        long_text = "word " * 4000  # ~20 000 chars
        result = prepare_notes(long_text)
        from backend.config import MAX_TEXT_CHARS
        assert len(result) <= MAX_TEXT_CHARS + 200  # +200 for truncation message

    def test_excessive_newlines_collapsed(self):
        text = "Line one.\n\n\n\n\nLine two."
        result = prepare_notes(text)
        assert "\n\n\n" not in result

    def test_empty_string_stays_empty(self):
        result = prepare_notes("   ")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_notes(self):
        notes = "The mitochondria is the powerhouse of the cell."
        prompt = build_prompt(notes, 3, "easy", ["mcq"])
        assert notes in prompt

    def test_prompt_contains_num_questions(self):
        prompt = build_prompt("Some notes here about biology.", 7, "hard", ["mcq"])
        assert "7" in prompt

    def test_prompt_contains_difficulty(self):
        prompt = build_prompt("Notes on chemistry.", 5, "hard", ["true_false"])
        assert "hard" in prompt

    def test_prompt_contains_question_types(self):
        prompt = build_prompt("Notes.", 3, "medium", ["mcq", "short_answer"])
        assert "mcq" in prompt
        assert "short_answer" in prompt

    def test_prompt_requests_json_output(self):
        prompt = build_prompt("Notes.", 3, "medium", ["mcq"])
        assert "JSON" in prompt or "json" in prompt


# ---------------------------------------------------------------------------
# parse_quiz_response
# ---------------------------------------------------------------------------

_VALID_MCQ_JSON = json.dumps({
    "questions": [
        {
            "type": "mcq",
            "question": "What is photosynthesis?",
            "options": {
                "A": "Converting light to energy",
                "B": "Digesting food",
                "C": "Cellular respiration",
                "D": "Water evaporation",
            },
            "answer": "A",
            "explanation": "Plants use sunlight to produce glucose.",
            "source_hint": "Plants convert sunlight into chemical energy.",
            "flagged": False,
        }
    ]
})

_VALID_TF_JSON = json.dumps({
    "questions": [
        {
            "type": "true_false",
            "question": "Plants release oxygen during photosynthesis.",
            "answer": "True",
            "explanation": "Oxygen is a by-product of the light reactions.",
            "source_hint": "oxygen released during light reactions",
            "flagged": False,
        }
    ]
})


class TestParseQuizResponse:
    def test_valid_mcq_json_parsed(self):
        questions = parse_quiz_response(_VALID_MCQ_JSON)
        assert len(questions) == 1
        assert questions[0]["type"] == "mcq"
        assert questions[0]["answer"] == "A"

    def test_valid_tf_json_parsed(self):
        questions = parse_quiz_response(_VALID_TF_JSON)
        assert len(questions) == 1
        assert questions[0]["type"] == "true_false"

    def test_json_with_markdown_fences_parsed(self):
        fenced = "```json\n" + _VALID_MCQ_JSON + "\n```"
        questions = parse_quiz_response(fenced)
        assert len(questions) == 1

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="parsed as JSON"):
            parse_quiz_response("This is not JSON at all.")

    def test_empty_questions_list_raises_value_error(self):
        empty = json.dumps({"questions": []})
        with pytest.raises(ValueError):
            parse_quiz_response(empty)

    def test_missing_questions_key_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_quiz_response(json.dumps({"data": []}))

    def test_defaults_filled_for_missing_fields(self):
        minimal = json.dumps({"questions": [{"type": "mcq", "question": "Q?"}]})
        questions = parse_quiz_response(minimal)
        assert questions[0]["answer"] == "N/A"
        assert questions[0]["flagged"] is False
        assert "id" in questions[0]   # id field now assigned

    def test_multiple_questions_parsed(self):
        data = {
            "questions": [
                {
                    "type": "mcq",
                    "question": f"Question {i}?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "answer": "A",
                    "explanation": "Because A.",
                    "source_hint": "hint",
                    "flagged": False,
                }
                for i in range(5)
            ]
        }
        questions = parse_quiz_response(json.dumps(data))
        assert len(questions) == 5


# ---------------------------------------------------------------------------
# extract_text_from_txt
# ---------------------------------------------------------------------------

class _FakeTxtFile:
    def __init__(self, content: bytes, name="notes.txt"):
        self.name = name
        self._buf = io.BytesIO(content)

    def seek(self, *args):
        self._buf.seek(*args)

    def read(self, *args):
        return self._buf.read(*args)


class TestExtractTextFromTxt:
    def test_utf8_decoded(self):
        content = "Hello, world! This is a test note.".encode("utf-8")
        result = extract_text_from_txt(_FakeTxtFile(content))
        assert "Hello" in result

    def test_latin1_decoded(self):
        content = "Caf\xe9 au lait is a popular drink.".encode("latin-1")
        result = extract_text_from_txt(_FakeTxtFile(content))
        assert "Caf" in result

    def test_empty_file_returns_empty_string(self):
        result = extract_text_from_txt(_FakeTxtFile(b""))
        assert result == ""
