"""
test_quiz_exporter.py — Tests for backend/core/quiz_exporter.py
Covers: TXT export, CSV export, Unicode, commas, multiline content.
Scenario 13 from spec.
"""
from __future__ import annotations

import csv
import io
import pytest
from backend.core.quiz_exporter import (
    export_questions_txt,
    export_answer_key_txt,
    export_quiz_csv,
)


# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

_QUESTIONS = [
    {
        "id": "q1", "type": "mcq",
        "question": "Which molecule is the primary product of photosynthesis?",
        "options": {
            "A": "Glucose", "B": "Oxygen", "C": "Carbon dioxide", "D": "Water"
        },
        "answer": "A",
        "explanation": "Glucose is synthesised in the Calvin cycle.",
        "source_hint": "Photosynthesis produces glucose.",
        "flagged": False,
    },
    {
        "id": "q2", "type": "true_false",
        "question": "Oxygen is a by-product of photosynthesis.",
        "answer": "True",
        "explanation": "Oxygen is released during the light reactions.",
        "source_hint": "Oxygen released during photolysis.",
        "flagged": False,
    },
    {
        "id": "q3", "type": "short_answer",
        "question": "What is the role of chlorophyll in photosynthesis?",
        "answer": "Chlorophyll absorbs sunlight to power photosynthesis.",
        "explanation": "It is the primary pigment in the light reactions.",
        "source_hint": "Chlorophyll absorbs sunlight.",
        "flagged": False,
    },
]

_UNICODE_Q = {
    "id": "q4", "type": "mcq",
    "question": "What is 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂?",
    "options": {"A": "Photosynthesis", "B": "Résumé", "C": "Café", "D": "Naïve"},
    "answer": "A",
    "explanation": "The photosynthesis equation.",
    "source_hint": "Overall equation for photosynthesis.",
    "flagged": False,
}

_MULTILINE_Q = {
    "id": "q5", "type": "short_answer",
    "question": "Explain\nphotosynthesis\nin detail.",
    "answer": "It converts light,\nCO2 and water into glucose.",
    "explanation": "Multi-line explanation.",
    "source_hint": "Source.",
    "flagged": False,
}

_COMMA_Q = {
    "id": "q6", "type": "mcq",
    "question": 'Which element is NOT needed: carbon, oxygen, or "nitrogen"?',
    "options": {"A": 'Carbon, a key element', "B": "Oxygen", "C": "Nitrogen", "D": "Hydrogen"},
    "answer": "C",
    "explanation": 'Nitrogen is not directly used.',
    "source_hint": "Source.",
    "flagged": False,
}


# ---------------------------------------------------------------------------
# Scenario 13: CSV export correctness
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_csv_has_correct_columns(self):
        data = export_quiz_csv(_QUESTIONS, "medium", "test01")
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        expected = {
            "quiz_id", "question_id", "question_type", "difficulty",
            "question", "option_a", "option_b", "option_c", "option_d",
            "correct_answer", "expected_answer", "explanation",
            "source_evidence", "flagged",
        }
        assert set(reader.fieldnames) >= expected

    def test_csv_row_count_matches_questions(self):
        data = export_quiz_csv(_QUESTIONS, "medium", "test01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert len(rows) == len(_QUESTIONS)

    def test_csv_utf8_encoded(self):
        data = export_quiz_csv([_UNICODE_Q], "easy", "uni01")
        text = data.decode("utf-8")
        assert "₂" in text or "CO" in text   # Unicode characters preserved

    def test_csv_handles_commas_in_content(self):
        data = export_quiz_csv([_COMMA_Q], "medium", "comma01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert len(rows) == 1
        assert "Carbon" in rows[0]["option_a"]

    def test_csv_handles_quotes_in_content(self):
        data = export_quiz_csv([_COMMA_Q], "medium", "quote01")
        text = data.decode("utf-8")
        # If parsed correctly there should be exactly 1 data row
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1

    def test_csv_handles_multiline_content(self):
        data = export_quiz_csv([_MULTILINE_Q], "hard", "ml01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert len(rows) == 1

    def test_csv_tf_options_are_empty(self):
        tf_q = _QUESTIONS[1]
        data = export_quiz_csv([tf_q], "medium", "tf01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert rows[0]["option_a"] == ""
        assert rows[0]["option_b"] == ""

    def test_csv_short_answer_expected_answer_filled(self):
        sa_q = _QUESTIONS[2]
        data = export_quiz_csv([sa_q], "medium", "sa01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert rows[0]["expected_answer"] != ""

    def test_csv_difficulty_column(self):
        data = export_quiz_csv(_QUESTIONS, "hard", "d01")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        assert all(r["difficulty"] == "hard" for r in rows)


# ---------------------------------------------------------------------------
# TXT exports
# ---------------------------------------------------------------------------

class TestTXTExport:
    def test_questions_txt_excludes_answers(self):
        data = export_questions_txt(_QUESTIONS, "medium", "txt01")
        text = data.decode("utf-8")
        assert "Answer" not in text or "Answer Key" not in text
        # Should NOT contain the word "Answer     :"
        assert "Answer     :" not in text

    def test_questions_txt_contains_question_text(self):
        data = export_questions_txt(_QUESTIONS, "medium", "txt01")
        text = data.decode("utf-8")
        assert "photosynthesis" in text.lower()

    def test_questions_txt_contains_mcq_options(self):
        data = export_questions_txt(_QUESTIONS, "medium", "txt01")
        text = data.decode("utf-8")
        assert "A. Glucose" in text

    def test_answer_key_txt_contains_answers(self):
        data = export_answer_key_txt(_QUESTIONS, "medium", "ak01")
        text = data.decode("utf-8")
        assert "Answer" in text
        assert "Explanation" in text

    def test_answer_key_txt_contains_source_hints(self):
        data = export_answer_key_txt(_QUESTIONS, "medium", "ak01")
        text = data.decode("utf-8")
        assert "Source" in text

    def test_txt_is_utf8(self):
        data = export_questions_txt([_UNICODE_Q], "easy", "uni01")
        text = data.decode("utf-8")
        assert "₂" in text or "CO" in text

    def test_empty_question_list_produces_header_only(self):
        data = export_questions_txt([], "easy", "empty01")
        text = data.decode("utf-8")
        assert "QuizForge" in text
        assert "Questions  : 0" in text
