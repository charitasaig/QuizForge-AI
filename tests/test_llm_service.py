"""
test_llm_service.py — Tests for backend/services/llm_service.py
All tests mock the LLM providers — no real API calls are made.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_groq_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _valid_json(n: int = 2) -> str:
    questions = [
        {
            "type": "mcq",
            "question": f"Q{i}?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer": "A",
            "explanation": "Exp.",
            "source_hint": "Hint.",
            "flagged": False,
        }
        for i in range(n)
    ]
    return json.dumps({"questions": questions})


# ---------------------------------------------------------------------------
# Mock Groq — successful call
# ---------------------------------------------------------------------------

class TestGroqProvider:
    @patch("backend.services.llm_service.get_groq_api_key", return_value="gsk_fake")
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    @patch("backend.services.llm_service._call_groq", return_value=_valid_json(3))
    def test_groq_returns_raw_text(self, mock_call, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 3, "medium", ["mcq"])
        assert provider == "groq"
        assert "questions" in raw

    @patch("backend.services.llm_service.get_groq_api_key", return_value="gsk_fake")
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    @patch("backend.services.llm_service._call_groq", side_effect=Exception("timeout"))
    def test_groq_failure_falls_back_to_demo(self, mock_call, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 2, "easy", ["mcq"])
        assert provider == "demo"
        data = json.loads(raw)
        assert len(data["questions"]) >= 1


# ---------------------------------------------------------------------------
# Mock Gemini — successful call
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    @patch("backend.services.llm_service.get_groq_api_key", return_value=None)
    @patch("backend.services.llm_service.get_gemini_api_key", return_value="AIza_fake")
    @patch("backend.services.llm_service._call_gemini", return_value=_valid_json(2))
    def test_gemini_used_when_groq_absent(self, mock_call, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 2, "medium", ["true_false"])
        assert provider == "gemini"

    @patch("backend.services.llm_service.get_groq_api_key", return_value=None)
    @patch("backend.services.llm_service.get_gemini_api_key", return_value="AIza_fake")
    @patch("backend.services.llm_service._call_gemini", side_effect=Exception("500"))
    def test_gemini_failure_falls_back_to_demo(self, mock_call, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 1, "hard", ["short_answer"])
        assert provider == "demo"


# ---------------------------------------------------------------------------
# Demo / mock mode
# ---------------------------------------------------------------------------

class TestDemoMode:
    @patch("backend.services.llm_service.get_groq_api_key", return_value=None)
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    def test_demo_mode_when_no_keys(self, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 3, "medium", ["mcq"])
        assert provider == "demo"
        data = json.loads(raw)
        assert len(data["questions"]) == 3

    @patch("backend.services.llm_service.get_groq_api_key", return_value="your_groq_api_key_here")
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    def test_placeholder_key_treated_as_missing(self, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        raw, provider = generate_quiz_raw("prompt", 2, "easy", ["true_false"])
        assert provider == "demo"

    @patch("backend.services.llm_service.get_groq_api_key", return_value=None)
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    def test_demo_respects_question_count(self, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        for n in (1, 5, 10):
            raw, _ = generate_quiz_raw("prompt", n, "easy", ["mcq"])
            data = json.loads(raw)
            assert len(data["questions"]) == n

    @patch("backend.services.llm_service.get_groq_api_key", return_value=None)
    @patch("backend.services.llm_service.get_gemini_api_key", return_value=None)
    def test_demo_all_three_types(self, mock_gem, mock_key):
        from backend.services.llm_service import generate_quiz_raw
        # Request 5 questions so all three types get slots in _mock_response
        raw, _ = generate_quiz_raw("prompt", 5, "medium", ["mcq", "true_false", "short_answer"])
        data = json.loads(raw)
        types = {q["type"] for q in data["questions"]}
        assert "mcq" in types
        assert "true_false" in types
        assert "short_answer" in types
