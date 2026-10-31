"""
test_api_client.py — Unit tests for QuizForge AI api_client.py.
"""
from unittest.mock import patch, MagicMock
import pytest

from backend.services.api_client import (
    check_backend_status,
    generate_quiz_remote,
    evaluate_quiz_remote,
)


def test_check_backend_status_rest_api():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "service": "QuizForge AI Backend API"}

    with patch("requests.get", return_value=mock_resp):
        res = check_backend_status("https://quizforge-ai-me4t.onrender.com")
        assert res["online"] is True
        assert res["api_ready"] is True
        assert "REST API" in res["mode"]


def test_check_backend_status_streamlit_deployment():
    def mock_get(url, **kwargs):
        resp = MagicMock()
        if "_stcore/health" in url:
            resp.status_code = 200
            resp.text = "ok"
            resp.headers = {"content-type": "text/plain"}
            return resp
        elif "/api/health" in url or "/health" in url:
            resp.status_code = 404
            return resp
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        return resp

    with patch("requests.get", side_effect=mock_get):
        res = check_backend_status("https://quizforge-ai-me4t.onrender.com")
        assert res["online"] is True
        assert res["api_ready"] is False
        assert "Live Render Instance" in res["mode"]


def test_check_backend_status_offline():
    with patch("requests.get", side_effect=Exception("Connection refused")):
        res = check_backend_status("https://offline-backend.onrender.com")
        assert res["online"] is False
        assert res["api_ready"] is False


def test_generate_quiz_remote_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "questions": [{"id": "q1", "question": "What is ATP?"}],
        "provider": "remote_groq",
        "agent_metrics": {"total_accepted": 1},
    }

    with patch("requests.post", return_value=mock_resp):
        result = generate_quiz_remote(
            notes="Cell energy is stored in ATP.",
            num_questions=1,
            difficulty="easy",
            question_types=["mcq"],
            base_url="https://quizforge-ai-me4t.onrender.com",
        )
        assert result is not None
        questions, provider, metrics = result
        assert len(questions) == 1
        assert provider == "remote_groq"


def test_generate_quiz_remote_failure():
    with patch("requests.post", side_effect=Exception("Timeout")):
        result = generate_quiz_remote(
            notes="Sample notes",
            num_questions=1,
            difficulty="easy",
            question_types=["mcq"],
            base_url="https://quizforge-ai-me4t.onrender.com",
        )
        assert result is None
