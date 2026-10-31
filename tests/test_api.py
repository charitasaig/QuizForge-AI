"""
test_api.py — Unit tests for QuizForge AI REST API (backend/api.py).
Tests endpoint handlers directly for fast, isolated execution.
"""
from unittest.mock import patch
import pytest
from fastapi import HTTPException

from backend.api import (
    root,
    health_check,
    generate_quiz,
    evaluate,
    get_history,
    get_quiz_by_id,
    delete_quiz_by_id,
    GenerateQuizRequest,
    EvaluateQuizRequest,
)


def test_root_handler():
    data = root()
    assert data["status"] == "online"
    assert "endpoints" in data
    assert data["endpoints"]["generate"] == "POST /api/generate"


def test_health_handler():
    data = health_check()
    assert data["status"] == "ok"
    assert "models" in data
    assert "groq" in data["models"]


def test_generate_handler_empty_notes():
    req = GenerateQuizRequest(
        notes="    ",
        num_questions=5,
        difficulty="medium",
        question_types=["mcq"],
    )
    with pytest.raises(HTTPException) as exc_info:
        generate_quiz(req)
    assert exc_info.value.status_code == 400


def test_generate_handler_success():
    sample_notes = (
        "Mitochondria are membrane-bound cell organelles that generate most of the "
        "chemical energy needed to power the biochemical reactions of the cell. "
        "Chemical energy produced by the mitochondria is stored in ATP molecules. "
        "Mitochondria contain their own small chromosomes generally inherited only from mother. "
        "They are commonly referred to as the powerhouse of the cell."
    )

    mock_questions = [
        {
            "id": "q1",
            "type": "mcq",
            "question": "What is the powerhouse of the cell?",
            "options": ["A) Mitochondria", "B) Ribosome", "C) Nucleus", "D) Cell wall"],
            "answer": "A",
            "explanation": "Mitochondria produce ATP.",
            "source_evidence": "powerhouse of the cell.",
        }
    ]

    req = GenerateQuizRequest(
        notes=sample_notes,
        num_questions=1,
        difficulty="easy",
        question_types=["mcq"],
    )

    with patch("backend.api.run_quiz_pipeline", return_value=(mock_questions, "mock_llm")):
        resp = generate_quiz(req)
        assert "questions" in resp
        assert len(resp["questions"]) == 1
        assert resp["provider"] == "mock_llm"
        assert "quiz_id" in resp


def test_evaluate_handler():
    sample_questions = [
        {
            "id": "q1",
            "type": "mcq",
            "question": "What is the powerhouse of the cell?",
            "options": ["A) Mitochondria", "B) Nucleus", "C) Ribosome", "D) Golgi"],
            "answer": "A",
            "explanation": "Mitochondria produce ATP.",
            "source_evidence": "Powerhouse of the cell.",
        }
    ]

    req = EvaluateQuizRequest(
        questions=sample_questions,
        user_answers={"0": "A"},
    )
    result = evaluate(req)
    assert result["score"] == 1
    assert result["total_questions"] == 1
    assert result["percentage"] == 100.0


def test_history_handlers():
    history = get_history(limit=10)
    assert isinstance(history, list)
