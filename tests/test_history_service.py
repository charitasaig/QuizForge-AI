"""
test_history_service.py — Tests for backend/core/history_service.py
Scenario 14 from spec: save, retrieve, delete quizzes.
Uses a temporary DB so tests never touch the real quiz_history.db.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures — redirect DB to a temp file for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Point history_service at a fresh temp DB for each test."""
    import backend.core.history_service as svc
    tmp_db_path = tmp_path / "test_history.db"
    monkeypatch.setattr(svc, "_DB_PATH", tmp_db_path)
    svc.init_db()
    yield tmp_db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_QUESTIONS = [
    {
        "id": "q1", "type": "mcq",
        "question": "What is X?",
        "options": {"A": "Correct", "B": "W1", "C": "W2", "D": "W3"},
        "answer": "A",
        "explanation": "Because A.",
        "source_hint": "Source.",
        "flagged": False,
    }
]


def _save(difficulty="medium", types=None, provider="groq", preview="Preview text"):
    from backend.core import history_service as svc
    return svc.save_quiz(
        questions=_SAMPLE_QUESTIONS,
        difficulty=difficulty,
        question_types=types or ["mcq"],
        provider=provider,
        notes_preview=preview,
        source_type="pasted",
    )


# ---------------------------------------------------------------------------
# Scenario 14: History CRUD
# ---------------------------------------------------------------------------

class TestSaveAndList:
    def test_save_returns_id(self):
        qid = _save()
        assert isinstance(qid, str)
        assert len(qid) > 0

    def test_saved_quiz_appears_in_list(self):
        from backend.core import history_service as svc
        _save()
        history = svc.list_quizzes()
        assert len(history) == 1

    def test_multiple_quizzes_listed_newest_first(self):
        from backend.core import history_service as svc
        import time
        id1 = _save(preview="First")
        time.sleep(1.1)   # ensure different timestamps (ISO seconds resolution)
        id2 = _save(preview="Second")
        history = svc.list_quizzes()
        assert len(history) == 2
        # Newest first
        assert history[0]["id"] == id2

    def test_list_respects_limit(self):
        from backend.core import history_service as svc
        for _ in range(5):
            _save()
        history = svc.list_quizzes(limit=3)
        assert len(history) == 3

    def test_list_fields_present(self):
        from backend.core import history_service as svc
        _save(difficulty="hard", types=["mcq", "true_false"], provider="gemini")
        row = svc.list_quizzes()[0]
        assert row["difficulty"] == "hard"
        assert "mcq" in row["question_types"]
        assert row["provider"] == "gemini"
        assert row["question_count"] == 1


class TestLoadQuiz:
    def test_load_returns_questions(self):
        from backend.core import history_service as svc
        qid = _save()
        data = svc.load_quiz(qid)
        assert data is not None
        assert len(data["questions"]) == 1
        assert data["questions"][0]["type"] == "mcq"

    def test_load_nonexistent_returns_none(self):
        from backend.core import history_service as svc
        result = svc.load_quiz("does-not-exist")
        assert result is None

    def test_load_result_initially_none(self):
        from backend.core import history_service as svc
        qid = _save()
        data = svc.load_quiz(qid)
        assert data["result"] is None


class TestSaveResult:
    def test_save_and_load_result(self):
        from backend.core import history_service as svc
        qid = _save()
        result_dict = {"score": 1.0, "max_score": 1.0, "percentage": 100.0,
                       "correct": 1, "incorrect": 0, "unanswered": 0, "uncertain": 0}
        svc.save_result(qid, result_dict)
        data = svc.load_quiz(qid)
        assert data["result"] is not None
        assert data["result"]["score"] == 1.0
        assert data["result"]["percentage"] == 100.0

    def test_save_result_nonexistent_quiz_silently_fails(self):
        from backend.core import history_service as svc
        # Should not raise
        svc.save_result("ghost-id", {"score": 0})

    def test_has_result_flag_in_list(self):
        from backend.core import history_service as svc
        qid = _save()
        svc.save_result(qid, {"score": 1.0, "max_score": 1.0, "percentage": 100.0,
                               "correct": 1, "incorrect": 0, "unanswered": 0, "uncertain": 0})
        row = svc.list_quizzes()[0]
        assert row["has_result"] == 1


class TestDeleteQuiz:
    def test_delete_removes_quiz(self):
        from backend.core import history_service as svc
        qid = _save()
        deleted = svc.delete_quiz(qid)
        assert deleted is True
        assert svc.load_quiz(qid) is None
        assert len(svc.list_quizzes()) == 0

    def test_delete_nonexistent_returns_false(self):
        from backend.core import history_service as svc
        deleted = svc.delete_quiz("ghost-id")
        assert deleted is False

    def test_delete_one_of_many(self):
        from backend.core import history_service as svc
        id1 = _save(preview="Keep me")
        id2 = _save(preview="Delete me")
        svc.delete_quiz(id2)
        history = svc.list_quizzes()
        assert len(history) == 1
        assert history[0]["id"] == id1
