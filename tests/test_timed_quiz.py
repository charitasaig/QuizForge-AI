"""
test_timed_quiz.py — Unit tests for Timed Quiz Mode, Autorefresh, and Answer Preservation.
"""
from __future__ import annotations

import datetime
from pathlib import Path
import pytest

from backend.core.quiz_evaluator import evaluate_quiz


class TestAutorefreshDependency:
    def test_streamlit_autorefresh_is_installed_and_importable(self):
        """Verify streamlit_autorefresh is installed and can be imported."""
        from streamlit_autorefresh import st_autorefresh
        assert callable(st_autorefresh)

    def test_requirements_includes_streamlit_autorefresh(self):
        """Verify requirements.txt explicitly tracks streamlit-autorefresh."""
        req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
        assert req_path.exists()
        content = req_path.read_text(encoding="utf-8")
        assert "streamlit-autorefresh" in content


class TestTimerCalculationsAndDuration:
    def test_timer_deadline_is_absolute_and_decreases_with_time(self):
        """Simulate timer ticks over wall-clock time and ensure deadline is never reset."""
        duration_minutes = 5
        start_time = datetime.datetime(2026, 10, 28, 12, 0, 0)
        deadline = start_time + datetime.timedelta(minutes=duration_minutes)

        # Initial tick at t=0
        remaining_t0 = (deadline - start_time).total_seconds()
        assert remaining_t0 == 300.0  # 5 * 60

        # Tick at t=1s (simulating first 1000ms autorefresh)
        tick_1s = start_time + datetime.timedelta(seconds=1)
        remaining_t1 = (deadline - tick_1s).total_seconds()
        mins, secs = divmod(int(remaining_t1), 60)
        assert remaining_t1 == 299.0
        assert f"{mins:02d}:{secs:02d}" == "04:59"

        # Tick at t=15s
        tick_15s = start_time + datetime.timedelta(seconds=15)
        remaining_t15 = (deadline - tick_15s).total_seconds()
        mins, secs = divmod(int(remaining_t15), 60)
        assert remaining_t15 == 285.0
        assert f"{mins:02d}:{secs:02d}" == "04:45"

        # Tick at t=299s
        tick_299s = start_time + datetime.timedelta(seconds=299)
        remaining_t299 = (deadline - tick_299s).total_seconds()
        mins, secs = divmod(int(remaining_t299), 60)
        assert remaining_t299 == 1.0
        assert f"{mins:02d}:{secs:02d}" == "00:01"

        # Deadline itself is immutable across reruns
        assert deadline == datetime.datetime(2026, 10, 28, 12, 5, 0)

    def test_timer_expiration_detected_at_deadline(self):
        """Verify remaining time reaches zero or negative when time expires."""
        start_time = datetime.datetime(2026, 10, 28, 12, 0, 0)
        deadline = start_time + datetime.timedelta(minutes=5)

        current_time = start_time + datetime.timedelta(minutes=5, seconds=1)
        remaining = max(0.0, (deadline - current_time).total_seconds())
        assert remaining == 0.0

    def test_quiz_start_time_preserved_for_accurate_time_taken(self):
        """Verify start_time calculation preserves elapsed time for scoring."""
        quiz_start = datetime.datetime.now() - datetime.timedelta(seconds=125)
        questions = [
            {
                "id": "q1",
                "type": "mcq",
                "question": "What is mitochondria?",
                "options": {"A": "Powerhouse", "B": "Control center", "C": "Storage", "D": "Wall"},
                "answer": "A",
                "explanation": "It produces ATP",
            }
        ]
        result = evaluate_quiz(questions, {0: "A"}, start_time=quiz_start)
        assert result.time_taken_seconds is not None
        assert 120 <= result.time_taken_seconds <= 135


class TestUserAnswersPreservationAcrossReruns:
    def test_mcq_index_mapping_preserves_selection(self):
        """Simulate how app.py maps saved answers to widget choices across reruns."""
        choices = ["A. Mitochondria", "B. Nucleus", "C. Ribosome", "D. Chloroplast"]

        def get_default_idx(prev_ans: str) -> int | None:
            if not prev_ans:
                return None
            for idx, choice in enumerate(choices):
                if choice.startswith(f"{prev_ans}."):
                    return idx
            return None

        # Unanswered initially
        assert get_default_idx("") is None

        # User selected 'B'
        assert get_default_idx("B") == 1

        # User switched to 'D'
        assert get_default_idx("D") == 3

    def test_true_false_index_mapping_preserves_selection(self):
        """Simulate how app.py maps True/False answers to widget choices across reruns."""
        def get_tf_idx(prev_ans: str) -> int | None:
            if prev_ans in ("True", "False"):
                return 0 if prev_ans == "True" else 1
            return None

        assert get_tf_idx("") is None
        assert get_tf_idx("True") == 0
        assert get_tf_idx("False") == 1

    def test_auto_submit_preserves_answered_questions_at_timeout(self):
        """Verify that when auto-submit triggers at timeout, answers entered so far are scored correctly."""
        questions = [
            {
                "id": "q1",
                "type": "mcq",
                "question": "Question 1",
                "options": {"A": "Correct", "B": "Wrong 1", "C": "Wrong 2", "D": "Wrong 3"},
                "answer": "A",
                "explanation": "Exp 1",
            },
            {
                "id": "q2",
                "type": "true_false",
                "question": "Question 2",
                "answer": "False",
                "explanation": "Exp 2",
            },
            {
                "id": "q3",
                "type": "short_answer",
                "question": "Question 3",
                "answer": "Photosynthesis is the process of converting light into chemical energy",
                "explanation": "Exp 3",
            },
        ]

        # Student answered Q1 ('A') and Q2 ('False'), left Q3 blank before time expired
        user_answers = {0: "A", 1: "False", 2: ""}

        start_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
        result = evaluate_quiz(questions, user_answers, start_time=start_time)

        assert result.total_questions == 3
        assert result.correct == 2
        assert result.unanswered == 1
        assert result.incorrect == 0
        assert result.score == 2.0
        assert result.time_taken_seconds is not None
