"""
test_scenarios.py — Scenario tests 1–3 from spec.
Uses mocked LLM responses to test full pipeline without API calls.

Scenario 1: Computer Science notes (MCQ + T/F + short-answer)
Scenario 2: Biology notes
Scenario 3: History notes
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Shared mock response builder
# ---------------------------------------------------------------------------

def _mock_llm_response(questions: list[dict]) -> str:
    return json.dumps({"questions": questions})


def _cs_response():
    return _mock_llm_response([
        {
            "type": "mcq",
            "question": "What does a compiler do?",
            "options": {
                "A": "Translates source code to machine code",
                "B": "Manages memory allocation",
                "C": "Handles network requests",
                "D": "Renders graphics",
            },
            "answer": "A",
            "explanation": "A compiler translates source code into machine code.",
            "source_hint": "A compiler translates source code into a target representation.",
            "flagged": False,
        },
        {
            "type": "true_false",
            "question": "An interpreter executes source code line by line.",
            "answer": "True",
            "explanation": "Interpreters process one statement at a time.",
            "source_hint": "Interpreters execute code line by line.",
            "flagged": False,
        },
        {
            "type": "short_answer",
            "question": "What is the difference between a compiler and an interpreter?",
            "answer": "A compiler translates the entire program before execution; an interpreter executes line by line.",
            "explanation": "Both translate code, but at different times relative to execution.",
            "source_hint": "Compilers translate beforehand; interpreters translate during execution.",
            "flagged": False,
        },
    ])


def _biology_response():
    return _mock_llm_response([
        {
            "type": "mcq",
            "question": "Which pigment absorbs sunlight in photosynthesis?",
            "options": {
                "A": "Chlorophyll",
                "B": "Melanin",
                "C": "Haemoglobin",
                "D": "Carotene",
            },
            "answer": "A",
            "explanation": "Chlorophyll is the primary pigment for light absorption.",
            "source_hint": "Chlorophyll absorbs sunlight to power photosynthesis.",
            "flagged": False,
        },
        {
            "type": "true_false",
            "question": "Oxygen is released as a by-product of photosynthesis.",
            "answer": "True",
            "explanation": "Oxygen is released when water is split during the light reactions.",
            "source_hint": "Oxygen released during photolysis of water.",
            "flagged": False,
        },
    ])


def _history_response():
    return _mock_llm_response([
        {
            "type": "mcq",
            "question": "What event triggered Britain's declaration of war against Germany in 1939?",
            "options": {
                "A": "Invasion of Poland",
                "B": "Annexation of Austria",
                "C": "Battle of Britain",
                "D": "Operation Barbarossa",
            },
            "answer": "A",
            "explanation": "Germany invaded Poland on 1 September 1939.",
            "source_hint": "invasion of Poland 1 September 1939",
            "flagged": False,
        },
        {
            "type": "true_false",
            "question": "The Treaty of Versailles imposed war reparations on Germany.",
            "answer": "True",
            "explanation": "The treaty required Germany to pay £6.6 billion in reparations.",
            "source_hint": "reparations of £6.6 billion",
            "flagged": False,
        },
        {
            "type": "short_answer",
            "question": "What was the policy of appeasement?",
            "answer": "Britain and France conceded to Hitler's demands to avoid war.",
            "explanation": "Appeasement aimed to prevent conflict by satisfying demands.",
            "source_hint": "hoped to avoid war by conceding to Hitler's demands",
            "flagged": False,
        },
    ])


# ---------------------------------------------------------------------------
# Scenario 1: Computer Science notes
# ---------------------------------------------------------------------------

class TestScenario1ComputerScience:
    _CS_NOTES = (
        "A compiler is a program that translates source code written in a high-level "
        "programming language into machine code or another low-level representation. "
        "An interpreter executes source code directly, processing one statement at a time. "
        "Python is an interpreted language; C++ is typically compiled. "
        "Programming languages can be classified as compiled or interpreted."
    )

    def test_cs_pipeline_produces_three_types(self):
        """Scenario 1 — verify structural validity across all three question types."""
        from backend.core.pipeline import run_quiz_pipeline
        with patch(
            "backend.services.llm_service.generate_quiz_raw",
            return_value=(_cs_response(), "demo"),
        ):
            questions, provider = run_quiz_pipeline(
                self._CS_NOTES, 3, "medium", ["mcq", "true_false", "short_answer"]
            )

        assert len(questions) == 3
        q_types = {q["type"] for q in questions}
        assert "mcq" in q_types
        assert "true_false" in q_types
        assert "short_answer" in q_types

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_cs_response(), "demo"))
    def test_cs_mcq_has_four_options(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._CS_NOTES, 3, "medium", ["mcq", "true_false", "short_answer"]
        )
        mcqs = [q for q in questions if q["type"] == "mcq"]
        for q in mcqs:
            assert len(q.get("options", {})) == 4

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_cs_response(), "demo"))
    def test_cs_all_questions_have_answer(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._CS_NOTES, 3, "medium", ["mcq", "true_false", "short_answer"]
        )
        for q in questions:
            assert q.get("answer") not in (None, "", "N/A")

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_cs_response(), "demo"))
    def test_cs_all_questions_have_explanation(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._CS_NOTES, 3, "medium", ["mcq", "true_false", "short_answer"]
        )
        for q in questions:
            assert q.get("explanation", "").strip() != ""

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_cs_response(), "demo"))
    def test_cs_all_questions_have_source_hint(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._CS_NOTES, 3, "medium", ["mcq", "true_false", "short_answer"]
        )
        for q in questions:
            assert q.get("source_hint", "").strip() != ""


# ---------------------------------------------------------------------------
# Scenario 2: Biology notes
# ---------------------------------------------------------------------------

class TestScenario2Biology:
    _BIO_NOTES = (
        "Photosynthesis is the process by which green plants use chlorophyll to absorb "
        "sunlight and convert carbon dioxide and water into glucose and oxygen. "
        "The overall equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. "
        "The light reactions occur in the thylakoid membranes. "
        "The Calvin cycle occurs in the stroma and fixes CO2 into glucose. "
        "Plant respiration releases energy from glucose through aerobic respiration."
    )

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_biology_response(), "groq"))
    def test_biology_questions_generated(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, provider = run_quiz_pipeline(
            self._BIO_NOTES, 2, "easy", ["mcq", "true_false"]
        )
        assert len(questions) == 2
        assert provider == "groq"

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_biology_response(), "groq"))
    def test_biology_mcq_correct_answer_in_options(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(self._BIO_NOTES, 2, "easy", ["mcq", "true_false"])
        for q in questions:
            if q["type"] == "mcq":
                assert q["answer"] in q.get("options", {})

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_biology_response(), "groq"))
    def test_biology_tf_answer_is_true_or_false(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(self._BIO_NOTES, 2, "easy", ["mcq", "true_false"])
        for q in questions:
            if q["type"] == "true_false":
                assert q["answer"] in ("True", "False")


# ---------------------------------------------------------------------------
# Scenario 3: History notes
# ---------------------------------------------------------------------------

class TestScenario3History:
    _HIST_NOTES = (
        "World War II began when Germany invaded Poland on 1 September 1939. "
        "Britain and France declared war on Germany on 3 September 1939. "
        "The Treaty of Versailles imposed harsh penalties on Germany including "
        "reparations of 6.6 billion pounds and the war guilt clause. "
        "The policy of appeasement, represented by the Munich Agreement of 1938, "
        "allowed Germany to annex the Sudetenland. "
        "D-Day, on 6 June 1944, saw Allied forces land in Normandy. "
        "VE Day on 8 May 1945 marked the end of the war in Europe."
    )

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_history_response(), "groq"))
    def test_history_three_question_types(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._HIST_NOTES, 3, "hard", ["mcq", "true_false", "short_answer"]
        )
        assert len(questions) == 3

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_history_response(), "groq"))
    def test_history_questions_have_ids(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._HIST_NOTES, 3, "hard", ["mcq", "true_false", "short_answer"]
        )
        for q in questions:
            assert "id" in q
            assert q["id"].strip() != ""

    @patch("backend.services.llm_service.generate_quiz_raw",
           return_value=(_history_response(), "groq"))
    def test_history_no_none_fields(self, mock_gen):
        from backend.core.pipeline import run_quiz_pipeline
        questions, _ = run_quiz_pipeline(
            self._HIST_NOTES, 3, "hard", ["mcq", "true_false", "short_answer"]
        )
        for q in questions:
            for key in ("question", "answer", "explanation"):
                assert q.get(key) is not None
