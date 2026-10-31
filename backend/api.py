"""
api.py — FastAPI REST API backend for QuizForge AI.
Exposes endpoints for quiz generation, evaluation, history, and health checks.
Can be deployed as a standalone REST service on Render, Heroku, or locally.
"""
from __future__ import annotations

import os
from typing import Any, List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.utils.validators import validate_text_input, validate_quiz_params
from backend.core.pipeline import run_quiz_pipeline, get_last_agent_metrics
from backend.core.quiz_evaluator import evaluate_quiz
from backend.core import history_service
from backend.config import (
    GROQ_MODEL,
    GEMINI_MODEL,
    DEFAULT_BACKEND_URL,
)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QuizForge AI — REST Backend API",
    description="Agentic RAG-powered quiz generation and evaluation engine.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit frontend and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class GenerateQuizRequest(BaseModel):
    notes: str = Field(..., description="Raw text notes or extracted text to generate quiz from")
    num_questions: int = Field(5, ge=1, le=20, description="Total questions to generate (1-20)")
    difficulty: str = Field("medium", description="Difficulty level: 'easy', 'medium', or 'hard'")
    question_types: List[str] = Field(
        default_factory=lambda: ["mcq", "true_false"],
        description="Allowed question types: 'mcq', 'true_false', 'short_answer'",
    )


class EvaluateQuizRequest(BaseModel):
    questions: List[Dict[str, Any]] = Field(..., description="List of generated question dicts")
    user_answers: Dict[str, str] = Field(..., description="Map of question index to user answer")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": "QuizForge AI Backend API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "generate": "POST /api/generate",
            "evaluate": "POST /api/evaluate",
            "history": "GET /api/history",
        },
    }


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint for Render, monitoring, and frontend probing."""
    return {
        "status": "ok",
        "service": "QuizForge AI Backend API",
        "models": {
            "groq": GROQ_MODEL,
            "gemini": GEMINI_MODEL,
        },
        "version": "1.0.0",
    }


@app.post("/api/generate", tags=["Quiz Generation"])
def generate_quiz(req: GenerateQuizRequest) -> Dict[str, Any]:
    """
    Generate questions from notes using the Agentic RAG pipeline.
    Validates input notes, chunks and retrieves evidence, prompts LLM, and gates quality.
    """
    ok, msg = validate_text_input(req.notes)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    ok, msg = validate_quiz_params(req.num_questions, req.difficulty, req.question_types)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    try:
        questions, provider = run_quiz_pipeline(
            req.notes, req.num_questions, req.difficulty, req.question_types
        )
        metrics = get_last_agent_metrics()

        # Save to SQLite history
        notes_preview = (req.notes[:120] + "…") if len(req.notes) > 120 else req.notes
        quiz_id = history_service.save_quiz(
            questions=questions,
            difficulty=req.difficulty,
            question_types=req.question_types,
            provider=provider,
            notes_preview=notes_preview,
            source_type="api",
        )

        return {
            "quiz_id": quiz_id,
            "provider": provider,
            "questions": questions,
            "count": len(questions),
            "agent_metrics": metrics,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(exc)}",
        )


@app.post("/api/evaluate", tags=["Evaluation"])
def evaluate(req: EvaluateQuizRequest) -> Dict[str, Any]:
    """
    Evaluate user answers against the question set and calculate scores.
    """
    try:
        # Normalize integer keys from request
        user_answers = {int(k): v for k, v in req.user_answers.items() if str(k).isdigit()}
        result = evaluate_quiz(req.questions, user_answers)

        return {
            "score": result.score,
            "max_score": result.max_score,
            "total_questions": result.total_questions,
            "correct": result.correct,
            "incorrect": result.incorrect,
            "unanswered": result.unanswered,
            "uncertain": result.uncertain,
            "percentage": result.percentage,
            "detailed_results": [
                {
                    "question_index": qr.question_index,
                    "question_id": qr.question_id,
                    "question_text": qr.question_text,
                    "question_type": qr.question_type,
                    "correct_answer": qr.correct_answer,
                    "student_answer": qr.student_answer,
                    "is_correct": qr.is_correct,
                    "status": qr.status,
                    "score": qr.score,
                    "explanation": qr.explanation,
                    "source_hint": qr.source_hint,
                }
                for qr in result.per_question
            ],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(exc)}",
        )


@app.get("/api/history", tags=["History"])
def get_history(limit: int = Query(50, ge=1, le=100)) -> List[Dict[str, Any]]:
    """Retrieve history of generated quizzes."""
    return history_service.list_quizzes(limit=limit)


@app.get("/api/history/{quiz_id}", tags=["History"])
def get_quiz_by_id(quiz_id: str) -> Dict[str, Any]:
    """Retrieve a specific quiz by its ID."""
    quiz = history_service.load_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")
    return quiz


@app.delete("/api/history/{quiz_id}", tags=["History"])
def delete_quiz_by_id(quiz_id: str) -> Dict[str, Any]:
    """Delete a quiz from history by ID."""
    deleted = history_service.delete_quiz(quiz_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")
    return {"message": "Quiz deleted successfully", "quiz_id": quiz_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.api:app", host="0.0.0.0", port=port, reload=True)
