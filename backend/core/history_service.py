"""
history_service.py — Persistent quiz history using SQLite.

Schema:
  table: quizzes
    id           TEXT PRIMARY KEY   (uuid)
    title        TEXT
    created_at   TEXT               (ISO timestamp)
    source_type  TEXT               ("pasted" | "uploaded")
    difficulty   TEXT
    question_count INTEGER
    question_types TEXT             (comma-separated)
    provider     TEXT
    questions_json TEXT             (JSON-encoded list of question dicts)
    notes_preview TEXT
    result_json  TEXT               (JSON-encoded QuizResult, nullable)
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# DB file lives next to this module's package root
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "quiz_history.db"


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


@contextmanager
def _connect():
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the quizzes table if it does not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                source_type     TEXT NOT NULL DEFAULT 'pasted',
                difficulty      TEXT NOT NULL,
                question_count  INTEGER NOT NULL,
                question_types  TEXT NOT NULL,
                provider        TEXT NOT NULL DEFAULT '',
                questions_json  TEXT NOT NULL DEFAULT '[]',
                notes_preview   TEXT NOT NULL DEFAULT '',
                result_json     TEXT
            )
        """)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_quiz(
    questions: list[dict[str, Any]],
    difficulty: str,
    question_types: list[str],
    provider: str,
    notes_preview: str,
    source_type: str = "pasted",
    title: str = "",
) -> str:
    """Persist a quiz and return its UUID."""
    init_db()
    quiz_id = str(uuid.uuid4())[:8]   # short 8-char ID for display
    if not title:
        title = f"Quiz {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO quizzes
                (id, title, created_at, source_type, difficulty,
                 question_count, question_types, provider,
                 questions_json, notes_preview, result_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
            """,
            (
                quiz_id,
                title,
                datetime.now().isoformat(timespec="seconds"),
                source_type,
                difficulty,
                len(questions),
                ", ".join(question_types),
                provider,
                json.dumps(questions, ensure_ascii=False),
                notes_preview,
            ),
        )
    return quiz_id


def save_result(quiz_id: str, result_dict: dict[str, Any]) -> None:
    """Attach a serialised QuizResult to an existing quiz row."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE quizzes SET result_json = ? WHERE id = ?",
            (json.dumps(result_dict, ensure_ascii=False, default=str), quiz_id),
        )


def list_quizzes(limit: int = 50) -> list[dict[str, Any]]:
    """Return quiz metadata rows, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, source_type, difficulty,
                   question_count, question_types, provider,
                   notes_preview,
                   CASE WHEN result_json IS NOT NULL THEN 1 ELSE 0 END AS has_result
            FROM quizzes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_quiz(quiz_id: str) -> dict[str, Any] | None:
    """Return the full quiz row (including questions_json and result_json)."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM quizzes WHERE id = ?", (quiz_id,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["questions"] = json.loads(data.get("questions_json") or "[]")
    data["result"] = json.loads(data["result_json"]) if data.get("result_json") else None
    return data


def delete_quiz(quiz_id: str) -> bool:
    """Delete a quiz by ID. Returns True if a row was removed."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
    return cursor.rowcount > 0
