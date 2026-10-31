"""
api_client.py — Client for communicating with the QuizForge AI Backend API.
Provides automatic health detection, remote generation, evaluation, and seamless
fallback to the local processing pipeline.
"""
from __future__ import annotations

import logging
from typing import Any
import requests

from backend.config import get_backend_url

logger = logging.getLogger(__name__)


def check_backend_status(base_url: str | None = None, timeout: float = 3.0) -> dict[str, Any]:
    """
    Check the reachability and capabilities of the target backend URL.
    Detects whether the URL provides the REST API, is a Streamlit host, or is offline.
    """
    url = (base_url or get_backend_url()).rstrip("/")
    if not url:
        return {
            "online": False,
            "api_ready": False,
            "url": "",
            "mode": "Not configured",
            "message": "No backend URL configured",
        }

    # 1. Try FastAPI REST API health endpoint
    try:
        resp = requests.get(f"{url}/api/health", timeout=timeout)
        if resp.status_code == 200:
            try:
                data = resp.json()
                return {
                    "online": True,
                    "api_ready": True,
                    "url": url,
                    "mode": "REST API",
                    "details": data,
                    "message": "Connected to QuizForge REST API",
                }
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"REST API health check failed on {url}: {exc}")

    # 2. Try generic /health endpoint
    try:
        resp = requests.get(f"{url}/health", timeout=timeout)
        if resp.status_code == 200 and "application/json" in resp.headers.get("content-type", ""):
            data = resp.json()
            return {
                "online": True,
                "api_ready": True,
                "url": url,
                "mode": "REST API",
                "details": data,
                "message": "Connected to QuizForge REST API",
            }
    except Exception:
        pass

    # 3. Try Streamlit health endpoint (for Streamlit-hosted instances like Render)
    try:
        resp = requests.get(f"{url}/_stcore/health", timeout=timeout)
        if resp.status_code == 200 and "ok" in resp.text.lower():
            return {
                "online": True,
                "api_ready": False,
                "url": url,
                "mode": "Live Render Instance",
                "message": "Connected to live deployment (Local engine active for generation)",
            }
    except Exception:
        pass

    # 4. Try root ping
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code < 500:
            return {
                "online": True,
                "api_ready": False,
                "url": url,
                "mode": "Reachable Host",
                "message": f"Host reachable (HTTP {resp.status_code})",
            }
    except Exception as exc:
        logger.debug(f"Root ping failed on {url}: {exc}")

    return {
        "online": False,
        "api_ready": False,
        "url": url,
        "mode": "Offline / Sleeping",
        "message": "Backend server is sleeping or unreachable",
    }


def generate_quiz_remote(
    notes: str,
    num_questions: int,
    difficulty: str,
    question_types: list[str],
    base_url: str | None = None,
    timeout: float = 40.0,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None] | None:
    """
    Call the backend REST API to generate a quiz.
    Returns (questions, provider, agent_metrics) if successful, or None on failure.
    """
    url = (base_url or get_backend_url()).rstrip("/")
    endpoint = f"{url}/api/generate"

    payload = {
        "notes": notes,
        "num_questions": num_questions,
        "difficulty": difficulty,
        "question_types": question_types,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            questions = data.get("questions", [])
            provider = data.get("provider", "remote_api")
            metrics = data.get("agent_metrics")
            if questions:
                return questions, provider, metrics
        logger.warning(f"Remote generation responded with status {resp.status_code}")
    except Exception as exc:
        logger.warning(f"Remote generation call failed: {exc}")

    return None


def evaluate_quiz_remote(
    questions: list[dict[str, Any]],
    user_answers: dict[str, str],
    base_url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """
    Call the backend REST API to evaluate quiz answers.
    Returns evaluation result dict if successful, or None on failure.
    """
    url = (base_url or get_backend_url()).rstrip("/")
    endpoint = f"{url}/api/evaluate"

    payload = {
        "questions": questions,
        "user_answers": user_answers,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning(f"Remote evaluation call failed: {exc}")

    return None
