"""
llm_service.py — LLM provider calls with timeout and error handling.

Priority order:
  1. Groq (openai/gpt-oss-20b)   — fast, free tier (Oct 2026)
  2. Gemini (gemini-2.0-flash)   — fallback via google-genai SDK
  3. Mock / demo mode            — always available, no API key needed
"""
from __future__ import annotations

import json
import streamlit as st

from backend.config import (
    get_groq_api_key,
    get_gemini_api_key,
    GROQ_MODEL,
    GEMINI_MODEL,
    LLM_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq chat completions API."""
    from groq import Groq  # type: ignore
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Google Gemini API using the current google-genai SDK."""
    try:
        # Current SDK: google-genai (replaces deprecated google-generativeai)
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4096,
            ),
        )
        # Extract text — new SDK stores it in candidates
        if response.text:
            return response.text
        # Fallback: dig into candidates manually
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
        raise ValueError("Gemini returned an empty response.")
    except ImportError as imp_err:
        raise RuntimeError(
            "google-genai package not found. Run: pip install google-genai"
        ) from imp_err


def _mock_response(
    num_questions: int,
    difficulty: str,
    question_types: list[str],
) -> str:
    """
    Return a hard-coded JSON response so the app is fully testable
    without any API key (demo / offline mode).
    """
    questions = []
    q_index = 0

    if "mcq" in question_types:
        for i in range(min(2, num_questions - q_index)):
            questions.append({
                "type": "mcq",
                "question": f"[DEMO MCQ {i+1}] Which of the following best describes photosynthesis?",
                "options": {
                    "A": "The process by which plants convert sunlight into chemical energy",
                    "B": "The process by which animals digest food",
                    "C": "The process of cellular respiration",
                    "D": "The process of water evaporation",
                },
                "answer": "A",
                "explanation": "Photosynthesis is the process used by plants to convert light energy into glucose.",
                "source_hint": "Demo mode — no real notes were analysed.",
                "flagged": False,
            })
            q_index += 1
            if q_index >= num_questions:
                break

    if "true_false" in question_types and q_index < num_questions:
        questions.append({
            "type": "true_false",
            "question": "[DEMO T/F] Plants release oxygen as a by-product of photosynthesis.",
            "answer": "True",
            "explanation": "Oxygen is released when water molecules are split during the light-dependent reactions.",
            "source_hint": "Demo mode — no real notes were analysed.",
            "flagged": False,
        })
        q_index += 1

    if "short_answer" in question_types and q_index < num_questions:
        questions.append({
            "type": "short_answer",
            "question": "[DEMO SA] What molecule is produced as the primary output of photosynthesis?",
            "answer": "Glucose (C₆H₁₂O₆)",
            "explanation": "Glucose is synthesised in the Calvin cycle using CO₂ and the energy stored in ATP and NADPH.",
            "source_hint": "Demo mode — no real notes were analysed.",
            "flagged": False,
        })
        q_index += 1

    # Pad to requested count with MCQ copies if needed
    while q_index < num_questions:
        questions.append({
            "type": "mcq",
            "question": f"[DEMO MCQ extra {q_index+1}] Sample question about the provided notes.",
            "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
            "answer": "A",
            "explanation": "This is a placeholder question in demo mode.",
            "source_hint": "Demo mode.",
            "flagged": False,
        })
        q_index += 1

    return json.dumps({"questions": questions})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_quiz_raw(
    prompt: str,
    num_questions: int,
    difficulty: str,
    question_types: list[str],
) -> tuple[str, str]:
    """
    Attempt to call a real LLM; fall back to demo mode.

    Returns:
        (raw_text, provider_used)
        provider_used is one of: "groq", "gemini", "demo"
    """
    groq_key = get_groq_api_key()
    gemini_key = get_gemini_api_key()

    # --- Try Groq ---
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            return _call_groq(prompt, groq_key), "groq"
        except Exception as exc:
            st.warning(f"Groq unavailable ({exc}). Trying Gemini…")

    # --- Try Gemini ---
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            return _call_gemini(prompt, gemini_key), "gemini"
        except Exception as exc:
            st.warning(f"Gemini unavailable ({exc}). Falling back to demo mode.")

    # --- Demo / mock mode ---
    st.info(
        "⚠️ **Demo mode** — No valid API key found. "
        "Showing sample questions. Add your GROQ_API_KEY or GEMINI_API_KEY to `.env` or `.streamlit/secrets.toml`."
    )
    return _mock_response(num_questions, difficulty, question_types), "demo"
