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
    Produces varied, distinct questions across requested types.
    """
    _MCQ_POOL = [
        {
            "type": "mcq",
            "question": "[DEMO MCQ 1] Which of the following best describes photosynthesis?",
            "options": {
                "A": "The process by which plants convert sunlight into chemical energy",
                "B": "The process by which animals digest food",
                "C": "The process of cellular respiration",
                "D": "The process of water evaporation",
            },
            "answer": "A",
            "explanation": "Photosynthesis is the process used by plants to convert light energy into glucose.",
            "source_hint": "Demo mode — converts sunlight into chemical energy.",
            "flagged": False,
        },
        {
            "type": "mcq",
            "question": "[DEMO MCQ 2] Where do the light-dependent reactions occur within plant chloroplasts?",
            "options": {
                "A": "Thylakoid membranes",
                "B": "Mitochondrial matrix",
                "C": "Cell wall",
                "D": "Cytoplasm",
            },
            "answer": "A",
            "explanation": "Light reactions occur in the thylakoid membranes.",
            "source_hint": "Demo mode — light reactions occur in thylakoids.",
            "flagged": False,
        },
        {
            "type": "mcq",
            "question": "[DEMO MCQ 3] Which green pigment absorbs sunlight to power photosynthesis?",
            "options": {
                "A": "Chlorophyll",
                "B": "Carotene",
                "C": "Melanin",
                "D": "Hemoglobin",
            },
            "answer": "A",
            "explanation": "Chlorophyll is the primary pigment for light absorption.",
            "source_hint": "Demo mode — chlorophyll absorbs sunlight.",
            "flagged": False,
        },
    ]

    _TF_POOL = [
        {
            "type": "true_false",
            "question": "[DEMO T/F 1] Plants release oxygen as a by-product of photosynthesis.",
            "answer": "True",
            "explanation": "Oxygen is released when water molecules are split during light reactions.",
            "source_hint": "Demo mode — oxygen released as a by-product.",
            "flagged": False,
        },
        {
            "type": "true_false",
            "question": "[DEMO T/F 2] The Calvin cycle occurs in the thylakoid membrane.",
            "answer": "False",
            "explanation": "The Calvin cycle occurs in the stroma of the chloroplast.",
            "source_hint": "Demo mode — Calvin cycle occurs in the stroma.",
            "flagged": False,
        },
    ]

    _SA_POOL = [
        {
            "type": "short_answer",
            "question": "[DEMO SA 1] What molecule is produced as the primary carbohydrate output of photosynthesis?",
            "answer": "Glucose (C₆H₁₂O₆)",
            "explanation": "Glucose is synthesised in the Calvin cycle using carbon dioxide and light energy.",
            "source_hint": "Demo mode — glucose is synthesized in the Calvin cycle.",
            "flagged": False,
        },
        {
            "type": "short_answer",
            "question": "[DEMO SA 2] What chemical energy carrier molecule powers cellular biochemical reactions?",
            "answer": "Adenosine triphosphate (ATP)",
            "explanation": "ATP stores and delivers energy for cellular processes.",
            "source_hint": "Demo mode — energy stored in ATP molecules.",
            "flagged": False,
        },
    ]

    questions = []
    mcq_idx = 0
    tf_idx = 0
    sa_idx = 0

    # Round-robin among requested question types to satisfy requested count
    for slot in range(num_questions):
        t = question_types[slot % len(question_types)]
        if t == "mcq":
            if mcq_idx < len(_MCQ_POOL):
                questions.append(dict(_MCQ_POOL[mcq_idx]))
                mcq_idx += 1
            else:
                questions.append({
                    "type": "mcq",
                    "question": f"[DEMO MCQ {slot+1}] Which factor directly affects the rate of photosynthesis?",
                    "options": {"A": "Light intensity", "B": "Wind direction", "C": "Sound waves", "D": "Magnetic field"},
                    "answer": "A",
                    "explanation": "Light intensity directly affects the rate of photosynthesis.",
                    "source_hint": "Demo mode — light intensity rate factor.",
                    "flagged": False,
                })
        elif t == "true_false":
            if tf_idx < len(_TF_POOL):
                questions.append(dict(_TF_POOL[tf_idx]))
                tf_idx += 1
            else:
                questions.append({
                    "type": "true_false",
                    "question": f"[DEMO T/F {slot+1}] Carbon dioxide enters plant leaves through microscopic pores called stomata.",
                    "answer": "True",
                    "explanation": "Stomata allow gas exchange.",
                    "source_hint": "Demo mode — stomata allow gas exchange.",
                    "flagged": False,
                })
        elif t == "short_answer":
            if sa_idx < len(_SA_POOL):
                questions.append(dict(_SA_POOL[sa_idx]))
                sa_idx += 1
            else:
                questions.append({
                    "type": "short_answer",
                    "question": f"[DEMO SA {slot+1}] Name the microscopic pores on plant leaves that facilitate gas exchange.",
                    "answer": "Stomata",
                    "explanation": "Stomata regulate carbon dioxide uptake and water vapor loss.",
                    "source_hint": "Demo mode — stomata facilitate gas exchange.",
                    "flagged": False,
                })

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
