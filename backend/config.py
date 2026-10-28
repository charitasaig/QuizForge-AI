"""
config.py — Reads API keys from st.secrets (deployed) or .env (local).
Never log or expose the actual key values.
"""
import os
import streamlit as st


def get_groq_api_key() -> str | None:
    """Return the Groq API key, or None if not configured."""
    # 1. Streamlit secrets (Streamlit Cloud deployment)
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Environment variable (local .env loaded by python-dotenv in app.py)
    return os.environ.get("GROQ_API_KEY")


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key, or None if not configured."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GEMINI_API_KEY")


# --- App-wide constants ---
MAX_FILE_SIZE_MB: int = 5
MAX_TEXT_CHARS: int = 15_000   # ~3 000 words — safe LLM context
ALLOWED_EXTENSIONS: tuple[str, ...] = (".txt", ".pdf")

# LLM settings
GROQ_MODEL: str = "openai/gpt-oss-20b"          # fast, free-tier model on Groq (Oct 2026)
GROQ_MODEL_LARGE: str = "openai/gpt-oss-120b"   # higher quality fallback
GEMINI_MODEL: str = "gemini-2.5-flash"           # current Gemini free-tier model
LLM_TIMEOUT_SECONDS: int = 30
