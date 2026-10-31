"""
main.py — FastAPI REST API entry point for QuizForge AI.
Run with:
    uvicorn main:app --reload --port 8000
Or:
    python main.py
"""
import os
import uvicorn
from backend.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
