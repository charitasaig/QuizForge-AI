# 🧠 QuizForge AI — Smart Notes-to-Quiz Generator

Turn class notes into revision quizzes in seconds using AI.  
Supports Multiple Choice, True/False and Short Answer questions with answer keys, scoring, timed mode, export and persistent history.

---

## What QuizForge AI Does

1. Accept pasted text or upload a TXT/PDF file.
2. Send the notes to an LLM (Groq or Gemini).
3. Return a structured quiz of MCQ, True/False and/or Short Answer questions.
4. Let the student take the quiz and submit answers.
5. Score the results and show an answer key with explanations and source hints.
6. Export questions and answers to TXT or CSV.
7. Save every quiz to a local SQLite database so history persists across sessions.

---

## Current Features

| Feature | Status |
|---------|--------|
| Paste notes | ✅ |
| Upload TXT / PDF | ✅ |
| Multiple Choice questions (4 options) | ✅ |
| True / False questions | ✅ |
| Short Answer questions | ✅ |
| Difficulty selector (easy / medium / hard) | ✅ |
| Question count selector (1–20) | ✅ |
| Separate quiz view — answers hidden | ✅ |
| Answer key + explanations + source hints | ✅ |
| Question quality validation (duplicates, invalid MCQ, flagged) | ✅ |
| Score calculation (MCQ, T/F, short-answer keyword baseline) | ✅ |
| Incorrect-answer review with filter | ✅ |
| Timed quiz mode (5/10/15/30 min) | ✅ |
| Export questions TXT | ✅ |
| Export answer key TXT | ✅ |
| Export full CSV | ✅ |
| Export results CSV (after submission) | ✅ |
| Persistent quiz history (SQLite) | ✅ |
| Delete individual / all history entries | ✅ |
| Groq → Gemini → Demo fallback chain | ✅ |
| Demo mode (no API key required) | ✅ |

---

## Prerequisites

- Python 3.11 or 3.12 (3.13 also tested and working)
- pip
- A free Groq API key (recommended) — <https://console.groq.com>
- Optional: Gemini API key — <https://aistudio.google.com/app/apikey>

---

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/your-username/quizforge-ai.git
cd quizforge-ai
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example file and fill in your key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_real_key_here
```

> **No API key?** Leave the placeholder. The app runs in **demo mode** automatically and shows sample questions.

### 5. Run the application

```powershell
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

All 142 tests run without a live API — LLM calls are mocked.

---

## How to Use Each Question Type

| Type | How to answer | Scoring |
|------|--------------|---------|
| Multiple Choice | Click one radio button (A–D) | 1 pt if exact match |
| True / False | Click True or False | 1 pt if exact match |
| Short Answer | Type a few sentences | Keyword-overlap baseline; marked Uncertain when partial |

Short-answer grading is approximate. The status **Uncertain** means the keyword overlap was 30–60 % — always review these manually.

---

## Difficulty Levels

| Level | Question style |
|-------|---------------|
| Easy | Definitions, direct recall, simple examples |
| Medium | Concepts, comparisons, relationships between ideas |
| Hard | Multi-step reasoning, analysis, synthesising multiple points |

Hard questions are still grounded in the supplied notes — no outside knowledge is required.

---

## Timed Quiz Mode

1. Enable **Timed Mode** in the sidebar.
2. Select a duration (5 / 10 / 15 / 30 minutes).
3. Click **Generate Quiz** — the timer starts immediately.
4. The deadline is stored as an absolute timestamp, so it is not reset by Streamlit reruns.
5. When time expires the quiz is submitted automatically.
6. Install `streamlit-autorefresh` for a live countdown display:
   ```powershell
   pip install streamlit-autorefresh
   ```
   Without it, the timer updates only on interaction but still submits correctly when the deadline passes.

---

## Score Calculation

After submitting, QuizForge shows:

- Total questions
- Correct / Incorrect / Unanswered / Uncertain
- Score (points / max points)
- Percentage
- Time taken (timed mode only)

MCQ and True/False use exact-match grading.  
Short answers use keyword-overlap: ≥ 60 % overlap = correct, 30–60 % = uncertain, < 30 % = incorrect.

---

## Exporting Quizzes

Four download buttons appear below every quiz:

| Button | Contents |
|--------|----------|
| Questions TXT | Questions and options only — **no answers** |
| Answer Key TXT | Answers, explanations and source hints |
| Full CSV | All fields including answers (UTF-8, properly escaped) |
| Results CSV | Student answers + grading status (appears after submission) |

Filenames include the quiz ID, e.g. `quizforge_a1b2c3d4_questions.txt`.

---

## Quiz History

All quizzes are saved to `data/quiz_history.db` (SQLite).  
History persists across browser refreshes and server restarts.

From the **History** page (sidebar):
- Browse all saved quizzes with metadata.
- Load and inspect question lists.
- Download questions or CSV for any past quiz.
- Delete individual quizzes (with confirmation).
- Delete all history.

---

## Data Flow

```
User (paste / upload)
  │
  ▼
validators.py          ← reject empty, oversized, wrong-type input
  │
  ▼
pipeline.extract_text()     ← TXT decode or PDF text extraction
  │
  ▼
pipeline.prepare_notes()    ← clean whitespace, truncate to 15 000 chars
  │
  ▼
pipeline.build_prompt()     ← inject notes + difficulty + types into prompt
  │
  ▼
llm_service.generate_quiz_raw()
  ├─ Try Groq  (openai/gpt-oss-20b)
  ├─ Fallback Gemini (gemini-2.5-flash)
  └─ Fallback demo/mock mode
  │
  ▼
pipeline.parse_quiz_response()   ← validate and normalise JSON
  │
  ▼
quiz_validator.validate_questions()  ← remove duplicates, invalid MCQs
  │
  ▼
history_service.save_quiz()     ← persist to SQLite
  │
  ▼
Streamlit UI  ← quiz form → submit → quiz_evaluator → results → export
```

---

## Prompt Design

The prompt instructs the LLM to:

1. Use **only** the provided notes — no invented facts.
2. Include a `source_hint` for every answer (≤ 15 words quoting the notes).
3. Set `"flagged": true` on questions that are vague or weakly supported.
4. Return **strict JSON** — no markdown fences, no commentary.

---

## LLM Providers

| Priority | Provider | Model | Notes |
|----------|----------|-------|-------|
| 1 | Groq | `openai/gpt-oss-20b` | Fast, free tier |
| 2 | Gemini | `gemini-2.5-flash` | Google free tier |
| 3 | Demo mode | — | No key needed; sample questions |

Model names and free-tier limits change. The versions above were verified in October 2026.

---

## How Scanned PDFs Are Handled

QuizForge uses `pypdf` for text extraction. If a PDF contains no extractable text (i.e. it is a scanned image), the app displays:

> "Could not extract text from this PDF. It may be a scanned image. Please use a text-based PDF or paste the content manually."

OCR is not built in. To use scanned PDFs, run an OCR tool (such as Tesseract) separately, save the output as a TXT file, and upload that instead.

---

## How Long Notes Are Processed

- Maximum input: **15 000 characters** (~3 000 words).
- Notes longer than this are truncated at a sentence boundary before being sent to the LLM.
- A visible `[… content truncated …]` marker is appended so you know truncation occurred.
- If your notes are too long, copy the most relevant section and paste that instead.

---

## Known Limitations

- Scanned / image-based PDFs are not supported.
- Short-answer grading is a keyword-overlap heuristic — always review **Uncertain** results manually.
- AI-generated questions may be incorrect or imprecise — verify against your source notes.
- The quality validator uses normalised text comparison for duplicate detection; near-synonyms are not detected.
- Free-tier rate limits may cause temporary failures; the app falls back to demo mode automatically.
- The timer countdown requires `streamlit-autorefresh` for live display; without it the timer still enforces the deadline but only updates on user interaction.
- Maximum 20 questions per generation.

---

## Project Structure

```
Quiz-Forge/
├── app.py                            Main Streamlit entry point
├── pages/
│   └── 1_History.py                 Persistent quiz history page
├── components/
│   └── quiz_card.py                 Reusable UI helpers
├── backend/
│   ├── config.py                    Reads secrets / env vars
│   ├── services/
│   │   └── llm_service.py           Groq / Gemini / demo fallback
│   ├── core/
│   │   ├── pipeline.py              Text extract → prepare → prompt → parse
│   │   ├── quiz_validator.py        Duplicate / invalid MCQ / flag detection
│   │   ├── quiz_evaluator.py        Score calculation and per-question results
│   │   ├── quiz_exporter.py         TXT and CSV export
│   │   └── history_service.py       SQLite persistence
│   └── utils/
│       └── validators.py            Input and file validation
├── data/
│   ├── sample/                      Three synthetic sample note files
│   └── quiz_history.db              SQLite database (auto-created)
├── tests/                           142 pytest tests, no live API calls
│   ├── test_validators.py
│   ├── test_pipeline.py
│   ├── test_quiz_validator.py
│   ├── test_quiz_evaluator.py
│   ├── test_quiz_exporter.py
│   ├── test_history_service.py
│   ├── test_llm_service.py
│   ├── test_document_processing.py
│   └── test_scenarios.py
├── .env                             Your real keys (never commit)
├── .env.example                     Placeholder keys (safe to commit)
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | UI framework |
| groq | Groq API client |
| google-genai | Gemini API client |
| pypdf | PDF text extraction |
| pandas | Data handling |
| python-dotenv | Load `.env` for local dev |

Optional: `streamlit-autorefresh` for live timer countdown.

---

## Deployment (Streamlit Community Cloud)

1. Push to a public GitHub repository (ensure `.env` and `secrets.toml` are in `.gitignore`).
2. Go to <https://share.streamlit.io> → **New app** → select repo and `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy. The SQLite history file will persist within the deployment container session.

---

## License

MIT — see `LICENSE` for details.
