# 🧠 QuizForge AI — Smart Notes-to-Quiz Generator (Agentic RAG Edition)

Turn class notes into revision quizzes in seconds using AI.  
Upgraded with **Agentic AI orchestration**, **Retrieval-Augmented Generation (RAG)**, and **NLP-powered semantic quality gates**. Supports Multiple Choice, True/False, and Short Answer questions with source citations, evidence quotes, answer keys, scoring, timed mode, export, and persistent history.

---

## What QuizForge AI Does

1. **Accept Notes**: Paste text or upload a `.txt` / `.pdf` file.
2. **Chunk & Index (RAG)**: Break notes into boundary-aware text chunks and index them into an in-memory vector store using local embeddings (TF-IDF or Sentence-Transformers).
3. **Agentic AI Orchestration**: A bounded state machine plans topics across chunks, retrieves focused context passages, prompts the LLM (Groq or Gemini), and evaluates candidates against a multi-stage quality gate.
4. **NLP Quality & Grounding Gate**:
   - **Semantic Similarity Detection**: Prevents duplicate or near-duplicate questions using cosine similarity (configurable threshold: `0.82`).
   - **Quality & Leakage Checks**: Rejects answer-in-stem leakage, identical distractors, and empty options.
   - **Evidence Grounding**: Verifies factual grounding against retrieved chunks and extracts real sentence quotes. Never fabricates citations.
5. **Targeted Self-Correction**: Candidates failing validation receive targeted feedback and are retried up to 2 times per question slot.
6. **Student Quiz Interface**: Take quizzes interactively, submit answers, and receive instant scoring with explanations and verifiable source evidence.
7. **Export & History**: Export questions, answer keys, and results to TXT and CSV with full citation traceability, persisted automatically in a local SQLite database.

---

## Features

| Feature | Category | Status |
|---------|----------|--------|
| Paste notes & Upload TXT / PDF | Core | ✅ |
| Multiple Choice, True / False, Short Answer | Core | ✅ |
| Difficulty (Easy / Medium / Hard) & Count (1–20) | Core | ✅ |
| Answer key + explanations + source hints | Core | ✅ |
| Score calculation & incorrect-answer review | Core | ✅ |
| Timed quiz mode (5 / 10 / 15 / 30 min) | Core | ✅ |
| Export TXT (Questions & Answer Key) & CSV | Core | ✅ |
| Persistent SQLite quiz history | Core | ✅ |
| Multi-provider LLM chain: Groq → Gemini → Demo | LLM | ✅ |
| Text chunking with character-offset tracking | RAG | ✅ |
| In-memory vector store with cosine similarity | RAG | ✅ |
| Diverse chunk sampling across source notes | RAG | ✅ |
| Traceable source citations (`Chunk X (chars Y–Z)`) | RAG | ✅ |
| Factual evidence extraction (real sentence quotes) | RAG | ✅ |
| Graceful zero-dependency TF-IDF embedding fallback | RAG / NLP | ✅ |
| Auto-upgrade to `sentence-transformers` when installed | RAG / NLP | ✅ |
| Semantic duplicate detection (configurable threshold) | NLP | ✅ |
| Answer-in-stem leakage detection | NLP | ✅ |
| Distractor quality & option deduplication | NLP | ✅ |
| Bounded Agent workflow (Plan → Retrieve → Gen → Gate → Retry) | Agentic AI | ✅ |
| Hard retry bounds (`MAX_AGENT_RETRIES = 2`) | Agentic AI | ✅ |
| Targeted retry feedback prompting | Agentic AI | ✅ |
| Agent execution metrics & rejection categorization | Agentic AI | ✅ |
| Streamlit metrics expander & citations on cards | UI | ✅ |

---

## Architecture & Data Flow

```
                      User Notes (Pasted / TXT / PDF)
                                   │
                                   ▼
                      [backend/utils/validators.py]
                        (validate length & format)
                                   │
                                   ▼
                 [backend/rag/chunker.py]
           (Split into chunks with character offsets)
                                   │
                                   ▼
                [backend/rag/embeddings.py]
    (Compute vectors: Sentence-Transformers or TF-IDF)
                                   │
                                   ▼
               [backend/rag/vector_store.py]
                (In-Memory Cosine Vector Store)
                                   │
                                   ▼
      ┌────────────────────────────────────────────────────────┐
      │               QUIZ AGENT ORCHESTRATOR                  │
      │             [backend/agent/quiz_agent.py]              │
      │                                                        │
      │  1. Topic Planning & Passage Retrieval                │
      │     (Sample diverse chunks across notes)              │
      │                                                        │
      │  2. Candidate Generation via LLM Service               │
      │     (Groq -> Gemini -> Demo Mock Pool)                │
      │                                                        │
      │  3. Multi-Stage Quality Gate:                          │
      │     ├─ Structural Validation (valid MCQ/TF, format)    │
      │     ├─ Semantic Deduplication (cosine similarity <0.82)│
      │     ├─ NLP Quality (no answer leakage, distinct options│
      │     └─ Grounding Verifier (factual overlap with chunk) │
      │                                                        │
      │  4. Bounded Retry Feedback Loop (max 2 retries):       │
      │     └─ Pass rejection reason back to prompt if failed  │
      └────────────────────────────────────────────────────────┘
                                   │
                    Validated Questions & Metrics
                                   │
                                   ▼
                    [backend/core/pipeline.py]
          (Preserves signature: run_quiz_pipeline())
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
[SQLite History]                                      [Streamlit UI]
(Save questions & citations)                 (Display cards, metrics,
                                              citations & take quiz)
                                                              │
                                                              ▼
                                                     [Quiz Evaluator]
                                              (Scoring, answer review,
                                               TXT / CSV exports)
```

---

## Detailed Component Overview

### 1. NLP Layer (`backend/nlp/`)
- **`semantic_similarity.py`**:
  - Computes pairwise cosine similarity across question stems using local embeddings.
  - Rejects candidates exceeding `SEMANTIC_SIMILARITY_THRESHOLD` (default: `0.82`).
  - Configurable threshold allows tuning between strict novelty and permissive phrasing.
- **`quality_checker.py`**:
  - **Answer Leakage**: Detects if the correct answer verbatim or stem keywords appear directly in the question prompt.
  - **Distractor Quality**: Rejects multiple choice questions where distractors repeat each other or duplicate the correct answer.
  - **Clarity & Length Bounds**: Flags ambiguous questions or prompts with insufficient length.

### 2. RAG Layer (`backend/rag/`)
- **`chunker.py`**:
  - Segments input text into boundary-aware windows (default: 800 chars, 150 char overlap) preserving `char_start` and `char_end` relative to the original document.
- **`embeddings.py`**:
  - Provides a unified vector representation.
  - Automatically detects if `sentence-transformers` (`all-MiniLM-L6-v2`) is available; if not, falls back seamlessly to **Scikit-Learn normalized term-frequency vectors (TF-IDF)**.
  - Zero cloud dependencies, zero latency, 100% offline-capable.
- **`vector_store.py`**:
  - In-memory cosine similarity store supporting top-$k$ retrieval and diverse note sampling.
- **`grounding_verifier.py`**:
  - Validates factual support by checking content-word overlap between candidate questions/answers and retrieved note chunks.
  - Extracts the exact supporting sentence quote as `source_evidence`.
  - **Anti-Hallucination Guarantee**: If a question cannot be grounded in the text, `grounded = False` and no source citation is invented.

### 3. Agentic AI Layer (`backend/agent/`)
- **`quiz_agent.py`**:
  - Orchestrates question synthesis through a bounded state machine:
    1. **Plan**: Partitions generation targets across available evidence chunks.
    2. **Generate**: Requests candidate questions with focused context snippets.
    3. **Evaluate**: Tests candidate against structural, semantic similarity, quality, and grounding gates.
    4. **Self-Correct**: If rejected, injects explicit error guidance (e.g. *"Previous question was too similar to Question 1; generate a question focusing on a different concept"*) and re-generates.
  - **Bounded Execution**: Hard cap of `MAX_AGENT_RETRIES = 2` retries per question slot to prevent infinite loops.
  - **Execution Metrics**: Tracks total candidates generated, accepted, rejected by category (`structural`, `duplicate`, `quality`, `unsupported`), and self-corrected count.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 or 3.12 (Python 3.12.10 recommended)
- `pip`

### 2. Virtual Environment & Dependencies
```powershell
# Clone or navigate to the repository
cd Quiz-Forge-master

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

> **Optional: Dense Neural Embeddings**  
> QuizForge includes a local TF-IDF embedding engine out-of-the-box. To optionally enable dense neural embeddings:
> ```powershell
> pip install sentence-transformers
> ```
> QuizForge will auto-detect it and switch from `TF-IDF` to `Sentence-Transformers`.

### 3. API Key & Backend Configuration
Copy the `.env.example` file to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
BACKEND_URL=https://quizforge-ai-me4t.onrender.com
```
> **No API Key?** Leave the placeholder. QuizForge runs automatically in **Demo Mode**, generating structured sample quizzes across all question types with complete RAG grounding and agent validation.

### 4. Running the Frontend & Backend

#### A. Streamlit Frontend (Connected to Backend)
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) (or `http://localhost:8502`) in your browser. The sidebar displays live backend connectivity to `https://quizforge-ai-me4t.onrender.com`.

#### B. FastAPI REST API Backend
```powershell
python main.py
# Or: uvicorn main:app --reload --port 8000
```
Interactive OpenAPI documentation will be accessible at: [http://localhost:8000/docs](http://localhost:8000/docs).

#### C. Standalone Web Frontend
Simply open `frontend/index.html` in your browser. It connects directly to the backend URL via Fetch API and provides a complete responsive quiz taking experience.

### 5. Live Deployment
- **Live Deployed Service (Render):** [https://quizforge-ai-me4t.onrender.com/](https://quizforge-ai-me4t.onrender.com/)


---

## Running the Test Suite

QuizForge AI features a comprehensive automated test suite with **182 passing tests** and 0 failures, covering all layers with mocked LLMs:

```powershell
python -m pytest tests -v
```

### Test Breakdown
- `test_validators.py` (18 tests): Input validation, character limits, file format rules.
- `test_document_processing.py` (19 tests): TXT encodings (UTF-8, Latin-1, CP1252), PDF text extraction, scanned PDF detection, sentence-boundary truncation.
- `test_nlp_validator.py` (16 tests): Semantic deduplication, threshold sensitivity, distractor quality, answer leakage detection.
- `test_rag_pipeline.py` (10 tests): Boundary chunker, offset accuracy, vector store indexing, diverse sampling, grounding verifier, and anti-fabrication rules.
- `test_agentic_workflow.py` (5 tests): QuizAgent lifecycle, bounded retries, self-correction, metrics persistence, and pipeline signature backward-compatibility.
- `test_edge_cases.py` (9 tests): Malformed JSON recovery, stopwords-only similarity, empty input, grounding edge cases, provider fallback, retry exhaustion termination.
- `test_pipeline.py` (18 tests): Prompt generation, JSON extraction, error recovery.
- `test_quiz_validator.py` (21 tests): Structural MCQ/TF verification, duplicate removals.
- `test_quiz_evaluator.py` (16 tests): Deterministic scoring, exact match, keyword overlap heuristics.
- `test_quiz_exporter.py` (20 tests): TXT questions/answers, CSV headers, citation escaping.
- `test_history_service.py` (15 tests): SQLite CRUD, deletion, listing, retrieval.
- `test_llm_service.py` (5 tests): Groq → Gemini → Demo fallback cascade.
- `test_scenarios.py` (10 tests): End-to-end multi-disciplinary scenario runs (CS, Biology, History).

---

## Known Limitations

1. **Scanned / Image PDFs**: Text extraction relies on `pypdf`. Image-only / scanned PDFs without embedded text streams are detected and rejected with a helpful user prompt; OCR is not embedded.
2. **Short-Answer Auto-Grading**: Grading for free-text short answers utilizes a keyword-overlap baseline (≥60% correct, 30–60% uncertain). Students should review uncertain questions using the displayed source citation.
3. **Active Embedding Default**: By default, the system operates in local `TF-IDF` mode to ensure instant startup with zero network dependencies or heavy PyTorch downloads. Installing `sentence-transformers` unlocks dense neural representations.
4. **Context Limit**: Input documents are truncated at 15,000 characters at sentence boundaries to fit within LLM token budgets.

---

## License

MIT License — see `LICENSE` for details.
