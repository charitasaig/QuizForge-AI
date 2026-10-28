"""
app.py — QuizForge AI  |  Streamlit entry point.

Data flow:
  User input (text / file)
    → validators.py          validate input
    → pipeline.py            extract → prepare → prompt → LLM → parse → validate
    → quiz_evaluator.py      score calculation
    → quiz_exporter.py       TXT / CSV downloads
    → history_service.py     SQLite persistence
    → UI                     quiz form / answer key / results / export / history
"""
from __future__ import annotations

import datetime
import time
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Back-end imports
from backend.utils.validators import (
    validate_text_input,
    validate_uploaded_file,
    validate_quiz_params,
)
from backend.core.pipeline import extract_text, run_quiz_pipeline
from backend.core.quiz_validator import validate_questions
from backend.core.quiz_evaluator import evaluate_quiz, QuizResult
from backend.core.quiz_exporter import (
    export_questions_txt,
    export_answer_key_txt,
    export_quiz_csv,
    export_results_csv,
)
from backend.core import history_service

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuizForge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "quiz": [],               # list of question dicts (validated)
    "quiz_id": "",            # current quiz UUID
    "provider": "",           # groq | gemini | demo
    "difficulty": "medium",   # persisted for export
    "question_types": [],     # persisted for export
    "notes_text": "",
    "source_type": "pasted",
    "timed_mode": False,
    "timer_deadline": None,   # absolute datetime when time expires
    "user_answers": {},       # {index: str}
    "submitted": False,
    "quiz_result": None,      # QuizResult after submission
    "show_answer_key": False,
    "validation_report": None,
    "new_quiz_pending": False,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Auto-refresh (timer countdown) — only when a timed quiz is active
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _has_autorefresh = True
except ImportError:
    _has_autorefresh = False

_quiz_active = (
    st.session_state.quiz
    and st.session_state.timed_mode
    and not st.session_state.submitted
    and st.session_state.timer_deadline is not None
)
if _quiz_active and _has_autorefresh:
    st_autorefresh(interval=1000, limit=None, key="timer_refresh")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("QuizForge AI")
    st.caption("Turn notes into quizzes in seconds.")
    st.divider()

    st.subheader("⚙️ Quiz Settings")

    _difficulty = st.select_slider(
        "Difficulty",
        options=["easy", "medium", "hard"],
        value="medium",
        help=(
            "Easy: definitions & recall | "
            "Medium: concepts & comparisons | "
            "Hard: analysis & synthesis"
        ),
    )

    _num_questions = st.slider(
        "Number of questions", min_value=1, max_value=20, value=5
    )

    st.markdown("**Question types**")
    _use_mcq = st.checkbox("Multiple Choice (MCQ)", value=True)
    _use_tf  = st.checkbox("True / False", value=True)
    _use_sa  = st.checkbox("Short Answer", value=False)

    st.divider()
    st.subheader("⏱️ Timed Mode")
    _timed_mode = st.checkbox("Enable timer", value=False)
    _timer_option = st.selectbox(
        "Duration",
        options=[5, 10, 15, 30],
        index=0,
        format_func=lambda m: f"{m} minutes",
        disabled=not _timed_mode,
    )

    st.divider()
    st.markdown(
        "**Limitations**\n"
        "- AI answers may be imperfect — verify against your notes.\n"
        "- Max input: 15 000 characters (~3 000 words).\n"
        "- Scanned/image PDFs are not supported.\n"
        "- Short-answer grading is approximate.\n"
        "- Free-tier rate limits apply."
    )

# ---------------------------------------------------------------------------
# Assemble question_types list
# ---------------------------------------------------------------------------
_question_types: list[str] = []
if _use_mcq:
    _question_types.append("mcq")
if _use_tf:
    _question_types.append("true_false")
if _use_sa:
    _question_types.append("short_answer")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 QuizForge AI — Smart Notes-to-Quiz Generator")
st.markdown(
    "Paste your class notes **or** upload a TXT/PDF file, set the options in the sidebar, "
    "then click **Generate Quiz**."
)
st.info(
    "💡 The AI generates questions based *only* on your notes. "
    "Always verify important answers against your source material."
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
tab_paste, tab_upload = st.tabs(["📋 Paste Notes", "📁 Upload File"])

_raw_notes: str = ""
_source_type: str = "pasted"

with tab_paste:
    _pasted = st.text_area(
        "Paste your notes here",
        height=250,
        placeholder="e.g. Photosynthesis is the process by which plants convert sunlight…",
    )
    if _pasted:
        _raw_notes = _pasted

with tab_upload:
    _uploaded_file = st.file_uploader(
        "Upload TXT or PDF (max 5 MB)",
        type=["txt", "pdf"],
        help="Text-based PDFs only — scanned image PDFs are not supported.",
    )
    if _uploaded_file is not None:
        _ok, _msg = validate_uploaded_file(_uploaded_file)
        if not _ok:
            st.error(_msg)
        else:
            try:
                _extracted = extract_text(_uploaded_file)
                _raw_notes = _extracted
                _source_type = "uploaded"
                st.success(
                    f"✅ Extracted {len(_extracted):,} characters from **{_uploaded_file.name}**."
                )
                with st.expander("Preview extracted text (first 500 chars)"):
                    st.text(_extracted[:500])
            except ValueError as _exc:
                st.error(str(_exc))
            except Exception as _exc:
                st.error(f"Unexpected error reading file: {_exc}")

# ---------------------------------------------------------------------------
# Generate Quiz button
# ---------------------------------------------------------------------------
st.divider()
_generate_clicked = st.button(
    "🚀 Generate Quiz", type="primary", use_container_width=True
)

if _generate_clicked:
    if not _raw_notes:
        st.error("Please paste notes or upload a file before generating.")
        st.stop()

    _ok, _msg = validate_text_input(_raw_notes)
    if not _ok:
        st.error(_msg)
        st.stop()

    _ok, _msg = validate_quiz_params(_num_questions, _difficulty, _question_types)
    if not _ok:
        st.error(_msg)
        st.stop()

    with st.spinner("🤖 Generating quiz… this may take up to 30 seconds."):
        try:
            _questions, _provider = run_quiz_pipeline(
                _raw_notes, _num_questions, _difficulty, _question_types
            )

            # Run (and store) validation report
            _report = validate_questions(_questions)

            # Persist to SQLite
            _notes_preview = (
                (_raw_notes[:120] + "…") if len(_raw_notes) > 120 else _raw_notes
            )
            _quiz_id = history_service.save_quiz(
                questions=_questions,
                difficulty=_difficulty,
                question_types=_question_types,
                provider=_provider,
                notes_preview=_notes_preview,
                source_type=_source_type,
            )

            # Store in session
            st.session_state.quiz               = _questions
            st.session_state.quiz_id            = _quiz_id
            st.session_state.provider           = _provider
            st.session_state.difficulty         = _difficulty
            st.session_state.question_types     = _question_types
            st.session_state.notes_text         = _raw_notes
            st.session_state.source_type        = _source_type
            st.session_state.timed_mode         = _timed_mode
            st.session_state.user_answers       = {}
            st.session_state.submitted          = False
            st.session_state.quiz_result        = None
            st.session_state.show_answer_key    = False
            st.session_state.validation_report  = _report
            st.session_state.new_quiz_pending   = False

            if _timed_mode:
                st.session_state.timer_deadline = (
                    datetime.datetime.now()
                    + datetime.timedelta(minutes=_timer_option)
                )
            else:
                st.session_state.timer_deadline = None

            # Provider badge
            _badge_colours = {"groq": "#F97316", "gemini": "#3B82F6", "demo": "#6B7280"}
            _badge_labels  = {"groq": "⚡ Groq", "gemini": "✨ Gemini", "demo": "🔧 Demo Mode"}
            _bc = _badge_colours.get(_provider, "#6B7280")
            _bl = _badge_labels.get(_provider, _provider.upper())
            st.success(
                f"✅ **{len(_questions)} questions generated** "
                f"· Quiz ID: `{_quiz_id}` "
            )
            st.markdown(
                f'<span style="background:{_bc};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:0.85rem;font-weight:600;">{_bl}</span>',
                unsafe_allow_html=True,
            )

        except ValueError as _exc:
            st.error(f"Failed to parse AI response: {_exc}")
            st.stop()
        except Exception as _exc:
            st.error(
                f"An unexpected error occurred: {_exc}\n\n"
                "Check your API key or try demo mode (leave API key blank)."
            )
            st.stop()

# ===========================================================================
# Quiz display section — only shown when a quiz is loaded
# ===========================================================================
if not st.session_state.quiz:
    st.stop()

_questions   = st.session_state.quiz
_quiz_id     = st.session_state.quiz_id
_difficulty  = st.session_state.difficulty
_q_types     = st.session_state.question_types
_report      = st.session_state.validation_report
_is_submitted = st.session_state.submitted

st.divider()

# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
if st.session_state.timed_mode and not _is_submitted:
    _deadline = st.session_state.timer_deadline
    if _deadline:
        _remaining = max(0.0, (_deadline - datetime.datetime.now()).total_seconds())
        _mins, _secs = divmod(int(_remaining), 60)
        _tc = "red" if _remaining < 60 else "orange" if _remaining < 180 else "green"
        st.markdown(
            f"<h3 style='color:{_tc}'>⏱️ Time remaining: {_mins:02d}:{_secs:02d}</h3>",
            unsafe_allow_html=True,
        )
        if _remaining <= 0 and not _is_submitted:
            st.warning("⏰ Time's up! Submitting automatically…")
            # Auto-submit with whatever answers are in session state
            _result = evaluate_quiz(
                _questions,
                st.session_state.user_answers,
                start_time=_deadline - datetime.timedelta(
                    seconds=_timer_option * 60 if "_timer_option" in dir() else 300
                ),
            )
            st.session_state.quiz_result = _result
            st.session_state.submitted   = True
            history_service.save_result(_quiz_id, {
                "score": _result.score,
                "max_score": _result.max_score,
                "percentage": _result.percentage,
                "correct": _result.correct,
                "incorrect": _result.incorrect,
                "unanswered": _result.unanswered,
                "uncertain": _result.uncertain,
            })
            st.rerun()
    if not _has_autorefresh:
        st.caption(
            "⚠️ Install `streamlit-autorefresh` for a live countdown: "
            "`pip install streamlit-autorefresh`"
        )

# ---------------------------------------------------------------------------
# Validation quality report (collapsible)
# ---------------------------------------------------------------------------
if _report:
    _issues = _report.issues
    _has_issues = _report.duplicate_count > 0 or _report.invalid_count > 0 or _report.flagged_count > 0
    with st.expander(
        f"🔍 Question Quality Report — {_report.valid_count}/{_report.total_generated} valid"
        + (" ⚠️" if _has_issues else " ✅"),
        expanded=_has_issues,
    ):
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Generated",   _report.total_generated)
        _c2.metric("Valid",       _report.valid_count)
        _c3.metric("Duplicates removed",  _report.duplicate_count)
        _c4.metric("Flagged",     _report.flagged_count)
        if _report.invalid_count:
            st.warning(f"⚠️ {_report.invalid_count} question(s) excluded due to structural issues.")
        for _iss in _issues:
            _icon = {"invalid_mcq": "❌", "duplicate": "🔁", "flagged": "🚩", "ambiguous": "⚠️"}.get(
                _iss.issue_type, "ℹ️"
            )
            _label = "excluded" if _iss.excluded else "flagged"
            st.markdown(
                f"{_icon} **Q{_iss.question_index + 1}** [{_iss.issue_type} — {_label}]: "
                f"{_iss.detail}"
            )

# ---------------------------------------------------------------------------
# Quiz heading
# ---------------------------------------------------------------------------
_diff_cap = _difficulty.capitalize() if _difficulty else "Medium"
st.subheader(f"📝 Quiz — {_diff_cap} | {len(_questions)} Questions")

# ---------------------------------------------------------------------------
# SECTION 1: Take the Quiz (hidden when submitted)
# ---------------------------------------------------------------------------
if not _is_submitted:
    with st.form("quiz_form"):
        for _i, _q in enumerate(_questions):
            _qtype = _q.get("type", "mcq")
            _flag  = " 🚩" if _q.get("flagged") else ""
            st.markdown(f"**Q{_i + 1}.** {_q.get('question', '')}{_flag}")

            _key = f"ans_{_i}"

            if _qtype == "mcq":
                _opts = _q.get("options") or {}
                _choices = [f"{k}. {v}" for k, v in _opts.items()]
                _ans = st.radio(
                    f"q_{_i}", _choices, key=_key,
                    index=None,                        # no option pre-selected
                    label_visibility="collapsed",
                )
                # Store only the letter (first char), empty string when nothing chosen
                st.session_state.user_answers[_i] = _ans[:1] if _ans else ""

            elif _qtype == "true_false":
                _ans = st.radio(
                    f"q_{_i}", ["True", "False"], key=_key,
                    index=None,                        # no option pre-selected
                    label_visibility="collapsed",
                )
                st.session_state.user_answers[_i] = _ans or ""

            elif _qtype == "short_answer":
                _ans = st.text_area(
                    f"Your answer for Q{_i + 1}",
                    key=_key,
                    height=80,
                    label_visibility="collapsed",
                )
                st.session_state.user_answers[_i] = _ans or ""

            st.markdown("---")

        _form_submitted = st.form_submit_button(
            "✅ Submit Quiz", use_container_width=True, type="primary"
        )

    if _form_submitted:
        _result = evaluate_quiz(
            _questions,
            st.session_state.user_answers,
            start_time=st.session_state.get("timer_deadline") and (
                st.session_state.timer_deadline
                - datetime.timedelta(minutes=_timer_option if "_timer_option" in dir() else 5)
            ),
        )
        st.session_state.quiz_result = _result
        st.session_state.submitted   = True
        history_service.save_result(_quiz_id, {
            "score": _result.score,
            "max_score": _result.max_score,
            "percentage": _result.percentage,
            "correct": _result.correct,
            "incorrect": _result.incorrect,
            "unanswered": _result.unanswered,
            "uncertain": _result.uncertain,
        })
        st.rerun()

# ---------------------------------------------------------------------------
# SECTION 2: Results + Answer Key (shown after submission)
# ---------------------------------------------------------------------------
if _is_submitted and st.session_state.quiz_result is not None:
    _result: QuizResult = st.session_state.quiz_result

    # --- Score card ---
    st.subheader("🎯 Results")
    _pct = _result.percentage

    _score_color = "#22C55E" if _pct == 100 else "#3B82F6" if _pct >= 70 else "#F59E0B" if _pct >= 40 else "#EF4444"
    _score_emoji = "🏆" if _pct == 100 else "👍" if _pct >= 70 else "📖" if _pct >= 40 else "💪"
    st.markdown(
        f"""<div style="background:{_score_color}22;border:2px solid {_score_color};
        border-radius:12px;padding:16px 24px;text-align:center;margin:12px 0;">
          <span style="font-size:2.5rem;">{_score_emoji}</span>
          <h2 style="color:{_score_color};margin:4px 0;">
            {int(_result.score)}/{int(_result.max_score)}
          </h2>
          <p style="color:{_score_color};font-weight:600;margin:0;">{_pct:.0f}%</p>
        </div>""",
        unsafe_allow_html=True,
    )

    _rc1, _rc2, _rc3, _rc4, _rc5 = st.columns(5)
    _rc1.metric("Total",       _result.total_questions)
    _rc2.metric("✅ Correct",  _result.correct)
    _rc3.metric("❌ Incorrect", _result.incorrect)
    _rc4.metric("⬜ Unanswered", _result.unanswered)
    _rc5.metric("❓ Uncertain", _result.uncertain)

    if _result.time_taken_seconds is not None and _result.time_taken_seconds > 0:
        _tm, _ts = divmod(int(_result.time_taken_seconds), 60)
        st.caption(f"⏱️ Time taken: {_tm:02d}:{_ts:02d}")

    if _pct == 100:
        st.balloons()
    elif _pct >= 70:
        st.success("Great job! Keep it up.")
    elif _pct >= 40:
        st.warning("Good effort — review the explanations and try again.")
    else:
        st.error("Keep studying — re-read your notes and retry.")

    if _result.uncertain > 0:
        st.info(
            f"ℹ️ {_result.uncertain} short-answer question(s) marked as **Uncertain** "
            "— keyword matching cannot fully verify these. Please review manually."
        )

    # --- Review section with filter ---
    st.subheader("📋 Answer Key & Review")
    _filter = st.radio(
        "Show",
        ["All questions", "❌ Incorrect", "⬜ Unanswered", "✅ Correct", "❓ Uncertain"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _filter_map = {
        "All questions": None,
        "❌ Incorrect": "incorrect",
        "⬜ Unanswered": "unanswered",
        "✅ Correct": "correct",
        "❓ Uncertain": "uncertain",
    }
    _status_filter = _filter_map[_filter]

    for _qr in _result.per_question:
        if _status_filter and _qr.status != _status_filter:
            continue

        _icon_map = {
            "correct": "✅", "incorrect": "❌",
            "unanswered": "⬜", "uncertain": "❓",
        }
        _q_icon = _icon_map.get(_qr.status, "ℹ️")
        _title  = f"{_q_icon} Q{_qr.question_index + 1}: {_qr.question_text[:80]}"
        if len(_qr.question_text) > 80:
            _title += "…"

        with st.expander(_title):
            if _qr.question_type == "mcq":
                _orig_q = _questions[_qr.question_index]
                for _k, _v in (_orig_q.get("options") or {}).items():
                    _m = "✅ " if _k == _qr.correct_answer else ""
                    st.markdown(f"- {_m}**{_k}.** {_v}")

            st.markdown(f"**Your answer:** {_qr.student_answer or '*(no answer)*'}")
            st.markdown(f"**Correct answer:** {_qr.correct_answer}")
            st.markdown(f"**Explanation:** {_qr.explanation}")
            if _qr.source_hint:
                st.caption(f"📌 Source hint: *{_qr.source_hint}*")
            if _qr.status == "uncertain":
                st.warning(
                    "Short-answer grading is approximate. "
                    "Please compare your answer against the expected answer manually."
                )

    # --- New quiz button ---
    st.divider()
    if st.button("🔄 Generate another quiz", use_container_width=True):
        for _k in ["quiz", "quiz_id", "submitted", "quiz_result",
                   "user_answers", "show_answer_key", "timed_mode",
                   "timer_deadline", "validation_report"]:
            st.session_state[_k] = _DEFAULTS.get(_k)
        st.rerun()

# ---------------------------------------------------------------------------
# Export section (always shown when a quiz exists)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📥 Export")

_exp_c1, _exp_c2, _exp_c3 = st.columns(3)

with _exp_c1:
    _q_txt = export_questions_txt(_questions, _difficulty, _quiz_id)
    st.download_button(
        "⬇️ Questions TXT",
        data=_q_txt,
        file_name=f"quizforge_{_quiz_id}_questions.txt",
        mime="text/plain",
        use_container_width=True,
        help="Questions only — no answers.",
    )

with _exp_c2:
    _ak_txt = export_answer_key_txt(_questions, _difficulty, _quiz_id)
    st.download_button(
        "⬇️ Answer Key TXT",
        data=_ak_txt,
        file_name=f"quizforge_{_quiz_id}_answers.txt",
        mime="text/plain",
        use_container_width=True,
        help="Answers + explanations.",
    )

with _exp_c3:
    _csv = export_quiz_csv(_questions, _difficulty, _quiz_id)
    st.download_button(
        "⬇️ Full CSV",
        data=_csv,
        file_name=f"quizforge_{_quiz_id}_quiz.csv",
        mime="text/csv",
        use_container_width=True,
        help="All fields including answers.",
    )

if _is_submitted and st.session_state.quiz_result is not None:
    _res_csv = export_results_csv(st.session_state.quiz_result, _quiz_id)
    st.download_button(
        "⬇️ Results CSV",
        data=_res_csv,
        file_name=f"quizforge_{_quiz_id}_results.csv",
        mime="text/csv",
        use_container_width=True,
        help="Your answers and scores.",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "QuizForge AI · Powered by Groq / Gemini · "
    "AI outputs may be incorrect — always verify against your source notes."
)
