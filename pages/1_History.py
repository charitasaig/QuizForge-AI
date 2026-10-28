"""
1_History.py — Persistent quiz history page (SQLite-backed).
"""
from __future__ import annotations

import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.core import history_service
from backend.core.quiz_exporter import (
    export_questions_txt,
    export_answer_key_txt,
    export_quiz_csv,
)

st.set_page_config(
    page_title="Quiz History | QuizForge AI",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Quiz History")
st.caption("All quizzes are saved to a local SQLite database and persist across sessions.")

# ---------------------------------------------------------------------------
# Load history
# ---------------------------------------------------------------------------
try:
    history = history_service.list_quizzes(limit=50)
except Exception as exc:
    st.error(f"Could not load history: {exc}")
    history = []

if not history:
    st.info(
        "No quizzes saved yet. Head to the home page to generate your first quiz!"
    )
    st.stop()

st.markdown(f"**{len(history)} quiz session(s)** saved.")
st.divider()

# ---------------------------------------------------------------------------
# Delete-confirmation state
# ---------------------------------------------------------------------------
if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None

# ---------------------------------------------------------------------------
# List quizzes
# ---------------------------------------------------------------------------
for entry in history:
    _qid    = entry["id"]
    _title  = entry.get("title", "Untitled")
    _ts     = entry.get("created_at", "")
    _diff   = entry.get("difficulty", "").capitalize()
    _count  = entry.get("question_count", 0)
    _types  = entry.get("question_types", "")
    _prov   = entry.get("provider", "").upper()
    _has_r  = bool(entry.get("has_result"))

    _header = f"**{_title}** · `{_qid}` · {_ts}"
    with st.expander(_header):
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Questions",  _count)
        _c2.metric("Difficulty", _diff)
        _c3.metric("Provider",   _prov)
        _c4.metric("Scored",     "Yes" if _has_r else "No")
        st.caption(f"Types: {_types}")
        st.caption(f"Source: {entry.get('source_type', 'pasted')}")
        st.caption(f"Preview: {entry.get('notes_preview', '')}")

        # --- Load full quiz for downloads ---
        _col_dl1, _col_dl2, _col_dl3 = st.columns(3)
        with _col_dl1:
            if st.button("📂 Load details", key=f"load_{_qid}"):
                _full = history_service.load_quiz(_qid)
                if _full:
                    _qs = _full.get("questions", [])
                    st.markdown(f"**{len(_qs)} questions loaded**")
                    for _i, _q in enumerate(_qs):
                        st.markdown(f"- Q{_i+1} [{_q.get('type','')}]: {_q.get('question','')[:80]}")
                    if _full.get("result"):
                        _r = _full["result"]
                        st.success(
                            f"Score: {_r.get('score',0):.0f}/{_r.get('max_score',0):.0f} "
                            f"({_r.get('percentage',0):.0f}%)"
                        )
                else:
                    st.warning("Quiz not found.")

        with _col_dl2:
            # Download buttons — we need to load the quiz
            _full2 = history_service.load_quiz(_qid)
            if _full2:
                _qs2 = _full2.get("questions", [])
                _d2  = _full2.get("difficulty", "medium")
                _qtxt = export_questions_txt(_qs2, _d2, _qid)
                st.download_button(
                    "⬇️ Questions TXT",
                    data=_qtxt,
                    file_name=f"quizforge_{_qid}_questions.txt",
                    mime="text/plain",
                    key=f"dl_q_{_qid}",
                )

        with _col_dl3:
            if _full2:
                _csv = export_quiz_csv(_qs2, _d2, _qid)
                st.download_button(
                    "⬇️ Full CSV",
                    data=_csv,
                    file_name=f"quizforge_{_qid}_quiz.csv",
                    mime="text/csv",
                    key=f"dl_csv_{_qid}",
                )

        # --- Delete ---
        st.markdown("")
        if st.session_state.confirm_delete_id == _qid:
            st.warning(f"Are you sure you want to delete quiz `{_qid}`? This cannot be undone.")
            _dc1, _dc2 = st.columns(2)
            if _dc1.button("✅ Yes, delete", key=f"confirm_yes_{_qid}"):
                history_service.delete_quiz(_qid)
                st.session_state.confirm_delete_id = None
                st.rerun()
            if _dc2.button("❌ Cancel", key=f"confirm_no_{_qid}"):
                st.session_state.confirm_delete_id = None
                st.rerun()
        else:
            if st.button("🗑️ Delete this quiz", key=f"del_{_qid}", type="secondary"):
                st.session_state.confirm_delete_id = _qid
                st.rerun()

st.divider()
if st.button("🗑️ Delete ALL history", type="secondary"):
    st.session_state.confirm_delete_id = "__ALL__"

if st.session_state.confirm_delete_id == "__ALL__":
    st.warning("This will permanently delete ALL saved quizzes.")
    _da1, _da2 = st.columns(2)
    if _da1.button("✅ Yes, delete all"):
        for _e in history:
            history_service.delete_quiz(_e["id"])
        st.session_state.confirm_delete_id = None
        st.rerun()
    if _da2.button("❌ Cancel"):
        st.session_state.confirm_delete_id = None
        st.rerun()
