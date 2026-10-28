"""
quiz_card.py — Reusable UI components for QuizForge AI.
Imported by app.py to keep the main file clean.
"""
from __future__ import annotations
import streamlit as st


def render_provider_badge(provider: str) -> None:
    """Show a small coloured badge indicating which LLM provider was used."""
    colours = {"groq": "#F97316", "gemini": "#3B82F6", "demo": "#6B7280"}
    labels = {"groq": "⚡ Groq", "gemini": "✨ Gemini", "demo": "🔧 Demo Mode"}
    colour = colours.get(provider, "#6B7280")
    label = labels.get(provider, provider.upper())
    st.markdown(
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_question_card(index: int, question: dict) -> None:
    """
    Render a single read-only question card (used in answer-key / review view).
    Not used inside forms — for display-only contexts.
    """
    q_type = question.get("type", "mcq")
    flagged = question.get("flagged", False)
    flag_icon = " 🚩" if flagged else ""

    with st.container():
        st.markdown(
            f"**Q{index + 1}.** {question.get('question', '')}{flag_icon}"
        )

        if q_type == "mcq":
            options = question.get("options", {})
            correct = question.get("answer", "")
            for key, val in options.items():
                icon = "✅" if key == correct else "◦"
                st.markdown(f"&nbsp;&nbsp;&nbsp;{icon} **{key}.** {val}")

        st.markdown(
            f"> **Answer:** {question.get('answer', '')}  \n"
            f"> **Explanation:** {question.get('explanation', '')}  \n"
            + (
                f"> 📌 *Source: {question.get('source_hint', '')}*"
                if question.get("source_hint")
                else ""
            )
        )
        if flagged:
            st.warning(
                "⚠️ This question was flagged as potentially vague or "
                "weakly supported by the notes."
            )
        st.markdown("---")


def render_score_card(correct: int, total: int) -> None:
    """Render a styled score summary."""
    pct = (correct / total * 100) if total else 0
    if pct == 100:
        colour, emoji = "#22C55E", "🏆"
    elif pct >= 70:
        colour, emoji = "#3B82F6", "👍"
    elif pct >= 40:
        colour, emoji = "#F59E0B", "📖"
    else:
        colour, emoji = "#EF4444", "💪"

    st.markdown(
        f"""
        <div style="background:{colour}22;border:2px solid {colour};border-radius:12px;
                    padding:16px 24px;text-align:center;margin:12px 0;">
          <span style="font-size:2rem;">{emoji}</span>
          <h2 style="color:{colour};margin:4px 0;">{correct} / {total}</h2>
          <p style="color:{colour};font-weight:600;margin:0;">{pct:.0f}% correct</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
