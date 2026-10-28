"""
grounding_verifier.py — Grounding and evidence validation for QuizForge AI.

Strictly verifies that generated questions and answers are substantiated
by the retrieved chunk text. Never invents citations or claims unsupported
questions are grounded.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from backend.rag.chunker import TextChunk


def _norm(text: str) -> str:
    """Normalize text: strip diacritics, lowercase, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem_word(w: str) -> str:
    """Lightweight suffix-stripper for robust inflection matching (e.g. releases -> releas)."""
    for suff in ("ing", "tion", "sion", "ment", "able", "ible", "ness", "ed", "es", "s"):
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            return w[:-len(suff)]
    return w


def _get_content_stems(text: str, min_len: int = 3) -> set[str]:
    """Extract content word stems ignoring standard english stopwords and boolean markers."""
    _STOPWORDS = {
        "the", "and", "is", "in", "it", "of", "to", "a", "an", "for", "on", "with",
        "as", "by", "at", "from", "that", "this", "these", "those", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "but", "not", "what", "which", "who", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "true", "false",  # boolean answers handled separately
    }
    words = _norm(text).split()
    return {_stem_word(w) for w in words if len(w) >= min_len and w not in _STOPWORDS}


def verify_grounding(
    question: dict[str, Any],
    chunk: TextChunk,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Verify whether a generated question and answer are grounded in a retrieved chunk.

    Returns:
        (is_grounded, reason, metadata_dict)
    """
    q_type = question.get("type", "mcq")
    stem = question.get("question", "")
    hint = question.get("source_hint", "")

    # Extract answer text
    if q_type == "mcq":
        options = question.get("options") or {}
        ans_key = str(question.get("answer", "")).strip().upper()
        answer_text = str(options.get(ans_key, "")).strip()
    elif q_type == "true_false":
        answer_text = ""  # Boolean truth value, not content
    else:
        answer_text = str(question.get("answer", "")).strip()

    chunk_norm = _norm(chunk.text)
    chunk_stems = _get_content_stems(chunk.text)

    # 1. Check stem relevance
    stem_stems = _get_content_stems(stem)
    stem_overlap = stem_stems & chunk_stems
    stem_supported = len(stem_overlap) >= 1 if stem_stems else True

    # 2. Check hint relevance
    hint_norm = _norm(hint)
    hint_stems = _get_content_stems(hint)
    hint_supported = False
    if hint_norm and hint_norm in chunk_norm:
        hint_supported = True
    elif hint_stems:
        hint_overlap = hint_stems & chunk_stems
        if len(hint_overlap) / len(hint_stems) >= 0.35:
            hint_supported = True

    # 3. Check answer relevance
    answer_norm = _norm(answer_text)
    answer_stems = _get_content_stems(answer_text)
    answer_supported = False
    if q_type == "true_false":
        # For True/False, the assertion is the stem itself; require substantial stem overlap
        answer_supported = (len(stem_overlap) / len(stem_stems) >= 0.30) if stem_stems else True
    elif answer_norm and answer_norm in chunk_norm:
        answer_supported = True
    elif answer_stems:
        answer_overlap = answer_stems & chunk_stems
        if len(answer_stems) == 1:
            answer_supported = bool(answer_overlap)
        else:
            answer_supported = (len(answer_overlap) / len(answer_stems) >= 0.30)

    # Decision: question stem must be present AND (answer supported OR hint supported)
    is_grounded = stem_supported and (answer_supported or hint_supported)

    if not is_grounded:
        missing = []
        if not stem_supported:
            missing.append("question topic not present in chunk")
        if not (answer_supported or hint_supported):
            missing.append("answer/evidence not substantiated by chunk")
        reason = f"Grounding check failed: {', '.join(missing)} in chunk {chunk.chunk_id}."
        return False, reason, {
            "grounded": False,
            "source_reference": "",
            "source_chunk_id": "",
            "source_evidence": "",
        }

    # Extract best sentence from chunk as concrete source evidence
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
    best_sent = ""
    best_score = 0
    all_query_stems = stem_stems | answer_stems | hint_stems

    for s in sentences:
        s_stems = _get_content_stems(s)
        match_count = len(s_stems & all_query_stems)
        if match_count > best_score:
            best_score = match_count
            best_sent = s.strip()

    evidence_quote = best_sent if best_sent else chunk.snippet

    metadata = {
        "grounded": True,
        "source_reference": f"Chunk {chunk.index + 1} (chars {chunk.char_start}–{chunk.char_end})",
        "source_chunk_id": chunk.chunk_id,
        "source_evidence": evidence_quote,
    }

    return True, "Validated: question and answer are grounded in retrieved evidence.", metadata
