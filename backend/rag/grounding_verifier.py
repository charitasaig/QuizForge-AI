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


def _get_content_words(text: str, min_len: int = 3) -> set[str]:
    """Extract content words ignoring standard english stopwords."""
    _STOPWORDS = {
        "the", "and", "is", "in", "it", "of", "to", "a", "an", "for", "on", "with",
        "as", "by", "at", "from", "that", "this", "these", "those", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "but", "not", "what", "which", "who", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
    }
    words = _norm(text).split()
    return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}


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
    else:
        answer_text = str(question.get("answer", "")).strip()

    chunk_norm = _norm(chunk.text)
    chunk_words = _get_content_words(chunk.text)

    # 1. Check if source_hint directly matches text in the chunk
    hint_norm = _norm(hint)
    hint_words = _get_content_words(hint)

    hint_supported = False
    if hint_norm and hint_norm in chunk_norm:
        hint_supported = True
    elif hint_words:
        overlap = hint_words & chunk_words
        if len(overlap) / len(hint_words) >= 0.60:
            hint_supported = True

    # 2. Check if answer content words are present in chunk
    answer_words = _get_content_words(answer_text)
    answer_supported = False
    if not answer_words:
        # e.g. T/F "True" or empty
        answer_supported = True
    else:
        overlap = answer_words & chunk_words
        ratio = len(overlap) / len(answer_words)
        # For multi-word answers, require at least 50% content words in chunk
        if len(answer_words) == 1:
            answer_supported = bool(overlap)
        else:
            answer_supported = ratio >= 0.50

    # 3. Check if stem topic has presence in chunk
    stem_words = _get_content_words(stem)
    stem_overlap = stem_words & chunk_words
    stem_supported = len(stem_overlap) >= 1 if stem_words else True

    # Overall grounding decision:
    # Must have stem relevance AND (answer supported OR hint supported)
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
    all_query_words = stem_words | answer_words | hint_words

    for s in sentences:
        s_words = _get_content_words(s)
        match_count = len(s_words & all_query_words)
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
