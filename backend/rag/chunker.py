"""
chunker.py — Document chunking with metadata preservation for QuizForge AI RAG.

Splits text into coherent chunks at paragraph and sentence boundaries,
preserving character offsets, chunk index, and approximate token counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    index: int
    text: str
    char_start: int
    char_end: int
    source_name: str = "notes"
    page_number: int | None = None

    @property
    def snippet(self) -> str:
        """First 120 characters of the chunk for preview."""
        clean = " ".join(self.text.split())
        return (clean[:117] + "…") if len(clean) > 120 else clean


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    source_name: str = "notes",
) -> list[TextChunk]:
    """
    Split text into overlapping chunks, respecting paragraph and sentence boundaries.

    Args:
        text: Raw document text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        source_name: Name or filename of the document.

    Returns:
        List of TextChunk instances with accurate character offsets.
    """
    if not text or not text.strip():
        return []

    # If text is smaller than chunk size, return single chunk
    stripped = text.strip()
    if len(stripped) <= chunk_size:
        start = text.find(stripped)
        return [
            TextChunk(
                chunk_id="chunk_0",
                index=0,
                text=stripped,
                char_start=start if start >= 0 else 0,
                char_end=(start + len(stripped)) if start >= 0 else len(stripped),
                source_name=source_name,
            )
        ]

    chunks: list[TextChunk] = []
    text_len = len(text)
    start = 0
    chunk_idx = 0

    while start < text_len:
        # Determine candidate end position
        target_end = min(start + chunk_size, text_len)

        if target_end < text_len:
            # Look backwards from target_end for a natural sentence or paragraph break
            search_window = text[start + int(chunk_size * 0.6) : target_end]
            # Try paragraph break
            para_break = search_window.rfind("\n\n")
            if para_break != -1:
                end = start + int(chunk_size * 0.6) + para_break + 2
            else:
                # Try sentence boundary (. followed by space or newline)
                sent_match = None
                for m in re.finditer(r"\.\s+", search_window):
                    sent_match = m
                if sent_match:
                    end = start + int(chunk_size * 0.6) + sent_match.end()
                else:
                    # Try newline or space
                    space_break = search_window.rfind(" ")
                    if space_break != -1:
                        end = start + int(chunk_size * 0.6) + space_break + 1
                    else:
                        end = target_end
        else:
            end = text_len

        chunk_text = text[start:end].strip()
        if chunk_text:
            actual_start = text.find(chunk_text, start)
            actual_end = actual_start + len(chunk_text) if actual_start >= 0 else end

            chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{chunk_idx}",
                    index=chunk_idx,
                    text=chunk_text,
                    char_start=actual_start if actual_start >= 0 else start,
                    char_end=actual_end,
                    source_name=source_name,
                )
            )
            chunk_idx += 1

        if end >= text_len:
            break

        # Move forward, stepping back by overlap
        step = max(1, (end - start) - chunk_overlap)
        start += step

    return chunks
