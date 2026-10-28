"""
test_rag_pipeline.py — Unit tests for RAG chunking, vector store, retrieval, and grounding.

Covers:
  1. Chunking with paragraph/sentence boundaries and accurate offsets.
  2. Local vector store similarity search and diverse passage sampling.
  3. Grounding verifier: true positive grounding vs rejection of unsupported questions.
  4. Non-hallucination guarantee: unsupported questions are never given false citations.
  5. RAGService integration.
"""
from __future__ import annotations

import pytest

from backend.rag.chunker import TextChunk, split_text_into_chunks
from backend.rag.embeddings import LocalEmbeddingEngine
from backend.rag.vector_store import LocalVectorStore
from backend.rag.grounding_verifier import verify_grounding
from backend.rag.rag_service import RAGService


_SAMPLE_NOTES = (
    "Photosynthesis is the process used by plants, algae and certain bacteria to turn sunlight, "
    "water and carbon dioxide into oxygen and energy in the form of sugar. "
    "The overall chemical equation is 6CO2 + 6H2O + light -> C6H12O6 + 6O2.\n\n"
    "Cellular respiration is a set of metabolic reactions that take place in the cells of organisms "
    "to convert chemical energy from oxygen molecules or nutrients into adenosine triphosphate (ATP), "
    "and then release waste products. Catabolic reactions include glycolysis, the citric acid cycle, "
    "and oxidative phosphorylation.\n\n"
    "Mitosis is a part of the cell cycle in which replicated chromosomes are separated into two new nuclei. "
    "Cell division gives rise to genetically identical cells in which the total number of chromosomes is maintained. "
    "The stages of mitosis are prophase, metaphase, anaphase, and telophase."
)


# ---------------------------------------------------------------------------
# 1. Chunking Tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_short_text_returns_single_chunk(self):
        text = "Short lecture note about ribosomes and protein synthesis."
        chunks = split_text_into_chunks(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_empty_text_returns_empty_list(self):
        assert split_text_into_chunks("") == []
        assert split_text_into_chunks("   ") == []

    def test_long_text_split_into_multiple_chunks(self):
        chunks = split_text_into_chunks(_SAMPLE_NOTES, chunk_size=300, chunk_overlap=50)
        assert len(chunks) >= 3
        # Ensure chunks have non-empty text and sequential indexes
        for i, c in enumerate(chunks):
            assert c.index == i
            assert len(c.text) > 0
            assert c.char_end > c.char_start

    def test_chunk_offsets_match_original_text(self):
        chunks = split_text_into_chunks(_SAMPLE_NOTES, chunk_size=250, chunk_overlap=40)
        for c in chunks:
            extracted = _SAMPLE_NOTES[c.char_start : c.char_end]
            assert extracted == c.text


# ---------------------------------------------------------------------------
# 2. Embedding & Vector Store Tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_vector_store_indexes_and_searches(self):
        chunks = split_text_into_chunks(_SAMPLE_NOTES, chunk_size=300, chunk_overlap=50)
        store = LocalVectorStore()
        store.add_chunks(chunks)

        # Query about mitosis
        results = store.similarity_search("chromosomes stages of mitosis prophase", top_k=2)
        assert len(results) > 0
        top_chunk, score = results[0]
        assert "mitosis" in top_chunk.text.lower()
        assert score > 0.0

    def test_diverse_chunk_sampling(self):
        chunks = split_text_into_chunks(_SAMPLE_NOTES, chunk_size=200, chunk_overlap=30)
        store = LocalVectorStore()
        store.add_chunks(chunks)

        sampled = store.get_diverse_chunks(3)
        assert len(sampled) == min(3, len(chunks))
        # Ensure distinct chunks sampled
        assert len({c.chunk_id for c in sampled}) == len(sampled)


# ---------------------------------------------------------------------------
# 3. Grounding Verifier Tests
# ---------------------------------------------------------------------------

class TestGroundingVerifier:
    def test_grounded_mcq_passes_and_gets_evidence(self):
        chunk = TextChunk(
            chunk_id="c1",
            index=0,
            text="Photosynthesis converts water and carbon dioxide into oxygen and glucose sugar.",
            char_start=0,
            char_end=80,
        )
        q = {
            "type": "mcq",
            "question": "What does photosynthesis produce from water and carbon dioxide?",
            "options": {
                "A": "Oxygen and glucose sugar",
                "B": "Methane gas",
                "C": "Nitrogen dioxide",
                "D": "Sodium chloride",
            },
            "answer": "A",
            "source_hint": "converts water and carbon dioxide into oxygen and glucose",
        }
        ok, reason, meta = verify_grounding(q, chunk)
        assert ok is True
        assert meta["grounded"] is True
        assert "Chunk 1" in meta["source_reference"]
        assert meta["source_chunk_id"] == "c1"
        assert len(meta["source_evidence"]) > 0

    def test_unsupported_question_rejected(self):
        """A question about an unrelated topic must NOT be verified as grounded."""
        chunk = TextChunk(
            chunk_id="c1",
            index=0,
            text="Photosynthesis converts water and carbon dioxide into oxygen and glucose sugar.",
            char_start=0,
            char_end=80,
        )
        unrelated_q = {
            "type": "mcq",
            "question": "Who was the first emperor of the Roman Empire?",
            "options": {"A": "Augustus", "B": "Nero", "C": "Caesar", "D": "Tiberius"},
            "answer": "A",
            "source_hint": "Augustus ruled Rome.",
        }
        ok, reason, meta = verify_grounding(unrelated_q, chunk)
        assert ok is False
        assert meta["grounded"] is False
        assert meta["source_reference"] == ""
        assert "Grounding check failed" in reason

    def test_never_invents_citations_when_unsupported(self):
        chunk = TextChunk(
            chunk_id="c2",
            index=1,
            text="Mitosis separates chromosomes into two identical daughter cells.",
            char_start=100,
            char_end=170,
        )
        hallucinated_q = {
            "type": "short_answer",
            "question": "What is the boiling point of ethanol at sea level?",
            "answer": "78.37 degrees Celsius",
            "source_hint": "Ethanol boiling point.",
        }
        ok, _, meta = verify_grounding(hallucinated_q, chunk)
        assert ok is False
        assert meta["grounded"] is False
        assert meta["source_reference"] == ""
        assert meta["source_evidence"] == ""


# ---------------------------------------------------------------------------
# 4. End-to-End RAGService Tests
# ---------------------------------------------------------------------------

class TestRAGServiceIntegration:
    def test_rag_service_lifecycle(self):
        rag = RAGService(chunk_size=300, chunk_overlap=50)
        chunks = rag.ingest_notes(_SAMPLE_NOTES)
        assert len(chunks) >= 3

        retrieved = rag.retrieve_chunks_for_quiz(num_questions=3)
        assert len(retrieved) == 3

        mode = rag.get_active_embedding_mode()
        assert mode in ("sentence-transformers", "tfidf")
