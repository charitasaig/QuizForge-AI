"""
test_nlp_validator.py — Comprehensive tests for NLP semantic similarity & quality checks.

Covers:
  1. Active mode reporting ('sentence-transformers' or 'tfidf').
  2. Paraphrased semantic duplicate detection.
  3. Configurable similarity threshold.
  4. False-positive resistance (distinct questions on the same subject must pass).
  5. False-negative resistance (paraphrased questions asking the same thing must be caught).
  6. Distractor quality validation.
  7. Answer-in-stem leakage detection.
  8. Linguistic clarity & length bounds.
  9. Full integration with validate_questions().
"""
from __future__ import annotations

import pytest

from backend.nlp.semantic_similarity import (
    SemanticSimilarityDetector,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from backend.nlp.quality_checker import (
    check_all_quality,
    check_distractor_quality,
    check_answer_leakage,
    check_linguistic_clarity,
)
from backend.core.quiz_validator import validate_questions, QuestionIssue


def _mcq(question="What is photosynthesis?", answer="A", options=None, idx=0):
    return {
        "id": f"q{idx}",
        "type": "mcq",
        "question": question,
        "options": options or {
            "A": "Converting light energy into glucose",
            "B": "Digesting cellular waste",
            "C": "Cellular respiration in mitochondria",
            "D": "Water evaporation from plant leaves",
        },
        "answer": answer,
        "explanation": "Plants use light to produce carbohydrates.",
        "source_hint": "Light conversion.",
        "flagged": False,
    }


# ---------------------------------------------------------------------------
# 1. Semantic Similarity Detector Unit Tests
# ---------------------------------------------------------------------------

class TestSemanticSimilarityDetector:
    def test_detector_reports_mode(self):
        detector = SemanticSimilarityDetector()
        mode = detector.get_active_mode()
        assert mode in ("sentence-transformers", "tfidf")

    def test_identical_questions_score_one(self):
        detector = SemanticSimilarityDetector()
        q = "What is the function of the cell nucleus?"
        score = detector.compute_similarity(q, q)
        assert score >= 0.99

    def test_completely_different_questions_score_low(self):
        detector = SemanticSimilarityDetector()
        q1 = "What is the function of the cell nucleus?"
        q2 = "Who was the prime minister of Britain during World War II?"
        score = detector.compute_similarity(q1, q2)
        assert score < 0.40

    def test_paraphrased_questions_detected(self):
        detector = SemanticSimilarityDetector()
        q1 = "What organelle produces ATP in cells?"
        q2 = "Which organelle produces ATP in animal cells?"
        score = detector.compute_similarity(q1, q2)
        assert score >= 0.70


# ---------------------------------------------------------------------------
# 2. Configurable Threshold, False Positives & False Negatives
# ---------------------------------------------------------------------------

class TestConfigurableThresholdAndEdgeCases:
    def test_configurable_threshold_sensitivity(self):
        detector = SemanticSimilarityDetector()
        q1 = "What is the capital of France?"
        q2 = "What is the capital city of France?"
        sim = detector.compute_similarity(q1, q2)

        # At low threshold, it must be flagged
        dups_low = detector.find_duplicates([q1, q2], threshold=0.60)
        assert len(dups_low) == 1

        # At ultra-high threshold (e.g. 0.999), it should not be flagged unless identical
        dups_ultra = detector.find_duplicates([q1, q2], threshold=0.999)
        assert len(dups_ultra) == 0

    def test_false_positive_resistance(self):
        """
        Two distinct questions about related biology topics must NOT be flagged as duplicates.
        """
        detector = SemanticSimilarityDetector()
        q1 = "What is the primary function of the cell wall in plant cells?"
        q2 = "What is the primary function of chloroplasts in plant cells?"
        dups = detector.find_duplicates([q1, q2], threshold=0.85)
        assert len(dups) == 0

    def test_false_negative_resistance(self):
        """
        Two questions asking the exact same question with different wording MUST be flagged.
        """
        detector = SemanticSimilarityDetector()
        q1 = "Which molecule is the main energy currency of the cell?"
        q2 = "Which molecule is the primary energy currency of the cell?"
        dups = detector.find_duplicates([q1, q2], threshold=0.75)
        assert len(dups) == 1
        dup_idx, orig_idx, score = dups[0]
        assert dup_idx == 1
        assert orig_idx == 0
        assert score >= 0.75


# ---------------------------------------------------------------------------
# 3. Question Quality & Distractor Checks
# ---------------------------------------------------------------------------

class TestQuestionQualityChecks:
    def test_duplicate_distractor_detected(self):
        q = _mcq(options={
            "A": "Glucose",
            "B": "Glucose",  # duplicate distractor
            "C": "Oxygen",
            "D": "Water",
        })
        issues = check_distractor_quality(q, 0)
        assert any(i.issue_type == "invalid_mcq" and "duplicate" in i.detail.lower() for i in issues)

    def test_single_char_trivial_distractor_flagged(self):
        q = _mcq(options={
            "A": "Photosynthesis",
            "B": "x",  # trivial distractor
            "C": "Cellular respiration",
            "D": "Fermentation",
        })
        issues = check_distractor_quality(q, 0)
        assert any(i.issue_type == "quality" and "too brief" in i.detail for i in issues)

    def test_answer_in_stem_leakage_detected(self):
        q = {
            "type": "mcq",
            "question": "What is the biological process called photosynthesis?",
            "options": {
                "A": "Photosynthesis",
                "B": "Respiration",
                "C": "Osmosis",
                "D": "Diffusion",
            },
            "answer": "A",
        }
        issues = check_answer_leakage(q, 0)
        assert any(i.issue_type == "quality" and "leak" in i.detail for i in issues)

    def test_no_leakage_for_subtle_answers(self):
        q = _mcq(
            question="Which process converts sunlight into chemical energy in plants?",
            answer="A",
            options={
                "A": "Photosynthesis",
                "B": "Respiration",
                "C": "Glycolysis",
                "D": "Krebs cycle",
            },
        )
        issues = check_answer_leakage(q, 0)
        assert len(issues) == 0

    def test_dangling_deictic_reference_flagged(self):
        q = _mcq(question="As mentioned above in the diagram, what does the arrow indicate?")
        issues = check_linguistic_clarity(q, 0)
        assert any(i.issue_type == "ambiguous" and "dangling" in i.detail for i in issues)

    def test_extremely_short_question_flagged(self):
        q = _mcq(question="Cells divide?")  # only 2 words
        issues = check_linguistic_clarity(q, 0)
        assert any(i.issue_type == "ambiguous" and "fewer than 4 words" in i.detail for i in issues)


# ---------------------------------------------------------------------------
# 4. Full Integration with validate_questions()
# ---------------------------------------------------------------------------

class TestValidateQuestionsWithNLP:
    def test_semantic_duplicate_excluded_in_report(self):
        q1 = _mcq(question="Which organelle produces ATP in cells?", idx=0)
        q2 = _mcq(question="Which organelle produces ATP in animal cells?", idx=1)
        report = validate_questions([q1, q2], similarity_threshold=0.75)
        assert report.duplicate_count >= 1
        assert report.valid_count == 1
        assert any(i.issue_type == "duplicate" and "semantically" in i.detail.lower() for i in report.issues)

    def test_distinct_questions_both_valid(self):
        q1 = _mcq(question="What is the function of the cell wall in plant cells?", idx=0)
        q2 = _mcq(question="What is the role of ribosomes in protein synthesis?", idx=1)
        report = validate_questions([q1, q2], similarity_threshold=0.82)
        assert report.duplicate_count == 0
        assert report.valid_count == 2

    def test_report_includes_embedding_mode(self):
        report = validate_questions([_mcq()])
        assert hasattr(report, "embedding_mode")
        assert report.embedding_mode in ("sentence-transformers", "tfidf")
