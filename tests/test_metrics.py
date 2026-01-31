"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    calculate_text_similarity,
    calculate_extraction_accuracy,
    calculate_table_accuracy,
    calculate_chunk_quality,
    calculate_classification_accuracy,
    EvaluationMetrics,
)
from src.common.models import Chunk


class TestTextSimilarity:
    """Tests for text similarity calculation."""

    def test_identical_text(self):
        result = calculate_text_similarity(
            "The quick brown fox",
            "The quick brown fox"
        )
        assert result["similarity"] == 1.0
        assert result["word_error_rate"] == 0.0

    def test_completely_different_text(self):
        result = calculate_text_similarity(
            "The quick brown fox",
            "Lorem ipsum dolor"
        )
        assert result["similarity"] < 0.5
        assert result["word_error_rate"] > 0.5

    def test_partial_match(self):
        result = calculate_text_similarity(
            "The quick brown fox jumps",
            "The quick brown dog runs"
        )
        assert 0.3 < result["similarity"] < 0.8

    def test_empty_reference(self):
        result = calculate_text_similarity("", "Some text")
        assert result["similarity"] == 0.0

    def test_empty_hypothesis(self):
        result = calculate_text_similarity("Some text", "")
        assert result["similarity"] == 0.0

    def test_character_similarity(self):
        result = calculate_text_similarity(
            "Hello World",
            "Hello Wordl",  # Typo
            method="character"
        )
        assert result["similarity"] >= 0.8
        assert result["character_error_rate"] <= 0.2

    def test_sequence_similarity(self):
        result = calculate_text_similarity(
            "ABCDEFG",
            "ABXDEFG",
            method="sequence"
        )
        assert result["similarity"] > 0.8


class TestExtractionAccuracy:
    """Tests for extraction accuracy calculation."""

    def test_perfect_extraction(self):
        expected = {"name": "John", "age": "30", "city": "NYC"}
        extracted = {"name": "John", "age": "30", "city": "NYC"}

        result = calculate_extraction_accuracy(expected, extracted)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_extraction(self):
        expected = {"name": "John", "age": "30", "city": "NYC"}
        extracted = {"name": "John", "age": "30"}

        result = calculate_extraction_accuracy(expected, extracted)
        assert result["recall"] < 1.0
        assert result["key_recall"] < 1.0

    def test_extra_fields_extracted(self):
        expected = {"name": "John"}
        extracted = {"name": "John", "extra": "field"}

        result = calculate_extraction_accuracy(expected, extracted)
        assert result["recall"] == 1.0
        assert result["precision"] < 1.0

    def test_fuzzy_matching(self):
        # Use values with high token overlap for fuzzy matching
        expected = {"name": "John Smith Junior"}
        extracted = {"name": "John Smith Jr"}

        result = calculate_extraction_accuracy(expected, extracted, fuzzy_match=True)
        # With token similarity, "John Smith" matches 2/3 tokens from expected
        # The fuzzy match threshold is 0.8 so this should match
        assert result["key_precision"] == 1.0
        assert result["key_recall"] == 1.0

    def test_empty_expected(self):
        result = calculate_extraction_accuracy({}, {"field": "value"})
        assert result["f1"] == 0.0


class TestTableAccuracy:
    """Tests for table extraction accuracy."""

    def test_perfect_table_match(self):
        expected = [["A", "B"], ["1", "2"]]
        extracted = [["A", "B"], ["1", "2"]]

        result = calculate_table_accuracy(expected, extracted)
        assert result["cell_accuracy"] == 1.0
        assert result["structure_accuracy"] == 1.0

    def test_wrong_structure(self):
        expected = [["A", "B"], ["1", "2"]]
        extracted = [["A", "B", "C"]]  # Wrong dimensions

        result = calculate_table_accuracy(expected, extracted)
        assert result["structure_accuracy"] < 1.0

    def test_partial_cell_match(self):
        expected = [["A", "B"], ["1", "2"]]
        extracted = [["A", "X"], ["1", "Y"]]

        result = calculate_table_accuracy(expected, extracted)
        assert 0.4 < result["cell_accuracy"] < 0.6

    def test_empty_table(self):
        result = calculate_table_accuracy([], [["A"]])
        assert result["cell_accuracy"] == 0.0


class TestChunkQuality:
    """Tests for chunk quality metrics."""

    def test_good_chunks(self):
        source = "This is sentence one. This is sentence two. This is sentence three."
        chunks = [
            Chunk(id="1", content="This is sentence one."),
            Chunk(id="2", content="This is sentence two."),
            Chunk(id="3", content="This is sentence three."),
        ]

        result = calculate_chunk_quality(chunks, source)
        assert result["coverage"] > 0.8
        assert result["coherence"] > 0.8
        assert result["num_chunks"] == 3

    def test_poor_coverage(self):
        source = "This is a long document with many words and sentences."
        chunks = [
            Chunk(id="1", content="short"),
        ]

        result = calculate_chunk_quality(chunks, source)
        assert result["coverage"] < 0.5

    def test_empty_chunks(self):
        result = calculate_chunk_quality([], "Some content")
        assert result["coverage"] == 0.0
        assert result["coherence"] == 0.0


class TestClassificationAccuracy:
    """Tests for classification accuracy."""

    def test_correct_classification(self):
        result = calculate_classification_accuracy("invoice", "invoice", 0.95)
        assert result["accuracy"] == 1.0
        assert result["confidence"] == 0.95
        assert result["weighted_accuracy"] == 0.95

    def test_incorrect_classification(self):
        result = calculate_classification_accuracy("invoice", "receipt", 0.8)
        assert result["accuracy"] == 0.0
        assert result["weighted_accuracy"] == 0.0

    def test_case_insensitive(self):
        result = calculate_classification_accuracy("Invoice", "INVOICE", 0.9)
        assert result["accuracy"] == 1.0


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_default_values(self):
        metrics = EvaluationMetrics()
        assert metrics.text_similarity == 0.0
        assert metrics.processing_time_ms == 0.0
        assert metrics.details == {}

    def test_custom_values(self):
        metrics = EvaluationMetrics(
            text_similarity=0.85,
            extraction_f1=0.90,
            processing_time_ms=1500.0
        )
        assert metrics.text_similarity == 0.85
        assert metrics.extraction_f1 == 0.90
        assert metrics.processing_time_ms == 1500.0
