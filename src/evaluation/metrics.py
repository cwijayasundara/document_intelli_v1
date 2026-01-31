"""Evaluation metrics for document processing quality.

Provides metrics for:
- Text extraction accuracy
- Table extraction accuracy
- Handwriting recognition quality
- Chunk coherence
- Classification accuracy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Text metrics
    text_similarity: float = 0.0
    character_error_rate: float = 0.0
    word_error_rate: float = 0.0

    # Extraction metrics
    extraction_precision: float = 0.0
    extraction_recall: float = 0.0
    extraction_f1: float = 0.0

    # Table metrics
    table_cell_accuracy: float = 0.0
    table_structure_accuracy: float = 0.0

    # Chunk metrics
    chunk_coherence: float = 0.0
    chunk_coverage: float = 0.0

    # Classification metrics
    classification_accuracy: float = 0.0
    classification_confidence: float = 0.0

    # Performance metrics
    processing_time_ms: float = 0.0
    api_credits_used: float = 0.0

    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)


def calculate_text_similarity(
    reference: str,
    hypothesis: str,
    method: str = "token"
) -> Dict[str, float]:
    """Calculate text similarity between reference and hypothesis.

    Args:
        reference: Ground truth text
        hypothesis: Extracted text
        method: Similarity method ("token", "character", "sequence")

    Returns:
        Dict with similarity metrics
    """
    if not reference or not hypothesis:
        return {
            "similarity": 0.0,
            "character_error_rate": 1.0,
            "word_error_rate": 1.0,
        }

    if method == "character":
        return _character_similarity(reference, hypothesis)
    elif method == "sequence":
        return _sequence_similarity(reference, hypothesis)
    else:  # token
        return _token_similarity(reference, hypothesis)


def _token_similarity(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate token-based similarity."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if not ref_tokens:
        return {"similarity": 0.0, "word_error_rate": 1.0}

    intersection = ref_tokens & hyp_tokens
    union = ref_tokens | hyp_tokens

    jaccard = len(intersection) / len(union) if union else 0.0

    # Word Error Rate approximation
    ref_list = reference.lower().split()
    hyp_list = hypothesis.lower().split()
    wer = _levenshtein_distance(ref_list, hyp_list) / len(ref_list) if ref_list else 0.0

    return {
        "similarity": jaccard,
        "word_error_rate": min(wer, 1.0),
        "precision": len(intersection) / len(hyp_tokens) if hyp_tokens else 0.0,
        "recall": len(intersection) / len(ref_tokens) if ref_tokens else 0.0,
    }


def _character_similarity(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate character-based similarity."""
    ref_clean = reference.lower().replace(" ", "")
    hyp_clean = hypothesis.lower().replace(" ", "")

    if not ref_clean:
        return {"similarity": 0.0, "character_error_rate": 1.0}

    # Character Error Rate
    cer = _levenshtein_distance(list(ref_clean), list(hyp_clean)) / len(ref_clean)

    return {
        "similarity": max(0.0, 1.0 - cer),
        "character_error_rate": min(cer, 1.0),
    }


def _sequence_similarity(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate sequence-based similarity using longest common subsequence."""
    if not reference or not hypothesis:
        return {"similarity": 0.0, "lcs_ratio": 0.0}

    lcs_length = _longest_common_subsequence(reference.lower(), hypothesis.lower())
    max_len = max(len(reference), len(hypothesis))

    return {
        "similarity": lcs_length / max_len if max_len else 0.0,
        "lcs_ratio": lcs_length / len(reference) if reference else 0.0,
    }


def _levenshtein_distance(s1: List, s2: List) -> int:
    """Calculate Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _longest_common_subsequence(s1: str, s2: str) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def calculate_extraction_accuracy(
    expected: Dict[str, Any],
    extracted: Dict[str, Any],
    fuzzy_match: bool = True
) -> Dict[str, float]:
    """Calculate extraction accuracy metrics.

    Args:
        expected: Expected/ground truth fields
        extracted: Extracted fields
        fuzzy_match: Use fuzzy matching for string values

    Returns:
        Dict with precision, recall, F1
    """
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    expected_keys = set(expected.keys())
    extracted_keys = set(extracted.keys())

    # Key-level metrics
    key_intersection = expected_keys & extracted_keys
    key_precision = len(key_intersection) / len(extracted_keys) if extracted_keys else 0.0
    key_recall = len(key_intersection) / len(expected_keys) if expected_keys else 0.0

    # Value-level metrics
    correct_values = 0
    total_expected = 0

    for key in expected_keys:
        expected_val = expected[key]
        if expected_val is None:
            continue

        total_expected += 1

        if key in extracted:
            extracted_val = extracted[key]

            if fuzzy_match and isinstance(expected_val, str) and isinstance(extracted_val, str):
                # Fuzzy string matching
                similarity = _token_similarity(expected_val, extracted_val)["similarity"]
                if similarity > 0.8:
                    correct_values += 1
            elif expected_val == extracted_val:
                correct_values += 1

    value_precision = correct_values / len(extracted_keys) if extracted_keys else 0.0
    value_recall = correct_values / total_expected if total_expected else 0.0
    value_f1 = (
        2 * value_precision * value_recall / (value_precision + value_recall)
        if (value_precision + value_recall) > 0 else 0.0
    )

    return {
        "precision": value_precision,
        "recall": value_recall,
        "f1": value_f1,
        "key_precision": key_precision,
        "key_recall": key_recall,
        "correct_values": correct_values,
        "total_expected": total_expected,
    }


def calculate_table_accuracy(
    expected_cells: List[List[str]],
    extracted_cells: List[List[str]]
) -> Dict[str, float]:
    """Calculate table extraction accuracy.

    Args:
        expected_cells: Expected table as 2D list
        extracted_cells: Extracted table as 2D list

    Returns:
        Dict with cell and structure accuracy
    """
    if not expected_cells:
        return {"cell_accuracy": 0.0, "structure_accuracy": 0.0}

    expected_rows = len(expected_cells)
    expected_cols = max(len(row) for row in expected_cells) if expected_cells else 0

    extracted_rows = len(extracted_cells)
    extracted_cols = max(len(row) for row in extracted_cells) if extracted_cells else 0

    # Structure accuracy
    row_match = 1.0 if expected_rows == extracted_rows else 0.0
    col_match = 1.0 if expected_cols == extracted_cols else 0.0
    structure_accuracy = (row_match + col_match) / 2

    # Cell accuracy
    correct_cells = 0
    total_cells = 0

    for i, expected_row in enumerate(expected_cells):
        for j, expected_cell in enumerate(expected_row):
            total_cells += 1

            if i < extracted_rows and j < len(extracted_cells[i]):
                extracted_cell = extracted_cells[i][j]
                # Fuzzy match
                if expected_cell.strip().lower() == extracted_cell.strip().lower():
                    correct_cells += 1
                elif _token_similarity(expected_cell, extracted_cell)["similarity"] > 0.8:
                    correct_cells += 0.5

    cell_accuracy = correct_cells / total_cells if total_cells else 0.0

    return {
        "cell_accuracy": cell_accuracy,
        "structure_accuracy": structure_accuracy,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
    }


def calculate_chunk_quality(
    chunks: List[Any],
    source_content: str
) -> Dict[str, float]:
    """Calculate chunk quality metrics.

    Args:
        chunks: List of chunk objects
        source_content: Original source content

    Returns:
        Dict with coherence and coverage metrics
    """
    if not chunks or not source_content:
        return {"coherence": 0.0, "coverage": 0.0, "avg_chunk_size": 0.0}

    # Calculate coverage
    chunk_texts = []
    for chunk in chunks:
        if hasattr(chunk, 'content'):
            chunk_texts.append(chunk.content)
        elif isinstance(chunk, dict):
            chunk_texts.append(chunk.get('content', chunk.get('text', '')))
        else:
            chunk_texts.append(str(chunk))

    combined_chunks = " ".join(chunk_texts)
    source_tokens = set(source_content.lower().split())
    chunk_tokens = set(combined_chunks.lower().split())

    coverage = len(source_tokens & chunk_tokens) / len(source_tokens) if source_tokens else 0.0

    # Calculate coherence (based on chunk boundaries)
    coherence_scores = []
    for text in chunk_texts:
        # Simple coherence heuristics
        sentences = text.split(".")
        if len(sentences) > 1:
            # Check if chunks end at sentence boundaries
            ends_properly = text.strip().endswith((".", "!", "?", ":"))
            coherence_scores.append(1.0 if ends_properly else 0.5)
        else:
            coherence_scores.append(0.8)

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    # Average chunk size
    avg_size = sum(len(t) for t in chunk_texts) / len(chunk_texts) if chunk_texts else 0.0

    return {
        "coherence": avg_coherence,
        "coverage": coverage,
        "avg_chunk_size": avg_size,
        "num_chunks": len(chunks),
    }


def calculate_classification_accuracy(
    expected_type: str,
    predicted_type: str,
    confidence: float = 1.0
) -> Dict[str, float]:
    """Calculate classification accuracy.

    Args:
        expected_type: Expected document type
        predicted_type: Predicted document type
        confidence: Prediction confidence

    Returns:
        Dict with accuracy metrics
    """
    is_correct = expected_type.lower() == predicted_type.lower()

    return {
        "accuracy": 1.0 if is_correct else 0.0,
        "confidence": confidence,
        "weighted_accuracy": confidence if is_correct else 0.0,
    }
