"""Evaluation framework for comparing document processing stacks.

Provides metrics, benchmarks, and comparison tools.
"""

from .metrics import (
    EvaluationMetrics,
    calculate_text_similarity,
    calculate_extraction_accuracy,
    calculate_chunk_quality,
)
from .benchmark import Benchmark, BenchmarkResult
from .compare import StackComparator, ComparisonReport

__all__ = [
    "EvaluationMetrics",
    "calculate_text_similarity",
    "calculate_extraction_accuracy",
    "calculate_chunk_quality",
    "Benchmark",
    "BenchmarkResult",
    "StackComparator",
    "ComparisonReport",
]
