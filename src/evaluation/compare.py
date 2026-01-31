"""Comparison framework for evaluating multiple processing stacks.

Enables side-by-side comparison of LlamaIndex and LandingAI stacks.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..common.models import ParsedDocument, DocumentType
from ..common.interfaces import ClassificationRule
from .metrics import (
    EvaluationMetrics,
    calculate_text_similarity,
    calculate_extraction_accuracy,
    calculate_chunk_quality,
    calculate_classification_accuracy,
)
from .benchmark import Benchmark, BenchmarkResult


@dataclass
class DocumentResult:
    """Results for a single document from one processor."""
    document_path: str
    processor_name: str
    parsed_document: Optional[ParsedDocument]
    benchmark: Optional[BenchmarkResult]
    metrics: Optional[EvaluationMetrics]
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison results for a single document across processors."""
    document_path: str
    document_name: str
    results: Dict[str, DocumentResult] = field(default_factory=dict)
    winner: Optional[str] = None
    comparison_notes: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Full comparison report across all documents and processors."""
    documents: List[ComparisonResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class StackComparator:
    """Compare document processing stacks side by side."""

    def __init__(
        self,
        processors: Dict[str, Any],
        ground_truth: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize comparator.

        Args:
            processors: Dict of processor name -> processor instance
            ground_truth: Optional ground truth data for evaluation
                Format: {document_path: {expected_fields}}
        """
        self.processors = processors
        self.ground_truth = ground_truth or {}
        self.benchmark = Benchmark(warmup_runs=0, benchmark_runs=1)

    async def compare_document(
        self,
        file_path: Union[str, Path],
        schema: Optional[Type[BaseModel]] = None,
        expected_type: Optional[DocumentType] = None,
        **options
    ) -> ComparisonResult:
        """Compare processors on a single document.

        Args:
            file_path: Path to document
            schema: Optional extraction schema
            expected_type: Optional expected document type for classification eval
            **options: Options passed to processors

        Returns:
            ComparisonResult with results from all processors
        """
        file_path = Path(file_path)
        comparison = ComparisonResult(
            document_path=str(file_path),
            document_name=file_path.name,
        )

        # Get ground truth if available
        ground_truth = self.ground_truth.get(str(file_path), {})

        # Run each processor
        for name, processor in self.processors.items():
            try:
                # Benchmark the processing
                benchmark_result = await self.benchmark.run_single(
                    processor, file_path, schema=schema, **options
                )

                # Get the parsed document
                parsed = await processor.process(file_path, schema=schema, **options)

                # Calculate metrics
                metrics = self._calculate_metrics(
                    parsed, ground_truth, expected_type
                )

                doc_result = DocumentResult(
                    document_path=str(file_path),
                    processor_name=name,
                    parsed_document=parsed,
                    benchmark=benchmark_result,
                    metrics=metrics,
                )

            except Exception as e:
                doc_result = DocumentResult(
                    document_path=str(file_path),
                    processor_name=name,
                    parsed_document=None,
                    benchmark=None,
                    metrics=None,
                    error=str(e),
                )

            comparison.results[name] = doc_result

        # Determine winner
        comparison.winner = self._determine_winner(comparison)
        comparison.comparison_notes = self._generate_notes(comparison)

        return comparison

    async def compare_batch(
        self,
        file_paths: List[Union[str, Path]],
        **options
    ) -> ComparisonReport:
        """Compare processors on multiple documents.

        Args:
            file_paths: List of document paths
            **options: Options passed to processors

        Returns:
            Full comparison report
        """
        report = ComparisonReport()

        for path in file_paths:
            comparison = await self.compare_document(path, **options)
            report.documents.append(comparison)

        # Generate summary
        report.summary = self._generate_summary(report.documents)
        report.recommendations = self._generate_recommendations(report)

        return report

    def _calculate_metrics(
        self,
        parsed: ParsedDocument,
        ground_truth: Dict[str, Any],
        expected_type: Optional[DocumentType]
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics for a parsed document."""
        metrics = EvaluationMetrics(
            processing_time_ms=parsed.metadata.processing_time_ms,
            api_credits_used=parsed.metadata.api_credits_used or 0.0,
        )

        # Text metrics
        if "text" in ground_truth:
            text_metrics = calculate_text_similarity(
                ground_truth["text"],
                parsed.markdown
            )
            metrics.text_similarity = text_metrics.get("similarity", 0.0)
            metrics.word_error_rate = text_metrics.get("word_error_rate", 1.0)

        # Extraction metrics
        if "fields" in ground_truth and parsed.extraction:
            extraction_metrics = calculate_extraction_accuracy(
                ground_truth["fields"],
                parsed.extraction.fields
            )
            metrics.extraction_precision = extraction_metrics.get("precision", 0.0)
            metrics.extraction_recall = extraction_metrics.get("recall", 0.0)
            metrics.extraction_f1 = extraction_metrics.get("f1", 0.0)

        # Classification metrics
        if expected_type and parsed.classification:
            class_metrics = calculate_classification_accuracy(
                expected_type.value,
                parsed.classification.document_type.value,
                parsed.classification.confidence
            )
            metrics.classification_accuracy = class_metrics.get("accuracy", 0.0)
            metrics.classification_confidence = class_metrics.get("confidence", 0.0)

        # Chunk metrics
        if parsed.chunks:
            chunk_metrics = calculate_chunk_quality(
                parsed.chunks,
                parsed.markdown
            )
            metrics.chunk_coherence = chunk_metrics.get("coherence", 0.0)
            metrics.chunk_coverage = chunk_metrics.get("coverage", 0.0)

        return metrics

    def _determine_winner(self, comparison: ComparisonResult) -> Optional[str]:
        """Determine which processor performed best."""
        scores: Dict[str, float] = {}

        for name, result in comparison.results.items():
            if result.error:
                scores[name] = 0.0
                continue

            if not result.metrics:
                scores[name] = 0.5
                continue

            # Composite score (weighted average)
            m = result.metrics
            score = (
                m.text_similarity * 0.3 +
                m.extraction_f1 * 0.25 +
                m.classification_accuracy * 0.15 +
                m.chunk_coherence * 0.15 +
                (1.0 - min(m.processing_time_ms / 30000, 1.0)) * 0.15  # Speed bonus
            )
            scores[name] = score

        if not scores:
            return None

        return max(scores, key=scores.get)

    def _generate_notes(self, comparison: ComparisonResult) -> List[str]:
        """Generate comparison notes."""
        notes = []

        results = list(comparison.results.values())
        if len(results) < 2:
            return notes

        # Compare processing times
        times = {
            r.processor_name: r.benchmark.total_time_ms
            for r in results if r.benchmark
        }
        if times:
            fastest = min(times, key=times.get)
            slowest = max(times, key=times.get)
            if times[fastest] > 0:
                speedup = times[slowest] / times[fastest]
                notes.append(f"{fastest} was {speedup:.1f}x faster than {slowest}")

        # Compare output lengths
        lengths = {
            r.processor_name: len(r.parsed_document.markdown)
            for r in results if r.parsed_document
        }
        if lengths:
            max_len = max(lengths.values())
            min_len = min(lengths.values())
            if min_len > 0 and max_len / min_len > 1.5:
                notes.append(f"Output length varied significantly: {min_len}-{max_len} chars")

        # Compare chunk counts
        chunks = {
            r.processor_name: len(r.parsed_document.chunks)
            for r in results if r.parsed_document
        }
        if chunks and len(set(chunks.values())) > 1:
            notes.append(f"Chunk counts: {dict(chunks)}")

        # Note errors
        errors = [r.processor_name for r in results if r.error]
        if errors:
            notes.append(f"Errors from: {', '.join(errors)}")

        return notes

    def _generate_summary(
        self,
        comparisons: List[ComparisonResult]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_documents": len(comparisons),
            "processors": {},
            "wins": {},
        }

        # Count wins
        for comp in comparisons:
            if comp.winner:
                summary["wins"][comp.winner] = summary["wins"].get(comp.winner, 0) + 1

        # Aggregate metrics by processor
        for proc_name in self.processors.keys():
            proc_results = [
                c.results.get(proc_name)
                for c in comparisons
                if c.results.get(proc_name) and not c.results[proc_name].error
            ]

            if not proc_results:
                continue

            times = [r.benchmark.total_time_ms for r in proc_results if r.benchmark]
            metrics_list = [r.metrics for r in proc_results if r.metrics]

            summary["processors"][proc_name] = {
                "successful": len(proc_results),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "avg_text_similarity": (
                    sum(m.text_similarity for m in metrics_list) / len(metrics_list)
                    if metrics_list else 0
                ),
                "avg_extraction_f1": (
                    sum(m.extraction_f1 for m in metrics_list) / len(metrics_list)
                    if metrics_list else 0
                ),
                "avg_chunk_coherence": (
                    sum(m.chunk_coherence for m in metrics_list) / len(metrics_list)
                    if metrics_list else 0
                ),
            }

        return summary

    def _generate_recommendations(self, report: ComparisonReport) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        summary = report.summary

        wins = summary.get("wins", {})
        if wins:
            best = max(wins, key=wins.get)
            recommendations.append(
                f"{best} performed best overall ({wins[best]}/{summary['total_documents']} documents)"
            )

        processors = summary.get("processors", {})
        if len(processors) >= 2:
            # Speed comparison
            times = {k: v["avg_time_ms"] for k, v in processors.items()}
            fastest = min(times, key=times.get)
            recommendations.append(f"Use {fastest} for speed-critical applications")

            # Quality comparison
            f1_scores = {k: v.get("avg_extraction_f1", 0) for k, v in processors.items()}
            if any(f1_scores.values()):
                best_extraction = max(f1_scores, key=f1_scores.get)
                recommendations.append(
                    f"Use {best_extraction} for extraction-heavy workflows"
                )

        return recommendations

    def print_report(self, report: ComparisonReport):
        """Print a formatted comparison report."""
        print("\n" + "=" * 60)
        print("DOCUMENT PROCESSING COMPARISON REPORT")
        print("=" * 60)

        print(f"\nDocuments analyzed: {report.summary['total_documents']}")
        print(f"Processors compared: {', '.join(self.processors.keys())}")

        print("\n--- WINS ---")
        for proc, wins in report.summary.get("wins", {}).items():
            print(f"  {proc}: {wins} documents")

        print("\n--- PROCESSOR PERFORMANCE ---")
        for proc, stats in report.summary.get("processors", {}).items():
            print(f"\n  {proc}:")
            print(f"    Successful: {stats['successful']}")
            print(f"    Avg time: {stats['avg_time_ms']:.0f}ms")
            print(f"    Avg text similarity: {stats['avg_text_similarity']:.2%}")
            print(f"    Avg extraction F1: {stats['avg_extraction_f1']:.2%}")
            print(f"    Avg chunk coherence: {stats['avg_chunk_coherence']:.2%}")

        print("\n--- RECOMMENDATIONS ---")
        for rec in report.recommendations:
            print(f"  • {rec}")

        print("\n" + "=" * 60)
