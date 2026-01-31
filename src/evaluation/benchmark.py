"""Benchmarking framework for document processing performance.

Measures:
- Processing time
- API costs
- Memory usage
- Throughput
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..common.models import ParsedDocument


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Identification
    processor_name: str
    document_path: str
    run_id: str

    # Timing
    start_time: datetime
    end_time: datetime
    total_time_ms: float
    parse_time_ms: Optional[float] = None
    classify_time_ms: Optional[float] = None
    extract_time_ms: Optional[float] = None
    split_time_ms: Optional[float] = None

    # Resource usage
    api_credits: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    # Output metrics
    output_length: int = 0
    chunk_count: int = 0
    page_count: int = 0

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """Benchmark runner for document processing pipelines."""

    # Cost estimates per operation (approximate)
    COST_ESTIMATES = {
        "llamaindex": {
            "parse_per_page": 0.01,  # ~1 cent per page
            "classify": 0.001,
            "extract": 0.005,
            "split": 0.005,
        },
        "landingai": {
            "parse_per_page": 0.008,
            "extract": 0.004,
            "split": 0.003,
        },
        "gemini": {
            "process_per_page": 0.002,
        },
    }

    def __init__(
        self,
        name: str = "benchmark",
        warmup_runs: int = 1,
        benchmark_runs: int = 3
    ):
        """Initialize benchmark.

        Args:
            name: Benchmark name
            warmup_runs: Number of warmup runs (not counted)
            benchmark_runs: Number of benchmark runs to average
        """
        self.name = name
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []

    async def run_single(
        self,
        processor: Any,
        file_path: Union[str, Path],
        **options
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            processor: Document processor instance
            file_path: Path to document
            **options: Options passed to processor

        Returns:
            BenchmarkResult
        """
        file_path = Path(file_path)
        run_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start_time = datetime.now()
        start_ms = time.time() * 1000

        try:
            # Run processing
            result: ParsedDocument = await processor.process(file_path, **options)

            end_ms = time.time() * 1000
            end_time = datetime.now()

            # Calculate metrics
            total_time = end_ms - start_ms

            benchmark_result = BenchmarkResult(
                processor_name=processor.name,
                document_path=str(file_path),
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                total_time_ms=total_time,
                parse_time_ms=result.metadata.processing_time_ms,
                api_credits=result.metadata.api_credits_used,
                output_length=len(result.markdown),
                chunk_count=len(result.chunks),
                page_count=result.metadata.page_count,
                success=True,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "file_type": file_path.suffix,
                    "has_tables": result.metadata.has_tables,
                    "has_images": result.metadata.has_images,
                }
            )

            # Estimate cost
            benchmark_result.estimated_cost_usd = self._estimate_cost(
                processor.name,
                result.metadata.page_count
            )

            return benchmark_result

        except Exception as e:
            end_ms = time.time() * 1000
            end_time = datetime.now()

            return BenchmarkResult(
                processor_name=processor.name,
                document_path=str(file_path),
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                total_time_ms=end_ms - start_ms,
                success=False,
                error_message=str(e),
            )

    async def run_batch(
        self,
        processor: Any,
        file_paths: List[Union[str, Path]],
        **options
    ) -> List[BenchmarkResult]:
        """Run benchmark on multiple documents.

        Args:
            processor: Document processor instance
            file_paths: List of document paths
            **options: Options passed to processor

        Returns:
            List of BenchmarkResults
        """
        results = []

        for path in file_paths:
            # Warmup runs
            for _ in range(self.warmup_runs):
                await self.run_single(processor, path, **options)

            # Benchmark runs
            run_results = []
            for _ in range(self.benchmark_runs):
                result = await self.run_single(processor, path, **options)
                run_results.append(result)

            # Average the results
            avg_result = self._average_results(run_results)
            results.append(avg_result)
            self.results.append(avg_result)

        return results

    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results."""
        if len(results) == 1:
            return results[0]

        successful = [r for r in results if r.success]
        if not successful:
            return results[0]

        avg_time = sum(r.total_time_ms for r in successful) / len(successful)

        # Take the first result as template
        avg_result = BenchmarkResult(
            processor_name=results[0].processor_name,
            document_path=results[0].document_path,
            run_id=f"{results[0].run_id}_avg",
            start_time=results[0].start_time,
            end_time=results[-1].end_time,
            total_time_ms=avg_time,
            output_length=results[0].output_length,
            chunk_count=results[0].chunk_count,
            page_count=results[0].page_count,
            success=True,
            metadata={
                "num_runs": len(successful),
                "min_time_ms": min(r.total_time_ms for r in successful),
                "max_time_ms": max(r.total_time_ms for r in successful),
            }
        )

        return avg_result

    def _estimate_cost(self, processor_name: str, page_count: int) -> float:
        """Estimate processing cost in USD."""
        costs = self.COST_ESTIMATES.get(processor_name, {})

        if processor_name == "llamaindex":
            return (
                costs.get("parse_per_page", 0) * page_count +
                costs.get("classify", 0) +
                costs.get("extract", 0) +
                costs.get("split", 0)
            )
        elif processor_name == "landingai":
            return (
                costs.get("parse_per_page", 0) * page_count +
                costs.get("extract", 0) +
                costs.get("split", 0)
            )
        elif processor_name == "gemini":
            return costs.get("process_per_page", 0) * page_count

        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        if not self.results:
            return {"error": "No benchmark results"}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        by_processor: Dict[str, List[BenchmarkResult]] = {}
        for r in successful:
            if r.processor_name not in by_processor:
                by_processor[r.processor_name] = []
            by_processor[r.processor_name].append(r)

        summary = {
            "total_runs": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "processors": {},
        }

        for proc_name, proc_results in by_processor.items():
            times = [r.total_time_ms for r in proc_results]
            summary["processors"][proc_name] = {
                "runs": len(proc_results),
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "total_pages": sum(r.page_count for r in proc_results),
                "avg_pages_per_doc": sum(r.page_count for r in proc_results) / len(proc_results),
                "estimated_cost": sum(r.estimated_cost_usd or 0 for r in proc_results),
            }

        return summary

    def export_results(self, format: str = "dict") -> Any:
        """Export benchmark results.

        Args:
            format: Export format ("dict", "csv", "json")

        Returns:
            Exported results
        """
        if format == "dict":
            return [
                {
                    "processor": r.processor_name,
                    "document": r.document_path,
                    "time_ms": r.total_time_ms,
                    "pages": r.page_count,
                    "chunks": r.chunk_count,
                    "success": r.success,
                    "cost_usd": r.estimated_cost_usd,
                }
                for r in self.results
            ]

        elif format == "csv":
            lines = ["processor,document,time_ms,pages,chunks,success,cost_usd"]
            for r in self.results:
                lines.append(
                    f"{r.processor_name},{r.document_path},{r.total_time_ms},"
                    f"{r.page_count},{r.chunk_count},{r.success},{r.estimated_cost_usd}"
                )
            return "\n".join(lines)

        return self.results
