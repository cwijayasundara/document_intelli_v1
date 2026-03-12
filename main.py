"""Main entry point for document extraction pipeline.

This module provides high-level functions for:
- Processing documents with either LlamaIndex or LandingAI stack
- Running comparative evaluations
- Batch processing

Example usage:
    from main import process_document, compare_stacks

    # Process a single document
    result = await process_document("document.pdf", stack="llamaindex")

    # Compare stacks
    report = await compare_stacks(["doc1.pdf", "doc2.pdf"])
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from src.common.models import ParsedDocument, DocumentType
from src.common.router import DocumentRouter, ProcessorType
from src.common.interfaces import ClassificationRule
from src.llamaindex_stack import LlamaIndexProcessor
from src.landingai_stack import LandingAIProcessor
from src.reducto_stack import ReductoProcessor
from src.gemini import GeminiHandwritingProcessor
from src.evaluation import StackComparator, ComparisonReport, Benchmark


async def process_document(
    file_path: Union[str, Path],
    stack: str = "llamaindex",
    schema: Optional[Type[BaseModel]] = None,
    **options
) -> ParsedDocument:
    """Process a document using the specified stack.

    Args:
        file_path: Path to the document
        stack: Processing stack ("llamaindex", "landingai", or "auto")
        schema: Optional Pydantic schema for extraction
        **options: Additional options passed to the processor

    Returns:
        ParsedDocument with all processing results

    Raises:
        ValueError: If stack is invalid or API key is missing
        FileNotFoundError: If document doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Auto-routing
    if stack == "auto":
        router = DocumentRouter()
        processor_type, route_options = router.route(file_path)
        stack = processor_type.value
        options.update(route_options)

    # Select processor
    if stack == "llamaindex":
        processor = LlamaIndexProcessor()
    elif stack == "landingai":
        processor = LandingAIProcessor()
    elif stack == "reducto":
        processor = ReductoProcessor()
    elif stack == "gemini":
        # Use Gemini for handwriting
        gemini = GeminiHandwritingProcessor()
        result = await gemini.process_handwriting(file_path, **options)
        # Convert to ParsedDocument format
        from src.common.models import DocumentMetadata
        return ParsedDocument(
            markdown=result["text"],
            metadata=DocumentMetadata(
                source_path=str(file_path),
                file_name=file_path.name,
                file_type=file_path.suffix,
                processor="gemini",
                has_handwriting=True,
            )
        )
    else:
        raise ValueError(f"Unknown stack: {stack}. Use 'llamaindex', 'landingai', 'reducto', or 'auto'")

    return await processor.process(file_path, schema=schema, **options)


async def process_batch(
    file_paths: List[Union[str, Path]],
    stack: str = "llamaindex",
    **options
) -> List[ParsedDocument]:
    """Process multiple documents.

    Args:
        file_paths: List of document paths
        stack: Processing stack
        **options: Additional options

    Returns:
        List of ParsedDocument results
    """
    results = []
    for path in file_paths:
        try:
            result = await process_document(path, stack=stack, **options)
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Create error result
            from src.common.models import DocumentMetadata
            error_result = ParsedDocument(
                markdown="",
                metadata=DocumentMetadata(
                    source_path=str(path),
                    file_name=Path(path).name,
                    file_type=Path(path).suffix,
                    processor=stack,
                    errors=[str(e)],
                )
            )
            results.append(error_result)

    return results


async def compare_stacks(
    file_paths: List[Union[str, Path]],
    stacks: Optional[List[str]] = None,
    **options
) -> ComparisonReport:
    """Compare processing stacks on multiple documents.

    Args:
        file_paths: List of document paths
        stacks: Stacks to compare (default: ["llamaindex", "landingai"])
        **options: Additional options

    Returns:
        ComparisonReport with detailed comparison
    """
    if stacks is None:
        stacks = ["llamaindex", "landingai", "reducto"]

    processors = {}
    for stack in stacks:
        if stack == "llamaindex":
            processors[stack] = LlamaIndexProcessor()
        elif stack == "landingai":
            processors[stack] = LandingAIProcessor()
        elif stack == "reducto":
            processors[stack] = ReductoProcessor()

    comparator = StackComparator(processors)
    report = await comparator.compare_batch(file_paths, **options)

    return report


async def run_benchmark(
    file_paths: List[Union[str, Path]],
    stack: str = "llamaindex",
    **options
) -> Dict[str, Any]:
    """Run performance benchmark on documents.

    Args:
        file_paths: List of document paths
        stack: Stack to benchmark
        **options: Additional options

    Returns:
        Benchmark summary
    """
    if stack == "llamaindex":
        processor = LlamaIndexProcessor()
    elif stack == "landingai":
        processor = LandingAIProcessor()
    elif stack == "reducto":
        processor = ReductoProcessor()
    else:
        raise ValueError(f"Unknown stack: {stack}")

    benchmark = Benchmark(name=f"{stack}_benchmark")
    await benchmark.run_batch(processor, file_paths, **options)

    return benchmark.get_summary()


def get_test_documents() -> List[Path]:
    """Get list of test documents from difficult_examples folder.

    Returns:
        List of document paths
    """
    examples_dir = Path(__file__).parent / "difficult_examples"

    if not examples_dir.exists():
        return []

    # Supported extensions
    extensions = {".pdf", ".png", ".jpg", ".jpeg"}

    documents = [
        f for f in examples_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    return sorted(documents)


async def demo():
    """Run a demo of the document extraction pipeline."""
    print("=" * 60)
    print("Document Extraction Pipeline Demo")
    print("=" * 60)

    # Check for API keys
    llama_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    landing_key = os.environ.get("LANDINGAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    reducto_key = os.environ.get("REDUCTO_API_KEY")

    print("\nAPI Key Status:")
    print(f"  LLAMA_CLOUD_API_KEY: {'Set' if llama_key else 'Not set'}")
    print(f"  LANDINGAI_API_KEY: {'Set' if landing_key else 'Not set'}")
    print(f"  GOOGLE_API_KEY: {'Set' if google_key else 'Not set'}")
    print(f"  REDUCTO_API_KEY: {'Set' if reducto_key else 'Not set'}")

    # Get test documents
    documents = get_test_documents()
    print(f"\nTest documents found: {len(documents)}")
    for doc in documents:
        print(f"  - {doc.name}")

    if not documents:
        print("\nNo test documents found in difficult_examples/")
        return

    # Demo with first document
    if documents and (llama_key or landing_key):
        print("\n--- Processing Demo ---")
        test_doc = documents[0]
        print(f"Processing: {test_doc.name}")

        if llama_key:
            print("\nUsing LlamaIndex stack...")
            try:
                result = await process_document(test_doc, stack="llamaindex")
                print(f"  Markdown length: {len(result.markdown)} chars")
                print(f"  Chunks: {result.total_chunks}")
                print(f"  Processing time: {result.metadata.processing_time_ms:.0f}ms")
                if result.classification:
                    print(f"  Classification: {result.classification.document_type.value}")
            except Exception as e:
                print(f"  Error: {e}")

        if landing_key:
            print("\nUsing LandingAI stack...")
            try:
                result = await process_document(test_doc, stack="landingai")
                print(f"  Markdown length: {len(result.markdown)} chars")
                print(f"  Chunks: {result.total_chunks}")
                print(f"  Processing time: {result.metadata.processing_time_ms:.0f}ms")
                print(f"  Has grounding: {result.has_grounding}")
            except Exception as e:
                print(f"  Error: {e}")

        if reducto_key:
            print("\nUsing Reducto stack...")
            try:
                result = await process_document(test_doc, stack="reducto")
                print(f"  Markdown length: {len(result.markdown)} chars")
                print(f"  Chunks: {result.total_chunks}")
                print(f"  Processing time: {result.metadata.processing_time_ms:.0f}ms")
                print(f"  Has grounding: {result.has_grounding}")
                if result.metadata.api_credits_used:
                    print(f"  Credits used: {result.metadata.api_credits_used}")
            except Exception as e:
                print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
