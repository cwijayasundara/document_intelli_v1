"""Evaluation script for comparing document processing stacks.

This script runs both LlamaIndex and LandingAI stacks on the test documents
and generates a comparison report.

Usage:
    python notebooks/evaluation.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.common.models import DocumentType
from src.common.router import DocumentRouter, ProcessorType
from src.llamaindex_stack import LlamaIndexProcessor
from src.landingai_stack import LandingAIProcessor
from src.gemini import GeminiHandwritingProcessor
from src.evaluation import StackComparator, Benchmark


# Test document ground truth (partial - for evaluation)
GROUND_TRUTH = {
    "certificate_of_origin.pdf": {
        "expected_type": DocumentType.CERTIFICATE,
        "fields": {
            "certificate_type": "Certificate of Origin",
        }
    },
    "patient_intake.pdf": {
        "expected_type": DocumentType.MEDICAL,
        "fields": {}
    },
    "ikea-assembly.pdf": {
        "expected_type": DocumentType.INSTRUCTIONS,
        "fields": {}
    },
    "Investor_Presentation_pg7.png": {
        "expected_type": DocumentType.PRESENTATION,
        "fields": {}
    },
    "hr_process_flowchart.png": {
        "expected_type": DocumentType.FLOWCHART,
        "fields": {}
    },
    "sales_volume.png": {
        "expected_type": DocumentType.SPREADSHEET,
        "fields": {}
    },
    "calculus_BC_answer_sheet.jpg": {
        "expected_type": DocumentType.HANDWRITTEN,
        "fields": {}
    },
    "ikea_infographic.jpg": {
        "expected_type": DocumentType.INFOGRAPHIC,
        "fields": {}
    },
}


def get_test_documents():
    """Get list of test documents."""
    examples_dir = Path(__file__).parent.parent / "difficult_examples"

    if not examples_dir.exists():
        print(f"Error: {examples_dir} not found")
        return []

    extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    documents = [
        f for f in examples_dir.iterdir()
        if f.suffix.lower() in extensions and not f.name.startswith(".")
    ]

    return sorted(documents)


async def evaluate_llamaindex(documents: list):
    """Evaluate LlamaIndex stack."""
    print("\n" + "=" * 50)
    print("EVALUATING LLAMAINDEX STACK")
    print("=" * 50)

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("LLAMA_CLOUD_API_KEY not set - skipping")
        return None

    processor = LlamaIndexProcessor()
    results = []

    for doc in documents:
        print(f"\nProcessing: {doc.name}")
        try:
            result = await processor.process(doc)
            print(f"  ✓ Success - {len(result.markdown)} chars, {result.total_chunks} chunks")
            if result.classification:
                print(f"    Type: {result.classification.document_type.value} ({result.classification.confidence:.2%})")
            results.append({
                "document": doc.name,
                "success": True,
                "markdown_length": len(result.markdown),
                "chunks": result.total_chunks,
                "time_ms": result.metadata.processing_time_ms,
                "classification": result.classification.document_type.value if result.classification else None,
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "document": doc.name,
                "success": False,
                "error": str(e),
            })

    return results


async def evaluate_landingai(documents: list):
    """Evaluate LandingAI stack."""
    print("\n" + "=" * 50)
    print("EVALUATING LANDINGAI STACK")
    print("=" * 50)

    api_key = os.environ.get("LANDINGAI_API_KEY")
    if not api_key:
        print("LANDINGAI_API_KEY not set - skipping")
        return None

    processor = LandingAIProcessor()
    results = []

    for doc in documents:
        print(f"\nProcessing: {doc.name}")
        try:
            result = await processor.process(doc)
            print(f"  ✓ Success - {len(result.markdown)} chars, {result.total_chunks} chunks")
            print(f"    Grounding: {'Yes' if result.has_grounding else 'No'}")
            results.append({
                "document": doc.name,
                "success": True,
                "markdown_length": len(result.markdown),
                "chunks": result.total_chunks,
                "time_ms": result.metadata.processing_time_ms,
                "has_grounding": result.has_grounding,
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "document": doc.name,
                "success": False,
                "error": str(e),
            })

    await processor.close()
    return results


async def evaluate_gemini_handwriting(documents: list):
    """Evaluate Gemini for handwritten documents."""
    print("\n" + "=" * 50)
    print("EVALUATING GEMINI HANDWRITING")
    print("=" * 50)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set - skipping")
        return None

    # Only process handwritten documents
    handwriting_docs = [
        d for d in documents
        if "answer" in d.name.lower() or "handwritten" in d.name.lower()
    ]

    if not handwriting_docs:
        print("No handwritten documents found")
        return None

    processor = GeminiHandwritingProcessor()
    results = []

    for doc in handwriting_docs:
        print(f"\nProcessing: {doc.name}")
        try:
            # First detect handwriting
            detection = await processor.detect_handwriting(doc)
            print(f"  Handwriting detected: {detection['has_handwriting']}")
            print(f"  Percentage: {detection['handwriting_percentage']:.0%}")

            # Process handwriting
            result = await processor.process_handwriting(doc, mode="math")
            print(f"  ✓ Extracted {len(result['text'])} chars")
            results.append({
                "document": doc.name,
                "success": True,
                "text_length": len(result["text"]),
                "has_handwriting": detection["has_handwriting"],
                "handwriting_percentage": detection["handwriting_percentage"],
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "document": doc.name,
                "success": False,
                "error": str(e),
            })

    return results


async def run_comparison(documents: list):
    """Run side-by-side comparison."""
    print("\n" + "=" * 50)
    print("RUNNING SIDE-BY-SIDE COMPARISON")
    print("=" * 50)

    processors = {}

    if os.environ.get("LLAMA_CLOUD_API_KEY"):
        processors["llamaindex"] = LlamaIndexProcessor()

    if os.environ.get("LANDINGAI_API_KEY"):
        processors["landingai"] = LandingAIProcessor()

    if len(processors) < 2:
        print("Need at least 2 API keys for comparison")
        return None

    comparator = StackComparator(processors)

    # Compare on subset of documents
    test_docs = documents[:3]  # First 3 for quick comparison

    print(f"\nComparing on {len(test_docs)} documents...")
    report = await comparator.compare_batch(test_docs)

    comparator.print_report(report)

    return report


def save_results(results: dict, output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    output_file = output_dir / f"evaluation_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


async def main():
    """Main evaluation entry point."""
    print("=" * 60)
    print("DOCUMENT EXTRACTION PIPELINE EVALUATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Get test documents
    documents = get_test_documents()
    print(f"\nFound {len(documents)} test documents:")
    for doc in documents:
        print(f"  - {doc.name}")

    if not documents:
        print("No test documents found!")
        return

    results = {
        "timestamp": datetime.now().isoformat(),
        "documents_tested": [d.name for d in documents],
        "llamaindex": None,
        "landingai": None,
        "gemini": None,
        "comparison": None,
    }

    # Run individual evaluations
    results["llamaindex"] = await evaluate_llamaindex(documents)
    results["landingai"] = await evaluate_landingai(documents)
    results["gemini"] = await evaluate_gemini_handwriting(documents)

    # Run comparison
    results["comparison"] = await run_comparison(documents)

    # Save results
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
