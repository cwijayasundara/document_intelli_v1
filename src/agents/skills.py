"""Document processing skills/tools for the LangChain agent.

These tools provide document AI capabilities:
- Parsing: Convert documents to markdown
- Extraction: Extract structured data using schemas
- Splitting: Chunk documents semantically
- Classification: Identify document types
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# Import our document processing components
from ..common.models import DocumentType
from ..common.interfaces import ClassificationRule


# Global processor instances (lazy initialized)
_llama_processor = None
_landing_processor = None


def _get_llama_processor():
    """Get or create LlamaIndex processor instance."""
    global _llama_processor
    if _llama_processor is None:
        from ..llamaindex_stack import LlamaIndexProcessor
        _llama_processor = LlamaIndexProcessor()
    return _llama_processor


def _get_landing_processor():
    """Get or create LandingAI processor instance."""
    global _landing_processor
    if _landing_processor is None:
        from ..landingai_stack import LandingAIProcessor
        _landing_processor = LandingAIProcessor()
    return _landing_processor


@tool
def parse_document(
    file_path: str,
    processor: str = "llamaindex",
    tier: str = "agentic",
    multimodal: bool = False
) -> str:
    """Parse a document and convert it to markdown format.

    This tool takes a document file (PDF, image, etc.) and extracts its content
    as structured markdown text. It handles tables, images, and complex layouts.

    Args:
        file_path: The path to the document file to parse.
                   Supported formats: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
        processor: Which processor to use - "llamaindex" or "landingai".
                   Default is "llamaindex".
        tier: Parsing tier for LlamaIndex - "cost_effective", "agentic",
              "agentic_plus", or "fast". Default is "agentic".
        multimodal: Enable multimodal mode for visual documents.
                    Uses more credits but better for charts/diagrams.

    Returns:
        The parsed document content as markdown text.

    Example:
        >>> result = parse_document("/path/to/invoice.pdf")
        >>> print(result[:500])
    """
    import asyncio

    file_path = Path(file_path)
    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    try:
        if processor == "llamaindex":
            proc = _get_llama_processor()
            from ..llamaindex_stack.parser import ParseTier
            tier_enum = ParseTier(tier)

            # Run async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()

            result = asyncio.run(proc.parser.parse(
                file_path=file_path,
                tier=tier_enum,
                multimodal=multimodal
            ))
            return result["markdown"]

        elif processor == "landingai":
            proc = _get_landing_processor()
            result = asyncio.run(proc.parser.parse(file_path=file_path))
            return result["markdown"]

        else:
            return f"Error: Unknown processor '{processor}'. Use 'llamaindex' or 'landingai'."

    except Exception as e:
        return f"Error parsing document: {str(e)}"


@tool
def extract_from_document(
    content: str,
    fields: str,
    processor: str = "llamaindex"
) -> str:
    """Extract structured data from document content using specified fields.

    This tool extracts key-value pairs from document text based on the fields
    you want to extract. It uses AI to understand the document and find the
    relevant information.

    Args:
        content: The document content (markdown or plain text) to extract from.
                 This is typically the output from parse_document.
        fields: A JSON string or comma-separated list of field names to extract.
                Examples:
                - "invoice_number, total, vendor_name, date"
                - '{"invoice_number": "string", "total": "number"}'
        processor: Which processor to use - "llamaindex" or "landingai".
                   Default is "llamaindex".

    Returns:
        A JSON string containing the extracted fields and their values.

    Example:
        >>> content = parse_document("invoice.pdf")
        >>> result = extract_from_document(content, "invoice_number, total, vendor_name")
        >>> print(result)
        {"invoice_number": "INV-001", "total": 150.00, "vendor_name": "Acme Corp"}
    """
    import asyncio

    if not content:
        return json.dumps({"error": "No content provided"})

    try:
        # Parse fields specification
        if fields.startswith("{"):
            # JSON schema provided
            field_schema = json.loads(fields)
            field_names = list(field_schema.keys())
        else:
            # Comma-separated list
            field_names = [f.strip() for f in fields.split(",") if f.strip()]

        if not field_names:
            return json.dumps({"error": "No fields specified"})

        # Build extraction schema
        from pydantic import create_model
        field_definitions = {name: (Optional[str], None) for name in field_names}
        DynamicSchema = create_model("DynamicExtraction", **field_definitions)

        if processor == "llamaindex":
            proc = _get_llama_processor()
            result = asyncio.run(proc.extractor.extract(
                content=content,
                schema=DynamicSchema
            ))
            return json.dumps(result.fields, indent=2)

        elif processor == "landingai":
            # LandingAI uses JSON schema directly
            json_schema = {
                "type": "object",
                "properties": {
                    name: {"type": "string", "description": f"Extract {name}"}
                    for name in field_names
                }
            }
            proc = _get_landing_processor()
            result = asyncio.run(proc.extractor.extract_with_json_schema(
                content=content,
                json_schema=json_schema
            ))
            return json.dumps(result.fields, indent=2)

        else:
            return json.dumps({"error": f"Unknown processor '{processor}'"})

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def split_document(
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 100,
    categories: str = ""
) -> str:
    """Split document content into semantic chunks for RAG applications.

    This tool divides document content into meaningful chunks that can be
    used for retrieval-augmented generation (RAG), vector storage, or
    other processing that requires smaller text segments.

    Args:
        content: The document content (markdown or plain text) to split.
                 This is typically the output from parse_document.
        max_chunk_size: Maximum size of each chunk in characters.
                        Default is 2000 characters.
        overlap: Number of characters to overlap between chunks.
                 Helps maintain context across chunk boundaries.
                 Default is 100 characters.
        categories: Optional comma-separated list of category labels for
                    classifying chunks. Example: "introduction,body,conclusion"

    Returns:
        A JSON string containing an array of chunks, each with:
        - id: Unique chunk identifier
        - content: The chunk text
        - category: Assigned category (if categories provided)
        - metadata: Additional chunk metadata

    Example:
        >>> content = parse_document("report.pdf")
        >>> chunks = split_document(content, max_chunk_size=1000)
        >>> chunks_data = json.loads(chunks)
        >>> print(f"Created {len(chunks_data)} chunks")
    """
    import asyncio

    if not content:
        return json.dumps({"error": "No content provided", "chunks": []})

    try:
        # Parse categories if provided
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(",") if c.strip()]

        proc = _get_llama_processor()
        chunks = asyncio.run(proc.splitter.split(
            content=content,
            categories=category_list,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        ))

        # Convert to serializable format
        chunks_data = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "category": chunk.category,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        return json.dumps({
            "total_chunks": len(chunks_data),
            "chunks": chunks_data
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "chunks": []})


@tool
def classify_document(
    content: str,
    custom_categories: str = ""
) -> str:
    """Classify a document into a category based on its content.

    This tool analyzes document content and determines what type of document
    it is (invoice, form, certificate, report, etc.). It returns the
    classification with a confidence score.

    Args:
        content: The document content (markdown or plain text) to classify.
                 This is typically the output from parse_document.
        custom_categories: Optional comma-separated list of custom categories.
                          If not provided, uses default categories:
                          form, invoice, receipt, certificate, medical,
                          presentation, diagram, flowchart, spreadsheet,
                          instructions, infographic, handwritten, report

    Returns:
        A JSON string containing:
        - document_type: The primary document classification
        - confidence: Confidence score (0-1)
        - reasoning: Explanation for the classification
        - all_scores: Scores for all categories considered

    Example:
        >>> content = parse_document("form.pdf")
        >>> result = classify_document(content)
        >>> data = json.loads(result)
        >>> print(f"Type: {data['document_type']} ({data['confidence']:.0%})")
    """
    import asyncio

    if not content:
        return json.dumps({
            "error": "No content provided",
            "document_type": "unknown",
            "confidence": 0.0
        })

    try:
        # Build classification rules
        if custom_categories:
            categories = [c.strip() for c in custom_categories.split(",") if c.strip()]
            rules = [
                ClassificationRule(
                    label=cat,
                    description=f"Documents related to {cat}",
                    keywords=[cat]
                )
                for cat in categories
            ]
        else:
            # Use default rules from classifier
            proc = _get_llama_processor()
            rules = proc.classifier.get_default_rules()

        proc = _get_llama_processor()
        classification = asyncio.run(proc.classifier.classify(
            content=content,
            rules=rules
        ))

        return json.dumps({
            "document_type": classification.document_type.value,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "all_scores": classification.labels
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "document_type": "unknown",
            "confidence": 0.0
        })


@tool
def process_document_full(
    file_path: str,
    extract_fields: str = "",
    chunk_size: int = 2000
) -> str:
    """Process a document through the complete pipeline: parse, classify, extract, and split.

    This is a convenience tool that runs the full document processing pipeline
    in one call. It parses the document, classifies it, optionally extracts
    fields, and splits it into chunks.

    Args:
        file_path: The path to the document file to process.
        extract_fields: Optional comma-separated list of fields to extract.
                        Example: "name, date, total, description"
        chunk_size: Maximum size of chunks in characters. Default is 2000.

    Returns:
        A JSON string containing the complete processing results:
        - markdown: The parsed document content
        - classification: Document type and confidence
        - extraction: Extracted fields (if extract_fields provided)
        - chunks: Array of document chunks
        - metadata: Processing metadata

    Example:
        >>> result = process_document_full(
        ...     "invoice.pdf",
        ...     extract_fields="invoice_number, total, vendor"
        ... )
        >>> data = json.loads(result)
        >>> print(f"Type: {data['classification']['document_type']}")
        >>> print(f"Chunks: {len(data['chunks'])}")
    """
    import asyncio

    file_path = Path(file_path)
    if not file_path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    try:
        proc = _get_llama_processor()

        # Determine extraction schema
        schema = None
        if extract_fields:
            from pydantic import create_model
            field_names = [f.strip() for f in extract_fields.split(",") if f.strip()]
            field_definitions = {name: (Optional[str], None) for name in field_names}
            schema = create_model("DynamicExtraction", **field_definitions)

        # Run full pipeline
        result = asyncio.run(proc.process(
            file_path=file_path,
            schema=schema,
            skip_extraction=not bool(extract_fields)
        ))

        # Build response
        response = {
            "file_name": result.metadata.file_name,
            "markdown": result.markdown,
            "markdown_length": len(result.markdown),
            "classification": {
                "document_type": result.classification.document_type.value if result.classification else "unknown",
                "confidence": result.classification.confidence if result.classification else 0.0,
                "reasoning": result.classification.reasoning if result.classification else None
            },
            "extraction": result.extraction.fields if result.extraction else {},
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                    "category": c.category
                }
                for c in result.chunks
            ],
            "total_chunks": len(result.chunks),
            "metadata": {
                "pages": result.metadata.page_count,
                "processing_time_ms": result.metadata.processing_time_ms,
                "has_tables": result.metadata.has_tables,
                "has_images": result.metadata.has_images
            }
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# List of all tools for easy import
ALL_TOOLS = [
    parse_document,
    extract_from_document,
    split_document,
    classify_document,
    process_document_full,
]
