"""Common interfaces for document processors.

Both LlamaIndex and LandingAI stacks implement these interfaces
to ensure consistent API and enable easy comparison.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type, Union, runtime_checkable

from pydantic import BaseModel

from .models import (
    Classification,
    Chunk,
    DocumentType,
    ExtractionResult,
    ParsedDocument,
)


class ClassificationRule(BaseModel):
    """Rule for document classification."""
    label: str
    description: str
    keywords: List[str] = []
    examples: List[str] = []


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for document parsing components."""

    async def parse(
        self,
        file_path: Union[str, Path],
        **options
    ) -> str:
        """Parse a document and return markdown content.

        Args:
            file_path: Path to the document file
            **options: Parser-specific options

        Returns:
            Parsed content as markdown string
        """
        ...


@runtime_checkable
class DocumentClassifier(Protocol):
    """Protocol for document classification components."""

    async def classify(
        self,
        content: str,
        rules: List[ClassificationRule],
        **options
    ) -> Classification:
        """Classify document content.

        Args:
            content: Document content (markdown or text)
            rules: Classification rules/categories
            **options: Classifier-specific options

        Returns:
            Classification result with type, confidence, and reasoning
        """
        ...


@runtime_checkable
class DocumentExtractor(Protocol):
    """Protocol for structured data extraction components."""

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        **options
    ) -> ExtractionResult:
        """Extract structured data according to a schema.

        Args:
            content: Document content (markdown or text)
            schema: Pydantic model defining extraction schema
            **options: Extractor-specific options

        Returns:
            ExtractionResult with extracted fields
        """
        ...


@runtime_checkable
class DocumentSplitter(Protocol):
    """Protocol for semantic chunking/splitting components."""

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split document into semantic chunks.

        Args:
            content: Document content (markdown or text)
            categories: Optional category labels for chunks
            **options: Splitter-specific options

        Returns:
            List of semantic chunks
        """
        ...


class DocumentProcessor(ABC):
    """Abstract base class for full document processing pipeline.

    Both LlamaIndex and LandingAI processors inherit from this
    to ensure consistent API across stacks.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize processor with optional API key.

        Args:
            api_key: API key for the service. If not provided,
                    will look for environment variable.
        """
        self.api_key = api_key

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the processor name (e.g., 'llamaindex', 'landingai')."""
        ...

    @abstractmethod
    async def parse(
        self,
        file_path: Union[str, Path],
        **options
    ) -> str:
        """Parse document to markdown.

        Args:
            file_path: Path to document
            **options: Parser options (tier, mode, etc.)

        Returns:
            Markdown content
        """
        ...

    @abstractmethod
    async def classify(
        self,
        content: str,
        rules: Optional[List[ClassificationRule]] = None,
        **options
    ) -> Classification:
        """Classify document content.

        Args:
            content: Document content
            rules: Optional classification rules
            **options: Classifier options

        Returns:
            Classification result
        """
        ...

    @abstractmethod
    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        **options
    ) -> ExtractionResult:
        """Extract structured data.

        Args:
            content: Document content
            schema: Pydantic schema for extraction
            **options: Extractor options

        Returns:
            Extraction result
        """
        ...

    @abstractmethod
    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split into semantic chunks.

        Args:
            content: Document content
            categories: Optional category labels
            **options: Splitter options

        Returns:
            List of chunks
        """
        ...

    @abstractmethod
    async def process(
        self,
        file_path: Union[str, Path],
        schema: Optional[Type[BaseModel]] = None,
        classification_rules: Optional[List[ClassificationRule]] = None,
        chunk_categories: Optional[List[str]] = None,
        **options
    ) -> ParsedDocument:
        """Run full processing pipeline.

        This is the main entry point that orchestrates:
        1. Parsing (convert to markdown)
        2. Classification (determine document type)
        3. Extraction (extract structured fields)
        4. Splitting (semantic chunking)

        Args:
            file_path: Path to document
            schema: Optional extraction schema (auto-detected if not provided)
            classification_rules: Optional classification rules
            chunk_categories: Optional chunk categories
            **options: Additional options passed to each step

        Returns:
            ParsedDocument with all processing results
        """
        ...

    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        **options
    ) -> List[ParsedDocument]:
        """Process multiple documents.

        Default implementation processes sequentially.
        Subclasses may override for parallel processing.

        Args:
            file_paths: List of document paths
            **options: Options passed to process()

        Returns:
            List of ParsedDocument results
        """
        results = []
        for path in file_paths:
            result = await self.process(path, **options)
            results.append(result)
        return results


class HandwritingProcessor(Protocol):
    """Protocol for handwriting-specific processing."""

    async def process_handwriting(
        self,
        file_path: Union[str, Path],
        **options
    ) -> str:
        """Process handwritten document content.

        Args:
            file_path: Path to document with handwriting
            **options: Processor-specific options

        Returns:
            Extracted text from handwriting
        """
        ...

    async def detect_handwriting(
        self,
        file_path: Union[str, Path]
    ) -> bool:
        """Detect if document contains handwriting.

        Args:
            file_path: Path to document

        Returns:
            True if handwriting is detected
        """
        ...


# Default classification rules for common document types
DEFAULT_CLASSIFICATION_RULES: List[ClassificationRule] = [
    ClassificationRule(
        label="form",
        description="Structured forms with fields, checkboxes, or fillable areas",
        keywords=["form", "checkbox", "fill in", "please complete"],
    ),
    ClassificationRule(
        label="invoice",
        description="Billing documents with line items and totals",
        keywords=["invoice", "bill to", "amount due", "payment"],
    ),
    ClassificationRule(
        label="certificate",
        description="Official certificates or credentials",
        keywords=["certificate", "certify", "hereby", "issued"],
    ),
    ClassificationRule(
        label="medical",
        description="Medical or healthcare related documents",
        keywords=["patient", "medical", "health", "diagnosis", "prescription"],
    ),
    ClassificationRule(
        label="presentation",
        description="Slides or presentation materials",
        keywords=["slide", "presentation", "overview", "summary"],
    ),
    ClassificationRule(
        label="diagram",
        description="Technical diagrams, flowcharts, or visual explanations",
        keywords=["diagram", "flow", "process", "step"],
    ),
    ClassificationRule(
        label="spreadsheet",
        description="Tabular data, spreadsheets, or data grids",
        keywords=["total", "sum", "data", "column", "row"],
    ),
    ClassificationRule(
        label="instructions",
        description="Assembly instructions, manuals, or how-to guides",
        keywords=["step", "instruction", "assembly", "guide", "how to"],
    ),
    ClassificationRule(
        label="handwritten",
        description="Documents containing handwritten text",
        keywords=["handwritten", "handwriting", "written by hand"],
    ),
    ClassificationRule(
        label="report",
        description="Reports, analyses, or detailed documents",
        keywords=["report", "analysis", "findings", "conclusion"],
    ),
]
