"""End-to-end loan processing pipeline.

Orchestrates document parsing, categorization, extraction, and validation.
Includes caching for parsed documents to avoid redundant API calls.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.pipelines.loan_processing.schemas import (
    LoanDocumentType,
    DocumentExtractionResult,
)
from src.pipelines.loan_processing.categorizer import LoanDocumentCategorizer
from src.pipelines.loan_processing.extractor import LoanFieldExtractor
from src.pipelines.loan_processing.validator import LoanValidator, LoanValidationResult
from src.pipelines.loan_processing.visualizer import DocumentVisualizer

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path("parsed")


class ProcessedDocument(BaseModel):
    """A single processed document with all results."""
    file_path: str = Field(..., description="Path to the source file")
    file_name: str = Field(..., description="Name of the source file")
    markdown: str = Field("", description="Parsed markdown content")
    document_type: LoanDocumentType = Field(
        LoanDocumentType.UNKNOWN,
        description="Classified document type"
    )
    classification_confidence: float = Field(
        0.0,
        description="Confidence of classification"
    )
    classification_reasoning: str = Field(
        "",
        description="Reasoning for classification"
    )
    extraction: DocumentExtractionResult = Field(
        ...,
        description="Extracted fields and grounding"
    )
    processing_time_ms: float = Field(
        0.0,
        description="Time to process this document"
    )
    from_cache: bool = Field(
        False,
        description="Whether the parsed content was loaded from cache"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )


class LoanApplicationResult(BaseModel):
    """Complete result for a loan application."""
    # Processing results
    documents: List[ProcessedDocument] = Field(
        default_factory=list,
        description="All processed documents"
    )
    validation: Optional[LoanValidationResult] = Field(
        None,
        description="Cross-document validation results"
    )

    # Summary statistics
    total_documents: int = Field(0, description="Total documents processed")
    successful_documents: int = Field(0, description="Documents processed successfully")
    failed_documents: int = Field(0, description="Documents that failed processing")
    documents_from_cache: int = Field(0, description="Documents loaded from parse cache")
    total_processing_time_ms: float = Field(0.0, description="Total processing time")

    # Timestamps
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the application was processed"
    )

    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="All errors encountered")
    warnings: List[str] = Field(default_factory=list, description="All warnings")

    def get_document_by_type(
        self,
        doc_type: LoanDocumentType
    ) -> Optional[ProcessedDocument]:
        """Get the first document of a specific type."""
        for doc in self.documents:
            if doc.document_type == doc_type:
                return doc
        return None

    def get_all_documents_by_type(
        self,
        doc_type: LoanDocumentType
    ) -> List[ProcessedDocument]:
        """Get all documents of a specific type."""
        return [doc for doc in self.documents if doc.document_type == doc_type]

    @property
    def is_complete(self) -> bool:
        """Check if the application has all required documents."""
        types_found = {doc.document_type for doc in self.documents}
        has_id = LoanDocumentType.ID in types_found
        has_income = (
            LoanDocumentType.W2 in types_found or
            LoanDocumentType.PAY_STUB in types_found
        )
        return has_id and has_income


class LoanProcessingPipeline:
    """End-to-end pipeline for processing loan application documents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        processor: str = "landingai"
    ):
        """Initialize the pipeline.

        Args:
            api_key: API key for the selected processor. Defaults to env var.
            min_year: Minimum acceptable year for documents
            max_year: Maximum acceptable year for documents
            cache_dir: Directory to store parsed document cache. Defaults to "parsed/"
            use_cache: Whether to use caching for parsed documents. Defaults to True.
            processor: Which processor to use - "landingai" or "reducto". Default is "landingai".
        """
        self.processor_name = processor

        if processor == "reducto":
            from src.reducto_stack.client import ReductoClient
            self.client = ReductoClient(api_key=api_key)
        else:
            from src.landingai_stack.client import ADEClient
            self.client = ADEClient(api_key=api_key)

        self.categorizer = LoanDocumentCategorizer(api_key=api_key, processor=processor)
        self.extractor = LoanFieldExtractor(api_key=api_key, processor=processor)
        self.validator = LoanValidator(min_year=min_year, max_year=max_year)
        self.visualizer = DocumentVisualizer()

        # Cache configuration
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Parse cache directory: {self.cache_dir.absolute()}")

    def _get_cache_path(self, file_path: Path) -> Path:
        """Get the cache file path for a document.

        Uses the original filename with .md extension.
        If multiple files have the same name, appends a hash of the full path.

        Args:
            file_path: Path to the original document

        Returns:
            Path to the cached markdown file
        """
        # Use original filename with .md extension
        cache_name = f"{file_path.stem}.md"
        cache_path = self.cache_dir / cache_name

        # If there might be naming conflicts, we can use a hash
        # For now, use simple filename-based caching
        return cache_path

    def _get_cache_path_with_hash(self, file_path: Path) -> Path:
        """Get cache path using file content hash for uniqueness.

        This ensures the same file always maps to the same cache,
        even if moved or renamed.

        Args:
            file_path: Path to the original document

        Returns:
            Path to the cached markdown file
        """
        # Create hash of file path for uniqueness
        path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
        cache_name = f"{file_path.stem}_{path_hash}.md"
        return self.cache_dir / cache_name

    def _load_from_cache(self, file_path: Path) -> Optional[str]:
        """Load parsed markdown from cache if available.

        Cache lookup is based on filename only - if a cache file exists with
        the same name, it will be used. This works well for uploaded files
        where the same document may be re-uploaded multiple times.

        Use force_reparse=True to bypass cache and get fresh content.

        Args:
            file_path: Path to the original document

        Returns:
            Cached markdown content, or None if not cached
        """
        cache_path = self._get_cache_path(file_path)

        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}")
                return None

        return None

    def _save_to_cache(self, file_path: Path, markdown: str) -> None:
        """Save parsed markdown to cache.

        Args:
            file_path: Path to the original document
            markdown: Parsed markdown content
        """
        cache_path = self._get_cache_path(file_path)

        try:
            cache_path.write_text(markdown, encoding="utf-8")
            logger.info(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def parse_document(
        self,
        file_path: Path,
        force_reparse: bool = False
    ) -> str:
        """Parse a document to markdown, using cache if available.

        Args:
            file_path: Path to the document file
            force_reparse: If True, ignore cache and re-parse the document

        Returns:
            Markdown content
        """
        file_path = Path(file_path)

        # Check cache first (unless force_reparse is True)
        if self.use_cache and not force_reparse:
            cached = self._load_from_cache(file_path)
            if cached is not None:
                logger.info(f"Using cached parse for: {file_path.name}")
                return cached

        # Parse the document
        logger.info(f"Parsing document (API call) with {self.processor_name}: {file_path}")
        if self.processor_name == "reducto":
            from src.reducto_stack.parser import ReductoParseWrapper
            parser = ReductoParseWrapper()
            result = await parser.parse(file_path=file_path)
        else:
            result = await self.client.parse(file_path=file_path)
        markdown = result.get("markdown", "")

        logger.info(f"Parsed {len(markdown)} characters from {file_path.name}")

        # Save to cache
        if self.use_cache and markdown:
            self._save_to_cache(file_path, markdown)

        return markdown

    def clear_cache(self, file_path: Optional[Path] = None) -> int:
        """Clear cached parsed documents.

        Args:
            file_path: If provided, only clear cache for this file.
                      If None, clear all cached files.

        Returns:
            Number of cache files deleted
        """
        if file_path:
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for: {file_path.name}")
                return 1
            return 0

        # Clear all cache files
        count = 0
        for cache_file in self.cache_dir.glob("*.md"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached files")
        return count

    def list_cached_documents(self) -> List[Path]:
        """List all cached document files.

        Returns:
            List of paths to cached markdown files
        """
        if not self.cache_dir.exists():
            return []
        return sorted(self.cache_dir.glob("*.md"))

    async def process_single_document(
        self,
        file_path: Path,
        force_reparse: bool = False
    ) -> ProcessedDocument:
        """Process a single document through the pipeline.

        Args:
            file_path: Path to the document file
            force_reparse: If True, ignore cache and re-parse the document

        Returns:
            ProcessedDocument with all results
        """
        import time
        start_time = time.time()

        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")

        result = ProcessedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            extraction=DocumentExtractionResult(
                document_type=LoanDocumentType.UNKNOWN,
                file_path=str(file_path),
                file_name=file_path.name
            )
        )

        try:
            # Check if we'll use cache (for tracking purposes)
            from_cache = False
            if self.use_cache and not force_reparse:
                cached = self._load_from_cache(file_path)
                if cached is not None:
                    from_cache = True

            # Step 1: Parse document (uses cache if available)
            markdown = await self.parse_document(file_path, force_reparse=force_reparse)
            result.markdown = markdown
            result.from_cache = from_cache

            # Step 2: Categorize document
            doc_type, confidence, reasoning = await self.categorizer.categorize(
                file_path,
                markdown_content=markdown
            )
            result.document_type = doc_type
            result.classification_confidence = confidence
            result.classification_reasoning = reasoning

            # Step 3: Extract fields based on document type
            if doc_type != LoanDocumentType.UNKNOWN:
                extraction = await self.extractor.extract(
                    file_path,
                    doc_type,
                    markdown_content=markdown
                )
                result.extraction = extraction
            else:
                result.extraction = DocumentExtractionResult(
                    document_type=doc_type,
                    confidence=confidence,
                    fields={},
                    file_path=str(file_path),
                    file_name=file_path.name
                )

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            result.error = str(e)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    async def process_application(
        self,
        documents: List[Path],
        validate: bool = True,
        force_reparse: bool = False
    ) -> LoanApplicationResult:
        """Process a complete loan application.

        Args:
            documents: List of document file paths
            validate: Whether to run cross-document validation
            force_reparse: If True, ignore cache and re-parse all documents

        Returns:
            LoanApplicationResult with all processing results
        """
        import time
        start_time = time.time()

        logger.info(f"Processing loan application with {len(documents)} documents")
        if force_reparse:
            logger.info("Force re-parse enabled, ignoring cache")

        result = LoanApplicationResult(
            total_documents=len(documents)
        )

        # Process all documents
        processed_docs = []
        for doc_path in documents:
            try:
                processed = await self.process_single_document(
                    Path(doc_path),
                    force_reparse=force_reparse
                )
                processed_docs.append(processed)

                if processed.error:
                    result.errors.append(f"{processed.file_name}: {processed.error}")
                    result.failed_documents += 1
                else:
                    result.successful_documents += 1
                    if processed.from_cache:
                        result.documents_from_cache += 1

            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
                result.errors.append(f"{Path(doc_path).name}: {str(e)}")
                result.failed_documents += 1

        result.documents = processed_docs

        # Run validation if requested
        if validate and processed_docs:
            extraction_results = [
                doc.extraction for doc in processed_docs
                if doc.extraction and not doc.error
            ]
            if extraction_results:
                result.validation = self.validator.validate(extraction_results)

                # Add validation issues to warnings
                if result.validation.issues:
                    result.warnings.extend(result.validation.issues)

        result.total_processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Processed {result.successful_documents}/{result.total_documents} documents "
            f"in {result.total_processing_time_ms:.0f}ms"
        )

        return result

    async def get_annotated_document(
        self,
        processed_doc: ProcessedDocument,
        page_number: int = 0
    ):
        """Get an annotated image of a processed document.

        Args:
            processed_doc: The processed document
            page_number: Page number to annotate (for PDFs)

        Returns:
            PIL Image with annotations, or None if failed
        """
        if not processed_doc.extraction.grounding:
            logger.info("No grounding data available for annotation")
            return None

        file_path = Path(processed_doc.file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        return self.visualizer.annotate_document(
            file_path,
            {
                "fields": processed_doc.extraction.fields,
                "grounding": processed_doc.extraction.grounding
            },
            page_number
        )


async def main():
    """Demo the loan processing pipeline."""
    import sys

    # Parse command line arguments
    args = sys.argv[1:]

    # Check for flags
    force_reparse = "--no-cache" in args or "--force" in args
    clear_cache = "--clear-cache" in args

    # Remove flags from args
    args = [a for a in args if not a.startswith("--")]

    if len(args) < 1 and not clear_cache:
        print("Usage: python -m src.pipelines.loan_processing.pipeline [options] <doc1> [doc2] ...")
        print()
        print("Options:")
        print("  --no-cache     Ignore cache and re-parse all documents")
        print("  --force        Same as --no-cache")
        print("  --clear-cache  Clear all cached parsed documents")
        print()
        print("Example: python -m src.pipelines.loan_processing.pipeline tests/loan_samples/*.pdf")
        print("Example: python -m src.pipelines.loan_processing.pipeline --no-cache doc.pdf")
        sys.exit(1)

    # Initialize pipeline
    pipeline = LoanProcessingPipeline()

    # Handle clear cache
    if clear_cache:
        count = pipeline.clear_cache()
        print(f"Cleared {count} cached files from {pipeline.cache_dir}")
        if len(args) == 0:
            sys.exit(0)

    documents = [Path(p) for p in args]

    # Validate paths
    for doc in documents:
        if not doc.exists():
            print(f"File not found: {doc}")
            sys.exit(1)

    print(f"Processing {len(documents)} documents...")
    if force_reparse:
        print("(Cache disabled - forcing re-parse)")
    else:
        print(f"(Using cache directory: {pipeline.cache_dir.absolute()})")

    # Process the application
    result = await pipeline.process_application(documents, force_reparse=force_reparse)

    # Print results
    print("\n" + "=" * 60)
    print("LOAN APPLICATION PROCESSING RESULTS")
    print("=" * 60)

    print(f"\nDocuments processed: {result.successful_documents}/{result.total_documents}")
    print(f"Documents from cache: {result.documents_from_cache}")
    print(f"Processing time: {result.total_processing_time_ms:.0f}ms")

    print("\n--- Document Summary ---")
    for doc in result.documents:
        status = "OK" if not doc.error else "FAILED"
        cache_indicator = " (cached)" if doc.from_cache else ""
        print(f"\n{doc.file_name}: {status}{cache_indicator}")
        print(f"  Type: {doc.document_type.value} ({doc.classification_confidence:.0%})")
        if doc.extraction.fields:
            print("  Extracted fields:")
            for field, value in doc.extraction.fields.items():
                print(f"    - {field}: {value}")
        if doc.error:
            print(f"  Error: {doc.error}")

    if result.validation:
        print("\n--- Validation Results ---")
        print(f"Validation passed: {result.validation.validation_passed}")
        print(f"Names found: {', '.join(result.validation.names_found)}")
        print(f"Years found: {', '.join(map(str, result.validation.years_found))}")
        print(f"Total bank balance: ${result.validation.total_bank_balance:,.2f}")
        print(f"Total investments: ${result.validation.total_investment_value:,.2f}")
        print(f"Total assets: ${result.validation.total_assets:,.2f}")

        if result.validation.annual_income:
            print(f"Estimated annual income: ${result.validation.annual_income:,.2f}")

        if result.validation.issues:
            print("\nIssues:")
            for issue in result.validation.issues:
                print(f"  - {issue}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
