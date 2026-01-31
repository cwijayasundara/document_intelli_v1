"""LlamaIndex full pipeline processor.

Orchestrates LlamaParse, LlamaClassify, LlamaExtract, and LlamaSplit
into a unified document processing pipeline.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..common.interfaces import ClassificationRule, DocumentProcessor
from ..common.models import (
    Chunk,
    Classification,
    DocumentMetadata,
    DocumentType,
    ExtractionResult,
    ParsedDocument,
    get_schema_for_type,
)
from .parser import LlamaParseWrapper, ParseTier
from .classifier import LlamaClassifyWrapper, ClassifyMode
from .extractor import LlamaExtractWrapper
from .splitter import LlamaSplitWrapper


class LlamaIndexProcessor(DocumentProcessor):
    """Full document processing pipeline using LlamaIndex stack.

    Orchestrates:
    1. LlamaParse - Document parsing to markdown
    2. LlamaClassify - Document type classification
    3. LlamaExtract - Structured data extraction
    4. LlamaSplit - Semantic chunking
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LlamaIndex processor.

        Args:
            api_key: LlamaCloud API key. Defaults to LLAMA_CLOUD_API_KEY env var.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set LLAMA_CLOUD_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize components
        self.parser = LlamaParseWrapper(self.api_key)
        self.classifier = LlamaClassifyWrapper(self.api_key)
        self.extractor = LlamaExtractWrapper(self.api_key)
        self.splitter = LlamaSplitWrapper(self.api_key)

    @property
    def name(self) -> str:
        return "llamaindex"

    async def parse(
        self,
        file_path: Union[str, Path],
        tier: ParseTier = ParseTier.AGENTIC,
        multimodal: bool = False,
        **options
    ) -> str:
        """Parse document to markdown.

        Args:
            file_path: Path to document
            tier: Parsing tier (cost_effective, agentic, agentic_plus, fast)
            multimodal: Enable multimodal mode for visual documents
            **options: Additional parser options

        Returns:
            Markdown content
        """
        result = await self.parser.parse(
            file_path=file_path,
            tier=tier,
            multimodal=multimodal,
            **options
        )
        return result["markdown"]

    async def classify(
        self,
        content: str,
        rules: Optional[List[ClassificationRule]] = None,
        mode: ClassifyMode = ClassifyMode.FAST,
        **options
    ) -> Classification:
        """Classify document content.

        Args:
            content: Document content (markdown or text)
            rules: Classification rules. Uses defaults if not provided.
            mode: Classification mode (fast or multimodal)
            **options: Additional classifier options

        Returns:
            Classification result
        """
        if rules is None:
            rules = self.classifier.get_default_rules()

        return await self.classifier.classify(
            content=content,
            rules=rules,
            mode=mode,
            **options
        )

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        **options
    ) -> ExtractionResult:
        """Extract structured data from content.

        Args:
            content: Document content
            schema: Pydantic model defining extraction schema
            **options: Additional extractor options

        Returns:
            ExtractionResult with extracted fields
        """
        return await self.extractor.extract(
            content=content,
            schema=schema,
            **options
        )

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split document into semantic chunks.

        Args:
            content: Document content
            categories: Optional category labels for chunks
            **options: Additional splitter options

        Returns:
            List of Chunk objects
        """
        return await self.splitter.split(
            content=content,
            categories=categories,
            **options
        )

    async def process(
        self,
        file_path: Union[str, Path],
        schema: Optional[Type[BaseModel]] = None,
        classification_rules: Optional[List[ClassificationRule]] = None,
        chunk_categories: Optional[List[str]] = None,
        parse_tier: ParseTier = ParseTier.AGENTIC,
        multimodal: bool = False,
        skip_classification: bool = False,
        skip_extraction: bool = False,
        skip_splitting: bool = False,
        **options
    ) -> ParsedDocument:
        """Run full document processing pipeline.

        Pipeline steps:
        1. Parse document to markdown (LlamaParse)
        2. Classify document type (LlamaClassify)
        3. Extract structured data (LlamaExtract)
        4. Split into semantic chunks (LlamaSplit)

        Args:
            file_path: Path to document
            schema: Extraction schema. Auto-detected from classification if not provided.
            classification_rules: Classification rules. Uses defaults if not provided.
            chunk_categories: Chunk category labels.
            parse_tier: Parsing tier
            multimodal: Enable multimodal parsing
            skip_classification: Skip classification step
            skip_extraction: Skip extraction step
            skip_splitting: Skip splitting step
            **options: Additional options passed to each step

        Returns:
            ParsedDocument with all processing results
        """
        file_path = Path(file_path)
        start_time = time.time()
        warnings = []
        errors = []
        total_credits = 0.0

        # Step 1: Parse document
        try:
            parse_result = await self.parser.parse(
                file_path=file_path,
                tier=parse_tier,
                multimodal=multimodal,
                **options.get("parse_options", {})
            )

            markdown = parse_result["markdown"]
            raw_text = parse_result.get("text", "")
            images = parse_result.get("images", [])
            tables = parse_result.get("tables", [])
            total_credits += parse_result["metadata"].get("credits_used", 0)
            page_count = parse_result["metadata"].get("pages", 1)
        except Exception as e:
            errors.append(f"Parsing failed: {str(e)}")
            markdown = ""
            raw_text = ""
            images = []
            tables = []
            page_count = 1

        # Step 2: Classify document
        classification = None
        if not skip_classification and markdown:
            try:
                classification = await self.classify(
                    content=markdown,
                    rules=classification_rules,
                    **options.get("classify_options", {})
                )
            except Exception as e:
                warnings.append(f"Classification failed: {str(e)}")
                classification = Classification(
                    document_type=DocumentType.OTHER,
                    confidence=0.0,
                    reasoning="Classification failed"
                )

        # Step 3: Extract structured data
        extraction = None
        if not skip_extraction and markdown:
            extraction_schema = schema
            if extraction_schema is None and classification:
                extraction_schema = get_schema_for_type(classification.document_type)

            if extraction_schema:
                try:
                    extraction = await self.extract(
                        content=markdown,
                        schema=extraction_schema,
                        **options.get("extract_options", {})
                    )
                except Exception as e:
                    warnings.append(f"Extraction failed: {str(e)}")
            else:
                # Auto-extract without schema
                try:
                    extraction = await self.extractor.auto_extract(
                        content=markdown,
                        **options.get("extract_options", {})
                    )
                except Exception as e:
                    warnings.append(f"Auto-extraction failed: {str(e)}")

        # Step 4: Split into chunks
        chunks = []
        if not skip_splitting and markdown:
            try:
                chunks = await self.split(
                    content=markdown,
                    categories=chunk_categories,
                    **options.get("split_options", {})
                )
            except Exception as e:
                warnings.append(f"Splitting failed: {str(e)}")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build metadata
        metadata = DocumentMetadata(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_path.suffix,
            file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            page_count=page_count,
            processor=self.name,
            processing_time_ms=processing_time_ms,
            processed_at=datetime.now(timezone.utc),
            api_credits_used=total_credits,
            has_tables=len(tables) > 0,
            has_images=len(images) > 0,
            has_handwriting=False,
            warnings=warnings,
            errors=errors,
        )

        return ParsedDocument(
            markdown=markdown,
            raw_text=raw_text,
            chunks=chunks,
            classification=classification,
            extraction=extraction,
            metadata=metadata,
            grounding=None,
            pages=[],
            images=images,
            tables=tables,
        )

    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        **options
    ) -> List[ParsedDocument]:
        """Process multiple documents.

        Args:
            file_paths: List of document paths
            **options: Options passed to process()

        Returns:
            List of ParsedDocument results
        """
        results = []
        for path in file_paths:
            try:
                result = await self.process(path, **options)
                results.append(result)
            except Exception as e:
                path = Path(path)
                error_result = ParsedDocument(
                    markdown="",
                    metadata=DocumentMetadata(
                        source_path=str(path),
                        file_name=path.name,
                        file_type=path.suffix,
                        processor=self.name,
                        errors=[str(e)],
                    ),
                )
                results.append(error_result)

        return results
