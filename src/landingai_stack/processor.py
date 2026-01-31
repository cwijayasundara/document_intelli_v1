"""LandingAI ADE full pipeline processor.

Orchestrates ADE Parse, ADE Extract, and ADE Split
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
    GroundingData,
    ParsedDocument,
    get_schema_for_type,
)
from .client import ADEClient, ADERegion
from .parser import ADEParseWrapper
from .extractor import ADEExtractWrapper
from .splitter import ADESplitWrapper


class LandingAIProcessor(DocumentProcessor):
    """Full document processing pipeline using LandingAI ADE stack.

    Orchestrates:
    1. ADE Parse - Document parsing with grounding
    2. ADE Extract - Structured data extraction
    3. ADE Split - Section classification and chunking

    Note: ADE does not have a dedicated classification endpoint,
    so classification is inferred from extraction or split results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: ADERegion = ADERegion.US
    ):
        """Initialize LandingAI processor.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
            region: API region (US or EU)
        """
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("LANDINGAI_API_KEY")
        self.region = region

        if not self.api_key:
            raise ValueError(
                "API key required. Set LANDINGAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize components
        self.parser = ADEParseWrapper(self.api_key, region)
        self.extractor = ADEExtractWrapper(self.api_key, region)
        self.splitter = ADESplitWrapper(self.api_key, region)

    @property
    def name(self) -> str:
        return "landingai"

    async def parse(
        self,
        file_path: Union[str, Path],
        include_grounding: bool = True,
        page_level: bool = False,
        **options
    ) -> str:
        """Parse document to markdown.

        Args:
            file_path: Path to document
            include_grounding: Include bounding box data
            page_level: Return page-level results
            **options: Additional parser options

        Returns:
            Markdown content
        """
        result = await self.parser.parse(
            file_path=file_path,
            include_grounding=include_grounding,
            page_level=page_level,
            **options
        )
        return result["markdown"]

    async def classify(
        self,
        content: str,
        rules: Optional[List[ClassificationRule]] = None,
        **options
    ) -> Classification:
        """Classify document content.

        ADE doesn't have a dedicated classification endpoint,
        so we infer from content analysis.

        Args:
            content: Document content (markdown or text)
            rules: Classification rules (used for category matching)
            **options: Additional options

        Returns:
            Classification result
        """
        # Use split to get section categories, then infer document type
        categories = [rule.label for rule in rules] if rules else self._get_default_categories()

        chunks = await self.splitter.split(
            content=content,
            categories=categories,
            **options
        )

        # Analyze chunk categories to determine document type
        category_counts: Dict[str, int] = {}
        for chunk in chunks:
            if chunk.category:
                category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1

        # Find most common category
        if category_counts:
            primary_category = max(category_counts, key=category_counts.get)
            total_chunks = len(chunks)
            confidence = category_counts[primary_category] / total_chunks if total_chunks > 0 else 0.0
        else:
            primary_category = "other"
            confidence = 0.0

        # Map to DocumentType
        doc_type = self._map_to_document_type(primary_category)

        return Classification(
            document_type=doc_type,
            confidence=confidence,
            reasoning=f"Inferred from section analysis: {dict(category_counts)}",
            labels={k: v / len(chunks) for k, v in category_counts.items()} if chunks else {}
        )

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data from content.

        Args:
            content: Document content
            schema: Pydantic model defining extraction schema
            file_path: Optional original file for grounding
            **options: Additional extractor options

        Returns:
            ExtractionResult with extracted fields
        """
        return await self.extractor.extract(
            content=content,
            schema=schema,
            file_path=file_path,
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
        include_grounding: bool = True,
        page_level: bool = False,
        skip_classification: bool = False,
        skip_extraction: bool = False,
        skip_splitting: bool = False,
        **options
    ) -> ParsedDocument:
        """Run full document processing pipeline.

        Pipeline steps:
        1. Parse document with grounding (ADE Parse)
        2. Classify document type (inferred from splitting)
        3. Extract structured data (ADE Extract)
        4. Split into semantic chunks (ADE Split)

        Args:
            file_path: Path to document
            schema: Extraction schema. Auto-detected from classification if not provided.
            classification_rules: Classification rules
            chunk_categories: Chunk category labels
            include_grounding: Include bounding box data
            page_level: Parse at page level
            skip_classification: Skip classification step
            skip_extraction: Skip extraction step
            skip_splitting: Skip splitting step
            **options: Additional options

        Returns:
            ParsedDocument with all processing results
        """
        file_path = Path(file_path)
        start_time = time.time()
        warnings = []
        errors = []

        # Step 1: Parse document
        parse_result = await self.parser.parse(
            file_path=file_path,
            include_grounding=include_grounding,
            page_level=page_level,
            **options.get("parse_options", {})
        )

        markdown = parse_result["markdown"]
        raw_text = parse_result.get("text", "")
        grounding = parse_result.get("grounding")
        tables = parse_result.get("tables", [])
        figures = parse_result.get("figures", [])
        pages = parse_result.get("pages", [])
        page_count = parse_result["metadata"].get("page_count", 1)

        # Step 2: Split into chunks (also used for classification)
        chunks = []
        if not skip_splitting:
            try:
                chunks = await self.split(
                    content=markdown,
                    categories=chunk_categories,
                    **options.get("split_options", {})
                )
            except Exception as e:
                warnings.append(f"Splitting failed: {str(e)}")

        # Step 3: Classify document (inferred from chunks)
        classification = None
        if not skip_classification:
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

        # Step 4: Extract structured data
        extraction = None
        if not skip_extraction:
            extraction_schema = schema
            if extraction_schema is None and classification:
                extraction_schema = get_schema_for_type(classification.document_type)

            if extraction_schema:
                try:
                    extraction = await self.extract(
                        content=markdown,
                        schema=extraction_schema,
                        file_path=file_path if include_grounding else None,
                        **options.get("extract_options", {})
                    )
                except Exception as e:
                    warnings.append(f"Extraction failed: {str(e)}")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build metadata
        metadata = DocumentMetadata(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_path.suffix,
            file_size_bytes=file_path.stat().st_size,
            page_count=page_count,
            processor=self.name,
            processing_time_ms=processing_time_ms,
            processed_at=datetime.now(timezone.utc),
            api_credits_used=None,  # LandingAI doesn't report credits
            has_tables=len(tables) > 0,
            has_images=len(figures) > 0,
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
            grounding=grounding,
            pages=pages,
            images=figures,
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

    async def close(self):
        """Close all client connections."""
        await self.parser.close()
        await self.extractor.close()
        await self.splitter.close()

    def _get_default_categories(self) -> List[str]:
        """Get default classification categories."""
        return [
            "form",
            "invoice",
            "receipt",
            "certificate",
            "medical",
            "presentation",
            "diagram",
            "flowchart",
            "spreadsheet",
            "instructions",
            "infographic",
            "report",
            "other",
        ]

    def _map_to_document_type(self, category: str) -> DocumentType:
        """Map category string to DocumentType enum."""
        mapping = {
            "form": DocumentType.FORM,
            "invoice": DocumentType.INVOICE,
            "receipt": DocumentType.RECEIPT,
            "certificate": DocumentType.CERTIFICATE,
            "medical": DocumentType.MEDICAL,
            "presentation": DocumentType.PRESENTATION,
            "diagram": DocumentType.DIAGRAM,
            "flowchart": DocumentType.FLOWCHART,
            "spreadsheet": DocumentType.SPREADSHEET,
            "instructions": DocumentType.INSTRUCTIONS,
            "infographic": DocumentType.INFOGRAPHIC,
            "report": DocumentType.REPORT,
            "handwritten": DocumentType.HANDWRITTEN,
            "contract": DocumentType.CONTRACT,
            "educational": DocumentType.EDUCATIONAL,
        }
        return mapping.get(category.lower(), DocumentType.OTHER)
