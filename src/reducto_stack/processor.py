"""Reducto full pipeline processor.

Orchestrates Reducto Parse, Extract, and Split
into a unified document processing pipeline.
"""

import os
import time
from collections import Counter
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
from .client import ReductoClient
from .parser import ReductoParseWrapper
from .extractor import ReductoExtractWrapper
from .splitter import ReductoSplitWrapper


class ReductoProcessor(DocumentProcessor):
    """Full document processing pipeline using Reducto stack.

    Orchestrates:
    1. Reducto Parse - Document parsing with bounding box grounding
    2. Reducto Extract - Structured data extraction
    3. Reducto Split - Section classification and splitting

    Note: Reducto does not have a dedicated classification endpoint,
    so classification is inferred from block types in the parse result.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Reducto processor.

        Args:
            api_key: Reducto API key. Defaults to REDUCTO_API_KEY env var.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("REDUCTO_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set REDUCTO_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize components
        self.parser = ReductoParseWrapper(self.api_key)
        self.extractor = ReductoExtractWrapper(self.api_key)
        self.splitter = ReductoSplitWrapper(self.api_key)

        # Shared client for upload caching
        self._client = ReductoClient(self.api_key)

    @property
    def name(self) -> str:
        return "reducto"

    async def parse(
        self,
        file_path: Union[str, Path],
        include_grounding: bool = True,
        **options
    ) -> str:
        """Parse document to markdown.

        Args:
            file_path: Path to document
            include_grounding: Include bounding box data
            **options: Additional parser options

        Returns:
            Markdown content
        """
        result = await self.parser.parse(
            file_path=file_path,
            include_grounding=include_grounding,
            **options
        )
        return result["markdown"]

    async def classify(
        self,
        content: str,
        rules: Optional[List[ClassificationRule]] = None,
        block_types: Optional[List[str]] = None,
        **options
    ) -> Classification:
        """Classify document content.

        Reducto doesn't have a dedicated classification endpoint,
        so we infer from block types and content analysis.

        Args:
            content: Document content (markdown or text)
            rules: Classification rules (used for keyword matching)
            block_types: Block types from parse result for type inference
            **options: Additional options

        Returns:
            Classification result
        """
        # First try block-type based classification
        if block_types:
            type_counts = Counter(block_types)
            doc_type, confidence, reasoning = self._classify_from_block_types(type_counts)
            if confidence > 0.3:
                return Classification(
                    document_type=doc_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    labels={k: v / len(block_types) for k, v in type_counts.items()},
                )

        # Fallback: keyword-based classification from content
        return self._keyword_classify(content, rules)

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data from content.

        Args:
            content: Document content
            schema: Pydantic model defining extraction schema
            file_path: Optional original file for direct extraction
            upload_ref: Pre-uploaded file reference
            **options: Additional extractor options

        Returns:
            ExtractionResult with extracted fields
        """
        return await self.extractor.extract(
            content=content,
            schema=schema,
            file_path=file_path,
            upload_ref=upload_ref,
            **options
        )

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        **options
    ) -> List[Chunk]:
        """Split document into semantic chunks.

        Args:
            content: Document content
            categories: Optional category labels for chunks
            file_path: Optional original file for API-based splitting
            upload_ref: Pre-uploaded file reference
            **options: Additional splitter options

        Returns:
            List of Chunk objects
        """
        return await self.splitter.split(
            content=content,
            categories=categories,
            file_path=file_path,
            upload_ref=upload_ref,
            **options
        )

    async def process(
        self,
        file_path: Union[str, Path],
        schema: Optional[Type[BaseModel]] = None,
        classification_rules: Optional[List[ClassificationRule]] = None,
        chunk_categories: Optional[List[str]] = None,
        include_grounding: bool = True,
        skip_classification: bool = False,
        skip_extraction: bool = False,
        skip_splitting: bool = False,
        **options
    ) -> ParsedDocument:
        """Run full document processing pipeline.

        Pipeline steps:
        1. Upload document once (cached for all operations)
        2. Parse document with grounding (Reducto Parse)
        3. Classify document type (inferred from block types)
        4. Extract structured data (Reducto Extract)
        5. Split into semantic chunks (Reducto Split)

        Args:
            file_path: Path to document
            schema: Extraction schema. Auto-detected from classification if not provided.
            classification_rules: Classification rules
            chunk_categories: Chunk category labels
            include_grounding: Include bounding box data
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
        total_credits = 0.0

        # Step 0: Upload once, reuse for all operations
        upload_ref = await self._client.upload(file_path)

        # Step 1: Parse document
        parse_result = await self.parser.parse(
            file_path=file_path,
            include_grounding=include_grounding,
            upload_ref=upload_ref,
            **options.get("parse_options", {})
        )

        markdown = parse_result["markdown"]
        raw_text = parse_result.get("text", "")
        grounding = parse_result.get("grounding")
        tables = parse_result.get("tables", [])
        figures = parse_result.get("figures", [])
        pages = parse_result.get("pages", [])
        page_count = parse_result["metadata"].get("page_count", 1)
        credits = parse_result["metadata"].get("credits_used")
        if credits:
            total_credits += credits

        # Collect block types for classification
        block_types = []
        for chunk_data in parse_result.get("chunks", []):
            if isinstance(chunk_data, dict) and "blocks" in chunk_data:
                for block in chunk_data["blocks"]:
                    if isinstance(block, dict) and "type" in block:
                        block_types.append(block["type"])

        # Step 2: Classify document (inferred from block types and content)
        classification = None
        if not skip_classification:
            try:
                classification = await self.classify(
                    content=markdown,
                    rules=classification_rules,
                    block_types=block_types if block_types else None,
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
        if not skip_extraction:
            extraction_schema = schema
            if extraction_schema is None and classification:
                extraction_schema = get_schema_for_type(classification.document_type)

            if extraction_schema:
                try:
                    extraction = await self.extract(
                        content=markdown,
                        schema=extraction_schema,
                        file_path=file_path,
                        upload_ref=upload_ref,
                        **options.get("extract_options", {})
                    )
                except Exception as e:
                    warnings.append(f"Extraction failed: {str(e)}")

        # Step 4: Split into chunks
        chunks = []
        if not skip_splitting:
            try:
                chunks = await self.split(
                    content=markdown,
                    categories=chunk_categories,
                    file_path=file_path,
                    upload_ref=upload_ref,
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
            file_size_bytes=file_path.stat().st_size,
            page_count=page_count,
            processor=self.name,
            processing_time_ms=processing_time_ms,
            processed_at=datetime.now(timezone.utc),
            api_credits_used=total_credits if total_credits > 0 else None,
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

    def _classify_from_block_types(
        self,
        type_counts: Counter
    ) -> tuple:
        """Classify document type from block type frequency.

        Args:
            type_counts: Counter of block types

        Returns:
            Tuple of (DocumentType, confidence, reasoning)
        """
        total = sum(type_counts.values())
        if total == 0:
            return DocumentType.OTHER, 0.0, "No blocks found"

        # Calculate proportions
        kv_ratio = type_counts.get("Key Value", 0) / total
        table_ratio = type_counts.get("Table", 0) / total
        figure_ratio = type_counts.get("Figure", 0) / total
        text_ratio = type_counts.get("Text", 0) / total

        if kv_ratio > 0.3:
            return DocumentType.FORM, min(kv_ratio + 0.2, 1.0), \
                f"High Key Value block ratio ({kv_ratio:.0%})"
        elif table_ratio > 0.3:
            return DocumentType.SPREADSHEET, min(table_ratio + 0.2, 1.0), \
                f"High Table block ratio ({table_ratio:.0%})"
        elif figure_ratio > 0.3:
            return DocumentType.DIAGRAM, min(figure_ratio + 0.2, 1.0), \
                f"High Figure block ratio ({figure_ratio:.0%})"
        elif text_ratio > 0.7:
            return DocumentType.REPORT, min(text_ratio * 0.6, 1.0), \
                f"Predominantly text content ({text_ratio:.0%})"

        return DocumentType.OTHER, 0.2, f"Mixed block types: {dict(type_counts)}"

    def _keyword_classify(
        self,
        content: str,
        rules: Optional[List[ClassificationRule]] = None
    ) -> Classification:
        """Fallback keyword-based classification.

        Args:
            content: Document content
            rules: Optional classification rules

        Returns:
            Classification result
        """
        if rules is None:
            rules = self._get_default_rules()

        content_lower = content.lower()
        scores = {}

        for rule in rules:
            score = 0
            for keyword in rule.keywords:
                if keyword.lower() in content_lower:
                    score += 1
            if rule.keywords:
                scores[rule.label] = score / len(rule.keywords)
            else:
                scores[rule.label] = 0.0

        if scores:
            best_label = max(scores, key=scores.get)
            confidence = min(scores[best_label], 1.0)
        else:
            best_label = "other"
            confidence = 0.0

        doc_type = self._map_to_document_type(best_label)

        return Classification(
            document_type=doc_type,
            confidence=confidence,
            reasoning=f"Keyword-based classification: {dict(sorted(scores.items(), key=lambda x: -x[1])[:5])}",
            labels=scores,
        )

    def _get_default_rules(self) -> List[ClassificationRule]:
        """Get default classification rules."""
        return [
            ClassificationRule(label="form", description="Forms", keywords=["form", "checkbox", "fill in", "please complete"]),
            ClassificationRule(label="invoice", description="Invoices", keywords=["invoice", "bill to", "amount due", "payment"]),
            ClassificationRule(label="certificate", description="Certificates", keywords=["certificate", "certify", "hereby", "issued"]),
            ClassificationRule(label="medical", description="Medical", keywords=["patient", "medical", "health", "diagnosis"]),
            ClassificationRule(label="presentation", description="Presentations", keywords=["slide", "presentation", "overview"]),
            ClassificationRule(label="report", description="Reports", keywords=["report", "analysis", "findings", "conclusion"]),
            ClassificationRule(label="spreadsheet", description="Spreadsheets", keywords=["total", "sum", "column", "row"]),
        ]

    def _get_default_categories(self) -> List[str]:
        """Get default classification categories."""
        return [
            "form", "invoice", "receipt", "certificate", "medical",
            "presentation", "diagram", "flowchart", "spreadsheet",
            "instructions", "infographic", "report", "other",
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
