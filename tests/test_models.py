"""Tests for common data models."""

import pytest
from datetime import datetime

from src.common.models import (
    BoundingBox,
    Chunk,
    Classification,
    DocumentMetadata,
    DocumentType,
    ExtractionField,
    ExtractionResult,
    GroundingData,
    GroundingReference,
    ParsedDocument,
    InvoiceSchema,
    FormSchema,
    get_schema_for_type,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_create_bounding_box(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50, page=1)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.page == 1

    def test_bounding_box_x2_y2(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self):
        chunk = Chunk(
            id="chunk_001",
            content="This is test content",
            chunk_type="text",
            page_numbers=[1, 2],
            category="introduction"
        )
        assert chunk.id == "chunk_001"
        assert chunk.content == "This is test content"
        assert chunk.chunk_type == "text"
        assert chunk.page_numbers == [1, 2]
        assert chunk.category == "introduction"

    def test_chunk_defaults(self):
        chunk = Chunk(id="test", content="content")
        assert chunk.chunk_type == "text"
        assert chunk.page_numbers == []
        assert chunk.category is None
        assert chunk.metadata == {}


class TestClassification:
    """Tests for Classification model."""

    def test_create_classification(self):
        classification = Classification(
            document_type=DocumentType.INVOICE,
            confidence=0.95,
            reasoning="Contains invoice number and line items"
        )
        assert classification.document_type == DocumentType.INVOICE
        assert classification.confidence == 0.95
        assert "invoice" in classification.reasoning.lower()

    def test_classification_with_secondary_types(self):
        classification = Classification(
            document_type=DocumentType.FORM,
            confidence=0.8,
            secondary_types=[DocumentType.MEDICAL, DocumentType.CONTRACT]
        )
        assert len(classification.secondary_types) == 2


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_create_metadata(self):
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            file_name="doc.pdf",
            file_type=".pdf",
            processor="llamaindex"
        )
        assert metadata.source_path == "/path/to/doc.pdf"
        assert metadata.file_name == "doc.pdf"
        assert metadata.processor == "llamaindex"

    def test_metadata_defaults(self):
        metadata = DocumentMetadata(
            source_path="/doc.pdf",
            file_name="doc.pdf",
            file_type=".pdf",
            processor="test"
        )
        assert metadata.page_count == 1
        assert metadata.has_tables is False
        assert metadata.has_images is False
        assert metadata.language == "en"


class TestParsedDocument:
    """Tests for ParsedDocument model."""

    def test_create_parsed_document(self):
        metadata = DocumentMetadata(
            source_path="/doc.pdf",
            file_name="doc.pdf",
            file_type=".pdf",
            processor="test"
        )
        doc = ParsedDocument(
            markdown="# Test Document\n\nContent here.",
            metadata=metadata
        )
        assert "Test Document" in doc.markdown
        assert doc.total_chunks == 0
        assert doc.has_grounding is False

    def test_parsed_document_with_chunks(self):
        metadata = DocumentMetadata(
            source_path="/doc.pdf",
            file_name="doc.pdf",
            file_type=".pdf",
            processor="test"
        )
        chunks = [
            Chunk(id="1", content="Chunk 1", page_numbers=[1]),
            Chunk(id="2", content="Chunk 2", page_numbers=[1]),
            Chunk(id="3", content="Chunk 3", page_numbers=[2]),
        ]
        doc = ParsedDocument(
            markdown="Full content",
            chunks=chunks,
            metadata=metadata
        )
        assert doc.total_chunks == 3
        assert len(doc.get_chunks_by_page(1)) == 2
        assert len(doc.get_chunks_by_page(2)) == 1

    def test_get_chunk_by_id(self):
        metadata = DocumentMetadata(
            source_path="/doc.pdf",
            file_name="doc.pdf",
            file_type=".pdf",
            processor="test"
        )
        chunks = [
            Chunk(id="chunk_a", content="A"),
            Chunk(id="chunk_b", content="B"),
        ]
        doc = ParsedDocument(markdown="", chunks=chunks, metadata=metadata)

        found = doc.get_chunk_by_id("chunk_b")
        assert found is not None
        assert found.content == "B"

        not_found = doc.get_chunk_by_id("chunk_c")
        assert not_found is None


class TestSchemaRegistry:
    """Tests for schema registry."""

    def test_get_invoice_schema(self):
        schema = get_schema_for_type(DocumentType.INVOICE)
        assert schema == InvoiceSchema

    def test_get_form_schema(self):
        schema = get_schema_for_type(DocumentType.FORM)
        assert schema == FormSchema

    def test_get_unknown_type_returns_none(self):
        schema = get_schema_for_type(DocumentType.OTHER)
        assert schema is None


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_extraction_result(self):
        result = ExtractionResult(
            fields={"invoice_number": "INV-001", "total": 100.00},
            schema_name="InvoiceSchema",
            extraction_confidence=0.9
        )
        assert result.fields["invoice_number"] == "INV-001"
        assert result.schema_name == "InvoiceSchema"
        assert result.extraction_confidence == 0.9

    def test_extraction_with_raw_fields(self):
        raw_fields = [
            ExtractionField(name="field1", value="value1", confidence=0.95),
            ExtractionField(name="field2", value="value2", confidence=0.85),
        ]
        result = ExtractionResult(
            fields={"field1": "value1", "field2": "value2"},
            raw_fields=raw_fields
        )
        assert len(result.raw_fields) == 2
        assert result.raw_fields[0].confidence == 0.95
