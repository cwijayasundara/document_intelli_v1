"""Unified data models for document extraction pipeline.

These models provide a consistent interface for output from both
LlamaIndex and LandingAI stacks, enabling direct comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Classification categories for documents."""
    FORM = "form"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    PRESENTATION = "presentation"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    SPREADSHEET = "spreadsheet"
    CERTIFICATE = "certificate"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    INFOGRAPHIC = "infographic"
    INSTRUCTIONS = "instructions"
    HANDWRITTEN = "handwritten"
    OTHER = "other"


class BoundingBox(BaseModel):
    """Bounding box coordinates for grounding data."""
    x: float = Field(..., description="X coordinate (left)")
    y: float = Field(..., description="Y coordinate (top)")
    width: float = Field(..., description="Width of the box")
    height: float = Field(..., description="Height of the box")
    page: int = Field(default=1, description="Page number (1-indexed)")

    @property
    def x2(self) -> float:
        """Right edge coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge coordinate."""
        return self.y + self.height


class GroundingReference(BaseModel):
    """Reference to source location in original document."""
    text: str = Field(..., description="The extracted text")
    bbox: BoundingBox = Field(..., description="Location in source document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GroundingData(BaseModel):
    """Grounding data with bounding boxes for source traceability."""
    references: List[GroundingReference] = Field(default_factory=list)
    page_dimensions: Optional[Dict[int, Dict[str, float]]] = Field(
        default=None,
        description="Page dimensions: {page_num: {width, height}}"
    )


class Chunk(BaseModel):
    """A semantic chunk of document content."""
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    chunk_type: str = Field(default="text", description="Type: text, table, figure, etc.")
    page_numbers: List[int] = Field(default_factory=list, description="Source pages")
    category: Optional[str] = Field(default=None, description="Classification category")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    grounding: Optional[List[GroundingReference]] = Field(default=None)

    # Semantic chunking metadata
    start_index: Optional[int] = Field(default=None, description="Start position in source")
    end_index: Optional[int] = Field(default=None, description="End position in source")
    embedding: Optional[List[float]] = Field(default=None, exclude=True)


class Classification(BaseModel):
    """Document classification result."""
    document_type: DocumentType = Field(..., description="Primary document type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: Optional[str] = Field(default=None, description="Explanation for classification")
    secondary_types: List[DocumentType] = Field(
        default_factory=list,
        description="Additional applicable types"
    )
    labels: Dict[str, float] = Field(
        default_factory=dict,
        description="All label scores"
    )


class ExtractionField(BaseModel):
    """A single extracted field with optional grounding."""
    name: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    grounding: Optional[GroundingReference] = None


class ExtractionResult(BaseModel):
    """Structured extraction results."""
    fields: Dict[str, Any] = Field(default_factory=dict)
    raw_fields: List[ExtractionField] = Field(default_factory=list)
    schema_name: Optional[str] = Field(default=None)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DocumentMetadata(BaseModel):
    """Metadata about the source document and processing."""
    source_path: str = Field(..., description="Original file path")
    file_name: str = Field(..., description="File name")
    file_type: str = Field(..., description="File extension")
    file_size_bytes: int = Field(default=0)
    page_count: int = Field(default=1)

    # Processing metadata
    processor: str = Field(..., description="Stack used: llamaindex or landingai")
    processing_time_ms: float = Field(default=0.0)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    api_credits_used: Optional[float] = Field(default=None)

    # Document characteristics
    has_tables: bool = Field(default=False)
    has_images: bool = Field(default=False)
    has_handwriting: bool = Field(default=False)
    language: str = Field(default="en")

    # Error tracking
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Unified output from either LlamaIndex or LandingAI stack.

    This is the primary output model that both stacks produce,
    enabling direct comparison of results.
    """
    # Core content
    markdown: str = Field(..., description="Full parsed content as Markdown")
    raw_text: Optional[str] = Field(default=None, description="Plain text without formatting")

    # Processed components
    chunks: List[Chunk] = Field(default_factory=list, description="Semantic segments")
    classification: Optional[Classification] = Field(default=None)
    extraction: Optional[ExtractionResult] = Field(default=None)

    # Metadata and grounding
    metadata: DocumentMetadata = Field(...)
    grounding: Optional[GroundingData] = Field(default=None)

    # Page-level content
    pages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-page parsed content"
    )

    # Images and figures extracted
    images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted images/figures with descriptions"
    )

    # Tables extracted
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted tables as structured data"
    )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_by_page(self, page_num: int) -> List[Chunk]:
        """Get all chunks from a specific page."""
        return [c for c in self.chunks if page_num in c.page_numbers]

    def get_chunks_by_category(self, category: str) -> List[Chunk]:
        """Get all chunks with a specific category."""
        return [c for c in self.chunks if c.category == category]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def has_grounding(self) -> bool:
        return self.grounding is not None and len(self.grounding.references) > 0


# Extraction schemas for common document types

class InvoiceSchema(BaseModel):
    """Schema for invoice extraction."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: str = "USD"


class FormSchema(BaseModel):
    """Schema for general form extraction."""
    form_title: Optional[str] = None
    form_date: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict)
    checkboxes: Dict[str, bool] = Field(default_factory=dict)
    signatures: List[Dict[str, Any]] = Field(default_factory=list)


class CertificateSchema(BaseModel):
    """Schema for certificate extraction."""
    certificate_type: Optional[str] = None
    certificate_number: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    issuer: Optional[str] = None
    recipient: Optional[str] = None
    description: Optional[str] = None
    signatures: List[Dict[str, Any]] = Field(default_factory=list)


class MedicalFormSchema(BaseModel):
    """Schema for medical/patient intake form extraction."""
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None
    insurance_info: Optional[Dict[str, str]] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    consent_signed: bool = False


class PresentationSchema(BaseModel):
    """Schema for presentation/investor deck extraction."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    company_name: Optional[str] = None
    date: Optional[str] = None
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    bullet_points: List[str] = Field(default_factory=list)


# Schema registry for automatic schema selection
SCHEMA_REGISTRY: Dict[DocumentType, type] = {
    DocumentType.INVOICE: InvoiceSchema,
    DocumentType.FORM: FormSchema,
    DocumentType.CERTIFICATE: CertificateSchema,
    DocumentType.MEDICAL: MedicalFormSchema,
    DocumentType.PRESENTATION: PresentationSchema,
}


def get_schema_for_type(doc_type: DocumentType) -> Optional[type]:
    """Get the appropriate extraction schema for a document type."""
    return SCHEMA_REGISTRY.get(doc_type)
