"""Document routing logic for directing documents to appropriate processors.

Routes documents to:
- Gemini for handwritten content
- LlamaIndex or LandingAI for standard documents
"""

import mimetypes
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

from .models import DocumentType


class ProcessorType(str, Enum):
    """Available document processors."""
    LLAMAINDEX = "llamaindex"
    LANDINGAI = "landingai"
    GEMINI = "gemini"
    REDUCTO = "reducto"


class DocumentRouter:
    """Routes documents to appropriate processing stack based on characteristics."""

    # Supported file types by processor
    LLAMAINDEX_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    LANDINGAI_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".csv"}
    GEMINI_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    REDUCTO_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".docx", ".xlsx", ".pptx"}

    # Keywords that suggest handwriting
    HANDWRITING_INDICATORS = {
        "handwritten", "handwriting", "written", "filled in",
        "answer_sheet", "answer sheet", "worksheet", "exam", "test"
    }

    def __init__(
        self,
        default_processor: ProcessorType = ProcessorType.LLAMAINDEX,
        prefer_gemini_for_handwriting: bool = True
    ):
        """Initialize router.

        Args:
            default_processor: Default processor for standard documents
            prefer_gemini_for_handwriting: Route handwritten docs to Gemini
        """
        self.default_processor = default_processor
        self.prefer_gemini_for_handwriting = prefer_gemini_for_handwriting

    def get_file_type(self, file_path: Union[str, Path]) -> str:
        """Get file extension."""
        return Path(file_path).suffix.lower()

    def get_mime_type(self, file_path: Union[str, Path]) -> Optional[str]:
        """Get MIME type of file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type

    def is_supported_format(
        self,
        file_path: Union[str, Path],
        processor: ProcessorType
    ) -> bool:
        """Check if file format is supported by processor."""
        ext = self.get_file_type(file_path)

        format_map = {
            ProcessorType.LLAMAINDEX: self.LLAMAINDEX_FORMATS,
            ProcessorType.LANDINGAI: self.LANDINGAI_FORMATS,
            ProcessorType.GEMINI: self.GEMINI_FORMATS,
            ProcessorType.REDUCTO: self.REDUCTO_FORMATS,
        }

        return ext in format_map.get(processor, set())

    def detect_handwriting_from_filename(self, file_path: Union[str, Path]) -> bool:
        """Detect potential handwriting from filename patterns."""
        filename = Path(file_path).stem.lower()
        return any(indicator in filename for indicator in self.HANDWRITING_INDICATORS)

    def route(
        self,
        file_path: Union[str, Path],
        force_processor: Optional[ProcessorType] = None,
        has_handwriting: Optional[bool] = None
    ) -> Tuple[ProcessorType, dict]:
        """Determine which processor should handle this document.

        Args:
            file_path: Path to document
            force_processor: Override automatic routing
            has_handwriting: Explicit handwriting flag (overrides detection)

        Returns:
            Tuple of (ProcessorType, options dict)
        """
        file_path = Path(file_path)
        ext = self.get_file_type(file_path)
        options = {}

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Force processor if specified
        if force_processor:
            if not self.is_supported_format(file_path, force_processor):
                raise ValueError(
                    f"File type {ext} not supported by {force_processor.value}"
                )
            return force_processor, options

        # Detect handwriting
        is_handwritten = has_handwriting
        if is_handwritten is None:
            is_handwritten = self.detect_handwriting_from_filename(file_path)

        # Route handwritten documents to Gemini
        if is_handwritten and self.prefer_gemini_for_handwriting:
            if self.is_supported_format(file_path, ProcessorType.GEMINI):
                options["handwriting_mode"] = True
                return ProcessorType.GEMINI, options

        # Route based on file type support
        if self.is_supported_format(file_path, self.default_processor):
            return self.default_processor, options

        # Fallback: try other processors
        for processor in ProcessorType:
            if self.is_supported_format(file_path, processor):
                return processor, options

        raise ValueError(f"No processor supports file type: {ext}")

    def get_classification_hint(self, file_path: Union[str, Path]) -> Optional[DocumentType]:
        """Get a classification hint based on filename patterns."""
        filename = Path(file_path).stem.lower()

        hints = {
            "invoice": DocumentType.INVOICE,
            "receipt": DocumentType.RECEIPT,
            "certificate": DocumentType.CERTIFICATE,
            "form": DocumentType.FORM,
            "patient": DocumentType.MEDICAL,
            "intake": DocumentType.MEDICAL,
            "flowchart": DocumentType.FLOWCHART,
            "diagram": DocumentType.DIAGRAM,
            "presentation": DocumentType.PRESENTATION,
            "investor": DocumentType.PRESENTATION,
            "assembly": DocumentType.INSTRUCTIONS,
            "manual": DocumentType.INSTRUCTIONS,
            "infographic": DocumentType.INFOGRAPHIC,
            "spreadsheet": DocumentType.SPREADSHEET,
            "sales": DocumentType.SPREADSHEET,
            "answer": DocumentType.HANDWRITTEN,
            "handwritten": DocumentType.HANDWRITTEN,
            "report": DocumentType.REPORT,
        }

        for keyword, doc_type in hints.items():
            if keyword in filename:
                return doc_type

        return None

    def get_recommended_options(
        self,
        file_path: Union[str, Path],
        processor: ProcessorType
    ) -> dict:
        """Get recommended processing options based on document characteristics."""
        options = {}
        hint = self.get_classification_hint(file_path)
        ext = self.get_file_type(file_path)

        # LlamaIndex specific options
        if processor == ProcessorType.LLAMAINDEX:
            # Use Agentic Plus for complex documents
            if hint in {
                DocumentType.DIAGRAM, DocumentType.FLOWCHART,
                DocumentType.SPREADSHEET, DocumentType.PRESENTATION
            }:
                options["tier"] = "agentic_plus"
                options["multimodal"] = True
            else:
                options["tier"] = "agentic"

            # Enable image handling for visual documents
            if ext in {".png", ".jpg", ".jpeg"}:
                options["multimodal"] = True

        # LandingAI specific options
        elif processor == ProcessorType.LANDINGAI:
            # Enable page-level processing for multi-page PDFs
            if ext == ".pdf":
                options["page_level"] = True

            # Enable grounding for audit trails
            options["include_grounding"] = True

        # Reducto specific options
        elif processor == ProcessorType.REDUCTO:
            options["include_grounding"] = True
            if hint in {
                DocumentType.DIAGRAM, DocumentType.FLOWCHART,
                DocumentType.SPREADSHEET
            }:
                options["enhance"] = {
                    "summarize_figures": True,
                    "agentic": [{"scope": "table"}],
                }

        # Gemini specific options
        elif processor == ProcessorType.GEMINI:
            if hint == DocumentType.HANDWRITTEN:
                options["handwriting_mode"] = True

            # Use pro model for complex visual documents
            if hint in {
                DocumentType.DIAGRAM, DocumentType.FLOWCHART,
                DocumentType.SPREADSHEET
            }:
                options["model"] = "gemini-2.0-flash"
            else:
                options["model"] = "gemini-2.0-flash"

        return options
