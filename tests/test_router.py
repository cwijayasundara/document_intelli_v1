"""Tests for document router."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.common.router import DocumentRouter, ProcessorType, DocumentType


class TestDocumentRouter:
    """Tests for DocumentRouter."""

    @pytest.fixture
    def router(self):
        return DocumentRouter()

    def test_get_file_type(self, router):
        assert router.get_file_type("/path/to/file.pdf") == ".pdf"
        assert router.get_file_type("/path/to/file.PNG") == ".png"
        assert router.get_file_type("document.xlsx") == ".xlsx"

    def test_is_supported_format_llamaindex(self, router):
        assert router.is_supported_format("doc.pdf", ProcessorType.LLAMAINDEX)
        assert router.is_supported_format("img.png", ProcessorType.LLAMAINDEX)
        assert router.is_supported_format("img.jpg", ProcessorType.LLAMAINDEX)
        assert not router.is_supported_format("doc.xlsx", ProcessorType.LLAMAINDEX)

    def test_is_supported_format_landingai(self, router):
        assert router.is_supported_format("doc.pdf", ProcessorType.LANDINGAI)
        assert router.is_supported_format("data.xlsx", ProcessorType.LANDINGAI)
        assert router.is_supported_format("data.csv", ProcessorType.LANDINGAI)

    def test_is_supported_format_gemini(self, router):
        assert router.is_supported_format("doc.pdf", ProcessorType.GEMINI)
        assert router.is_supported_format("img.webp", ProcessorType.GEMINI)
        assert not router.is_supported_format("doc.docx", ProcessorType.GEMINI)

    def test_detect_handwriting_from_filename(self, router):
        assert router.detect_handwriting_from_filename("calculus_answer_sheet.jpg")
        assert router.detect_handwriting_from_filename("handwritten_notes.pdf")
        assert router.detect_handwriting_from_filename("exam_worksheet.png")
        assert not router.detect_handwriting_from_filename("invoice.pdf")
        assert not router.detect_handwriting_from_filename("report.pdf")

    def test_get_classification_hint(self, router):
        assert router.get_classification_hint("invoice_2024.pdf") == DocumentType.INVOICE
        assert router.get_classification_hint("patient_intake.pdf") == DocumentType.MEDICAL
        assert router.get_classification_hint("flowchart.png") == DocumentType.FLOWCHART
        assert router.get_classification_hint("assembly_guide.pdf") == DocumentType.INSTRUCTIONS
        assert router.get_classification_hint("random_document.pdf") is None

    def test_route_forces_processor(self, router, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        processor, options = router.route(
            test_file,
            force_processor=ProcessorType.LANDINGAI
        )
        assert processor == ProcessorType.LANDINGAI

    def test_route_handwriting_to_gemini(self, router, tmp_path):
        # Create a test file with handwriting indicator in name
        test_file = tmp_path / "answer_sheet.jpg"
        test_file.touch()

        processor, options = router.route(test_file)
        assert processor == ProcessorType.GEMINI
        assert options.get("handwriting_mode") is True

    def test_route_standard_document(self, router, tmp_path):
        # Create a standard PDF
        test_file = tmp_path / "report.pdf"
        test_file.touch()

        router_default = DocumentRouter(default_processor=ProcessorType.LLAMAINDEX)
        processor, options = router_default.route(test_file)
        assert processor == ProcessorType.LLAMAINDEX

    def test_route_file_not_found(self, router):
        with pytest.raises(FileNotFoundError):
            router.route("/nonexistent/file.pdf")

    def test_get_recommended_options_llamaindex(self, router, tmp_path):
        test_file = tmp_path / "flowchart.png"
        test_file.touch()

        options = router.get_recommended_options(test_file, ProcessorType.LLAMAINDEX)
        assert options.get("tier") == "agentic_plus"
        assert options.get("multimodal") is True

    def test_get_recommended_options_landingai(self, router, tmp_path):
        test_file = tmp_path / "document.pdf"
        test_file.touch()

        options = router.get_recommended_options(test_file, ProcessorType.LANDINGAI)
        assert options.get("page_level") is True
        assert options.get("include_grounding") is True

    def test_get_recommended_options_gemini(self, router, tmp_path):
        test_file = tmp_path / "handwritten.jpg"
        test_file.touch()

        options = router.get_recommended_options(test_file, ProcessorType.GEMINI)
        assert options.get("handwriting_mode") is True
