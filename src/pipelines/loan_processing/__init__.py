"""Loan processing pipeline.

This module provides end-to-end document processing for loan applications,
including parsing, categorization, extraction, validation, and visualization.
"""

from src.pipelines.loan_processing.schemas import (
    LoanDocumentType,
    IDSchema,
    W2Schema,
    PayStubSchema,
    BankStatementSchema,
    InvestmentStatementSchema,
    DocumentExtractionResult,
)
from src.pipelines.loan_processing.categorizer import LoanDocumentCategorizer
from src.pipelines.loan_processing.extractor import LoanFieldExtractor
from src.pipelines.loan_processing.validator import LoanValidator, LoanValidationResult
from src.pipelines.loan_processing.pipeline import (
    LoanProcessingPipeline,
    LoanApplicationResult,
    ProcessedDocument,
)
from src.pipelines.loan_processing.visualizer import DocumentVisualizer

__all__ = [
    # Schemas
    "LoanDocumentType",
    "IDSchema",
    "W2Schema",
    "PayStubSchema",
    "BankStatementSchema",
    "InvestmentStatementSchema",
    "DocumentExtractionResult",
    # Components
    "LoanDocumentCategorizer",
    "LoanFieldExtractor",
    "LoanValidator",
    "LoanValidationResult",
    "DocumentVisualizer",
    # Pipeline
    "LoanProcessingPipeline",
    "LoanApplicationResult",
    "ProcessedDocument",
]
