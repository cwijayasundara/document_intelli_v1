"""Document processing pipelines.

This package contains specialized pipelines for different document types.
"""

from src.pipelines.loan_processing import (
    LoanProcessingPipeline,
    LoanDocumentType,
    LoanApplicationResult,
    LoanValidationResult,
)

__all__ = [
    "LoanProcessingPipeline",
    "LoanDocumentType",
    "LoanApplicationResult",
    "LoanValidationResult",
]
