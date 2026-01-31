"""Validation logic for loan applications.

Validates consistency across documents in a loan application:
- Name matching across all documents
- Year verification (documents from recent years)
- Asset totals calculation
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from src.pipelines.loan_processing.schemas import LoanDocumentType, DocumentExtractionResult

logger = logging.getLogger(__name__)


class LoanValidationResult(BaseModel):
    """Result of validating a loan application's documents."""
    # Name validation
    name_match: bool = Field(
        False,
        description="Whether all names across documents match"
    )
    names_found: List[str] = Field(
        default_factory=list,
        description="All unique names found across documents"
    )
    name_issues: List[str] = Field(
        default_factory=list,
        description="Issues found with name matching"
    )

    # Year validation
    years_valid: bool = Field(
        False,
        description="Whether all documents are from acceptable years"
    )
    years_found: List[int] = Field(
        default_factory=list,
        description="All years found in documents"
    )
    year_issues: List[str] = Field(
        default_factory=list,
        description="Issues found with document years"
    )

    # Asset totals
    total_bank_balance: float = Field(
        0.0,
        description="Sum of all bank account balances"
    )
    total_investment_value: float = Field(
        0.0,
        description="Sum of all investment account values"
    )
    total_assets: float = Field(
        0.0,
        description="Total assets (bank + investment)"
    )

    # Income information
    annual_income: Optional[float] = Field(
        None,
        description="Annual income from W2 or annualized from pay stubs"
    )
    monthly_income: Optional[float] = Field(
        None,
        description="Monthly income estimate"
    )

    # Overall validation
    validation_passed: bool = Field(
        False,
        description="Whether all validations passed"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="All issues found during validation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-critical warnings"
    )

    # Document summary
    documents_validated: int = Field(
        0,
        description="Number of documents validated"
    )
    document_types_found: List[str] = Field(
        default_factory=list,
        description="Types of documents found"
    )


class LoanValidator:
    """Validates loan application documents for consistency."""

    def __init__(
        self,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        require_id: bool = True,
        require_income_proof: bool = True
    ):
        """Initialize the validator.

        Args:
            min_year: Minimum acceptable year for documents (default: 2 years ago)
            max_year: Maximum acceptable year for documents (default: current year)
            require_id: Whether an ID document is required
            require_income_proof: Whether income proof (W2 or pay stub) is required
        """
        current_year = datetime.now().year
        self.min_year = min_year or (current_year - 2)
        self.max_year = max_year or current_year
        self.require_id = require_id
        self.require_income_proof = require_income_proof

    def validate(
        self,
        extraction_results: List[DocumentExtractionResult]
    ) -> LoanValidationResult:
        """Validate a set of document extraction results.

        Args:
            extraction_results: List of extraction results from loan documents

        Returns:
            LoanValidationResult with validation details
        """
        logger.info(f"Validating {len(extraction_results)} documents")

        result = LoanValidationResult()
        result.documents_validated = len(extraction_results)

        # Collect data from all documents
        all_names: List[str] = []
        all_years: List[int] = []
        bank_balances: List[float] = []
        investment_values: List[float] = []
        w2_wages: Optional[float] = None
        pay_stub_gross: Optional[float] = None
        doc_types_found: Set[str] = set()

        for doc_result in extraction_results:
            doc_types_found.add(doc_result.document_type.value)

            # Extract names based on document type
            names = self._extract_names(doc_result)
            all_names.extend(names)

            # Extract years based on document type
            years = self._extract_years(doc_result)
            all_years.extend(years)

            # Extract financial data
            if doc_result.document_type == LoanDocumentType.BANK_STATEMENT:
                balance = doc_result.fields.get("closing_balance")
                if balance is not None:
                    bank_balances.append(float(balance))

            elif doc_result.document_type == LoanDocumentType.INVESTMENT_STATEMENT:
                value = doc_result.fields.get("total_value")
                if value is not None:
                    investment_values.append(float(value))

            elif doc_result.document_type == LoanDocumentType.W2:
                wages = doc_result.fields.get("wages_box_1")
                if wages is not None:
                    w2_wages = float(wages)

            elif doc_result.document_type == LoanDocumentType.PAY_STUB:
                gross = doc_result.fields.get("gross_pay")
                if gross is not None:
                    pay_stub_gross = float(gross)

        result.document_types_found = sorted(doc_types_found)

        # Validate names
        result.names_found = list(set(all_names))
        result.name_match, result.name_issues = self._validate_names(all_names)

        # Validate years
        result.years_found = sorted(set(all_years))
        result.years_valid, result.year_issues = self._validate_years(all_years)

        # Calculate totals
        result.total_bank_balance = sum(bank_balances)
        result.total_investment_value = sum(investment_values)
        result.total_assets = result.total_bank_balance + result.total_investment_value

        # Calculate income
        if w2_wages is not None:
            result.annual_income = w2_wages
            result.monthly_income = w2_wages / 12
        elif pay_stub_gross is not None:
            # Estimate annual income (assuming bi-weekly pay)
            result.annual_income = pay_stub_gross * 26
            result.monthly_income = pay_stub_gross * 26 / 12

        # Check required documents
        if self.require_id and LoanDocumentType.ID.value not in doc_types_found:
            result.issues.append("Missing required document: ID (driver's license, passport, etc.)")

        if self.require_income_proof:
            has_w2 = LoanDocumentType.W2.value in doc_types_found
            has_pay_stub = LoanDocumentType.PAY_STUB.value in doc_types_found
            if not has_w2 and not has_pay_stub:
                result.issues.append("Missing required income proof: W2 or pay stub")

        # Collect all issues
        result.issues.extend(result.name_issues)
        result.issues.extend(result.year_issues)

        # Determine overall pass/fail
        result.validation_passed = (
            result.name_match and
            result.years_valid and
            len(result.issues) == 0
        )

        if result.validation_passed:
            logger.info("Validation passed")
        else:
            logger.warning(f"Validation failed with {len(result.issues)} issues")

        return result

    def _extract_names(self, doc_result: DocumentExtractionResult) -> List[str]:
        """Extract all name fields from a document."""
        names = []
        fields = doc_result.fields

        # Common name field patterns
        name_fields = [
            "name", "full_name", "employee_name", "account_owner",
            "account_holder", "customer_name", "holder_name"
        ]

        for field in name_fields:
            if field in fields and fields[field]:
                name = str(fields[field]).strip()
                if name:
                    names.append(name)

        return names

    def _extract_years(self, doc_result: DocumentExtractionResult) -> List[int]:
        """Extract all year information from a document."""
        years = []
        fields = doc_result.fields

        # Direct year fields
        year_fields = ["w2_year", "investment_year", "tax_year", "year"]
        for field in year_fields:
            if field in fields and fields[field]:
                try:
                    years.append(int(fields[field]))
                except (ValueError, TypeError):
                    pass

        # Extract years from date fields
        date_fields = [
            "issue_date", "expiration_date", "statement_date",
            "statement_period_end", "pay_date", "end_date"
        ]
        for field in date_fields:
            if field in fields and fields[field]:
                year = self._extract_year_from_date(str(fields[field]))
                if year:
                    years.append(year)

        return years

    def _extract_year_from_date(self, date_str: str) -> Optional[int]:
        """Extract a year from a date string."""
        # Try various date formats
        patterns = [
            r"\b(20\d{2})\b",  # 4-digit year starting with 20
            r"\b(19\d{2})\b",  # 4-digit year starting with 19
        ]

        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                return int(match.group(1))

        return None

    def _validate_names(self, names: List[str]) -> tuple[bool, List[str]]:
        """Validate that names across documents match.

        Returns:
            Tuple of (names_match, list of issues)
        """
        if not names:
            return True, []  # No names to validate

        # Normalize names for comparison
        normalized = [self._normalize_name(n) for n in names]
        unique_normalized = set(normalized)

        if len(unique_normalized) == 1:
            return True, []

        # Check for minor variations
        issues = []
        if len(unique_normalized) > 1:
            issues.append(
                f"Name mismatch detected: {', '.join(sorted(set(names)))}"
            )

        return False, issues

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Convert to lowercase
        name = name.lower()

        # Remove common titles and suffixes
        prefixes = ["mr.", "mrs.", "ms.", "dr.", "mr", "mrs", "ms", "dr"]
        suffixes = ["jr.", "sr.", "ii", "iii", "iv", "jr", "sr"]

        words = name.split()
        if words and words[0] in prefixes:
            words = words[1:]
        if words and words[-1] in suffixes:
            words = words[:-1]

        # Remove extra whitespace and punctuation
        name = " ".join(words)
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name).strip()

        return name

    def _validate_years(self, years: List[int]) -> tuple[bool, List[str]]:
        """Validate that document years are within acceptable range.

        Returns:
            Tuple of (years_valid, list of issues)
        """
        if not years:
            return True, []  # No years to validate

        issues = []
        for year in set(years):
            if year < self.min_year:
                issues.append(
                    f"Document from {year} is too old (minimum: {self.min_year})"
                )
            elif year > self.max_year:
                issues.append(
                    f"Document from {year} is in the future (maximum: {self.max_year})"
                )

        return len(issues) == 0, issues
