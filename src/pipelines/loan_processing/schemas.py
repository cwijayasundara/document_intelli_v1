"""Pydantic schemas for loan document processing.

Defines document types and extraction schemas for each loan document category.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LoanDocumentType(str, Enum):
    """Classification categories for loan documents."""
    ID = "ID"
    W2 = "W2"
    PAY_STUB = "pay_stub"
    BANK_STATEMENT = "bank_statement"
    INVESTMENT_STATEMENT = "investment_statement"
    UNKNOWN = "unknown"


# Document-specific extraction schemas

class IDSchema(BaseModel):
    """Schema for ID document extraction (driver's license, passport, etc.)."""
    name: Optional[str] = Field(None, description="Full name on the ID")
    issuer: Optional[str] = Field(None, description="Issuing authority (state, country)")
    issue_date: Optional[str] = Field(None, description="Date the ID was issued")
    expiration_date: Optional[str] = Field(None, description="Expiration date of the ID")
    identifier: Optional[str] = Field(None, description="ID number (license number, passport number)")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    address: Optional[str] = Field(None, description="Address on the ID")


class W2Schema(BaseModel):
    """Schema for W2 tax form extraction."""
    employee_name: Optional[str] = Field(None, description="Employee's full name")
    employee_ssn: Optional[str] = Field(None, description="Employee's SSN (last 4 digits)")
    employer_name: Optional[str] = Field(None, description="Employer's name")
    employer_ein: Optional[str] = Field(None, description="Employer Identification Number")
    w2_year: Optional[int] = Field(None, description="Tax year of the W2")
    wages_box_1: Optional[float] = Field(None, description="Box 1: Wages, tips, other compensation")
    federal_tax_withheld: Optional[float] = Field(None, description="Box 2: Federal income tax withheld")
    social_security_wages: Optional[float] = Field(None, description="Box 3: Social security wages")
    medicare_wages: Optional[float] = Field(None, description="Box 5: Medicare wages and tips")


class PayStubSchema(BaseModel):
    """Schema for pay stub extraction."""
    employee_name: Optional[str] = Field(None, description="Employee's full name")
    employer_name: Optional[str] = Field(None, description="Employer's name")
    pay_period_start: Optional[str] = Field(None, description="Start date of pay period")
    pay_period_end: Optional[str] = Field(None, description="End date of pay period")
    pay_date: Optional[str] = Field(None, description="Date of payment")
    gross_pay: Optional[float] = Field(None, description="Gross pay amount")
    net_pay: Optional[float] = Field(None, description="Net pay amount after deductions")
    ytd_gross: Optional[float] = Field(None, description="Year-to-date gross pay")
    ytd_net: Optional[float] = Field(None, description="Year-to-date net pay")
    hours_worked: Optional[float] = Field(None, description="Hours worked in pay period")
    hourly_rate: Optional[float] = Field(None, description="Hourly pay rate")


class BankStatementSchema(BaseModel):
    """Schema for bank statement extraction."""
    account_owner: Optional[str] = Field(None, description="Name of account holder")
    bank_name: Optional[str] = Field(None, description="Name of the bank")
    account_number: Optional[str] = Field(None, description="Account number (may be masked)")
    account_type: Optional[str] = Field(None, description="Account type (checking, savings)")
    statement_period_start: Optional[str] = Field(None, description="Statement period start date")
    statement_period_end: Optional[str] = Field(None, description="Statement period end date")
    opening_balance: Optional[float] = Field(None, description="Opening balance")
    closing_balance: Optional[float] = Field(None, description="Closing/ending balance")
    total_deposits: Optional[float] = Field(None, description="Total deposits during period")
    total_withdrawals: Optional[float] = Field(None, description="Total withdrawals during period")


class InvestmentStatementSchema(BaseModel):
    """Schema for investment/brokerage statement extraction."""
    account_owner: Optional[str] = Field(None, description="Name of account holder")
    institution_name: Optional[str] = Field(None, description="Name of investment institution")
    account_number: Optional[str] = Field(None, description="Account number (may be masked)")
    statement_date: Optional[str] = Field(None, description="Statement date")
    statement_period_start: Optional[str] = Field(None, description="Statement period start date")
    statement_period_end: Optional[str] = Field(None, description="Statement period end date")
    total_value: Optional[float] = Field(None, description="Total account value")
    total_cash: Optional[float] = Field(None, description="Cash/money market balance")
    total_securities: Optional[float] = Field(None, description="Total securities value")
    change_in_value: Optional[float] = Field(None, description="Change in value during period")


# Schema registry mapping document types to their schemas
LOAN_SCHEMA_REGISTRY: Dict[LoanDocumentType, type] = {
    LoanDocumentType.ID: IDSchema,
    LoanDocumentType.W2: W2Schema,
    LoanDocumentType.PAY_STUB: PayStubSchema,
    LoanDocumentType.BANK_STATEMENT: BankStatementSchema,
    LoanDocumentType.INVESTMENT_STATEMENT: InvestmentStatementSchema,
}


def get_schema_for_loan_document(doc_type: LoanDocumentType) -> Optional[type]:
    """Get the appropriate extraction schema for a loan document type."""
    return LOAN_SCHEMA_REGISTRY.get(doc_type)


class DocumentExtractionResult(BaseModel):
    """Result of extracting fields from a document."""
    document_type: LoanDocumentType = Field(..., description="Classified document type")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Classification confidence")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Extracted field values")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw API response")
    grounding: Optional[List[Dict[str, Any]]] = Field(None, description="Bounding box grounding data")
    file_path: Optional[str] = Field(None, description="Source file path")
    file_name: Optional[str] = Field(None, description="Source file name")
    error: Optional[str] = Field(None, description="Error message if extraction failed")
