"""Document categorization for loan applications.

Uses LandingAI ADE to classify loan documents into their appropriate types.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field

from src.pipelines.loan_processing.schemas import LoanDocumentType

logger = logging.getLogger(__name__)


class CategorizationSchema(BaseModel):
    """Schema for document categorization."""
    document_type: str = Field(
        ...,
        description=(
            "The type of loan document. Must be one of: "
            "'ID' (driver's license, passport, state ID), "
            "'W2' (W-2 tax form), "
            "'pay_stub' (paycheck stub, earnings statement), "
            "'bank_statement' (checking or savings account statement), "
            "'investment_statement' (brokerage, 401k, IRA statement), "
            "'unknown' (cannot determine document type)"
        )
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this document type was chosen"
    )


class LoanDocumentCategorizer:
    """Categorizes loan documents using LandingAI ADE or Reducto."""

    def __init__(self, api_key: Optional[str] = None, processor: str = "landingai"):
        """Initialize the categorizer.

        Args:
            api_key: API key for the selected processor. Defaults to env var.
            processor: Which processor to use - "landingai" or "reducto".
        """
        self.processor_name = processor
        if processor == "reducto":
            from src.reducto_stack.client import ReductoClient
            self.client = ReductoClient(api_key=api_key)
        else:
            from src.landingai_stack.client import ADEClient
            self.client = ADEClient(api_key=api_key)

    async def categorize(
        self,
        file_path: Path,
        markdown_content: Optional[str] = None
    ) -> Tuple[LoanDocumentType, float, str]:
        """Categorize a loan document.

        Args:
            file_path: Path to the document file
            markdown_content: Optional pre-parsed markdown content

        Returns:
            Tuple of (document_type, confidence, reasoning)
        """
        logger.info(f"Categorizing document: {file_path}")

        # Build the JSON schema for categorization
        json_schema = {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": ["ID", "W2", "pay_stub", "bank_statement", "investment_statement", "unknown"],
                    "description": (
                        "The type of loan document. Determine based on content: "
                        "ID = driver's license, passport, state ID with photo and personal info; "
                        "W2 = W-2 tax form showing annual wages and taxes withheld; "
                        "pay_stub = paycheck stub showing pay period, gross/net pay; "
                        "bank_statement = bank account statement with transactions and balance; "
                        "investment_statement = brokerage or retirement account statement with holdings"
                    )
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the classification"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this document type was chosen"
                }
            },
            "required": ["document_type", "confidence", "reasoning"]
        }

        try:
            if self.processor_name == "reducto":
                # Reducto extracts from the uploaded file directly
                from src.reducto_stack.extractor import ReductoExtractWrapper
                extractor = ReductoExtractWrapper(client=self.client)
                result_obj = await extractor.extract(
                    content=markdown_content or "",
                    schema=json_schema,
                    file_path=file_path
                )
                data = result_obj.fields
            else:
                # Use ADE extract with the categorization schema
                result = await self.client.extract(
                    content=markdown_content or "",
                    schema=json_schema,
                    file_path=file_path
                )

                logger.debug(f"Categorization result: {result}")

                # Parse the result - SDK returns data under 'extraction' key
                if "extraction" in result:
                    data = result["extraction"]
                elif "data" in result:
                    data = result["data"]
                elif "document_type" in result:
                    data = result
                else:
                    logger.warning(f"Unexpected categorization result format: {result}")
                    return LoanDocumentType.UNKNOWN, 0.0, "Failed to parse categorization result"

            # Extract values
            doc_type_str = data.get("document_type", "unknown")
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "No reasoning provided")

            # Map to enum
            try:
                doc_type = LoanDocumentType(doc_type_str)
            except ValueError:
                logger.warning(f"Unknown document type: {doc_type_str}")
                doc_type = LoanDocumentType.UNKNOWN

            logger.info(f"Categorized as {doc_type.value} with {confidence:.1%} confidence")
            return doc_type, confidence, reasoning

        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return LoanDocumentType.UNKNOWN, 0.0, f"Categorization failed: {str(e)}"

    async def categorize_batch(
        self,
        documents: list[Tuple[Path, Optional[str]]]
    ) -> list[Tuple[LoanDocumentType, float, str]]:
        """Categorize multiple documents.

        Args:
            documents: List of (file_path, optional_markdown) tuples

        Returns:
            List of (document_type, confidence, reasoning) tuples
        """
        results = []
        for file_path, markdown in documents:
            result = await self.categorize(file_path, markdown)
            results.append(result)
        return results
