"""Field extraction for loan documents.

Extracts relevant fields based on document type using type-specific schemas.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from src.pipelines.loan_processing.schemas import (
    LoanDocumentType,
    IDSchema,
    W2Schema,
    PayStubSchema,
    BankStatementSchema,
    InvestmentStatementSchema,
    DocumentExtractionResult,
    get_schema_for_loan_document,
)

logger = logging.getLogger(__name__)


def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to a JSON schema for extraction."""
    schema = model.model_json_schema()

    # Simplify for LandingAI ADE
    json_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    # Get properties from the Pydantic schema
    properties = schema.get("properties", {})
    for field_name, field_info in properties.items():
        # Get the field type
        field_type = field_info.get("type", "string")
        if field_type == "integer":
            field_type = "number"

        # Handle nullable fields
        any_of = field_info.get("anyOf", [])
        if any_of:
            for option in any_of:
                if option.get("type") and option["type"] != "null":
                    field_type = option["type"]
                    if field_type == "integer":
                        field_type = "number"
                    break

        json_schema["properties"][field_name] = {
            "type": field_type,
            "description": field_info.get("description", f"Extract {field_name}")
        }

    return json_schema


class LoanFieldExtractor:
    """Extracts fields from loan documents based on their type."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the extractor.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
        """
        from src.landingai_stack.client import ADEClient
        self.client = ADEClient(api_key=api_key)

    async def extract(
        self,
        file_path: Path,
        document_type: LoanDocumentType,
        markdown_content: Optional[str] = None
    ) -> DocumentExtractionResult:
        """Extract fields from a document based on its type.

        Args:
            file_path: Path to the document file
            document_type: The classified document type
            markdown_content: Optional pre-parsed markdown content

        Returns:
            DocumentExtractionResult with extracted fields
        """
        logger.info(f"Extracting fields from {document_type.value} document: {file_path}")

        # Get the appropriate schema for this document type
        schema_class = get_schema_for_loan_document(document_type)

        if schema_class is None:
            logger.warning(f"No schema for document type: {document_type}")
            return DocumentExtractionResult(
                document_type=document_type,
                confidence=0.0,
                fields={},
                file_path=str(file_path),
                file_name=file_path.name,
                error="No extraction schema available for this document type"
            )

        # Convert Pydantic schema to JSON schema
        json_schema = pydantic_to_json_schema(schema_class)
        logger.debug(f"Using JSON schema: {json_schema}")

        try:
            # Use ADE extract with the type-specific schema
            result = await self.client.extract(
                content=markdown_content or "",
                schema=json_schema,
                file_path=file_path
            )

            logger.debug(f"Extraction result: {result}")

            # Parse the result - SDK returns data under 'extraction' key
            if "extraction" in result:
                fields = result["extraction"]
            elif "data" in result:
                fields = result["data"]
            elif "error" in result and result["error"]:
                return DocumentExtractionResult(
                    document_type=document_type,
                    confidence=0.0,
                    fields={},
                    raw_response=result,
                    file_path=str(file_path),
                    file_name=file_path.name,
                    error=result["error"]
                )
            else:
                # Assume the result itself is the extracted data
                fields = {k: v for k, v in result.items() if k not in ["error", "grounding", "extraction_metadata", "metadata"]}

            # Extract grounding/references if available
            grounding = result.get("grounding", [])

            # extraction_metadata contains reference info but in dict format, not list
            # Convert to list format if needed for compatibility
            if not grounding and "extraction_metadata" in result:
                extraction_meta = result["extraction_metadata"]
                # Convert dict to list of grounding items
                grounding_list = []
                for field_name, field_meta in extraction_meta.items():
                    if isinstance(field_meta, dict) and "references" in field_meta:
                        grounding_list.append({
                            "field": field_name,
                            "value": field_meta.get("value"),
                            "references": field_meta.get("references", [])
                        })
                grounding = grounding_list if grounding_list else None

            # Clean up null/None values
            cleaned_fields = {k: v for k, v in fields.items() if v is not None}

            logger.info(f"Extracted {len(cleaned_fields)} fields from {document_type.value}")

            return DocumentExtractionResult(
                document_type=document_type,
                confidence=1.0 if cleaned_fields else 0.0,
                fields=cleaned_fields,
                raw_response=result,
                grounding=grounding if grounding else None,
                file_path=str(file_path),
                file_name=file_path.name
            )

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return DocumentExtractionResult(
                document_type=document_type,
                confidence=0.0,
                fields={},
                file_path=str(file_path),
                file_name=file_path.name,
                error=str(e)
            )

    async def extract_with_custom_schema(
        self,
        file_path: Path,
        json_schema: Dict[str, Any],
        markdown_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract fields using a custom JSON schema.

        Args:
            file_path: Path to the document file
            json_schema: Custom JSON schema for extraction
            markdown_content: Optional pre-parsed markdown content

        Returns:
            Dictionary of extracted fields
        """
        logger.info(f"Extracting with custom schema from: {file_path}")

        try:
            result = await self.client.extract(
                content=markdown_content or "",
                schema=json_schema,
                file_path=file_path
            )

            if "data" in result:
                return result["data"]
            return result

        except Exception as e:
            logger.error(f"Custom extraction failed: {e}")
            return {"error": str(e)}
