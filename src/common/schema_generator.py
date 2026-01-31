"""Automatic schema generation from document content.

Uses LLM to analyze document content and derive a comprehensive
extraction schema without losing any information.
"""

import json
import os
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


class FieldDefinition(BaseModel):
    """Definition of a single extractable field."""
    name: str = Field(..., description="Field name (snake_case)")
    field_type: str = Field(..., description="Type: string, number, date, boolean, list, object")
    description: str = Field(..., description="What this field represents")
    is_required: bool = Field(default=False, description="Whether the field is required")
    is_list: bool = Field(default=False, description="Whether this field contains multiple values")


class DerivedSchema(BaseModel):
    """Schema derived from document analysis."""
    document_type: str = Field(..., description="Detected document type")
    fields: List[FieldDefinition] = Field(..., description="List of extractable fields")
    reasoning: str = Field(..., description="Explanation of schema derivation")


SCHEMA_DERIVATION_PROMPT = """Analyze this document and extract ALL fields that contain information.

Return a JSON object with this EXACT structure:
{{
    "document_type": "<type>",
    "reasoning": "<why this type>",
    "fields": [
        {{"name": "<snake_case_name>", "field_type": "string", "description": "<what it is>", "is_required": true, "is_list": false}}
    ]
}}

For the document below, identify EVERY piece of data including:
- Document IDs, certificate numbers, reference numbers
- All names (exporter, importer, producer, companies, people)
- All addresses (full addresses, cities, countries)
- All dates (issue date, departure date, expiry)
- Contact info (phone, fax, email)
- Financial info (amounts, values, currencies, FOB/CIF)
- Product details (descriptions, quantities, HS codes)
- Transportation (vessel, port of loading, port of discharge)
- Any table data as individual fields
- Signatures, stamps, official marks

Document:
---
{content}
---

Output ONLY the JSON object, nothing else."""


class SchemaGenerator:
    """Generates extraction schemas from document content using LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Initialize the schema generator.

        Args:
            model: LLM model to use for analysis.
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            temperature: Model temperature for consistency. None uses model default.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Build kwargs - only include temperature if specified
        kwargs = {
            "model": model,
            "model_provider": "openai",
            "api_key": self.api_key
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        self.llm = init_chat_model(**kwargs)

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response, handling various formats.

        Args:
            response_text: Raw response text from LLM.

        Returns:
            Cleaned JSON string.
        """
        import re

        text = response_text.strip()

        # Try to extract from markdown code blocks first
        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                return match.group(1).strip()

        if "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                return match.group(1).strip()

        # Try to find JSON object pattern
        # Look for content between first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace + 1]

        # Return as-is if no patterns found
        return text

    def derive_schema(self, content: str) -> DerivedSchema:
        """Analyze document content and derive an extraction schema.

        Args:
            content: Document content (markdown or plain text).

        Returns:
            DerivedSchema with document type, fields, and reasoning.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Truncate very long content to fit in context
        max_chars = 15000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated...]"

        prompt = SCHEMA_DERIVATION_PROMPT.format(content=content)

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a document analysis expert. Always respond with valid JSON only, no explanations."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content
            logger.debug(f"Raw LLM response: {response_text[:500]}...")

            # Extract JSON from response
            json_text = self._extract_json_from_response(response_text)
            logger.debug(f"Extracted JSON: {json_text[:500]}...")

            # Parse JSON
            data = json.loads(json_text)

            # Validate required fields exist
            if "document_type" not in data:
                data["document_type"] = "unknown"
            if "fields" not in data:
                data["fields"] = []
            if "reasoning" not in data:
                data["reasoning"] = "Schema derived from document analysis."

            return DerivedSchema(**data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"Response text was: {response_text[:1000] if 'response_text' in dir() else 'N/A'}")

            # Try pattern-based fallback
            logger.info("Trying pattern-based schema derivation...")
            pattern_schema = self._derive_schema_from_patterns(content)
            if pattern_schema:
                logger.info(f"Pattern-based derivation found {len(pattern_schema.fields)} fields")
                return pattern_schema

            # Return a basic schema if all else fails
            return DerivedSchema(
                document_type="unknown",
                fields=[
                    FieldDefinition(
                        name="content",
                        field_type="string",
                        description="Full document content",
                        is_required=True,
                        is_list=False
                    )
                ],
                reasoning=f"Failed to parse LLM response as JSON: {str(e)}. Returning basic schema."
            )
        except Exception as e:
            logger.error(f"Schema derivation failed: {str(e)}")

            # Try pattern-based fallback
            logger.info("Trying pattern-based schema derivation...")
            pattern_schema = self._derive_schema_from_patterns(content)
            if pattern_schema:
                logger.info(f"Pattern-based derivation found {len(pattern_schema.fields)} fields")
                return pattern_schema

            return DerivedSchema(
                document_type="unknown",
                fields=[
                    FieldDefinition(
                        name="content",
                        field_type="string",
                        description="Full document content",
                        is_required=True,
                        is_list=False
                    )
                ],
                reasoning=f"Failed to derive schema: {str(e)}. Returning basic schema."
            )

    def create_pydantic_model(
        self,
        schema: DerivedSchema,
        model_name: str = "DynamicExtraction"
    ) -> Type[BaseModel]:
        """Create a Pydantic model from the derived schema.

        Args:
            schema: The derived schema definition.
            model_name: Name for the generated model class.

        Returns:
            A dynamically created Pydantic model class.
        """
        field_definitions = {}

        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "date": str,  # Keep as string for flexibility
            "boolean": bool,
            "object": dict,
            "list": list,
        }

        for field in schema.fields:
            python_type = type_mapping.get(field.field_type, str)

            # Handle list types
            if field.is_list:
                python_type = List[python_type]

            # Make optional if not required
            if not field.is_required:
                python_type = Optional[python_type]
                default = None
            else:
                default = ...

            field_definitions[field.name] = (
                python_type,
                Field(default=default, description=field.description)
            )

        return create_model(model_name, **field_definitions)

    def derive_and_create_model(
        self,
        content: str,
        model_name: str = "DynamicExtraction"
    ) -> tuple[DerivedSchema, Type[BaseModel]]:
        """Convenience method to derive schema and create model in one step.

        Args:
            content: Document content to analyze.
            model_name: Name for the generated model class.

        Returns:
            Tuple of (DerivedSchema, Pydantic model class).
        """
        schema = self.derive_schema(content)
        model = self.create_pydantic_model(schema, model_name)
        return schema, model

    def _derive_schema_from_patterns(self, content: str) -> DerivedSchema:
        """Fallback method to derive schema using regex patterns.

        Args:
            content: Document content.

        Returns:
            DerivedSchema with detected fields.
        """
        import re

        fields = []
        detected_type = "document"

        # Common patterns to detect
        patterns = {
            # IDs and numbers
            "certificate_number": r'certificate\s*(?:no|number|#)[:\s]*([A-Z0-9-]+)',
            "invoice_number": r'invoice\s*(?:no|number|#)[:\s]*([A-Z0-9-]+)',
            "reference_number": r'ref(?:erence)?\s*(?:no|number|#)?[:\s]*([A-Z0-9-]+)',
            "hs_code": r'hs\s*code[:\s]*(\d{4,10}(?:\.\d+)?)',

            # Dates
            "date": r'date[:\s]*(\w+\.?\s*\d{1,2},?\s*\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            "departure_date": r'departure\s*date[:\s]*(\w+\.?\s*\d{1,2},?\s*\d{4})',
            "issue_date": r'issue[d]?\s*(?:date|on)?[:\s]*(\w+\.?\s*\d{1,2},?\s*\d{4})',

            # Names and parties
            "exporter_name": r"exporter['']?s?\s*name[:\s]*([^\n]+)",
            "importer_name": r"importer['']?s?\s*name[:\s]*([^\n]+)",
            "producer_name": r"producer['']?s?\s*name[:\s]*([^\n]+)",
            "company_name": r'(?:company|corp|ltd|inc)[:\s]*([^\n]+)',

            # Addresses
            "address": r'address[:\s]*([^\n]+(?:\n[^\n]+)?)',
            "country": r'country[:\s]*([A-Za-z\s]+)',
            "city": r'city[:\s]*([A-Za-z\s]+)',

            # Contact
            "phone": r'(?:tel|phone)[:\s]*([\d\s\-\+\(\)]+)',
            "fax": r'fax[:\s]*([\d\s\-\+\(\)]+)',
            "email": r'email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',

            # Financial
            "total_amount": r'total[:\s]*\$?([\d,]+\.?\d*)',
            "value": r'value[:\s]*(?:USD|EUR|GBP)?\s*\$?([\d,]+\.?\d*)',
            "currency": r'(USD|EUR|GBP|CNY)',
            "fob_value": r'fob[:\s]*(?:USD)?\s*\$?([\d,]+\.?\d*)',

            # Transportation
            "vessel_name": r'vessel[:\s]*([^\n]+)',
            "port_of_loading": r'port\s*of\s*loading[:\s]*([^\n]+)',
            "port_of_discharge": r'port\s*of\s*discharge[:\s]*([^\n]+)',

            # Products
            "description": r'description[:\s]*([^\n]+)',
            "quantity": r'quantity[:\s]*([\d,]+\s*\w+)',
            "weight": r'(?:gross\s*)?weight[:\s]*([\d,]+\.?\d*\s*\w+)',
        }

        content_lower = content.lower()

        # Detect document type
        if 'certificate of origin' in content_lower:
            detected_type = 'certificate_of_origin'
        elif 'invoice' in content_lower:
            detected_type = 'invoice'
        elif 'receipt' in content_lower:
            detected_type = 'receipt'
        elif 'form' in content_lower:
            detected_type = 'form'

        # Find matches for each pattern
        for field_name, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                fields.append(FieldDefinition(
                    name=field_name,
                    field_type="string",
                    description=f"Extracted {field_name.replace('_', ' ')}",
                    is_required=True,
                    is_list=len(matches) > 1
                ))

        # If we found meaningful fields, return them
        if len(fields) >= 3:
            return DerivedSchema(
                document_type=detected_type,
                fields=fields,
                reasoning=f"Schema derived using pattern matching. Found {len(fields)} fields."
            )

        # Otherwise return None to indicate fallback should be used
        return None

    def schema_to_json_schema(self, schema: DerivedSchema) -> Dict[str, Any]:
        """Convert derived schema to JSON Schema format.

        Useful for LandingAI which uses JSON Schema.

        Args:
            schema: The derived schema definition.

        Returns:
            JSON Schema dictionary.
        """
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "date": "string",
            "boolean": "boolean",
            "object": "object",
            "list": "array",
        }

        properties = {}
        required = []

        for field in schema.fields:
            json_type = type_mapping.get(field.field_type, "string")

            if field.is_list:
                prop = {
                    "type": "array",
                    "items": {"type": json_type},
                    "description": field.description
                }
            else:
                prop = {
                    "type": json_type,
                    "description": field.description
                }

            properties[field.name] = prop

            if field.is_required:
                required.append(field.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
