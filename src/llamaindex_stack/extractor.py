"""LlamaExtract wrapper for structured data extraction.

LlamaExtract supports:
- Pydantic schema-based extraction
- JSON schema extraction
- Automatic field detection
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel
from llama_cloud import LlamaCloud

from ..common.models import ExtractionField, ExtractionResult

logger = logging.getLogger(__name__)


class LlamaExtractWrapper:
    """Wrapper for LlamaExtract structured data extraction API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        extraction_mode: Literal["FAST", "BALANCED", "PREMIUM", "MULTIMODAL"] = "BALANCED",
        extract_model: Optional[str] = None
    ):
        """Initialize LlamaExtract wrapper.

        Args:
            api_key: LlamaCloud API key. Defaults to LLAMA_CLOUD_API_KEY env var.
            extraction_mode: Mode for extraction (FAST, BALANCED, PREMIUM, MULTIMODAL).
            extract_model: Optional specific model to use for extraction.
        """
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set LLAMA_CLOUD_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = LlamaCloud(api_key=self.api_key)
        self.extraction_mode = extraction_mode
        self.extract_model = extract_model

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        **options
    ) -> ExtractionResult:
        """Extract structured data from content using a Pydantic schema.

        Args:
            content: Document content (markdown or text)
            schema: Pydantic model defining the extraction schema
            **options: Additional extraction options

        Returns:
            ExtractionResult with extracted fields
        """
        # Convert Pydantic schema to JSON schema
        json_schema = schema.model_json_schema()

        return await self.extract_with_json_schema(
            content=content,
            json_schema=json_schema,
            schema_name=schema.__name__,
            **options
        )

    async def extract_with_json_schema(
        self,
        content: str,
        json_schema: Dict[str, Any],
        schema_name: Optional[str] = None,
        extraction_mode: Optional[str] = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data using a JSON schema.

        Args:
            content: Document content
            json_schema: JSON schema defining fields to extract
            schema_name: Optional name for the schema
            extraction_mode: Override extraction mode (FAST, BALANCED, PREMIUM, MULTIMODAL)
            **options: Additional extraction options

        Returns:
            ExtractionResult with extracted fields
        """
        try:
            # Build extraction config
            config = {
                "extraction_mode": extraction_mode or self.extraction_mode,
                "extraction_target": "PER_DOC",
                "cite_sources": False,
                "confidence_scores": True,
            }

            # Add model if specified
            if self.extract_model:
                config["extract_model"] = self.extract_model

            logger.info(f"Calling LlamaCloud extraction with mode={config['extraction_mode']}")
            logger.debug(f"Schema has {len(json_schema.get('properties', {}))} properties")

            # Call extraction API with correct parameters
            result = self.client.extraction.extract(
                config=config,
                data_schema=json_schema,
                text=content,
                **options
            )

            logger.info(f"Extraction API returned result type: {type(result)}")

            # Parse result - the API returns a JobGetResultResponse
            extracted_fields = {}
            raw_fields = []
            confidence = 1.0

            # Try different result formats
            if hasattr(result, 'extraction'):
                # Newer API format
                extraction_data = result.extraction
                if isinstance(extraction_data, dict):
                    extracted_fields = extraction_data
                elif hasattr(extraction_data, 'data'):
                    extracted_fields = extraction_data.data if isinstance(extraction_data.data, dict) else {}
            elif hasattr(result, 'data'):
                extracted_fields = result.data if isinstance(result.data, dict) else {}
            elif hasattr(result, 'output'):
                extracted_fields = result.output if isinstance(result.output, dict) else {}
            elif isinstance(result, dict):
                extracted_fields = result.get('extraction', result.get('data', result.get('output', result)))

            logger.info(f"Extracted {len(extracted_fields)} fields")

            # Convert to ExtractionField objects with confidence
            for field_name, value in extracted_fields.items():
                if value is not None:
                    raw_fields.append(ExtractionField(
                        name=field_name,
                        value=value,
                        confidence=1.0,
                        grounding=None
                    ))

            return ExtractionResult(
                fields=extracted_fields,
                raw_fields=raw_fields,
                schema_name=schema_name,
                extraction_confidence=confidence
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            logger.exception("Full extraction error traceback:")
            # Return empty extraction on error
            return ExtractionResult(
                fields={},
                raw_fields=[],
                schema_name=schema_name,
                extraction_confidence=0.0
            )

    async def extract_from_file(
        self,
        file_path: Union[str, Path],
        schema: Type[BaseModel],
        extraction_mode: Optional[str] = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data directly from a file.

        Args:
            file_path: Path to document file
            schema: Pydantic model defining extraction schema
            extraction_mode: Override extraction mode
            **options: Additional options

        Returns:
            ExtractionResult with extracted fields
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Convert schema
        json_schema = schema.model_json_schema()

        try:
            # Build extraction config
            config = {
                "extraction_mode": extraction_mode or self.extraction_mode,
                "extraction_target": "PER_DOC",
                "cite_sources": False,
                "confidence_scores": True,
            }

            if self.extract_model:
                config["extract_model"] = self.extract_model

            logger.info(f"Extracting from file: {file_path.name}")

            # Extract from file using correct API parameters
            result = self.client.extraction.extract(
                config=config,
                data_schema=json_schema,
                file=(file_path.name, file_content, "application/octet-stream"),
                **options
            )

            # Parse result
            extracted_fields = {}
            raw_fields = []

            if hasattr(result, 'extraction'):
                extraction_data = result.extraction
                if isinstance(extraction_data, dict):
                    extracted_fields = extraction_data
                elif hasattr(extraction_data, 'data'):
                    extracted_fields = extraction_data.data if isinstance(extraction_data.data, dict) else {}
            elif hasattr(result, 'data'):
                extracted_fields = result.data if isinstance(result.data, dict) else {}
            elif isinstance(result, dict):
                extracted_fields = result.get('extraction', result.get('data', result))

            for field_name, value in extracted_fields.items():
                if value is not None:
                    raw_fields.append(ExtractionField(
                        name=field_name,
                        value=value,
                        confidence=1.0,
                        grounding=None
                    ))

            return ExtractionResult(
                fields=extracted_fields,
                raw_fields=raw_fields,
                schema_name=schema.__name__,
                extraction_confidence=1.0
            )

        except Exception as e:
            logger.error(f"File extraction failed: {str(e)}")
            return ExtractionResult(
                fields={},
                raw_fields=[],
                schema_name=schema.__name__,
                extraction_confidence=0.0
            )

    async def auto_extract(
        self,
        content: str,
        hints: Optional[List[str]] = None,
        **options
    ) -> ExtractionResult:
        """Automatically detect and extract structured data.

        Uses LLM to identify extractable fields without a predefined schema.

        Args:
            content: Document content
            hints: Optional hints about what to extract
            **options: Additional options

        Returns:
            ExtractionResult with auto-detected fields
        """
        # Build a generic schema for auto-extraction
        auto_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }

        if hints:
            # Add hinted fields to schema
            for hint in hints:
                field_name = hint.lower().replace(" ", "_")
                auto_schema["properties"][field_name] = {
                    "type": "string",
                    "description": f"Extract: {hint}"
                }

        return await self.extract_with_json_schema(
            content=content,
            json_schema=auto_schema,
            schema_name="auto_extracted",
            **options
        )

    def create_schema_from_fields(
        self,
        fields: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a JSON schema from a list of field definitions.

        Args:
            fields: List of field definitions, each with:
                - name: Field name
                - type: Field type (string, number, boolean, array, object)
                - description: Optional description
                - required: Whether field is required

        Returns:
            JSON schema dictionary
        """
        properties = {}
        required = []

        for field in fields:
            field_name = field["name"]
            field_type = field.get("type", "string")
            description = field.get("description", "")

            properties[field_name] = {
                "type": field_type,
                "description": description
            }

            if field.get("required", False):
                required.append(field_name)

        schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return schema
