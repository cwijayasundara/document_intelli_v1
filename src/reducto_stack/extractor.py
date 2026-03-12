"""Reducto Extract wrapper for structured data extraction.

Reducto Extract provides:
- JSON schema-based extraction directly from documents
- Field-level extraction with confidence
- Citation support for extracted values
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..common.models import ExtractionField, ExtractionResult
from .client import ReductoClient

logger = logging.getLogger(__name__)


class ReductoExtractWrapper:
    """Wrapper for Reducto Extract structured data extraction API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Reducto Extract wrapper.

        Args:
            api_key: Reducto API key. Defaults to REDUCTO_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("REDUCTO_API_KEY")
        self.client = ReductoClient(api_key=self.api_key)

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data using a Pydantic schema.

        Args:
            content: Document content (used as fallback context)
            schema: Pydantic model defining extraction schema
            file_path: Path to original document file (preferred for Reducto)
            upload_ref: Pre-uploaded file reference
            **options: Additional extraction options

        Returns:
            ExtractionResult with extracted fields
        """
        json_schema = schema.model_json_schema()

        return await self.extract_with_json_schema(
            content=content,
            json_schema=json_schema,
            schema_name=schema.__name__,
            file_path=file_path,
            upload_ref=upload_ref,
            **options
        )

    async def extract_with_json_schema(
        self,
        content: str,
        json_schema: Dict[str, Any],
        schema_name: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        **options
    ) -> ExtractionResult:
        """Extract structured data using a JSON schema.

        Reducto extract operates on the uploaded file directly, not on text.
        If file_path or upload_ref is provided, uses the Reducto API.

        Args:
            content: Document content (not directly used by Reducto API)
            json_schema: JSON schema defining fields to extract
            schema_name: Optional name for the schema
            file_path: Path to original document file
            upload_ref: Pre-uploaded file reference
            **options: Additional extraction options

        Returns:
            ExtractionResult with extracted fields
        """
        # Reducto extract needs an upload reference
        if upload_ref is None and file_path:
            file_path = Path(file_path)
            if file_path.exists():
                upload_ref = await self.client.upload(file_path)

        if upload_ref is None:
            logger.warning("No file_path or upload_ref provided; extraction may be limited")
            return ExtractionResult(
                fields={},
                raw_fields=[],
                schema_name=schema_name,
                extraction_confidence=0.0,
            )

        try:
            result = await self.client.extract(
                upload_ref=upload_ref,
                schema=json_schema,
                **options
            )

            return self._process_extract_result(result, schema_name)

        except Exception as e:
            logger.error(f"Reducto extract failed: {str(e)}")
            return ExtractionResult(
                fields={},
                raw_fields=[],
                schema_name=schema_name,
                extraction_confidence=0.0,
            )

    def _process_extract_result(
        self,
        result: Any,
        schema_name: Optional[str] = None
    ) -> ExtractionResult:
        """Process Reducto extract result into ExtractionResult.

        Args:
            result: Raw Reducto extract result
            schema_name: Optional schema name

        Returns:
            ExtractionResult with extracted fields
        """
        # Extract fields from result
        extracted_fields = {}
        if hasattr(result, 'result'):
            if isinstance(result.result, dict):
                extracted_fields = result.result
            elif isinstance(result.result, list) and result.result:
                # If result is a list, take the first item
                extracted_fields = result.result[0] if isinstance(result.result[0], dict) else {}
            elif hasattr(result.result, 'model_dump'):
                extracted_fields = result.result.model_dump()

        # Build raw fields with metadata
        raw_fields = []
        for field_name, value in extracted_fields.items():
            if value is not None:
                raw_fields.append(ExtractionField(
                    name=field_name,
                    value=value,
                    confidence=1.0,
                ))

        # Extract usage info
        credits_used = None
        if hasattr(result, 'usage') and hasattr(result.usage, 'credits'):
            credits_used = result.usage.credits

        return ExtractionResult(
            fields=extracted_fields,
            raw_fields=raw_fields,
            schema_name=schema_name,
            extraction_confidence=1.0 if extracted_fields else 0.0,
        )

    async def extract_key_values(
        self,
        content: str,
        keys: List[str],
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        **options
    ) -> ExtractionResult:
        """Extract specific key-value pairs.

        Args:
            content: Document content
            keys: List of field names to extract
            file_path: Optional original file for direct extraction
            upload_ref: Pre-uploaded file reference
            **options: Additional options

        Returns:
            ExtractionResult with extracted fields
        """
        json_schema = {
            "type": "object",
            "properties": {
                key: {"type": "string", "description": f"Extract value for {key}"}
                for key in keys
            }
        }

        return await self.extract_with_json_schema(
            content=content,
            json_schema=json_schema,
            schema_name="key_value_extraction",
            file_path=file_path,
            upload_ref=upload_ref,
            **options
        )
