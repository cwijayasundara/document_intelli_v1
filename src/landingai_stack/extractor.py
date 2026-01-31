"""ADE Extract wrapper for structured data extraction.

ADE Extract provides:
- JSON schema-based extraction
- Key-value pair extraction
- Grounding references for extracted values
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..common.models import (
    BoundingBox,
    ExtractionField,
    ExtractionResult,
    GroundingReference,
)
from .client import ADEClient, ADERegion


class ADEExtractWrapper:
    """Wrapper for ADE Extract structured data extraction API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: ADERegion = ADERegion.US
    ):
        """Initialize ADE Extract wrapper.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
            region: API region (US or EU)
        """
        self.api_key = api_key or os.environ.get("LANDINGAI_API_KEY")
        self.region = region
        self.client = ADEClient(api_key=self.api_key, region=region)

    async def extract(
        self,
        content: str,
        schema: Type[BaseModel],
        file_path: Optional[Union[str, Path]] = None,
        include_grounding: bool = True,
        **options
    ) -> ExtractionResult:
        """Extract structured data using a Pydantic schema.

        Args:
            content: Document content (markdown or text)
            schema: Pydantic model defining extraction schema
            file_path: Optional original file for grounding
            include_grounding: Include bounding box references
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
            file_path=file_path,
            include_grounding=include_grounding,
            **options
        )

    async def extract_with_json_schema(
        self,
        content: str,
        json_schema: Dict[str, Any],
        schema_name: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        include_grounding: bool = True,
        **options
    ) -> ExtractionResult:
        """Extract structured data using a JSON schema.

        Args:
            content: Document content
            json_schema: JSON schema defining fields to extract
            schema_name: Optional name for the schema
            file_path: Optional original file for grounding
            include_grounding: Include bounding box references
            **options: Additional extraction options

        Returns:
            ExtractionResult with extracted fields
        """
        # Call ADE Extract API
        result = await self.client.extract(
            content=content,
            schema=json_schema,
            file_path=file_path,
            include_grounding=include_grounding,
            **options
        )

        # Parse response
        extracted_fields = result.get("data", result.get("fields", {}))
        grounding_data = result.get("grounding", {})

        # Convert to ExtractionField objects
        raw_fields = []
        for field_name, value in extracted_fields.items():
            if value is not None:
                # Find grounding for this field if available
                grounding = None
                if include_grounding and field_name in grounding_data:
                    grounding = self._parse_field_grounding(grounding_data[field_name])

                raw_fields.append(ExtractionField(
                    name=field_name,
                    value=value,
                    confidence=result.get("confidence", {}).get(field_name, 1.0),
                    grounding=grounding
                ))

        return ExtractionResult(
            fields=extracted_fields,
            raw_fields=raw_fields,
            schema_name=schema_name,
            extraction_confidence=result.get("overall_confidence", 1.0)
        )

    def _parse_field_grounding(self, grounding_info: Dict[str, Any]) -> Optional[GroundingReference]:
        """Parse grounding info for a single field.

        Args:
            grounding_info: Grounding data for a field

        Returns:
            GroundingReference or None
        """
        if not grounding_info:
            return None

        bbox_data = grounding_info.get("bbox", grounding_info.get("bounding_box", {}))

        if not bbox_data:
            return None

        bbox = BoundingBox(
            x=bbox_data.get("x", bbox_data.get("left", 0)),
            y=bbox_data.get("y", bbox_data.get("top", 0)),
            width=bbox_data.get("width", bbox_data.get("w", 0)),
            height=bbox_data.get("height", bbox_data.get("h", 0)),
            page=grounding_info.get("page", 1)
        )

        return GroundingReference(
            text=grounding_info.get("text", ""),
            bbox=bbox,
            confidence=grounding_info.get("confidence", 1.0)
        )

    async def extract_key_values(
        self,
        content: str,
        keys: List[str],
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> ExtractionResult:
        """Extract specific key-value pairs.

        Simpler interface when you just need to extract specific fields
        without a full schema.

        Args:
            content: Document content
            keys: List of field names to extract
            file_path: Optional original file for grounding
            **options: Additional options

        Returns:
            ExtractionResult with extracted fields
        """
        # Build simple schema from keys
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
            **options
        )

    async def extract_from_file(
        self,
        file_path: Union[str, Path],
        schema: Type[BaseModel],
        **options
    ) -> ExtractionResult:
        """Extract structured data directly from a file.

        Combines parsing and extraction in one call for convenience.

        Args:
            file_path: Path to document file
            schema: Pydantic model defining extraction schema
            **options: Additional options

        Returns:
            ExtractionResult with extracted fields
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # First parse the file
        from .parser import ADEParseWrapper
        parser = ADEParseWrapper(api_key=self.api_key, region=self.region)

        parse_result = await parser.parse(file_path, include_grounding=True)
        await parser.close()

        # Then extract from the parsed content
        return await self.extract(
            content=parse_result["markdown"],
            schema=schema,
            file_path=file_path,
            **options
        )

    async def close(self):
        """Close the client connection."""
        await self.client.close()
