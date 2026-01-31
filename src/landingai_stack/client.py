"""LandingAI ADE API client using official SDK.

Uses the landingai-ade SDK for:
- Parse: Document parsing with grounding
- Extract: Structured extraction with JSON schemas
- Split: Section classification and splitting

Supports both US and EU endpoints for data residency.
"""

import mimetypes
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from landingai_ade import LandingAIADE
except ImportError:
    raise ImportError(
        "landingai-ade package not found. Please install it with:\n"
        "  pip install landingai-ade\n"
        "Or install all requirements:\n"
        "  pip install -r requirements.txt"
    )


class ADERegion(str, Enum):
    """ADE API regions."""
    US = "us"
    EU = "eu"


# Environment URLs
ENVIRONMENT_URLS = {
    ADERegion.US: "https://api.va.landing.ai",
    ADERegion.EU: "https://api.va.eu-west-1.landing.ai",
}

# MIME type mapping
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}


class ADEClient:
    """Client for LandingAI ADE API using official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: ADERegion = ADERegion.US,
        timeout: float = 120.0
    ):
        """Initialize ADE client.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
            region: API region (US or EU)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("LANDINGAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set LANDINGAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.region = region
        self.base_url = ENVIRONMENT_URLS[region]

        self.client = LandingAIADE(
            apikey=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )

    async def close(self):
        """Close the client."""
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file."""
        ext = file_path.suffix.lower()
        return MIME_TYPES.get(ext, "application/octet-stream")

    async def parse(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        split_by_page: bool = False,
        **options
    ) -> Dict[str, Any]:
        """Call ADE Parse endpoint.

        Args:
            file_path: Path to document file
            file_content: Raw file bytes (alternative to file_path)
            file_name: Filename (required if using file_content)
            url: URL to document (alternative to file)
            split_by_page: Return page-level results
            **options: Additional API options

        Returns:
            Parse result dictionary
        """
        # Prepare file for upload
        if file_path:
            file_path = Path(file_path)
            with open(file_path, 'rb') as f:
                file_content = f.read()
            file_name = file_path.name
            mime_type = self._get_mime_type(file_path)
        elif file_content and file_name:
            mime_type = MIME_TYPES.get(Path(file_name).suffix.lower(), "application/octet-stream")
        elif url:
            # Use URL directly
            result = self.client.parse(document_url=url, split="page" if split_by_page else None)
            return self._parse_response_to_dict(result)
        else:
            raise ValueError("Must provide file_path, file_content with file_name, or url")

        # Create file tuple for SDK
        document = (file_name, file_content, mime_type)

        # Call parse API
        result = self.client.parse(
            document=document,
            split="page" if split_by_page else None
        )

        return self._parse_response_to_dict(result)

    def _parse_response_to_dict(self, result) -> Dict[str, Any]:
        """Convert SDK response to dictionary."""
        response = {
            "markdown": "",
            "text": "",
            "chunks": [],
            "grounding": {},
            "metadata": {
                "page_count": 1
            }
        }

        # Extract markdown
        if hasattr(result, 'markdown') and result.markdown:
            if isinstance(result.markdown, str):
                response["markdown"] = result.markdown
            else:
                response["markdown"] = str(result.markdown)

        # Use markdown as text if no separate text
        response["text"] = response["markdown"]

        # Extract chunks if available
        if hasattr(result, 'chunks') and result.chunks:
            response["chunks"] = [
                {
                    "content": c.text if hasattr(c, 'text') else str(c),
                    "id": str(i),
                    "grounding": c.grounding if hasattr(c, 'grounding') else None
                }
                for i, c in enumerate(result.chunks)
            ]

        # Extract grounding if available
        if hasattr(result, 'grounding') and result.grounding:
            response["grounding"] = {
                "items": [
                    g.model_dump() if hasattr(g, 'model_dump') else g
                    for g in result.grounding
                ] if isinstance(result.grounding, list) else []
            }

        return response

    async def extract(
        self,
        content: str,
        schema: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Call ADE Extract endpoint.

        Args:
            content: Document content (markdown or text)
            schema: JSON schema defining fields to extract
            file_path: Optional original file for grounding (not used - extract works on text)
            file_content: Optional raw file bytes (not used)
            file_name: Optional filename (not used)
            **options: Additional API options

        Returns:
            Extraction result dictionary
        """
        import json
        import logging
        logger = logging.getLogger(__name__)

        if hasattr(self.client, 'extract') and callable(self.client.extract):
            try:
                # Convert schema to JSON string
                schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
                logger.info(f"Calling ADE extract with schema: {schema_str[:200]}...")

                # The LandingAI ADE SDK extract() takes 'markdown' and 'schema' parameters
                # It works on markdown/text content, not on document files directly
                result = self.client.extract(
                    markdown=content,
                    schema=schema_str
                )

                logger.info(f"ADE extract result type: {type(result)}")

                if hasattr(result, 'model_dump'):
                    data = result.model_dump()
                    logger.info(f"Extracted data keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                    return data
                elif isinstance(result, dict):
                    logger.info(f"Result is dict with keys: {list(result.keys())}")
                    return result
                else:
                    logger.info(f"Result is: {result}")
                    return {"data": result}

            except Exception as e:
                logger.error(f"ADE extract failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {"error": str(e), "data": {}}

        logger.warning("Extract method not available on client")
        return {"error": "Extract not supported", "data": {}}

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        file_path: Optional[Union[str, Path]] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Call ADE Split endpoint.

        Args:
            content: Document content (markdown or text)
            categories: List of category labels for classification
            file_path: Optional original file
            file_content: Optional raw file bytes
            file_name: Optional filename
            **options: Additional API options

        Returns:
            Split result dictionary
        """
        if hasattr(self.client, 'split') and callable(self.client.split):
            try:
                # Prepare document if provided
                document = None
                if file_path:
                    file_path = Path(file_path)
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    file_name = file_path.name
                    mime_type = self._get_mime_type(file_path)
                    document = (file_name, file_content, mime_type)
                elif file_content and file_name:
                    mime_type = MIME_TYPES.get(Path(file_name).suffix.lower(), "application/octet-stream")
                    document = (file_name, file_content, mime_type)

                result = self.client.split(
                    document=document,
                    **options
                )
                if hasattr(result, 'model_dump'):
                    return result.model_dump()
                return {"sections": result}
            except Exception as e:
                return {"error": str(e), "sections": []}

        return {"error": "Split not supported", "sections": []}

    async def health_check(self) -> bool:
        """Check API health."""
        return True
