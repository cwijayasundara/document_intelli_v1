"""Reducto API client using official SDK.

Uses the reductoai SDK for:
- Upload: Upload documents for processing
- Parse: Document parsing with bounding box grounding
- Extract: Structured extraction with JSON schemas
- Split: Section-based document splitting
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from reducto import Reducto
except ImportError:
    raise ImportError(
        "reductoai package not found. Please install it with:\n"
        "  pip install reductoai\n"
        "Or install all requirements:\n"
        "  pip install -r requirements.txt"
    )


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
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class ReductoClient:
    """Client for Reducto API using official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0
    ):
        """Initialize Reducto client.

        Args:
            api_key: Reducto API key. Defaults to REDUCTO_API_KEY env var.
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("REDUCTO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set REDUCTO_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Set env var so the SDK picks it up
        os.environ["REDUCTO_API_KEY"] = self.api_key
        self.client = Reducto()

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file."""
        ext = file_path.suffix.lower()
        return MIME_TYPES.get(ext, "application/octet-stream")

    async def upload(self, file_path: Union[str, Path]) -> Any:
        """Upload a file to Reducto for processing.

        Args:
            file_path: Path to the document file

        Returns:
            Upload reference for use in subsequent API calls
        """
        file_path = Path(file_path)
        return await asyncio.to_thread(self.client.upload, file=file_path)

    async def parse(self, upload_ref: Any, **options) -> Any:
        """Call Reducto parse endpoint.

        Args:
            upload_ref: Upload reference from upload()
            **options: Additional parse options (enhance, formatting, settings, retrieval)

        Returns:
            Parse result object
        """
        return await asyncio.to_thread(
            self.client.parse.run, input=upload_ref, **options
        )

    async def extract(
        self,
        upload_ref: Any,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **options
    ) -> Any:
        """Call Reducto extract endpoint.

        Args:
            upload_ref: Upload reference from upload()
            schema: JSON schema defining fields to extract
            system_prompt: Optional system prompt for extraction
            **options: Additional extract options

        Returns:
            Extract result object
        """
        instructions = {"schema": schema}
        if system_prompt:
            instructions["system_prompt"] = system_prompt

        return await asyncio.to_thread(
            self.client.extract.run,
            input=upload_ref,
            instructions=instructions,
            **options
        )

    async def split(
        self,
        upload_ref: Any,
        split_description: List[Dict[str, str]],
        split_rules: Optional[str] = None,
        **options
    ) -> Any:
        """Call Reducto split endpoint.

        Args:
            upload_ref: Upload reference from upload()
            split_description: List of {name, description} dicts for section categories
            split_rules: Optional custom splitting rules prompt
            **options: Additional split options

        Returns:
            Split result object
        """
        kwargs = {
            "input": upload_ref,
            "split_description": split_description,
        }
        if split_rules:
            kwargs["split_rules"] = split_rules
        kwargs.update(options)

        return await asyncio.to_thread(
            self.client.split.run, **kwargs
        )

    async def health_check(self) -> bool:
        """Check API health."""
        return True
