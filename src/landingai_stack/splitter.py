"""ADE Split wrapper for document section classification and splitting.

ADE Split provides:
- Section classification
- Document segmentation
- Category-based splitting
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import Chunk
from .client import ADEClient, ADERegion


class ADESplitWrapper:
    """Wrapper for ADE Split document segmentation API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: ADERegion = ADERegion.US
    ):
        """Initialize ADE Split wrapper.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
            region: API region (US or EU)
        """
        self.api_key = api_key or os.environ.get("LANDINGAI_API_KEY")
        self.region = region
        self.client = ADEClient(api_key=self.api_key, region=region)

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> List[Chunk]:
        """Split document into classified sections.

        Args:
            content: Document content (markdown or text)
            categories: Category labels for classification
            file_path: Optional original file
            **options: Additional splitting options

        Returns:
            List of Chunk objects
        """
        # Call ADE Split API
        result = await self.client.split(
            content=content,
            categories=categories,
            file_path=file_path,
            **options
        )

        # Parse response into Chunk objects
        chunks = []

        sections = result.get("sections", result.get("chunks", []))

        for i, section in enumerate(sections):
            if isinstance(section, str):
                # Simple string chunk
                chunk = Chunk(
                    id=f"chunk_{uuid.uuid4().hex[:8]}",
                    content=section,
                    chunk_type="text",
                    page_numbers=[],
                    category=None,
                    metadata={"index": i}
                )
            else:
                # Structured chunk object
                chunk = Chunk(
                    id=section.get("id", f"chunk_{uuid.uuid4().hex[:8]}"),
                    content=section.get("content", section.get("text", "")),
                    chunk_type=section.get("type", "text"),
                    page_numbers=section.get("pages", []),
                    category=section.get("category", section.get("label")),
                    metadata={
                        "index": i,
                        "confidence": section.get("confidence"),
                        "start": section.get("start"),
                        "end": section.get("end"),
                    },
                    start_index=section.get("start"),
                    end_index=section.get("end"),
                )

            chunks.append(chunk)

        return chunks

    async def split_file(
        self,
        file_path: Union[str, Path],
        categories: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split a document file into sections.

        Args:
            file_path: Path to document file
            categories: Category labels for classification
            **options: Additional options

        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # First parse the file
        from .parser import ADEParseWrapper
        parser = ADEParseWrapper(api_key=self.api_key, region=self.region)

        parse_result = await parser.parse(file_path)
        await parser.close()

        # Then split the parsed content
        return await self.split(
            content=parse_result["markdown"],
            categories=categories,
            file_path=file_path,
            **options
        )

    async def classify_sections(
        self,
        content: str,
        categories: List[str],
        **options
    ) -> List[Dict[str, Any]]:
        """Classify document sections into categories.

        Returns raw classification results without converting to Chunks.

        Args:
            content: Document content
            categories: Category labels
            **options: Additional options

        Returns:
            List of section classification results
        """
        result = await self.client.split(
            content=content,
            categories=categories,
            **options
        )

        return result.get("sections", result.get("classifications", []))

    def get_default_categories(self) -> List[str]:
        """Get default section categories.

        Returns:
            List of common document section categories
        """
        return [
            "header",
            "title",
            "abstract",
            "introduction",
            "body",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "appendix",
            "table",
            "figure",
            "caption",
            "footnote",
            "sidebar",
            "list",
            "form_field",
        ]

    async def close(self):
        """Close the client connection."""
        await self.client.close()
