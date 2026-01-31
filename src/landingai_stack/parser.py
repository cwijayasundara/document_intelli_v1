"""ADE Parse wrapper for document parsing with grounding.

ADE Parse provides:
- Structured markdown output
- Bounding box grounding for source traceability
- Page-level segmentation
- Table and figure extraction
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import BoundingBox, GroundingData, GroundingReference
from .client import ADEClient, ADERegion


class ADEParseWrapper:
    """Wrapper for ADE Parse document parsing API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: ADERegion = ADERegion.US
    ):
        """Initialize ADE Parse wrapper.

        Args:
            api_key: LandingAI API key. Defaults to LANDINGAI_API_KEY env var.
            region: API region (US or EU)
        """
        self.api_key = api_key or os.environ.get("LANDINGAI_API_KEY")
        self.region = region
        self.client = ADEClient(api_key=self.api_key, region=region)

    async def parse(
        self,
        file_path: Union[str, Path],
        include_grounding: bool = True,
        page_level: bool = False,
        **options
    ) -> Dict[str, Any]:
        """Parse a document using ADE Parse.

        Args:
            file_path: Path to the document file
            include_grounding: Include bounding box data for source traceability
            page_level: Return results segmented by page
            **options: Additional ADE Parse options

        Returns:
            Dict containing:
                - markdown: Parsed content as markdown
                - grounding: GroundingData with bounding boxes (if include_grounding)
                - pages: Per-page content (if page_level)
                - chunks: Pre-split chunks from parsing
                - tables: Extracted tables
                - figures: Extracted figures
                - metadata: Parsing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Call ADE Parse API
        result = await self.client.parse(
            file_path=file_path,
            include_grounding=include_grounding,
            page_level=page_level,
            **options
        )

        # Process response
        parsed_result = {
            "markdown": result.get("markdown", result.get("content", "")),
            "text": result.get("text", ""),
            "chunks": [],
            "tables": [],
            "figures": [],
            "pages": [],
            "grounding": None,
            "metadata": {
                "page_count": result.get("page_count", 1),
                "file_type": file_path.suffix,
            }
        }

        # Extract chunks if available
        if "chunks" in result:
            parsed_result["chunks"] = result["chunks"]

        # Extract tables
        if "tables" in result:
            parsed_result["tables"] = result["tables"]

        # Extract figures
        if "figures" in result:
            parsed_result["figures"] = result["figures"]

        # Process page-level results
        if page_level and "pages" in result:
            parsed_result["pages"] = result["pages"]

        # Process grounding data
        if include_grounding and "grounding" in result:
            parsed_result["grounding"] = self._parse_grounding(result["grounding"])

        return parsed_result

    def _parse_grounding(self, grounding_data: Dict[str, Any]) -> GroundingData:
        """Parse grounding data from API response.

        Args:
            grounding_data: Raw grounding data from API

        Returns:
            GroundingData object
        """
        references = []

        items = grounding_data.get("items", grounding_data.get("references", []))

        for item in items:
            bbox_data = item.get("bbox", item.get("bounding_box", {}))

            if bbox_data:
                bbox = BoundingBox(
                    x=bbox_data.get("x", bbox_data.get("left", 0)),
                    y=bbox_data.get("y", bbox_data.get("top", 0)),
                    width=bbox_data.get("width", bbox_data.get("w", 0)),
                    height=bbox_data.get("height", bbox_data.get("h", 0)),
                    page=item.get("page", 1)
                )

                references.append(GroundingReference(
                    text=item.get("text", item.get("content", "")),
                    bbox=bbox,
                    confidence=item.get("confidence", 1.0)
                ))

        # Parse page dimensions if available
        page_dimensions = None
        if "page_dimensions" in grounding_data:
            page_dimensions = grounding_data["page_dimensions"]

        return GroundingData(
            references=references,
            page_dimensions=page_dimensions
        )

    async def parse_url(
        self,
        url: str,
        include_grounding: bool = True,
        **options
    ) -> Dict[str, Any]:
        """Parse a document from URL.

        Args:
            url: URL to document
            include_grounding: Include bounding box data
            **options: Additional options

        Returns:
            Parsed result dictionary
        """
        result = await self.client.parse(
            url=url,
            include_grounding=include_grounding,
            **options
        )

        return {
            "markdown": result.get("markdown", result.get("content", "")),
            "text": result.get("text", ""),
            "grounding": self._parse_grounding(result.get("grounding", {})) if include_grounding else None,
            "metadata": {
                "source_url": url,
                "page_count": result.get("page_count", 1),
            }
        }

    async def parse_batch(
        self,
        file_paths: List[Union[str, Path]],
        **options
    ) -> List[Dict[str, Any]]:
        """Parse multiple documents.

        Args:
            file_paths: List of file paths
            **options: Options passed to parse()

        Returns:
            List of parsed results
        """
        results = []
        for path in file_paths:
            try:
                result = await self.parse(path, **options)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "file_path": str(path),
                    "markdown": "",
                })
        return results

    async def close(self):
        """Close the client connection."""
        await self.client.close()
