"""Reducto Parse wrapper for document parsing with grounding.

Reducto Parse provides:
- Structured content extraction with block-level typing
- Bounding box grounding for source traceability
- Table, figure, and key-value detection
- Multiple output formats for tables (HTML, markdown, JSON)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import BoundingBox, GroundingData, GroundingReference
from .client import ReductoClient

logger = logging.getLogger(__name__)


class ReductoParseWrapper:
    """Wrapper for Reducto Parse document parsing API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Reducto Parse wrapper.

        Args:
            api_key: Reducto API key. Defaults to REDUCTO_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("REDUCTO_API_KEY")
        self.client = ReductoClient(api_key=self.api_key)

    async def parse(
        self,
        file_path: Union[str, Path],
        include_grounding: bool = True,
        upload_ref: Any = None,
        **options
    ) -> Dict[str, Any]:
        """Parse a document using Reducto Parse.

        Args:
            file_path: Path to the document file
            include_grounding: Include bounding box data for source traceability
            upload_ref: Pre-uploaded file reference (avoids re-uploading)
            **options: Additional Reducto Parse options

        Returns:
            Dict containing:
                - markdown: Assembled markdown from blocks
                - text: Plain text content
                - chunks: Raw chunks from Reducto
                - tables: Extracted tables
                - figures: Extracted figures
                - grounding: GroundingData with bounding boxes
                - metadata: Parsing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Upload if no ref provided
        if upload_ref is None:
            upload_ref = await self.client.upload(file_path)

        # Build parse options
        parse_kwargs = {}
        if "formatting" not in options:
            parse_kwargs["formatting"] = {"table_output_format": "md"}
        if "enhance" in options:
            parse_kwargs["enhance"] = options.pop("enhance")
        if "settings" in options:
            parse_kwargs["settings"] = options.pop("settings")
        if "retrieval" in options:
            parse_kwargs["retrieval"] = options.pop("retrieval")
        parse_kwargs.update(options)

        # Call Reducto Parse API
        result = await self.client.parse(upload_ref, **parse_kwargs)

        # Process response
        return self._process_parse_result(result, file_path, include_grounding)

    def _process_parse_result(
        self,
        result: Any,
        file_path: Path,
        include_grounding: bool
    ) -> Dict[str, Any]:
        """Process Reducto parse result into unified format.

        Args:
            result: Raw Reducto parse result
            file_path: Original file path
            include_grounding: Whether to extract grounding data

        Returns:
            Processed result dictionary
        """
        markdown_parts = []
        text_parts = []
        tables = []
        figures = []
        grounding_refs = []
        page_numbers = set()

        # Extract chunks and blocks
        chunks_data = []
        if hasattr(result, 'result') and hasattr(result.result, 'chunks'):
            for chunk in result.result.chunks:
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                markdown_parts.append(chunk_content)
                text_parts.append(chunk_content)

                chunks_data.append({
                    "content": chunk_content,
                    "embed": chunk.embed if hasattr(chunk, 'embed') else None,
                })

                # Process blocks within each chunk
                if hasattr(chunk, 'blocks'):
                    for block in chunk.blocks:
                        block_type = block.type if hasattr(block, 'type') else "Text"
                        block_content = block.content if hasattr(block, 'content') else ""

                        # Collect tables and figures
                        if block_type == "Table":
                            tables.append({
                                "content": block_content,
                                "page": block.bbox.page if hasattr(block, 'bbox') and hasattr(block.bbox, 'page') else None
                            })
                        elif block_type == "Figure":
                            figures.append({
                                "content": block_content,
                                "image_url": block.image_url if hasattr(block, 'image_url') else None,
                                "page": block.bbox.page if hasattr(block, 'bbox') and hasattr(block.bbox, 'page') else None
                            })

                        # Extract grounding from bounding boxes
                        if include_grounding and hasattr(block, 'bbox') and block.bbox:
                            bbox = block.bbox
                            page = bbox.page if hasattr(bbox, 'page') else 1
                            page_numbers.add(page)

                            confidence_val = 1.0
                            if hasattr(block, 'confidence') and block.confidence:
                                confidence_val = 1.0 if block.confidence == "high" else 0.5

                            grounding_refs.append(GroundingReference(
                                text=block_content[:200] if block_content else "",
                                bbox=BoundingBox(
                                    x=bbox.left if hasattr(bbox, 'left') else 0,
                                    y=bbox.top if hasattr(bbox, 'top') else 0,
                                    width=bbox.width if hasattr(bbox, 'width') else 0,
                                    height=bbox.height if hasattr(bbox, 'height') else 0,
                                    page=page,
                                ),
                                confidence=confidence_val,
                            ))

        markdown = "\n\n".join(markdown_parts) if markdown_parts else ""
        text = "\n\n".join(text_parts) if text_parts else ""

        # Extract usage metadata
        num_pages = 1
        credits_used = None
        if hasattr(result, 'usage'):
            num_pages = result.usage.num_pages if hasattr(result.usage, 'num_pages') else 1
            credits_used = result.usage.credits if hasattr(result.usage, 'credits') else None

        # Build grounding data
        grounding = None
        if include_grounding and grounding_refs:
            grounding = GroundingData(references=grounding_refs)

        return {
            "markdown": markdown,
            "text": text,
            "chunks": chunks_data,
            "tables": tables,
            "figures": figures,
            "pages": [],
            "grounding": grounding,
            "metadata": {
                "page_count": num_pages,
                "credits_used": credits_used,
                "file_type": file_path.suffix,
                "job_id": result.job_id if hasattr(result, 'job_id') else None,
                "studio_link": result.studio_link if hasattr(result, 'studio_link') else None,
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
