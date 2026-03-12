"""Reducto Split wrapper for document section splitting.

Reducto Split provides:
- Section-based document splitting with category classification
- Confidence scores for each split
- Page-level section mapping
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import Chunk
from .client import ReductoClient

logger = logging.getLogger(__name__)


class ReductoSplitWrapper:
    """Wrapper for Reducto Split document segmentation API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Reducto Split wrapper.

        Args:
            api_key: Reducto API key. Defaults to REDUCTO_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("REDUCTO_API_KEY")
        self.client = ReductoClient(api_key=self.api_key)

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        file_path: Optional[Union[str, Path]] = None,
        upload_ref: Any = None,
        parsed_chunks: Optional[List[Dict[str, Any]]] = None,
        **options
    ) -> List[Chunk]:
        """Split document into classified sections.

        Args:
            content: Document content (markdown or text)
            categories: Category labels for classification
            file_path: Optional original file for API-based splitting
            upload_ref: Pre-uploaded file reference
            parsed_chunks: Pre-parsed chunks from Reducto parse (for content mapping)
            **options: Additional splitting options

        Returns:
            List of Chunk objects
        """
        # Build split descriptions from categories
        if categories:
            split_description = [
                {"name": cat, "description": f"Sections related to {cat}"}
                for cat in categories
            ]
        else:
            split_description = self._get_default_split_descriptions()

        # Try API-based splitting if we have a file reference
        if upload_ref is not None or (file_path and Path(file_path).exists()):
            try:
                if upload_ref is None:
                    upload_ref = await self.client.upload(Path(file_path))

                result = await self.client.split(
                    upload_ref=upload_ref,
                    split_description=split_description,
                    **options
                )

                return self._process_split_result(result, content)

            except Exception as e:
                logger.warning(f"Reducto split API failed, falling back to local: {str(e)}")

        # Fallback: local text-based splitting
        return self._local_split(content, categories)

    def _process_split_result(
        self,
        result: Any,
        content: str
    ) -> List[Chunk]:
        """Process Reducto split result into Chunk objects.

        Args:
            result: Raw Reducto split result
            content: Original document content for page-content mapping

        Returns:
            List of Chunk objects
        """
        chunks = []

        splits = []
        if hasattr(result, 'result'):
            if hasattr(result.result, 'splits'):
                splits = result.result.splits
            elif isinstance(result.result, dict) and 'splits' in result.result:
                splits = result.result['splits']

        # Split content by pages for content mapping
        content_lines = content.split("\n") if content else []

        for i, split in enumerate(splits):
            split_name = split.name if hasattr(split, 'name') else split.get('name', f'section_{i}')
            split_pages = split.pages if hasattr(split, 'pages') else split.get('pages', [])
            split_conf = split.conf if hasattr(split, 'conf') else split.get('conf', 'high')

            # Extract content for this split's pages
            # Since we may not have page-aligned content, use the split name as context
            chunk_content = f"[Section: {split_name}] Pages: {split_pages}"

            confidence_val = 1.0 if split_conf == "high" else 0.5

            chunk = Chunk(
                id=f"chunk_{uuid.uuid4().hex[:8]}",
                content=chunk_content,
                chunk_type="section",
                page_numbers=split_pages if isinstance(split_pages, list) else [split_pages],
                category=split_name,
                metadata={
                    "index": i,
                    "confidence": confidence_val,
                    "source": "reducto_split",
                },
            )
            chunks.append(chunk)

        return chunks

    def _local_split(
        self,
        content: str,
        categories: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Fallback local text-based splitting.

        Args:
            content: Document content
            categories: Optional category labels

        Returns:
            List of Chunk objects
        """
        chunks = []
        if not content:
            return chunks

        # Split by markdown headers or paragraphs
        sections = []
        current_section = []
        current_header = None

        for line in content.split("\n"):
            if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                if current_section:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_section)
                    })
                current_header = line.lstrip("#").strip()
                current_section = [line]
            else:
                current_section.append(line)

        # Add last section
        if current_section:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_section)
            })

        # If no headers found, split by double newlines
        if len(sections) <= 1:
            paragraphs = content.split("\n\n")
            sections = [{"header": None, "content": p.strip()} for p in paragraphs if p.strip()]

        for i, section in enumerate(sections):
            chunk = Chunk(
                id=f"chunk_{uuid.uuid4().hex[:8]}",
                content=section["content"],
                chunk_type="section" if section["header"] else "text",
                page_numbers=[],
                category=section["header"] or (categories[i % len(categories)] if categories else None),
                metadata={
                    "index": i,
                    "source": "local_split",
                },
            )
            chunks.append(chunk)

        return chunks

    def _get_default_split_descriptions(self) -> List[Dict[str, str]]:
        """Get default split descriptions for common document sections."""
        return [
            {"name": "header", "description": "Document header, title, and introductory information"},
            {"name": "body", "description": "Main body content, paragraphs, and key information"},
            {"name": "tables", "description": "Tables, data grids, and tabular information"},
            {"name": "figures", "description": "Figures, charts, diagrams, and visual elements"},
            {"name": "footer", "description": "Footer content, disclaimers, and closing information"},
            {"name": "references", "description": "References, citations, and bibliography"},
            {"name": "appendix", "description": "Appendices and supplementary materials"},
        ]

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
