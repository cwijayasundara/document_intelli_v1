"""LlamaParse wrapper for document parsing.

LlamaParse offers 4 parsing tiers:
- Cost Effective: Basic parsing, 1 credit/page
- Agentic: Enhanced parsing, 1 credit/page
- Agentic Plus: Complex layouts, 2 credits/page
- Fast: Quick processing, 1 credit/page
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_cloud import LlamaCloud


class ParseTier(str, Enum):
    """LlamaParse parsing tiers."""
    COST_EFFECTIVE = "cost_effective"
    AGENTIC = "agentic"
    AGENTIC_PLUS = "agentic_plus"
    FAST = "fast"


class LlamaParseWrapper:
    """Wrapper for LlamaParse document parsing API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LlamaParse wrapper.

        Args:
            api_key: LlamaCloud API key. Defaults to LLAMA_CLOUD_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set LLAMA_CLOUD_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = LlamaCloud(api_key=self.api_key)

    async def parse(
        self,
        file_path: Union[str, Path],
        tier: ParseTier = ParseTier.AGENTIC,
        multimodal: bool = False,
        language: str = "en",
        output_format: str = "markdown",
        extract_images: bool = True,
        extract_tables: bool = True,
        **options
    ) -> Dict[str, Any]:
        """Parse a document using LlamaParse.

        Args:
            file_path: Path to the document file
            tier: Parsing tier (cost_effective, agentic, agentic_plus, fast)
            multimodal: Enable multimodal mode for visual documents (2 credits/page)
            language: Document language code
            output_format: Output format (markdown, text, html)
            extract_images: Extract and describe images
            extract_tables: Extract tables as structured data
            **options: Additional LlamaParse options

        Returns:
            Dict containing:
                - markdown: Parsed content as markdown
                - text: Plain text content
                - images: List of extracted images with descriptions
                - tables: List of extracted tables
                - metadata: Parsing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Build parsing options
        parse_options = {
            "tier": tier.value,
            "version": "latest",
            "upload_file": (file_path.name, file_content),
            "expand": ["markdown", "text", "items"],
        }

        # Add processing options
        if multimodal:
            parse_options["processing_options"] = {
                "multimodal_model": "anthropic-sonnet-4"
            }

        # Call parsing API
        result = self.client.parsing.parse(**parse_options)

        # Extract content from result
        markdown_content = ""
        text_content = ""
        images = []
        tables = []
        page_count = 1

        # Extract markdown from Markdown object with pages
        if result.markdown and hasattr(result.markdown, 'pages') and result.markdown.pages:
            markdown_parts = []
            for page in result.markdown.pages:
                if hasattr(page, 'markdown') and page.markdown:
                    markdown_parts.append(page.markdown)
            markdown_content = "\n\n".join(markdown_parts)
            page_count = len(result.markdown.pages)

        # Extract text from Text object with pages
        if result.text and hasattr(result.text, 'pages') and result.text.pages:
            text_parts = []
            for page in result.text.pages:
                if hasattr(page, 'text') and page.text:
                    text_parts.append(page.text)
            text_content = "\n\n".join(text_parts)

        # Fallback: if markdown is empty but items exist, build from items
        if not markdown_content and result.items:
            if hasattr(result.items, 'pages') and result.items.pages:
                md_parts = []
                for page in result.items.pages:
                    if hasattr(page, 'items') and page.items:
                        for item in page.items:
                            if hasattr(item, 'md') and item.md:
                                md_parts.append(item.md)
                            elif hasattr(item, 'value') and item.value:
                                md_parts.append(item.value)
                            elif hasattr(item, 'csv') and item.csv:
                                md_parts.append(self._csv_to_markdown_table(item.csv))
                markdown_content = "\n\n".join(md_parts)

        # Use markdown as text if text is empty
        if not text_content:
            text_content = markdown_content

        # Build result
        parsed_result = {
            "markdown": markdown_content,
            "text": text_content,
            "images": images,
            "tables": tables,
            "metadata": {
                "job_id": result.job.id if result.job else None,
                "tier": tier.value,
                "multimodal": multimodal,
                "pages": page_count,
                "credits_used": self._calculate_credits(
                    page_count,
                    tier,
                    multimodal
                ),
            }
        }

        return parsed_result

    def _csv_to_markdown_table(self, csv_content: str) -> str:
        """Convert CSV content to markdown table."""
        lines = csv_content.strip().split('\n')
        if not lines:
            return ""

        import csv
        import io

        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        if not rows:
            return ""

        # Build markdown table
        md_lines = []

        # Header row
        header = rows[0]
        md_lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Data rows
        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")
            md_lines.append("| " + " | ".join(str(cell) for cell in row[:len(header)]) + " |")

        return "\n".join(md_lines)

    def _calculate_credits(
        self,
        pages: int,
        tier: ParseTier,
        multimodal: bool
    ) -> float:
        """Calculate credits used for parsing."""
        if multimodal:
            return pages * 2.0
        elif tier == ParseTier.AGENTIC_PLUS:
            return pages * 2.0
        else:
            return pages * 1.0

    async def parse_batch(
        self,
        file_paths: List[Union[str, Path]],
        **options
    ) -> List[Dict[str, Any]]:
        """Parse multiple documents."""
        results = []
        for path in file_paths:
            result = await self.parse(path, **options)
            results.append(result)
        return results
