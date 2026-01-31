"""LlamaSplit wrapper for semantic document chunking.

LlamaSplit provides:
- AI-driven semantic segmentation
- Category-based splitting
- Chunk metadata and relationships
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import Chunk


class LlamaSplitWrapper:
    """Wrapper for semantic chunking.

    Note: LlamaCloud doesn't have a dedicated split API, so this uses
    rule-based splitting with optional semantic analysis.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LlamaSplit wrapper.

        Args:
            api_key: LlamaCloud API key. Defaults to LLAMA_CLOUD_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")

    async def split(
        self,
        content: str,
        categories: Optional[List[str]] = None,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        overlap: int = 50,
        **options
    ) -> List[Chunk]:
        """Split document into semantic chunks.

        Args:
            content: Document content (markdown or text)
            categories: Optional category labels for classification
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between consecutive chunks
            **options: Additional splitting options

        Returns:
            List of Chunk objects
        """
        # Use semantic splitting based on markdown structure
        chunks = []

        # First try to split by headers
        header_chunks = self._split_by_headers(content)

        if header_chunks:
            # Further split large chunks
            for i, chunk_text in enumerate(header_chunks):
                if len(chunk_text) > max_chunk_size:
                    # Split large chunks by paragraphs
                    sub_chunks = self._split_by_paragraphs(
                        chunk_text, max_chunk_size, overlap
                    )
                    for j, sub_text in enumerate(sub_chunks):
                        chunks.append(Chunk(
                            id=f"chunk_{uuid.uuid4().hex[:8]}",
                            content=sub_text,
                            chunk_type="text",
                            page_numbers=[],
                            category=self._classify_chunk(sub_text, categories),
                            metadata={"index": len(chunks)},
                        ))
                elif len(chunk_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        id=f"chunk_{uuid.uuid4().hex[:8]}",
                        content=chunk_text,
                        chunk_type="section",
                        page_numbers=[],
                        category=self._classify_chunk(chunk_text, categories),
                        metadata={"index": len(chunks)},
                    ))
        else:
            # Fall back to paragraph-based splitting
            para_chunks = self._split_by_paragraphs(content, max_chunk_size, overlap)
            for i, chunk_text in enumerate(para_chunks):
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        id=f"chunk_{uuid.uuid4().hex[:8]}",
                        content=chunk_text,
                        chunk_type="text",
                        page_numbers=[],
                        category=self._classify_chunk(chunk_text, categories),
                        metadata={"index": i},
                    ))

        # If still no chunks, create one from the whole content
        if not chunks and content.strip():
            chunks.append(Chunk(
                id=f"chunk_{uuid.uuid4().hex[:8]}",
                content=content.strip(),
                chunk_type="text",
                page_numbers=[],
                category=self._classify_chunk(content, categories),
                metadata={"index": 0},
            ))

        return chunks

    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by markdown headers."""
        chunks = []
        current_chunk = []
        lines = content.split('\n')

        header_markers = ["# ", "## ", "### ", "#### "]

        for line in lines:
            is_header = any(line.startswith(marker) for marker in header_markers)

            if is_header and current_chunk:
                # Save previous chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _split_by_paragraphs(
        self,
        content: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """Split content by paragraphs with size limit."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if current_size + para_size > max_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Include last paragraph for overlap
                    current_chunk = [current_chunk[-1], para]
                    current_size = len(current_chunk[-2]) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _classify_chunk(
        self,
        content: str,
        categories: Optional[List[str]]
    ) -> Optional[str]:
        """Classify a chunk into a category using keyword matching."""
        if not categories:
            return None

        content_lower = content.lower()

        # Simple keyword-based classification
        category_keywords = {
            "introduction": ["introduction", "overview", "background"],
            "methodology": ["method", "approach", "procedure"],
            "results": ["results", "findings", "outcome"],
            "discussion": ["discussion", "analysis", "interpretation"],
            "conclusion": ["conclusion", "summary", "final"],
            "references": ["references", "bibliography", "citations"],
            "table": ["table", "|", "---"],
            "figure": ["figure", "image", "diagram"],
            "list": ["- ", "* ", "1.", "2."],
        }

        for category in categories:
            cat_lower = category.lower()
            if cat_lower in category_keywords:
                keywords = category_keywords[cat_lower]
                if any(kw in content_lower for kw in keywords):
                    return category

        return None

    async def split_file(
        self,
        file_path: Union[str, Path],
        categories: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split a document file into semantic chunks.

        Args:
            file_path: Path to document file
            categories: Optional category labels
            **options: Additional options

        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file as text if possible
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Binary file, return empty
            return []

        return await self.split(content, categories=categories, **options)

    async def split_by_sections(
        self,
        content: str,
        section_markers: Optional[List[str]] = None,
        **options
    ) -> List[Chunk]:
        """Split document by section headers or markers.

        Args:
            content: Document content
            section_markers: List of strings that indicate section breaks
            **options: Additional options

        Returns:
            List of Chunk objects, one per section
        """
        if section_markers is None:
            section_markers = ["# ", "## ", "### "]

        chunks = []
        current_chunk = []
        current_header = None
        lines = content.split('\n')

        for line in lines:
            is_header = any(line.startswith(marker) for marker in section_markers)

            if is_header and current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(Chunk(
                        id=f"section_{len(chunks)}",
                        content=chunk_text,
                        chunk_type="section",
                        category=current_header,
                        metadata={"header": current_header}
                    ))
                current_chunk = [line]
                current_header = line.lstrip('#').strip()
            else:
                current_chunk.append(line)
                if is_header:
                    current_header = line.lstrip('#').strip()

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(Chunk(
                    id=f"section_{len(chunks)}",
                    content=chunk_text,
                    chunk_type="section",
                    category=current_header,
                    metadata={"header": current_header}
                ))

        return chunks

    def get_default_categories(self) -> List[str]:
        """Get default chunk categories.

        Returns:
            List of common document section categories
        """
        return [
            "title",
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "appendix",
            "table",
            "figure",
            "list",
            "header",
            "footer",
            "sidebar",
        ]
