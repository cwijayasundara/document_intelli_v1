"""Reducto Stack for document processing.

Uses Reducto API for:
- Parse: Document parsing with bounding box grounding
- Extract: Structured extraction with JSON schemas
- Split: Section-based document splitting
"""

from .processor import ReductoProcessor
from .client import ReductoClient
from .parser import ReductoParseWrapper
from .extractor import ReductoExtractWrapper
from .splitter import ReductoSplitWrapper

__all__ = [
    "ReductoProcessor",
    "ReductoClient",
    "ReductoParseWrapper",
    "ReductoExtractWrapper",
    "ReductoSplitWrapper",
]
