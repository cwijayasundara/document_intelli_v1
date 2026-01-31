"""LandingAI ADE Stack for document processing.

Uses LandingAI ADE API for:
- ADE Parse: Document parsing with grounding
- ADE Extract: Structured extraction with JSON schemas
- ADE Split: Section classification and splitting
"""

from .processor import LandingAIProcessor
from .client import ADEClient
from .parser import ADEParseWrapper
from .extractor import ADEExtractWrapper
from .splitter import ADESplitWrapper

__all__ = [
    "LandingAIProcessor",
    "ADEClient",
    "ADEParseWrapper",
    "ADEExtractWrapper",
    "ADESplitWrapper",
]
