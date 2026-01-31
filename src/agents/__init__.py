"""LangChain-based document processing agent.

Provides an AI agent with skills for:
- Document parsing
- Document extraction
- Document splitting/chunking
- Document classification
"""

from .agent import DocumentAgent, create_document_agent
from .skills import (
    parse_document,
    extract_from_document,
    split_document,
    classify_document,
)

__all__ = [
    "DocumentAgent",
    "create_document_agent",
    "parse_document",
    "extract_from_document",
    "split_document",
    "classify_document",
]
