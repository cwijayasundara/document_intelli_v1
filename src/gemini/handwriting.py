"""Gemini-based handwriting recognition processor.

Uses Google Gemini's multimodal capabilities for:
- Handwritten text recognition
- Mixed printed/handwritten content
- Mathematical notation
"""

import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Union

from google import genai
from google.genai import types


class GeminiHandwritingProcessor:
    """Processor for handwritten document content using Gemini."""

    # Supported image formats
    SUPPORTED_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

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
    }

    # Handwriting-specific prompts
    HANDWRITING_PROMPT = """Analyze this document image and extract all handwritten content.

Instructions:
1. Identify all handwritten text, including:
   - Handwritten words and sentences
   - Handwritten numbers and mathematical expressions
   - Filled-in form fields
   - Signatures
   - Annotations and notes

2. For each handwritten element, provide:
   - The transcribed text
   - Location description (e.g., "top right", "next to question 3")
   - Confidence level (high, medium, low)

3. If there is printed text mixed with handwriting:
   - Clearly distinguish between printed and handwritten content
   - Maintain the relationship between printed questions and handwritten answers

4. For mathematical notation:
   - Use standard mathematical notation (e.g., x^2 for x squared)
   - Preserve fractions, integrals, and other mathematical symbols

Output the content in a structured format with clear sections."""

    MATH_HANDWRITING_PROMPT = """This is a document containing handwritten mathematical content (possibly an exam or worksheet).

Instructions:
1. Extract all handwritten mathematical expressions and solutions
2. For each problem/answer:
   - Identify the problem number if visible
   - Transcribe the handwritten solution step by step
   - Use LaTeX-style notation for mathematical expressions (e.g., \\frac{}{}, \\int, \\sqrt{})
   - Preserve the work/steps shown

3. If there are multiple choice answers:
   - Note which option was selected
   - Include any work shown

4. Maintain the original structure and order of problems

Output in markdown format with clear problem/answer separation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash"
    ):
        """Initialize Gemini handwriting processor.

        Args:
            api_key: Google AI API key. Defaults to GOOGLE_API_KEY env var.
            model: Gemini model to use (default: gemini-2.0-flash)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    async def process_handwriting(
        self,
        file_path: Union[str, Path],
        mode: str = "general",
        custom_prompt: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Process handwritten document content.

        Args:
            file_path: Path to document with handwriting
            mode: Processing mode - "general", "math", or "form"
            custom_prompt: Optional custom prompt to override default
            **options: Additional processing options

        Returns:
            Dict containing:
                - text: Extracted text content
                - structured: Structured extraction if applicable
                - confidence: Overall confidence score
                - metadata: Processing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Select prompt based on mode
        if custom_prompt:
            prompt = custom_prompt
        elif mode == "math":
            prompt = self.MATH_HANDWRITING_PROMPT
        else:
            prompt = self.HANDWRITING_PROMPT

        # Read file and create content
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = self.MIME_TYPES.get(ext, "application/octet-stream")

        # Create parts for the request
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        text_part = types.Part.from_text(text=prompt)

        # Generate content
        response = self.client.models.generate_content(
            model=self.model,
            contents=[file_part, text_part]
        )

        result_text = response.text if hasattr(response, 'text') else ""

        return {
            "text": result_text,
            "structured": self._parse_structured_content(result_text, mode),
            "confidence": 0.85,  # Gemini doesn't provide explicit confidence
            "metadata": {
                "model": self.model,
                "mode": mode,
                "file_name": file_path.name,
            }
        }

    def _parse_structured_content(
        self,
        text: str,
        mode: str
    ) -> Optional[Dict[str, Any]]:
        """Parse structured content from Gemini response.

        Args:
            text: Raw text response
            mode: Processing mode

        Returns:
            Structured content if parseable
        """
        if mode == "math":
            # Try to extract problems and solutions
            problems = []
            current_problem = None

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Look for problem numbers
                if any(line.startswith(p) for p in ["Problem", "Question", "#", "1.", "2.", "3."]):
                    if current_problem:
                        problems.append(current_problem)
                    current_problem = {"problem": line, "solution": []}
                elif current_problem:
                    current_problem["solution"].append(line)

            if current_problem:
                problems.append(current_problem)

            return {"problems": problems} if problems else None

        elif mode == "form":
            # Try to extract field-value pairs
            fields = {}
            for line in text.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if key and value:
                            fields[key] = value

            return {"fields": fields} if fields else None

        return None

    async def detect_handwriting(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Detect if document contains handwriting.

        Args:
            file_path: Path to document

        Returns:
            Dict with detection results:
                - has_handwriting: bool
                - handwriting_percentage: float (0-1)
                - regions: List of regions with handwriting
        """
        file_path = Path(file_path)

        detection_prompt = """Analyze this document image and determine if it contains handwritten content.

Report:
1. Does the document contain any handwriting? (yes/no)
2. Approximately what percentage of the content is handwritten? (0-100)
3. What types of handwriting are present? (text, numbers, signatures, annotations, etc.)
4. Describe the regions where handwriting appears

Respond in this exact format:
HANDWRITING_PRESENT: yes/no
PERCENTAGE: [number]
TYPES: [comma-separated list]
REGIONS: [description]"""

        # Read file
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        ext = file_path.suffix.lower()
        mime_type = self.MIME_TYPES.get(ext, "application/octet-stream")

        # Create parts
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        text_part = types.Part.from_text(text=detection_prompt)

        # Generate
        response = self.client.models.generate_content(
            model=self.model,
            contents=[file_part, text_part]
        )

        response_text = response.text if hasattr(response, 'text') else ""

        # Parse response
        has_handwriting = False
        percentage = 0.0
        types_list = []
        regions = ""

        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("HANDWRITING_PRESENT:"):
                has_handwriting = "yes" in line.lower()
            elif line.startswith("PERCENTAGE:"):
                try:
                    percentage = float(line.split(":")[1].strip().replace("%", "")) / 100
                except (ValueError, IndexError):
                    pass
            elif line.startswith("TYPES:"):
                types_str = line.split(":", 1)[1].strip() if ":" in line else ""
                types_list = [t.strip() for t in types_str.split(",") if t.strip()]
            elif line.startswith("REGIONS:"):
                regions = line.split(":", 1)[1].strip() if ":" in line else ""

        return {
            "has_handwriting": has_handwriting,
            "handwriting_percentage": percentage,
            "types": types_list,
            "regions": regions,
        }

    async def process_with_context(
        self,
        file_path: Union[str, Path],
        context: str,
        **options
    ) -> Dict[str, Any]:
        """Process handwriting with additional context.

        Useful when you know the document type or expected content.

        Args:
            file_path: Path to document
            context: Context about the document (e.g., "This is a calculus exam")
            **options: Additional options

        Returns:
            Processing results
        """
        custom_prompt = f"""Context: {context}

{self.HANDWRITING_PROMPT}

Given the context, pay special attention to domain-specific notation and terminology."""

        return await self.process_handwriting(
            file_path=file_path,
            custom_prompt=custom_prompt,
            **options
        )
