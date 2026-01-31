"""Visualization helpers for loan document processing.

Provides bounding box visualization and document annotation.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# Color palette for different field types
FIELD_COLORS = {
    "name": (76, 175, 80, 180),      # Green
    "date": (33, 150, 243, 180),     # Blue
    "amount": (255, 152, 0, 180),    # Orange
    "number": (156, 39, 176, 180),   # Purple
    "address": (0, 188, 212, 180),   # Cyan
    "default": (233, 30, 99, 180),   # Pink
}


def get_field_color(field_name: str) -> Tuple[int, int, int, int]:
    """Get a color for a field based on its name/type."""
    field_lower = field_name.lower()

    if any(x in field_lower for x in ["name", "employee", "owner", "holder"]):
        return FIELD_COLORS["name"]
    elif any(x in field_lower for x in ["date", "period", "year"]):
        return FIELD_COLORS["date"]
    elif any(x in field_lower for x in ["amount", "balance", "value", "pay", "wage", "total"]):
        return FIELD_COLORS["amount"]
    elif any(x in field_lower for x in ["number", "ssn", "ein", "account", "identifier"]):
        return FIELD_COLORS["number"]
    elif any(x in field_lower for x in ["address", "location"]):
        return FIELD_COLORS["address"]

    return FIELD_COLORS["default"]


class DocumentVisualizer:
    """Visualizes extracted fields on documents with bounding boxes."""

    def __init__(self):
        """Initialize the visualizer."""
        self._font = None

    def _get_font(self, size: int = 12) -> ImageFont.FreeTypeFont:
        """Get a font for annotations."""
        if self._font is None:
            try:
                # Try to use a system font
                self._font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
            except (OSError, IOError):
                try:
                    self._font = ImageFont.truetype("arial.ttf", size)
                except (OSError, IOError):
                    # Fall back to default font
                    self._font = ImageFont.load_default()
        return self._font

    def render_pdf_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        dpi: int = 150
    ) -> Optional[Image.Image]:
        """Render a PDF page as an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to render (0-indexed)
            dpi: Resolution for rendering

        Returns:
            PIL Image of the rendered page, or None if failed
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            return None

        try:
            doc = fitz.open(pdf_path)
            if page_number >= len(doc):
                logger.warning(f"Page {page_number} out of range (document has {len(doc)} pages)")
                return None

            page = doc[page_number]
            # Render at higher resolution for better quality
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            return img

        except Exception as e:
            logger.error(f"Failed to render PDF page: {e}")
            return None

    def load_image(self, file_path: Path) -> Optional[Image.Image]:
        """Load an image from file.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image, or None if failed
        """
        try:
            return Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def draw_bounding_boxes(
        self,
        image: Image.Image,
        grounding_data: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None
    ) -> Image.Image:
        """Draw bounding boxes on an image.

        Args:
            image: PIL Image to annotate
            grounding_data: List of grounding items with bounding box info
            labels: Optional dict mapping field names to display labels
            page_width: Original page width (for coordinate scaling)
            page_height: Original page height (for coordinate scaling)

        Returns:
            Annotated PIL Image
        """
        # Create a copy to draw on
        img = image.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        font = self._get_font(14)

        img_width, img_height = img.size

        for item in grounding_data:
            # Extract bounding box coordinates
            bbox = self._extract_bbox(item)
            if bbox is None:
                continue

            x, y, width, height = bbox

            # Scale coordinates if page dimensions provided
            if page_width and page_height:
                scale_x = img_width / page_width
                scale_y = img_height / page_height
                x *= scale_x
                y *= scale_y
                width *= scale_x
                height *= scale_y

            # Get field name and color
            field_name = item.get("field", item.get("name", "unknown"))
            color = get_field_color(field_name)

            # Draw filled rectangle with transparency
            draw.rectangle(
                [x, y, x + width, y + height],
                fill=(*color[:3], 50),
                outline=color[:3],
                width=2
            )

            # Draw label
            label = labels.get(field_name, field_name) if labels else field_name
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position label above the box
            label_x = x
            label_y = y - text_height - 4
            if label_y < 0:
                label_y = y + height + 4

            # Draw label background
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
                fill=color[:3]
            )
            draw.text((label_x + 2, label_y + 2), label, fill="white", font=font)

        return img

    def _extract_bbox(self, item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounding box coordinates from a grounding item.

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        # Try different bbox formats
        if "bbox" in item:
            bbox = item["bbox"]
            if isinstance(bbox, dict):
                return (
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("width", 0),
                    bbox.get("height", 0)
                )
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return tuple(bbox[:4])

        # Try x, y, width, height directly
        if all(k in item for k in ["x", "y", "width", "height"]):
            return (item["x"], item["y"], item["width"], item["height"])

        # Try coordinates format [x1, y1, x2, y2]
        if "coordinates" in item:
            coords = item["coordinates"]
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                return (x1, y1, x2 - x1, y2 - y1)

        return None

    def annotate_document(
        self,
        file_path: Path,
        extraction_result: Dict[str, Any],
        page_number: int = 0
    ) -> Optional[Image.Image]:
        """Create an annotated version of a document with extracted fields highlighted.

        Args:
            file_path: Path to the document file
            extraction_result: Extraction result containing fields and grounding
            page_number: Page number to annotate (for PDFs)

        Returns:
            Annotated PIL Image, or None if failed
        """
        # Load the document
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image = self.render_pdf_page(file_path, page_number)
        else:
            image = self.load_image(file_path)

        if image is None:
            logger.warning(f"Could not load document: {file_path}")
            return None

        # Get grounding data
        grounding = extraction_result.get("grounding", [])
        if not grounding:
            logger.info("No grounding data available for visualization")
            return image

        # Get field values for labels
        fields = extraction_result.get("fields", {})
        labels = {k: f"{k}: {v}" for k, v in fields.items() if v is not None}

        # Draw bounding boxes
        annotated = self.draw_bounding_boxes(image, grounding, labels)

        return annotated

    def create_summary_image(
        self,
        extraction_results: List[Dict[str, Any]],
        max_width: int = 800
    ) -> Optional[Image.Image]:
        """Create a summary image showing all documents with annotations.

        Args:
            extraction_results: List of extraction results
            max_width: Maximum width for each thumbnail

        Returns:
            Combined PIL Image, or None if failed
        """
        thumbnails = []

        for result in extraction_results:
            file_path = result.get("file_path")
            if not file_path:
                continue

            file_path = Path(file_path)
            if not file_path.exists():
                continue

            annotated = self.annotate_document(file_path, result)
            if annotated:
                # Resize to thumbnail
                ratio = max_width / annotated.width
                new_height = int(annotated.height * ratio)
                thumbnail = annotated.resize((max_width, new_height), Image.Resampling.LANCZOS)
                thumbnails.append(thumbnail)

        if not thumbnails:
            return None

        # Combine thumbnails vertically
        total_height = sum(t.height for t in thumbnails) + (len(thumbnails) - 1) * 10
        combined = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        for thumbnail in thumbnails:
            combined.paste(thumbnail, (0, y_offset))
            y_offset += thumbnail.height + 10

        return combined

    def save_annotated(
        self,
        image: Image.Image,
        output_path: Path,
        format: str = "PNG"
    ) -> bool:
        """Save an annotated image to file.

        Args:
            image: PIL Image to save
            output_path: Path to save to
            format: Image format (PNG, JPEG, etc.)

        Returns:
            True if saved successfully
        """
        try:
            image.save(output_path, format=format)
            logger.info(f"Saved annotated image to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False

    def get_image_bytes(
        self,
        image: Image.Image,
        format: str = "PNG"
    ) -> bytes:
        """Get image as bytes.

        Args:
            image: PIL Image
            format: Image format

        Returns:
            Image bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
