"""LlamaClassify wrapper for document classification.

LlamaClassify provides:
- Fast mode: Quick classification based on text
- Multimodal mode: Classification considering visual elements
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_cloud import LlamaCloud

from ..common.interfaces import ClassificationRule
from ..common.models import Classification, DocumentType


class ClassifyMode(str, Enum):
    """LlamaClassify modes."""
    FAST = "fast"
    MULTIMODAL = "multimodal"


class LlamaClassifyWrapper:
    """Wrapper for LlamaClassify document classification API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LlamaClassify wrapper.

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

    async def classify(
        self,
        content: str,
        rules: List[ClassificationRule],
        mode: ClassifyMode = ClassifyMode.FAST,
        return_reasoning: bool = True,
        **options
    ) -> Classification:
        """Classify document content.

        Args:
            content: Document content (markdown or text)
            rules: Classification rules defining categories
            mode: Classification mode (fast or multimodal)
            return_reasoning: Include reasoning in response
            **options: Additional classification options

        Returns:
            Classification result with type, confidence, and reasoning
        """
        # Build classification request
        labels = [rule.label for rule in rules]
        label_descriptions = {rule.label: rule.description for rule in rules}

        try:
            # Call classification API
            result = self.client.classifier.classify(
                text=content,
                labels=labels,
                label_descriptions=label_descriptions,
                **options
            )

            # Parse response
            primary_label = "other"
            confidence = 0.0
            reasoning = None
            label_scores = {}

            if hasattr(result, 'label'):
                primary_label = result.label
            if hasattr(result, 'confidence'):
                confidence = result.confidence
            if hasattr(result, 'reasoning') and return_reasoning:
                reasoning = result.reasoning
            if hasattr(result, 'scores'):
                label_scores = result.scores or {}

            # Map to DocumentType
            doc_type = self._map_to_document_type(primary_label)

            return Classification(
                document_type=doc_type,
                confidence=confidence,
                reasoning=reasoning,
                secondary_types=[],
                labels=label_scores
            )

        except Exception as e:
            # Fallback to simple keyword-based classification
            return self._fallback_classify(content, rules)

    def _fallback_classify(
        self,
        content: str,
        rules: List[ClassificationRule]
    ) -> Classification:
        """Fallback classification using keyword matching."""
        content_lower = content.lower()
        scores = {}

        for rule in rules:
            score = 0
            for keyword in rule.keywords:
                if keyword.lower() in content_lower:
                    score += 1
            scores[rule.label] = score

        if scores:
            best_label = max(scores, key=scores.get)
            max_score = scores[best_label]
            total_keywords = sum(len(r.keywords) for r in rules)
            confidence = max_score / max(total_keywords, 1) if max_score > 0 else 0.0
        else:
            best_label = "other"
            confidence = 0.0

        doc_type = self._map_to_document_type(best_label)

        return Classification(
            document_type=doc_type,
            confidence=min(confidence, 1.0),
            reasoning="Keyword-based fallback classification",
            labels={k: v / max(sum(scores.values()), 1) for k, v in scores.items()}
        )

    async def classify_file(
        self,
        file_path: Union[str, Path],
        rules: List[ClassificationRule],
        mode: ClassifyMode = ClassifyMode.MULTIMODAL,
        **options
    ) -> Classification:
        """Classify a document file directly.

        Useful for visual documents where content extraction
        might lose important classification signals.

        Args:
            file_path: Path to document file
            rules: Classification rules
            mode: Classification mode (multimodal recommended)
            **options: Additional options

        Returns:
            Classification result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Build labels
        labels = [rule.label for rule in rules]
        label_descriptions = {rule.label: rule.description for rule in rules}

        try:
            # Call classification API with file
            result = self.client.classifier.classify(
                upload_file=(file_path.name, file_content),
                labels=labels,
                label_descriptions=label_descriptions,
                **options
            )

            primary_label = result.label if hasattr(result, 'label') else "other"
            confidence = result.confidence if hasattr(result, 'confidence') else 0.0
            reasoning = result.reasoning if hasattr(result, 'reasoning') else None

            doc_type = self._map_to_document_type(primary_label)

            return Classification(
                document_type=doc_type,
                confidence=confidence,
                reasoning=reasoning,
                secondary_types=[],
                labels={}
            )

        except Exception as e:
            # Return unknown classification on error
            return Classification(
                document_type=DocumentType.OTHER,
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )

    def _map_to_document_type(self, label: str) -> DocumentType:
        """Map classification label to DocumentType enum.

        Args:
            label: Classification label string

        Returns:
            Corresponding DocumentType
        """
        label_lower = label.lower().strip()

        mapping = {
            "form": DocumentType.FORM,
            "invoice": DocumentType.INVOICE,
            "receipt": DocumentType.RECEIPT,
            "contract": DocumentType.CONTRACT,
            "report": DocumentType.REPORT,
            "presentation": DocumentType.PRESENTATION,
            "diagram": DocumentType.DIAGRAM,
            "flowchart": DocumentType.FLOWCHART,
            "spreadsheet": DocumentType.SPREADSHEET,
            "certificate": DocumentType.CERTIFICATE,
            "medical": DocumentType.MEDICAL,
            "educational": DocumentType.EDUCATIONAL,
            "infographic": DocumentType.INFOGRAPHIC,
            "instructions": DocumentType.INSTRUCTIONS,
            "handwritten": DocumentType.HANDWRITTEN,
        }

        return mapping.get(label_lower, DocumentType.OTHER)

    def get_default_rules(self) -> List[ClassificationRule]:
        """Get default classification rules covering common document types.

        Returns:
            List of default ClassificationRule objects
        """
        return [
            ClassificationRule(
                label="form",
                description="Structured forms with input fields, checkboxes, or areas to fill",
                keywords=["form", "please fill", "checkbox", "input"],
            ),
            ClassificationRule(
                label="invoice",
                description="Billing documents with itemized charges and payment information",
                keywords=["invoice", "bill to", "total", "payment due"],
            ),
            ClassificationRule(
                label="receipt",
                description="Transaction receipts showing purchases or payments",
                keywords=["receipt", "paid", "transaction", "purchase"],
            ),
            ClassificationRule(
                label="certificate",
                description="Official certificates, credentials, or certifications",
                keywords=["certificate", "certify", "hereby", "awarded"],
            ),
            ClassificationRule(
                label="medical",
                description="Medical or healthcare documents including patient forms",
                keywords=["patient", "medical", "health", "doctor", "diagnosis"],
            ),
            ClassificationRule(
                label="presentation",
                description="Presentation slides or investor materials",
                keywords=["slide", "presentation", "overview", "Q1", "revenue"],
            ),
            ClassificationRule(
                label="diagram",
                description="Technical diagrams or architectural drawings",
                keywords=["diagram", "architecture", "component", "flow"],
            ),
            ClassificationRule(
                label="flowchart",
                description="Process flowcharts with decision branches",
                keywords=["flowchart", "process", "decision", "start", "end"],
            ),
            ClassificationRule(
                label="spreadsheet",
                description="Data tables or spreadsheet views",
                keywords=["column", "row", "total", "sum", "data"],
            ),
            ClassificationRule(
                label="instructions",
                description="Assembly instructions or how-to guides",
                keywords=["step", "instruction", "assembly", "how to", "guide"],
            ),
            ClassificationRule(
                label="infographic",
                description="Visual infographics combining text and graphics",
                keywords=["infographic", "timeline", "visual", "stats"],
            ),
            ClassificationRule(
                label="handwritten",
                description="Documents containing handwritten content",
                keywords=["handwritten", "handwriting", "written"],
            ),
            ClassificationRule(
                label="report",
                description="Reports, analyses, or formal documents",
                keywords=["report", "analysis", "summary", "conclusion"],
            ),
        ]
