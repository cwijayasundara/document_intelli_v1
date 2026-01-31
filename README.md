# Document Extraction Pipeline

A multi-stack document parsing and extraction platform that provides comparative evaluation between **LlamaIndex** and **LandingAI** commercial APIs, with **Gemini** integration for handwriting recognition.

## Overview

This pipeline processes documents through multiple stages:
1. **Parsing** - Convert documents to structured markdown
2. **Classification** - Identify document type (invoice, form, certificate, etc.)
3. **Extraction** - Extract structured data using schemas
4. **Chunking** - Split into semantic segments for RAG applications

## Features

| Feature | LlamaIndex | LandingAI | Gemini |
|---------|------------|-----------|--------|
| PDF Parsing | ✅ | ✅ | ✅ |
| Image Parsing | ✅ | ✅ | ✅ |
| Classification | ✅ Native | ✅ Inferred | ✅ Prompt-based |
| Schema Extraction | ✅ Pydantic | ✅ JSON Schema | ✅ Prompt-based |
| Semantic Chunking | ✅ | ✅ | ❌ |
| Grounding/BBox | Partial | ✅ | ❌ |
| Handwriting OCR | ✅ | Unknown | ✅ Specialized |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd doc_extraction_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Add your API keys:

```env
# LlamaCloud API Key (for LlamaIndex stack)
# Get from: https://cloud.llamaindex.ai
LLAMA_CLOUD_API_KEY=your_key_here

# LandingAI API Key (for LandingAI ADE stack)
# Get from: https://landing.ai
LANDINGAI_API_KEY=your_key_here

# Google AI API Key (for Gemini handwriting processing)
# Get from: https://aistudio.google.com
GOOGLE_API_KEY=your_key_here

# OpenAI API Key (for LangChain Agent with GPT-5-mini)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_key_here
```

## Quick Start

### Using the Web UI

The easiest way to test the pipeline is through the Streamlit web interface.

#### Running the UI

```bash
# Make sure you're in the project directory with virtual environment activated
cd doc_extraction_pipeline
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Ensure dependencies are installed
pip install -r requirements.txt

# Run the Streamlit app
streamlit run ui/app.py
```

The app will open in your browser at `http://localhost:8501`.

#### UI Features

**1. Document Upload**
- Drag and drop or click to upload documents
- Supported formats: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- Preview images directly in the browser

**2. Processor Selection** (Sidebar)
- **LlamaIndex**: Uses LlamaParse, LlamaClassify, LlamaExtract
- **LandingAI**: Uses ADE Parse, Extract, Split

**3. Operations** (Sidebar)
- **Parse**: Convert document to markdown
  - LlamaIndex: Choose tier (cost_effective, agentic, agentic_plus, fast)
  - Enable multimodal mode for charts/diagrams
- **Classify**: Identify document type with confidence scores
  - Optional custom categories
- **Extract**: Pull structured data from document
  - **Auto-derive schema**: AI analyzes content and identifies ALL extractable fields
  - **Manual fields**: Specify field names to extract
- **Split**: Chunk document for RAG applications
  - Configure chunk size (500-5000 chars)
  - Set overlap between chunks
  - Optional category labels
- **Full Pipeline**: Run all operations in sequence

**4. Results**
- View parsed markdown with syntax highlighting
- See classification confidence scores with progress bars
- Download extracted data as JSON
- Browse chunks with expandable sections

#### Example Workflow

1. Upload `difficult_examples/certificate_of_origin.pdf`
2. Select **LlamaIndex** processor
3. Choose **Extract** operation
4. Select **Auto-derive from content** schema mode
5. Click **Process Document**
6. Review the auto-detected fields (shipper, consignee, goods, dates, etc.)
7. Download the extracted JSON

### Using the LangChain Agent (Recommended)

The easiest way to interact with the pipeline is through the LangChain-powered agent:

```python
from src.agents import create_document_agent

# Create agent with GPT-5-mini
agent = create_document_agent(model="gpt-5-mini")

# Chat naturally about documents
response = agent.chat("Parse the document at 'invoice.pdf' and extract the total amount")
print(response)

# Or use the convenience method
result = agent.process_document("invoice.pdf", task="extract invoice details")
print(result)
```

Run the interactive demo:

```bash
python -m src.agents.demo
```

### Process a Single Document

```python
import asyncio
from main import process_document

async def main():
    # Auto-route to best processor
    result = await process_document("document.pdf", stack="auto")

    print(f"Markdown: {result.markdown[:500]}...")
    print(f"Classification: {result.classification.document_type}")
    print(f"Chunks: {result.total_chunks}")

asyncio.run(main())
```

### Use a Specific Stack

```python
# LlamaIndex stack
result = await process_document("invoice.pdf", stack="llamaindex")

# LandingAI stack
result = await process_document("form.pdf", stack="landingai")

# Gemini for handwriting
result = await process_document("handwritten.jpg", stack="gemini")
```

### Extract Structured Data

```python
from pydantic import BaseModel
from src.llamaindex_stack import LlamaIndexProcessor

class InvoiceData(BaseModel):
    invoice_number: str
    total: float
    vendor_name: str

processor = LlamaIndexProcessor()
result = await processor.process("invoice.pdf", schema=InvoiceData)

print(result.extraction.fields)
# {'invoice_number': 'INV-001', 'total': 150.00, 'vendor_name': 'Acme Corp'}
```

### Compare Stacks

```python
from main import compare_stacks

report = await compare_stacks(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    stacks=["llamaindex", "landingai"]
)

# Print comparison report
from src.evaluation import StackComparator
comparator = StackComparator({})
comparator.print_report(report)
```

## Project Structure

```
doc_extraction_pipeline/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
│
├── src/
│   ├── common/                # Shared components
│   │   ├── models.py          # Data models (ParsedDocument, Chunk, etc.)
│   │   ├── interfaces.py      # Protocols and base classes
│   │   └── router.py          # Document routing logic
│   │
│   ├── llamaindex_stack/      # LlamaIndex integration
│   │   ├── parser.py          # LlamaParse wrapper
│   │   ├── classifier.py      # LlamaClassify wrapper
│   │   ├── extractor.py       # LlamaExtract wrapper
│   │   ├── splitter.py        # Semantic chunking
│   │   └── processor.py       # Full pipeline
│   │
│   ├── landingai_stack/       # LandingAI ADE integration
│   │   ├── client.py          # ADE SDK client
│   │   ├── parser.py          # ADE Parse wrapper
│   │   ├── extractor.py       # ADE Extract wrapper
│   │   ├── splitter.py        # ADE Split wrapper
│   │   └── processor.py       # Full pipeline
│   │
│   ├── gemini/                # Google Gemini integration
│   │   └── handwriting.py     # Handwriting processor
│   │
│   ├── agents/                # LangChain Agent
│   │   ├── __init__.py        # Package exports
│   │   ├── agent.py           # DocumentAgent with GPT-5-mini
│   │   ├── skills.py          # Document processing tools
│   │   └── demo.py            # Interactive demo script
│   │
│   └── evaluation/            # Benchmarking & comparison
│       ├── metrics.py         # Accuracy metrics
│       ├── benchmark.py       # Performance benchmarks
│       └── compare.py         # Stack comparison
│
├── tests/                     # Unit tests
│   ├── test_models.py
│   ├── test_router.py
│   └── test_metrics.py
│
├── notebooks/
│   └── evaluation.py          # Evaluation script
│
├── ui/                        # Streamlit Web UI
│   ├── __init__.py
│   └── app.py                 # Main UI application
│
└── difficult_examples/        # Test documents
    ├── certificate_of_origin.pdf
    ├── patient_intake.pdf
    ├── calculus_BC_answer_sheet.jpg
    └── ...
```

## Core Components

### ParsedDocument

The unified output model from all processors:

```python
class ParsedDocument(BaseModel):
    markdown: str                    # Full parsed content
    raw_text: Optional[str]          # Plain text
    chunks: List[Chunk]              # Semantic segments
    classification: Classification   # Document type + confidence
    extraction: ExtractionResult     # Structured fields
    metadata: DocumentMetadata       # Processing info
    grounding: Optional[GroundingData]  # Bounding boxes
```

### Document Types

Supported classification categories:

- `form` - Structured forms with fields
- `invoice` - Billing documents
- `receipt` - Transaction receipts
- `certificate` - Official certificates
- `medical` - Healthcare documents
- `presentation` - Slides/decks
- `diagram` - Technical diagrams
- `flowchart` - Process flowcharts
- `spreadsheet` - Data tables
- `instructions` - Assembly guides
- `infographic` - Visual infographics
- `handwritten` - Handwritten content
- `report` - Reports and analyses

### Extraction Schemas

Built-in schemas for common document types:

```python
from src.common.models import (
    InvoiceSchema,
    FormSchema,
    CertificateSchema,
    MedicalFormSchema,
    PresentationSchema,
)
```

## API Reference

### LlamaIndexProcessor

```python
from src.llamaindex_stack import LlamaIndexProcessor

processor = LlamaIndexProcessor(api_key="optional_override")

# Full pipeline
result = await processor.process(
    file_path="document.pdf",
    schema=MySchema,                    # Optional extraction schema
    classification_rules=rules,         # Optional custom rules
    chunk_categories=["intro", "body"], # Optional chunk labels
    parse_tier="agentic",              # cost_effective|agentic|agentic_plus|fast
    multimodal=False,                   # Enable for visual documents
)

# Individual steps
markdown = await processor.parse("doc.pdf", tier="agentic")
classification = await processor.classify(markdown, rules)
extraction = await processor.extract(markdown, MySchema)
chunks = await processor.split(markdown, categories)
```

### LandingAIProcessor

```python
from src.landingai_stack import LandingAIProcessor

processor = LandingAIProcessor(
    api_key="optional_override",
    region="us"  # or "eu" for EU data residency
)

result = await processor.process(
    file_path="document.pdf",
    include_grounding=True,  # Get bounding boxes
    page_level=False,        # Page-by-page results
)

# Don't forget to close
await processor.close()
```

### GeminiHandwritingProcessor

```python
from src.gemini import GeminiHandwritingProcessor

processor = GeminiHandwritingProcessor(
    model="gemini-2.0-flash"  # or gemini-2.0-pro for complex docs
)

# Detect handwriting
detection = await processor.detect_handwriting("document.jpg")
print(f"Has handwriting: {detection['has_handwriting']}")
print(f"Percentage: {detection['handwriting_percentage']:.0%}")

# Extract handwritten content
result = await processor.process_handwriting(
    "exam.jpg",
    mode="math"  # general|math|form
)
print(result["text"])
```

### Document Router

Automatically route documents to the best processor:

```python
from src.common.router import DocumentRouter, ProcessorType

router = DocumentRouter(
    default_processor=ProcessorType.LLAMAINDEX,
    prefer_gemini_for_handwriting=True
)

# Get routing decision
processor_type, options = router.route("document.pdf")
print(f"Use: {processor_type.value}")  # llamaindex, landingai, or gemini

# Get classification hint from filename
hint = router.get_classification_hint("patient_intake.pdf")
print(hint)  # DocumentType.MEDICAL
```

### SchemaGenerator (Auto-derive Extraction Schemas)

Automatically analyze document content and derive a comprehensive extraction schema:

```python
from src.common import SchemaGenerator

# Create generator
generator = SchemaGenerator(model="gpt-5-mini")

# Derive schema from document content
markdown = "Invoice #12345\nDate: 2024-01-15\nVendor: Acme Corp\nTotal: $1,500.00"
derived_schema = generator.derive_schema(markdown)

print(f"Document Type: {derived_schema.document_type}")
print(f"Fields Found: {len(derived_schema.fields)}")
for field in derived_schema.fields:
    print(f"  - {field.name}: {field.field_type} ({'required' if field.is_required else 'optional'})")

# Create Pydantic model from schema
Model = generator.create_pydantic_model(derived_schema)

# Or get JSON Schema for LandingAI
json_schema = generator.schema_to_json_schema(derived_schema)
```

The generator uses GPT-5-mini to analyze document content and identify:
- All extractable fields (names, dates, numbers, IDs, etc.)
- Appropriate data types for each field
- Whether fields are required or optional
- Whether fields contain multiple values (lists)

### DocumentAgent (LangChain)

The agent provides a conversational interface to all document processing capabilities:

```python
from src.agents import create_document_agent, DocumentAgent

# Create with defaults (GPT-5-mini, temperature=0)
agent = create_document_agent()

# Or customize
agent = DocumentAgent(
    model_name="gpt-5-mini",
    temperature=0.0,
    api_key="optional_override",  # Uses OPENAI_API_KEY env var by default
    memory=True,                  # Enable conversation memory
    verbose=False
)

# Simple chat interface
response = agent.chat(
    "Parse invoice.pdf and tell me the total",
    thread_id="my-session"
)

# Full response with metadata
result = agent.invoke("What type of document is this?", thread_id="my-session")
print(result["response"])      # Agent's text response
print(result["tool_calls"])    # Tools that were called
print(result["thread_id"])     # Thread ID for continuity

# Streaming responses
for chunk in agent.stream("Process this document"):
    print(chunk)

# Reset conversation memory
agent.reset_memory()
```

#### Available Agent Tools

| Tool | Description |
|------|-------------|
| `parse_document` | Convert PDF/image to markdown |
| `classify_document` | Identify document type |
| `extract_from_document` | Extract structured fields |
| `split_document` | Create semantic chunks |
| `process_document_full` | Run complete pipeline |

#### Example Prompts

```python
# Parsing
agent.chat("Parse the document at 'form.pdf'")

# Classification
agent.chat("What type of document is this?")

# Extraction
agent.chat("Extract the invoice number, date, and total")

# Chunking
agent.chat("Split this into chunks for RAG")

# Full pipeline
agent.chat("Process invoice.pdf: parse, classify, extract all fields, and chunk")
```

## Evaluation

### Run Full Evaluation

```bash
python notebooks/evaluation.py
```

This processes all documents in `difficult_examples/` through both stacks and generates a comparison report.

### Metrics

- **Text Similarity** - Token/character overlap with ground truth
- **Extraction F1** - Precision/recall for structured extraction
- **Table Accuracy** - Cell-level accuracy for tables
- **Chunk Quality** - Coherence and coverage metrics
- **Classification Accuracy** - Document type prediction

### Benchmarking

```python
from src.evaluation import Benchmark

benchmark = Benchmark(warmup_runs=1, benchmark_runs=3)
results = await benchmark.run_batch(processor, file_paths)

summary = benchmark.get_summary()
print(f"Avg time: {summary['processors']['llamaindex']['avg_time_ms']}ms")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Supported File Formats

| Format | LlamaIndex | LandingAI | Gemini |
|--------|------------|-----------|--------|
| PDF | ✅ | ✅ | ✅ |
| PNG | ✅ | ✅ | ✅ |
| JPG/JPEG | ✅ | ✅ | ✅ |
| GIF | ✅ | ❌ | ✅ |
| BMP | ✅ | ❌ | ✅ |
| TIFF | ✅ | ❌ | ✅ |
| WebP | ✅ | ❌ | ✅ |
| XLSX | ❌ | ✅ | ❌ |
| CSV | ❌ | ✅ | ❌ |

## Cost Estimates

Approximate costs per 1,000 pages:

| Stack | Operation | Cost |
|-------|-----------|------|
| LlamaIndex | Parse (Agentic) | ~$10 |
| LlamaIndex | Parse (Agentic Plus) | ~$20 |
| LlamaIndex | Classify | ~$1 |
| LlamaIndex | Extract | ~$5 |
| LandingAI | Parse | ~$8 |
| LandingAI | Extract | ~$4 |
| Gemini Flash | Process | ~$2 |

## Troubleshooting

### API Key Errors

```
ValueError: API key required. Set LLAMA_CLOUD_API_KEY environment variable
```

Ensure your `.env` file is properly configured and loaded:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Timeout Errors

For large documents, increase the timeout:

```python
processor = LandingAIProcessor(timeout=300.0)  # 5 minutes
```

### Unsupported Format

```
ValueError: No processor supports file type: .docx
```

Convert DOCX files to PDF before processing, or use a dedicated converter.

### Streamlit UI Issues

**Port already in use:**
```bash
streamlit run ui/app.py --server.port 8502
```

**Module not found errors:**
```bash
# Ensure you're in the project root directory
cd doc_extraction_pipeline
pip install -r requirements.txt
```

**API key not detected in UI:**
- Check that `.env` file exists in the project root
- Verify the key names match exactly (e.g., `LLAMA_CLOUD_API_KEY`)
- Restart the Streamlit app after modifying `.env`

## Web UI Reference

### Starting the UI

```bash
# Basic startup
streamlit run ui/app.py

# With custom port
streamlit run ui/app.py --server.port 8502

# With auto-reload disabled (for production)
streamlit run ui/app.py --server.runOnSave false
```

### UI Configuration Options

| Setting | Location | Options |
|---------|----------|---------|
| Processor | Sidebar | LlamaIndex, LandingAI |
| Operation | Sidebar | Parse, Classify, Extract, Split, Full Pipeline |
| Parse Tier | Sidebar (Parse only) | cost_effective, agentic, agentic_plus, fast |
| Multimodal | Sidebar (Parse only) | Enable for visual documents |
| Schema Mode | Sidebar (Extract only) | Auto-derive, Manual fields |
| Chunk Size | Sidebar (Split only) | 500-5000 characters |

### Auto-Schema Derivation

When using **Extract** with **Auto-derive from content**:

1. The document is first parsed to markdown
2. GPT-5-mini analyzes the content to identify ALL extractable fields
3. A dynamic schema is generated with:
   - Field names (snake_case)
   - Data types (string, number, date, boolean, list)
   - Required/optional flags
   - Field descriptions
4. You can review and edit the schema before extraction
5. The schema is used for structured extraction

This approach ensures no information is lost because the AI comprehensively scans the entire document.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## Loan Processing Pipeline

A specialized end-to-end document processing pipeline for loan applications, inspired by the LandingAI course (L9). This pipeline automates the processing of common loan application documents.

### Overview

The loan processor handles:
1. **Parse** - Convert uploaded documents to markdown
2. **Categorize** - Classify each document type (ID, W2, pay stub, bank statement, investment statement)
3. **Extract** - Pull relevant fields using document-specific schemas
4. **Validate** - Cross-document consistency checks (name matching, year verification, asset totals)
5. **Visualize** - Display extracted fields with bounding boxes

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoanProcessingPipeline                       │
│                      (pipeline.py)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   ADEClient   │   │  Categorizer  │   │   Extractor   │
│   (parse)     │   │  (classify)   │   │   (extract)   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌───────────────┐
                    │   Validator   │
                    │ (cross-doc)   │
                    └───────────────┘
```

**Note:** This is a sequential async pipeline, not a LangGraph flow. The design is intentionally simple since the workflow is linear and deterministic.

### Schema-Driven Auto-Extraction

The key design principle is **schema-driven extraction**. Each document type has a predefined Pydantic schema that defines what fields to extract:

#### 1. Define Schemas per Document Type

```python
# src/pipelines/loan_processing/schemas.py

class W2Schema(BaseModel):
    employee_name: Optional[str] = Field(None, description="Employee's full name")
    employer_name: Optional[str] = Field(None, description="Employer's name")
    w2_year: Optional[int] = Field(None, description="Tax year of the W2")
    wages_box_1: Optional[float] = Field(None, description="Box 1: Wages, tips, other compensation")
    federal_tax_withheld: Optional[float] = Field(None, description="Box 2: Federal income tax withheld")
    # ... more fields

class BankStatementSchema(BaseModel):
    account_owner: Optional[str] = Field(None, description="Name of account holder")
    bank_name: Optional[str] = Field(None, description="Name of the bank")
    closing_balance: Optional[float] = Field(None, description="Closing/ending balance")
    # ... more fields

# Registry maps document types to schemas
LOAN_SCHEMA_REGISTRY = {
    LoanDocumentType.ID: IDSchema,
    LoanDocumentType.W2: W2Schema,
    LoanDocumentType.PAY_STUB: PayStubSchema,
    LoanDocumentType.BANK_STATEMENT: BankStatementSchema,
    LoanDocumentType.INVESTMENT_STATEMENT: InvestmentStatementSchema,
}
```

#### 2. Convert Pydantic to JSON Schema

```python
# src/pipelines/loan_processing/extractor.py

def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to JSON schema for LandingAI ADE."""
    schema = model.model_json_schema()

    json_schema = {"type": "object", "properties": {}, "required": []}

    for field_name, field_info in schema.get("properties", {}).items():
        json_schema["properties"][field_name] = {
            "type": field_info.get("type", "string"),
            "description": field_info.get("description", f"Extract {field_name}")
        }

    return json_schema
```

#### 3. Extract Using LandingAI ADE

```python
# The extraction flow:
async def extract(self, file_path, document_type, markdown_content):
    # 1. Get schema for document type
    schema_class = get_schema_for_loan_document(document_type)  # e.g., W2Schema

    # 2. Convert to JSON schema
    json_schema = pydantic_to_json_schema(schema_class)

    # 3. Call LandingAI ADE extract API
    result = await self.client.extract(
        content=markdown_content,  # Parsed document text
        schema=json_schema         # Fields to extract
    )

    # 4. Return structured data
    return DocumentExtractionResult(fields=result["extraction"], ...)
```

The LandingAI ADE API uses a vision-language model to:
- Read the markdown content
- Understand the JSON schema (field names + descriptions)
- Locate and extract each field value
- Return structured data matching the schema

### Supported Document Types

| Type | Description | Key Fields Extracted |
|------|-------------|---------------------|
| **ID** | Driver's license, passport, state ID | name, issuer, issue_date, expiration_date, identifier, DOB, address |
| **W2** | W-2 tax form | employee_name, employer_name, w2_year, wages_box_1, federal_tax_withheld |
| **Pay Stub** | Paycheck/earnings statement | employee_name, employer_name, pay_period, gross_pay, net_pay, ytd_gross |
| **Bank Statement** | Checking/savings statement | account_owner, bank_name, account_number, closing_balance, total_deposits |
| **Investment Statement** | Brokerage, 401k, IRA | account_owner, institution_name, total_value, total_cash, total_securities |

### Cross-Document Validation

The validator performs these checks:

```python
# src/pipelines/loan_processing/validator.py

class LoanValidationResult(BaseModel):
    name_match: bool              # Names consistent across documents
    names_found: List[str]        # All names extracted
    years_valid: bool             # Documents from acceptable years
    years_found: List[int]        # All years extracted
    total_bank_balance: float     # Sum of bank balances
    total_investment_value: float # Sum of investment values
    total_assets: float           # Bank + investments
    annual_income: Optional[float] # From W2 or pay stubs
    monthly_income: Optional[float]
    validation_passed: bool       # Overall pass/fail
    issues: List[str]             # List of problems found
```

### Document Parse Caching

The pipeline caches parsed documents to avoid redundant API calls:

```python
# Cache is stored in ./parsed/ directory
pipeline = LoanProcessingPipeline(
    use_cache=True,           # Enable caching (default)
    cache_dir="./parsed"      # Cache directory
)

# Force re-parse (ignore cache)
result = await pipeline.process_application(documents, force_reparse=True)

# Clear cache
pipeline.clear_cache()  # Clear all
pipeline.clear_cache(file_path)  # Clear specific file
```

### Using the Loan Processor

#### Via Streamlit UI

```bash
streamlit run ui/app.py
# Navigate to "Loan Processor" page in sidebar
```

Features:
- Multi-file upload (drag & drop or Ctrl+click)
- Files accumulate across uploads
- Processing status with cache indicators
- Results dashboard with validation summary
- Export to JSON/CSV

#### Via Python API

```python
from src.pipelines.loan_processing import LoanProcessingPipeline
from pathlib import Path

# Initialize
pipeline = LoanProcessingPipeline(
    min_year=2023,    # Reject documents older than this
    max_year=2025,    # Reject documents from future years
    use_cache=True
)

# Process documents
documents = [
    Path("id_card.jpg"),
    Path("w2_form.pdf"),
    Path("bank_statement.pdf")
]

result = await pipeline.process_application(documents, validate=True)

# Access results
for doc in result.documents:
    print(f"{doc.file_name}: {doc.document_type.value}")
    print(f"  Fields: {doc.extraction.fields}")

# Validation summary
if result.validation:
    print(f"Validation passed: {result.validation.validation_passed}")
    print(f"Total assets: ${result.validation.total_assets:,.2f}")
```

#### Via Command Line

```bash
# Process documents
python -m src.pipelines.loan_processing.pipeline doc1.pdf doc2.jpg doc3.pdf

# Force re-parse (ignore cache)
python -m src.pipelines.loan_processing.pipeline --no-cache doc1.pdf

# Clear cache
python -m src.pipelines.loan_processing.pipeline --clear-cache
```

### Pipeline Components

| Component | File | Purpose |
|-----------|------|---------|
| **Schemas** | `schemas.py` | Pydantic models for each document type |
| **Categorizer** | `categorizer.py` | Classify document type using ADE extract |
| **Extractor** | `extractor.py` | Extract fields using type-specific schemas |
| **Validator** | `validator.py` | Cross-document validation logic |
| **Pipeline** | `pipeline.py` | Orchestration with caching |
| **Visualizer** | `visualizer.py` | Bounding box annotations (optional) |

### Project Structure

```
src/pipelines/loan_processing/
├── __init__.py           # Package exports
├── schemas.py            # Document type enum + extraction schemas
├── categorizer.py        # Document classification
├── extractor.py          # Schema-based field extraction
├── validator.py          # Cross-document validation
├── pipeline.py           # Main orchestrator with caching
└── visualizer.py         # Bounding box visualization

ui/pages/
└── 1_Loan_Processor.py   # Streamlit UI page
```

### Why Not LangGraph?

The loan pipeline is **linear and deterministic**:

```
Parse → Categorize → Extract → Validate → Done
```

LangGraph would add value if we needed:
- **Conditional routing** (different flows for different document types)
- **Human-in-the-loop** (manual review of low-confidence extractions)
- **Retry logic** (re-extract with different prompts if confidence is low)
- **Parallel branching** (extract using multiple models and compare)

For simple sequential workflows, an async pipeline is simpler and more efficient.

---

## License

This project is for research and evaluation purposes.

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) - LlamaParse, LlamaExtract, LlamaClassify
- [LandingAI](https://landing.ai/) - Agentic Document Extraction (ADE)
- [Google AI](https://ai.google.dev/) - Gemini multimodal models
