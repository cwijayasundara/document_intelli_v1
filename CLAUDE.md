# CLAUDE.md - AI Assistant Guide

This file provides context for AI assistants working with this codebase.

## Project Overview

This is a **document extraction pipeline** that compares multiple commercial document AI platforms:

1. **LlamaIndex Stack** - Uses LlamaCloud APIs (LlamaParse, LlamaClassify, LlamaExtract)
2. **LandingAI Stack** - Uses LandingAI ADE (Agentic Document Extraction) API
3. **Gemini Integration** - Uses Google Gemini for handwriting recognition

The goal is to evaluate and compare these platforms for document parsing, classification, structured extraction, and semantic chunking.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interfaces                           │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────┐  ┌────────────────────────┐     │
│  │   Streamlit Web UI     │  │  LangChain Agent       │     │
│  │   (ui/app.py)          │  │  (src/agents/)         │     │
│  │  - Visual interface    │  │  - Chat interface      │     │
│  │  - File upload         │  │  - Conversation memory │     │
│  │  - Auto-schema derive  │  │  - Tool selection      │     │
│  └───────────┬────────────┘  └───────────┬────────────┘     │
│              │                           │                   │
│              └─────────────┬─────────────┘                   │
│                            │                                 │
└────────────────────────────┼────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Document Input                           │
│         (PDF, PNG, JPG, etc.)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Document Router                            │
│   (Determines best processor based on file type/content)    │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ LlamaIndex  │ │  LandingAI  │ │   Gemini    │
│   Stack     │ │    Stack    │ │ Handwriting │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ParsedDocument                             │
│   (Unified output: markdown, chunks, classification,        │
│    extraction, grounding, metadata)                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

### Entry Points
- `main.py` - High-level functions: `process_document()`, `compare_stacks()`, `run_benchmark()`
- `ui/app.py` - Streamlit web interface: `streamlit run ui/app.py`
- `src/agents/demo.py` - Interactive agent demo: `python -m src.agents.demo`
- `notebooks/evaluation.py` - Evaluation script for test documents

### Common Module (`src/common/`)
- `models.py` - Pydantic models: `ParsedDocument`, `Chunk`, `Classification`, `ExtractionResult`
- `interfaces.py` - `DocumentProcessor` protocol and `ClassificationRule`
- `router.py` - `DocumentRouter` for automatic processor selection
- `schema_generator.py` - `SchemaGenerator` for auto-deriving extraction schemas from content

### LlamaIndex Stack (`src/llamaindex_stack/`)
- `parser.py` - `LlamaParseWrapper` - Document to markdown
- `classifier.py` - `LlamaClassifyWrapper` - Document type classification
- `extractor.py` - `LlamaExtractWrapper` - Schema-based extraction
- `splitter.py` - `LlamaSplitWrapper` - Semantic chunking
- `processor.py` - `LlamaIndexProcessor` - Full pipeline orchestration

### LandingAI Stack (`src/landingai_stack/`)
- `client.py` - `ADEClient` - Wraps official `landingai-ade` SDK
- `parser.py` - `ADEParseWrapper` - Document parsing with grounding
- `extractor.py` - `ADEExtractWrapper` - JSON schema extraction
- `splitter.py` - `ADESplitWrapper` - Section classification
- `processor.py` - `LandingAIProcessor` - Full pipeline orchestration

### Gemini (`src/gemini/`)
- `handwriting.py` - `GeminiHandwritingProcessor` - Handwriting recognition

### Evaluation (`src/evaluation/`)
- `metrics.py` - Text similarity, extraction accuracy, chunk quality metrics
- `benchmark.py` - Performance benchmarking
- `compare.py` - Side-by-side stack comparison

### LangChain Agent (`src/agents/`)
- `skills.py` - Document processing tools using `@tool` decorator
- `agent.py` - `DocumentAgent` class using `create_react_agent` and `init_chat_model()`
- `demo.py` - Interactive demo script

### Web UI (`ui/`)
- `app.py` - Streamlit application for interactive testing
- Features:
  - Document upload (PDF, images)
  - Processor selection (LlamaIndex/LandingAI)
  - Operation selection (Parse/Classify/Extract/Split/Full Pipeline)
  - Auto-schema derivation for extraction
  - Results display with download options
- Run with: `streamlit run ui/app.py`

### Schema Generator (`src/common/schema_generator.py`)
- `SchemaGenerator` - Auto-derives extraction schemas from document content
- Uses GPT-5-mini to analyze documents and identify all extractable fields
- Generates Pydantic models or JSON Schema for extraction

## Important Patterns

### Async/Await
All processing methods are async:
```python
result = await processor.process(file_path)
```

### Pydantic Models
All data models use Pydantic v2:
```python
from pydantic import BaseModel, Field

class MySchema(BaseModel):
    field_name: str = Field(..., description="Field description")
```

### Environment Variables
API keys are loaded from environment:
```python
api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
```

### Error Handling
Processing steps catch errors and add to warnings/errors lists:
```python
try:
    result = await self.classify(content)
except Exception as e:
    warnings.append(f"Classification failed: {str(e)}")
```

## SDK Dependencies

### LlamaCloud SDK
```python
from llama_cloud import LlamaCloud

client = LlamaCloud(api_key=api_key)
result = client.parsing.parse(
    tier="agentic",
    version="latest",
    upload_file=(filename, content),
    expand=["markdown", "text", "items"]
)
# Access: result.markdown.pages[0].markdown
```

### LandingAI ADE SDK
```python
from landingai_ade import LandingAIADE

client = LandingAIADE(apikey=api_key)
result = client.parse(
    document=(filename, content, mime_type)  # Tuple format required
)
# Access: result.markdown (string)
```

### Google Gemini SDK
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        types.Part.from_text(text=prompt)
    ]
)
# Access: response.text
```

### LangChain Agent SDK
```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Initialize LLM
llm = init_chat_model(
    model="gpt-5-mini",
    model_provider="openai",
    temperature=0.0,
    api_key=api_key
)

# Create tool
@tool
def parse_document(file_path: str) -> str:
    """Parse a document to markdown."""
    # Implementation
    return markdown

# Create agent
agent = create_react_agent(
    model=llm,
    tools=[parse_document],
    checkpointer=MemorySaver(),
    state_modifier="System prompt here"
)

# Invoke
result = agent.invoke(
    {"messages": [HumanMessage(content="Parse doc.pdf")]},
    config={"configurable": {"thread_id": "session-1"}}
)
```

## Common Tasks

### Adding a New Document Type
1. Add to `DocumentType` enum in `src/common/models.py`
2. Create extraction schema in `src/common/models.py`
3. Register in `SCHEMA_REGISTRY`
4. Add classification rule in `get_default_rules()`

### Adding a New Processor
1. Implement `DocumentProcessor` protocol from `src/common/interfaces.py`
2. Add to `ProcessorType` enum in `src/common/router.py`
3. Update routing logic in `DocumentRouter.route()`
4. Register supported formats

### Adding New Metrics
1. Add metric function to `src/evaluation/metrics.py`
2. Integrate into `EvaluationMetrics` dataclass
3. Call from `StackComparator._calculate_metrics()`

### Adding a New Agent Skill
1. Create function with `@tool` decorator in `src/agents/skills.py`
2. Add comprehensive docstring (used by the LLM to understand the tool)
3. Add to `ALL_TOOLS` list
4. Tools are automatically available to the agent

## Test Documents

Located in `difficult_examples/`:

| Document | Challenge | Best For |
|----------|-----------|----------|
| certificate_of_origin.pdf | Tables, formal layout | Testing table extraction |
| patient_intake.pdf | Forms, checkboxes | Testing form extraction |
| calculus_BC_answer_sheet.jpg | **Handwritten math** | Testing Gemini |
| hr_process_flowchart.png | Flowchart, spatial | Testing visual understanding |
| sales_volume.png | Dense data grid | Testing table accuracy |
| ikea-assembly.pdf | Diagrams, minimal text | Testing visual parsing |
| Investor_Presentation_pg7.png | Charts, financial data | Testing chart extraction |
| ikea_infographic.jpg | Timeline, mixed content | Testing layout understanding |
| virology_pg2.pdf | Scientific text | Testing text extraction |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_models.py::TestParsedDocument -v

# With output
pytest tests/ -v -s
```

## Running Evaluation

```bash
# Full evaluation
python notebooks/evaluation.py

# Results saved to notebooks/results/
```

## Debugging Tips

### Check API Response Structure
```python
result = client.parsing.parse(...)
print(type(result))
print([a for a in dir(result) if not a.startswith('_')])
```

### Inspect Pydantic Model
```python
print(result.model_dump_json(indent=2))
```

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Web UI Architecture

### Key Functions (`ui/app.py`)

| Function | Purpose |
|----------|---------|
| `render_sidebar()` | Renders configuration options, returns settings dict |
| `render_file_upload()` | Handles document upload, returns UploadedFile |
| `run_parse()` | Async wrapper for parsing operation |
| `run_classify()` | Async wrapper for classification |
| `run_extract()` | Async wrapper for extraction with schema |
| `run_split()` | Async wrapper for chunking |
| `run_full_pipeline()` | Runs all operations sequentially |
| `render_results()` | Displays results based on operation type |
| `render_schema_derivation()` | Shows derived schema with edit option |

### Session State
```python
st.session_state.settings  # Current configuration
st.session_state.results   # Last processing results
st.session_state.operation # Last operation type
```

### Adding New UI Features
1. Add settings to `render_sidebar()`
2. Create async handler function (e.g., `run_new_operation()`)
3. Add to operation selection in sidebar
4. Update `render_results()` to display new output

## Known Issues

1. **LlamaIndex Classification Confidence** - Returns low confidence (3-5%) due to keyword-based fallback
2. **LandingAI Chunking** - SDK doesn't return chunks by default, uses local splitter
3. **Gemini File Upload** - Must use `Part.from_bytes()` not file upload API
4. **Streamlit Async** - Use `run_async()` helper to call async functions from sync Streamlit code

## Code Style

- Python 3.9+ with type hints
- Async/await for all I/O operations
- Pydantic v2 for data validation
- pytest for testing
- Google-style docstrings

## Environment Setup

```bash
# Required environment variables
LLAMA_CLOUD_API_KEY=llx-...
LANDINGAI_API_KEY=...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...  # For LangChain Agent
```

## Quick Reference

### Process a Document
```python
from main import process_document
result = await process_document("doc.pdf", stack="auto")
```

### Access Results
```python
result.markdown          # Full text
result.chunks            # List[Chunk]
result.classification    # Classification
result.extraction        # ExtractionResult
result.metadata          # DocumentMetadata
```

### Custom Extraction
```python
from pydantic import BaseModel

class MySchema(BaseModel):
    field1: str
    field2: Optional[int]

result = await processor.process("doc.pdf", schema=MySchema)
print(result.extraction.fields)
```

### Using the LangChain Agent
```python
from src.agents import create_document_agent

# Create agent
agent = create_document_agent(model="gpt-5-mini")

# Chat interface
response = agent.chat("Parse invoice.pdf and extract the total")

# With conversation memory
agent.chat("What was the total?", thread_id="session-1")

# Process document shortcut
result = agent.process_document("form.pdf", task="classify and extract all fields")

# Run demo
python -m src.agents.demo
```

### Agent Tools
```python
from src.agents.skills import (
    parse_document,      # Convert to markdown
    classify_document,   # Identify document type
    extract_from_document,  # Extract fields
    split_document,      # Semantic chunking
    process_document_full,  # Full pipeline
)

# Direct tool usage (bypassing agent)
markdown = parse_document.invoke({"file_path": "doc.pdf"})
classification = classify_document.invoke({"content": markdown})
```

### Schema Derivation
```python
from src.common import SchemaGenerator

generator = SchemaGenerator()
derived_schema = generator.derive_schema(markdown_content)

# Get Pydantic model
Model = generator.create_pydantic_model(derived_schema)

# Get JSON Schema
json_schema = generator.schema_to_json_schema(derived_schema)
```

### Run Web UI
```bash
# Basic startup
streamlit run ui/app.py

# Custom port
streamlit run ui/app.py --server.port 8502

# Opens at http://localhost:8501
```

### UI Workflow
1. Upload document (drag & drop or click)
2. Select processor in sidebar (LlamaIndex or LandingAI)
3. Choose operation (Parse, Classify, Extract, Split, Full Pipeline)
4. Configure operation-specific settings
5. Click "Process Document"
6. View/download results

### UI Auto-Schema Derivation
When using Extract with "Auto-derive from content":
```python
# The UI internally does:
from src.common import SchemaGenerator

generator = SchemaGenerator()
# 1. Parse document to markdown
# 2. Derive schema from content
derived_schema = generator.derive_schema(markdown)
# 3. Display fields for review/edit
# 4. Create model and extract
Model = generator.create_pydantic_model(derived_schema)
```
