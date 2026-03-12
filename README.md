# Document Extraction Pipeline

A multi-stack document parsing and extraction platform that provides comparative evaluation between **LlamaIndex**, **LandingAI**, and **Reducto** commercial APIs, with **Gemini** integration for handwriting recognition.

## Overview

This pipeline processes documents through multiple stages:
1. **Parsing** - Convert documents to structured markdown
2. **Classification** - Identify document type (invoice, form, certificate, etc.)
3. **Extraction** - Extract structured data using schemas
4. **Chunking** - Split into semantic segments for RAG applications

## Features

| Feature | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence | Gemini |
|---------|------------|-----------|---------|----------------------|--------|
| PDF Parsing | ✅ | ✅ | ✅ | ✅ | ✅ |
| Image Parsing | ✅ | ✅ | ✅ | ✅ | ✅ |
| Classification | ✅ Native | ✅ Inferred | ✅ Inferred (block types) | ✅ Prebuilt + custom | ✅ Prompt-based |
| Schema Extraction | ✅ Pydantic | ✅ JSON Schema | ✅ JSON Schema | ✅ Prebuilt models + custom | ✅ Prompt-based |
| Semantic Chunking | ✅ | ✅ | ✅ (section + variable) | ✅ (paragraphs, sections) | ❌ |
| Grounding/BBox | Partial | ✅ | ✅ (per-block) | ✅ (word + line + paragraph) | ❌ |
| Handwriting OCR | ✅ | Unknown | ✅ (agentic text) | ✅ Native | ✅ Specialized |
| DOCX/PPTX Support | ✅ | ✅ (converted to PDF internally) | ✅ | ✅ | ❌ |
| Spreadsheet Support | ✅ (XLSX, CSV, XLS, TSV) | ✅ (XLSX, CSV) | ✅ (XLSX, CSV) | ✅ (XLSX) | ❌ |
| Audio Transcription | ✅ (MP3, WAV, MP4) | ❌ | ❌ | ❌ | ❌ |

## Stack Comparison

### Comprehensive Comparison: LlamaIndex vs LandingAI vs Reducto vs Azure Document Intelligence

> **Note:** Azure Document Intelligence (formerly Azure Form Recognizer) is included for reference as a leading hyperscaler solution commonly evaluated by enterprise customers. It is **not implemented** in this pipeline but is shown here to help teams make informed build-vs-buy decisions.

#### Document Parsing

| Aspect | LlamaIndex (LlamaParse) | LandingAI (ADE Parse) | Reducto (Parse) | Azure Doc Intelligence |
|--------|------------------------|----------------------|-----------------|----------------------|
| **Output Format** | Markdown (direct) | Markdown (direct) | Blocks → assembled to markdown | Structured JSON with content + roles |
| **Parsing Tiers** | 4 tiers (cost_effective, agentic, agentic_plus, fast) | Single tier | Single tier with agentic enhancement | Prebuilt models + custom models |
| **Multimodal** | Optional (2x credits) | Built-in | Built-in with figure summarization | Built-in (vision + OCR fused) |
| **Table Handling** | Markdown tables | Markdown tables | Configurable (HTML, markdown, JSON, CSV) | Structured cells with row/col spans, headers, bounding boxes |
| **Figure Handling** | Extracted as images | Extracted with metadata | Extracted + optional AI summarization | Extracted with captions and bounding regions |
| **OCR Quality** | Good, improves with higher tiers | Good | Hybrid mode (OCR + embedded text) | Excellent — enterprise-grade OCR with 300+ language support |
| **Page-Level Output** | ✅ Per-page markdown | ✅ Per-page segments | ✅ Page markers + block-level pages | ✅ Per-page with word/line/paragraph hierarchy |
| **Processing Speed** | Varies by tier (fast ~2s, agentic ~10s) | ~5-8s per page | ~3-6s per page | Varies by model and document complexity |
| **Scanned Docs** | Supported via multimodal tier | Supported | Enhanced via agentic text correction | Excellent — purpose-built for scanned docs |
| **Output Granularity** | Page-level chunks | Document-level markdown + chunks | Block-level (Title, Text, Table, Figure, Key Value) | Word → line → paragraph → section hierarchy |

#### Structured Extraction

| Aspect | LlamaIndex (LlamaExtract) | LandingAI (ADE Extract) | Reducto (Extract) | Azure Doc Intelligence |
|--------|--------------------------|------------------------|-------------------|----------------------|
| **Input** | Markdown text | Markdown text | Uploaded document (file-based) | Uploaded document (file or URL) |
| **Schema Format** | Pydantic models | JSON Schema string | JSON Schema | Prebuilt schemas + custom trained models |
| **Auto-Extraction** | ✅ (detect fields without schema) | ❌ | ❌ | ✅ (prebuilt models for invoices, receipts, IDs, W-2s, etc.) |
| **Extraction Modes** | 4 modes (fast, balanced, premium, multimodal) | Single mode | Single mode with parsing config | Prebuilt, custom, composed models |
| **Field Confidence** | Per-field confidence scores | Per-field confidence | Per-field (via block confidence) | ✅ Per-field confidence (0-1) |
| **Grounding** | Not available | ✅ Per-field bounding boxes | ✅ Via block bounding boxes | ✅ Per-field bounding regions with polygon coordinates |
| **Nested Objects** | ✅ | ✅ | ✅ | ✅ (via composed/custom models) |
| **List Fields** | ✅ | ✅ | ✅ | ✅ (line items in invoices, etc.) |
| **Citation Support** | ❌ | ❌ | ✅ (configurable) | ✅ (spans reference source content) |
| **Prebuilt Doc Types** | ❌ | ❌ | ❌ | ✅ Invoice, receipt, ID, W-2, health insurance, US tax forms, contract, US mortgage, marriage certificate, credit/debit card, business card, pay stub, bank statement |

#### Semantic Splitting / Chunking

| Aspect | LlamaIndex (LlamaSplit) | LandingAI (ADE Split) | Reducto (Split) | Azure Doc Intelligence |
|--------|------------------------|----------------------|-----------------|----------------------|
| **API-Based** | ❌ (local rule-based) | ✅ | ✅ | ✅ (Layout model) |
| **Chunk Modes** | Header-based + size limits | Category-based | Section-based with descriptions | Paragraph, section heading, page-based |
| **Category Classification** | Keyword matching | API-powered | AI-powered with confidence | Role-based (title, sectionHeading, pageHeader, pageFooter, footnote) |
| **Custom Categories** | ✅ | ✅ | ✅ (name + description pairs) | ❌ (fixed roles from Layout model) |
| **Overlap Support** | ✅ Configurable | ❌ | ✅ (variable mode) | ❌ (structural boundaries) |
| **Page Mapping** | ❌ | ✅ | ✅ (pages per split) | ✅ (page number + bounding regions) |
| **Confidence Scores** | ❌ | ✅ | ✅ (high/low) | ✅ (per-element confidence) |
| **Chunking Strategies** | Header-based, paragraph | Category-based | variable, section, page, block, page_sections | Paragraph, section, table, figure (structural) |

#### Document Classification

| Aspect | LlamaIndex (LlamaClassify) | LandingAI | Reducto | Azure Doc Intelligence |
|--------|---------------------------|-----------|---------|----------------------|
| **Dedicated API** | ✅ | ❌ | ❌ | ✅ (Custom classification model) |
| **Method** | Native classifier API | Inferred from split categories | Inferred from block type frequency | Custom-trained classifier or prebuilt model routing |
| **Confidence** | Returns confidence scores (often low 3-5%) | Calculated from category distribution | Calculated from block type ratios | Per-class confidence scores |
| **Custom Labels** | ✅ With descriptions | ✅ Via split categories | ✅ Via keyword rules | ✅ Custom-trained with labeled samples |
| **Multimodal** | ✅ (fast or multimodal mode) | ❌ | ❌ (uses parsed content) | ✅ (visual + text features) |
| **Fallback** | Keyword-based if API fails | Always keyword/category | Block-type → keyword fallback | N/A (model-based) |
| **Training Required** | ❌ | ❌ | ❌ | ✅ (minimum 5 samples per class) |

#### Grounding & Traceability

| Aspect | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence |
|--------|------------|-----------|---------|----------------------|
| **Bounding Boxes** | Partial (items only) | ✅ Full document | ✅ Per-block | ✅ Word, line, paragraph, field-level polygons |
| **Coordinate System** | Varies | Absolute pixels | Normalized (0-1) | Points-based polygons (inches from origin) |
| **Page Reference** | ✅ | ✅ | ✅ (1-indexed) | ✅ (1-indexed with page dimensions) |
| **Block Type Labels** | ❌ | ❌ | ✅ (Title, Table, Figure, Key Value, etc.) | ✅ (paragraph roles: title, sectionHeading, header, footer, footnote, pageNumber) |
| **Visual Debugger** | ❌ | ❌ | ✅ (Studio Link) | ✅ (Document Intelligence Studio — label, train, test in browser) |
| **Confidence Per Block** | ❌ | ✅ | ✅ (high/low) | ✅ (0-1 float per word, line, field) |
| **Selection Marks** | ❌ | ❌ | ❌ | ✅ (checkbox/radio detection with selected/unselected state) |

#### Cost & Pricing

| Operation | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence |
|-----------|------------|-----------|---------|----------------------|
| **Pricing Model** | Credit-based (per page) | Per-page | Credit-based (per page) | Per-page (tiered by volume) |
| **Parse (per 1K pages)** | ~$10 (agentic), ~$20 (agentic_plus) | ~$8 | ~$5-10 (varies by features) | $1.50 (Read), $10 (Layout), $15 (prebuilt) |
| **Extract (per 1K pages)** | ~$5 | ~$4 | Included with parse or ~$5 | Included in prebuilt/custom model pricing |
| **Split (per 1K pages)** | Free (local) | ~$3 | ~$3 | Included in Layout model pricing |
| **Classify (per 1K pages)** | ~$1 | Free (inferred) | Free (inferred) | $10 (custom classification model) |
| **Credit Tracking** | ✅ (1-2 credits/page) | ❌ | ✅ (credits reported per call) | ✅ (Azure Cost Management + usage metrics) |
| **Free Tier** | Limited | Limited | Limited | ✅ 500 pages/month free (all prebuilt models) |
| **Agentic Enhancement** | Higher tier = more cost | N/A | Additional cost per enhanced block | N/A (model-based quality tiers) |
| **Enterprise Agreements** | ❌ | ❌ | ❌ | ✅ (Azure EA, CSP, PAYG, reserved capacity) |

#### SDK & Integration

| Aspect | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence |
|--------|------------|-----------|---------|----------------------|
| **Python SDK** | `llama-cloud` | `landingai-ade` | `reductoai` | `azure-ai-documentintelligence` |
| **SDK Style** | Sync | Sync | Sync | Sync + native async (`aio` client) |
| **Async Support** | Wrapped with `to_thread` | Direct (some methods) | Wrapped with `to_thread` | Native async client (`DocumentIntelligenceClient` from `aio`) |
| **Authentication** | `LLAMA_CLOUD_API_KEY` env var | `LANDINGAI_API_KEY` env var | `REDUCTO_API_KEY` env var | Azure AD (Entra ID), API key, or managed identity |
| **File Upload** | Inline (pass bytes) | Inline (pass tuple) | Two-step (upload → process) | Inline (bytes/stream) or URL |
| **Region Support** | Single | US / EU | Single | 30+ Azure regions worldwide |
| **Batch Processing** | ✅ | ✅ | ✅ (multi-URL input) | ✅ (Batch API for large volumes) |
| **Async/Webhook** | ❌ | ❌ | ✅ (webhook callbacks) | ✅ (long-running operations with polling) |
| **IAM / RBAC** | API key only | API key only | API key only | ✅ Azure RBAC, private endpoints, VNet, managed identity |
| **Compliance** | Varies | SOC 2 | Varies | ✅ SOC 1/2/3, ISO 27001, HIPAA BAA, FedRAMP, PCI DSS |
| **On-Premises** | ❌ | ❌ | ❌ | ✅ (disconnected containers for air-gapped environments) |

#### Strengths & Best Use Cases

| Stack | Best For | Key Strengths | Limitations |
|-------|----------|---------------|-------------|
| **LlamaIndex** | General-purpose parsing with tiered quality control | Multiple parsing tiers, native classification, auto-extraction without schema, 50+ file formats (incl. Office, audio), strong ecosystem | Lower classification confidence, no native grounding, higher cost at premium tiers |
| **LandingAI** | Documents needing grounding/audit trails | Full bounding box grounding, region support (US/EU), integrated parse→extract flow, DOCX/PPTX support | No dedicated classification API, limited chunking options |
| **Reducto** | Complex documents needing block-level analysis | Block-type detection, configurable table output, visual debugger (Studio), DOCX/PPTX support, agentic text correction | Two-step upload flow, extract requires file (not text), newer SDK |
| **Azure Doc Intelligence** | Enterprise/regulated environments, high-volume production | ~22 prebuilt doc types, enterprise auth (Entra ID/RBAC), on-premises containers, 300+ language OCR, HIPAA/FedRAMP/PCI compliance, Azure ecosystem integration, 500 free pages/month | Requires Azure subscription, custom models need training data, less flexible schema definition (no arbitrary JSON schema), higher learning curve |

#### Recommendation Matrix

| Document Type | Recommended Stack | Reason |
|--------------|-------------------|--------|
| **Standard PDFs** | Any | All handle well |
| **Scanned/OCR docs** | Azure or Reducto | Azure has best-in-class OCR; Reducto has agentic correction |
| **Forms with checkboxes** | Azure or LandingAI | Azure detects selection marks natively; LandingAI has strong grounding |
| **Financial tables** | Azure or Reducto | Azure has prebuilt invoice/receipt models; Reducto has configurable output |
| **DOCX/PPTX files** | LlamaIndex, Reducto, LandingAI, or Azure | All four support Office formats (LandingAI converts to PDF internally) |
| **Visual documents** | LlamaIndex (agentic_plus) | Best multimodal tier options |
| **Audit/compliance** | Azure | SOC, HIPAA, FedRAMP, PCI DSS compliance; audit logs via Azure Monitor |
| **Budget-conscious** | Azure (free tier) or LlamaIndex (cost_effective) | Azure: 500 free pages/month; LlamaIndex: cheapest paid tier |
| **Handwritten content** | Azure or Gemini | Azure has native handwriting OCR; Gemini excels at math handwriting |
| **Enterprise / regulated** | Azure | RBAC, private endpoints, managed identity, on-premises containers, enterprise agreements |
| **US tax forms (W-2, 1098, 1099)** | Azure | Purpose-built prebuilt models with field-level extraction |
| **Identity documents** | Azure | Prebuilt ID model supports passports, driver's licenses, 150+ countries |
| **Rapid prototyping** | LlamaIndex or Reducto | Simplest API key setup, no cloud subscription needed |
| **Air-gapped / offline** | Azure | Only option with disconnected container deployment |

#### Benchmark Results (Real Document Tests)

The following results were obtained by running all three implemented stacks on actual test documents from `difficult_examples/`.

##### Certificate of Origin (PDF — tables, formal layout)

| Metric | LlamaIndex | LandingAI | Reducto |
|--------|------------|-----------|---------|
| **Markdown length** | 3,985 chars | 4,201 chars | 4,018 chars |
| **Chunks produced** | 3 | 0 | 4 |
| **Classification** | certificate (4%) | other (0%) | certificate (75%) |
| **Fields extracted** | 8 | 0 | 7 |
| **Processing time** | 24.3s | 12.4s | 71.2s |

##### Patient Intake Form (PDF — checkboxes, mixed fields)

| Metric | LlamaIndex | LandingAI | Reducto |
|--------|------------|-----------|---------|
| **Markdown length** | 3,689 chars | 2,565 chars | 4,753 chars |
| **Chunks produced** | 5 | 0 | 4 |
| **Classification** | medical (4%) | other (0%) | form (25%) |
| **Fields extracted** | 0 (schema validation error) | 0 | 4 (incl. nested objects + checkboxes) |
| **Processing time** | 21.8s | 24.5s | 65.9s |

##### Key Observations

| Dimension | LlamaIndex | LandingAI | Reducto |
|-----------|------------|-----------|---------|
| **Classification accuracy** | Correct label but very low confidence (3-5%) | Does not classify unless schema matches | Best accuracy and confidence (25-75%) |
| **Extraction depth** | Good for flat schemas; fails on nested Pydantic types (LlamaExtract API limitation) | Depends on schema match; no extraction without it | Richest — handles nested objects, checkboxes, key-value pairs |
| **Speed** | Mid (~20-35s) | Fastest (~12-25s) | Slowest (~60-70s due to upload + multiple API calls) |
| **Markdown quality** | Clean HTML tables with rowspan/colspan | Longest output with anchor IDs for grounding | Block-assembled; most content for form-type docs |
| **Chunking** | Local rule-based (always produces chunks) | No chunks by default | API-powered section-based chunks |

> **Test environment:** macOS, Python 3.12, documents from `difficult_examples/`. Processing times include network latency and vary by document complexity and API load.

#### Large Document Handling (200+ Pages)

> **Problem:** A 200-300 page document produces ~200K-500K+ tokens of markdown. Most LLM-backed extraction APIs have context limits of 100K-200K tokens. Sending the full markdown to an LLM for schema-based extraction will **fail, truncate, or produce incomplete results**.

| Aspect | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence |
|--------|------------|-----------|---------|----------------------|
| **Parsing (200-300pp)** | ✅ Server-side, no page limit in code | ✅ Server-side, no page limit in code | ✅ Server-side, no page limit in code | ✅ Server-side, up to 2000 pages |
| **Extraction input** | Full markdown text → LLM | Full markdown text → LLM | Uploaded file ref (server-side) | Uploaded file (server-side) |
| **Token limit risk** | 🔴 HIGH — entire markdown sent as `text=` param | 🔴 HIGH — entire markdown sent as `markdown=` param | 🟢 LOW — server processes the file directly | 🟢 LOW — prebuilt models process natively |
| **Chunked extraction** | ❌ Not implemented | ❌ Not implemented | N/A (server handles internally) | N/A (server handles internally) |
| **Max extraction context** | ~100K-200K tokens (API-dependent) | ~100K-200K tokens (API-dependent) | File-based (no token limit) | File-based (no token limit) |
| **Schema derivation** | ⚠️ Truncated to first ~15K chars (~5-10 pages) | ⚠️ Same (shared component) | ⚠️ Same (shared component) | N/A (prebuilt schemas) |
| **Page-level extraction** | ❌ No page attribution | Partial | ✅ Block-level page refs | ✅ Word/line/field page refs |
| **Splitting large docs** | ✅ Local (no API cost) | ✅ Server-side | ✅ Server-side | ✅ Layout model |

##### Extraction Strategies for Large Documents

The core challenge: **you need structured data from a 300-page document, but the LLM can only see ~50-100 pages at once**. Here are the viable strategies, ranked by complexity:

**Strategy 1: File-Based Extraction (Recommended for structured docs)**
- Use APIs that extract directly from the uploaded file, not from text
- **Reducto Extract** and **Azure prebuilt models** both operate on the file server-side — the API handles pagination internally
- Best for: invoices, forms, contracts, tax documents, any document with a known schema
- Limitation: you're dependent on the API's extraction quality; no prompt engineering possible

**Strategy 2: Chunked Extraction (Map-Reduce)**
- Parse document → split into chunks → extract from each chunk independently → merge results
- Each chunk stays within LLM token limits
- Works with any LLM-backed extraction API (LlamaIndex, LandingAI, or direct OpenAI/Anthropic calls)
- Best for: documents where target fields are scattered across pages (e.g., a 300-page contract with clauses on different pages)
- Limitation: fields that span multiple chunks may be missed; merge logic can be complex for nested/list fields
```
Pipeline: Parse → Split(~2K tokens/chunk) → Extract(chunk₁) + Extract(chunk₂) + ... → Merge
```

**Strategy 3: RAG-Based Extraction (Recommended for sparse extraction)**
- Parse document → embed chunks into vector store → for each schema field, retrieve top-K relevant chunks → extract from retrieved context only
- Only sends relevant chunks to the LLM, not the entire document
- Best for: extracting a small number of specific fields from a very large document (e.g., "find the total contract value" from a 300-page agreement)
- Limitation: requires vector store infrastructure; retrieval quality affects extraction accuracy
```
Pipeline: Parse → Embed(chunks) → VectorStore
          For each field: Query(field_description) → Retrieve(top-K) → Extract(context)
```

**Strategy 4: Hierarchical Extraction (Two-Pass)**
- First pass: extract a table of contents / section index from the document
- Second pass: for each section relevant to your schema, extract targeted fields
- Best for: well-structured documents (reports, filings, manuals) where you know fields live in specific sections
- Limitation: requires documents with clear section structure

**Strategy 5: Agent-Driven Extraction (Most flexible, highest cost)**
- An LLM agent iteratively reads pages/sections, decides what to extract, and builds up the result
- The agent can ask follow-up questions like "I found a reference to Amendment 3 on page 45 — let me go read it"
- Best for: complex documents requiring cross-referencing (e.g., legal contracts with amendments, financial filings with footnotes)
- Limitation: highest cost (many LLM calls); slowest; harder to make deterministic

##### When to Use What

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| **Known schema, structured doc** (invoice, form, tax) | File-based (Reducto/Azure) | API handles pagination natively, no token limit |
| **Known schema, unstructured doc** (contract, report) | Chunked extraction (map-reduce) | Fields scattered across pages; each chunk fits in context |
| **Few fields from large doc** (find 5 values in 300 pages) | RAG-based extraction | Only retrieves relevant pages; most token-efficient |
| **Well-structured doc** (manual, filing, spec) | Hierarchical (two-pass) | Section index narrows search space |
| **Complex cross-referencing** (legal, financial) | Agent-driven | Needs reasoning across non-adjacent sections |
| **Unknown schema** (auto-derive fields) | RAG + schema derivation | Sample multiple sections (not just first 15K chars) |

##### Practical Token Budget Estimates

| Document Size | Approx. Markdown Tokens | Fits in GPT-4o (128K)? | Fits in Claude (200K)? | Fits in Gemini 1.5 Pro (1M)? |
|--------------|------------------------|----------------------|----------------------|------------------------------|
| 10 pages | ~5K-15K | ✅ | ✅ | ✅ |
| 50 pages | ~25K-75K | ✅ (tight) | ✅ | ✅ |
| 100 pages | ~50K-150K | ⚠️ May exceed | ✅ (tight) | ✅ |
| 200 pages | ~100K-300K | ❌ Likely exceeds | ⚠️ May exceed | ✅ |
| 300 pages | ~150K-500K | ❌ Exceeds | ❌ Likely exceeds | ✅ (tight) |
| 500+ pages | ~250K-800K+ | ❌ | ❌ | ⚠️ May exceed |

> **Note:** Token counts vary significantly based on document density. A text-heavy academic paper produces ~1K-2K tokens/page, while a form with sparse fields produces ~200-500 tokens/page. Tables and images (as markdown) can be especially token-dense.

##### Recommendations for This Pipeline

For 200-300 page documents in the current pipeline:

1. **Reducto is the safest choice** — extraction operates on the uploaded file, so no token limit applies to the extraction step
2. **For LlamaIndex/LandingAI extraction**, implement chunked extraction: parse → split into ~2K-token chunks → extract per chunk → merge results
3. **Schema derivation** (auto-derive fields) currently only sees the first ~15K characters — for large docs, sample from multiple sections or use a table-of-contents approach
4. **Consider RAG** if you only need a handful of fields from a large document — it's more token-efficient than extracting from every chunk
5. **Gemini 1.5 Pro's 1M token context** can handle most 200-300 page documents in a single call, but at higher cost and latency

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

# Reducto API Key (for Reducto stack)
# Get from: https://reducto.ai
REDUCTO_API_KEY=your_key_here

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
- **Reducto**: Uses Reducto Parse, Extract, Split

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

# Reducto stack
result = await process_document("report.pdf", stack="reducto")

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
    stacks=["llamaindex", "landingai", "reducto"]
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
│   ├── reducto_stack/         # Reducto integration
│   │   ├── client.py          # Reducto SDK client
│   │   ├── parser.py          # Reducto Parse wrapper
│   │   ├── extractor.py       # Reducto Extract wrapper
│   │   ├── splitter.py        # Reducto Split wrapper
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

### ReductoProcessor

```python
from src.reducto_stack import ReductoProcessor

processor = ReductoProcessor(api_key="optional_override")

# Full pipeline
result = await processor.process(
    file_path="document.pdf",
    schema=MySchema,                    # Optional extraction schema
    classification_rules=rules,         # Optional custom rules
    chunk_categories=["intro", "body"], # Optional chunk labels
    include_grounding=True,             # Get bounding boxes per block
)

# Individual steps
markdown = await processor.parse("doc.pdf")
classification = await processor.classify(markdown, rules)
extraction = await processor.extract(markdown, MySchema, file_path="doc.pdf")
chunks = await processor.split(markdown, categories, file_path="doc.pdf")
```

**Key differences from other stacks:**
- Reducto requires a two-step upload→process flow. The `ReductoProcessor.process()` method handles this automatically, uploading once and reusing the reference across parse, extract, and split.
- Extraction operates on the uploaded file (not text), so passing `file_path` to `extract()` yields better results.
- Classification is inferred from block-type frequency analysis (Key Value → form, Table → spreadsheet, etc.).
- Parse output includes block-level metadata (type, bounding box, confidence) in addition to assembled markdown.

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

| Format | LlamaIndex | LandingAI | Reducto | Azure Doc Intelligence | Gemini |
|--------|------------|-----------|---------|----------------------|--------|
| PDF | ✅ | ✅ | ✅ | ✅ | ✅ |
| PNG | ✅ | ✅ | ✅ | ✅ | ✅ |
| JPG/JPEG | ✅ | ✅ | ✅ | ✅ | ✅ |
| GIF | ✅ | ✅ | ✅ | ❌ | ✅ |
| BMP | ✅ | ✅ | ✅ | ✅ | ✅ |
| TIFF | ✅ | ✅ | ✅ | ✅ | ✅ |
| WebP | ✅ | ✅ | ❌ | ❌ | ✅ |
| DOCX | ✅ | ✅ (→ PDF) | ✅ | ✅ | ❌ |
| XLSX | ✅ | ✅ | ✅ | ✅ | ❌ |
| PPTX | ✅ | ✅ (→ PDF) | ✅ | ✅ | ❌ |
| CSV | ✅ | ✅ | ✅ | ❌ | ❌ |
| HTML | ✅ | ❌ | ❌ | ✅ | ❌ |
| Audio (MP3/WAV) | ✅ | ❌ | ❌ | ❌ | ❌ |

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
| Reducto | Parse | ~$5-10 |
| Reducto | Extract | ~$5 |
| Reducto | Split | ~$3 |
| Azure | Read (OCR only) | ~$1.50 |
| Azure | Layout (tables + structure) | ~$10 |
| Azure | Prebuilt (invoice, receipt, etc.) | ~$15 |
| Azure | Custom (trained models) | ~$15 |
| Azure | Custom Classify | ~$10 |
| Gemini Flash | Process | ~$2 |

> **Azure free tier:** 500 pages/month free across all prebuilt models. Enterprise customers can negotiate volume discounts via Azure EA or reserved capacity.

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

Use the Reducto stack for DOCX/PPTX files, or convert to PDF before processing with other stacks.

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
| Processor | Sidebar | LlamaIndex, LandingAI, Reducto |
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
- [Reducto](https://reducto.ai/) - Document parsing, extraction, and splitting
- [Google AI](https://ai.google.dev/) - Gemini multimodal models
