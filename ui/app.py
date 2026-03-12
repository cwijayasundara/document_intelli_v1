"""Streamlit UI for Document Extraction Pipeline Testing.

This app allows users to:
1. Upload documents (PDF, images)
2. Select a processor (LlamaIndex or LandingAI)
3. Choose an operation (Parse, Classify, Extract, Split)
4. For extraction, automatically derive schemas from content
5. View and compare results

Run with:
    streamlit run ui/app.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logger.info("Starting Document Extraction Pipeline UI")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")


# Page config
st.set_page_config(
    page_title="Document Extraction Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }

    /* Dataframe/Table styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame thead th {
        background-color: #f8f9fa !important;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f5f5f5;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6c757d;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        background-color: #f8f9fa;
        border-radius: 8px;
    }

    /* Success/Error boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 15px;
    }

    /* Log box */
    .log-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 12px;
        padding: 12px;
        border-radius: 8px;
        max-height: 200px;
        overflow-y: auto;
    }

    /* Document structure overview */
    .doc-overview {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    /* Table of contents */
    .toc-item {
        padding: 4px 8px;
        border-left: 2px solid #dee2e6;
        margin-left: 8px;
    }
    .toc-item:hover {
        border-left-color: #667eea;
        background-color: #f8f9fa;
    }

    /* Chunk cards */
    .chunk-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        background-color: #ffffff;
    }
    .chunk-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Classification badge */
    .classification-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        background-color: #667eea;
        color: white;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Better button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 4px;
    }

    /* JSON display */
    .stJson {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Run async function in sync context."""
    logger.info("Running async coroutine")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., in Jupyter), use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(coro)
        logger.info("Async coroutine completed successfully")
        return result
    except Exception as e:
        logger.error(f"Async coroutine failed: {str(e)}")
        raise


def check_api_keys():
    """Check which API keys are configured."""
    keys = {
        "LLAMA_CLOUD_API_KEY": bool(os.environ.get("LLAMA_CLOUD_API_KEY")),
        "LANDINGAI_API_KEY": bool(os.environ.get("LANDINGAI_API_KEY")),
        "REDUCTO_API_KEY": bool(os.environ.get("REDUCTO_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
    }
    logger.info(f"API Key status: {keys}")
    return keys


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        logger.info(f"Saved uploaded file to: {tmp.name}")
        return tmp.name


def log_to_ui(message: str, level: str = "info"):
    """Add a log message to the UI log display."""
    if "ui_logs" not in st.session_state:
        st.session_state.ui_logs = []

    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{level.upper()}] {message}"
    st.session_state.ui_logs.append(log_entry)

    # Keep only last 50 logs
    if len(st.session_state.ui_logs) > 50:
        st.session_state.ui_logs = st.session_state.ui_logs[-50:]

    # Also log to console
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)


def get_llama_processor():
    """Get LlamaIndex processor with error handling."""
    log_to_ui("Initializing LlamaIndex processor...")
    try:
        from src.llamaindex_stack import LlamaIndexProcessor
        processor = LlamaIndexProcessor()
        log_to_ui("LlamaIndex processor initialized successfully", "info")
        return processor
    except ImportError as e:
        log_to_ui(f"Failed to import LlamaIndex: {str(e)}", "error")
        raise
    except Exception as e:
        log_to_ui(f"Failed to initialize LlamaIndex processor: {str(e)}", "error")
        raise


def get_landing_processor():
    """Get LandingAI processor with error handling."""
    log_to_ui("Initializing LandingAI processor...")
    try:
        from src.landingai_stack import LandingAIProcessor
        processor = LandingAIProcessor()
        log_to_ui("LandingAI processor initialized successfully", "info")
        return processor
    except ImportError as e:
        error_msg = str(e)
        if "landingai_ade" in error_msg or "landingai-ade" in error_msg:
            log_to_ui("LandingAI SDK not installed. Run: pip install landingai-ade", "error")
            raise ImportError(
                "LandingAI SDK not installed.\n"
                "Please run: pip install landingai-ade"
            )
        log_to_ui(f"Failed to import LandingAI: {error_msg}", "error")
        raise
    except Exception as e:
        log_to_ui(f"Failed to initialize LandingAI processor: {str(e)}", "error")
        raise


def get_reducto_processor():
    """Get Reducto processor with error handling."""
    log_to_ui("Initializing Reducto processor...")
    try:
        from src.reducto_stack import ReductoProcessor
        processor = ReductoProcessor()
        log_to_ui("Reducto processor initialized successfully", "info")
        return processor
    except ImportError as e:
        error_msg = str(e)
        if "reducto" in error_msg.lower():
            log_to_ui("Reducto SDK not installed. Run: pip install reductoai", "error")
            raise ImportError(
                "Reducto SDK not installed.\n"
                "Please run: pip install reductoai"
            )
        log_to_ui(f"Failed to import Reducto: {error_msg}", "error")
        raise
    except Exception as e:
        log_to_ui(f"Failed to initialize Reducto processor: {str(e)}", "error")
        raise


def get_schema_generator():
    """Get schema generator with error handling."""
    log_to_ui("Initializing schema generator...")
    try:
        from src.common.schema_generator import SchemaGenerator
        generator = SchemaGenerator()
        log_to_ui("Schema generator initialized successfully", "info")
        return generator
    except ImportError as e:
        log_to_ui(f"Failed to import SchemaGenerator: {str(e)}", "error")
        raise
    except Exception as e:
        log_to_ui(f"Failed to initialize schema generator: {str(e)}", "error")
        raise


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # API Key Status
        st.subheader("🔑 API Key Status")
        keys = check_api_keys()
        for key, configured in keys.items():
            if configured:
                st.success(f"✓ {key}")
            else:
                st.warning(f"✗ {key}")

        st.divider()

        # Processor Selection
        st.subheader("🔧 Processor")
        processor = st.radio(
            "Select Document Processor",
            ["LlamaIndex", "LandingAI", "Reducto"],
            help="Choose which AI service to use for processing"
        )

        st.divider()

        # Operation Selection
        st.subheader("📋 Operation")
        operation = st.radio(
            "Select Operation",
            ["Parse", "Classify", "Extract", "Split", "Full Pipeline"],
            help="Choose what to do with the document"
        )

        st.divider()

        # Initialize default values
        tier = "agentic"
        multimodal = False
        custom_categories = ""
        schema_mode = "Auto-derive from content"
        manual_fields = ""
        chunk_size = 2000
        overlap = 100
        categories = ""

        # Operation-specific settings
        if operation == "Parse":
            st.subheader("Parse Settings")
            if processor == "LlamaIndex":
                tier = st.selectbox(
                    "Parsing Tier",
                    ["agentic", "agentic_plus", "cost_effective", "fast"],
                    help="Higher tiers are more accurate but cost more"
                )
                multimodal = st.checkbox(
                    "Multimodal Mode",
                    help="Better for charts/diagrams (2x credits)"
                )

        elif operation == "Classify":
            st.subheader("Classification Settings")
            custom_categories = st.text_area(
                "Custom Categories (optional)",
                placeholder="invoice, receipt, form, report",
                help="Comma-separated list of custom categories"
            )

        elif operation == "Extract":
            st.subheader("Extraction Settings")
            schema_mode = st.radio(
                "Schema Mode",
                ["Auto-derive from content", "Manual fields"],
                help="Auto-derive uses AI to detect all extractable fields"
            )
            if schema_mode == "Manual fields":
                manual_fields = st.text_area(
                    "Fields to Extract",
                    placeholder="invoice_number, date, total, vendor_name",
                    help="Comma-separated list of field names"
                )

        elif operation == "Split":
            st.subheader("Split Settings")
            chunk_size = st.slider(
                "Max Chunk Size",
                min_value=500,
                max_value=5000,
                value=2000,
                step=100,
                help="Maximum characters per chunk"
            )
            overlap = st.slider(
                "Overlap",
                min_value=0,
                max_value=500,
                value=100,
                step=50,
                help="Characters to overlap between chunks"
            )
            categories = st.text_area(
                "Chunk Categories (optional)",
                placeholder="header, body, conclusion, references",
                help="Comma-separated categories for chunk classification"
            )

        # Store settings in session state
        st.session_state.settings = {
            "processor": processor,
            "operation": operation,
            "tier": tier,
            "multimodal": multimodal,
            "custom_categories": custom_categories,
            "schema_mode": schema_mode,
            "manual_fields": manual_fields,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_categories": categories,
        }

        return st.session_state.settings


def render_file_upload():
    """Render the file upload section."""
    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
        help="Supported formats: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WebP"
    )

    if uploaded_file:
        log_to_ui(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")

        with col2:
            # Show preview for images
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption="Document Preview", width="stretch")

    return uploaded_file


async def run_parse(file_path: str, settings: dict) -> dict:
    """Run document parsing."""
    log_to_ui(f"Starting parse operation with {settings['processor']} processor")
    log_to_ui(f"File: {file_path}")

    if settings["processor"] == "LlamaIndex":
        log_to_ui(f"Parse tier: {settings['tier']}, Multimodal: {settings['multimodal']}")
        try:
            from src.llamaindex_stack.parser import LlamaParseWrapper, ParseTier
            log_to_ui("LlamaParseWrapper imported successfully")

            parser = LlamaParseWrapper()
            log_to_ui("LlamaParseWrapper initialized")

            tier = ParseTier(settings["tier"])
            result = await parser.parse(
                file_path=Path(file_path),
                tier=tier,
                multimodal=settings["multimodal"]
            )
            log_to_ui(f"Parse complete: {len(result['markdown'])} characters extracted")
            return {
                "markdown": result["markdown"],
                "pages": result.get("metadata", {}).get("pages", 1),
                "processor": "LlamaIndex"
            }
        except Exception as e:
            log_to_ui(f"LlamaIndex parse failed: {str(e)}", "error")
            raise
    elif settings["processor"] == "Reducto":
        try:
            from src.reducto_stack.parser import ReductoParseWrapper
            log_to_ui("ReductoParseWrapper imported successfully")

            parser = ReductoParseWrapper()
            log_to_ui("ReductoParseWrapper initialized")

            result = await parser.parse(file_path=Path(file_path))
            log_to_ui(f"Parse complete: {len(result['markdown'])} characters extracted")

            page_count = result.get("metadata", {}).get("page_count", 1)

            return {
                "markdown": result["markdown"],
                "pages": page_count,
                "processor": "Reducto"
            }
        except ImportError as e:
            error_msg = str(e)
            if "reducto" in error_msg.lower():
                log_to_ui("Reducto SDK not installed. Run: pip install reductoai", "error")
                raise ImportError(
                    "Reducto SDK not installed.\n"
                    "Please run: pip install reductoai"
                )
            log_to_ui(f"Reducto parse failed: {error_msg}", "error")
            raise
        except Exception as e:
            log_to_ui(f"Reducto parse failed: {str(e)}", "error")
            raise
    else:
        try:
            from src.landingai_stack.parser import ADEParseWrapper
            log_to_ui("ADEParseWrapper imported successfully")

            parser = ADEParseWrapper()
            log_to_ui("ADEParseWrapper initialized")

            result = await parser.parse(file_path=Path(file_path))
            log_to_ui(f"Parse complete: {len(result['markdown'])} characters extracted")

            # Get page count - pages might be a list or the count might be in metadata
            pages = result.get("pages", [])
            if isinstance(pages, list):
                page_count = len(pages) if pages else result.get("metadata", {}).get("page_count", 1)
            else:
                page_count = pages if isinstance(pages, int) else 1

            return {
                "markdown": result["markdown"],
                "pages": page_count,
                "processor": "LandingAI"
            }
        except ImportError as e:
            error_msg = str(e)
            if "landingai_ade" in error_msg or "landingai-ade" in error_msg:
                log_to_ui("LandingAI SDK not installed. Run: pip install landingai-ade", "error")
                raise ImportError(
                    "LandingAI SDK not installed.\n"
                    "Please run: pip install landingai-ade"
                )
            log_to_ui(f"LandingAI parse failed: {error_msg}", "error")
            raise
        except Exception as e:
            log_to_ui(f"LandingAI parse failed: {str(e)}", "error")
            raise


async def run_classify(content: str, settings: dict) -> dict:
    """Run document classification."""
    log_to_ui(f"Starting classification with {settings['processor']} processor")
    log_to_ui(f"Content length: {len(content)} characters")

    from src.common.interfaces import ClassificationRule

    # Build rules
    if settings["custom_categories"]:
        categories = [c.strip() for c in settings["custom_categories"].split(",") if c.strip()]
        log_to_ui(f"Using custom categories: {categories}")
        rules = [
            ClassificationRule(
                label=cat,
                description=f"Documents related to {cat}",
                keywords=[cat]
            )
            for cat in categories
        ]
    else:
        rules = None
        log_to_ui("Using default classification rules")

    try:
        from src.llamaindex_stack.classifier import LlamaClassifyWrapper
        classifier = LlamaClassifyWrapper()
        if rules is None:
            rules = classifier.get_default_rules()
        result = await classifier.classify(content=content, rules=rules)
        log_to_ui(f"Classification complete: {result.document_type.value} ({result.confidence:.1%})")

        return {
            "document_type": result.document_type.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "all_scores": result.labels,
            "processor": settings["processor"]
        }
    except Exception as e:
        log_to_ui(f"Classification failed: {str(e)}", "error")
        raise


async def run_extract(content: str, settings: dict, schema_info: dict = None) -> dict:
    """Run document extraction."""
    log_to_ui(f"Starting extraction with {settings['processor']} processor")
    log_to_ui(f"Schema mode: {settings['schema_mode']}")

    from pydantic import create_model, Field
    from typing import Optional

    # Build schema
    if settings["schema_mode"] == "Auto-derive from content" and schema_info:
        log_to_ui("Using auto-derived schema")
        from src.common.schema_generator import SchemaGenerator
        generator = get_schema_generator()
        schema = generator.create_pydantic_model(schema_info["derived_schema"])
        json_schema = generator.schema_to_json_schema(schema_info["derived_schema"])
        log_to_ui(f"Schema has {len(schema_info['derived_schema'].fields)} fields")
    else:
        # Manual fields
        fields = [f.strip() for f in settings["manual_fields"].split(",") if f.strip()]
        if not fields:
            fields = ["content"]
        log_to_ui(f"Using manual fields: {fields}")
        field_definitions = {name: (Optional[str], Field(None, description=f"Extract {name}")) for name in fields}
        schema = create_model("ManualExtraction", **field_definitions)
        json_schema = {
            "type": "object",
            "properties": {name: {"type": "string", "description": f"Extract {name}"} for name in fields}
        }

    try:
        if settings["processor"] == "LlamaIndex":
            from src.llamaindex_stack.extractor import LlamaExtractWrapper
            extractor = LlamaExtractWrapper()
            result = await extractor.extract(content=content, schema=schema)
            log_to_ui(f"Extraction complete: {len(result.fields)} fields extracted")
            return {
                "fields": result.fields,
                "confidence": result.extraction_confidence,
                "processor": "LlamaIndex",
                "schema_used": json_schema
            }
        elif settings["processor"] == "Reducto":
            from src.reducto_stack.extractor import ReductoExtractWrapper
            extractor = ReductoExtractWrapper()
            result = await extractor.extract_with_json_schema(content=content, json_schema=json_schema)
            log_to_ui(f"Extraction complete: {len(result.fields)} fields extracted")
            return {
                "fields": result.fields,
                "confidence": result.extraction_confidence,
                "processor": "Reducto",
                "schema_used": json_schema
            }
        else:
            from src.landingai_stack.extractor import ADEExtractWrapper
            extractor = ADEExtractWrapper()
            result = await extractor.extract_with_json_schema(content=content, json_schema=json_schema)
            log_to_ui(f"Extraction complete: {len(result.fields)} fields extracted")
            return {
                "fields": result.fields,
                "confidence": result.extraction_confidence,
                "processor": "LandingAI",
                "schema_used": json_schema
            }
    except Exception as e:
        log_to_ui(f"Extraction failed: {str(e)}", "error")
        raise


async def run_split(content: str, settings: dict) -> dict:
    """Run document splitting."""
    log_to_ui(f"Starting split with {settings['processor']} processor")
    log_to_ui(f"Chunk size: {settings['chunk_size']}, Overlap: {settings['overlap']}")

    categories = None
    if settings["chunk_categories"]:
        categories = [c.strip() for c in settings["chunk_categories"].split(",") if c.strip()]
        log_to_ui(f"Using categories: {categories}")

    try:
        if settings["processor"] == "LlamaIndex":
            from src.llamaindex_stack.splitter import LlamaSplitWrapper
            splitter = LlamaSplitWrapper()
            chunks = await splitter.split(
                content=content,
                categories=categories,
                max_chunk_size=settings["chunk_size"],
                overlap=settings["overlap"]
            )
        elif settings["processor"] == "Reducto":
            from src.reducto_stack.splitter import ReductoSplitWrapper
            splitter = ReductoSplitWrapper()
            chunks = await splitter.split(
                content=content,
                categories=categories,
            )
        else:
            from src.landingai_stack.splitter import ADESplitWrapper
            splitter = ADESplitWrapper()
            chunks = await splitter.split(
                content=content,
                categories=categories,
                max_chunk_size=settings["chunk_size"]
            )

        log_to_ui(f"Split complete: {len(chunks)} chunks created")

        return {
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                    "full_content": c.content,
                    "category": c.category,
                    "chunk_type": c.chunk_type,
                    "char_count": len(c.content)
                }
                for c in chunks
            ],
            "processor": settings["processor"]
        }
    except Exception as e:
        log_to_ui(f"Split failed: {str(e)}", "error")
        raise


async def run_full_pipeline(file_path: str, settings: dict, schema_info: dict = None) -> dict:
    """Run the full processing pipeline."""
    log_to_ui("Starting full pipeline")
    results = {}

    # Parse
    log_to_ui("Step 1/4: Parsing document...")
    parse_result = await run_parse(file_path, settings)
    results["parse"] = parse_result

    # Classify
    log_to_ui("Step 2/4: Classifying document...")
    classify_result = await run_classify(parse_result["markdown"], settings)
    results["classify"] = classify_result

    # Extract
    log_to_ui("Step 3/4: Extracting fields...")
    extract_result = await run_extract(parse_result["markdown"], settings, schema_info)
    results["extract"] = extract_result

    # Split
    log_to_ui("Step 4/4: Splitting into chunks...")
    split_result = await run_split(parse_result["markdown"], settings)
    results["split"] = split_result

    log_to_ui("Full pipeline complete!")
    return results


def render_results(results: dict, operation: str):
    """Render the processing results using enhanced components."""
    from ui.components import (
        render_parsed_document,
        render_classification_results,
        render_extraction_results,
        render_chunks,
        render_full_pipeline_results,
        parse_markdown_tables,
        extract_document_structure
    )

    st.header("📊 Results")

    if operation == "Parse":
        markdown = results.get("markdown", "")

        # Show processor info
        col1, col2, col3 = st.columns(3)
        with col1:
            # Handle pages being either a number or a list
            pages = results.get("pages", 1)
            if isinstance(pages, list):
                pages = len(pages) if pages else 1
            st.metric("📄 Pages", pages)
        with col2:
            st.metric("📝 Characters", f"{len(markdown):,}")
        with col3:
            st.metric("🔧 Processor", results.get("processor", "N/A"))

        st.divider()

        # Render the parsed document with proper formatting
        render_parsed_document(markdown)

        # Download buttons
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Markdown",
                markdown,
                file_name="parsed_document.md",
                mime="text/markdown",
                width="stretch"
            )
        with col2:
            # Also offer plain text
            st.download_button(
                "📥 Download Plain Text",
                markdown,
                file_name="parsed_document.txt",
                mime="text/plain",
                width="stretch"
            )

    elif operation == "Classify":
        render_classification_results(results)

    elif operation == "Extract":
        render_extraction_results(
            results.get("fields", {}),
            results.get("schema_used")
        )

        # Show schema used
        if results.get("schema_used"):
            with st.expander("📋 Schema Used", expanded=False):
                st.json(results["schema_used"])

    elif operation == "Split":
        render_chunks(results.get("chunks", []))

    elif operation == "Full Pipeline":
        render_full_pipeline_results(results)


def render_schema_derivation(content: str) -> dict:
    """Render schema derivation UI and return derived schema."""
    from src.common.schema_generator import SchemaGenerator, DerivedSchema, FieldDefinition

    st.subheader("🔍 Schema Derivation")
    log_to_ui("Starting schema derivation from document content")

    derived_schema = None

    try:
        generator = get_schema_generator()

        with st.spinner("Analyzing document content to derive extraction schema..."):
            derived_schema = generator.derive_schema(content)

        log_to_ui(f"Schema derived: {len(derived_schema.fields)} fields found")

    except Exception as e:
        log_to_ui(f"Schema derivation failed: {str(e)}", "error")
        st.warning(f"Auto-schema derivation encountered an issue: {str(e)}")
        st.info("Using fallback schema. You can manually add fields below.")

        # Create a fallback schema
        derived_schema = DerivedSchema(
            document_type="unknown",
            fields=[
                FieldDefinition(
                    name="content",
                    field_type="string",
                    description="Full document content",
                    is_required=True,
                    is_list=False
                )
            ],
            reasoning="Fallback schema due to derivation error."
        )

    # Display derived schema
    col1, col2 = st.columns([1, 1])

    with col1:
        if derived_schema.document_type != "unknown":
            st.success(f"**Detected Document Type:** {derived_schema.document_type}")
        else:
            st.warning(f"**Document Type:** {derived_schema.document_type}")
        st.write(f"**Reasoning:** {derived_schema.reasoning}")

    with col2:
        st.write(f"**Fields Found:** {len(derived_schema.fields)}")

    # Show fields in a table
    if derived_schema.fields:
        st.subheader("Derived Fields")
        field_data = []
        for field in derived_schema.fields:
            field_data.append({
                "Field Name": field.name,
                "Type": field.field_type,
                "Required": "Yes" if field.is_required else "No",
                "List": "Yes" if field.is_list else "No",
                "Description": field.description[:50] + "..." if len(field.description) > 50 else field.description
            })

        st.dataframe(field_data, width="stretch")

    # Allow editing
    with st.expander("✏️ Edit Schema (Advanced)", expanded=(derived_schema.document_type == "unknown")):
        st.write("You can modify the schema JSON below:")
        schema_json = json.dumps(derived_schema.model_dump(), indent=2)
        edited_json = st.text_area("Schema JSON", schema_json, height=300, key="schema_editor")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Edits", key="apply_schema"):
                try:
                    derived_schema = DerivedSchema(**json.loads(edited_json))
                    st.success("Schema updated!")
                    log_to_ui("Schema manually edited")
                    st.session_state.derived_schema = derived_schema
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    log_to_ui(f"Schema edit failed: {e}", "error")

        with col2:
            # Quick add field
            st.write("**Quick Add Field:**")

    # Quick add field form
    with st.expander("➕ Add Field Manually"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_field_name = st.text_input("Field Name", placeholder="field_name", key="new_field_name")
        with col2:
            new_field_type = st.selectbox("Type", ["string", "number", "date", "boolean", "list", "object"], key="new_field_type")
        with col3:
            new_field_required = st.checkbox("Required", key="new_field_required")

        new_field_desc = st.text_input("Description", placeholder="What this field represents", key="new_field_desc")

        if st.button("Add Field", key="add_field_btn"):
            if new_field_name:
                new_field = FieldDefinition(
                    name=new_field_name.lower().replace(" ", "_"),
                    field_type=new_field_type,
                    description=new_field_desc or f"Extract {new_field_name}",
                    is_required=new_field_required,
                    is_list=False
                )
                derived_schema.fields.append(new_field)
                st.success(f"Added field: {new_field_name}")
                log_to_ui(f"Manually added field: {new_field_name}")
            else:
                st.warning("Please enter a field name")

    return {"derived_schema": derived_schema}


def render_logs():
    """Render the log display section."""
    if "ui_logs" in st.session_state and st.session_state.ui_logs:
        with st.expander("📋 Processing Logs", expanded=False):
            log_html = "<div class='log-box'>"
            for log in st.session_state.ui_logs[-20:]:  # Show last 20 logs
                if "[ERROR]" in log:
                    log_html += f"<span style='color: #f44336;'>{log}</span><br>"
                elif "[WARNING]" in log:
                    log_html += f"<span style='color: #ff9800;'>{log}</span><br>"
                else:
                    log_html += f"<span>{log}</span><br>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

            if st.button("Clear Logs"):
                st.session_state.ui_logs = []
                st.rerun()


def main():
    """Main application entry point."""
    st.title("📄 Document Extraction Pipeline")
    st.write("Test and compare document processing capabilities across different AI platforms.")

    # Initialize logs
    if "ui_logs" not in st.session_state:
        st.session_state.ui_logs = []

    log_to_ui("Application started")

    # Render sidebar and get settings
    settings = render_sidebar()

    # Main content area
    uploaded_file = render_file_upload()

    if uploaded_file is None:
        st.info("👆 Upload a document to get started.")
        render_logs()
        return

    # Save file temporarily
    file_path = save_uploaded_file(uploaded_file)

    # Schema derivation for extraction
    schema_info = None
    parsed_content_for_schema = None

    if settings["operation"] in ["Extract", "Full Pipeline"] and settings.get("schema_mode") == "Auto-derive from content":
        # First parse to get content for schema derivation
        log_to_ui("Parsing document for schema derivation")
        with st.spinner("Parsing document for schema derivation..."):
            try:
                parse_result = run_async(run_parse(file_path, settings))
                parsed_content_for_schema = parse_result["markdown"]
                log_to_ui(f"Document parsed: {len(parsed_content_for_schema)} characters")
            except Exception as e:
                st.error(f"Failed to parse document: {str(e)}")
                log_to_ui(f"Parse for schema derivation failed: {str(e)}", "error")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                render_logs()
                return

        # Derive schema (this now handles errors gracefully)
        if parsed_content_for_schema:
            schema_info = render_schema_derivation(parsed_content_for_schema)

    # Process button
    st.divider()
    if st.button("🚀 Process Document", type="primary", width="stretch"):
        try:
            operation = settings["operation"]
            log_to_ui(f"Processing document with operation: {operation}")

            if operation == "Parse":
                with st.spinner("Parsing document..."):
                    results = run_async(run_parse(file_path, settings))
            elif operation == "Classify":
                with st.spinner("Processing..."):
                    # First parse
                    log_to_ui("Parsing document before classification")
                    parse_result = run_async(run_parse(file_path, settings))
                    results = run_async(run_classify(parse_result["markdown"], settings))
            elif operation == "Extract":
                with st.spinner("Processing..."):
                    # First parse if not already done
                    if schema_info:
                        content = run_async(run_parse(file_path, settings))["markdown"]
                    else:
                        log_to_ui("Parsing document before extraction")
                        content = run_async(run_parse(file_path, settings))["markdown"]
                    results = run_async(run_extract(content, settings, schema_info))
            elif operation == "Split":
                with st.spinner("Processing..."):
                    log_to_ui("Parsing document before splitting")
                    parse_result = run_async(run_parse(file_path, settings))
                    results = run_async(run_split(parse_result["markdown"], settings))
            elif operation == "Full Pipeline":
                with st.spinner("Running full pipeline..."):
                    results = run_async(run_full_pipeline(file_path, settings, schema_info))

            # Store results in session state
            st.session_state.results = results
            st.session_state.operation = operation
            log_to_ui("Processing complete!")

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            log_to_ui(f"Processing failed: {str(e)}", "error")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    # Display results if available
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.divider()
        render_results(st.session_state.results, st.session_state.operation)

    # Always show logs at the bottom
    render_logs()

    # Cleanup temp file
    try:
        os.unlink(file_path)
    except:
        pass


if __name__ == "__main__":
    main()
