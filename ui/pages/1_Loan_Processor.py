"""Loan Processing Pipeline - Streamlit UI Page.

This page provides an interface for processing loan application documents:
1. Upload multiple loan documents (ID, W2, pay stubs, bank statements, investment statements)
2. Automatically categorize each document
3. Extract relevant fields based on document type
4. Validate consistency across documents
5. Visualize extracted fields with bounding boxes

Run with:
    streamlit run ui/app.py
    (Navigate to "Loan Processor" page)
"""

import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Loan Document Processor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Document card styling */
    .doc-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .doc-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Status badges */
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }

    /* Document type badge */
    .doc-type-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .doc-type-id { background-color: #e3f2fd; color: #1565c0; }
    .doc-type-w2 { background-color: #f3e5f5; color: #7b1fa2; }
    .doc-type-pay_stub { background-color: #e8f5e9; color: #2e7d32; }
    .doc-type-bank_statement { background-color: #fff8e1; color: #f57f17; }
    .doc-type-investment_statement { background-color: #fce4ec; color: #c2185b; }
    .doc-type-unknown { background-color: #f5f5f5; color: #616161; }

    /* Validation panel */
    .validation-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .validation-passed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .validation-failed {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }

    /* Asset summary cards */
    .asset-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .asset-value {
        font-size: 2em;
        font-weight: 700;
        color: #1a237e;
    }
    .asset-label {
        font-size: 0.9em;
        color: #6c757d;
        margin-top: 8px;
    }

    /* Field table */
    .field-table {
        width: 100%;
        border-collapse: collapse;
    }
    .field-table th, .field-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    .field-table th {
        background-color: #f8f9fa;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def check_api_key(processor: str = "landingai"):
    """Check if the API key for the selected processor is configured."""
    if processor == "reducto":
        return bool(os.environ.get("REDUCTO_API_KEY"))
    return bool(os.environ.get("LANDINGAI_API_KEY"))


def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temp directory, preserving original filenames.

    This preserves original filenames so that the parse cache uses
    meaningful names (e.g., 'bank_statement.md' instead of 'tmp123.md').
    """
    # Create a temp directory for this session
    temp_dir = Path(tempfile.mkdtemp(prefix="loan_docs_"))

    paths = []
    for uploaded_file in uploaded_files:
        # Use the original filename
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getvalue())
        paths.append(str(file_path))

    return paths


def get_doc_type_badge_class(doc_type: str) -> str:
    """Get CSS class for document type badge."""
    return f"doc-type-{doc_type.lower().replace(' ', '_')}"


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Processor selection
        st.subheader("🔧 Processor")
        processor = st.radio(
            "Select Processor",
            ["LandingAI", "Reducto"],
            index=0,
            help="Choose which document AI processor to use"
        )
        processor_key = processor.lower()

        # API Key Status
        st.subheader("🔑 API Key Status")
        if processor_key == "reducto":
            if check_api_key("reducto"):
                st.success("✓ REDUCTO_API_KEY configured")
            else:
                st.error("✗ REDUCTO_API_KEY not set")
                st.info("Set the REDUCTO_API_KEY environment variable to use Reducto.")
        else:
            if check_api_key("landingai"):
                st.success("✓ LANDINGAI_API_KEY configured")
            else:
                st.error("✗ LANDINGAI_API_KEY not set")
                st.info("Set the LANDINGAI_API_KEY environment variable to use LandingAI.")

        st.divider()

        # Processing options
        st.subheader("📋 Processing Options")

        run_validation = st.checkbox(
            "Run Cross-Document Validation",
            value=True,
            help="Validate name consistency, years, and calculate asset totals"
        )

        require_id = st.checkbox(
            "Require ID Document",
            value=True,
            help="Mark validation as failed if no ID is provided"
        )

        require_income = st.checkbox(
            "Require Income Proof",
            value=True,
            help="Mark validation as failed if no W2 or pay stub is provided"
        )

        st.divider()

        # Cache options
        st.subheader("💾 Parse Cache")

        use_cache = st.checkbox(
            "Use Parse Cache",
            value=True,
            help="Cache parsed documents to avoid redundant API calls"
        )

        force_reparse = st.checkbox(
            "Force Re-parse",
            value=False,
            help="Ignore cache and re-parse all documents",
            disabled=not use_cache
        )

        # Show cache info
        try:
            from src.pipelines.loan_processing.pipeline import LoanProcessingPipeline
            pipeline = LoanProcessingPipeline(use_cache=True)
            cached_files = pipeline.list_cached_documents()
            if cached_files:
                st.caption(f"📁 {len(cached_files)} cached document(s)")
                if st.button("🗑️ Clear Cache", key="clear_cache"):
                    pipeline.clear_cache()
                    st.success("Cache cleared!")
                    st.rerun()
        except Exception:
            pass

        st.divider()

        # Year validation
        st.subheader("📅 Year Validation")
        current_year = datetime.now().year

        min_year = st.number_input(
            "Minimum Acceptable Year",
            min_value=2000,
            max_value=current_year,
            value=current_year - 2,
            help="Documents older than this year will be flagged"
        )

        max_year = st.number_input(
            "Maximum Acceptable Year",
            min_value=2000,
            max_value=current_year + 1,
            value=current_year,
            help="Documents from future years will be flagged"
        )

        st.divider()

        # Help section
        st.subheader("📚 Supported Documents")
        st.markdown("""
        - **ID**: Driver's license, passport, state ID
        - **W2**: W-2 tax form
        - **Pay Stub**: Paycheck stub, earnings statement
        - **Bank Statement**: Checking/savings statement
        - **Investment Statement**: Brokerage, 401k, IRA
        """)

        return {
            "run_validation": run_validation,
            "require_id": require_id,
            "require_income": require_income,
            "min_year": min_year,
            "max_year": max_year,
            "use_cache": use_cache,
            "force_reparse": force_reparse if use_cache else True,
            "processor": processor_key
        }


def render_file_upload():
    """Render the file upload section."""
    st.header("📤 Upload Loan Documents")

    # Initialize accumulated files in session state
    if "accumulated_files" not in st.session_state:
        st.session_state.accumulated_files = {}

    st.info(
        "Upload documents for the loan application. "
        "You can upload multiple files at once (Ctrl/Cmd+click) or one at a time - they will accumulate. "
        "Supported formats: PDF, PNG, JPG, JPEG"
    )

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select multiple files at once, or upload one at a time - files accumulate until cleared",
        key="loan_doc_uploader"
    )

    # Accumulate newly uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Store by name to avoid duplicates
            st.session_state.accumulated_files[uploaded_file.name] = uploaded_file

    # Get all accumulated files
    all_files = list(st.session_state.accumulated_files.values())

    if all_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✓ {len(all_files)} document(s) ready for processing")
        with col2:
            if st.button("🗑️ Clear All", key="clear_files"):
                st.session_state.accumulated_files = {}
                st.rerun()

        # Show file previews
        st.markdown("**Uploaded Documents:**")
        cols = st.columns(min(len(all_files), 5))
        for i, uploaded_file in enumerate(all_files):
            with cols[i % 5]:
                st.markdown(f"📄 **{uploaded_file.name}**")
                st.caption(f"{uploaded_file.size / 1024:.1f} KB")
                # Add individual remove button
                if st.button("✕", key=f"remove_{uploaded_file.name}", help=f"Remove {uploaded_file.name}"):
                    del st.session_state.accumulated_files[uploaded_file.name]
                    st.rerun()

    return all_files


def render_document_results(doc):
    """Render results for a single processed document."""
    from src.pipelines.loan_processing.schemas import LoanDocumentType

    # Document type badge
    doc_type_class = get_doc_type_badge_class(doc.document_type.value)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### 📄 {doc.file_name}")
        st.markdown(
            f'<span class="doc-type-badge {doc_type_class}">{doc.document_type.value}</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Confidence: {doc.classification_confidence:.0%}")

    with col2:
        if doc.error:
            st.error(f"❌ Error: {doc.error}")
        else:
            status_msg = "✓ Processed successfully"
            if doc.from_cache:
                status_msg += " (cached)"
            st.success(status_msg)
        st.caption(f"Time: {doc.processing_time_ms:.0f}ms")

    # Show classification reasoning
    if doc.classification_reasoning:
        with st.expander("🔍 Classification Reasoning"):
            st.write(doc.classification_reasoning)

    # Show extracted fields
    if doc.extraction and doc.extraction.fields:
        st.markdown("#### Extracted Fields")

        # Convert to dataframe for nice display
        fields_df = pd.DataFrame([
            {"Field": k, "Value": str(v)}
            for k, v in doc.extraction.fields.items()
            if v is not None
        ])

        if not fields_df.empty:
            st.dataframe(fields_df, use_container_width=True, hide_index=True)
        else:
            st.info("No fields extracted")
    else:
        st.info("No fields extracted")

    # Show raw markdown (collapsed)
    if doc.markdown:
        with st.expander("📝 Raw Markdown"):
            st.markdown(doc.markdown[:2000] + "..." if len(doc.markdown) > 2000 else doc.markdown)


def render_validation_results(validation):
    """Render validation results panel."""
    if validation.validation_passed:
        panel_class = "validation-passed"
        icon = "✅"
        title = "Validation Passed"
    else:
        panel_class = "validation-failed"
        icon = "❌"
        title = "Validation Failed"

    st.markdown(f"""
    <div class="validation-panel {panel_class}">
        <h2>{icon} {title}</h2>
        <p>{len(validation.issues)} issue(s) found</p>
    </div>
    """, unsafe_allow_html=True)

    # Asset summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="asset-card">
            <div class="asset-value">${validation.total_bank_balance:,.2f}</div>
            <div class="asset-label">Bank Balance</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="asset-card">
            <div class="asset-value">${validation.total_investment_value:,.2f}</div>
            <div class="asset-label">Investments</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="asset-card">
            <div class="asset-value">${validation.total_assets:,.2f}</div>
            <div class="asset-label">Total Assets</div>
        </div>
        """, unsafe_allow_html=True)

    # Income if available
    if validation.annual_income:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Annual Income", f"${validation.annual_income:,.2f}")
        with col2:
            st.metric("💵 Monthly Income", f"${validation.monthly_income:,.2f}")

    # Validation details
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Names Found")
        if validation.names_found:
            for name in validation.names_found:
                st.write(f"• {name}")
            if validation.name_match:
                st.success("✓ Names match across documents")
            else:
                st.warning("⚠️ Name mismatch detected")
        else:
            st.info("No names extracted")

    with col2:
        st.markdown("##### 📅 Years Found")
        if validation.years_found:
            for year in validation.years_found:
                st.write(f"• {year}")
            if validation.years_valid:
                st.success("✓ All years are valid")
            else:
                st.warning("⚠️ Year validation issues")
        else:
            st.info("No years extracted")

    # Issues
    if validation.issues:
        st.markdown("---")
        st.markdown("##### ⚠️ Issues")
        for issue in validation.issues:
            st.warning(issue)

    # Document types found
    st.markdown("---")
    st.markdown("##### 📋 Document Types Processed")
    doc_types_str = ", ".join(validation.document_types_found)
    st.write(doc_types_str)


def render_results(result):
    """Render the complete processing results."""
    from src.pipelines.loan_processing.schemas import LoanDocumentType

    st.header("📊 Processing Results")

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📄 Total Documents", result.total_documents)
    with col2:
        st.metric("✅ Successful", result.successful_documents)
    with col3:
        st.metric("💾 From Cache", result.documents_from_cache)
    with col4:
        st.metric("❌ Failed", result.failed_documents)
    with col5:
        st.metric("⏱️ Processing Time", f"{result.total_processing_time_ms:.0f}ms")

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Documents", "✓ Validation", "📥 Export"])

    with tab1:
        st.subheader("Processed Documents")
        for doc in result.documents:
            with st.container():
                st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                render_document_results(doc)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if result.validation:
            st.subheader("Cross-Document Validation")
            render_validation_results(result.validation)
        else:
            st.info("Validation was not run")

    with tab3:
        st.subheader("Export Results")

        # JSON export
        result_dict = result.model_dump(mode="json")
        result_json = json.dumps(result_dict, indent=2, default=str)

        st.download_button(
            "📥 Download Full Results (JSON)",
            result_json,
            file_name="loan_processing_results.json",
            mime="application/json",
            use_container_width=True
        )

        # Summary CSV export
        if result.documents:
            summary_data = []
            for doc in result.documents:
                row = {
                    "File": doc.file_name,
                    "Type": doc.document_type.value,
                    "Confidence": f"{doc.classification_confidence:.0%}",
                    "Status": "Error" if doc.error else "Success",
                    "Fields Extracted": len(doc.extraction.fields) if doc.extraction else 0
                }
                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)

            st.download_button(
                "📥 Download Summary (CSV)",
                csv,
                file_name="loan_processing_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Validation summary
        if result.validation:
            validation_summary = {
                "Validation Passed": result.validation.validation_passed,
                "Names Match": result.validation.name_match,
                "Years Valid": result.validation.years_valid,
                "Total Bank Balance": f"${result.validation.total_bank_balance:,.2f}",
                "Total Investments": f"${result.validation.total_investment_value:,.2f}",
                "Total Assets": f"${result.validation.total_assets:,.2f}",
                "Annual Income": f"${result.validation.annual_income:,.2f}" if result.validation.annual_income else "N/A",
                "Issues": "; ".join(result.validation.issues) if result.validation.issues else "None"
            }

            validation_df = pd.DataFrame([validation_summary])
            validation_csv = validation_df.to_csv(index=False)

            st.download_button(
                "📥 Download Validation Summary (CSV)",
                validation_csv,
                file_name="loan_validation_summary.csv",
                mime="text/csv",
                use_container_width=True
            )


async def process_documents(file_paths: List[str], settings: dict):
    """Process documents through the loan pipeline."""
    from src.pipelines.loan_processing.pipeline import LoanProcessingPipeline
    from src.pipelines.loan_processing.validator import LoanValidator

    # Create pipeline with settings
    pipeline = LoanProcessingPipeline(
        min_year=settings["min_year"],
        max_year=settings["max_year"],
        use_cache=settings.get("use_cache", True),
        processor=settings.get("processor", "landingai")
    )

    # Override validator settings
    pipeline.validator = LoanValidator(
        min_year=settings["min_year"],
        max_year=settings["max_year"],
        require_id=settings["require_id"],
        require_income_proof=settings["require_income"]
    )

    # Process the application
    result = await pipeline.process_application(
        [Path(p) for p in file_paths],
        validate=settings["run_validation"],
        force_reparse=settings.get("force_reparse", False)
    )

    return result


def main():
    """Main application entry point."""
    st.title("💰 Loan Document Processor")
    st.write("Process and validate loan application documents using AI-powered extraction.")

    # Check for required SDK
    try:
        from src.pipelines.loan_processing import LoanProcessingPipeline
    except ImportError as e:
        st.error("Failed to import loan processing pipeline. Make sure all dependencies are installed.")
        st.code(str(e))
        return

    # Render sidebar and get settings
    settings = render_sidebar()

    # Check API key for selected processor
    selected_processor = settings.get("processor", "landingai")
    if not check_api_key(selected_processor):
        key_name = "REDUCTO_API_KEY" if selected_processor == "reducto" else "LANDINGAI_API_KEY"
        st.warning(
            f"⚠️ {key_name} not configured. "
            f"Please set this environment variable to use the loan processor with {selected_processor}."
        )

    # Main content area
    uploaded_files = render_file_upload()

    if not uploaded_files:
        st.info("👆 Upload loan documents to get started.")

        # Show example workflow
        with st.expander("📖 How It Works", expanded=True):
            st.markdown("""
            ### Loan Document Processing Pipeline

            1. **Upload Documents**: Upload all documents for the loan application
            2. **Automatic Categorization**: Each document is classified (ID, W2, pay stub, etc.)
            3. **Field Extraction**: Relevant fields are extracted based on document type
            4. **Cross-Document Validation**:
               - Name consistency check
               - Year validation
               - Asset totals calculation
            5. **Review & Export**: Review results and export to JSON/CSV

            ### Supported Document Types

            | Type | Examples |
            |------|----------|
            | ID | Driver's license, passport, state ID |
            | W2 | W-2 tax form |
            | Pay Stub | Paycheck stub, earnings statement |
            | Bank Statement | Checking/savings account statement |
            | Investment Statement | Brokerage, 401k, IRA statement |
            """)
        return

    # Save uploaded files
    file_paths = save_uploaded_files(uploaded_files)

    # Process button
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        process_button = st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True
        )
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("loan_results", None)
            st.rerun()

    if process_button:
        if not check_api_key(selected_processor):
            key_name = "REDUCTO_API_KEY" if selected_processor == "reducto" else "LANDINGAI_API_KEY"
            st.error(f"Cannot process: {key_name} not configured")
            return

        with st.spinner(f"Processing {len(file_paths)} documents..."):
            try:
                result = run_async(process_documents(file_paths, settings))
                st.session_state.loan_results = result
                st.success("✓ Processing complete!")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                logger.exception("Processing failed")

    # Display results if available
    if "loan_results" in st.session_state:
        st.divider()
        render_results(st.session_state.loan_results)

    # Cleanup temp directory (files are in a temp dir with original names)
    if file_paths:
        try:
            temp_dir = Path(file_paths[0]).parent
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    main()
