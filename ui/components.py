"""UI Components for Document Extraction Pipeline.

This module provides reusable components for rendering parsed documents,
including proper table rendering (both HTML and Markdown), structured data display, and more.
"""

import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from io import StringIO


def parse_html_tables(html_content: str) -> List[Tuple[str, pd.DataFrame]]:
    """Extract HTML tables and convert to DataFrames.

    Args:
        html_content: Content containing HTML tables.

    Returns:
        List of tuples (table_title, DataFrame) for each table found.
    """
    tables = []

    try:
        # Use pandas to read HTML tables
        dfs = pd.read_html(StringIO(html_content))
        for i, df in enumerate(dfs):
            # Clean up the dataframe
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # Replace NaN with empty string for display
            df = df.fillna('')

            # Convert all to string for consistent display
            df = df.astype(str)

            # Remove columns that are just empty strings
            df = df.loc[:, (df != '').any(axis=0)]

            # Remove rows that are just empty strings
            df = df.loc[(df != '').any(axis=1)]

            if not df.empty:
                tables.append((f"Table {i+1}", df))
    except Exception as e:
        # No valid HTML tables found
        pass

    return tables


def parse_markdown_tables(markdown: str) -> List[Tuple[str, pd.DataFrame]]:
    """Extract markdown tables and convert to DataFrames.

    Args:
        markdown: The markdown content containing tables.

    Returns:
        List of tuples (table_title, DataFrame) for each table found.
    """
    tables = []

    # Pattern to match markdown tables
    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n?)+)'

    matches = re.finditer(table_pattern, markdown)

    for i, match in enumerate(matches):
        table_text = match.group(1)
        lines = table_text.strip().split('\n')

        if len(lines) < 3:
            continue

        # Parse header
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]

        # Parse data rows
        data_rows = []
        for line in lines[2:]:
            if line.strip():
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                while len(cells) < len(headers):
                    cells.append('')
                cells = cells[:len(headers)]
                data_rows.append(cells)

        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
            tables.append((f"Table {i+1}", df))

    return tables


def extract_all_tables(content: str) -> List[Tuple[str, pd.DataFrame]]:
    """Extract all tables from content (both HTML and Markdown).

    Args:
        content: Document content that may contain HTML or markdown tables.

    Returns:
        List of tuples (table_title, DataFrame) for each table found.
    """
    tables = []

    # First try HTML tables
    html_tables = parse_html_tables(content)
    tables.extend(html_tables)

    # Then try markdown tables (if no HTML tables found)
    if not tables:
        md_tables = parse_markdown_tables(content)
        tables.extend(md_tables)

    return tables


def split_content_by_tables(content: str) -> List[Dict[str, Any]]:
    """Split content into segments, separating tables from text.

    Args:
        content: The document content.

    Returns:
        List of dicts with 'type' ('text', 'html_table', or 'md_table') and 'content'.
    """
    segments = []

    # Check for HTML tables first
    html_table_pattern = r'(<table[^>]*>[\s\S]*?</table>)'

    last_end = 0
    for match in re.finditer(html_table_pattern, content, re.IGNORECASE):
        # Add text before table
        if match.start() > last_end:
            text = content[last_end:match.start()].strip()
            if text:
                segments.append({'type': 'text', 'content': text})

        # Parse HTML table to DataFrame
        table_html = match.group(1)
        try:
            dfs = pd.read_html(StringIO(table_html))
            if dfs:
                df = dfs[0]
                df = df.dropna(how='all').dropna(axis=1, how='all')
                df = df.fillna('')
                df = df.astype(str)
                df = df.loc[:, (df != '').any(axis=0)]
                df = df.loc[(df != '').any(axis=1)]
                if not df.empty:
                    segments.append({'type': 'html_table', 'content': df, 'raw_html': table_html})
        except:
            # If parsing fails, treat as text
            segments.append({'type': 'text', 'content': table_html})

        last_end = match.end()

    # Add remaining text
    if last_end < len(content):
        text = content[last_end:].strip()
        if text:
            # Check for markdown tables in remaining text
            md_segments = split_markdown_by_tables(text)
            segments.extend(md_segments)

    # If no HTML tables found, try markdown tables
    if not segments:
        segments = split_markdown_by_tables(content)

    return segments


def split_markdown_by_tables(markdown: str) -> List[Dict[str, Any]]:
    """Split markdown into segments, separating tables from text.

    Args:
        markdown: The markdown content.

    Returns:
        List of dicts with 'type' ('text' or 'md_table') and 'content'.
    """
    segments = []

    table_pattern = r'(\|[^\n]+\|\n\|[-:\| ]+\|\n(?:\|[^\n]+\|\n?)+)'

    last_end = 0
    for match in re.finditer(table_pattern, markdown):
        # Add text before table
        if match.start() > last_end:
            text = markdown[last_end:match.start()].strip()
            if text:
                segments.append({'type': 'text', 'content': text})

        # Parse table
        table_text = match.group(1)
        lines = table_text.strip().split('\n')

        if len(lines) >= 3:
            header_line = lines[0]
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]

            data_rows = []
            for line in lines[2:]:
                if line.strip():
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    while len(cells) < len(headers):
                        cells.append('')
                    cells = cells[:len(headers)]
                    data_rows.append(cells)

            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                segments.append({'type': 'md_table', 'content': df})

        last_end = match.end()

    # Add remaining text
    if last_end < len(markdown):
        text = markdown[last_end:].strip()
        if text:
            segments.append({'type': 'text', 'content': text})

    # If no tables found, return whole content as text
    if not segments and markdown.strip():
        segments.append({'type': 'text', 'content': markdown.strip()})

    return segments


def extract_document_structure(content: str) -> Dict[str, Any]:
    """Extract document structure from content.

    Args:
        content: The document content (HTML or markdown).

    Returns:
        Dict with document structure information.
    """
    structure = {
        'headings': [],
        'table_count': 0,
        'image_count': 0,
        'list_count': 0,
        'code_block_count': 0,
        'word_count': 0,
        'char_count': len(content),
    }

    # Count HTML tables
    html_tables = len(re.findall(r'<table[^>]*>', content, re.IGNORECASE))

    # Count markdown tables
    md_tables = len(re.findall(r'\|[^\n]+\|\n\|[-:\| ]+\|', content))

    structure['table_count'] = html_tables + md_tables

    # Count headings (both HTML and markdown)
    # HTML headings
    for match in re.finditer(r'<h([1-6])[^>]*>([^<]+)</h\1>', content, re.IGNORECASE):
        level = int(match.group(1))
        text = match.group(2).strip()
        structure['headings'].append({'level': level, 'text': text})

    # Markdown headings
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        structure['headings'].append({'level': level, 'text': text})

    # Count images
    html_images = len(re.findall(r'<img[^>]+>', content, re.IGNORECASE))
    md_images = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content))
    structure['image_count'] = html_images + md_images

    # Count lists
    html_lists = len(re.findall(r'<[ou]l[^>]*>', content, re.IGNORECASE))
    md_lists = len(re.findall(r'^[\s]*[-*+]\s|^[\s]*\d+\.\s', content, re.MULTILINE))
    structure['list_count'] = html_lists + md_lists

    # Count code blocks
    structure['code_block_count'] = len(re.findall(r'```[\s\S]*?```|<code[^>]*>[\s\S]*?</code>', content))

    # Word count (strip HTML tags first)
    text_only = re.sub(r'<[^>]+>', ' ', content)
    words = re.findall(r'\b\w+\b', text_only)
    structure['word_count'] = len(words)

    return structure


def render_document_structure(structure: Dict[str, Any]):
    """Render document structure overview in Streamlit.

    Args:
        structure: Document structure from extract_document_structure().
    """
    st.subheader("📊 Document Overview")

    cols = st.columns(5)
    with cols[0]:
        st.metric("📝 Words", f"{structure['word_count']:,}")
    with cols[1]:
        st.metric("📋 Tables", structure['table_count'])
    with cols[2]:
        st.metric("🖼️ Images", structure['image_count'])
    with cols[3]:
        st.metric("📑 Lists", structure['list_count'])
    with cols[4]:
        st.metric("📄 Characters", f"{structure['char_count']:,}")

    # Show table of contents if there are headings
    if structure['headings']:
        with st.expander("📚 Table of Contents", expanded=False):
            for heading in structure['headings']:
                indent = "  " * (heading['level'] - 1)
                st.markdown(f"{indent}• {heading['text']}")


def render_parsed_document(content: str, show_raw: bool = False):
    """Render parsed document with proper formatting.

    Args:
        content: The parsed document content (HTML or markdown).
        show_raw: Whether to show raw content view.
    """
    if not content:
        st.warning("No content to display")
        return

    # Extract and show document structure
    structure = extract_document_structure(content)
    render_document_structure(structure)

    st.divider()

    # Create tabs for different views
    tabs = st.tabs(["📄 Structured View", "📊 Tables Only", "📝 Raw Content"])

    with tabs[0]:
        render_formatted_content(content)

    with tabs[1]:
        render_tables_only(content)

    with tabs[2]:
        st.code(content, language="html" if "<table" in content.lower() else "markdown")


def render_formatted_content(content: str):
    """Render content with tables as proper dataframes.

    Args:
        content: The document content to render.
    """
    segments = split_content_by_tables(content)

    if not segments:
        st.markdown(content)
        return

    for i, segment in enumerate(segments):
        if segment['type'] == 'text':
            text = segment['content']
            # Clean up HTML tags for display, but keep structure
            # Convert common HTML to markdown-friendly format
            text = re.sub(r'<br\s*/?>', '\n', text)
            text = re.sub(r'</?p[^>]*>', '\n\n', text)
            text = re.sub(r'<strong>([^<]+)</strong>', r'**\1**', text)
            text = re.sub(r'<b>([^<]+)</b>', r'**\1**', text)
            text = re.sub(r'<em>([^<]+)</em>', r'*\1*', text)
            text = re.sub(r'<i>([^<]+)</i>', r'*\1*', text)
            text = re.sub(r'<h1[^>]*>([^<]+)</h1>', r'# \1', text)
            text = re.sub(r'<h2[^>]*>([^<]+)</h2>', r'## \1', text)
            text = re.sub(r'<h3[^>]*>([^<]+)</h3>', r'### \1', text)
            text = re.sub(r'<sup>([^<]+)</sup>', r'^(\1)', text)
            text = re.sub(r'<[^>]+>', '', text)  # Remove remaining HTML tags
            text = text.strip()
            if text:
                st.markdown(text)

        elif segment['type'] in ['html_table', 'md_table']:
            df = segment['content']
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                height=min(400, (len(df) + 1) * 35 + 10)  # Dynamic height based on rows
            )


def render_tables_only(content: str):
    """Render only the tables from the document.

    Args:
        content: The document content.
    """
    tables = extract_all_tables(content)

    if not tables:
        st.info("No tables found in this document")
        return

    st.success(f"Found **{len(tables)}** table(s) in the document")

    for i, (title, df) in enumerate(tables):
        st.subheader(f"📊 {title}")

        # Show table info
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                height=min(400, (len(df) + 1) * 35 + 10)
            )

        with col2:
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))

        with col3:
            # Download buttons
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 CSV",
                csv,
                file_name=f"table_{i+1}.csv",
                mime="text/csv",
                key=f"download_csv_{i}"
            )

            # Excel download
            try:
                from io import BytesIO
                buffer = BytesIO()
                df.to_excel(buffer, index=False, engine='openpyxl')
                st.download_button(
                    "📥 Excel",
                    buffer.getvalue(),
                    file_name=f"table_{i+1}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{i}"
                )
            except:
                pass  # openpyxl not installed

        st.divider()


def render_extraction_results(fields: Dict[str, Any], schema: Dict[str, Any] = None):
    """Render extraction results in a user-friendly format.

    Args:
        fields: The extracted fields.
        schema: Optional schema information.
    """
    if not fields:
        st.warning("No fields extracted")
        return

    st.subheader("📋 Extracted Information")

    # Group fields by type for better display
    simple_fields = {}
    complex_fields = {}
    list_fields = {}

    for key, value in fields.items():
        if value is None:
            continue
        elif isinstance(value, list):
            list_fields[key] = value
        elif isinstance(value, dict):
            complex_fields[key] = value
        else:
            simple_fields[key] = value

    # Render simple fields as a clean table
    if simple_fields:
        st.write("**📝 Basic Fields**")

        # Create a nice display
        field_data = []
        for key, value in simple_fields.items():
            field_data.append({
                "Field": key.replace("_", " ").title(),
                "Value": str(value)
            })

        df = pd.DataFrame(field_data)
        st.dataframe(df, width="stretch", hide_index=True)

    # Render list fields
    if list_fields:
        st.write("**📋 List Fields**")
        for key, values in list_fields.items():
            with st.expander(f"{key.replace('_', ' ').title()} ({len(values)} items)"):
                if values and isinstance(values[0], dict):
                    # List of objects - show as table
                    df = pd.DataFrame(values)
                    st.dataframe(df, width="stretch", hide_index=True)
                else:
                    # Simple list
                    for i, v in enumerate(values, 1):
                        st.write(f"{i}. {v}")

    # Render complex fields
    if complex_fields:
        st.write("**🔗 Nested Fields**")
        for key, value in complex_fields.items():
            with st.expander(key.replace("_", " ").title()):
                st.json(value)


def render_classification_results(classification: Dict[str, Any]):
    """Render classification results with visual indicators.

    Args:
        classification: Classification result dict.
    """
    st.subheader("🏷️ Document Classification")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        doc_type = classification.get("document_type", "unknown")
        # Add emoji based on type
        type_emojis = {
            "invoice": "🧾",
            "receipt": "🧾",
            "form": "📝",
            "certificate": "📜",
            "medical": "🏥",
            "report": "📊",
            "presentation": "📽️",
            "diagram": "📐",
            "flowchart": "🔀",
            "spreadsheet": "📈",
            "instructions": "📋",
            "infographic": "🎨",
            "handwritten": "✍️",
        }
        emoji = type_emojis.get(doc_type, "📄")
        st.markdown(f"### {emoji} {doc_type.replace('_', ' ').title()}")

    with col2:
        confidence = classification.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.1%}")

    with col3:
        processor = classification.get("processor", "N/A")
        st.metric("Processor", processor)

    # Show reasoning
    if classification.get("reasoning"):
        with st.expander("💭 Classification Reasoning"):
            st.write(classification["reasoning"])

    # Show all scores as a bar chart
    if classification.get("all_scores"):
        st.write("**Category Scores**")
        scores = classification["all_scores"]

        # Convert to dataframe for chart
        score_data = pd.DataFrame([
            {"Category": k.replace("_", " ").title(), "Score": float(v)}
            for k, v in scores.items()
        ]).sort_values("Score", ascending=False)

        st.bar_chart(score_data.set_index("Category"))


def render_chunks(chunks: List[Dict[str, Any]]):
    """Render document chunks with metadata.

    Args:
        chunks: List of chunk dictionaries.
    """
    st.subheader("✂️ Document Chunks")

    if not chunks:
        st.warning("No chunks generated")
        return

    # Summary
    total_chars = sum(c.get("char_count", len(c.get("content", ""))) for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Total Characters", f"{total_chars:,}")
    with col3:
        st.metric("Avg Chunk Size", f"{avg_chars:,}")

    st.divider()

    # Chunk navigation
    view_mode = st.radio(
        "View Mode",
        ["Cards", "Table", "Full Text"],
        horizontal=True
    )

    if view_mode == "Cards":
        # Show as expandable cards
        for i, chunk in enumerate(chunks):
            category = chunk.get("category", "uncategorized")
            char_count = chunk.get("char_count", len(chunk.get("content", "")))

            with st.expander(
                f"**Chunk {i+1}** | {category} | {char_count:,} chars",
                expanded=(i == 0)  # Expand first chunk
            ):
                content = chunk.get("full_content") or chunk.get("content", "")
                st.markdown(content)

    elif view_mode == "Table":
        # Show as table with preview
        table_data = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            table_data.append({
                "Chunk": i + 1,
                "Category": chunk.get("category", "N/A"),
                "Characters": chunk.get("char_count", len(content)),
                "Preview": preview
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch", hide_index=True)

    else:  # Full Text
        # Show all chunks concatenated
        st.write("All chunks combined:")
        full_text = "\n\n---\n\n".join(
            f"**[Chunk {i+1} - {c.get('category', 'N/A')}]**\n\n{c.get('full_content') or c.get('content', '')}"
            for i, c in enumerate(chunks)
        )
        st.markdown(full_text)


def render_full_pipeline_results(results: Dict[str, Any]):
    """Render complete pipeline results with organized tabs.

    Args:
        results: Full pipeline results dict.
    """
    st.header("📊 Processing Results")

    # Quick summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        chars = len(results.get("parse", {}).get("markdown", ""))
        st.metric("📝 Characters", f"{chars:,}")

    with col2:
        doc_type = results.get("classify", {}).get("document_type", "N/A")
        st.metric("🏷️ Type", doc_type.title())

    with col3:
        fields = len(results.get("extract", {}).get("fields", {}))
        st.metric("📋 Fields", fields)

    with col4:
        chunks = results.get("split", {}).get("total_chunks", 0)
        st.metric("✂️ Chunks", chunks)

    st.divider()

    # Detailed tabs
    tabs = st.tabs([
        "📄 Parsed Document",
        "🏷️ Classification",
        "📋 Extraction",
        "✂️ Chunks"
    ])

    with tabs[0]:
        markdown = results.get("parse", {}).get("markdown", "")
        if markdown:
            render_parsed_document(markdown)
        else:
            st.warning("No parsed content available")

    with tabs[1]:
        classify = results.get("classify", {})
        if classify:
            render_classification_results(classify)
        else:
            st.warning("No classification results available")

    with tabs[2]:
        extract = results.get("extract", {})
        if extract:
            render_extraction_results(
                extract.get("fields", {}),
                extract.get("schema_used")
            )

            # Download button
            if extract.get("fields"):
                import json
                st.download_button(
                    "📥 Download Extracted Data (JSON)",
                    json.dumps(extract["fields"], indent=2),
                    file_name="extracted_data.json",
                    mime="application/json"
                )
        else:
            st.warning("No extraction results available")

    with tabs[3]:
        split = results.get("split", {})
        if split:
            render_chunks(split.get("chunks", []))
        else:
            st.warning("No chunk results available")
