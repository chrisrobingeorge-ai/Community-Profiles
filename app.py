import io
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_mapping,
    load_data,
    list_communities,
    get_record,
    gather_column_status,
)
from utils.narrative import compose_context, render_markdown, markdown_to_docx


# ----------------------
# Streamlit page config
# ----------------------
st.set_page_config(
    page_title="Community Profiles (StatCan)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Community Profiles (Statistics Canada)")

# ----------------------
# Load mapping + data
# ----------------------
@st.cache_data(show_spinner=True)
def _load_mapping_cached():
    return load_mapping(Path("metadata/fields.yaml"))

@st.cache_data(show_spinner=True)
def _load_data_cached():
    return load_data(Path("data/raw"))
with st.sidebar:
    st.subheader("Upload CSVs (optional)")
    uploads = st.file_uploader(
        "Drop StatCan CSV files here",
        type=["csv", "csv.gz", "csv.zip"],
        accept_multiple_files=True
    )
    if uploads:
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        for f in uploads:
            # Keep original file name if present
            out_path = Path("data/raw") / (f.name or "uploaded.csv")
            with open(out_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to data/raw/.")
        st.caption("Refresh to load them.")

mapping = _load_mapping_cached()
df = _load_data_cached()

if df.empty:
    st.warning(
        "No data was found in `data/raw/`. Please place your Statistics Canada CSV file(s) there and refresh."
    )
    st.stop()

# ----------------------
# Sidebar: controls
# ----------------------
with st.sidebar:
    st.header("Controls")
    name_col = mapping.get("identity", {}).get("name_col", "Geographic name")
    geoid_col = mapping.get("identity", {}).get("geoid_col", "GeoUID")

    communities = list_communities(df, geoid_col, name_col)
    community = st.selectbox("Select a community", communities, index=0, help="Choose by community name")
    st.caption(f"Using name column: **{name_col}** · ID column: **{geoid_col}**")

    with st.expander("Advanced"):
        geoid_input = st.text_input("Search by GeoUID (optional)", placeholder="e.g., 4806016")
        language = st.selectbox("Language", options=["en"], index=0, help="French template can be added later")

    run_button = st.button("Generate narrative", type="primary")

# Resolve selection
if geoid_input:
    # If GeoUID provided, prefer that record
    row = get_record(df, geoid=geoid_input, geoid_col=geoid_col, name_col=name_col)
else:
    row = get_record(df, name=community, geoid_col=geoid_col, name_col=name_col)

if row is None:
    st.error("Could not find a matching record. Check your `identity` columns in `metadata/fields.yaml` and your CSV.")
    st.stop()

# ----------------------
# Data Health Check
# ----------------------
with st.sidebar:
    st.subheader("Data Health Check")
    status_df, coverage = gather_column_status(df, mapping)
    st.metric("Section coverage", f"{coverage:.0f}%")
    st.dataframe(status_df, use_container_width=True, height=350)

# ----------------------
# Generate narrative
# ----------------------
if run_button:
    with st.spinner("Composing narrative..."):
        context = compose_context(row, mapping)
        md = render_markdown(context, lang=language)

    st.success("Narrative generated.")
    st.markdown(md)

    # Downloads
    md_bytes = md.encode("utf-8")
    st.download_button(
        "⬇️ Download Markdown",
        data=md_bytes,
        file_name=f"{context['geoid']}_{context['community_name']}.md",
        mime="text/markdown",
    )

    docx_bytes = markdown_to_docx(md)
    st.download_button(
        "⬇️ Download Word (.docx)",
        data=docx_bytes,
        file_name=f"{context['geoid']}_{context['community_name']}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption(
    "Data source: Statistics Canada. This application is provided under the Statistics Canada Open Licence. "
    "Narratives are generated from supplied CSVs; verify mappings in `metadata/fields.yaml`."
)
