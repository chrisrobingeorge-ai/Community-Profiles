import io
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Community Profile Extractor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Community Profile Extractor (Statistics Canada 2021 Census)")
st.caption(
    "Upload a Census Profile CSV (Statistics Canada). "
    "We'll pull only the fields relevant to community planning for Growing Up Strong."
)

# -------------------------------------------------------------------
# 1. Controlled list of sections / fields we care about
# -------------------------------------------------------------------

# These are high-level Topic names in the Census Profile, where possible.
TARGET_TOPICS = [
    "Population and dwellings",
    "Age characteristics",
    "Household and dwelling characteristics",
    "Household type",
    "Income of households in 2020",
    "Low income and income inequality in 2020",
    "Knowledge of official languages",
    "First official language spoken",
    "Mother tongue",
    "Language spoken most often at home",
    "Immigrant status and period of immigration",
    "Selected places of birth for the immigrant population",
    "Selected places of birth for the recent immigrant population",
    "Indigenous population",
    "Indigenous ancestry",
    "Visible minority",
    "Secondary (high) school diploma or equivalency certificate",
    "Highest certificate, diploma or degree",
    "Mobility status 1 year ago",
    "Mobility status 5 years ago",
    "Main mode of commuting",
    "Commuting duration",
    "Children eligible for instruction in the minority official language",
    "Eligibility and instruction in the minority official language for school-aged children",
]

# Some of those (like "Secondary (high) school diploma...") might not
# actually appear in the Topic column. Sometimes they're Characteristics
# under a Topic like "Education". To be safe, we'll also match by
# substring in the Characteristic column.
TARGET_CHARACTERISTIC_KEYWORDS = [
    "Secondary (high) school diploma or equivalency certificate",
    "Highest certificate, diploma or degree",
    "Mobility status 1 year ago",
    "Mobility status 5 years ago",
    "Main mode of commuting",
    "Commuting duration",
    "Children eligible for instruction in the minority official language",
    "Eligibility and instruction in the minority official language for school-aged children",
]

# -------------------------------------------------------------------
# 2. Helper: clean uploaded CSV into a usable DataFrame
# -------------------------------------------------------------------
def load_statcan_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """
    Attempt to read the uploaded Census Profile CSV into a tidy DataFrame.
    We assume:
    - There's a 'Topic' column
    - There's a 'Characteristic' column
    - Then one or more columns for geographies (e.g. Cypress County, etc.)

    We'll also drop any fully-empty columns/rows.
    """
    df = pd.read_csv(uploaded_file)

    # Strip whitespace from headers and cells to normalize matching
    df.columns = [str(c).strip() for c in df.columns]
    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).str.strip()
    if "Characteristic" in df.columns:
        df["Characteristic"] = df["Characteristic"].astype(str).str.strip()

    # Remove rows that are completely empty
    df = df.dropna(how="all")
    # Remove duplicate header rows (StatCan sometimes repeats headers in the body)
    df = df[~(df["Topic"] == "Topic")]

    return df


# -------------------------------------------------------------------
# 3. Helper: filter rows we care about
# -------------------------------------------------------------------
def filter_relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    We keep rows if:
    - 'Topic' matches any of TARGET_TOPICS (case-insensitive),
      OR
    - 'Characteristic' contains any keyword in TARGET_CHARACTERISTIC_KEYWORDS
      (also case-insensitive).

    Then we return only those rows/columns.
    """
    if "Topic" not in df.columns or "Characteristic" not in df.columns:
        st.error(
            "CSV doesn't look like a standard Census Profile export. "
            "It must include 'Topic' and 'Characteristic' columns."
        )
        return pd.DataFrame()

    # Case-insensitive match on Topic
    topic_mask = df["Topic"].str.lower().isin([t.lower() for t in TARGET_TOPICS])

    # Substring match in Characteristic (case-insensitive)
    char_mask = False
    for kw in TARGET_CHARACTERISTIC_KEYWORDS:
        char_mask = char_mask | df["Characteristic"].str.lower().str.contains(kw.lower(), na=False)

    # Combine
    keep_mask = topic_mask | char_mask
    filtered = df[keep_mask].copy()

    # Optional: tidy up formatting a bit for display
    # We'll group by Topic so the report feels "sectioned"
    filtered.sort_values(by=["Topic", "Characteristic"], inplace=True, ignore_index=True)

    return filtered


# -------------------------------------------------------------------
# 4. Helper: render a friendly "report view"
# -------------------------------------------------------------------
def render_report(df: pd.DataFrame):
    """
    Display the filtered DataFrame in a readable way:
    - Show as an expander per Topic, with all related rows.
    """
    if df.empty:
        st.warning("No matching rows were found for the selected fields.")
        return

    # Identify numeric columns (geography columns etc.) to keep them visible
    non_meta_cols = [c for c in df.columns if c not in ("Topic", "Characteristic")]

    # For each Topic, show a subtable
    for topic, sub in df.groupby("Topic", dropna=False):
        with st.expander(f"📂 {topic}", expanded=True):
            # Just show Characteristic + value columns
            pretty = sub[["Characteristic"] + non_meta_cols].reset_index(drop=True)
            st.dataframe(pretty, use_container_width=True)


# -------------------------------------------------------------------
# 5. Helper: build downloadable CSV / PDF-ish export
# -------------------------------------------------------------------
def build_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_html_report(df: pd.DataFrame) -> str:
    """
    We'll generate a minimal HTML string so the user can save-as-PDF
    from their browser. True PDF export would require an extra library.
    """
    if df.empty:
        return "<html><body><h1>No data</h1></body></html>"

    html_parts = [
        "<html><head><meta charset='UTF-8'>",
        "<style>",
        "body { font-family: sans-serif; margin: 2rem; }",
        "h1 { font-size: 1.4rem; margin-bottom: 0.5rem; }",
        "h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #999; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }",
        "th, td { border: 1px solid #ccc; padding: 0.4rem; font-size: 0.9rem; text-align: left; }",
        "th { background: #f5f5f5; }",
        "</style></head><body>",
        "<h1>Community Profile Extract</h1>",
        "<p>Filtered fields aligned to Growing Up Strong community planning.</p>",
    ]

    non_meta_cols = [c for c in df.columns if c not in ("Topic", "Characteristic")]

    for topic, sub in df.groupby("Topic", dropna=False):
        html_parts.append(f"<h2>{topic}</h2>")
        # Build a small HTML table for this topic
        tmp = sub[["Characteristic"] + non_meta_cols].reset_index(drop=True)
        html_parts.append(tmp.to_html(index=False, escape=False))

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


# -------------------------------------------------------------------
# 6. Streamlit UI flow
# -------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Statistics Canada Census Profile CSV",
    type=["csv"],
    help="Use the 'Download CSV' option from a Census Profile table on the Statistics Canada site."
)

if uploaded_file is not None:
    # Load
    raw_df = load_statcan_csv(uploaded_file)

    st.subheader("Raw Preview (first 20 rows)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # Filter
    cleaned_df = filter_relevant_rows(raw_df)

    st.subheader("Filtered Report")
    render_report(cleaned_df)

    # Downloads
    st.subheader("Export")
    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = build_csv_bytes(cleaned_df)
        st.download_button(
            label="⬇️ Download filtered CSV",
            data=csv_bytes,
            file_name="community_profile_filtered.csv",
            mime="text/csv",
        )

    with col2:
        # We'll give them an .html file that they can open in a browser and print to PDF.
        html_str = build_html_report(cleaned_df)
        html_bytes = html_str.encode("utf-8")
        st.download_button(
            label="⬇️ Download printable report (HTML)",
            data=html_bytes,
            file_name="community_profile_report.html",
            mime="text/html",
        )

else:
    st.info(
        "No file uploaded yet. Upload a CSV to generate the community profile report."
    )


# -------------------------------------------------------------------
# 7. (Optional) True direct-to-PDF note
# -------------------------------------------------------------------
st.markdown(
    """
**About PDF export:**  
The "Printable report (HTML)" download is meant for you to open in your browser and print/save as PDF.  
If you want true 1-click PDF creation inside the app, you can add a library like `weasyprint` or `reportlab`
to convert that HTML to PDF on the server — but that adds dependencies and sometimes OS packages.
"""
)