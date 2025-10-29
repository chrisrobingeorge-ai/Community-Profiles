import io
import pandas as pd
import streamlit as st

# ------------------------------------------------
# Streamlit page config
# ------------------------------------------------
st.set_page_config(
    page_title="Community Profile Extractor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Community Profile Extractor (Statistics Canada 2021 Census)")
st.caption(
    "Upload a Census Profile CSV from Statistics Canada. "
    "This tool keeps only the fields we care about for community planning."
)

# ------------------------------------------------
# Selected sections / topics you care about
# ------------------------------------------------
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

# Sometimes these aren't in the Topic column, but appear in Characteristic instead,
# so we also match using keywords inside the Characteristic column:
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

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def load_statcan_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """
    Read the uploaded Census Profile CSV.
    We assume columns like:
    - Topic
    - Characteristic
    - One or more geography columns (Cypress, Taber MD, etc.)
    """

    df = pd.read_csv(uploaded_file)

    # normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # normalize key text columns if present
    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).strip()
    if "Characteristic" in df.columns:
        df["Characteristic"] = df["Characteristic"].astype(str).str.strip()

    # drop totally empty rows (sometimes StatCan dumps blank separators)
    df = df.dropna(how="all")

    # drop rows that are literally repeated headers
    if "Topic" in df.columns:
        df = df[df["Topic"] != "Topic"]

    return df


def filter_relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep any row where:
    - Topic is exactly one of the selected sections
      (case-insensitive match)
    OR
    - Characteristic contains one of our keyword phrases.
    """

    if "Topic" not in df.columns or "Characteristic" not in df.columns:
        st.error(
            "This file doesn't look like the standard Census Profile format. "
            "It must include columns named 'Topic' and 'Characteristic'."
        )
        return pd.DataFrame()

    # match on Topic (case-insensitive equality)
    topic_mask = df["Topic"].str.lower().isin([t.lower() for t in TARGET_TOPICS])

    # match on Characteristic (case-insensitive substring)
    char_mask = False
    for kw in TARGET_CHARACTERISTIC_KEYWORDS:
        char_mask = char_mask | df["Characteristic"].str.lower().str.contains(
            kw.lower(), na=False
        )

    keep_mask = topic_mask | char_mask
    filtered = df[keep_mask].copy()

    # sort nicely for display
    filtered.sort_values(
        by=["Topic", "Characteristic"],
        inplace=True,
        ignore_index=True,
    )

    return filtered


def render_report(df: pd.DataFrame):
    """
    Show results in collapsible sections by Topic.
    """
    if df.empty:
        st.warning("No matching rows found in this CSV for the selected fields.")
        return

    value_cols = [c for c in df.columns if c not in ("Topic", "Characteristic")]

    for topic, sub in df.groupby("Topic", dropna=False):
        with st.expander(f"📂 {topic}", expanded=True):
            pretty = sub[["Characteristic"] + value_cols].reset_index(drop=True)
            st.dataframe(pretty, use_container_width=True)


def build_filtered_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_printable_html(df: pd.DataFrame) -> str:
    """
    Build a simple HTML report the user can 'Save as PDF' in their browser.
    Streamlit will just download it as .html.
    """
    if df.empty:
        return "<html><body><h1>No data</h1></body></html>"

    styles = """
    <style>
    body { font-family: sans-serif; margin: 2rem; }
    h1 { font-size: 1.4rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #999; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    th, td {
        border: 1px solid #ccc;
        padding: 0.4rem;
        font-size: 0.9rem;
        text-align: left;
    }
    th { background: #f5f5f5; }
    </style>
    """

    parts = [
        "<html><head><meta charset='UTF-8'>",
        styles,
        "</head><body>",
        "<h1>Community Profile Extract</h1>",
        "<p>Filtered fields aligned to community planning / Growing Up Strong.</p>",
    ]

    value_cols = [c for c in df.columns if c not in ("Topic", "Characteristic")]

    for topic, sub in df.groupby("Topic", dropna=False):
        parts.append(f"<h2>{topic}</h2>")
        tmp = sub[["Characteristic"] + value_cols].reset_index(drop=True)
        parts.append(tmp.to_html(index=False, escape=False))

    parts.append("</body></html>")
    return "\n".join(parts)


# ------------------------------------------------
# UI
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Statistics Canada Census Profile CSV",
    type=["csv"],
    help="Use the 'Download CSV' option from a community's Census Profile table.",
)

if uploaded_file is None:
    st.info("Upload a CSV to generate the filtered community profile.")
else:
    # load raw
    raw_df = load_statcan_csv(uploaded_file)

    st.subheader("Raw Preview (first 20 rows)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # filter
    cleaned_df = filter_relevant_rows(raw_df)

    # show report
    st.subheader("Filtered Report")
    render_report(cleaned_df)

    # export buttons
    st.subheader("Export")
    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = build_filtered_csv(cleaned_df)
        st.download_button(
            label="⬇️ Download filtered CSV",
            data=csv_bytes,
            file_name="community_profile_filtered.csv",
            mime="text/csv",
        )

    with col2:
        html_str = build_printable_html(cleaned_df)
        html_bytes = html_str.encode("utf-8")
        st.download_button(
            label="⬇️ Download printable report (HTML)",
            data=html_bytes,
            file_name="community_profile_report.html",
            mime="text/html",
        )

st.markdown(
    """
**Note:** You can open the downloaded HTML file in any browser and use 'Print' → 'Save as PDF'.
"""
)