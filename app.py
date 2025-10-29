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

    # Try UTF-8 first. If that fails, fall back to latin1 (common for StatCan downloads)
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)  # rewind file handle before retry
        df = pd.read_csv(uploaded_file, encoding="latin1")

    # normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # normalize key text columns if present
    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).str.strip()
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

def _coerce_number(val):
    """
    Try to turn a cell like '123', '12.3', '12.3 %', '0', '..', etc.
    into a float. Return None if it's not usable.
    """
    if pd.isna(val):
        return None
    text = str(val).strip()
    if text in ["", "..", "...", "F", "X"]:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        num = float(text)
        return num
    except ValueError:
        return None


def generate_summary(df: pd.DataFrame) -> str:
    """
    Build a short narrative about the community using the filtered dataframe.
    We skip empty/zero/insignificant values.
    """

    if df.empty:
        return "No summary available."

    # figure out which geography columns exist (anything that's not Topic/Characteristic)
    value_cols = [c for c in df.columns if c not in ("Topic", "Characteristic")]
    if not value_cols:
        return "No summary available."
    geo_col = value_cols[0]  # assume first geography column is the community

    lines = []

    # Population
    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = None
    if not pop_rows.empty:
        pop_val_raw = pop_rows.iloc[0][geo_col]
        pop_val_num = _coerce_number(pop_val_raw)
        if pop_val_num:
            lines.append(
                f"The community has a population of about {int(round(pop_val_num, 0))} people."
            )

    # Children / youth
    kids_rows = df[
        (df["Topic"].str.contains("Age characteristics", case=False, na=False)) &
        (df["Characteristic"].str.contains("0 to 14 years", case=False, na=False))
    ]
    if not kids_rows.empty:
        kids_val_raw = kids_rows.iloc[0][geo_col]
        kids_val_num = _coerce_number(kids_val_raw)

        if kids_val_num and kids_val_num > 0:
            pct_kids = None
            if pop_val_num and pop_val_num > 0 and kids_val_num < pop_val_num:
                pct_kids = (kids_val_num / pop_val_num) * 100.0

            if pct_kids and pct_kids >= 1.0:
                lines.append(
                    f"Children (0–14) make up roughly {pct_kids:.1f}% of residents, "
                    "which highlights the importance of youth and family programming."
                )
            else:
                lines.append(
                    "There is a noticeable population of children (ages 0–14), "
                    "indicating demand for child and youth services."
                )

    # Indigenous population / ancestry
    indig_rows = df[
        df["Topic"].str.contains("Indigenous population|Indigenous ancestry", case=False, na=False)
    ]
    indig_val_num = None
    if not indig_rows.empty:
        for _, r in indig_rows.iterrows():
            cand = _coerce_number(r[geo_col])
            if cand and cand > 0:
                indig_val_num = cand
                break
    if indig_val_num:
        lines.append(
            "The community includes an Indigenous population that should be considered "
            "in program design and partnership."
        )

    # Immigration / newcomers
    imm_rows = df[
        df["Topic"].str.contains("Immigrant status and period of immigration", case=False, na=False)
        | df["Topic"].str.contains("Selected places of birth for the recent immigrant population",
                                   case=False, na=False)
    ]
    imm_val_num = None
    if not imm_rows.empty:
        for _, r in imm_rows.iterrows():
            cand = _coerce_number(r[geo_col])
            if cand and cand > 0:
                imm_val_num = cand
                break
    if imm_val_num:
        lines.append(
            "There is a visible newcomer / recent immigrant presence, which suggests a need "
            "for culturally responsive outreach and language-aware communication with parents."
        )

    # Language profile
    lang_rows = df[
        df["Topic"].str.contains(
            "Mother tongue|Language spoken most often at home|Knowledge of official languages|First official language spoken",
            case=False,
            na=False,
        )
    ]

    english_only_flag = False
    french_presence_flag = False
    other_lang_flag = False

    for _, r in lang_rows.iterrows():
        char_lower = r["Characteristic"].lower()
        val_num = _coerce_number(r[geo_col])
        if not val_num or val_num <= 0:
            continue

        if "english only" in char_lower:
            english_only_flag = True
        if "french" in char_lower and "only" in char_lower:
            french_presence_flag = True
        if "neither english nor french" in char_lower:
            other_lang_flag = True
        if "most often at home" in char_lower and "english" not in char_lower and "french" not in char_lower:
            other_lang_flag = True

    lang_bits = []
    if english_only_flag:
        lang_bits.append("English is dominant.")
    if french_presence_flag:
        lang_bits.append("French is present in daily life.")
    if other_lang_flag:
        lang_bits.append("Multiple other home languages are actively spoken.")
    if lang_bits:
        lines.append("Language profile: " + " ".join(lang_bits))

    # Low income / inequality
    low_income_rows = df[
        df["Topic"].str.contains("Low income and income inequality", case=False, na=False)
    ]
    low_income_flag = False
    if not low_income_rows.empty:
        for _, r in low_income_rows.iterrows():
            v = _coerce_number(r[geo_col])
            if v and v > 0:
                low_income_flag = True
                break
    if low_income_flag:
        lines.append(
            "There are indicators of financial vulnerability (low income / income inequality), "
            "which may limit access to paid activities without subsidy."
        )

    # Mobility (are people new / moving around)
    mobility_rows = df[
        df["Topic"].str.contains("Mobility status 1 year ago|Mobility status 5 years ago",
                                 case=False, na=False)
    ]
    mobility_flag = False
    if not mobility_rows.empty:
        for _, r in mobility_rows.iterrows():
            row_char = r["Characteristic"].lower()
            mv = _coerce_number(r[geo_col])
            if (("moved" in row_char) or ("different" in row_char)) and mv and mv > 0:
                mobility_flag = True
                break
    if mobility_flag:
        lines.append(
            "The community shows recent in-migration and residential movement, "
            "which can mean families are still forming local connections."
        )

    # Commuting patterns
    commute_rows = df[
        df["Topic"].str.contains("Main mode of commuting|Commuting duration", case=False, na=False)
    ]
    long_commute_flag = False
    car_commute_flag = False
    for _, r in commute_rows.iterrows():
        row_char = r["Characteristic"].lower()
        v = _coerce_number(r[geo_col])
        if not v or v <= 0:
            continue
        if "60 minutes" in row_char or "longer" in row_char:
            long_commute_flag = True
        if "car" in row_char or "automobile" in row_char or "driver" in row_char:
            car_commute_flag = True

    commute_bits = []
    if car_commute_flag:
        commute_bits.append("Most adults rely on driving.")
    if long_commute_flag:
        commute_bits.append("Some workers face long daily commutes.")
    if commute_bits:
        lines.append("Commuting pattern: " + " ".join(commute_bits))

    # Final
    if not lines:
        return (
            "This community profile does not surface notable demographic features "
            "from the selected census fields."
        )

    return " ".join(lines)

# ------------------------------------------------
# UI
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Statistics Canada Census Profile CSV",
    type=["csv"],
    help="Use the 'Download CSV' option from a community's Census Profile table.",
)

if uploaded_file is None:
    st.info("Upload a CSV to generate the community profile.")
else:
    raw_df = load_statcan_csv(uploaded_file)

    st.subheader("Raw Preview (first 20 rows)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    cleaned_df = filter_relevant_rows(raw_df)

    # NEW: Summary
    st.subheader("Community Profile Summary")
    summary_text = generate_summary(cleaned_df)
    st.write(summary_text)

    # Existing table view
    st.subheader("Filtered Report")
    render_report(cleaned_df)

    # Existing export section
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
