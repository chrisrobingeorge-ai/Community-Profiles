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

def pick_geo_col(df: pd.DataFrame) -> str | None:
    """
    Try to guess which column in df is the actual geography data column
    (e.g. 'Cypress County, Alberta') instead of metadata like 'Notes'.

    Strategy:
    1. Exclude known non-data columns.
    2. Among remaining columns, pick the one that has the highest count
       of numeric-looking values.
    """
    exclude = {"Topic", "Characteristic", "Notes", "Note", "Symbol", "Flags", "Flag"}
    candidates = [c for c in df.columns if c not in exclude]

    best_col = None
    best_numeric_score = -1

    for col in candidates:
        numeric_score = 0
        for val in df[col].head(50):  # sample first 50 rows to judge
            num = _coerce_number(val)
            if num is not None:
                numeric_score += 1
        if numeric_score > best_numeric_score:
            best_numeric_score = numeric_score
            best_col = col

    return best_col

def _best_numeric_from(df, topic_regex=None, char_regex=None, geo_col=None, min_pct=None):
    """
    Convenience finder:
    - Filter df by Topic containing topic_regex (case-insensitive) if provided
    - AND/OR Characteristic containing char_regex if provided
    - Look at the first nonzero numeric in geo_col
    - If min_pct is provided, require that numeric >= min_pct (treating it as percent or share)
    Returns that numeric (float) or None.
    """
    sub = df.copy()
    if topic_regex:
        sub = sub[sub["Topic"].str.contains(topic_regex, case=False, na=False)]
    if char_regex:
        sub = sub[sub["Characteristic"].str.contains(char_regex, case=False, na=False)]
    for _, r in sub.iterrows():
        num = _coerce_number(r[geo_col])
        if num is None or num <= 0:
            continue
        # if user wants to skip "insignificant", apply threshold
        if min_pct is not None and num < min_pct:
            continue
        return num
    return None

def extract_significant_languages(df, geo_col, pop_val_num):
    """
    Look at home languages from relevant topics and pull out
    specific non-English/non-French languages that appear
    in meaningful numbers.

    We'll scan characteristics like "German", "Tagalog (Filipino)",
    "Punjabi (Panjabi)", "Cree languages", etc.

    We'll ignore junk rows like "English", "French", "Both English and French",
    "Other languages", "Multiple responses".

    Heuristic for 'meaningful':
    - If we know total pop: language count/pop >= 1%
    - Else: raw count >= 50 (tunable)
    """

    if pop_val_num is None or pop_val_num <= 0:
        pop_val_num = None  # we'll fall back to raw-count threshold

    LANGUAGE_TOPIC_REGEX = (
        "Mother tongue|Language spoken most often at home|Other language spoken regularly at home"
    )

    lang_rows = df[
        df["Topic"].str.contains(LANGUAGE_TOPIC_REGEX, case=False, na=False)
    ].copy()

    significant = []

    for _, r in lang_rows.iterrows():
        label = str(r["Characteristic"]).strip()
        val_num = _coerce_number(r[geo_col])

        if val_num is None or val_num <= 0:
            continue

        # skip English/French buckets
        low_label = label.lower()
        if "english" in low_label or "french" in low_label:
            continue

        # skip generic buckets
        if "other languages" in low_label:
            continue
        if "multiple" in low_label:
            continue
        if "not included elsewhere" in low_label:
            continue

        # If StatCan labels like "German", "Cree languages", "Tagalog (Filipino)"
        # we keep the raw label but tidy slightly:
        clean_label = label
        # Optional cleanup: remove trailing footnote markers like "[...]"
        # or "(incl. dialects)" etc. Keep it simple for now.
        clean_label = clean_label.replace(" languages", "")
        clean_label = clean_label.replace(" language", "")
        clean_label = clean_label.replace(" (Filipino)", "")
        clean_label = clean_label.replace(" (Panjabi)", "")
        clean_label = clean_label.replace(";", ",")
        clean_label = clean_label.strip()

        # materiality test
        is_material = False
        if pop_val_num:
            pct = (val_num / pop_val_num) * 100.0
            if pct >= 1.0:
                is_material = True
        else:
            if val_num >= 50:
                is_material = True

        if is_material:
            significant.append(clean_label)

    # dedupe while preserving order
    seen = set()
    ordered_unique = []
    for lang in significant:
        base = lang.strip()
        if base not in seen:
            seen.add(base)
            ordered_unique.append(base)

    return ordered_unique

def extract_indigenous_nations(df, geo_col):
    """
    Look inside Indigenous-related topics and try to identify which
    nations / identities appear in non-trivial numbers.

    We'll look for labels like:
    - Cree
    - Dene
    - Blackfoot
    - Stoney
    - Saulteaux / Anishinaabe variants
    - Métis
    - Inuit

    We'll filter out generic totals and zero/blank rows.
    We don't currently force a % threshold because sometimes those counts are smaller
    but still culturally important; instead we'll require val_num >= 20 for now.
    You can adjust that threshold.
    """

    INDIG_TOPIC_REGEX = "Indigenous population|Indigenous ancestry|Indigenous identity"
    possible_markers = [
        "cree",
        "dene",
        "blackfoot",
        "stoney",
        "saulteaux",
        "anishinaabe",
        "ojibwe",  # just in case
        "Métis".lower(),
        "metis",   # accent-safe
        "inuit",
        "first nations",
        "north american indian",
    ]

    sub = df[df["Topic"].str.contains(INDIG_TOPIC_REGEX, case=False, na=False)].copy()

    found_groups = []

    for _, r in sub.iterrows():
        label = str(r["Characteristic"]).strip()
        val_num = _coerce_number(r[geo_col])
        if val_num is None or val_num <= 0:
            continue

        lower_label = label.lower()

        # skip rows that are clearly just overall totals
        if "total" in lower_label and "aboriginal" in lower_label:
            continue
        if "total" in lower_label and "indigenous" in lower_label:
            continue
        if "total" == lower_label.strip():
            continue

        # try to detect explicit nations / peoples
        hit_any = False
        for marker in possible_markers:
            if marker in lower_label:
                hit_any = True
                break

        if hit_any:
            # simplify label for readability
            clean_label = (
                label.replace("First Nations (North American Indian)", "First Nations")
                     .replace("Cree First Nations", "Cree")
                     .replace("Cree nations", "Cree")
                     .replace("Dene First Nations", "Dene")
                     .replace("Blackfoot First Nations", "Blackfoot")
                     .replace("Métis", "Métis")
                     .replace("Metis", "Métis")
                     .replace("Inuit", "Inuit")
            )
            clean_label = clean_label.strip()
            # If StatCan phrases it "Cree First Nations individuals", we just want "Cree"
            # quick trim:
            for word in ["First Nations individuals", "First Nations persons", "First Nations people"]:
                clean_label = clean_label.replace(word, "")
            clean_label = clean_label.strip()
            found_groups.append(clean_label)

    # Deduplicate, keep order
    seen = set()
    ordered_unique = []
    for grp in found_groups:
        base = grp.strip()
        if base not in seen:
            seen.add(base)
            ordered_unique.append(base)

    return ordered_unique

def generate_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No summary available."

    geo_col = pick_geo_col(df)
    if not geo_col:
        return "No summary available."

    lines = []

    # --- Population (for % math later) ---
    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = None
    if not pop_rows.empty:
        pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col])
        if pop_val_num and pop_val_num > 0:
            lines.append(
                f"This community has approximately {int(round(pop_val_num, 0))} residents."
            )

    # --- Age structure ---
    kids_val = _best_numeric_from(
        df,
        topic_regex="Age characteristics",
        char_regex=r"\b0\s*to\s*14\s*years\b",
        geo_col=geo_col,
    )
    seniors_val = _best_numeric_from(
        df,
        topic_regex="Age characteristics",
        char_regex=r"65\s*years",
        geo_col=geo_col,
    )

    kids_pct = None
    seniors_pct = None
    if pop_val_num and pop_val_num > 0:
        if kids_val and kids_val > 0 and kids_val < pop_val_num:
            kids_pct = (kids_val / pop_val_num) * 100.0
        if seniors_val and seniors_val > 0 and seniors_val < pop_val_num:
            seniors_pct = (seniors_val / seniors_val if seniors_val == pop_val_num else seniors_val / pop_val_num) * 100.0

    age_bits = []
    if kids_pct and kids_pct >= 1.0:
        age_bits.append(f"Children (0–14) are about {kids_pct:.1f}% of the population")
    elif kids_val and kids_val > 0:
        age_bits.append("There is a meaningful number of children (0–14)")

    if seniors_pct and seniors_pct >= 1.0:
        age_bits.append(f"older adults (65+) are about {seniors_pct:.1f}%")
    elif seniors_val and seniors_val > 0:
        age_bits.append("there is also a visible older adult population (65+)")

    if age_bits:
        if len(age_bits) == 2:
            lines.append(
                f"{age_bits[0]}, and {age_bits[1]}. This shapes demand for youth programs, family supports, and age-appropriate services."
            )
        else:
            lines.append(
                f"{age_bits[0]}. This shapes demand for family supports, school-age programming, and community services."
            )

    # --- Household structure ---
    single_parent_share = _best_numeric_from(
        df,
        topic_regex="Household type|Household and dwelling characteristics",
        char_regex="one-parent|single-parent",
        geo_col=geo_col,
        min_pct=1.0,
    )
    hh_size = _best_numeric_from(
        df,
        topic_regex="Household and dwelling characteristics",
        char_regex="Average household size",
        geo_col=geo_col,
    )
    renters_share = _best_numeric_from(
        df,
        topic_regex="Household and dwelling characteristics|Household type",
        char_regex="rented",
        geo_col=geo_col,
        min_pct=1.0,
    )
    owners_share = _best_numeric_from(
        df,
        topic_regex="Household and dwelling characteristics|Household type",
        char_regex="owned",
        geo_col=geo_col,
        min_pct=1.0,
    )

    household_bits = []
    if single_parent_share:
        household_bits.append("There is a notable share of one-parent households, meaning many caregivers are doing this on their own.")
    if hh_size and hh_size >= 2.5:
        household_bits.append(f"Average household size is about {hh_size:.1f} people, suggesting many multi-child or multi-generational homes.")
    if renters_share and (not owners_share or renters_share > owners_share):
        household_bits.append("Housing leans toward renting, which can mean less stability and more turnover.")
    elif owners_share and owners_share >= 1.0:
        household_bits.append("Most homes appear to be owner-occupied, which often signals longer-term roots in the area.")

    if household_bits:
        lines.append(" ".join(household_bits))

    # --- Indigenous nations / identities ---
    nations = extract_indigenous_nations(df, geo_col)
    if nations:
        if len(nations) == 1:
            lines.append(
                f"There is an Indigenous community presence, including {nations[0]}, which should shape how programs are offered, led, and communicated."
            )
        else:
            nation_list_text = ", ".join(nations[:-1]) + f", and {nations[-1]}"
            lines.append(
                f"There is an Indigenous community presence, including {nation_list_text}, which should shape how programs are offered, led, and communicated."
            )

    # --- Newcomers / immigration ---
    newcomer_val = _best_numeric_from(
        df,
        topic_regex="Immigrant status and period of immigration|Selected places of birth for the recent immigrant population",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )
    if newcomer_val:
        lines.append(
            "There is a visible newcomer / recent immigrant population. Cultural responsiveness, translation support, and parent-facing communication will matter."
        )

    # --- Language landscape, including specific home languages ---
    specific_langs = extract_significant_languages(df, geo_col, pop_val_num)
    minority_lang_val = _best_numeric_from(
        df,
        topic_regex="Children eligible for instruction in the minority official language|Eligibility and instruction in the minority official language",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )

    lang_sentences = []
    if specific_langs:
        if len(specific_langs) == 1:
            lang_sentences.append(f"Families speak {specific_langs[0]} at home in meaningful numbers.")
        else:
            lang_sentences.append(
                "Families speak " +
                ", ".join(specific_langs[:-1]) +
                f", and {specific_langs[-1]} at home in meaningful numbers."
            )

    if minority_lang_val:
        lang_sentences.append(
            "Some children are legally entitled to Francophone minority-language education."
        )

    if lang_sentences:
        lines.append(
            "Language and culture: " +
            " ".join(lang_sentences) +
            " This affects how outreach is delivered and which partners you engage."
        )

    # --- Education level ---
    hs_val = _best_numeric_from(
        df,
        topic_regex="Secondary \\(high\\) school diploma|Highest certificate, diploma or degree",
        char_regex="high school diploma|secondary",
        geo_col=geo_col,
        min_pct=1.0,
    )
    uni_val = _best_numeric_from(
        df,
        topic_regex="Highest certificate, diploma or degree",
        char_regex="bachelor|university|degree",
        geo_col=geo_col,
        min_pct=1.0,
    )
    if hs_val and uni_val:
        lines.append(
            "Education levels are mixed: many adults report high school or trades credentials, and there is also a group with university-level education."
        )
    elif hs_val and not uni_val:
        lines.append(
            "Most adults appear to hold high school or trades-level credentials rather than university degrees."
        )
    elif uni_val and not hs_val:
        lines.append(
            "A significant share of adults report university-level credentials."
        )

    # --- Low income / affordability ---
    low_income_val = _best_numeric_from(
        df,
        topic_regex="Low income and income inequality",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )
    if low_income_val:
        lines.append(
            "Affordability is a real factor: some households are living with low income or inequality, so cost can be a barrier without subsidy."
        )

    # --- Mobility / rootedness ---
    mobility_rows = df[
        df["Topic"].str.contains("Mobility status 1 year ago|Mobility status 5 years ago",
                                 case=False, na=False)
    ]
    mobility_flag = False
    for _, r in mobility_rows.iterrows():
        row_char = r["Characteristic"].lower()
        mv = _coerce_number(r[geo_col])
        if mv and mv > 0 and ("moved" in row_char or "different" in row_char):
            mobility_flag = True
            break
    if mobility_flag:
        lines.append(
            "Families are still moving in and settling, which means not everyone has long-standing local supports yet."
        )

    # --- Commuting / time pressure ---
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
    if car_commute_flag or long_commute_flag:
        commute_bits = []
        if car_commute_flag:
            commute_bits.append("most working adults rely on driving")
        if long_commute_flag:
            commute_bits.append("some families are dealing with long daily commutes")
        if len(commute_bits) == 2:
            commute_sentence = commute_bits[0] + ", and " + commute_bits[1] + "."
        else:
            commute_sentence = commute_bits[0] + "."
        lines.append(
            commute_sentence + " This shapes after-school timing, pickup logistics, and evening availability for programs."
        )

    if not lines:
        return "This community profile did not surface notable demographic features from the selected census fields."

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
