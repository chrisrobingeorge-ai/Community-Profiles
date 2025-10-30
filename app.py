import io
import os
import re
from collections import OrderedDict
from datetime import date, datetime

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
    "Age characteristics",
    "Household and dwelling characteristics",
    "Household type",
    "Income of households in 2020",
    "Low income and income inequality in 2020",
    "Knowledge of official languages",
    "Mother tongue",
    "Language spoken most often at home",
    "Selected places of birth for the recent immigrant population",
    "Ethnic origin",
    "Ethnic or cultural origin",
    "Visible minority",
    "Highest certificate, diploma or degree",
    "Mobility status 1 year ago",
    "Commuting duration",
]

# Sometimes these aren't in the Topic column, but appear in Characteristic instead,
# so we also match using keywords inside the Characteristic column:
TARGET_CHARACTERISTIC_KEYWORDS = [
    "Highest certificate, diploma or degree",
    "Commuting duration",
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


def _characteristic_sort_key(label: str) -> tuple:
    """
    Generate a tuple sort key that:
    - Orders numeric bands (ages, income ranges, commute times) in ascending numeric order.
    - Groups income 'Total - ...' rows after the numeric bands.
    - Pushes 'Average ...', 'Median ...' rows to the bottom.
    - Keeps stable alphabetical fallback inside each group.

    Returns (group_rank, primary_num, secondary_num, text)
    Lower group_rank comes first.
    """

    if label is None:
        return (9999, 9999999, 9999999, "")

    text = str(label).strip()
    low = text.lower()

    # ---------- 1) Special handling for Income tables ----------
    # Bucket: income-bracket ranges like "$5,000 to $9,999", "Under $5,000", "$150,000 and over"
    income_range_patterns = [
        r"^\$?([\d,]+)\s*to\s*\$?([\d,]+)",          # "$5,000 to $9,999"
        r"^under\s*\$?([\d,]+)",                      # "Under $5,000"
        r"^\$?([\d,]+)\s*(and over|\+)",              # "$200,000 and over"
    ]
    for patt in income_range_patterns:
        m = re.match(patt, text, flags=re.IGNORECASE)
        if m:
            # Try to pull the first number
            n1_raw = m.group(1)
            n1 = int(n1_raw.replace(",", "")) if n1_raw else 0

            # Default n2 = n1
            n2 = n1

            # "$5,000 to $9,999"
            m_to = re.match(r"^\$?([\d,]+)\s*to\s*\$?([\d,]+)", text, flags=re.IGNORECASE)
            if m_to:
                n1 = int(m_to.group(1).replace(",", ""))
                n2 = int(m_to.group(2).replace(",", ""))

            # "Under $5,000" -> treat as (0 .. 5000)
            m_under = re.match(r"^under\s*\$?([\d,]+)", text, flags=re.IGNORECASE)
            if m_under:
                n1 = 0
                n2 = int(m_under.group(1).replace(",", ""))

            # "$200,000 and over" -> (200000 .. inf)
            m_over = re.match(r"^\$?([\d,]+)\s*(and over|\+)", text, flags=re.IGNORECASE)
            if m_over:
                n1 = int(m_over.group(1).replace(",", ""))
                n2 = n1 + 999999  # shove it to the end of the numeric ladder

            # group_rank 0 = income ranges at the top of the Income table
            return (0, n1, n2, low)

    # Bucket: "Total - Income statistics ..." etc.
    if low.startswith("total -"):
        # group_rank 1 = totals block, comes after numeric ranges
        return (1, 0, 0, low)

    # Bucket: averages / medians
    if low.startswith("average ") or low.startswith("median "):
        # group_rank 2 = averages/medians at the bottom
        return (2, 0, 0, low)

    # ---------- 2) Generic numeric band handling (ages, commute minutes, etc.) ----------
    # Try to pull “0 to 4 years”, “15 to 19 years”, “5 to 14 minutes”, etc.
    nums = re.findall(r"\d+", text)
    if nums:
        first_num = int(nums[0])
        second_num = int(nums[1]) if len(nums) > 1 else first_num
        return (0, first_num, second_num, low)

    # Special case like "65 years and over", "60 minutes or more"
    m_over_generic = re.match(r"^\s*(\d+).*(and over|or more)", text, flags=re.IGNORECASE)
    if m_over_generic:
        n = int(m_over_generic.group(1))
        return (0, n, n + 1000, low)

    # ---------- 3) Fallback ----------
    # Give it a later rank (1) so numeric bands still sort first.
    return (1, 9999998, 9999998, low)


def resolve_topic_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Topic_norm by:
    - Using Topic when it looks like a proper label (not just '200', '201', empty).
    - Promoting Characteristic lines like '200: Main mode of commuting ...' to become the current topic label.
    - Forward-filling that topic until the next header.
    Also drops pure description header rows (e.g., lines whose Characteristic is the long sentence that
    'refers to ...' and have no data).
    """
    if df.empty:
        df["Topic_norm"] = df.get("Topic", "")
        return df

    df = df.copy()

    # Drop any pure description rows like "Commuting duration refers to ..."
    desc_mask = df["Characteristic"].str.contains(r"\brefers to\b", case=False, na=False)

    # Keep only rows that have at least one non-empty numeric-ish value
    value_cols_tmp = [
        c for c in df.columns
        if c not in ("Topic", "Characteristic", "Notes", "Note", "Symbol", "Flags", "Flag")
    ]
    has_data = df[value_cols_tmp].applymap(lambda x: str(x).strip()).apply(
        lambda row: any(
            v not in ["", "None", "nan", "NaN", "F", "X", "..", "..."] for v in row
        ),
        axis=1,
    )
    df = df[~(desc_mask & ~has_data)].copy()

    # Start Topic_norm as Topic (string)
    df["Topic_norm"] = df.get("Topic", "").astype(str).str.strip()

    current_topic = None
    rows_to_drop = []

    # Identify numeric-only Topic entries (e.g., '200', '201')
    def looks_numeric_topic(t: str) -> bool:
        t = (t or "").strip()
        return bool(re.fullmatch(r"\d{1,3}", t))

    # Loop rows to detect header lines in Characteristic like "200: Main mode of commuting ..."
    for idx, row in df.iterrows():
        topic = str(row.get("Topic", "")).strip()
        char = str(row.get("Characteristic", "")).strip()

        # Case A: a proper Topic string already present and not numeric-only
        if topic and not looks_numeric_topic(topic) and topic.lower() not in ["topic", ""]:
            current_topic = topic
            df.at[idx, "Topic_norm"] = current_topic
            continue

        # Case B: Characteristic contains a "###: Title" header → promote to topic
        m = re.match(r"^\s*(\d{1,3})\s*:\s*(.+)$", char)
        if m:
            # Use the text after the colon as the topic label
            label = m.group(2).strip()
            # Trim off long descriptions after "refers to"
            label = re.split(r"\s+refers to\s+", label, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            current_topic = label if label else char
            df.at[idx, "Topic_norm"] = current_topic

            # Mark this row for dropping if it's really just a header/description
            value_cols2 = [
                c for c in df.columns
                if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")
            ]

            maybe_desc = ("refers to" in char.lower())
            if maybe_desc:
                rows_to_drop.append(idx)
            else:
                # also drop if all value cells are empty/placeholders
                placeholders = {"", "..", "...", "F", "X"}
                col_vals = []
                for c2 in value_cols2:
                    v = df.at[idx, c2] if c2 in df.columns else None
                    s = "" if pd.isna(v) else str(v).strip()
                    col_vals.append(s)
                if all((v2 == "" or v2.upper() in placeholders) for v2 in col_vals):
                    rows_to_drop.append(idx)
            continue

        # Case C: no good Topic; carry forward the last known topic
        if current_topic:
            df.at[idx, "Topic_norm"] = current_topic
        else:
            df.at[idx, "Topic_norm"] = topic  # fallback

    # Drop the header/description rows we flagged
    if rows_to_drop:
        df = df.drop(index=rows_to_drop)

    return df


def filter_relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Topic" not in df.columns or "Characteristic" not in df.columns:
        st.error(
            "This file doesn't look like the standard Census Profile format. "
            "It must include columns named 'Topic' and 'Characteristic'."
        )
        return pd.DataFrame()

    # Normalize topics so '200: Main mode of commuting ...' becomes a real topic label
    df = resolve_topic_column(df)

    # match on normalized topic (case-insensitive equality)
    topic_mask = df["Topic_norm"].str.lower().isin([t.lower() for t in TARGET_TOPICS])

    # match on Characteristic (case-insensitive substring)
    char_mask = False
    for kw in TARGET_CHARACTERISTIC_KEYWORDS:
        char_mask = char_mask | df["Characteristic"].str.lower().str.contains(
            kw.lower(), na=False
        )

    keep_mask = topic_mask | char_mask
    filtered = df[keep_mask].copy()

    # numeric-aware sort within each topic
    filtered["__char_sort_key__"] = filtered["Characteristic"].apply(_characteristic_sort_key)

    filtered.sort_values(
        by=["Topic_norm", "__char_sort_key__"],
        inplace=True,
        ignore_index=True,
    )

    # drop helper col from the version we return
    filtered.drop(columns=["__char_sort_key__"], inplace=True, errors="ignore")

    return filtered


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


def row_has_nonzero_data(row: pd.Series, value_cols: list[str]) -> bool:
    """Return True if at least one of the given columns has a number > 0."""
    for c in value_cols:
        if c not in row:
            continue
        num = _coerce_number(row[c])
        if num is not None and num > 0:
            return True
    return False


def render_report(df: pd.DataFrame):
    """
    Show results in collapsible sections by Topic, but:
    - hide topics we don't want as tables (but maybe we still use for summary)
    - hide rows that have no meaningful (>0) values in any numeric column
    - keep expanders collapsed by default
    """
    if df.empty:
        st.warning("No matching rows found in this CSV for the selected fields.")
        return

    # columns that actually carry values (exclude metadata-ish columns)
    value_cols = [
        c for c in df.columns
        if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")
    ]

    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"

    # Topics you said you don't need as visible tables:
    HIDE_THESE_TOPICS = {
        "Mobility status 1 year ago",
        "Mobility status 5 years ago",
        "Selected places of birth for the recent immigrant population",
    }

    for topic, sub in df.groupby(topic_col, dropna=False):
        if topic in HIDE_THESE_TOPICS:
            continue

        # keep only rows where at least one of the value_cols is >0
        filtered_rows = []
        for _, r in sub.iterrows():
            if row_has_nonzero_data(r, value_cols):
                filtered_rows.append(r)

        if not filtered_rows:
            # if nothing survives in this topic, don't show that topic at all
            continue

        pretty_df = pd.DataFrame(filtered_rows)[["Characteristic"] + value_cols].reset_index(drop=True)

        with st.expander(f"📂 {topic}", expanded=False):
            st.dataframe(pretty_df, use_container_width=True)


def drop_zero_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with rows removed if they have no >0 numeric values
    in any value column (same rule we use for on-screen tables).
    """
    if df.empty:
        return df.copy()

    value_cols = [
        c for c in df.columns
        if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")
    ]

    keep_mask = []
    for _, r in df.iterrows():
        keep_mask.append(row_has_nonzero_data(r, value_cols))

    out = df[keep_mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def build_filtered_csv(df: pd.DataFrame) -> bytes:
    cleaned_no_zeros = drop_zero_only_rows(df)
    return cleaned_no_zeros.to_csv(index=False).encode("utf-8-sig")


def build_printable_html(df: pd.DataFrame) -> str:
    """
    Build a simple HTML report the user can 'Save as PDF'.
    We also hide rows that are 0-only the same way the UI does.
    """
    cleaned = drop_zero_only_rows(df)

    if cleaned.empty:
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

    # value columns = the numeric / counts columns we actually care about
    value_cols = [
        c for c in cleaned.columns
        if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")
    ]
    topic_col = "Topic_norm" if "Topic_norm" in cleaned.columns else "Topic"

    # group by topic, same as UI
    for topic, sub in cleaned.groupby(topic_col, dropna=False):
        if sub.empty:
            continue

        parts.append(f"<h2>{topic}</h2>")
        tmp = sub[["Characteristic"] + value_cols].reset_index(drop=True)
        parts.append(tmp.to_html(index=False, escape=False))

    parts.append("</body></html>")
    return "\n".join(parts)


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
    Return a list of concrete languages spoken at home in meaningful numbers,
    excluding English/French and excluding generic/statistical buckets.

    We'll scan "Mother tongue", "Language spoken most often at home",
    "Other language spoken regularly at home".
    """

    LANGUAGE_TOPIC_REGEX = (
        "Mother tongue|Language spoken most often at home|Other language spoken regularly at home"
    )

    # terms we do NOT want to output in the summary because they're buckets/families/stat headers:
    block_terms = [
        "total",
        "official",
        "non-official",
        "non official",
        "single responses",
        "multiple responses",
        "indo-european",
        "germanic",
        "balto-slavic",
        "slavic",
        "germanic languages",
        "germanic language",
        "language family",
        "not included elsewhere",
        "other languages",
        "languages not included elsewhere",
        "aboriginal languages",
        "indigenous languages",  # we'll handle Indigenous through nations instead
    ]

    # languages we ALWAYS ignore in this function (they show up elsewhere anyway)
    ignore_exact = [
        "english",
        "french",
        "english and french",
    ]

    # materiality rules
    MIN_PCT = 1.0   # at least ~1% of total pop
    MIN_COUNT = 50  # fallback if we don't know pct

    if pop_val_num is None or pop_val_num <= 0:
        pop_val_num = None  # so we use raw count fallback

    lang_rows = df[
        df["Topic"].str.contains(LANGUAGE_TOPIC_REGEX, case=False, na=False)
    ].copy()

    significant = []

    for _, r in lang_rows.iterrows():
        raw_label = str(r["Characteristic"]).strip()
        val_num = _coerce_number(r[geo_col])
        if val_num is None or val_num <= 0:
            continue

        low_label = raw_label.lower()

        # filter obvious junk
        if any(bt in low_label for bt in block_terms):
            continue
        if any(low_label == ig for ig in ignore_exact):
            continue
        if any(low_label.startswith(ig) for ig in ignore_exact):
            continue

        # Also skip anything that's literally just "English", "French",
        # "Both English and French", those we don't need here.
        if "english" in low_label and "french" not in low_label and len(low_label.split()) <= 3:
            continue
        if "french" in low_label and "english" not in low_label and len(low_label.split()) <= 3:
            continue

        # Heuristic: keep things that look like named languages / ethnolinguistic communities
        clean_label = raw_label
        clean_label = clean_label.replace(" (Filipino)", "")
        clean_label = clean_label.replace(" (Panjabi)", "")
        clean_label = clean_label.replace(" languages", "")
        clean_label = clean_label.replace(" language", "")
        clean_label = clean_label.strip()

        # if we know population, test percent
        is_material = False
        if pop_val_num:
            pct = (val_num / pop_val_num) * 100.0
            if pct >= MIN_PCT:
                is_material = True
        else:
            if val_num >= MIN_COUNT:
                is_material = True

        if is_material:
            significant.append(clean_label)

    # Dedupe and keep order
    seen = set()
    ordered_unique = []
    for lang in significant:
        key = lang.lower()
        if key not in seen:
            seen.add(key)
            ordered_unique.append(lang)

    return ordered_unique


def derive_indigenous_from_ethnic_origin(
    df: pd.DataFrame,
    geo_col: str,
    pop_val_num: float | None = None
) -> pd.DataFrame:
    """
    Build a compact table of Indigenous groups using the Ethnic origin / Ethnic or cultural origin topic.
    Output: columns ['Group', 'Count', 'Percent'] (Percent only if pop_val_num provided)
    """

    if df.empty or not geo_col:
        return pd.DataFrame(columns=["Group", "Count", "Percent"])

    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"
    ETHNIC_TOPIC_REGEX = r"(Ethnic origin|Ethnic or cultural origin|Origine ethnique ou culturelle)"

    sub = df[df[topic_col].str.contains(ETHNIC_TOPIC_REGEX, case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Group", "Count", "Percent"])

    # Canonical buckets for AB-relevant Nations/Peoples (+ common variants)
    groups_patterns: dict[str, list[str]] = {
        "Cree / Anishinaabe (incl. Saulteaux, Ojibwe, Oji-Cree)": [
            r"\bcree\b", r"\bplains\s*cree\b", r"\bwoodland\s*cree\b",
            r"\boji[\-\s]?cree\b", r"\bsaulteaux\b", r"\banishinaabe\b", r"\bojibw?e?y?\b",
        ],
        "Dene": [r"\bdene\b"],
        "Tsuut’ina": [r"\btsu+?t[’'\- ]?ina\b"],
        "Blackfoot": [r"\bblackfoot\b"],
        "Stoney Nakoda": [r"\bstoney\b", r"\bnakoda\b"],
        "Métis": [r"\bm[ée]tis\b", r"\bmetis\b"],
        "Inuit": [r"\binuit\b"],
        "First Nations (unspecified)": [
            r"first\s+nations\s*\(north american indian\).*n\.o\.s\.",
            r"\bfirst\s+nations\b.*\bn\.o\.s\.",
            r"\bnorth american indigenous\b.*\bn\.o\.s\.",
        ],
    }

    # Ignore obvious non-ethnic / irrelevant categories
    ignore_if_matches = [
        r"\bchristian\b", r"\bbuddhist\b", r"\bhindu\b", r"\bmuslim\b", r"\bsikh\b",
        r"\bontarian\b", r"\balbertan\b", r"\bmanitoban\b", r"\bnewfoundlander\b",
        r"\bcanadian\b", r"\bqu[eè]b[ée]cois\b", r"\bfranco\s*ontarian\b",
        r"\bafrican\b.*\bn\.o\.s\.", r"\basian\b.*\bn\.o\.s\.", r"\bslavic\b.*\bn\.o\.s\.",
        r"\bnorth american\b.*\bn\.o\.s\.",
    ]
    ignore_re = re.compile("|".join(ignore_if_matches), re.IGNORECASE)

    tallies: dict[str, float] = {k: 0.0 for k in groups_patterns.keys()}

    for _, r in sub.iterrows():
        label = str(r["Characteristic"]).strip().lower()
        if ignore_re.search(label):
            continue
        val = _coerce_number(r[geo_col])
        if val is None or val <= 0:
            continue

        for group, patt_list in groups_patterns.items():
            if any(re.search(p, label, flags=re.IGNORECASE) for p in patt_list):
                tallies[group] += float(val)
                break  # stop at first matching group

    rows = []
    for group, count in tallies.items():
        if count > 0:
            pct = (count / pop_val_num * 100.0) if (pop_val_num and pop_val_num > 0) else None
            rows.append({
                "Group": group,
                "Count": int(round(count)),
                "Percent": (f"{pct:.1f}%" if pct is not None else None),
            })

    out = pd.DataFrame(rows, columns=["Group", "Count", "Percent"])
    if not out.empty:
        out = out.sort_values(by="Count", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def build_indigenous_table(df: pd.DataFrame, geo_col: str, pop_val_num: float | None):
    """
    Safe wrapper around derive_indigenous_from_ethnic_origin.
    Returns a DataFrame with columns ['Group','Count','Percent'] or
    an empty DataFrame if we can't build it.
    """
    try:
        tbl = derive_indigenous_from_ethnic_origin(
            df,
            geo_col=geo_col,
            pop_val_num=pop_val_num
        )
        if tbl is None:
            return pd.DataFrame(columns=["Group", "Count", "Percent"])
        return tbl
    except Exception:
        # If anything goes sideways (missing topic, weird column), fail soft.
        return pd.DataFrame(columns=["Group", "Count", "Percent"])


def summarize_indigenous_nations_from_ethnic(df: pd.DataFrame) -> list[str]:
    """
    Return a cleaned list of Nation/People names (no counts) based on the
    ethnic-origin-derived Indigenous table.
    """
    geo_col = pick_geo_col(df)
    if not geo_col:
        return []

    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"
    pop_rows = df[
        (df[topic_col].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col]) if not pop_rows.empty else None

    nations_df = derive_indigenous_from_ethnic_origin(df, geo_col, pop_val_num)
    if nations_df.empty:
        return []

    # Dedupe groups in order
    groups = nations_df["Group"].astype(str).tolist()
    seen = set()
    ordered = []
    for g in groups:
        low = g.lower().strip()
        if low not in seen:
            seen.add(low)
            ordered.append(g)
    return ordered


# --- Age band logic and cohort-aging ---
CENSUS_REFERENCE_DATE = date(2021, 5, 11)  # 2021 Census Day

AGE_BANDS_ORDER = [
    "0 to 4 years",
    "5 to 9 years",
    "10 to 14 years",
    "15 to 19 years",
    "20 to 24 years",
    "25 to 29 years",
    "30 to 34 years",
    "35 to 39 years",
    "40 to 44 years",
    "45 to 49 years",
    "50 to 54 years",
    "55 to 59 years",
    "60 to 64 years",
    "65 to 69 years",
    "70 to 74 years",
    "75 to 79 years",
    "80 to 84 years",
    "85 years and over",
]


def _find_age_value(df, geo_col, band_label):
    row = df[
        (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
        (df["Characteristic"].str.fullmatch(rf"\s*{band_label}\s*", case=False, na=False))
    ]
    if row.empty:
        # try contains (some extracts have indenting/spaces)
        row = df[
            (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
            (df["Characteristic"].str.contains(rf"{band_label}", case=False, na=False))
        ]
    if row.empty:
        return 0.0
    return _coerce_number(row.iloc[0][geo_col]) or 0.0


def extract_age_bands(df, geo_col) -> OrderedDict:
    """
    Returns an ordered dict of {band_label: count} for standard 5-year bands and 85+.
    Missing bands default to 0.
    """
    bands = OrderedDict()
    for lab in AGE_BANDS_ORDER:
        bands[lab] = float(_find_age_value(df, geo_col, lab))
    return bands


def _shift_once_one_year(bands: OrderedDict) -> OrderedDict:
    """
    Shift 1 year forward: move 1/5 of each 5-year band up into the next band.
    Top band (85+) only receives; 0–4 loses but we don't add births.
    """
    out = OrderedDict((k, 0.0) for k in bands.keys())
    keys = list(bands.keys())
    for i, k in enumerate(keys):
        v = bands[k]
        if k == "85 years and over":
            # receives inflow from previous band; keeps its current residents
            out[k] += v
        else:
            stay = v * 4.0 / 5.0
            move = v * 1.0 / 5.0
            out[k] += stay
            # push to next band
            nxt = keys[i + 1]
            out[nxt] += move
    return out


def age_bands_adjust_to_date(bands: OrderedDict, as_of: date) -> OrderedDict:
    """
    Age the 2021 bands forward to 'as_of' date with 1/5 per year movement.
    Works for fractional years too.
    """
    # years since census day
    delta_years = max(0.0, (as_of - CENSUS_REFERENCE_DATE).days / 365.25)
    if delta_years == 0:
        return bands

    # integer years
    y = int(delta_years)
    frac = delta_years - y

    cur = bands.copy()
    for _ in range(y):
        cur = _shift_once_one_year(cur)

    if frac > 0:
        # apply fractional shift: move (frac/5) of each non-top band up
        out = OrderedDict((k, 0.0) for k in cur.keys())
        keys = list(cur.keys())
        for i, k in enumerate(keys):
            v = cur[k]
            if k == "85 years and over":
                out[k] += v
            else:
                stay = v * (1.0 - frac / 5.0)
                move = v * (frac / 5.0)
                out[k] += stay
                nxt = keys[i + 1]
                out[nxt] += move
        cur = out

    return cur


def sum_bands(bands: OrderedDict, wanted_labels: list[str]) -> float:
    return float(sum(bands.get(l, 0.0) for l in wanted_labels))


def get_childteen_bands(df: pd.DataFrame, geo_col: str, as_of_date: date | None):
    """
    Return a dict with counts for:
      '0 to 4 years'
      '5 to 9 years'
      '10 to 14 years'
      '15 to 19 years'

    Uses cohort-aging if as_of_date is provided.
    Falls back to 2021 raw values otherwise.
    """

    wanted_labels = [
        "0 to 4 years",
        "5 to 9 years",
        "10 to 14 years",
        "15 to 19 years",
    ]

    # OPTION A: user chose "age forward"
    if as_of_date is not None:
        # build full 5-year bands from 2021
        bands_2021 = extract_age_bands(df, geo_col)
        # age them forward to as_of_date
        bands_adj = age_bands_adjust_to_date(bands_2021, as_of=as_of_date)

        result = {}
        for lab in wanted_labels:
            result[lab] = float(bands_adj.get(lab, 0.0))
        return result

    # OPTION B: use raw 2021 numbers directly from the CSV
    result = {}
    for lab in wanted_labels:
        # try exact match first
        row = df[
            (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
            (df["Characteristic"].str.fullmatch(rf"\s*{lab}\s*", case=False, na=False))
        ]
        if row.empty:
            # lose the strictness, try contains
            row = df[
                (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
                (df["Characteristic"].str.contains(rf"{lab}", case=False, na=False))
            ]

        if row.empty:
            result[lab] = 0.0
        else:
            result[lab] = float(_coerce_number(row.iloc[0][geo_col]) or 0.0)

    return result


def extract_place_name(upload_name: str) -> str:
    """
    Try to turn something like
    'CensusProfile2021-ProfilRecensement2021-CountyOfStettlerNo6.csv'
    into 'County Of Stettler No 6'.
    """
    stem = os.path.splitext(upload_name)[0]

    # break on common separators
    parts = re.split(r"[-_]", stem)
    # prefer the last part that actually has alphabetic characters
    candidate = None
    for p in reversed(parts):
        if re.search(r"[A-Za-z]", p):
            candidate = p
            break
    if candidate is None:
        candidate = stem

    # If the chosen chunk is just "CensusProfile2021" etc., fall back to whole stem
    if re.match(r"(?i)censusprofile|profilrecensement", candidate):
        candidate = stem

    # Insert spaces before capital letters or digits to get
    # "CountyOfStettlerNo6" -> "County Of Stettler No 6"
    friendly = re.sub(r"(?<!^)([A-Z0-9])", r" \1", candidate).strip()

    # Collapse multiple spaces
    friendly = re.sub(r"\s+", " ", friendly)

    # Title case
    friendly = friendly.strip()
    if friendly:
        friendly = friendly[0].upper() + friendly[1:]

    return friendly


def generate_summary(
    df: pd.DataFrame,
    as_of_date: date | None = None,
    place_name: str | None = None,
) -> str:
    """
    Build a multi-section narrative for staff.
    Adds scale ("~480 kids"), implications ("expect siblings"), and delivery guidance.
    """

    if df.empty:
        return "No summary available."

    geo_col = pick_geo_col(df)
    if not geo_col:
        return "No summary available."

    community_label = place_name if place_name else "this community"

    # ---------------------------
    # 1. Population base
    # ---------------------------
    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = None
    if not pop_rows.empty:
        pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col])

    # ---------------------------
    # 2. Age structure
    # ---------------------------
    kids_band_labels = ["0 to 4 years", "5 to 9 years", "10 to 14 years"]
    teen_band_labels = ["15 to 19 years"]
    senior_band_labels = [
        "65 to 69 years", "70 to 74 years", "75 to 79 years",
        "80 to 84 years", "85 years and over"
    ]

    if as_of_date is not None:
        age_bands = extract_age_bands(df, geo_col)
        adj = age_bands_adjust_to_date(age_bands, as_of=as_of_date)
        kids_val = sum_bands(adj, kids_band_labels)
        teens_val = sum_bands(adj, teen_band_labels)
        seniors_val = sum_bands(adj, senior_band_labels)
    else:
        def grab_sum_for(labels):
            total = 0.0
            for lab in labels:
                sub = df[
                    (df["Topic"].str.contains("Age characteristics", case=False, na=False)) &
                    (df["Characteristic"].str.contains(lab, case=False, na=False))
                ]
                if not sub.empty:
                    v = _coerce_number(sub.iloc[0][geo_col])
                    if v:
                        total += v
            return total

        kids_val = grab_sum_for(kids_band_labels)
        teens_val = grab_sum_for(teen_band_labels)
        seniors_val = grab_sum_for(senior_band_labels)

    def pct_and_count(val):
        if not val or val <= 0 or not pop_val_num or pop_val_num <= 0:
            return None, None
        pct = (val / pop_val_num) * 100.0
        return pct, int(round(val))

    kids_pct, kids_cnt = pct_and_count(kids_val)
    teens_pct, teens_cnt = pct_and_count(teens_val)
    seniors_pct, seniors_cnt = pct_and_count(seniors_val)

    # ---------------------------
    # 3. Household structure & cost / stability
    # ---------------------------
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

    low_income_val = _best_numeric_from(
        df,
        topic_regex="Low income and income inequality",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )

    # mobility / turnover
    mobility_rows = df[
        df["Topic"].str.contains(
            "Mobility status 1 year ago|Mobility status 5 years ago",
            case=False,
            na=False,
        )
    ]
    mobility_flag = False
    for _, r in mobility_rows.iterrows():
        mv = _coerce_number(r[geo_col])
        row_char = str(r["Characteristic"]).lower()
        if mv and mv > 0 and ("moved" in row_char or "different" in row_char):
            mobility_flag = True
            break

    # commute stress
    commute_rows = df[
        df["Topic"].str.contains("Main mode of commuting|Commuting duration", case=False, na=False)
    ]
    long_commute_flag = False
    car_commute_flag = False
    for _, r in commute_rows.iterrows():
        row_char = str(r["Characteristic"]).lower()
        v = _coerce_number(r[geo_col])
        if not v or v <= 0:
            continue
        if "60 minutes" in row_char or "longer" in row_char:
            long_commute_flag = True
        if ("car" in row_char or "automobile" in row_char or "driver" in row_char):
            car_commute_flag = True

    # ---------------------------
    # 4. Languages / Indigenous Nations
    # ---------------------------
    sig_langs = extract_significant_languages(df, geo_col, pop_val_num)

    nations = build_indigenous_table(df, geo_col, pop_val_num)
    nation_bits = []
    if nations is not None and not nations.empty:
        for _, row in nations.iterrows():
            g = str(row["Group"])
            p = (
                str(row["Percent"])
                if ("Percent" in nations.columns and pd.notna(row["Percent"]))
                else None
            )
            if p and p != "None":
                nation_bits.append(f"{g} ({p})")
            else:
                nation_bits.append(g)

    # ---------------------------
    # BUILD SECTIONS
    # ---------------------------

    # A) Youth & caregivers
    youth_lines = []
    if kids_cnt and kids_pct:
        youth_lines.append(
            f"There are roughly {kids_cnt} children under 15 "
            f"({kids_pct:.1f}% of the population) living in {community_label}. "
            "That supports recurring, in-community kids programming — not just a one-time visit."
        )
    else:
        youth_lines.append(
            "There is a meaningful number of young children here, which supports recurring, "
            "in-community kids programming — not just a one-time visit."
        )

    if teens_cnt and teens_pct:
        youth_lines.append(
            f"There is also a visible group of about {teens_cnt} teens (15–19). "
            "This is the age where girls in particular tend to drop out of physical activity, "
            "so programs for this band need to feel socially safe first, physically demanding second."
        )
    else:
        youth_lines.append(
            "There is also a visible teen (15–19) population. This is where drop-off in participation "
            "starts, especially for girls, so programming must feel socially safe before it asks for performance."
        )

    if seniors_cnt and seniors_pct:
        youth_lines.append(
            f"About {seniors_cnt} residents ({seniors_pct:.1f}%) are 65+. "
            "Grandparents often act as drivers and supervisors, so it matters that our spaces feel welcoming "
            "and have sightlines/seating."
        )
    else:
        youth_lines.append(
            "There is also a notable older adult (65+) presence. Grandparents often handle pickup and supervision, "
            "so we should assume they’ll be physically in the space."
        )

    youth_block = " ".join(youth_lines)

    # B) Household reality & cost
    hh_lines = []

    if single_parent_share:
        hh_lines.append(
            "A meaningful share of households are led by a single caregiver. "
            "That means one adult is handling work, transport, meals, siblings, and bedtime. "
            "Programs that assume “two parents can drive and wait” will underperform."
        )

    if hh_size and hh_size >= 2.5:
        hh_lines.append(
            f"Average household size is about {hh_size:.1f} people. "
            "We should expect siblings to arrive together, cousins staying over, or grandparents in the same home. "
            "In practice: if we invite one child, two or three may actually show up."
        )

    if renters_share and (not owners_share or renters_share > owners_share):
        hh_lines.append(
            "Housing leans toward renting. Renting is usually linked to more moves and less long-term stability, "
            "so retention will depend on keeping re-entry easy (families should be able to miss a week and return)."
        )
    elif owners_share:
        hh_lines.append(
            "Most homes appear to be owner-occupied, which usually signals more stability and long-term community roots. "
            "That’s helpful when we’re trying to keep a recurring program going in one physical place."
        )

    if low_income_val:
        hh_lines.append(
            "Income stress is present. We should assume cost is a real blocker — not just class fees, "
            "but also shoes, rides, snacks, and anything that sounds like “tuition.” "
            "If it sounds like school fees, some families will self-select out before we ever meet them."
        )

    if mobility_flag:
        hh_lines.append(
            "Families are still moving in and settling. Some do not have long-standing local support networks yet, "
            "so we should not assume parents “already know who to talk to” for help."
        )

    hh_block = " ".join(hh_lines) if hh_lines else (
        "Household patterns suggest we should design for multiple kids per household and assume some level of financial pressure. "
        "Keeping programs low-cost, predictable, and local will matter."
    )

    # C) Language, Nations, trust channels
    lang_lines = []

    if sig_langs:
        if len(sig_langs) == 1:
            lang_lines.append(
                f"Families are using {sig_langs[0]} at home in meaningful numbers. "
                "Written outreach, consent forms, and first-contact conversations should not assume English is the default language of trust."
            )
        else:
            lang_lines.append(
                "Families are using " +
                ", ".join(sig_langs[:-1]) +
                f", and {sig_langs[-1]} at home in meaningful numbers. "
                "Outreach and consent materials should reflect that — if the parent can’t read the invite, "
                "the child never shows up."
            )

    if nation_bits:
        if len(nation_bits) == 1:
            lang_lines.append(
                f"Indigenous presence is visible: {nation_bits[0]}. "
                "Programming should be co-presented with Indigenous partners, not just offered “to” them. "
                "Youth participation is more reliable when leadership they already trust is in the room."
            )
        else:
            lang_lines.append(
                "Indigenous presence is visible: " +
                ", ".join(nation_bits[:-1]) +
                f", and {nation_bits[-1]}. "
                "Programming should be co-presented with Indigenous partners, not just offered “to” them. "
                "Youth participation is more reliable when leadership they already trust is in the room."
            )

    if not lang_lines:
        lang_lines.append(
            "We should not assume English-only outreach. Even when English is widely spoken, "
            "trust and comfort often sit with a first or home language."
        )

    # D) Schedule & delivery practicality
    sched_lines = []
    if car_commute_flag or long_commute_flag:
        if car_commute_flag and long_commute_flag:
            sched_lines.append(
                "Caregivers are already spending serious time driving for work, often 60+ minutes daily. "
                "That means our best delivery window is after school / early evening, in the local school or hall, "
                "because extra travel will kill attendance."
            )
        elif car_commute_flag:
            sched_lines.append(
                "Most adults rely on driving. We should plan for in-town delivery so no one has to add another car trip across the region."
            )
        elif long_commute_flag:
            sched_lines.append(
                "Some people are commuting long distances already. If we add more driving, attendance will fall off fast."
            )
    else:
        sched_lines.append(
            "Scheduling should assume tight evenings. Short, predictable blocks in a local space "
            "(school gym, community hall) will work better than destination programs."
        )

    sched_block = " ".join(sched_lines)

    # E) Final ops guidance
    ops_block = (
        "Operational guidance: bring programming directly into the school or trusted local space; "
        "keep point-of-use cost effectively zero; welcome siblings and grandparents in the room; "
        "use languages and messengers families already trust (school staff, settlement workers, Indigenous leadership, church/community connectors). "
        "Success here is not “did kids attend once,” it’s “did the family feel safe enough to come back next week.”"
    )

    # ---------------------------
    # Assemble final narrative
    # ---------------------------
    final_sections = [
        f"Community Profile Summary for {community_label}",
        "YOUTH & CAREGIVERS: " + youth_block,
        "HOUSEHOLD REALITY & COST: " + hh_block,
        "LANGUAGE, TRUST & INDIGENOUS PRESENCE: " + " ".join(lang_lines),
        "SCHEDULING & ACCESS: " + sched_block,
        "PROGRAM DELIVERY NOTE: " + ops_block,
    ]

    return "\n\n".join(final_sections)


def prune_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove low-value metadata columns and empty columns, while keeping Topic/Characteristic
    and any geography/value columns.

    Rules:
    - Drop columns whose name matches common flag/metadata patterns (case-insensitive):
      endswith '_flag', contains 'flag', 'symbol', 'note', 'quality', 'status'
    - Drop exact duplicates created by CSV tools (e.g., '.1', '.2' suffixes) *if* the
      base column also exists and the duplicate is entirely empty or identical.
    - Drop columns that are completely empty or only contain placeholders ('..', '...', 'F', 'X', '').
    """

    if df.empty:
        return df

    KEEP_ALWAYS = {"Topic", "Characteristic"}

    # 1) Drop obvious flag/metadata columns by name
    flag_like = []
    for c in df.columns:
        cl = c.lower()
        if c in KEEP_ALWAYS:
            continue
        if (
            cl.endswith("_flag")
            or "flag" in cl
            or "symbol" in cl
            or (cl.startswith("note") or cl == "note" or "notes" in cl)
            or "quality" in cl
            or "status" in cl
        ):
            flag_like.append(c)

    df2 = df.drop(columns=flag_like, errors="ignore")

    # 2) Remove fully empty / placeholder-only columns (except KEEP_ALWAYS)
    PLACEHOLDERS = {"", "..", "...", "f", "x"}
    drop_empty = []
    for c in df2.columns:
        if c in KEEP_ALWAYS:
            continue
        col = df2[c]
        # Treat as string for placeholder check
        as_str = col.astype(str).str.strip().str.lower()
        # Consider NaN or placeholder as "empty"
        empties = as_str.isna() | as_str.isin(PLACEHOLDERS)
        if empties.all():
            drop_empty.append(c)
    df2 = df2.drop(columns=drop_empty, errors="ignore")

    # 3) Drop duplicate-suffix columns like '.1', '.2' when base exists and dup is empty/identical
    dup_like = []
    for c in df2.columns:
        m = re.match(r"^(.*)\.(\d+)$", c)
        if not m:
            continue
        base = m.group(1)
        if base in df2.columns:
            # drop the suffixed one if it's identical or empty
            if df2[c].equals(df2[base]):
                dup_like.append(c)
            else:
                # If it's just empty/placeholder, drop
                as_str = df2[c].astype(str).str.strip().str.lower()
                if as_str.isin(PLACEHOLDERS).all():
                    dup_like.append(c)
    df2 = df2.drop(columns=dup_like, errors="ignore")

    return df2


def collapse_duplicate_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    StatCan often repeats the same 'Characteristic' twice in a row:
    - first row has counts + %
    - second row re-states % only
    We only need the first one.

    We define "duplicate" as: same topic label (Topic_norm if available, else Topic)
    AND same Characteristic text. We keep the first occurrence.
    """

    if df.empty:
        return df

    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"

    seen = set()
    keep_rows = []

    for idx, row in df.iterrows():
        key = (
            str(row.get(topic_col, "")).strip(),
            str(row.get("Characteristic", "")).strip(),
        )

        if key in seen:
            continue
        seen.add(key)
        keep_rows.append(idx)

    out = df.loc[keep_rows].copy().reset_index(drop=True)
    return out


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
    st.stop()
else:
    # 1) Load raw census CSV
    raw_df = load_statcan_csv(uploaded_file)

    # 2) Filter to the topics we care about
    cleaned_df = filter_relevant_rows(raw_df)

    # 3) Drop the annoying duplicate "percent-only" rows
    cleaned_df = collapse_duplicate_characteristics(cleaned_df)

    # 4) Strip metadata / placeholder columns from the filtered set
    cleaned_df = prune_columns(cleaned_df)

    # 5) Sidebar controls (age-to-date)
    with st.sidebar:
        st.markdown("### Real-time age adjustment")
        use_age_adjust = st.checkbox("Age cohorts forward from 2021", value=True)
        as_of = st.date_input("As-of date", value=date.today())

    # 6) Summary (uses the age controls and the inferred place name)
    st.subheader("Community Profile Summary")
    place_guess = extract_place_name(uploaded_file.name)

    summary_text = generate_summary(
        cleaned_df,
        as_of_date=(as_of if use_age_adjust else None),
        place_name=place_guess,
    )
    st.write(summary_text)

    # 7) Filtered table view (collapsible topics, zero-row suppression)
    st.subheader("Filtered Report")
    render_report(cleaned_df)

    # 8) Indigenous rollup from Ethnic origin
    geo_col = pick_geo_col(cleaned_df)

    topic_col_for_pop = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"
    pop_rows = cleaned_df[
        (cleaned_df[topic_col_for_pop].str.contains("Population and dwellings", case=False, na=False))
        &
        (cleaned_df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col]) if not pop_rows.empty else None

    st.subheader("Indigenous Population (Ethnic origin–derived)")
    indig_from_ethnic = derive_indigenous_from_ethnic_origin(cleaned_df, geo_col, pop_val_num)
    if indig_from_ethnic.empty:
        st.caption("No Alberta-relevant Indigenous groups detected in Ethnic origin.")
    else:
        st.dataframe(indig_from_ethnic, use_container_width=True)

    # 9) Exports
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
