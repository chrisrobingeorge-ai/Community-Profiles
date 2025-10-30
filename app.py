import io
import os
import re
import pandas as pd
from datetime import date
from collections import OrderedDict

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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


def _percent_or_none(val, pop_val_num):
    """
    Return (pct, count_int) if we can calculate a percent of total pop.
    Otherwise (None, rounded_count_if_any).
    """
    if val is None or val <= 0:
        return (None, None)
    if pop_val_num and pop_val_num > 0:
        pct = (val / pop_val_num) * 100.0
        return (pct, int(round(val)))
    else:
        return (None, int(round(val)))


def _top_n(seq, n):
    """Return first n unique non-empty strings from seq."""
    out = []
    seen = set()
    for item in seq:
        s = str(item).strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
        if len(out) >= n:
            break
    return out


def _safe_pct(x):
    """Format a percent or return None gracefully."""
    if x is None:
        return None
    try:
        return f"{x:.1f}%"
    except Exception:
        return None


def generate_summary(
    df: pd.DataFrame,
    as_of_date: date | None = None,
    place_name: str | None = None,
) -> str:
    """
    Build an expert-facing community brief with:
    - scale
    - implications
    - ops guidance
    - risks / mitigations
    - early KPIs

    Assumptions:
    - df is already cleaned_df (so duplicate rows dropped etc.)
    - pick_geo_col(df) works
    """

    if df.empty:
        return "No summary available."

    geo_col = pick_geo_col(df)
    if not geo_col:
        return "No summary available."

    community_label = place_name if place_name else "this community"

    # -------------------------------------------------
    # 1. Population base (2021)
    # -------------------------------------------------
    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = None
    if not pop_rows.empty:
        pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col])

    # -------------------------------------------------
    # 2. Age structure (children, teens, seniors)
    # -------------------------------------------------
    # buckets:
    kids_band_labels   = ["0 to 4 years", "5 to 9 years", "10 to 14 years"]
    teens_band_labels  = ["15 to 19 years"]
    seniors_band_labels = [
        "65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"
    ]

    if as_of_date is not None:
        # cohort-adjusted forward from 2021
        bands_2021 = extract_age_bands(df, geo_col)
        adj = age_bands_adjust_to_date(bands_2021, as_of=as_of_date)

        kids_val   = sum_bands(adj, kids_band_labels)
        teens_val  = sum_bands(adj, teens_band_labels)
        seniors_val = sum_bands(adj, seniors_band_labels)
    else:
        # raw 2021 values
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

        kids_val   = grab_sum_for(kids_band_labels)
        teens_val  = grab_sum_for(teens_band_labels)
        seniors_val = grab_sum_for(seniors_band_labels)

    kids_pct, kids_cnt         = _percent_or_none(kids_val, pop_val_num)
    teens_pct, teens_cnt       = _percent_or_none(teens_val, pop_val_num)
    seniors_pct, seniors_cnt   = _percent_or_none(seniors_val, pop_val_num)

    # -------------------------------------------------
    # 3. Household structure / cost pressure
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 4. Mobility / commute stress
    # -------------------------------------------------
    mobility_rows = df[
        df["Topic"].str.contains("Mobility status 1 year ago|Mobility status 5 years ago",
                                 case=False, na=False)
    ]
    mobility_flag = False
    for _, r in mobility_rows.iterrows():
        mv = _coerce_number(r[geo_col])
        row_char = str(r["Characteristic"]).lower()
        if mv and mv > 0 and ("moved" in row_char or "different" in row_char):
            mobility_flag = True
            break

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

    # -------------------------------------------------
    # 5. Languages spoken at home (non-English/French)
    # -------------------------------------------------
    sig_langs_all = extract_significant_languages(df, geo_col, pop_val_num)
    top_langs = _top_n(sig_langs_all, 2)  # take 2 strongest signals for messaging

    # -------------------------------------------------
    # 6. Indigenous Nations / Peoples via ethnic origin table
    # -------------------------------------------------
    # We use the wrapper build_indigenous_table() which returns df[Group,Count,Percent]
    nations_tbl = build_indigenous_table(df, geo_col, pop_val_num)
    nation_list_for_text = []
    if nations_tbl is not None and not nations_tbl.empty:
        # we will pick top 3 Groups by Count (table is already sorted desc by Count in build_indigenous_table)
        top_nations_rows = nations_tbl.head(3).to_dict("records")
        for row in top_nations_rows:
            g = str(row.get("Group", "")).strip()
            p = str(row.get("Percent", "")).strip() if row.get("Percent", None) else ""
            if p:
                nation_list_for_text.append(f"{g} ({p})")
            else:
                nation_list_for_text.append(g)

    # -------------------------------------------------
    # 7. Rough capacity modeling for planning
    # -------------------------------------------------
    # Seat planning heuristic:
    # - kids seats target = ~5–8% of kids_cnt (rounded up)
    # - teen seats target = ~5–8% of teens_cnt (rounded up), but floor it at 12 so we get a viable teen cohort
    def seat_range(count_val):
        if not count_val or count_val <= 0:
            return (None, None)
        low = int(round(count_val * 0.05))
        hi  = int(round(count_val * 0.08))
        # avoid zeros
        low = max(low, 10)
        hi  = max(hi, low)
        return (low, hi)

    kids_low, kids_hi = seat_range(kids_cnt if kids_cnt else 0)
    teen_low, teen_hi = seat_range(teens_cnt if teens_cnt else 0)

    # Pick a single teen target (center-ish)
    teen_target = None
    if teen_low and teen_hi:
        teen_target = max(12, int(round((teen_low + teen_hi) / 2.0)))

    # -------------------------------------------------
    # 8. Build narrative blocks
    # -------------------------------------------------

    # --- Snapshot / scale ---
    snapshot_lines = []
    snapshot_lines.append(
        f"{community_label} has an established resident base. "
        "The points below are meant to guide how we deliver programming here, not just describe statistics."
    )

    # Give numeric snapshot if we have it
    snap_bits = []
    if kids_cnt is not None and kids_cnt > 0:
        snap_bits.append(
            f"~{kids_cnt} children 0–14"
            + (f" ({_safe_pct(kids_pct)})" if kids_pct else "")
        )
    if teens_cnt is not None and teens_cnt > 0:
        snap_bits.append(
            f"~{teens_cnt} teens 15–19"
            + (f" ({_safe_pct(teens_pct)})" if teens_pct else "")
        )
    if seniors_cnt is not None and seniors_cnt > 0:
        snap_bits.append(
            f"~{seniors_cnt} seniors 65+"
            + (f" ({_safe_pct(seniors_pct)})" if seniors_pct else "")
        )

    if snap_bits:
        snapshot_lines.append(
            "Key age groups: " + "; ".join(snap_bits) + "."
        )

    if single_parent_share:
        snapshot_lines.append(
            f"Single-caregiver households are present (≈{single_parent_share:.1f}%)."
        )
    if renters_share or owners_share:
        renter_owner_txt = []
        if renters_share:
            renter_owner_txt.append(f"{renters_share:.1f}% renting")
        if owners_share:
            renter_owner_txt.append(f"{owners_share:.1f}% owning")
        if renter_owner_txt:
            snapshot_lines.append(
                "Housing mix: " + " / ".join(renter_owner_txt) + "."
            )
    if low_income_val:
        snapshot_lines.append(
            f"Low-income pressure shows up in census indicators (≈{low_income_val:.1f}%)."
        )

    if top_langs:
        snapshot_lines.append(
            "Families are using languages at home beyond English/French, notably " +
            ", ".join(top_langs) + "."
        )

    if nation_list_for_text:
        snapshot_lines.append(
            "Indigenous Nations / Peoples present in meaningful numbers include " +
            ", ".join(nation_list_for_text) + "."
        )

    snapshot_block = " ".join(snapshot_lines)

    # --- Youth & Caregivers ---
    youth_lines = []

    # kids/teens
    if kids_cnt:
        youth_lines.append(
            f"There are a lot of younger children here, which justifies recurring in-community programming instead of one-off outreach. "
            "This needs to look like 'every Tuesday/Thursday after school in the same place,' not a special event."
        )
    else:
        youth_lines.append(
            "There are meaningful numbers of younger children. We should plan for repeating programs, not pop-ins."
        )

    if teens_cnt:
        youth_lines.append(
            "There is also a visible 15–19 group. This is exactly the age when girls tend to drop out of structured activity. "
            "For that band, emotional safety and peer belonging have to come first, and 'performance' comes later if at all."
        )

    if seniors_cnt:
        youth_lines.append(
            "A notable 65+ population is also present. Grandparents are often the drivers, the sit-and-wait adults, "
            "and sometimes the default childcare at pickup. We should assume they're physically in the room or just outside it."
        )

    youth_block = " ".join(youth_lines)

    # --- Household Reality & Cost ---
    hh_lines = []

    if hh_size:
        hh_lines.append(
            f"Average household size is about {hh_size:.1f} people. "
            "In practice that means: if we register one child, two or three (siblings / cousins) may come. "
            "We should expect strollers, snacks, and kids hanging out on the side."
        )
    else:
        hh_lines.append(
            "Households commonly include multiple kids or extended family. Expect siblings and cousins to arrive together."
        )

    if single_parent_share:
        hh_lines.append(
            "Single caregivers are doing work, transport, meals, homework, and bedtime without backup. "
            "Programs that assume two parents can each drive a child to different places will fail. "
            "We need one location, predictable timing, and zero 'you must stay to supervise' requirements."
        )

    if renters_share and (not owners_share or renters_share > owners_share):
        hh_lines.append(
            "Renting is common, which usually means more moves and less scheduling stability. "
            "We should build for easy re-entry: if a family disappears for two weeks, they are still welcome back without penalty."
        )
    elif owners_share:
        hh_lines.append(
            "Owner-occupied housing is strong. That's helpful for continuity — one consistent site in town can actually hold a stable group over multiple months."
        )

    if low_income_val:
        hh_lines.append(
            "Cost is not a small barrier. We have to assume that even words like 'tuition', 'fee', 'uniform', or 'recital cost' will screen families out "
            "before they ever talk to us. Shoes and clothing must be provided, not just 'recommended'."
        )

    if mobility_flag:
        hh_lines.append(
            "Some families are newly arrived or recently moved. We cannot assume parents already know where to get help or who to talk to. "
            "We should act as if we’re introducing people to each other, not just to us."
        )

    hh_block = " ".join(hh_lines)

    # --- Language, Trust & Indigenous Partnerships ---
    lang_lines = []

    if top_langs:
        if len(top_langs) == 1:
            lang_lines.append(
                f"Parent communication cannot assume English-only. We should produce invites / consent in {top_langs[0]} as well as English. "
                "If the parent can't read the form, the child never shows up."
            )
        else:
            lang_lines.append(
                "Parent communication cannot assume English-only. We should produce invites and consent forms in " +
                " and ".join(top_langs) +
                " as well as English. If the parent can’t read the form, the child never shows up."
            )
    else:
        lang_lines.append(
            "Even if English looks dominant on paper, trust often sits in the first/home language. "
            "We should still check which languages school staff are actually using with caregivers."
        )

    if nation_list_for_text:
        lang_lines.append(
            "Indigenous partnership is not optional. Programs should be co-presented with local Indigenous leadership "
            "(not just 'welcomed to attend'), and Indigenous adults should be visibly in the room. "
            "That is what makes families feel like it’s theirs and not ours."
        )

    lang_block = " ".join(lang_lines)

    # --- Access / Schedule / Operations ---
    ops_lines = []

    # seats
    if kids_low and kids_hi:
        ops_lines.append(
            f"For younger kids, aim to hold between {kids_low} and {kids_hi} active seats across recurring weekly sessions. "
            "That scale is big enough to matter but still manageable with two facilitators."
        )
    else:
        ops_lines.append(
            "We should size kids' programming to a meaningful slice of the local child population (around 5–8%), "
            "not just a token pilot."
        )

    if teen_target:
        ops_lines.append(
            f"For teens (especially girls 15–19), build one dedicated block with ~{teen_target} seats. "
            "Keep it socially safe: low public performance pressure, zero extra uniform/gear cost."
        )

    # commute
    if long_commute_flag or car_commute_flag:
        ops_lines.append(
            "Location and timing: deliver in-town, ideally at (or beside) a school, right after school or early evening. "
            "If participation adds another drive across town after a 60+ minute work commute, attendance will collapse."
        )
    else:
        ops_lines.append(
            "Timing: after-school / early evening, in the closest possible school or hall, with predictable days (e.g., Tue/Thu every week). "
            "Predictability beats 'special events.'"
        )

    ops_lines.append(
        "Rooms need clear sightlines and a place for caregivers and siblings to sit. We should plan for stroller parking, snack/water table, "
        "and a gentle 'you can stay and watch' environment for grandparents."
    )

    ops_lines.append(
        "We also need a gear library on site (loaner shoes / clothing) and explicit 'no fee / no uniform / no recital cost' messaging."
    )

    ops_block = " ".join(ops_lines)

    # --- Risk / Mitigation / Early KPIs ---
    risk_lines = []
    risk_lines.append(
        "Main risks: cost stigma, commute fatigue, social safety for teen girls, and language disconnect with caregivers."
    )
    risk_lines.append(
        "Mitigations: zero-dollar entry; in-neighbourhood delivery; teen block built around belonging (not technique); "
        "and first contact through existing trust channels like school staff, settlement workers, FCSS, Indigenous leadership, or church/community connectors."
    )
    risk_lines.append(
        "90-day success test: (1) 60%+ of enrolled kids are still showing up in week 6; "
        "(2) teen girls are ≥40% of the teen block and ≥60% of them are still attending in week 6; "
        "(3) ≥75% of caregivers say they felt welcome/safe and would return next term."
    )
    kpi_block = " ".join(risk_lines)

    # -------------------------------------------------
    # Final assembly
    # -------------------------------------------------
    final_sections = [
        f"COMMUNITY PROFILE SUMMARY — {community_label.upper()}",
        "SNAPSHOT & SCALE: " + snapshot_block,
        "YOUTH & CAREGIVERS: " + youth_block,
        "HOUSEHOLD REALITY & COST: " + hh_block,
        "LANGUAGE, TRUST & INDIGENOUS PARTNERSHIPS: " + lang_block,
        "ACCESS / DELIVERY MODEL: " + ops_block,
        "RISK, MITIGATION & 90-DAY KPIs: " + kpi_block,
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

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import pandas as pd
from bs4 import BeautifulSoup

def create_full_pdf(summary_text: str, place_name: str, cleaned_df: pd.DataFrame) -> bytes:
    """
    Build a single PDF that includes:
    1. Title + place_name
    2. Narrative summary (multi-paragraph, human-readable)
    3. All topic tables (Characteristic + values) with zero-only rows already removed
    """

    # --- Prep styles ---
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="HeadingPlace",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="BodyTextTight",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
        spaceAfter=6,
    ))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    story = []

    # --- 1. Title ---
    safe_place = place_name or "Community Profile"
    story.append(Paragraph(f"Community Profile Summary for {safe_place}", styles["HeadingPlace"]))
    story.append(Spacer(1, 6))

    # --- 2. Narrative summary text ---
    # summary_text currently comes back as multiple sections separated by double newlines.
    # We'll split by blank lines and make each paragraph its own <Paragraph>.
    for block in summary_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        story.append(Paragraph(block, styles["BodyTextTight"]))
    story.append(PageBreak())

    # --- 3. Data tables by topic ---
    # We'll mirror what render_report() shows: topic sections, rows without all-zero values.
    topic_col = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"

    # figure out the value columns we actually care about
    value_cols = [
        c for c in cleaned_df.columns
        if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")
    ]

    def row_has_nonzero_data_pdf(row: pd.Series) -> bool:
        for c in value_cols:
            if c not in row:
                continue
            num = _coerce_number(row[c])
            if num is not None and num > 0:
                return True
        return False

    # group and render
    for topic, sub in cleaned_df.groupby(topic_col, dropna=False):
        # filter out zero-only rows (same rule as the app UI / CSV export)
        rows_keep = [r for _, r in sub.iterrows() if row_has_nonzero_data_pdf(r)]
        if not rows_keep:
            continue

        pretty_df = pd.DataFrame(rows_keep)[["Characteristic"] + value_cols].reset_index(drop=True)

        # topic header
        story.append(Paragraph(str(topic), styles["SectionHeader"]))

        # build table data: header row + each row’s cells
        table_data = [pretty_df.columns.tolist()]
        for _, r in pretty_df.iterrows():
            table_data.append([str(r[c]) if pd.notna(r[c]) else "" for c in pretty_df.columns])

        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f5f5f5")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.black),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0), 8),

            ("FONTSIZE",    (0,1), (-1,-1), 8),
            ("LEADING",     (0,1), (-1,-1), 10),

            ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#999999")),
            ("ALIGN",       (1,1), (-1,-1), "RIGHT"),  # keep numbers right-aligned
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ]))

        story.append(t)
        story.append(Spacer(1, 12))

    # build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


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
    # ---- PDF Download Button ----
    if summary_text:
        pdf_bytes = create_full_pdf(
            summary_text=summary_text,
            place_name=place_guess or "Community Profile",
            cleaned_df=cleaned_df,
        )
    
        st.download_button(
            label="📄 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"{(place_guess or 'community_profile').replace(' ', '_').lower()}_report.pdf",
            mime="application/pdf",
        )

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
