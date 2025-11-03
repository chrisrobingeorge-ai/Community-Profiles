from __future__ import annotations  # first line

import io, os, re
from datetime import date, datetime
from collections import OrderedDict

import pandas as pd
import streamlit as st  # <-- must be above st.set_page_config

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT   # 👈 add this line
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

# Sometimes these aren't in the Topic column, but appear in Characteristic instead
TARGET_CHARACTERISTIC_KEYWORDS = [
    "Highest certificate, diploma or degree",
    "Commuting duration",
]

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def load_statcan_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]
    if "Topic" in df.columns:
        df["Topic"] = df["Topic"].astype(str).str.strip()
    if "Characteristic" in df.columns:
        df["Characteristic"] = df["Characteristic"].astype(str).str.strip()
    df = df.dropna(how="all")
    if "Topic" in df.columns:
        df = df[df["Topic"] != "Topic"]
    return df


def _characteristic_sort_key(label: str) -> tuple:
    if label is None:
        return (9999, 9999999, 9999999, "")
    text = str(label).strip()
    low = text.lower()

    income_range_patterns = [
        r"^\$?([\d,]+)\s*to\s*\$?([\d,]+)",
        r"^under\s*\$?([\d,]+)",
        r"^\$?([\d,]+)\s*(and over|\+)",
    ]
    for patt in income_range_patterns:
        m = re.match(patt, text, flags=re.IGNORECASE)
        if m:
            n1_raw = m.group(1)
            n1 = int(n1_raw.replace(",", "")) if n1_raw else 0
            n2 = n1
            m_to = re.match(r"^\$?([\d,]+)\s*to\s*\$?([\d,]+)", text, flags=re.IGNORECASE)
            if m_to:
                n1 = int(m_to.group(1).replace(",", ""))
                n2 = int(m_to.group(2).replace(",", ""))
            m_under = re.match(r"^under\s*\$?([\d,]+)", text, flags=re.IGNORECASE)
            if m_under:
                n1, n2 = 0, int(m_under.group(1).replace(",", ""))
            m_over = re.match(r"^\$?([\d,]+)\s*(and over|\+)", text, flags=re.IGNORECASE)
            if m_over:
                n1 = int(m_over.group(1).replace(",", ""))
                n2 = n1 + 999999
            return (0, n1, n2, low)

    if low.startswith("total -"):
        return (1, 0, 0, low)
    if low.startswith("average ") or low.startswith("median "):
        return (2, 0, 0, low)

    nums = re.findall(r"\d+", text)
    if nums:
        first_num = int(nums[0])
        second_num = int(nums[1]) if len(nums) > 1 else first_num
        return (0, first_num, second_num, low)

    m_over_generic = re.match(r"^\s*(\d+).*(and over|or more)", text, flags=re.IGNORECASE)
    if m_over_generic:
        n = int(m_over_generic.group(1))
        return (0, n, n + 1000, low)

    return (1, 9999998, 9999998, low)


def resolve_topic_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["Topic_norm"] = df.get("Topic", "")
        return df
    df = df.copy()

    desc_mask = df["Characteristic"].str.contains(r"\brefers to\b", case=False, na=False)
    value_cols_tmp = [c for c in df.columns if c not in ("Topic", "Characteristic", "Notes", "Note", "Symbol", "Flags", "Flag")]
    has_data = df[value_cols_tmp].applymap(lambda x: str(x).strip()).apply(
        lambda row: any(v not in ["", "None", "nan", "NaN", "F", "X", "..", "..."] for v in row), axis=1
    )
    df = df[~(desc_mask & ~has_data)].copy()

    df["Topic_norm"] = df.get("Topic", "").astype(str).str.strip()
    current_topic, rows_to_drop = None, []

    def looks_numeric_topic(t: str) -> bool:
        t = (t or "").strip()
        return bool(re.fullmatch(r"\d{1,3}", t))

    for idx, row in df.iterrows():
        topic = str(row.get("Topic", "")).strip()
        char = str(row.get("Characteristic", "")).strip()

        if topic and not looks_numeric_topic(topic) and topic.lower() not in ["topic", ""]:
            current_topic = topic
            df.at[idx, "Topic_norm"] = current_topic
            continue

        m = re.match(r"^\s*(\d{1,3})\s*:\s*(.+)$", char)
        if m:
            label = m.group(2).strip()
            label = re.split(r"\s+refers to\s+", label, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            current_topic = label if label else char
            df.at[idx, "Topic_norm"] = current_topic

            value_cols2 = [c for c in df.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]
            maybe_desc = ("refers to" in char.lower())
            if maybe_desc:
                rows_to_drop.append(idx)
            else:
                placeholders = {"", "..", "...", "F", "X"}
                col_vals = []
                for c2 in value_cols2:
                    v = df.at[idx, c2] if c2 in df.columns else None
                    s = "" if pd.isna(v) else str(v).strip()
                    col_vals.append(s)
                if all((v2 == "" or v2.upper() in placeholders) for v2 in col_vals):
                    rows_to_drop.append(idx)
            continue

        if current_topic:
            df.at[idx, "Topic_norm"] = current_topic
        else:
            df.at[idx, "Topic_norm"] = topic

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

    df = resolve_topic_column(df)
    topic_mask = df["Topic_norm"].str.lower().isin([t.lower() for t in TARGET_TOPICS])

    char_mask = False
    for kw in TARGET_CHARACTERISTIC_KEYWORDS:
        char_mask = char_mask | df["Characteristic"].str.lower().str.contains(kw.lower(), na=False)

    keep_mask = topic_mask | char_mask
    filtered = df[keep_mask].copy()
    filtered["__char_sort_key__"] = filtered["Characteristic"].apply(_characteristic_sort_key)
    filtered.sort_values(by=["Topic_norm", "__char_sort_key__"], inplace=True, ignore_index=True)
    filtered.drop(columns=["__char_sort_key__"], inplace=True, errors="ignore")
    return filtered


def _coerce_number(val):
    if pd.isna(val):
        return None
    text = str(val).strip()
    if text in ["", "..", "...", "F", "X"]:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def row_has_nonzero_data(row: pd.Series, value_cols: list[str]) -> bool:
    for c in value_cols:
        if c not in row:
            continue
        num = _coerce_number(row[c])
        if num is not None and num > 0:
            return True
    return False


def render_report(df: pd.DataFrame):
    if df.empty:
        st.warning("No matching rows found in this CSV for the selected fields.")
        return

    value_cols = [c for c in df.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]
    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"

    HIDE_THESE_TOPICS = {
        "Mobility status 1 year ago",
        "Mobility status 5 years ago",
        "Selected places of birth for the recent immigrant population",
    }

    for topic, sub in df.groupby(topic_col, dropna=False):
        if topic in HIDE_THESE_TOPICS:
            continue

        filtered_rows = []
        for _, r in sub.iterrows():
            if row_has_nonzero_data(r, value_cols):
                filtered_rows.append(r)

        if not filtered_rows:
            continue

        pretty_df = pd.DataFrame(filtered_rows)[["Characteristic"] + value_cols].reset_index(drop=True)
        with st.expander(f"📂 {topic}", expanded=False):
            st.dataframe(pretty_df, use_container_width=True)


def drop_zero_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    value_cols = [c for c in df.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]
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
    cleaned = drop_zero_only_rows(df)
    if cleaned.empty:
        return "<html><body><h1>No data</h1></body></html>"

    styles = """
    <style>
    body { font-family: sans-serif; margin: 2rem; }
    h1 { font-size: 1.4rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #999; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    th, td { border: 1px solid #ccc; padding: 0.4rem; font-size: 0.9rem; text-align: left; }
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

    value_cols = [c for c in cleaned.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]
    topic_col = "Topic_norm" if "Topic_norm" in cleaned.columns else "Topic"

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
    Try to guess which column in df is the actual geography data column.
    Prefer 'Combined' if present (multi-file rollup), else fall back to
    the most numeric-looking column.
    """
    # --- NEW: prefer the rollup if present ---
    if "Combined" in df.columns:
        return "Combined"

    # --- original logic ---
    exclude = {"Topic", "Characteristic", "Notes", "Note", "Symbol", "Flags", "Flag", "Topic_norm"}
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
    sub = df.copy()
    if topic_regex:
        sub = sub[sub["Topic"].str.contains(topic_regex, case=False, na=False)]
    if char_regex:
        sub = sub[sub["Characteristic"].str.contains(char_regex, case=False, na=False)]
    for _, r in sub.iterrows():
        num = _coerce_number(r[geo_col])
        if num is None or num <= 0:
            continue
        if min_pct is not None and num < min_pct:
            continue
        return num
    return None


def extract_significant_languages(df, geo_col, pop_val_num):
    LANGUAGE_TOPIC_REGEX = (
        "Mother tongue|Language spoken most often at home|Other language spoken regularly at home"
    )
    block_terms = [
        "total","official","non-official","non official","single responses","multiple responses",
        "indo-european","germanic","balto-slavic","slavic","germanic languages","germanic language",
        "language family","not included elsewhere","other languages","languages not included elsewhere",
        "aboriginal languages","indigenous languages",
    ]
    ignore_exact = ["english","french","english and french"]
    MIN_PCT, MIN_COUNT = 1.0, 50
    if pop_val_num is None or pop_val_num <= 0:
        pop_val_num = None

    lang_rows = df[df["Topic"].str.contains(LANGUAGE_TOPIC_REGEX, case=False, na=False)].copy()
    significant = []

    for _, r in lang_rows.iterrows():
        raw_label = str(r["Characteristic"]).strip()
        val_num = _coerce_number(r[geo_col])
        if val_num is None or val_num <= 0:
            continue
        low_label = raw_label.lower()
        if any(bt in low_label for bt in block_terms):
            continue
        if any(low_label == ig for ig in ignore_exact):
            continue
        if any(low_label.startswith(ig) for ig in ignore_exact):
            continue
        if "english" in low_label and "french" not in low_label and len(low_label.split()) <= 3:
            continue
        if "french" in low_label and "english" not in low_label and len(low_label.split()) <= 3:
            continue
        clean_label = raw_label.replace(" (Filipino)", "").replace(" (Panjabi)", "").replace(" languages", "").replace(" language", "").strip()
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

    seen, ordered_unique = set(), []
    for lang in significant:
        key = lang.lower()
        if key not in seen:
            seen.add(key)
            ordered_unique.append(lang)
    return ordered_unique


def derive_indigenous_from_ethnic_origin(df: pd.DataFrame, geo_col: str, pop_val_num: float | None = None) -> pd.DataFrame:
    if df.empty or not geo_col:
        return pd.DataFrame(columns=["Group", "Count", "Percent"])

    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"
    ETHNIC_TOPIC_REGEX = r"(Ethnic origin|Ethnic or cultural origin|Origine ethnique ou culturelle)"
    sub = df[df[topic_col].str.contains(ETHNIC_TOPIC_REGEX, case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Group", "Count", "Percent"])

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
                break

    rows = []
    for group, count in tallies.items():
        if count > 0:
            pct = (count / pop_val_num * 100.0) if (pop_val_num and pop_val_num > 0) else None
            rows.append({"Group": group, "Count": int(round(count)), "Percent": (f"{pct:.1f}%" if pct is not None else None)})

    out = pd.DataFrame(rows, columns=["Group", "Count", "Percent"])
    if not out.empty:
        out = out.sort_values(by="Count", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def build_indigenous_table(df: pd.DataFrame, geo_col: str, pop_val_num: float | None):
    try:
        tbl = derive_indigenous_from_ethnic_origin(df, geo_col, pop_val_num)
        if tbl is None:
            return pd.DataFrame(columns=["Group", "Count", "Percent"])
        return tbl
    except Exception:
        return pd.DataFrame(columns=["Group", "Count", "Percent"])


def summarize_indigenous_nations_from_ethnic(df: pd.DataFrame) -> list[str]:
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

    groups = nations_df["Group"].astype(str).tolist()
    seen, ordered = set(), []
    for g in groups:
        low = g.lower().strip()
        if low not in seen:
            seen.add(low)
            ordered.append(g)
    return ordered


# --- Age band logic and cohort-aging ---
CENSUS_REFERENCE_DATE = date(2021, 5, 11)

AGE_BANDS_ORDER = [
    "0 to 4 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years","25 to 29 years",
    "30 to 34 years","35 to 39 years","40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years",
    "60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over",
]


def _find_age_value(df, geo_col, band_label):
    """
    Return the *count* for the age band. Ignore percent/proportion rows.
    Fallback: if multiple count rows remain, take the largest numeric.
    """
    # Narrow to Age topic
    sub = df[df["Topic"].str.contains(r"Age characteristics", case=False, na=False)].copy()

    # Candidate label matches (exact first, then contains)
    exact = sub[sub["Characteristic"].str.fullmatch(rf"\s*{band_label}\s*", case=False, na=False)]
    cand = exact if not exact.empty else sub[sub["Characteristic"].str.contains(rf"{band_label}", case=False, na=False)]

    if cand.empty:
        return 0.0

    # Strictly reject percent-like rows
    def looks_percent(s: str) -> bool:
        s = s.lower()
        return (
            "(%)" in s or " percent" in s or "percentage" in s or
            "% of" in s or "proportion" in s
        )

    cand = cand[~cand["Characteristic"].astype(str).apply(looks_percent)]
    if cand.empty:
        return 0.0

    # Collect numeric candidates from geo_col
    vals = []
    for _, r in cand.iterrows():
        v = _coerce_number(r.get(geo_col))
        if v is not None and v > 0:
            vals.append(float(v))

    if not vals:
        return 0.0

    # If multiple count-like matches, take the largest (counts >> any residual small figures)
    return max(vals)

def extract_age_bands(df, geo_col) -> OrderedDict:
    bands = OrderedDict()
    for lab in AGE_BANDS_ORDER:
        bands[lab] = float(_find_age_value(df, geo_col, lab))
    return bands


def _shift_once_one_year(bands: OrderedDict) -> OrderedDict:
    out = OrderedDict((k, 0.0) for k in bands.keys())
    keys = list(bands.keys())
    for i, k in enumerate(keys):
        v = bands[k]
        if k == "85 years and over":
            out[k] += v
        else:
            stay = v * 4.0 / 5.0
            move = v * 1.0 / 5.0
            out[k] += stay
            nxt = keys[i + 1]
            out[nxt] += move
    return out


def age_bands_adjust_to_date(bands: OrderedDict, as_of: date) -> OrderedDict:
    delta_years = max(0.0, (as_of - CENSUS_REFERENCE_DATE).days / 365.25)
    if delta_years == 0:
        return bands
    y = int(delta_years)
    frac = delta_years - y
    cur = bands.copy()
    for _ in range(y):
        cur = _shift_once_one_year(cur)
    if frac > 0:
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
    wanted_labels = ["0 to 4 years","5 to 9 years","10 to 14 years","15 to 19 years"]
    if as_of_date is not None:
        bands_2021 = extract_age_bands(df, geo_col)
        bands_adj = age_bands_adjust_to_date(bands_2021, as_of=as_of_date)
        return {lab: float(bands_adj.get(lab, 0.0)) for lab in wanted_labels}

    result = {}
    for lab in wanted_labels:
        row = df[
            (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
            (df["Characteristic"].str.fullmatch(rf"\s*{lab}\s*", case=False, na=False))
        ]
        if row.empty:
            row = df[
                (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
                (df["Characteristic"].str.contains(rf"{lab}", case=False, na=False))
            ]
        result[lab] = float(_coerce_number(row.iloc[0][geo_col]) or 0.0) if not row.empty else 0.0
    return result


def extract_place_name(upload_name: str) -> str:
    stem = os.path.splitext(upload_name)[0]
    parts = re.split(r"[-_]", stem)
    candidate = None
    for p in reversed(parts):
        if re.search(r"[A-Za-z]", p):
            candidate = p
            break
    if candidate is None:
        candidate = stem
    if re.match(r"(?i)censusprofile|profilrecensement", candidate):
        candidate = stem
    friendly = re.sub(r"(?<!^)([A-Z0-9])", r" \1", candidate).strip()
    friendly = re.sub(r"\s+", " ", friendly)
    friendly = friendly.strip()
    if friendly:
        friendly = friendly[0].upper() + friendly[1:]
    return friendly


def _percent_or_none(val, pop_val_num):
    if val is None or val <= 0:
        return (None, None)
    if pop_val_num and pop_val_num > 0:
        pct = (val / pop_val_num) * 100.0
        return (pct, int(round(val)))
    else:
        return (None, int(round(val)))


def _top_n(seq, n):
    out, seen = [], set()
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
    if x is None:
        return None
    try:
        return f"{x:.1f}%"
    except Exception:
        return None
        
def generate_summary(df: pd.DataFrame, as_of_date: date | None = None, place_name: str | None = None) -> str:
    if df.empty:
        return "No summary available."

    geo_col = pick_geo_col(df)
    if not geo_col:
        return "No summary available."

    community_label = place_name or "this community"
    pop_val_num = _get_population(df, geo_col)

    # Age bands and basic counts
    kids_pct, kids_cnt, seniors_pct, teens_cnt = _get_age_markers(df, geo_col, pop_val_num, as_of_date)
    hh_size = _best_numeric_from(df, "Household and dwelling characteristics", "Average household size", geo_col)
    single_parent_ratio = _best_numeric_from(df, "Household.*", "single-parent|one-parent", geo_col, min_pct=1.0)

    # Income and renters
    lim_at_rate = _best_numeric_from(df, "Low income.*", None, geo_col, min_pct=1.0)
    if lim_at_rate and lim_at_rate > 100:
        lim_at_rate = 100.0
    renters_pct = _best_numeric_from(df, "Household.*", "rented", geo_col, min_pct=1.0)

    # Language and newcomer patterns
    sig_langs_all = extract_significant_languages(df, geo_col, pop_val_num)
    if not isinstance(sig_langs_all, dict):
        sig_langs_all = {}
    top_langs = _top_n(sig_langs_all, 2)
    largest_non_official_language = top_langs[0] if top_langs else None
    largest_non_official_language_percent = sig_langs_all.get(largest_non_official_language, 0.0) if largest_non_official_language else 0.0
    newcomer_percent = sum([v for k, v in sig_langs_all.items() if v >= 1.0])

    # Indigenous
    nations_tbl = build_indigenous_table(df, geo_col, pop_val_num)
    indigenous_percent = 0.0
    if nations_tbl is not None and not nations_tbl.empty:
        for row in nations_tbl.to_dict("records"):
            p = str(row.get("Percent", "")).replace("%", "")
            try: indigenous_percent += float(p)
            except: continue

    # Commute
    commute_rows = df[df["Topic"].str.contains("Commuting", case=False, na=False)]
    long_commute = False
    for _, row in commute_rows.iterrows():
        if "60 minutes" in row["Characteristic"].lower():
            val = _coerce_number(row[geo_col])
            if val and val > 5:
                long_commute = True
                break

    summary_parts = []

    # Youth
    if kids_pct and kids_pct > 20:
        summary_parts.append("Youth density is strong. Programs should run weekly at predictable times and places—avoid one-offs.")
    elif kids_pct:
        summary_parts.append("Some younger children are present—consistent visibility helps build familiarity and trust.")

    # Household structure
    if hh_size and hh_size >= 3.5:
        summary_parts.append("Households are large. Expect strollers, cousins, siblings arriving together.")
    if single_parent_ratio and single_parent_ratio >= 5:
        summary_parts.append("Many single caregivers. No assumptions about supervision—convenience is key.")

    # Income
    if lim_at_rate and lim_at_rate >= 20:
        summary_parts.append(f"Low-income share is high (≈{lim_at_rate:.1f}%). Cost-free programming and gear access are essential.")
    elif lim_at_rate and lim_at_rate >= 10:
        summary_parts.append(f"Some income pressure detected (≈{lim_at_rate:.1f}%). Consider tiered pricing or donation-supported models.")

    # Language
    if largest_non_official_language and largest_non_official_language_percent >= 5:
        summary_parts.append(f"Families speak languages beyond English, notably {largest_non_official_language} (≈{largest_non_official_language_percent:.1f}%). Translate outreach where possible.")
    elif top_langs:
        summary_parts.append("Linguistic diversity exists. Home languages matter for trust—check with schools about parent communications.")

    # Indigenous
    if indigenous_percent >= 10:
        summary_parts.append(f"Indigenous identity is significant (≈{indigenous_percent:.1f}%). Powwow-based programming and Indigenous co-leadership are important.")
    elif indigenous_percent >= 3:
        summary_parts.append(f"Indigenous presence is modest (≈{indigenous_percent:.1f}%) but meaningful—co-delivery recommended.")
    else:
        summary_parts.append("Even with limited Indigenous data, culturally respectful practices remain vital.")

    # Commute & geography
    if long_commute:
        summary_parts.append("Long commute times suggest programs should be delivered in-neighbourhood, ideally after school.")

    # Rural
    if pop_val_num and pop_val_num < 10000:
        summary_parts.append("This is a small rural community. Hub-and-spoke or satellite delivery is recommended.")

    # EDIA
    equity_index = sum([
        (indigenous_percent or 0.0),
        (lim_at_rate or 0.0),
        (renters_pct or 0.0),
        (newcomer_percent or 0.0)
    ])
    if equity_index >= 60:
        summary_parts.append("Multiple equity flags suggest eligibility for EDIA-focused outreach grants and tailored program supports.")

    return " ".join(summary_parts)


# Helper used inside this version:
def _get_population(df, geo_col):
    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    return _coerce_number(pop_rows.iloc[0][geo_col]) if not pop_rows.empty else None

def _get_age_markers(df, geo_col, pop_val, as_of_date):
    labels_kids = ["0 to 4 years", "5 to 9 years", "10 to 14 years"]
    labels_teens = ["15 to 19 years"]
    labels_seniors = ["65 to 69 years", "70 to 74 years", "75 to 79 years", "80 to 84 years", "85 years and over"]
    if as_of_date:
        bands = extract_age_bands(df, geo_col)
        adj = age_bands_adjust_to_date(bands, as_of=as_of_date)
        kids = sum_bands(adj, labels_kids)
        teens = sum_bands(adj, labels_teens)
        seniors = sum_bands(adj, labels_seniors)
    else:
        kids = _grab_sum_for(df, geo_col, labels_kids)
        teens = _grab_sum_for(df, geo_col, labels_teens)
        seniors = _grab_sum_for(df, geo_col, labels_seniors)
    kids_pct, kids_cnt = _percent_or_none(kids, pop_val)
    seniors_pct, _ = _percent_or_none(seniors, pop_val)
    _, teens_cnt = _percent_or_none(teens, pop_val)
    return kids_pct, kids_cnt, seniors_pct, teens_cnt

def _grab_sum_for(df, geo_col, labels):
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
    
def prune_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    KEEP_ALWAYS = {"Topic", "Characteristic"}

    flag_like = []
    for c in df.columns:
        cl = c.lower()
        if c in KEEP_ALWAYS:
            continue
        if (cl.endswith("_flag") or "flag" in cl or "symbol" in cl or
            (cl.startswith("note") or cl == "note" or "notes" in cl) or
            "quality" in cl or "status" in cl):
            flag_like.append(c)
    df2 = df.drop(columns=flag_like, errors="ignore")

    PLACEHOLDERS = {"", "..", "...", "f", "x"}
    drop_empty = []
    for c in df2.columns:
        if c in KEEP_ALWAYS:
            continue
        as_str = df2[c].astype(str).str.strip().str.lower()
        empties = as_str.isna() | as_str.isin(PLACEHOLDERS)
        if empties.all():
            drop_empty.append(c)
    df2 = df2.drop(columns=drop_empty, errors="ignore")

    dup_like = []
    for c in df2.columns:
        m = re.match(r"^(.*)\.(\d+)$", c)
        if not m:
            continue
        base = m.group(1)
        if base in df2.columns:
            if df2[c].equals(df2[base]):
                dup_like.append(c)
            else:
                as_str = df2[c].astype(str).str.strip().str.lower()
                if as_str.isin(PLACEHOLDERS).all():
                    dup_like.append(c)
    df2 = df2.drop(columns=dup_like, errors="ignore")
    return df2


def collapse_duplicate_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    topic_col = "Topic_norm" if "Topic_norm" in df.columns else "Topic"
    seen, keep_rows = set(), []
    for idx, row in df.iterrows():
        key = (str(row.get(topic_col, "")).strip(), str(row.get("Characteristic", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        keep_rows.append(idx)
    out = df.loc[keep_rows].copy().reset_index(drop=True)
    return out

def _calc_col_widths(avail_w: float, n_values: int) -> list[float]:
    """
    Return [char_w] + [num_w]*n_values that fits within avail_w.
    - numeric columns get a compact, equal width
    - characteristic column gets the rest (and can wrap)
    """
    # Guard rails / heuristics in points
    MIN_CHAR = 120.0   # Characteristic must get at least this
    MAX_CHAR = avail_w * 0.75
    # Start with a reasonable numeric width
    num_w = 56.0
    if n_values > 8:    num_w = 48.0
    if n_values > 10:   num_w = 42.0
    if n_values > 12:   num_w = 38.0

    total_num = num_w * n_values
    char_w = max(MIN_CHAR, min(MAX_CHAR, avail_w - total_num))

    # If still too tight, squeeze numerics but never below 34pt
    if char_w + total_num > avail_w:
        num_w = max(34.0, (avail_w - MIN_CHAR) / max(1, n_values))
        total_num = num_w * n_values
        char_w = max(MIN_CHAR, avail_w - total_num)

    return [char_w] + [num_w] * n_values


def _make_wrapped_table(table_df: pd.DataFrame, styles, base_font: int, avail_w: float):
    """
    Build a Table with wrapped text in col 0 and right-aligned numerics in others.
    Returns the (Table, effective_font_size).
    """
    # Styles for cells
    p_body = ParagraphStyle(
        "tbl-body", parent=styles["BodyText"], fontSize=base_font, leading=base_font + 2, alignment=TA_LEFT
    )
    p_num = ParagraphStyle(
        "tbl-num", parent=styles["BodyText"], fontSize=base_font, leading=base_font + 2, alignment=TA_RIGHT
    )
    p_head = ParagraphStyle(
        "tbl-head", parent=styles["BodyText"], fontSize=base_font, leading=base_font + 2
    )

    # Build data with Paragraphs (needed for word wrap)
    headers = table_df.columns.tolist()
    head_row = [Paragraph(str(h), p_head) for h in headers]

    data_rows = []
    for _, r in table_df.iterrows():
        row_cells = []
        for j, c in enumerate(headers):
            val = "" if pd.isna(r[c]) else str(r[c])
            if j == 0:
                row_cells.append(Paragraph(val, p_body))  # Characteristic (wrap)
            else:
                row_cells.append(Paragraph(val, p_num))   # numeric-ish (right)
        data_rows.append(row_cells)

    # Compute widths to fit
    n_values = len(headers) - 1
    col_widths = _calc_col_widths(avail_w, n_values)

    # Construct table
    t = Table([head_row] + data_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.black),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#999999")),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("WORDWRAP",    (0,0), (-1,-1), True),
        # Ensure numeric columns render as a block (helps right-align Paragraphs)
        ("ALIGN",       (1,1), (-1,-1), "RIGHT"),
    ]))
    return t

def create_full_pdf(summary_text: str, place_name: str, cleaned_df: pd.DataFrame) -> bytes:
    """
    Auto-fits tables to the page by wrapping text, sizing columns, and
    switching to landscape when many columns are present.
    """
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HeadingPlace", parent=styles["Heading1"], fontSize=16, leading=20, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHeader", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=12, spaceAfter=6, textColor=colors.black))
    styles.add(ParagraphStyle(name="BodyTextTight", parent=styles["BodyText"], fontSize=10, leading=13, spaceAfter=6))

    # Work out column count once (all tables share same schema after cleaning)
    topic_col = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"
    value_cols = [c for c in cleaned_df.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]
    total_cols = 1 + len(value_cols)

    # Pick orientation based on width pressure
    # Portrait LETTER usable width ≈ 612-72 = 540pt; Landscape gives ~ 720pt usable.
    use_landscape = len(value_cols) >= 8  # heuristic threshold
    pagesize = landscape(LETTER) if use_landscape else LETTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
    )

    # Available width for tables
    avail_w = doc.pagesize[0] - doc.leftMargin - doc.rightMargin

    # Base font: smaller for very wide tables
    base_font = 8
    if len(value_cols) >= 9:
        base_font = 7
    if len(value_cols) >= 12:
        base_font = 6

    story = []

    # Title
    safe_place = place_name or "Community Profile"
    story.append(Paragraph(f"Community Profile Summary for {safe_place}", styles["HeadingPlace"]))
    story.append(Spacer(1, 6))

    # Narrative
    for block in summary_text.split("\n\n"):
        block = block.strip()
        if block:
            story.append(Paragraph(block, styles["BodyTextTight"]))
    story.append(PageBreak())

    # Group tables by topic
    def row_has_nonzero_data_pdf(row: pd.Series) -> bool:
        for c in value_cols:
            if c not in row:
                continue
            num = _coerce_number(row[c])
            if num is not None and num > 0:
                return True
        return False

    for topic, sub in cleaned_df.groupby(topic_col, dropna=False):
        rows_keep = [r for _, r in sub.iterrows() if row_has_nonzero_data_pdf(r)]
        if not rows_keep:
            continue

        pretty_df = pd.DataFrame(rows_keep)[["Characteristic"] + value_cols].reset_index(drop=True)

        # Topic header
        story.append(Paragraph(str(topic), styles["SectionHeader"]))

        # Table with wrapped content and fitted widths
        t = _make_wrapped_table(pretty_df, styles, base_font, avail_w)
        story.append(t)
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def build_chatgpt_prompt(
    summary_text: str,
    cleaned_df: pd.DataFrame,
    place_name: str | None,
    as_of_date: date | None,
) -> tuple[str, str]:
    """
    Returns (prompt_text_for_copy, full_prompt_txt_download).
    The 'copy' version trims the table if very long; the 'download' version includes all rows.
    """
    # Identify value column (the geo column)
    geo_col = pick_geo_col(cleaned_df)
    topic_col = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"
    place = place_name or "Community"

    # Population (best-effort)
    pop_rows = cleaned_df[
        (cleaned_df[topic_col].str.contains("Population and dwellings", case=False, na=False)) &
        (cleaned_df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col]) if (geo_col and not pop_rows.empty) else None

    # Top languages & Indigenous groups (re-use your logic)
    top_langs = _top_n(extract_significant_languages(cleaned_df, geo_col, pop_val_num), 3) if geo_col else []
    indig_tbl = build_indigenous_table(cleaned_df, geo_col, pop_val_num) if geo_col else pd.DataFrame()
    indig_list = indig_tbl["Group"].head(4).tolist() if (indig_tbl is not None and not indig_tbl.empty) else []

    # Compact table: Topic, Characteristic, Value
    trimmed = drop_zero_only_rows(cleaned_df)
    cols = [c for c in [topic_col, "Characteristic", geo_col] if c]
    compact = trimmed[cols].rename(columns={topic_col: "Topic", geo_col: "Value"}) if cols else pd.DataFrame()

    # Limit paste size to keep it friendly
    MAX_ROWS_FOR_COPY = 300
    compact_for_copy = compact.head(MAX_ROWS_FOR_COPY) if len(compact) > MAX_ROWS_FOR_COPY else compact

    # Build CSV text (no index, UTF-8, simple)
    csv_copy = compact_for_copy.to_csv(index=False)
    csv_full = compact.to_csv(index=False)

    # Format the instruction prompt
    header_lines = []
    header_lines.append(f"Community: {place}")
    if as_of_date:
        header_lines.append(f"As-of date for age adjustments: {as_of_date.isoformat()}")
    if pop_val_num:
        header_lines.append(f"Population (2021): {int(pop_val_num)}")
    if top_langs:
        header_lines.append("Notable home languages (non-English/French): " + ", ".join(top_langs))
    if indig_list:
        header_lines.append("Indigenous Nations/Peoples (from ethnic origin table): " + ", ".join(indig_list))

    instructions = (
        "You are a community program designer and evaluator. Using the narrative summary and the table below, "
        "produce a deeper analysis and program impact assessment for Alberta Ballet’s ‘Growing Up Strong’ campaign. "
        "Deliver:\n"
        "1) Key insights about children/teens, caregivers, cost barriers, mobility/commute, and language/ trust.\n"
        "2) Recommended delivery model (sites, schedule, seat targets by age band), including rationale tied to the data.\n"
        "3) Equity considerations (single-caregiver households, renters, low-income indicators), with concrete mitigations.\n"
        "4) Indigenous partnership approach (co-presentation/roles), language access steps, and caregiver engagement tactics.\n"
        "5) A 90-day learning plan with 3–5 measurable indicators aligned to Growing Up Strong pillars "
        "(Access to Tickets; Professional Training School; Recreational Classes; Community Programs).\n"
        "6) Risks & mitigations, plus assumptions and open questions to validate with local partners.\n\n"
        "Use short paragraphs and bullet points where helpful. Keep it practical and decision-ready."
    )

    # Assemble copy-prompt (trimmed CSV)
    prompt_copy = (
        "=== CONTEXT ===\n"
        + "\n".join(header_lines) + "\n\n"
        + "=== NARRATIVE SUMMARY ===\n"
        + summary_text.strip() + "\n\n"
        + "=== INSTRUCTIONS ===\n"
        + instructions + "\n\n"
        + "=== DATA (CSV) ===\n"
        + csv_copy
        + ("\n\n[Note: table truncated for copy. Full table is in the attached/downloaded TXT.]" if len(compact) > MAX_ROWS_FOR_COPY else "")
    )

    # Assemble full downloadable prompt (full CSV)
    prompt_full = (
        "=== CONTEXT ===\n"
        + "\n".join(header_lines) + "\n\n"
        + "=== NARRATIVE SUMMARY ===\n"
        + summary_text.strip() + "\n\n"
        + "=== INSTRUCTIONS ===\n"
        + instructions + "\n\n"
        + "=== DATA (CSV) ===\n"
        + csv_full
    )

    return prompt_copy, prompt_full

from functools import reduce

def load_and_clean(uploaded_file) -> pd.DataFrame:
    """Load one CSV and run your existing cleaning pipeline."""
    raw = load_statcan_csv(uploaded_file)
    df = filter_relevant_rows(raw)
    df = collapse_duplicate_characteristics(df)
    df = prune_columns(df)
    return df


def merge_cleaned_profiles(named_cleaned: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, str, list[str]]:
    """
    named_cleaned: list of (label, cleaned_df) where each df has Topic/Topic_norm/Characteristic and one geo column.
    Returns: (merged_df, combined_label, value_cols_kept)
    - Keeps each area as its own column (by label)
    - Adds 'Combined' = row-wise sum across those columns (ignoring NaN)
    """
    # Build minimal frames with a single numeric column per area, labeled by that area
    minimal_frames = []
    labels = []
    for label, df in named_cleaned:
        geo_col = pick_geo_col(df)
        if not geo_col:
            continue
        minimal = df[["Topic", "Topic_norm", "Characteristic", geo_col]].copy()
        minimal = minimal.rename(columns={geo_col: label})
        minimal_frames.append(minimal)
        labels.append(label)

    if not minimal_frames:
        return pd.DataFrame(), "Combined", []

    # Outer-merge on (Topic_norm, Characteristic, Topic) to preserve headings and sort keys
    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Topic_norm", "Characteristic", "Topic"], how="outer"
        ),
        minimal_frames
    )

    # Create a Combined column (numeric sum across included labels)

    def _to_num(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Convert %, comma-formatted strings to numeric.
        Works for both Series and DataFrames.
        """
        if isinstance(obj, pd.Series):
            return pd.to_numeric(
                obj.astype(str).str.replace(r"[,%]", "", regex=True),
                errors="coerce"
            )
        # DataFrame path
        return obj.apply(
            lambda col: pd.to_numeric(
                col.astype(str).str.replace(r"[,%]", "", regex=True),
                errors="coerce"
            )
        )

    # columns to sum (only those that actually survived the merge)
    value_cols = [c for c in labels if c in merged.columns]
    if value_cols:
        merged["Combined"] = _to_num(merged[value_cols]).sum(axis=1, skipna=True)
    else:
        merged["Combined"] = pd.NA

    # Reuse your sort logic within topics
    merged["__char_sort_key__"] = merged["Characteristic"].apply(_characteristic_sort_key)
    merged = (
        merged.sort_values(by=["Topic_norm", "__char_sort_key__"], kind="mergesort")
              .drop(columns="__char_sort_key__")
              .reset_index(drop=True)
    )

    # For downstream functions that look for a numeric "best" column, Combined will win.
    return merged, "Combined", value_cols


# ------------------------------------------------
# UI (multi-CSV regional rollup supported)
# ------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more Statistics Canada Census Profile CSVs",
    type=["csv"],
    accept_multiple_files=True,
    help="Tip: upload both the municipality and the surrounding MD if the program will draw from both.",
)

if not uploaded_files:
    st.info("Upload at least one CSV to generate the community profile.")
    st.stop()
else:
    # Load & clean each file
    cleaned_named: list[tuple[str, pd.DataFrame]] = []
    for uf in uploaded_files:
        df_clean = load_and_clean(uf)
        label = extract_place_name(uf.name) or os.path.splitext(uf.name)[0]
        cleaned_named.append((label, df_clean))

    # Merge if multiple; or just take the single frame
    if len(cleaned_named) == 1:
        place_guess = cleaned_named[0][0]
        cleaned_df = cleaned_named[0][1]
    else:
        merged_df, combined_label, indiv_cols = merge_cleaned_profiles(cleaned_named)
        cleaned_df = merged_df
        # Combined label for the report title
        place_guess = " + ".join([nm for nm, _ in cleaned_named])

        # Optional: let user choose which value column to use for the analysis (default = Combined)
        with st.sidebar:
            st.markdown("### Value column for analysis")
            default_choice = "Combined"
            choices = [default_choice] + [nm for nm, _ in cleaned_named]
            chosen_value_col = st.selectbox(
                "Which column should drive the summary, PDF, and prompt?",
                choices,
                index=0,
                help="Use 'Combined' for a regional rollup, or pick an individual area to view it on its own."
            )
        # If user picked an individual area, make that the dominant numeric column by moving it to 'Combined'
        if chosen_value_col != "Combined" and chosen_value_col in cleaned_df.columns:
            cleaned_df["Combined"] = pd.to_numeric(
                cleaned_df[chosen_value_col].astype(str).str.replace("%","").str.replace(",",""),
                errors="coerce"
            )

    # Sidebar age controls
    with st.sidebar:
        st.markdown("### Real-time age adjustment")
        use_age_adjust = st.checkbox("Age cohorts forward from 2021", value=True)
        as_of = st.date_input("As-of date", value=date.today())

    # ---- Summary (uses 'Combined' implicitly via pick_geo_col) ----
    st.subheader("Community Profile Summary")
    summary_text = generate_summary(
        cleaned_df,
        as_of_date=(as_of if use_age_adjust else None),
        place_name=place_guess,
    )
    st.write(summary_text)

    # ---- PDF ----
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

    # ---- ChatGPT prompt (expander; uses your existing helper) ----
    with st.expander("Deeper analysis (copy-ready prompt)", expanded=False):
        prompt_copy, prompt_full = build_chatgpt_prompt(
            summary_text=summary_text or "",
            cleaned_df=cleaned_df,
            place_name=place_guess,
            as_of_date=(as_of if use_age_adjust else None),
        )
        st.caption(
            "Copy this prompt into ChatGPT to get a deeper analysis and program impact assessment tailored to Growing Up Strong."
        )
        st.code(prompt_copy, language="markdown")
        st.download_button(
            label="⬇️ Download full prompt (TXT with complete table)",
            data=prompt_full.encode("utf-8"),
            file_name=f"{(place_guess or 'community')}_deep_analysis_prompt.txt",
            mime="text/plain",
        )

    # ---- Filtered Report ----
    st.subheader("Filtered Report")
    render_report(cleaned_df)

    # ---- Indigenous rollup (uses pick_geo_col → will prefer 'Combined') ----
    geo_col = pick_geo_col(cleaned_df)
    topic_col_for_pop = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"
    pop_rows = cleaned_df[
        (cleaned_df[topic_col_for_pop].str.contains("Population and dwellings", case=False, na=False)) &
        (cleaned_df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col]) if not pop_rows.empty else None

    st.subheader("Indigenous Population (Ethnic origin–derived)")
    indig_from_ethnic = derive_indigenous_from_ethnic_origin(cleaned_df, geo_col, pop_val_num)
    if indig_from_ethnic.empty:
        st.caption("No Alberta-relevant Indigenous groups detected in Ethnic origin.")
    else:
        st.dataframe(indig_from_ethnic, use_container_width=True)

    # ---- Export ----
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
    st.markdown("**Note:** If you uploaded multiple CSVs, all summary figures above reflect the selected value column (default: Combined).")
