import io, os, re
from datetime import date, datetime
from collections import OrderedDict

import pandas as pd
import streamlit as st  # <-- must be above st.set_page_config

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
    exclude = {"Topic", "Characteristic", "Notes", "Note", "Symbol", "Flags", "Flag"}
    candidates = [c for c in df.columns if c not in exclude]
    best_col, best_numeric_score = None, -1
    for col in candidates:
        numeric_score = 0
        for val in df[col].head(50):
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
    row = df[
        (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
        (df["Characteristic"].str.fullmatch(rf"\s*{band_label}\s*", case=False, na=False))
    ]
    if row.empty:
        row = df[
            (df["Topic"].str.contains(r"Age characteristics", case=False, na=False)) &
            (df["Characteristic"].str.contains(rf"{band_label}", case=False, na=False))
        ]
    if row.empty:
        return 0.0
    return _coerce_number(row.iloc[0][geo_col]) or 0.0


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
    community_label = place_name if place_name else "this community"

    pop_rows = df[
        (df["Topic"].str.contains("Population and dwellings", case=False, na=False)) &
        (df["Characteristic"].str.contains("Population, 2021", case=False, na=False))
    ]
    pop_val_num = _coerce_number(pop_rows.iloc[0][geo_col]) if not pop_rows.empty else None

    kids_band_labels = ["0 to 4 years", "5 to 9 years", "10 to 14 years"]
    teens_band_labels = ["15 to 19 years"]
    seniors_band_labels = ["65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"]

    if as_of_date is not None:
        bands_2021 = extract_age_bands(df, geo_col)
        adj = age_bands_adjust_to_date(bands_2021, as_of=as_of_date)
        kids_val = sum_bands(adj, kids_band_labels)
        teens_val = sum_bands(adj, teens_band_labels)
        seniors_val = sum_bands(adj, seniors_band_labels)
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
        teens_val = grab_sum_for(teens_band_labels)
        seniors_val = grab_sum_for(seniors_band_labels)

    kids_pct, kids_cnt = _percent_or_none(kids_val, pop_val_num)
    teens_pct, teens_cnt = _percent_or_none(teens_val, pop_val_num)
    seniors_pct, seniors_cnt = _percent_or_none(seniors_val, pop_val_num)

    single_parent_share = _best_numeric_from(
        df, topic_regex="Household type|Household and dwelling characteristics",
        char_regex="one-parent|single-parent", geo_col=geo_col, min_pct=1.0,
    )
    hh_size = _best_numeric_from(
        df, topic_regex="Household and dwelling characteristics",
        char_regex="Average household size", geo_col=geo_col,
    )
    renters_share = _best_numeric_from(
        df, topic_regex="Household and dwelling characteristics|Household type",
        char_regex="rented", geo_col=geo_col, min_pct=1.0,
    )
    owners_share = _best_numeric_from(
        df, topic_regex="Household and dwelling characteristics|Household type",
        char_regex="owned", geo_col=geo_col, min_pct=1.0,
    )
    low_income_val = _best_numeric_from(
        df, topic_regex="Low income and income inequality", char_regex=None,
        geo_col=geo_col, min_pct=1.0,
    )

    mobility_rows = df[df["Topic"].str.contains("Mobility status 1 year ago|Mobility status 5 years ago", case=False, na=False)]
    mobility_flag = False
    for _, r in mobility_rows.iterrows():
        mv = _coerce_number(r[geo_col])
        row_char = str(r["Characteristic"]).lower()
        if mv and mv > 0 and ("moved" in row_char or "different" in row_char):
            mobility_flag = True
            break

    commute_rows = df[df["Topic"].str.contains("Main mode of commuting|Commuting duration", case=False, na=False)]
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

    sig_langs_all = extract_significant_languages(df, geo_col, pop_val_num)
    top_langs = _top_n(sig_langs_all, 2)

    nations_tbl = build_indigenous_table(df, geo_col, pop_val_num)
    nation_list_for_text = []
    if nations_tbl is not None and not nations_tbl.empty:
        top_nations_rows = nations_tbl.head(3).to_dict("records")
        for row in top_nations_rows:
            g = str(row.get("Group", "")).strip()
            p = str(row.get("Percent", "")).strip() if row.get("Percent", None) else ""
            nation_list_for_text.append(f"{g} ({p})" if p else g)

    def seat_range(count_val):
        if not count_val or count_val <= 0:
            return (None, None)
        low = int(round(count_val * 0.05))
        hi  = int(round(count_val * 0.08))
        low = max(low, 10)
        hi  = max(hi, low)
        return (low, hi)

    kids_low, kids_hi = seat_range(kids_cnt if kids_cnt else 0)
    teen_low, teen_hi = seat_range(teens_cnt if teens_cnt else 0)
    teen_target = None
    if teen_low and teen_hi:
        teen_target = max(12, int(round((teen_low + teen_hi) / 2.0)))

    snapshot_lines = [
        f"{community_label} has an established resident base. "
        "The points below are meant to guide how we deliver programming here, not just describe statistics."
    ]
    snap_bits = []
    if kids_cnt and kids_cnt > 0:
        snap_bits.append(f"~{kids_cnt} children 0–14" + (f" ({_safe_pct(kids_pct)})" if kids_pct else ""))
    if teens_cnt and teens_cnt > 0:
        snap_bits.append(f"~{teens_cnt} teens 15–19" + (f" ({_safe_pct(teens_pct)})" if teens_pct else ""))
    if seniors_cnt and seniors_cnt > 0:
        snap_bits.append(f"~{seniors_cnt} seniors 65+" + (f" ({_safe_pct(seniors_pct)})" if seniors_pct else ""))
    if snap_bits:
        snapshot_lines.append("Key age groups: " + "; ".join(snap_bits) + ".")
    if single_parent_share:
        snapshot_lines.append(f"Single-caregiver households are present (≈{single_parent_share:.1f}%).")
    if renters_share or owners_share:
        parts = []
        if renters_share: parts.append(f"{renters_share:.1f}% renting")
        if owners_share:  parts.append(f"{owners_share:.1f}% owning")
        if parts: snapshot_lines.append("Housing mix: " + " / ".join(parts) + ".")
    if low_income_val:
        snapshot_lines.append(f"Low-income pressure shows up in census indicators (≈{low_income_val:.1f}%).")
    if top_langs:
        snapshot_lines.append("Families are using languages at home beyond English/French, notably " + ", ".join(top_langs) + ".")
    if nation_list_for_text:
        snapshot_lines.append("Indigenous Nations / Peoples present in meaningful numbers include " + ", ".join(nation_list_for_text) + ".")
    snapshot_block = " ".join(snapshot_lines)

    youth_lines = []
    if kids_cnt:
        youth_lines.append(
            "There are a lot of younger children here, which justifies recurring in-community programming instead of one-off outreach. "
            "This needs to look like 'every Tuesday/Thursday after school in the same place,' not a special event."
        )
    else:
        youth_lines.append("There are meaningful numbers of younger children. We should plan for repeating programs, not pop-ins.")
    if teens_cnt:
        youth_lines.append(
            "There is also a visible 15–19 group. This is exactly the age when girls tend to drop out of structured activity. "
            "For that band, emotional safety and peer belonging have to come first, and 'performance' comes later if at all."
        )
    if seniors_cnt:
        youth_lines.append(
            "A notable 65+ population is also present. Grandparents are often the drivers, sit-and-wait adults, "
            "and sometimes the default childcare at pickup."
        )
    youth_block = " ".join(youth_lines)

    hh_lines = []
    if hh_size:
        hh_lines.append(
            f"Average household size is about {hh_size:.1f} people. "
            "Expect siblings/cousins to arrive together; plan for strollers and snacks."
        )
    else:
        hh_lines.append("Households commonly include multiple kids or extended family.")
    if single_parent_share:
        hh_lines.append(
            "Single caregivers handle work, transport, meals, homework, and bedtime without backup. "
            "One location, predictable timing, and no 'must stay to supervise' assumptions."
        )
    if renters_share and (not owners_share or renters_share > owners_share):
        hh_lines.append(
            "Renting is common → more moves and less scheduling stability. Build easy re-entry: if a family disappears, they’re welcome back without penalty."
        )
    elif owners_share:
        hh_lines.append("Owner-occupied housing is strong → one consistent site can hold a stable group over multiple months.")
    if low_income_val:
        hh_lines.append(
            "Cost is a barrier. Avoid terms like 'tuition/fee/uniform/recital cost.' Provide shoes/clothing, don’t merely recommend them."
        )
    if mobility_flag:
        hh_lines.append(
            "Some families are newly arrived or recently moved. Don’t assume parents know where to get help—act as bridge and connector."
        )
    hh_block = " ".join(hh_lines)

    lang_lines = []
    if top_langs:
        if len(top_langs) == 1:
            lang_lines.append(
                f"Parent communication cannot assume English-only. Provide invites/consent in {top_langs[0]} as well as English."
            )
        else:
            lang_lines.append(
                "Parent communication cannot assume English-only. Provide invites/consent in " + " and ".join(top_langs) + " as well as English."
            )
    else:
        lang_lines.append("Even if English dominates on paper, trust often sits in the home language. Check what schools use with caregivers.")
    if nation_list_for_text:
        lang_lines.append(
            "Indigenous partnership is not optional. Co-present with local Indigenous leadership; Indigenous adults should be visibly in the room."
        )
    lang_block = " ".join(lang_lines)

    ops_lines = []
    if kids_low and kids_hi:
        ops_lines.append(
            f"For younger kids, aim for {kids_low}–{kids_hi} active seats across recurring weekly sessions (manageable with two facilitators)."
        )
    else:
        ops_lines.append("Size kids’ programming to ~5–8% of the local child population, not just a token pilot.")
    if teen_target:
        ops_lines.append(
            f"For teens (especially girls 15–19), build one dedicated block with ~{teen_target} seats. Low performance pressure; zero uniform/gear cost."
        )
    if long_commute_flag or car_commute_flag:
        ops_lines.append(
            "Deliver in-town at/near a school, right after school or early evening. Another cross-town drive after a 60+ min commute kills attendance."
        )
    else:
        ops_lines.append("After-school / early evening at the closest possible school or hall with predictable days (e.g., Tue/Thu).")
    ops_lines.append("Room setup: clear sightlines, space for caregivers/siblings, stroller parking, water/snack table, 'you can stay and watch'.")
    ops_lines.append("Have a gear library on site (loaner shoes/clothes) and explicit 'no fee / no uniform / no recital cost' messaging.")
    ops_block = " ".join(ops_lines)

    risk_lines = [
        "Main risks: cost stigma, commute fatigue, social safety for teen girls, and language disconnect with caregivers.",
        "Mitigations: zero-dollar entry; in-neighbourhood delivery; teen block focused on belonging; first contact via trusted channels (schools, settlement, FCSS, Indigenous leadership, churches).",
        "90-day success test: (1) ≥60% of enrolled kids still attending in week 6; (2) teen girls ≥40% of teen block and ≥60% still attending in week 6; (3) ≥75% of caregivers felt welcome/safe and would return.",
    ]
    kpi_block = " ".join(risk_lines)

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


def create_full_pdf(summary_text: str, place_name: str, cleaned_df: pd.DataFrame) -> bytes:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HeadingPlace", parent=styles["Heading1"], fontSize=16, leading=20, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHeader", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=12, spaceAfter=6, textColor=colors.black))
    styles.add(ParagraphStyle(name="BodyTextTight", parent=styles["BodyText"], fontSize=10, leading=13, spaceAfter=6))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    safe_place = place_name or "Community Profile"
    story.append(Paragraph(f"Community Profile Summary for {safe_place}", styles["HeadingPlace"]))
    story.append(Spacer(1, 6))

    for block in summary_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        story.append(Paragraph(block, styles["BodyTextTight"]))
    story.append(PageBreak())

    topic_col = "Topic_norm" if "Topic_norm" in cleaned_df.columns else "Topic"
    value_cols = [c for c in cleaned_df.columns if c not in ("Topic", "Characteristic", "Topic_norm", "Notes", "Note", "Symbol", "Flags", "Flag")]

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
        story.append(Paragraph(str(topic), styles["SectionHeader"]))

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
            ("ALIGN",       (1,1), (-1,-1), "RIGHT"),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

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
    raw_df = load_statcan_csv(uploaded_file)
    cleaned_df = filter_relevant_rows(raw_df)
    cleaned_df = collapse_duplicate_characteristics(cleaned_df)
    cleaned_df = prune_columns(cleaned_df)

    with st.sidebar:
        st.markdown("### Real-time age adjustment")
        use_age_adjust = st.checkbox("Age cohorts forward from 2021", value=True)
        as_of = st.date_input("As-of date", value=date.today())

    st.subheader("Community Profile Summary")
    place_guess = extract_place_name(uploaded_file.name)
    summary_text = generate_summary(
        cleaned_df,
        as_of_date=(as_of if use_age_adjust else None),
        place_name=place_guess,
    )
    st.write(summary_text)

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

    st.subheader("Filtered Report")
    render_report(cleaned_df)

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
    st.markdown("**Note:** Open the downloaded HTML in a browser and use Print → Save as PDF.")
