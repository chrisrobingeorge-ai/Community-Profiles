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
        # e.g. "German", "Low German", "Tagalog (Filipino)", "Punjabi (Panjabi)",
        # "Cree languages", "Blackfoot", "Dene", etc.
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

def extract_indigenous_nations(df, geo_col):
    """
    Return a list of Indigenous Nations / Peoples present in meaningful numbers.
    We only surface culturally meaningful identifiers, not ancestry bookkeeping combos.
    """

    INDIG_TOPIC_REGEX = "Indigenous population|Indigenous ancestry|Indigenous identity"

    # Words/phrases that we consider valid to surface
    allowed_markers = [
        "cree",
        "dene",
        "blackfoot",
        "stoney",
        "saulteaux",
        "anishinaabe",
        "ojibwe",
        "métis",
        "metis",
        "inuit",
        # we'll include general 'first nations' only if we don't get more specific
        "first nations",
    ]

    # Phrases that we should IGNORE because they're structural/statistical, not communities
    block_markers = [
        "ancestry only",
        "and non-indigenous",
        "single ancestry",
        "multiple aboriginal responses",
        "first nations and métis",
        "métis and non-indigenous",
        "first nations, inuk (inuit), and métis",
        "total",
    ]

    sub = df[df["Topic"].str.contains(INDIG_TOPIC_REGEX, case=False, na=False)].copy()

    found_groups = []

    for _, r in sub.iterrows():
        raw_label = str(r["Characteristic"]).strip()
        val_num = _coerce_number(r[geo_col])
        if val_num is None or val_num <= 0:
            continue

        label_lower = raw_label.lower()

        # if it's clearly a total row or ancestry bookkeeping row, skip it
        if any(b in label_lower for b in block_markers):
            continue

        # does this row look like it names a specific Nation / People?
        matched_marker = None
        for marker in allowed_markers:
            if marker in label_lower:
                matched_marker = marker
                break
        if not matched_marker:
            continue

        # Clean up wording
        clean_label = raw_label
        # Simplifications for nicer output:
        replacements = {
            "First Nations (North American Indian)": "First Nations",
            "Cree First Nations": "Cree",
            "Cree nations": "Cree",
            "Dene First Nations": "Dene",
            "Blackfoot First Nations": "Blackfoot",
            "Métis": "Métis",
            "Metis": "Métis",
            "Inuit": "Inuit",
        }
        for old, new in replacements.items():
            clean_label = clean_label.replace(old, new)

        # Strip noisy trailing phrases
        strip_phrases = [
            "First Nations individuals",
            "First Nations persons",
            "First Nations people",
        ]
        for phrase in strip_phrases:
            clean_label = clean_label.replace(phrase, "")

        clean_label = clean_label.strip()

        found_groups.append(clean_label)

    # Deduplicate while keeping order
    seen = set()
    ordered_unique = []
    for grp in found_groups:
        key = grp.lower()
        if key not in seen:
            seen.add(key)
            ordered_unique.append(grp)

    # If we captured both specific Nations (Cree, Dene, etc.) and a generic "First Nations",
    # we can drop the generic "First Nations" because the specifics are better.
    specific_terms = [g for g in ordered_unique if g.lower() not in ["first nations"]]
    if specific_terms:
        ordered_unique = specific_terms

    return ordered_unique

from datetime import date, datetime
from collections import OrderedDict

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
            out[k] += v  # current
            # inflow added below from previous band
        else:
            stay = v * 4.0/5.0
            move = v * 1.0/5.0
            out[k] += stay
            # push to next band
            nxt = keys[i+1]
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
                stay = v * (1.0 - frac/5.0)
                move = v * (frac/5.0)
                out[k] += stay
                nxt = keys[i+1]
                out[nxt] += move
        cur = out

    return cur

def sum_bands(bands: OrderedDict, wanted_labels: list[str]) -> float:
    return float(sum(bands.get(l, 0.0) for l in wanted_labels))

def generate_summary(df: pd.DataFrame, as_of_date: date | None = None) -> str:
    if df.empty:
        return "No summary available."

    geo_col = pick_geo_col(df)
    if not geo_col:
        return "No summary available."

    lines = []

    # -------------------------------------------------
    # 1. Population (for % math)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2. Age structure (kids, seniors) with optional aging to 'as_of_date'
    # -------------------------------------------------
    kids_bands = ["0 to 4 years", "5 to 9 years", "10 to 14 years"]
    seniors_bands = ["65 to 69 years", "70 to 74 years", "75 to 79 years", "80 to 84 years", "85 years and over"]

    kids_val = None
    seniors_val = None

    if as_of_date is not None:
        # Use cohort-aging to adjust age bands to the chosen date
        geo_col = pick_geo_col(df)
        age_bands = extract_age_bands(df, geo_col)
        adj = age_bands_adjust_to_date(age_bands, as_of=as_of_date)
        kids_val = sum_bands(adj, kids_bands)
        seniors_val = sum_bands(adj, seniors_bands)
    else:
        # Use 2021 raw values
        kids_val = _best_numeric_from(
            df, topic_regex="Age characteristics", char_regex=r"\b0\s*to\s*14\s*years\b", geo_col=geo_col
        )
        seniors_val = _best_numeric_from(
            df, topic_regex="Age characteristics", char_regex=r"65\s*years", geo_col=geo_col
        )

    kids_pct = None
    seniors_pct = None
    if pop_val_num and pop_val_num > 0:
        if kids_val and 0 < kids_val < pop_val_num:
            kids_pct = (kids_val / pop_val_num) * 100.0
        if seniors_val and 0 < seniors_val < pop_val_num:
            seniors_pct = (seniors_val / pop_val_num) * 100.0


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
                f"{age_bits[0]}, and {age_bits[1]}. We should expect to serve a lot of school-age kids and also grandparents / older caregivers in the same community spaces."
            )
        else:
            lines.append(
                f"{age_bits[0]}. We should expect to serve a large number of school-age kids in this community."
            )

    # -------------------------------------------------
    # 3. Household structure / stability
    #    (single caregivers, household size, renter/owner)
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

    # -------------------------------------------------
    # 4. Income pressure (define it EARLY so we can use it below)
    # -------------------------------------------------
    low_income_val = _best_numeric_from(
        df,
        topic_regex="Low income and income inequality",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )

    household_barrier_lines = []

    if single_parent_share:
        household_barrier_lines.append(
            "There is a notable share of one-parent households, meaning many caregivers are doing this alone."
        )

    if hh_size and hh_size >= 2.5:
        household_barrier_lines.append(
            f"Average household size is about {hh_size:.1f} people, so programming should assume multiple kids per family and sometimes multigenerational households."
        )

    if renters_share and (not owners_share or renters_share > owners_share):
        household_barrier_lines.append(
            "Housing leans toward renting, which usually means less stability and more moves."
        )
    elif owners_share and owners_share >= 1.0:
        household_barrier_lines.append(
            "Most homes appear to be owner-occupied, which suggests some families are rooted here long-term."
        )

    if low_income_val:
        household_barrier_lines.append(
            "Income data suggests affordability will be a barrier. Families report program fees, gear, and travel costs as major blockers, especially outside big cities. We should plan for low- or no-cost access, not assume families can pay."
        )

    if household_barrier_lines:
        lines.append(" ".join(household_barrier_lines))

    # -------------------------------------------------
    # 5. Indigenous Nations / Peoples
    # -------------------------------------------------
    nations = extract_indigenous_nations(df, geo_col)
    if nations:
        if len(nations) == 1:
            lines.append(
                f"Indigenous presence: {nations[0]} is present in meaningful numbers. Programming should be coordinated with local Indigenous leadership and should expect Indigenous youth participation."
            )
        else:
            nation_list_text = ", ".join(nations[:-1]) + f", and {nations[-1]}"
            lines.append(
                f"Indigenous presence: {nation_list_text} are present in meaningful numbers. Programming should be coordinated with local Indigenous leadership and should expect Indigenous youth participation."
            )

    # -------------------------------------------------
    # 6. Language / newcomer communities
    # -------------------------------------------------
    specific_langs = extract_significant_languages(df, geo_col, pop_val_num)

    newcomer_val = _best_numeric_from(
        df,
        topic_regex="Immigrant status and period of immigration|Selected places of birth for the recent immigrant population",
        char_regex=None,
        geo_col=geo_col,
        min_pct=1.0,
    )

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
            lang_sentences.append(
                f"Families speak {specific_langs[0]} at home in meaningful numbers. Parent communication and registration may need to happen in that language, not just English."
            )
        else:
            lang_sentences.append(
                "Families speak " +
                ", ".join(specific_langs[:-1]) +
                f", and {specific_langs[-1]} at home in meaningful numbers. Parent communication and registration may need to include these languages, not just English."
            )

    if newcomer_val:
        lang_sentences.append(
            "There is a visible newcomer / recent immigrant population. Trust-building may require connectors who already work with these families (schools, settlement services, cultural associations)."
        )

    if minority_lang_val:
        lang_sentences.append(
            "Some children here are legally entitled to minority-language (Francophone) education, so French-language access is an expectation."
        )

    if lang_sentences:
        lines.append(" ".join(lang_sentences))

    # -------------------------------------------------
    # 7. Mobility / rootedness (are families newly arrived / moving)
    # -------------------------------------------------
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
            "Families are still moving in and settling. Not everyone has long-standing local support networks yet."
        )

    # -------------------------------------------------
    # 8. Commuting / scheduling pressure
    # -------------------------------------------------
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
        if ("car" in row_char or "automobile" in row_char or "driver" in row_char):
            car_commute_flag = True

    if car_commute_flag or long_commute_flag:
        if car_commute_flag and long_commute_flag:
            commute_sentence = (
                "Most working adults rely on driving, and some families are already doing long daily commutes."
            )
        elif car_commute_flag:
            commute_sentence = "Most working adults rely on driving."
        else:
            commute_sentence = "Some families are already doing long daily commutes."

        lines.append(
            commute_sentence +
            " Programs should run locally (school / community space) right after school or early evening, because asking families to add more travel time is unrealistic."
        )

    # -------------------------------------------------
    # 9. Operational takeaway for staff
    # -------------------------------------------------
    lines.append(
        "Operational takeaway: deliver programming in-town, keep direct cost as close to zero as possible, coordinate early with schools and Indigenous leadership, and plan for multilingual parent communication."
    )

    # -------------------------------------------------
    # Final assembly
    # -------------------------------------------------
    # Join into one readable paragraph
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

with st.sidebar:
    st.markdown("### Real-time age adjustment")
    use_age_adjust = st.checkbox("Age cohorts forward from 2021", value=True)
    as_of = st.date_input("As-of date", value=date.today())

# ...

summary_text = generate_summary(
    cleaned_df,
    as_of_date=(as_of if use_age_adjust else None)
)
st.write(summary_text)
    
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
