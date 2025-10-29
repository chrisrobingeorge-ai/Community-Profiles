from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_mapping(path: Path) -> Dict:
    """
    Load YAML mapping that defines identity columns and section-to-column mappings.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f) or {}
    if "identity" not in mapping:
        mapping["identity"] = {"geoid_col": "GeoUID", "name_col": "Geographic name"}
    return mapping


def _read_csv_any(path: Path) -> pd.DataFrame:
    """
    Read a CSV with forgiving options:
    - try utf-8-sig, then latin-1
    - if still failing, auto-sniff delimiter (engine='python')
    - last resort: explicitly try semicolon
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            pass
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        pass
    return pd.read_csv(path, sep=";", low_memory=False)


def normalize_col(col: str) -> str:
    """
    Canonicalize a column name for matching:
    - cast to str
    - collapse whitespace to single space
    - strip
    - lower
    """
    if col is None:
        return ""
    c = re.sub(r"\s+", " ", str(col)).strip().lower()
    return c


def _is_topic_characteristic_matrix(df: pd.DataFrame) -> bool:
    """
    Heuristic: detect StatCan matrix with 'Topic' + 'Characteristic' columns,
    and many geography columns to the right.
    """
    cols_norm = [normalize_col(c) for c in df.columns]
    return ("topic" in cols_norm) and ("characteristic" in cols_norm) and (len(df.columns) >= 4)


def _reshape_topic_characteristic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a Topic/Characteristic × Geography-columns matrix into a
    row-per-geography table using 'Characteristic' as the feature columns.

    Output columns:
    - 'Geographic name'
    - one column per distinct Characteristic (values are the cell entries)
    """
    # Keep only the first occurrence if duplicate column names slipped in
    df = df.loc[:, ~df.columns.duplicated()]

    # Identify core columns
    # Use original casing to preserve characteristic strings
    def find_col(target: str) -> Optional[str]:
        for c in df.columns:
            if normalize_col(c) == target:
                return c
        return None

    topic_col = find_col("topic")
    char_col = find_col("characteristic")
    if topic_col is None or char_col is None:
        return pd.DataFrame()

    # Drop fully empty columns (lots of leading commas in some extracts)
    df = df.dropna(axis=1, how="all")

    # Geography columns = everything except Topic/Characteristic and common junk columns
    junk_norm = {
        "note", "total", "total_flag", "men+", "men+_flag", "women+", "women+_flag",
        "rate", "rates"
    }
    geo_cols = []
    for c in df.columns:
        n = normalize_col(c)
        if n in (normalize_col(topic_col), normalize_col(char_col)):
            continue
        if n in junk_norm:
            continue
        geo_cols.append(c)

    # If nothing looks like geography columns, bail
    if not geo_cols:
        return pd.DataFrame()

    # Melt to long (Topic, Characteristic, Geography name, Value)
    long = df.melt(
        id_vars=[topic_col, char_col],
        value_vars=geo_cols,
        var_name="Geographic name",
        value_name="value",
    )

    # Some files have extra header rows, blanks, or '...' placeholders — clean them
    # Keep rows where Characteristic is non-null and not just empty/commas
    mask_valid_char = long[char_col].astype(str).str.strip().ne("")
    long = long[mask_valid_char].copy()

    # We only want the numeric/text value cells; drop empty or '...' markers
    def _clean_cell(x):
        s = str(x).strip()
        if s in {"...", "", "nan", "None"}:
            return pd.NA
        return s

    long["value"] = long["value"].map(_clean_cell)

    # Pivot so each Characteristic becomes a column; keep the FIRST non-null per (Geo, Characteristic)
    # In case the same Characteristic appears under multiple Topics, we'll aggregate by first valid.
    pivot = (
        long.dropna(subset=["value"])
            .groupby(["Geographic name", char_col], as_index=False)["value"]
            .first()
            .pivot(index="Geographic name", columns=char_col, values="value")
            .reset_index()
    )

    # Rename index column to match app's identity name_col
    pivot = pivot.rename(columns={"Geographic name": "Geographic name"})

    # Normalize headers for later matching (but keep readable values)
    pivot.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in pivot.columns]

    return pivot


def _align_and_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Given a list of per-community wide tables, normalize headers and align columns.
    """
    if not frames:
        return pd.DataFrame()
    # Normalize headers like in original loader
    for i in range(len(frames)):
        frames[i].columns = [normalize_col(c) for c in frames[i].columns]
    all_cols = set()
    for df in frames:
        all_cols.update(df.columns)
    all_cols = list(all_cols)
    aligned = []
    for df in frames:
        for c in all_cols:
            if c not in df.columns:
                df[c] = pd.NA
        aligned.append(df[all_cols])
    return pd.concat(aligned, axis=0, ignore_index=True)


def load_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load one or multiple CSVs under data/raw. If multiple, reshape as needed,
    then vertically concatenate and align columns.

    - Case-insensitive extension match (.csv, .CSV, etc.)
    - Recursive into subfolders
    - Accept compressed CSVs (.csv.gz, .csv.zip) that pandas can read
    - If a file is in Topic/Characteristic × Geography matrix format,
      reshape it into a per-community wide table
    """
    raw_dir = Path(raw_dir)

    def is_csv_like(p: Path) -> bool:
        low = p.name.lower()
        return (
            p.is_file()
            and (low.endswith(".csv") or low.endswith(".csv.gz") or low.endswith(".csv.zip"))
        )

    paths = sorted([p for p in raw_dir.rglob("*") if is_csv_like(p)])
    if not paths:
        return pd.DataFrame()

    reshaped_frames: List[pd.DataFrame] = []

    for p in paths:
        df = _read_csv_any(p)

        # Fast path: if it already looks like row-per-community with identity columns, just normalize headers
        cols_norm = [normalize_col(c) for c in df.columns]
        has_name = "geographic name" in cols_norm or "geo name" in cols_norm or "name" in cols_norm
        has_geouid = "geouid" in cols_norm or "geo uid" in cols_norm
        if has_name or has_geouid:
            df.columns = [normalize_col(c) for c in df.columns]
            reshaped_frames.append(df)
            continue

        # Matrix path
        if _is_topic_characteristic_matrix(df):
            wide = _reshape_topic_characteristic_matrix(df)
            if not wide.empty:
                wide.columns = [normalize_col(c) for c in wide.columns]
                # Ensure identity column exists for app (use name; GeoUID may be missing in these extracts)
                if "geographic name" not in wide.columns:
                    wide = wide.rename(columns={wide.columns[0]: "geographic name"})
                reshaped_frames.append(wide)
                continue

        # If we get here, we couldn't recognize the shape; try a lenient normalization and keep as-is
        df.columns = [normalize_col(c) for c in df.columns]
        reshaped_frames.append(df)

    merged = _align_and_concat(reshaped_frames)

    return merged


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Given human-readable candidate names, return the actual df column key (normalized) if found.
    We compare on normalized representations, with a punctuation-stripped fallback.
    """
    if df is None or df.empty or not candidates:
        return None
    normalized_cols = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized_cols:
            return normalized_cols[key]

    def strip_punct(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s)

    normalized_no_punct = {strip_punct(k): v for k, v in normalized_cols.items()}
    for cand in candidates:
        key = strip_punct(normalize_col(cand))
        if key in normalized_no_punct:
            return normalized_no_punct[key]
    return None


def list_communities(df: pd.DataFrame, geoid_col: str, name_col: str) -> List[str]:
    """
    Return a sorted unique list of community names.
    """
    geoid_key = resolve_column(df, [geoid_col, "geouid"])
    name_key = resolve_column(df, [name_col, "geographic name", "geo name", "name"])
    if name_key is None and geoid_key in df.columns:
        return sorted(df[geoid_key].astype(str).fillna("Unknown").unique().tolist())
    if name_key is None:
        return []
    names = df[name_key].astype(str).fillna("Unknown")
    return sorted(names.unique().tolist())


def get_record(
    df: pd.DataFrame,
    geoid: Optional[str] = None,
    name: Optional[str] = None,
    geoid_col: str = "GeoUID",
    name_col: str = "Geographic name",
) -> Optional[pd.Series]:
    """
    Return the first matching row by GeoUID or by community name (case-insensitive).
    """
    geoid_key = resolve_column(df, [geoid_col, "geouid"]) or geoid_col
    name_key = resolve_column(df, [name_col, "geographic name", "geo name", "name"]) or name_col

    if geoid and geoid_key in df.columns:
        mask = df[geoid_key].astype(str).str.strip() == str(geoid).strip()
        if mask.any():
            return df.loc[mask].iloc[0]

    if name and name_key in df.columns:
        mask = df[name_key].astype(str).str.strip().str.lower() == str(name).strip().lower()
        if mask.any():
            return df.loc[mask].iloc[0]
    return None


def _mapping_to_required_columns(mapping: Dict) -> Dict[str, List[str]]:
    """
    Flatten mapping to a dict of section -> list of column names (as given).
    """
    required: Dict[str, List[str]] = {}
    skip_keys = {"identity"}
    for section, fields in mapping.items():
        if section in skip_keys or not isinstance(fields, dict):
            continue
        cols = [v for v in fields.values() if isinstance(v, str)]
        required[section] = cols
    return required


def gather_column_status(df: pd.DataFrame, mapping: Dict) -> Tuple[pd.DataFrame, float]:
    """
    Build a status dataframe with counts of mapped columns present per section and overall coverage.
    """
    req = _mapping_to_required_columns(mapping)
    rows = []
    present_total = 0
    required_total = 0
    df_norm_cols = {normalize_col(c) for c in df.columns}
    for section, cols in req.items():
        norm_cols = [normalize_col(c) for c in cols]
        present = sum(1 for c in norm_cols if c in df_norm_cols)
        total = len(norm_cols) if norm_cols else 0
        pct = (present / total * 100) if total else 0
        rows.append(
            {"section": section, "present": present, "required": total, "coverage_pct": round(pct)}
        )
        present_total += present
        required_total += total
    overall = (present_total / required_total * 100) if required_total else 0
    status_df = pd.DataFrame(rows).sort_values("section").reset_index(drop=True)
    return status_df, round(overall, 1)
