from __future__ import annotations

import glob
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
    # 1) Preferred encodings
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            pass
    # 2) Auto-sniff delimiter
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        pass
    # 3) Semicolon fallback (some StatCan extracts)
    return pd.read_csv(path, sep=";", low_memory=False)


def load_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load one or multiple CSVs under data/raw. If multiple, vertically concatenate and align columns.

    Improvements:
    - Case-insensitive extension match (.csv, .CSV, etc.)
    - Recursive into subfolders
    - Accept compressed CSVs (.csv.gz, .csv.zip) that pandas can read
    """
    raw_dir = Path(raw_dir)

    def is_csv_like(p: Path) -> bool:
        low = p.name.lower()
        return (
            p.is_file()
            and (
                low.endswith(".csv")
                or low.endswith(".csv.gz")
                or low.endswith(".csv.zip")
            )
        )

    csvs = sorted([p for p in raw_dir.rglob("*") if is_csv_like(p)])
    if not csvs:
        return pd.DataFrame()

    frames = []
    all_cols: set = set()
    for p in csvs:
        df = _read_csv_any(p)
        # Strip BOM/whitespace in headers and normalize for matching
        df.columns = [normalize_col(c) for c in df.columns]
        frames.append(df)
        all_cols.update(df.columns)

    # Align columns across files
    aligned_frames: List[pd.DataFrame] = []
    all_cols_list = list(all_cols)
    for df in frames:
        for c in all_cols_list:
            if c not in df.columns:
                df[c] = pd.NA
        aligned_frames.append(df[all_cols_list])

    merged = pd.concat(aligned_frames, axis=0, ignore_index=True)
    return merged


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

    # Try looser matching: remove punctuation
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
        # Fall back to GeoUID as a string
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
            {
                "section": section,
                "present": present,
                "required": total,
                "coverage_pct": round(pct),
            }
        )
        present_total += present
        required_total += total
    overall = (present_total / required_total * 100) if required_total else 0
    status_df = pd.DataFrame(rows).sort_values("section").reset_index(drop=True)
    return status_df, round(overall, 1)
