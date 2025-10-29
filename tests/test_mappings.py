from pathlib import Path

import pandas as pd
import pytest
import yaml

from utils.data_loader import load_data, load_mapping, gather_column_status


def test_can_load_mapping():
    mapping = load_mapping(Path("metadata/fields.yaml"))
    assert "identity" in mapping
    assert "geoid_col" in mapping["identity"] or "name_col" in mapping["identity"]

def test_data_directory_exists():
    assert Path("data/raw").exists()

def test_status_coverage_runs():
    df = load_data(Path("data/raw"))
    if df.empty:
        pytest.skip("No CSVs in data/raw; add your StatCan CSVs to run full tests.")
    mapping = load_mapping(Path("metadata/fields.yaml"))
    status_df, coverage = gather_column_status(df, mapping)
    assert "section" in status_df.columns
    assert coverage >= 0
