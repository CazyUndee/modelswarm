#!/usr/bin/env python3
"""
Data validation script for S6E8.

Run this before any experiment to verify you're using real data.
Prevents agents from fabricating synthetic datasets.

Usage:
    python competitions/s6e8/validate_data.py
"""

import hashlib
import os
import sys

import pandas as pd

DATA_DIR = "competitions/s6e8/data"

# Expected properties of the real dataset (from Kaggle playground-series-s6e8)
# Verified against the actual downloaded files on 2026-08-24.
EXPECTED = {
    "train": {
        "n_rows": 691369,  # approximate — within 10%
        "n_cols": 14,  # id + 12 features + target
        "columns": ["id", "age", "daily_screen_time_hours", "social_media_hours",
                     "gaming_hours", "work_study_hours", "sleep_hours",
                     "notifications_per_day", "app_opens_per_day",
                     "weekend_screen_time", "gender", "stress_level",
                     "academic_work_impact", "addicted_label"],
        "target": "addicted_label",
        "target_values": [0, 1],
    },
    "test": {
        "n_rows": 296302,  # approximate — within 10%
        "n_cols": 13,  # id + 12 features (no target)
        "columns": ["id", "age", "daily_screen_time_hours", "social_media_hours",
                     "gaming_hours", "work_study_hours", "sleep_hours",
                     "notifications_per_day", "app_opens_per_day",
                     "weekend_screen_time", "gender", "stress_level",
                     "academic_work_impact"],
    },
}


def validate_file(name: str) -> bool:
    """Validate a dataset file."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"❌ FAIL: {path} not found")
        return False

    df = pd.read_csv(path)
    expected = EXPECTED[name]

    # Check row count (within 10%)
    row_count = len(df)
    expected_rows = expected["n_rows"]
    if abs(row_count - expected_rows) > expected_rows * 0.1:
        print(f"❌ FAIL: {name}.csv has {row_count} rows, expected ~{expected_rows}")
        return False

    # Check column count
    if len(df.columns) != expected["n_cols"]:
        print(f"❌ FAIL: {name}.csv has {len(df.columns)} columns, expected {expected['n_cols']}")
        return False

    # Check target exists (train only)
    if "target" in expected:
        if expected["target"] not in df.columns:
            print(f"❌ FAIL: target column '{expected['target']}' not found")
            return False

        # Check target values
        unique_vals = set(df[expected["target"]].unique())
        if not unique_vals.issubset(set(expected["target_values"])):
            print(f"❌ FAIL: target has unexpected values: {unique_vals}")
            return False

    # Check for obvious fabrication (all same value, perfect patterns)
    for col in df.columns:
        if col == "id":
            continue
        if df[col].nunique() == 1:
            print(f"⚠️  WARNING: {col} has only one unique value — possible fabrication")
        if df[col].isnull().all():
            print(f"⚠️  WARNING: {col} is all null — possible fabrication")

    print(f"✅ PASS: {name}.csv validated ({row_count} rows, {len(df.columns)} cols)")
    return True


def check_not_synthetic(df: pd.DataFrame, name: str) -> bool:
    """Heuristic checks to detect obviously synthetic data."""
    issues = []

    # Check for perfectly uniform distributions
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if col == "id":
            continue
        # Real data rarely has perfectly round means
        mean_val = df[col].mean()
        if mean_val == int(mean_val) and len(df) > 1000:
            issues.append(f"{col} has suspiciously round mean: {mean_val}")

    # Check for duplicate rows (real data has few exact duplicates)
    dup_pct = df.duplicated().mean() * 100
    if dup_pct > 5:
        issues.append(f"{dup_pct:.1f}% duplicate rows — unusual for real data")

    if issues:
        print(f"⚠️  {name}.csv synthetic data warnings:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    return True


def main():
    # Windows consoles may default to cp1252 and crash on emoji markers.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("=" * 60)
    print("S6E8 Data Validation")
    print("=" * 60)

    all_pass = True
    for name in ["train", "test"]:
        if not validate_file(name):
            all_pass = False
        else:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
            check_not_synthetic(df, name)

    print("=" * 60)
    if all_pass:
        print("✅ All validations passed — data appears legitimate")
    else:
        print("❌ Validation failed — DO NOT use this data for experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()
