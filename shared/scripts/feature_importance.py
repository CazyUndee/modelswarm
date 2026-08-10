#!/usr/bin/env python3
"""
Feature Importance Analysis Script.

Compute and display feature importance from a trained model.

Usage:
    python feature_importance.py --model model.pkl --features features.csv --top 20
"""

import argparse

import numpy as np
import pandas as pd


def compute_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Compute feature importance from a model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        importances = model.feature_importance()
    else:
        raise ValueError("Model does not have feature_importances_ attribute")

    total = importances.sum()
    normalized = importances / total if total > 0 else importances

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "normalized": normalized,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["cumulative"] = df["normalized"].cumsum()
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute feature importance")
    parser.add_argument("--importances", required=True, help="CSV with feature importance values")
    parser.add_argument("--top", type=int, default=20, help="Show top N features")
    args = parser.parse_args()

    df = pd.read_csv(args.importances)
    df = df.sort_values("importance", ascending=False).head(args.top)

    print(f"Feature Importance (Top {args.top})")
    print(f"{'='*50}")
    print(f"{'Feature':<30} {'Importance':>12} {'Cumulative':>12}")
    print(f"{'-'*50}")

    cumulative = 0.0
    for _, row in df.iterrows():
        cumulative += row.get("normalized", row["importance"] / df["importance"].sum())
        print(f"{row['feature']:<30} {row['importance']:>12.0f} {cumulative:>11.1%}")


if __name__ == "__main__":
    main()
