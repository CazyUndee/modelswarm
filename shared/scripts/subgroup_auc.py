#!/usr/bin/env python3
"""
Subgroup AUC Analysis Script.

Compute AUC for subgroups to identify where models perform well or poorly.

Usage:
    python subgroup_auc.py --predictions preds.csv --group-col age_group
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_subgroup_auc(df: pd.DataFrame, group_col: str,
                          target_col: str = "target",
                          pred_col: str = "prediction") -> pd.DataFrame:
    """Compute AUC for each subgroup."""
    results = []
    for group_name, group_df in df.groupby(group_col):
        if len(group_df[target_col].unique()) < 2:
            continue
        auc = roc_auc_score(group_df[target_col], group_df[pred_col])
        results.append({
            "group": group_name,
            "n_samples": len(group_df),
            "positive_rate": group_df[target_col].mean(),
            "auc": auc,
        })

    return pd.DataFrame(results).sort_values("auc", ascending=True)


def main():
    parser = argparse.ArgumentParser(description="Compute subgroup AUC")
    parser.add_argument("--predictions", required=True, help="CSV with predictions and groups")
    parser.add_argument("--group-col", required=True, help="Column to group by")
    parser.add_argument("--target-col", default="target", help="Target column name")
    parser.add_argument("--pred-col", default="prediction", help="Prediction column name")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    results = compute_subgroup_auc(df, args.group_col, args.target_col, args.pred_col)

    print("Subgroup AUC Analysis")
    print("=" * 50)
    print(f"{'Group':<20} {'N':>8} {'Pos Rate':>10} {'AUC':>8}")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"{row['group']:<20} {row['n_samples']:>8} {row['positive_rate']:>10.3f} {row['auc']:>8.4f}")


if __name__ == "__main__":
    main()
