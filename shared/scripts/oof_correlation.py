#!/usr/bin/env python3
"""
OOF Correlation Analysis Script.

Compute Pearson correlation between two OOF prediction files.
Lower correlation = more complementary = better ensemble potential.

Usage:
    python oof_correlation.py --file-a preds_a.csv --file-b preds_b.csv
    python oof_correlation.py --file-a preds_a.csv --file-b preds_b.csv --target-col target
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser(description="Compute OOF correlation between two prediction files")
    parser.add_argument("--file-a", required=True, help="Path to first OOF predictions CSV")
    parser.add_argument("--file-b", required=True, help="Path to second OOF predictions CSV")
    parser.add_argument("--pred-col", default="prediction", help="Column name for predictions")
    parser.add_argument("--id-col", default="id", help="Column name for sample IDs")
    args = parser.parse_args()

    df_a = pd.read_csv(args.file_a)
    df_b = pd.read_csv(args.file_b)

    # Align by ID
    merged = df_a.merge(df_b, on=args.id_col, suffixes=("_a", "_b"))
    pred_col_a = f"{args.pred_col}_a"
    pred_col_b = f"{args.pred_col}_b"

    corr, p_value = pearsonr(merged[pred_col_a], merged[pred_col_b])

    print(f"OOF Correlation Analysis")
    print(f"{'='*40}")
    print(f"File A: {args.file_a}")
    print(f"File B: {args.file_b}")
    print(f"Samples: {len(merged)}")
    print(f"Pearson correlation: {corr:.6f}")
    print(f"P-value: {p_value:.2e}")
    print(f"")
    if corr < 0.5:
        print(f"LOW correlation — models are complementary (good for ensembling)")
    elif corr < 0.8:
        print(f"MEDIUM correlation — some complementarity")
    else:
        print(f"HIGH correlation — models capture similar signal")


if __name__ == "__main__":
    main()
