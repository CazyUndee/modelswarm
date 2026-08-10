#!/usr/bin/env python3
"""
Rank Blend Script.

Blend multiple submission files via rank averaging.
Rank blending is robust to differences in prediction scale.

Usage:
    python rank_blend.py --files sub1.csv sub2.csv sub3.csv --weights 0.5 0.3 0.2
    python rank_blend.py --files sub1.csv sub2.csv sub3.csv --output blended.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_blend(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Blend predictions using rank averaging."""
    n = len(predictions[0])
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    blended = np.zeros(n)
    for pred, w in zip(predictions, weights):
        ranks = rankdata(pred, method="average")
        normalized = (ranks - 1) / (len(ranks) - 1)
        blended += w * normalized

    return blended


def main():
    parser = argparse.ArgumentParser(description="Blend submissions via rank averaging")
    parser.add_argument("--files", nargs="+", required=True, help="Submission CSV files")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each file")
    parser.add_argument("--id-col", default="id", help="ID column name")
    parser.add_argument("--pred-col", default="prediction", help="Prediction column name")
    parser.add_argument("--output", default="blended_submission.csv", help="Output file")
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.files):
        raise ValueError("Number of weights must match number of files")

    # Load predictions
    predictions = []
    for f in args.files:
        df = pd.read_csv(f)
        predictions.append(df[args.pred_col].values)

    # Blend
    blended = rank_blend(predictions, args.weights)

    # Save
    result = pd.DataFrame({
        args.id_col: pd.read_csv(args.files[0])[args.id_col],
        args.pred_col: blended,
    })
    result.to_csv(args.output, index=False)

    print(f"Rank blend complete.")
    print(f"Files: {', '.join(args.files)}")
    print(f"Weights: {args.weights or 'equal'}")
    print(f"Output: {args.output}")
    print(f"Samples: {len(result)}")


if __name__ == "__main__":
    main()
