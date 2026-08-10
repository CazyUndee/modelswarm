#!/usr/bin/env python3
"""
Experiment runner — reads experiment config, trains model, writes results.

This script is designed to run inside GitHub Actions.
It reads an experiment config YAML, trains the model, and writes results back.

Usage:
    python scripts/run_experiment.py --config competitions/s6e8/experiments/EXP-014.yaml --output-dir experiments/output/EXP-014/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(competition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition data."""
    data_dir = Path(f"competitions/{competition}/data")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def apply_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply feature engineering from config."""
    fe = config.get("feature_engineering", {})

    # Interactions
    for interaction in fe.get("interactions", []):
        if len(interaction) == 2:
            name = f"{interaction[0]}_x_{interaction[1]}"
            df[name] = df[interaction[0]] * df[interaction[1]]

    # Ratios
    for ratio in fe.get("ratios", []):
        name = ratio["name"]
        num = ratio["numerator"]
        den = ratio["denominator"]
        df[name] = df[num] / (df[den] + 1e-8)

    return df


def train_model(train: pd.DataFrame, config: dict) -> dict:
    """Train model according to config. Returns results dict."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    model_config = config.get("model", {})
    model_name = model_config.get("name", "lightgbm")
    params = model_config.get("parameters", {})

    features = config.get("features", [])
    target = "addicted_label"  # Should come from competition config

    # Ensure all features exist
    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]
    if missing:
        print(f"⚠️  Missing features (skipping): {missing}")

    X = train[available].fillna(0)
    y = train[target]

    # Validation
    val_config = config.get("validation", {})
    n_folds = val_config.get("n_folds", 5)
    random_state = val_config.get("random_state", 42)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Train model based on type
    oof_preds = np.zeros(len(X))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "eval_metric"})
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 50), verbose=False)]
            )
        elif model_name == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "eval_metric" and k != "tree_method"})
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**{k: v for k, v in params.items() if k != "eval_metric"})
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_metrics.append(fold_auc)
        print(f"  Fold {fold}: AUC = {fold_auc:.5f}")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.5f}")

    return {
        "oof_metric": float(oof_auc),
        "fold_metrics": [float(m) for m in fold_metrics],
        "oof_predictions": oof_preds.tolist(),
        "features_used": available,
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a ModelSwarm experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    experiment_id = config.get("experiment_id", "unknown")
    competition = config.get("competition", "s6e8")

    print(f"{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"Competition: {competition}")
    print(f"Hypothesis: {config.get('hypothesis', 'N/A')}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        train, test = load_data(competition)
        print(f"  Train: {train.shape}")
        print(f"  Test: {test.shape}")
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("Download from Kaggle first:")
        print(f"  kaggle competitions download -c {competition} -p competitions/{competition}/data/")
        sys.exit(1)

    # Feature engineering
    print("\n[2/4] Applying feature engineering...")
    train = apply_feature_engineering(train, config)
    test = apply_feature_engineering(test, config)
    print(f"  Features after engineering: {train.shape[1]}")

    # Train
    print("\n[3/4] Training model...")
    start_time = time.time()
    results = train_model(train, config)
    runtime = time.time() - start_time
    results["runtime_seconds"] = runtime

    # Save OOF predictions
    print("\n[4/4] Saving results...")
    oof_df = pd.DataFrame({
        "id": train["id"],
        "target": train["addicted_label"],
        "prediction": results["oof_predictions"],
    })
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions: {oof_path}")

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Experiment {experiment_id} complete")
    print(f"   OOF AUC: {results['oof_metric']:.5f}")
    print(f"   Fold AUCs: {[f'{m:.5f}' for m in results['fold_metrics']]}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"{'='*60}")

    # Output for GitHub Actions
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"oof_metric={results['oof_metric']}\n")
            f.write(f"runtime={runtime}\n")


if __name__ == "__main__":
    main()
