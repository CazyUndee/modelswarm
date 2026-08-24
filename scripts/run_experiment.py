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
    """Apply feature engineering from config.

    Supports two formats:
    1. Structured op list (preferred):
       feature_engineering:
         - {op: ratio,   name: social_ratio, numerator: social_media_hours, denominator: daily_screen_time_hours}
         - {op: product, name: screen_x_social, terms: [daily_screen_time_hours, social_media_hours]}
         - {op: sum,     name: total_leisure, terms: [social_media_hours, gaming_hours]}
    2. Legacy dict format:
       feature_engineering:
         ratios:
           - {name: ..., numerator: ..., denominator: ...}
    """
    df = df.copy()
    fe = config.get("feature_engineering", []) or []

    def _apply_ratio(name: str, num: str, den: str) -> None:
        df[name] = df[num] / (df[den] + 1e-8)

    ops: list[dict] = []
    if isinstance(fe, list):
        ops = [op for op in fe if isinstance(op, dict)]
    elif isinstance(fe, dict):
        # Legacy format
        for ratio in fe.get("ratios", []):
            ops.append({"op": "ratio", **ratio})

    for op in ops:
        kind = op.get("op", "ratio")
        name = op.get("name")
        if not name:
            print(f"⚠️  Skipping FE op without name: {op}")
            continue
        try:
            if kind == "ratio":
                _apply_ratio(name, op["numerator"], op["denominator"])
            elif kind == "product":
                terms = op["terms"]
                df[name] = df[terms[0]]
                for t in terms[1:]:
                    df[name] = df[name] * df[t]
            elif kind == "sum":
                terms = op["terms"]
                df[name] = df[terms[0]]
                for t in terms[1:]:
                    df[name] = df[name] + df[t]
            elif kind == "diff":
                terms = op["terms"]
                df[name] = df[terms[0]]
                for t in terms[1:]:
                    df[name] = df[name] - df[t]
            else:
                print(f"⚠️  Unknown FE op '{kind}' — skipping")
        except KeyError as e:
            print(f"⚠️  FE op '{name}' references missing column {e} — skipping")

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

    X = train[available].copy()
    y = train[target]

    # Categorical columns → pandas 'category' dtype (native LightGBM handling).
    cat_cols = [c for c in available if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype("category")

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
            ctor = {k: v for k, v in params.items()
                    if k not in ("eval_metric", "early_stopping_rounds")}
            model = lgb.LGBMClassifier(**ctor)
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
            # CatBoost needs NaN-free input: median-fill numerics, encode categoricals.
            X_tr, X_va = X_train.copy(), X_val.copy()
            for c in available:
                if c in cat_cols:
                    codes = pd.Categorical(X[c]).codes
                    X_tr[c] = codes[train_idx]
                    X_va[c] = codes[val_idx]
                else:
                    med = X[c].median()
                    X_tr[c] = X_tr[c].fillna(med)
                    X_va[c] = X_va[c].fillna(med)
            model = CatBoostClassifier(**{k: v for k, v in params.items() if k != "eval_metric"})
            model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
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
        "categorical_features": cat_cols,
        "model_name": model_name,
    }


def predict_test(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> np.ndarray:
    """Train on the full dataset and predict the test set (for submission output)."""
    model_config = config.get("model", {})
    model_name = model_config.get("name", "lightgbm")
    params = model_config.get("parameters", {})

    target = "addicted_label"
    features = [f for f in config.get("features", []) if f in train.columns]
    X, y = train[features].copy(), train[target]
    X_test = test[features].copy()

    if model_name == "lightgbm":
        cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
        for c in cat_cols:
            X[c] = X[c].astype("category")
            X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**{k: v for k, v in params.items()
                                      if k not in ("eval_metric", "early_stopping_rounds")})
        model.fit(X, y, callbacks=[])
        return model.predict_proba(X_test)[:, 1]

    raise NotImplementedError(
        f"Test-set prediction not implemented for model '{model_name}'; "
        "extend predict_test() or set training.predict_test: false in the config."
    )


def main():
    # Windows consoles may default to cp1252 and crash on emoji markers.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Run a ModelSwarm experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Smoke-test only: cap training rows (results NOT valid for research)")
    parser.add_argument("--no-submission", action="store_true",
                        help="Skip full-data test prediction / submission.csv output")
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
    print("\n[1/5] Loading data...")
    try:
        train, test = load_data(competition)
        print(f"  Train: {train.shape}")
        print(f"  Test: {test.shape}")
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("Data must be committed under competitions/<id>/data/ — GitHub Actions has no other source.")
        sys.exit(1)

    if args.max_rows:
        print(f"⚠️  SMOKE TEST MODE: capping train to {args.max_rows} rows (invalid for research)")
        train = train.sample(n=min(args.max_rows, len(train)), random_state=42).reset_index(drop=True)

    # Feature engineering
    print("\n[2/5] Applying feature engineering...")
    train = apply_feature_engineering(train, config)
    test = apply_feature_engineering(test, config)
    print(f"  Features after engineering: {train.shape[1]}")

    # Train (OOF CV)
    print("\n[3/5] Training model (stratified CV)...")
    start_time = time.time()
    results = train_model(train, config)
    runtime = time.time() - start_time
    results["runtime_seconds"] = runtime

    # Save OOF predictions
    print("\n[4/5] Saving results...")
    oof_df = pd.DataFrame({
        "id": train["id"],
        "target": train["addicted_label"],
        "prediction": results["oof_predictions"],
    })
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions: {oof_path}")

    # Full-data test prediction → submission
    submission_note = None
    if not args.no_submission and config.get("training", {}).get("predict_test", True):
        print("  Predicting test set on full data...")
        try:
            test_preds = predict_test(train, test, config)
            submission = pd.DataFrame({"id": test["id"], "addicted_label": test_preds})
            sub_path = output_dir / "submission.csv"
            submission.to_csv(sub_path, index=False)
            print(f"  Submission: {sub_path}")
        except NotImplementedError as e:
            submission_note = str(e)
            print(f"  ⚠️  Skipping submission: {e}")

    # Save results JSON (full OOF vector excluded — it lives in oof_predictions.csv)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "oof_predictions"}, f, indent=2)
    print(f"  Results: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    smoke_tag = " [SMOKE TEST]" if args.max_rows else ""
    print(f"✅ Experiment {experiment_id} complete{smoke_tag}")
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
