"""Train models on the ORIGINAL dataset and predict the competition test set.

Hypothesis: Models trained on the real (original) data distribution may
capture patterns that synthetic-data-trained models miss. The original
dataset has a different distribution from the synthetic competition data,
and predictions based on it carry genuinely different signal.

This is the simplest form of "original data leakage" that many
playground competition winners exploit.

Run: gh workflow run analysis.yml -f script_path=shared/analysis/orig_train_member.py
"""
import sys
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
print("=== ORIGINAL-DATASET TRAINING MEMBER ===", flush=True)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test_df = pd.read_csv("competitions/s6e8/data/test.csv")

feature_cols = ["age", "daily_screen_time_hours", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours",
                "notifications_per_day", "app_opens_per_day",
                "weekend_screen_time", "gender", "stress_level",
                "academic_work_impact"]

# ============================================================
# 1. LOAD AND PREPARE ORIGINAL DATASET
# ============================================================
print("\nLoading original dataset...", flush=True)
import kagglehub
orig_root = kagglehub.dataset_download("jayjoshi37/smartphone-usage-and-addiction-prediction")
orig = pd.read_csv(os.path.join(orig_root, "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"))
print(f"Original: {orig.shape}", flush=True)
print(f"Original label rate: {orig['addicted_label'].mean():.4f}", flush=True)
print(f"Original addiction_level distribution:", flush=True)
print(orig["addiction_level"].value_counts(dropna=False).to_string(), flush=True)

# Prep original features same as competition
orig_X = orig[feature_cols].copy()
orig_y = orig["addicted_label"].values

# Map categoricals
orig_X["gender"] = orig_X["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.5).astype(np.float32)
orig_X["stress_level"] = orig_X["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5).astype(np.float32)
orig_X["academic_work_impact"] = orig_X["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.0).astype(np.float32)
for col in orig_X.columns:
    orig_X[col] = orig_X[col].astype(float)
orig_X = orig_X.fillna(orig_X.median(numeric_only=True))

# Prep competition features
tr_X = tr[feature_cols].copy()
tr_X["gender"] = tr_X["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.5).astype(np.float32)
tr_X["stress_level"] = tr_X["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5).astype(np.float32)
tr_X["academic_work_impact"] = tr_X["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.0).astype(np.float32)
for col in tr_X.columns:
    tr_X[col] = tr_X[col].astype(float)
tr_X = tr_X.fillna(tr_X.median(numeric_only=True))

test_X = test_df[feature_cols].copy()
test_X["gender"] = test_X["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.5).astype(np.float32)
test_X["stress_level"] = test_X["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5).astype(np.float32)
test_X["academic_work_impact"] = test_X["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.0).astype(np.float32)
for col in test_X.columns:
    test_X[col] = test_X[col].astype(float)
test_X = test_X.fillna(test_X.median(numeric_only=True))

print(f"\nOriginal: {orig_X.shape}, Competition train: {tr_X.shape}, Test: {test_X.shape}", flush=True)

# ============================================================
# 2. STRATEGY A: Train on original only, predict competition test
# ============================================================
print(f"\n{'='*60}", flush=True)
print("STRATEGY A: Train on original (7500 rows), predict competition test", flush=True)
print(f"{'='*60}", flush=True)

try:
    import lightgbm as lgb
    import xgboost as xgb_lib

    # LightGBM trained on original
    print("\nLightGBM (trained on original):", flush=True)
    dtrain_orig = lgb.Dataset(orig_X.values, label=orig_y)
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 15,
        "min_child_samples": 20, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    lgb_orig = lgb.train(params, dtrain_orig, num_boost_round=500)

    # Predict on competition train for OOF proxy
    lgb_tr_pred = lgb_orig.predict(tr_X.values)
    # Predict on competition test
    lgb_test_pred = lgb_orig.predict(test_X.values)

    # OOF on competition train (using original-trained model — this is a proxy)
    # We can't do proper OOF since we didn't train on competition data
    # But we can check if the predictions are meaningful
    train_auc = roc_auc_score(y, lgb_tr_pred)
    print(f"  Competition train AUC (original-trained model): {train_auc:.6f}", flush=True)
    print(f"  Test prediction range: [{lgb_test_pred.min():.4f}, {lgb_test_pred.max():.4f}]", flush=True)

    # XGBoost trained on original
    print("\nXGBoost (trained on original):", flush=True)
    dtrain_orig_xgb = xgb_lib.DMatrix(orig_X.values, label=orig_y)
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "learning_rate": 0.03, "max_depth": 4,
        "min_child_weight": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 1.0,
        "reg_lambda": 1.0, "verbosity": 0, "seed": 42,
    }
    xgb_orig = xgb_lib.train(xgb_params, dtrain_orig_xgb, num_boost_round=500)

    xgb_tr_pred = xgb_orig.predict(xgb_lib.DMatrix(tr_X.values))
    xgb_test_pred = xgb_orig.predict(xgb_lib.DMatrix(test_X.values))
    train_auc_xgb = roc_auc_score(y, xgb_tr_pred)
    print(f"  Competition train AUC: {train_auc_xgb:.6f}", flush=True)
    print(f"  Test prediction range: [{xgb_test_pred.min():.4f}, {xgb_test_pred.max():.4f}]", flush=True)

except Exception as e:
    import traceback
    print(f"FAILED: {e}", flush=True)
    traceback.print_exc()
    lgb_test_pred = None
    xgb_test_pred = None

# ============================================================
# 3. STRATEGY B: Combined training (original + competition)
# ============================================================
print(f"\n{'='*60}", flush=True)
print("STRATEGY B: Train on combined (original + competition train), predict test", flush=True)
print(f"{'='*60}", flush=True)

try:
    # Combine original and competition train
    combined_X = np.vstack([orig_X.values, tr_X.values])
    combined_y = np.concatenate([orig_y, y])
    print(f"Combined: {combined_X.shape[0]} rows, label rate: {combined_y.mean():.4f}", flush=True)

    # Cross-validate on competition train portion only
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(N_TRAIN), y))

    oof_combined_lgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        # Train on: all original + competition train fold fi
        train_X = np.vstack([orig_X.values, tr_X.values[fi]])
        train_y = np.concatenate([orig_y, y[fi]])

        dtrain = lgb.Dataset(train_X, label=train_y)
        model = lgb.train(params, dtrain, num_boost_round=500)
        oof_combined_lgb[vi] = model.predict(tr_X.values[vi])

    combined_lgb_auc = roc_auc_score(y, oof_combined_lgb)
    print(f"\nCombined LGB OOF (on competition train): {combined_lgb_auc:.6f}", flush=True)

    # Full combined training for test
    dtrain_full = lgb.Dataset(combined_X, label=combined_y)
    lgb_combined_full = lgb.train(params, dtrain_full, num_boost_round=500)
    lgb_combined_test = lgb_combined_full.predict(test_X.values)
    print(f"Combined LGB test range: [{lgb_combined_test.min():.4f}, {lgb_combined_test.max():.4f}]", flush=True)

    # XGBoost combined
    oof_combined_xgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        train_X = np.vstack([orig_X.values, tr_X.values[fi]])
        train_y = np.concatenate([orig_y, y[fi]])
        dtrain = xgb_lib.DMatrix(train_X, label=train_y)
        model = xgb_lib.train(xgb_params, dtrain, num_boost_round=500)
        oof_combined_xgb[vi] = model.predict(xgb_lib.DMatrix(tr_X.values[vi]))

    combined_xgb_auc = roc_auc_score(y, oof_combined_xgb)
    print(f"Combined XGB OOF: {combined_xgb_auc:.6f}", flush=True)

    # Full combined XGB for test
    dtrain_full = xgb_lib.DMatrix(combined_X, label=combined_y)
    xgb_combined_full = xgb_lib.train(xgb_params, dtrain_full, num_boost_round=500)
    xgb_combined_test = xgb_combined_full.predict(xgb_lib.DMatrix(test_X.values))

except Exception as e:
    import traceback
    print(f"FAILED: {e}", flush=True)
    traceback.print_exc()
    lgb_combined_test = None
    xgb_combined_test = None

# ============================================================
# 4. SAVE RESULTS
# ============================================================
print(f"\n{'='*60}", flush=True)
print("SAVE RESULTS", flush=True)
print(f"{'='*60}", flush=True)

# Save the best combined models as members for the stack
if 'lgb_combined_test' in dir() and lgb_combined_test is not None:
    np.save("shared/analysis/data/oof_orig_combined_lgb.npy", oof_combined_lgb)
    np.save("shared/analysis/data/test_orig_combined_lgb.npy", lgb_combined_test)
    print("Saved oof_orig_combined_lgb.npy + test_orig_combined_lgb.npy", flush=True)

if 'xgb_combined_test' in dir() and xgb_combined_test is not None:
    np.save("shared/analysis/data/oof_orig_combined_xgb.npy", oof_combined_xgb)
    np.save("shared/analysis/data/test_orig_combined_xgb.npy", xgb_combined_test)
    print("Saved oof_orig_combined_xgb.npy + test_orig_combined_xgb.npy", flush=True)

# Also save the original-only models (no proper OOF, but test predictions available)
if lgb_test_pred is not None:
    np.save("shared/analysis/data/test_orig_only_lgb.npy", lgb_test_pred)
    print("Saved test_orig_only_lgb.npy (original-only, no OOF)", flush=True)
if xgb_test_pred is not None:
    np.save("shared/analysis/data/test_orig_only_xgb.npy", xgb_test_pred)
    print("Saved test_orig_only_xgb.npy (original-only, no OOF)", flush=True)

# ============================================================
# 5. SUMMARY
# ============================================================
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Current stack OOF: 0.970221", flush=True)
if 'combined_lgb_auc' in dir():
    print(f"Combined LGB (orig+comp) OOF: {combined_lgb_auc:.6f} (delta: {combined_lgb_auc - 0.970221:+.6f})", flush=True)
if 'combined_xgb_auc' in dir():
    print(f"Combined XGB (orig+comp) OOF: {combined_xgb_auc:.6f} (delta: {combined_xgb_auc - 0.970221:+.6f})", flush=True)
print(f"\nOriginal-only model train AUC: {train_auc:.6f}", flush=True)
print(f"Original-only model captures patterns from real data distribution.", flush=True)

print(f"\n=== DONE ===", flush=True)
