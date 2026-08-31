"""Original-dataset reference features for S6E8.

Hypothesis: The original smartphone addiction dataset (7500 rows) has a
different distribution from the synthetic competition data. Models that
exploit this distributional difference via CDF-based features carry
genuinely new signal not captured by any existing member.

This is the approach used by kodaifukuda0311 (RealMLP author) who
reported strong CV scores using ORIG-derived features.

Run: gh workflow run analysis.yml -f script_path=shared/analysis/orig_reference_features.py
"""
import sys
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
print("=== ORIGINAL-DATASET REFERENCE FEATURES ===", flush=True)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test = pd.read_csv("competitions/s6e8/data/test.csv")

feature_cols = [c for c in tr.columns if c not in ("id", "addicted_label")]
print(f"Features: {feature_cols}", flush=True)

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================
print("\nLoading original dataset...", flush=True)
import kagglehub
orig_root = kagglehub.dataset_download("jayjoshi37/smartphone-usage-and-addiction-prediction")
orig = pd.read_csv(os.path.join(orig_root, "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"))
print(f"Original: {orig.shape}", flush=True)
print(f"Original label rate: {orig['addicted_label'].mean():.4f}", flush=True)

# Map categorical columns the same way as competition data
for df in [orig, tr, test]:
    df["gender_enc"] = df["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.5)
    df["stress_enc"] = df["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5)
    df["impact_enc"] = df["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.5)

numeric_cols = ["age", "daily_screen_time_hours", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours",
                "notifications_per_day", "app_opens_per_day",
                "weekend_screen_time"]
cat_cols = ["gender_enc", "stress_enc", "impact_enc"]
all_model_cols = numeric_cols + cat_cols

# ============================================================
# 2. COMPUTE REFERENCE STATISTICS FROM ORIGINAL DATASET
# ============================================================
print("\nComputing reference statistics from original dataset...", flush=True)

# Overall CDF: for each value, what fraction of original data is <= this value?
# This tells us how "typical" or "extreme" a value is relative to the original distribution
orig_numeric = orig[numeric_cols].values
orig_target = orig["addicted_label"].values.astype(int)

# Class-conditional statistics
orig_addicted = orig[orig["addicted_label"] == 1]
orig_not_addicted = orig[orig["addicted_label"] == 0]
print(f"Original: {len(orig_addicted)} addicted, {len(orig_not_addicted)} not addicted", flush=True)

def compute_cdf_features(values, orig_ref):
    """Compute CDF-based features: what fraction of original data is <= each value."""
    features = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        col_vals = values[:, j]
        ref_vals = orig_ref[:, j]
        for i in range(len(values)):
            features[i, j] = np.mean(ref_vals <= col_vals[i])
    return features

def compute_conditional_cdf_features(values, orig_pos, orig_neg):
    """Compute class-conditional CDF features."""
    features_pos = np.zeros((len(values), len(numeric_cols)))
    features_neg = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        col_vals = values[:, j]
        for i in range(len(values)):
            features_pos[i, j] = np.mean(orig_pos[:, j] <= col_vals[i])
            features_neg[i, j] = np.mean(orig_neg[:, j] <= col_vals[i])
    return features_pos, features_neg

def compute_frequency_encoding(values, orig_values):
    """Count how often each exact value appears in the original dataset."""
    freq = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        unique_vals, counts = np.unique(orig_values[:, j], return_counts=True)
        freq_map = dict(zip(unique_vals, counts / len(orig_values)))
        for i in range(len(values)):
            freq[i, j] = freq_map.get(values[i, j], 0)
    return freq

def compute_class_frequency_ratio(values, orig_pos, orig_neg):
    """Ratio of P(value|addicted) / P(value|not_addicted) from original."""
    ratio = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        unique_pos, counts_pos = np.unique(orig_pos[:, j], return_counts=True)
        unique_neg, counts_neg = np.unique(orig_neg[:, j], return_counts=True)
        freq_pos = dict(zip(unique_pos, counts_pos / len(orig_pos)))
        freq_neg = dict(zip(unique_neg, counts_neg / len(orig_neg)))
        for i in range(len(values)):
            p_pos = freq_pos.get(values[i, j], 1e-6)
            p_neg = freq_neg.get(values[i, j], 1e-6)
            ratio[i, j] = np.log(p_pos / p_neg + 1e-6)
    return ratio

# Compute features for train, test, and original
print("Computing CDF features (slow — one-by-one)...", flush=True)

# For speed, use vectorized percentile-based CDF approximation
def fast_cdf_features(values, orig_ref):
    """Vectorized CDF approximation using numpy searchsorted."""
    features = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        sorted_ref = np.sort(orig_ref[:, j])
        features[:, j] = np.searchsorted(sorted_ref, values[:, j]) / len(sorted_ref)
    return features

def fast_conditional_cdf(values, orig_pos, orig_neg):
    features_pos = np.zeros((len(values), len(numeric_cols)))
    features_neg = np.zeros((len(values), len(numeric_cols)))
    for j in range(len(numeric_cols)):
        sorted_pos = np.sort(orig_pos[:, j])
        sorted_neg = np.sort(orig_neg[:, j])
        features_pos[:, j] = np.searchsorted(sorted_pos, values[:, j]) / len(sorted_pos)
        features_neg[:, j] = np.searchsorted(sorted_neg, values[:, j]) / len(sorted_neg)
    return features_pos, features_neg

train_vals = tr[numeric_cols].values.astype(float)
test_vals = test[numeric_cols].values.astype(float)
orig_vals = orig_numeric.values.astype(float)
orig_pos_vals = orig_addicted[numeric_cols].values.astype(float)
orig_neg_vals = orig_not_addicted[numeric_cols].values.astype(float)

# Fill NaN with median for CDF computation
for j in range(train_vals.shape[1]):
    med = np.nanmedian(train_vals[:, j])
    train_vals[np.isnan(train_vals[:, j]), j] = med
    test_vals[np.isnan(test_vals[:, j]), j] = med

# Features
print("  Overall CDF...", flush=True)
cdf_train = fast_cdf_features(train_vals, orig_vals)
cdf_test = fast_cdf_features(test_vals, orig_vals)

print("  Class-conditional CDF...", flush=True)
ccdf_pos_train, ccdf_neg_train = fast_conditional_cdf(train_vals, orig_pos_vals, orig_neg_vals)
ccdf_pos_test, ccdf_neg_test = fast_conditional_cdf(test_vals, orig_pos_vals, orig_neg_vals)

# Class-conditional ratio features
ccdf_ratio_train = ccdf_pos_train - ccdf_neg_train
ccdf_ratio_test = ccdf_pos_test - ccdf_neg_test

# Missing-value features
miss_train = tr[numeric_cols].isna().astype(float).values
miss_test = test[numeric_cols].isna().astype(float).values

# Combine all features
X_train = np.column_stack([
    cdf_train,           # 9 features: overall CDF
    ccdf_pos_train,      # 9 features: CDF among addicted
    ccdf_neg_train,      # 9 features: CDF among not-addicted
    ccdf_ratio_train,    # 9 features: class-conditional ratio
    miss_train,          # 9 features: missing flags
    train_vals,          # 9 features: raw values
]).astype(np.float32)

X_test = np.column_stack([
    cdf_test,
    ccdf_pos_test,
    ccdf_neg_test,
    ccdf_ratio_test,
    miss_test,
    test_vals,
]).astype(np.float32)

print(f"Feature matrix: train={X_train.shape}, test={X_test.shape}", flush=True)

# ============================================================
# 3. TRAIN MODELS WITH OOF VALIDATION
# ============================================================
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
    np.zeros(N_TRAIN), y))

# --- LightGBM ---
print("\n=== LightGBM on orig-reference features ===", flush=True)
try:
    import lightgbm as lgb

    oof_lgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/5...", end=" ", flush=True)
        dtrain = lgb.Dataset(X_train[fi], label=y[fi])
        dval = lgb.Dataset(X_train[vi], label=y[vi], reference=dtrain)
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 500, "feature_fraction": 0.5,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        }
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[vi] = model.predict(X_train[vi])
        print(f"AUC={roc_auc_score(y[vi], oof_lgb[vi]):.6f} iters={model.best_iteration}", flush=True)

    lgb_auc = roc_auc_score(y, oof_lgb)
    print(f"\nLightGBM OOF: {lgb_auc:.6f}", flush=True)

    # Full-data LightGBM for test
    print("Training full LightGBM for test...", flush=True)
    dtrain_full = lgb.Dataset(X_train, label=y)
    lgb_full = lgb.train(params, dtrain_full, num_boost_round=1000)
    lgb_test = lgb_full.predict(X_test)
    print(f"Test range: [{lgb_test.min():.4f}, {lgb_test.max():.4f}]", flush=True)

    # Save OOF and test as .npy for the stack
    np.save("shared/analysis/data/oof_orig_lgb.npy", oof_lgb)
    np.save("shared/analysis/data/test_orig_lgb.npy", lgb_test)
    print("Saved oof_orig_lgb.npy and test_orig_lgb.npy", flush=True)

except Exception as e:
    import traceback
    print(f"LightGBM FAILED: {e}", flush=True)
    traceback.print_exc()
    lgb_auc = 0

# --- XGBoost ---
print("\n=== XGBoost on orig-reference features ===", flush=True)
try:
    import xgboost as xgb_lib

    oof_xgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/5...", end=" ", flush=True)
        dtrain = xgb_lib.DMatrix(X_train[fi], label=y[fi])
        dval = xgb_lib.DMatrix(X_train[vi], label=y[vi])
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.03, "max_depth": 4,
            "min_child_weight": 500, "subsample": 0.8,
            "colsample_bytree": 0.5, "reg_alpha": 1.0,
            "reg_lambda": 1.0, "verbosity": 0, "seed": 42,
        }
        model = xgb_lib.train(
            params, dtrain, num_boost_round=2000,
            evals=[(dval, "val")], early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_xgb[vi] = model.predict(dval)
        print(f"AUC={roc_auc_score(y[vi], oof_xgb[vi]):.6f}", flush=True)

    xgb_auc = roc_auc_score(y, oof_xgb)
    print(f"\nXGBoost OOF: {xgb_auc:.6f}", flush=True)

    # Full-data XGBoost for test
    print("Training full XGBoost for test...", flush=True)
    dtrain_full = xgb_lib.DMatrix(X_train, label=y)
    xgb_full = xgb_lib.train(params, dtrain_full, num_boost_round=1000)
    xgb_test = xgb_full.predict(xgb_lib.DMatrix(X_test))
    print(f"Test range: [{xgb_test.min():.4f}, {xgb_test.max():.4f}]", flush=True)

    # Save
    np.save("shared/analysis/data/oof_orig_xgb.npy", oof_xgb)
    np.save("shared/analysis/data/test_orig_xgb.npy", xgb_test)
    print("Saved oof_orig_xgb.npy and test_orig_xgb.npy", flush=True)

except Exception as e:
    import traceback
    print(f"XGBoost FAILED: {e}", flush=True)
    traceback.print_exc()
    xgb_auc = 0

# ============================================================
# 4. CORRELATION WITH EXISTING STACK MEMBERS
# ============================================================
print(f"\n{'='*60}", flush=True)
print("CORRELATION WITH EXISTING STACK MEMBERS", flush=True)
print(f"{'='*60}", flush=True)

# Load the stack's best member for comparison
try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    bolt_oof = pd.read_parquet(os.path.join(bolt, "oof_predictions.parquet"))
    # Pick the best bolt member
    bolt_names = [c for c in bolt_oof.columns if c != "id"]
    bolt_aucs = [roc_auc_score(y, bolt_oof[c].values) for c in bolt_names]
    best_bolt_idx = np.argmax(bolt_aucs)
    best_bolt = bolt_oof[bolt_names[best_bolt_idx]].values
    print(f"Best bolt member: {bolt_names[best_bolt_idx]} (AUC={bolt_aucs[best_bolt_idx]:.6f})", flush=True)
except:
    best_bolt = None

# Compare our new models with existing stack
if lgb_auc > 0:
    print(f"\nLightGBM orig-ref features OOF: {lgb_auc:.6f}", flush=True)
if xgb_auc > 0:
    print(f"XGBoost orig-ref features OOF: {xgb_auc:.6f}", flush=True)

# ============================================================
# 5. SUMMARY
# ============================================================
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Current stack OOF: 0.970221", flush=True)
if lgb_auc > 0:
    print(f"LightGBM orig-ref OOF: {lgb_auc:.6f} (delta vs stack: {lgb_auc - 0.970221:+.6f})", flush=True)
if xgb_auc > 0:
    print(f"XGBoost orig-ref OOF: {xgb_auc:.6f} (delta vs stack: {xgb_auc - 0.970221:+.6f})", flush=True)

print(f"\nHypothesis: ORIG-distribution features carry signal not in existing 252 members.", flush=True)
print(f"If OOF > 0.968, this member is worth adding to the rank-gauss stack.", flush=True)
print(f"If correlation with existing members is low, it could shift the stack OOF.", flush=True)

print(f"\n=== DONE ===", flush=True)
