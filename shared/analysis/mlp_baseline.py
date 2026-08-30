"""Simple MLP baseline for S6E8 — fast, lightweight NN diversity test.

Hypothesis: A simple multi-layer perceptron with proper regularization
provides genuinely different signal from tree models and from the more
complex TabFM/FT-Transformer architectures.

Protocol:
1. Build OOF predictions with 5-fold CV
2. Test standalone AUC
3. Test blend with existing models
4. Report complementarity metrics
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier

np.random.seed(42)

# Load data
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# Feature columns
feature_cols = ["age", "daily_screen_time_hours", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours",
                "notifications_per_day", "app_opens_per_day",
                "weekend_screen_time", "gender", "stress_level",
                "academic_work_impact"]

X_df = tr[feature_cols].copy()

# Encode all categorical columns
print(f"  Gender values: {X_df['gender'].unique()}")
X_df["gender"] = X_df["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.0).astype(np.float32)
print(f"  Stress values: {X_df['stress_level'].unique()}")
X_df["stress_level"] = X_df["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5).astype(np.float32)
print(f"  Impact values: {X_df['academic_work_impact'].unique()}")
X_df["academic_work_impact"] = X_df["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.0).astype(np.float32)

# Add free_time_slack
X_df["free_time_slack"] = X_df["daily_screen_time_hours"] - X_df["social_media_hours"] - X_df["gaming_hours"] - X_df["work_study_hours"]

# Standardize features
from sklearn.preprocessing import StandardScaler
feature_names = list(X_df.columns)
X_all = X_df.values.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"Data: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")

# ============================================================
# MLP OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("MLP OOF PREDICTIONS (5-fold CV)")
print("=" * 70)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
mlp_oof = np.zeros(N)

for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
    print(f"  Fold {fold+1}/{n_folds}...", end=" ", flush=True)
    
    # Leakage-safe imputation: fill NaNs with TRAIN-fold medians
    # (numeric columns have missing values; MLPClassifier rejects NaN)
    tr_med = np.nanmedian(X_scaled[tr_idx], axis=0)
    tr_X = np.where(np.isnan(X_scaled[tr_idx]), tr_med, X_scaled[tr_idx])
    va_X = np.where(np.isnan(X_scaled[va_idx]), tr_med, X_scaled[va_idx])

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=1024,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(tr_X, y[tr_idx])
    preds = clf.predict_proba(va_X)[:, 1]
    mlp_oof[va_idx] = preds
    
    fold_auc = roc_auc_score(y[va_idx], preds)
    print(f"AUC={fold_auc:.6f}")

mlp_auc = roc_auc_score(y, mlp_oof)
print(f"\nMLP OOF: {mlp_auc:.6f}")

# Save OOF
np.save("shared/artifacts/stacking_vectors/mlp_oof.npy", mlp_oof)
print("Saved mlp_oof.npy")

# ============================================================
# LOAD EXISTING MODELS FOR BLEND TEST
# ============================================================
print("\n" + "=" * 70)
print("BLEND TEST: MLP + existing models")
print("=" * 70)

import kagglehub
V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  lib_{n}: OOF {roc_auc_score(y, V[f'lib_{n}']):.6f}")

V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
V["mlp"] = mlp_oof
print(f"  mlp: OOF {roc_auc_score(y, V['mlp']):.6f}")

# ============================================================
# HELPER
# ============================================================
def fit_weights(cols, y_fit, M_fit):
    Mf = M_fit[:, cols]
    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y_fit * np.log(p) + (1 - y_fit) * np.log(1 - p))
    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 1000, "maxfev": 2000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()

keys = list(V.keys())
M = np.column_stack([V[k] for k in keys])
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

# ============================================================
# BLEND CONFIGURATIONS
# ============================================================
configs = [
    ("MLP alone", ["mlp"]),
    ("Library NN (lookup+tabm)", ["lib_lookup", "lib_tabm_seed3"]),
    ("MLP + Library NN", ["mlp", "lib_lookup", "lib_tabm_seed3"]),
    ("MLP + Owned", ["mlp", "owned_EXP122", "owned_EXP130"]),
    ("MLP + Library + Owned", ["mlp", "lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]),
    ("Full pool", ["mlp", "lib_lookup", "lib_tabm_seed3", "lib_naji03", "owned_EXP122", "owned_EXP130"]),
]

print(f"\n{'Config':<45} {'OOF AUC':>10} {'Held-out':>10} {'Δ vs mlp':>12}")
print("-" * 80)

for name, members in configs:
    cols = [keys.index(m) for m in members]
    if len(members) == 1:
        auc = roc_auc_score(y, V[members[0]])
        held = roc_auc_score(y[hold_idx], V[members[0]][hold_idx])
    else:
        w = fit_weights(cols, y[fit_idx], M[fit_idx])
        pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, pred)
        held = roc_auc_score(y[hold_idx], pred[hold_idx])
    delta = auc - roc_auc_score(y, V["mlp"])
    print(f"{name:<45} {auc:>10.6f} {held:>10.6f} {delta:>+12.6f}")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print(f"\n--- Correlation: MLP vs existing models ---")
for k in ["lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]:
    sp = spearmanr(V["mlp"], V[k]).correlation
    print(f"  mlp ~ {k}: Spearman={sp:.5f}")

print("\n--- DONE ---")
