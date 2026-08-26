"""Test Google TabFM — tabular foundation model for S6E8.

TabFM is a pretrained tabular foundation model that uses in-context
learning. It could provide the missing NN diversity we need.

Protocol:
1. Install TabFM from GitHub (not PyPI)
2. Generate OOF predictions with 5-fold CV
3. Test standalone AUC
4. Test blend with existing models
5. Report complementarity metrics
"""
import os
import subprocess
import sys

# Install TabFM from GitHub
print("Installing TabFM and dependencies...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "safetensors", "git+https://github.com/google-research/tabfm.git"],
               capture_output=False, text=True)
if result.returncode != 0:
    print(f"WARNING: pip install returned {result.returncode}")
print("TabFM install complete.")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.stats import spearmanr

np.random.seed(42)

# Load data
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# Feature columns (same as our models)
feature_cols = ["age", "daily_screen_time_hours", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours",
                "notifications_per_day", "app_opens_per_day",
                "weekend_screen_time", "gender", "stress_level",
                "academic_work_impact"]

X_df = tr[feature_cols].copy()
# Encode gender: Male=1, Female=Other=0.5, Female=0, NaN=0
print(f"  Gender values: {X_df['gender'].unique()}")
gender_map = {"Male": 1.0, "Female": 0.0, "Other": 0.5}
X_df["gender"] = X_df["gender"].map(gender_map).fillna(0.0).astype(np.float32)
print(f"  Encoded gender: Male=1, Female=0, Other=0.5")

X = X_df.values.astype(np.float32)

# Free time slack feature
X_slack = np.column_stack([X, X[:, 1] - X[:, 2] - X[:, 3] - X[:, 4]])  # daily - social - gaming - work
feature_names = feature_cols + ["free_time_slack"]

print(f"Data: {X_slack.shape[0]} rows, {X_slack.shape[1]} features")

# ============================================================
# TABFM OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("TABFM OOF PREDICTIONS (5-fold CV)")
print("=" * 70)

try:
    from tabfm import TabFMClassifier
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    
    print("Loading TabFM model...")
    tabfm_model = tabfm_v1_0_0.load()
    print("TabFM model loaded.")
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    tabfm_oof = np.zeros(N)
    
    # Use DataFrame for TabFM (it expects DataFrame input)
    # Use float32 to reduce memory
    X_df = pd.DataFrame(X_slack.astype(np.float32), columns=feature_names)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
        print(f"  Fold {fold+1}/{n_folds}...", end=" ", flush=True)
        
        clf = TabFMClassifier(model=tabfm_model)
        # Use smaller batch to reduce memory
        clf.fit(X_df.iloc[tr_idx], y[tr_idx])
        preds = clf.predict_proba(X_df.iloc[va_idx])[:, 1]
        tabfm_oof[va_idx] = preds
        
        fold_auc = roc_auc_score(y[va_idx], preds)
        print(f"AUC={fold_auc:.6f}")
        
        # Free memory after each fold
        del clf
        import gc; gc.collect()
    
    tabfm_auc = roc_auc_score(y, tabfm_oof)
    print(f"\nTabFM OOF: {tabfm_auc:.6f}")
    
    # Save OOF
    np.save("shared/artifacts/stacking_vectors/tabfm_oof.npy", tabfm_oof)
    print("Saved tabfm_oof.npy")
    
except Exception as e:
    print(f"TabFM failed: {e}")
    import traceback
    traceback.print_exc()
    print("No foundation model available. Exiting.")
    sys.exit(1)

# ============================================================
# LOAD EXISTING MODELS FOR BLEND TEST
# ============================================================
print("\n" + "=" * 70)
print("BLEND TEST: TabFM + existing models")
print("=" * 70)

import kagglehub
V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  lib_{n}: OOF {roc_auc_score(y, V[f'lib_{n}']):.6f}")

V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
V["tabfm"] = tabfm_oof
print(f"  tabfm: OOF {roc_auc_score(y, V['tabfm']):.6f}")

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
    ("TabFM alone", ["tabfm"]),
    ("Library NN (lookup+tabm)", ["lib_lookup", "lib_tabm_seed3"]),
    ("TabFM + Library NN", ["tabfm", "lib_lookup", "lib_tabm_seed3"]),
    ("TabFM + Owned", ["tabfm", "owned_EXP122", "owned_EXP130"]),
    ("TabFM + Library + Owned", ["tabfm", "lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]),
    ("Full pool", ["tabfm", "lib_lookup", "lib_tabm_seed3", "lib_naji03", "owned_EXP122", "owned_EXP130"]),
]

print(f"\n{'Config':<45} {'OOF AUC':>10} {'Held-out':>10} {'Δ vs tabfm':>12}")
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
    delta = auc - roc_auc_score(y, V["tabfm"])
    print(f"{name:<45} {auc:>10.6f} {held:>10.6f} {delta:>+12.6f}")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print(f"\n--- Correlation: TabFM vs existing models ---")
for k in ["lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]:
    sp = spearmanr(V["tabfm"], V[k]).correlation
    print(f"  tabfm ~ {k}: Spearman={sp:.5f}")

print("\n--- DONE ---")
