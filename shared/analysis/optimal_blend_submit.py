"""Optimal Blend Submission Generator — loads all available OOF vectors,
computes pairwise correlations, finds the best blend, and generates a
submission.csv for Kaggle.

Runs on GHA where kagglehub is available.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import spearmanr
import kagglehub
import glob
import json

np.random.seed(42)

# ============================================================
# LOAD TRAINING LABELS
# ============================================================
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
te = pd.read_csv("competitions/s6e8/data/test.csv")
N = len(y)
Nt = len(te)
print(f"Train: {N} rows, Test: {Nt} rows")

V = {}  # OOF predictions
T = {}  # Test predictions

# ============================================================
# LOAD LIBRARY VECTORS (public OOF library)
# ============================================================
print("\n=== LIBRARY VECTORS ===")
try:
    lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
    print(f"Library: {lib_dir}")
    lib_members = ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb",
                   "latr1_cat", "digit_cat", "latr1_lgbm", "digit_lgbm"]
    for n in lib_members:
        oof_path = os.path.join(lib_dir, "oof", f"oof_{n}.npy")
        test_path = os.path.join(lib_dir, "oof", f"test_{n}.npy")
        if os.path.exists(oof_path) and os.path.exists(test_path):
            V[f"lib_{n}"] = np.load(oof_path)
            T[f"lib_{n}"] = np.load(test_path)
            auc = roc_auc_score(y, V[f"lib_{n}"])
            print(f"  lib_{n}: OOF {auc:.6f}")
except Exception as e:
    print(f"  Library download failed: {e}")

# ============================================================
# LOAD OWNED VECTORS (from experiment artifacts)
# ============================================================
print("\n=== OWNED VECTORS ===")

# EXP-135 just completed - load its individual member OOFs
for member_file, member_name in [
    ("shared/artifacts/exp135/oof_member_lightgbm[0].csv", "owned_EXP135_lgbm"),
    ("shared/artifacts/exp135/oof_member_logistic[1].csv", "owned_EXP135_logistic"),
]:
    if os.path.exists(member_file):
        df = pd.read_csv(member_file)
        pred_col = "prediction" if "prediction" in df.columns else df.columns[-1]
        V[member_name] = df[pred_col].values
        print(f"  {member_name}: OOF {roc_auc_score(y, V[member_name]):.6f}")

# EXP-119 artifacts (champion)
for art_path in [
    "shared/artifacts/exp135/oof_predictions.csv",
]:
    if os.path.exists(art_path):
        df = pd.read_csv(art_path)
        pred_col = "prediction" if "prediction" in df.columns else df.columns[-1]
        V["owned_EXP135_blend"] = df[pred_col].values
        print(f"  owned_EXP135_blend: OOF {roc_auc_score(y, V['owned_EXP135_blend']):.6f}")

# ============================================================
# PAIRWISE SPEARMAN CORRELATIONS
# ============================================================
print("\n=== PAIRWISE CORRELATIONS ===")
keys = list(V.keys())
print(f"\nAvailable models ({len(keys)}):")
for k in keys:
    auc = roc_auc_score(y, V[k])
    source = "LIB" if k.startswith("lib_") else "OWN"
    print(f"  {k:<35} OOF={auc:.6f}  [{source}]")

# Find best single model
best_single = max(keys, key=lambda k: roc_auc_score(y, V[k]))
best_auc = roc_auc_score(y, V[best_single])
print(f"\nBest single: {best_single} = {best_auc:.6f}")

# ============================================================
# OPTIMIZED BLEND
# ============================================================
print("\n=== OPTIMIZED BLEND ===")

M = np.column_stack([V[k] for k in keys])
n_models = len(keys)

def neg_auc(weights):
    pred = np.clip(M @ weights, 1e-9, 1 - 1e-9)
    return -roc_auc_score(y, pred)

# Uniform blend
uniform_pred = np.mean(M, axis=1)
uniform_auc = roc_auc_score(y, uniform_pred)
print(f"Uniform blend: {uniform_auc:.6f}")

# Optimal weights
result = minimize(neg_auc, np.ones(n_models) / n_models, method="Nelder-Mead",
                  options={"maxiter": 10000})
opt_weights = result.x / result.x.sum()
opt_pred = np.clip(M @ opt_weights, 1e-9, 1 - 1e-9)
opt_auc = roc_auc_score(y, opt_pred)
print(f"Optimized blend: {opt_auc:.6f}")

print("\nOptimal weights:")
for k, w in zip(keys, opt_weights):
    if w > 0.01:
        print(f"  {k:<35} {w:.4f}")

# ============================================================
# RANK-AVERAGE BLEND (often better than probability average)
# ============================================================
print("\n=== RANK-AVERAGE BLEND ===")
M_rank = np.column_stack([pd.Series(V[k]).rank(pct=True).values for k in keys])

def neg_auc_rank(weights):
    pred = np.clip(M_rank @ weights, 1e-9, 1 - 1e-9)
    return -roc_auc_score(y, pred)

result_rank = minimize(neg_auc_rank, np.ones(n_models) / n_models, method="Nelder-Mead",
                       options={"maxiter": 10000})
opt_weights_rank = result_rank.x / result_rank.x.sum()
opt_pred_rank = np.clip(M_rank @ opt_weights_rank, 1e-9, 1 - 1e-9)
opt_auc_rank = roc_auc_score(y, opt_pred_rank)
print(f"Rank-average optimized: {opt_auc_rank:.6f}")

print("\nRank-average optimal weights:")
for k, w in zip(keys, opt_weights_rank):
    if w > 0.01:
        print(f"  {k:<35} {w:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n=== GENERATING SUBMISSION ===")

# Use whichever blend was better
if opt_auc_rank > opt_auc:
    print(f"Using rank-average blend (OOF={opt_auc_rank:.6f})")
    # Build test predictions
    T_stack = np.column_stack([T[k] for k in keys])
    T_rank = np.column_stack([pd.Series(T[k]).rank(pct=True).values for k in keys])
    test_pred = np.clip(T_rank @ opt_weights_rank, 1e-9, 1 - 1e-9)
    suffix = "rank_blend"
    auc_used = opt_auc_rank
else:
    print(f"Using probability-average blend (OOF={opt_auc:.6f})")
    T_stack = np.column_stack([T.get(k, np.zeros(Nt)) for k in keys])
    test_pred = np.clip(T_stack @ opt_weights, 1e-9, 1 - 1e-9)
    suffix = "prob_blend"
    auc_used = opt_auc

# Filter out models with zero test predictions
valid_cols = [i for i, k in enumerate(keys) if k in T]
if len(valid_cols) < n_models:
    print(f"  WARNING: {n_models - len(valid_cols)} models missing test predictions")
    # Regenerate with available test vectors
    valid_keys = [keys[i] for i in valid_cols]
    M_valid = np.column_stack([V[k] for k in valid_keys])
    T_valid = np.column_stack([T[k] for k in valid_keys])
    
    # Re-optimize on valid subset
    def neg_auc_v(w):
        return -roc_auc_score(y, np.clip(M_valid @ w, 1e-9, 1 - 1e-9))
    res_v = minimize(neg_auc_v, np.ones(len(valid_keys)) / len(valid_keys), method="Nelder-Mead")
    w_v = res_v.x / res_v.x.sum()
    oof_v = roc_auc_score(y, np.clip(M_valid @ w_v, 1e-9, 1 - 1e-9))
    
    T_rank_v = np.column_stack([pd.Series(T[k]).rank(pct=True).values for k in valid_keys])
    def neg_auc_rv(w):
        return -roc_auc_score(y, np.column_stack([pd.Series(V[k]).rank(pct=True).values for k in valid_keys]) @ w)
    res_rv = minimize(neg_auc_rv, np.ones(len(valid_keys)) / len(valid_keys), method="Nelder-Mead")
    w_rv = res_rv.x / res_rv.x.sum()
    oof_rv = roc_auc_score(y, np.column_stack([pd.Series(V[k]).rank(pct=True).values for k in valid_keys]) @ w_rv)
    
    print(f"  Valid subset: prob OOF={oof_v:.6f}, rank OOF={oof_rv:.6f}")
    if oof_rv > oof_v:
        test_pred = np.clip(T_rank_v @ w_rv, 1e-9, 1 - 1e-9)
        auc_used = oof_rv
        suffix = "rank_blend_valid"
        print(f"  Using rank-average on valid subset")
    else:
        test_pred = np.clip(T_valid @ w_v, 1e-9, 1 - 1e-9)
        auc_used = oof_v
        suffix = "prob_blend_valid"

sub = pd.DataFrame({
    "id": te["id"],
    "addicted_label": test_pred
})
out_path = f"submission_optimal_{suffix}.csv"
sub.to_csv(out_path, index=False)
print(f"Submission: {out_path} ({len(sub)} rows)")
print(f"Prediction range: [{test_pred.min():.6f}, {test_pred.max():.6f}]")

# Also generate a simple champion-only submission as fallback
if "lib_lookup" in T:
    sub_champ = pd.DataFrame({
        "id": te["id"],
        "addicted_label": T["lib_lookup"]
    })
    sub_champ.to_csv("submission_champion_lookup.csv", index=False)
    print(f"Fallback: submission_champion_lookup.csv")

# Save blend analysis
analysis = {
    "oof_auc_optimized": float(auc_used),
    "n_models": n_models,
    "blend_method": suffix,
    "models_used": keys,
    "weights": {k: float(w) for k, w in zip(keys, opt_weights)},
    "rank_weights": {k: float(w) for k, w in zip(keys, opt_weights_rank)},
}
with open("blend_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("\n=== DONE ===")
