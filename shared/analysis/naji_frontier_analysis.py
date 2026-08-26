"""Naji Ama Frontier Analysis — reverse-engineer the 0.97128 blend composition.

Naji published 20 OOF/submission pairs on Kaggle. This script:
1. Loads all OOF vectors and computes individual AUCs
2. Measures pairwise correlations
3. Finds the optimal blend weights
4. Identifies which models are most complementary
5. Tests whether our owned models add value on top

Provenance: ALL Naji vectors are EXTERNAL (public, CC0-like).
Our owned vectors are clearly labeled.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import spearmanr
import glob

np.random.seed(42)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# Load Naji's published OOF vectors
naji_dir = "/tmp/naji_oof/naji_data"
V = {}
for f in sorted(glob.glob(os.path.join(naji_dir, "*_oof_predictions.csv"))):
    name = os.path.basename(f).replace("_oof_predictions.csv", "")
    df = pd.read_csv(f)
    if "prediction" in df.columns:
        V[f"naji_{name}"] = df["prediction"].values
    elif "addicted_label" in df.columns:
        V[f"naji_{name}"] = df["addicted_label"].values
    else:
        print(f"  Skipping {name}: columns = {list(df.columns)}")
        continue

# Also load our owned vectors
lib_dir = None
for p in ["competitions/s6e8/data"]:
    if os.path.exists(p):
        break

# Load library NN
import kagglehub
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))

# Load owned vectors
V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")

# ============================================================
# INDIVIDUAL MODEL AUCS
# ============================================================
print("=" * 70)
print("NAJI FRONTIER ANALYSIS — Model Inventory")
print("=" * 70)

keys = list(V.keys())
aucs = {k: roc_auc_score(y, V[k]) for k in keys}

print(f"\n{'Model':<30} {'OOF AUC':>10} {'Source':>12}")
print("-" * 55)
for k in sorted(keys, key=lambda x: aucs[x], reverse=True):
    source = "EXTERNAL" if k.startswith("naji_") else ("LIBRARY" if k.startswith("lib_") else "OWNED")
    print(f"{k:<30} {aucs[k]:>10.6f} {source:>12}")

# ============================================================
# PAIRWISE CORRELATIONS (Naji models only)
# ============================================================
print("\n" + "=" * 70)
print("PAIRWISE SPEARMAN CORRELATIONS (Naji models)")
print("=" * 70)

naji_keys = [k for k in keys if k.startswith("naji_")]
M_naji = np.column_stack([V[k] for k inaji_keys])

print(f"\n{'Pair':<40} {'Spearman':>10} {'AUC Δ':>10}")
print("-" * 65)
for i in range(len(naji_keys)):
    for j in range(i + 1, len(naji_keys)):
        sp = spearmanr(V[naji_keys[i]], V[naji_keys[j]]).correlation
        auc_diff = aucs[naji_keys[j]] - aucs[naji_keys[i]]
        if sp < 0.99:  # Only show decorrelated pairs
            print(f"{naji_keys[i]:20s} ~ {naji_keys[j]:17s} {sp:>10.5f} {auc_diff:>+10.6f}")

# ============================================================
# BLEND ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("BLEND ANALYSIS — Finding optimal Naji model combination")
print("=" * 70)

# Find which Naji models are most useful in a blend
def fit_weights(cols, y_fit, M_fit):
    Mf = M_fit[:, cols]
    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y_fit * np.log(p) + (1 - y_fit) * np.log(1 - p))
    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 1000, "maxfev": 2000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()

# OOS split
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

# Forward selection over Naji models
print("\n--- Forward selection over Naji models ---")
remaining = list(range(len(naji_keys)))
selected = []
best_auc = 0

for step in range(min(10, len(naji_keys))):
    step_best = (-1, -1, None)
    for candidate in remaining:
        test_cols = selected + [candidate]
        w = fit_weights(test_cols, y[fit_idx], M_naji[fit_idx])
        pred = np.clip(M_naji[:, test_cols] @ w, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, pred)
        if auc > step_best[1]:
            step_best = (candidate, auc, w)
    
    if step_best[0] < 0 or step_best[1] <= best_auc + 0.00001:
        break
    
    selected.append(step_best[0])
    remaining.remove(step_best[0])
    best_auc = step_best[1]
    
    w = step_best[2]
    w_str = ", ".join(f"{naji_keys[selected[i]]}={w[i]:.3f}" for i in range(len(selected)) if w[i] > 0.01)
    held_pred = np.clip(M_naji[hold_idx][:, selected] @ w, 1e-9, 1 - 1e-9)
    held_auc = roc_auc_score(y[hold_idx], held_pred)
    print(f"  step {step+1}: +{naji_keys[step_best[0]]:25s} → OOF={best_auc:.6f} held={held_auc:.6f} [{w_str}]")

# ============================================================
# OUR MODELS vs NAJI FRONTIER
# ============================================================
print("\n" + "=" * 70)
print("OUR MODELS vs NAJI FRONTIER — Complementarity test")
print("=" * 70)

# Use the best Naji blend as the frontier reference
M_all = np.column_stack([V[k] for k in keys])
all_cols = list(range(len(keys)))

# Test: does adding our owned models improve the Naji blend?
best_naji_cols = selected
best_naji_w = fit_weights(best_naji_cols, y[fit_idx], M_naji[fit_idx])
best_naji_pred = np.clip(M_naji[:, best_naji_cols] @ best_naji_w, 1e-9, 1 - 1e-9)
best_naji_auc = roc_auc_score(y, best_naji_pred)
print(f"\nBest Naji blend: OOF={best_naji_auc:.6f} ({len(best_naji_cols)} models)")

# Add each of our models one at a time
our_models = ["owned_EXP122", "owned_EXP130", "lib_lookup", "lib_tabm_seed3"]
for model in our_models:
    if model not in keys:
        continue
    model_col = keys.index(model)
    test_cols = best_naji_cols + [model_col]
    w = fit_weights(test_cols, y[fit_idx], M_all[fit_idx])
    pred = np.clip(M_all[:, test_cols] @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    delta = auc - best_naji_auc
    our_weight = w[-1]
    print(f"  +{model:25s}: OOF={auc:.6f} Δ={delta:+.6f} (our_weight={our_weight:.3f})")

# Test: does adding our owned models to the full Naji blend help?
print(f"\n--- Adding owned models to ALL Naji models ---")
all_naji_cols = list(range(len(naji_keys)))
for model in our_models:
    if model not in keys:
        continue
    model_col = keys.index(model)
    test_cols = all_naji_cols + [model_col]
    w = fit_weights(test_cols, y[fit_idx], M_all[fit_idx])
    pred = np.clip(M_all[:, test_cols] @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    delta = auc - best_naji_auc
    our_weight = w[-1]
    print(f"  +{model:25s}: OOF={auc:.6f} Δ={delta:+.6f} (our_weight={our_weight:.3f})")

# ============================================================
# KEY FINDINGS
# ============================================================
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print(f"\nTotal Naji models published: {len(naji_keys)}")
print(f"Best Naji blend OOF: {best_naji_auc:.6f}")
print(f"Best Naji blend LB: ~0.97128 (verified)")
print(f"Live LB frontier: 0.97186 (Chris Deotte)")
print(f"\nOur best owned blend: OOF 0.96942")
print(f"Gap to Naji blend: {best_naji_auc - 0.96942:+.6f}")
print(f"\nIf our models add value to the Naji blend, that's a path to the frontier.")
print(f"If not, the Naji blend already captures everything our models know.")
