"""Test: Can new NN architectures improve Naji's frontier blend?

Hypothesis: If TabFM or FT-Transformer provide genuinely new signal
(not captured by Naji's 14 models), they could improve the 0.970099 OOF.

This script:
1. Downloads Naji's published OOF predictions
2. Downloads our existing OOF vectors
3. Loads TabFM/FT-Transformer OOF if available (from previous analysis)
4. Tests if new NN architectures add value to Naji's best blend
5. Reports complementarity metrics
"""
import os
import glob
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

print(f"Data: {N} rows")
print(f"Label rate: {y.mean():.4f}")

# ============================================================
# LOAD NAJI OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("LOADING NAJI OOF PREDICTIONS")
print("=" * 70)

import kagglehub

naji_dir = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
print(f"Naji dataset: {naji_dir}")

V = {}
for f in sorted(glob.glob(os.path.join(naji_dir, "*_oof_predictions.csv"))):
    name = os.path.basename(f).replace("_oof_predictions.csv", "")
    df = pd.read_csv(f)
    if "prediction" in df.columns:
        V[f"naji_{name}"] = df["prediction"].values
    elif "addicted_label" in df.columns:
        V[f"naji_{name}"] = df["addicted_label"].values
    else:
        continue
    print(f"  naji_{name}: OOF {roc_auc_score(y, V[f'naji_{name}']):.6f}")

# ============================================================
# LOAD LIBRARY MODELS
# ============================================================
print("\n" + "=" * 70)
print("LOADING LIBRARY MODELS")
print("=" * 70)

lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  lib_{n}: OOF {roc_auc_score(y, V[f'lib_{n}']):.6f}")

# ============================================================
# LOAD OWNED VECTORS
# ============================================================
print("\n" + "=" * 70)
print("LOADING OWNED VECTORS")
print("=" * 70)

V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
print(f"  owned_EXP122: OOF {roc_auc_score(y, V['owned_EXP122']):.6f}")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
print(f"  owned_EXP130: OOF {roc_auc_score(y, V['owned_EXP130']):.6f}")

# ============================================================
# LOAD NEW NN OOF IF AVAILABLE
# ============================================================
print("\n" + "=" * 70)
print("LOADING NEW NN OOF (TabFM, FT-Transformer)")
print("=" * 70)

new_nn = {}
for name, path in [("tabfm", "shared/artifacts/stacking_vectors/tabfm_oof.npy"),
                    ("ft_transformer", "shared/artifacts/stacking_vectors/ft_transformer_oof.npy")]:
    if os.path.exists(path):
        V[name] = np.load(path)
        new_nn[name] = V[name]
        print(f"  {name}: OOF {roc_auc_score(y, V[name]):.6f}")
    else:
        print(f"  {name}: NOT AVAILABLE (previous analysis may not have completed)")

if not new_nn:
    print("\nWARNING: No new NN OOF available. Only testing existing models.")
    print("This script needs TabFM or FT-Transformer OOF to test the hypothesis.")
    print("Run tabfm_test.py and ft_transformer_test.py first.\n")

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

# ============================================================
# TEST: NAJI BEST BLEND + NEW NN
# ============================================================
print("\n" + "=" * 70)
print("TEST: Naji's Best Blend + New NN Architectures")
print("=" * 70)

# Get naji_19_blend as the base
if "naji_19_blend" not in V:
    print("ERROR: naji_19_blend not found")
    exit(1)

naji_base = V["naji_19_blend"]
naji_base_auc = roc_auc_score(y, naji_base)
print(f"\nNaji base (naji_19_blend): OOF {naji_base_auc:.6f}")

# Split for weight fitting
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

# Test each new NN individually
for nn_name in ["tabm_seed3", "lookup", "owned_EXP122", "owned_EXP130"] + list(new_nn.keys()):
    if nn_name not in V:
        continue
    nn_pred = V[nn_name]
    
    # Stack: [naji_base, nn_pred]
    M = np.column_stack([naji_base, nn_pred])
    w = fit_weights([0, 1], y[fit_idx], M[fit_idx])
    pred = np.clip(M @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    held = roc_auc_score(y[hold_idx], pred[hold_idx])
    delta = auc - naji_base_auc
    sp = spearmanr(naji_base, nn_pred).correlation
    
    print(f"  +{nn_name:<25}: OOF={auc:.6f} Δ={delta:>+.6f} held={held:.6f} nn_weight={w[1]:.3f} spearman={sp:.5f}")

# ============================================================
# TEST: MULTIPLE NEW NNS TOGETHER
# ============================================================
if len(new_nn) > 1:
    print("\n--- Multiple new NNs combined ---")
    nn_names = list(new_nn.keys())
    keys = ["naji_19_blend"] + nn_names
    M = np.column_stack([V[k] for k in keys])
    cols = list(range(len(keys)))
    w = fit_weights(cols, y[fit_idx], M[fit_idx])
    pred = np.clip(M @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    held = roc_auc_score(y[hold_idx], pred[hold_idx])
    delta = auc - naji_base_auc
    print(f"  naji + all new NNs: OOF={auc:.6f} Δ={delta:>+.6f} held={held:.6f}")
    for k, wi in zip(keys, w):
        print(f"    {k}: {wi:.4f}")

# ============================================================
# CORRELATION: NEW NNS vs NAJI
# ============================================================
print("\n--- Correlation: New NNs vs Naji models ---")
for nn_name in ["tabm_seed3", "lookup"] + list(new_nn.keys()):
    if nn_name not in V:
        continue
    for naji_name in ["naji_03", "naji_05", "naji_07_blend", "naji_19_blend"]:
        if naji_name not in V:
            continue
        sp = spearmanr(V[nn_name], V[naji_name]).correlation
        print(f"  {nn_name:<20} ~ {naji_name:<20}: Spearman={sp:.5f}")

print("\n--- DONE ---")
