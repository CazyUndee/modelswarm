"""FM lattice blend check — OOS validation of FM members against greedy74 base.

Tests whether raykkretzschmar/s6e8-fm-lattice-blend-members improve held-out AUC
when added to our existing model pool. Protocol: OOS half-split, logloss-fit weights.

Usage: python shared/analysis/fm_blend_check.py
"""
import os
import numpy as np
import pandas as pd
import kagglehub
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# --- Load data ---
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values

# --- Load FM lattice OOF vectors ---
fm_dir = kagglehub.dataset_download("raykkretzschmar/s6e8-fm-lattice-blend-members")
print(f"FM lattice downloaded to: {fm_dir}")

fm_names = ["fmplr", "fmnum", "fmpure", "fmwide", "fmdeep"]
V, T = {}, {}
for n in fm_names:
    oof_path = os.path.join(fm_dir, f"oof_{n}.npy")
    test_path = os.path.join(fm_dir, f"test_{n}.npy")
    if os.path.exists(oof_path):
        V[n] = np.load(oof_path)
        T[n] = np.load(test_path)
        print(f"  {n}: OOF shape {V[n].shape}, AUC {roc_auc_score(y, V[n]):.6f}")
    else:
        print(f"  {n}: NOT FOUND at {oof_path}")

# --- Load library OOF vectors (key members from greedy74) ---
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
print(f"Library downloaded to: {lib_dir}")

lib_members = ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb"]
for n in lib_members:
    oof_path = os.path.join(lib_dir, "oof", f"oof_{n}.npy")
    test_path = os.path.join(lib_dir, "oof", f"test_{n}.npy")
    if os.path.exists(oof_path):
        V[n] = np.load(oof_path)
        T[n] = np.load(test_path)
        print(f"  {n}: OOF shape {V[n].shape}, AUC {roc_auc_score(y, V[n]):.6f}")

# --- Load our committed OOF CSVs (tier decomposition models) ---
oof_files = {
    "FAST": "oof_FAST.csv",
    "MED-B": "oof_MED-B.csv",
    "NUC-REF": "oof_NUC-REF.csv",
    "XGB": "oof_XGB.csv",
    "HGB": "oof_HGB.csv",
    "CAT": "oof_CAT.csv",
}
for name, path in oof_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        V[name] = df["pred"].values
        print(f"  {name}: OOF shape {V[name].shape}, AUC {roc_auc_score(y, V[name]):.6f}")

# --- Build matrix ---
keys = list(V.keys())
M = np.column_stack([V[k] for k in keys])
print(f"\nPool: {len(keys)} members: {keys}")

# --- OOS half-split ---
rng = np.random.RandomState(42)
idx = rng.permutation(len(y))
fit_idx, hold_idx = idx[:200000], idx[200000:500000]

def fit_weights(cols, fit_y, fit_M):
    Mf = fit_M[:, cols]
    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(fit_y * np.log(p) + (1 - fit_y) * np.log(1 - p))
    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 800, "maxfev": 1200, "xatol": 1e-4, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()

# --- Test 1: Greedy74-like base (library + our tier models) ---
base_keys = ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb",
             "FAST", "MED-B", "NUC-REF", "XGB", "HGB", "CAT"]
base_cols = [keys.index(k) for k in base_keys if k in keys]
base_w = fit_weights(base_cols, y[fit_idx], M[fit_idx])
base_hold = roc_auc_score(y[hold_idx], M[hold_idx][:, base_cols] @ base_w)
print(f"\n=== BASE (greedy74-like, {len(base_cols)} members) ===")
for c, w in zip(base_cols, base_w):
    if w > 0.01:
        print(f"  {keys[c]}: {w:.4f}")
print(f"  Held-out AUC: {base_hold:.6f}")

# --- Test 2: Base + each FM member one at a time ---
print(f"\n=== BASE + FM members (one at a time) ===")
for fm in fm_names:
    if fm not in keys:
        continue
    test_cols = base_cols + [keys.index(fm)]
    w = fit_weights(test_cols, y[fit_idx], M[fit_idx])
    hold = roc_auc_score(y[hold_idx], M[hold_idx][:, test_cols] @ w)
    delta = hold - base_hold
    marker = " GAIN" if delta > 0.0001 else (" FLAT" if abs(delta) < 0.0001 else " LOSS")
    print(f"  +{fm}: held-out {hold:.6f} (delta {delta:+.6f}){marker}")

# --- Test 3: Base + all FM members together ---
all_fm_cols = base_cols + [keys.index(fm) for fm in fm_names if fm in keys]
w = fit_weights(all_fm_cols, y[fit_idx], M[fit_idx])
hold = roc_auc_score(y[hold_idx], M[hold_idx][:, all_fm_cols] @ w)
delta = hold - base_hold
print(f"\n=== BASE + ALL FM ({len(all_fm_cols)} members) ===")
print(f"  Held-out AUC: {hold:.6f} (delta {delta:+.6f})")
for c, wi in zip(all_fm_cols, w):
    if wi > 0.01:
        print(f"  {keys[c]}: {wi:.4f}")

# --- Test 4: FM-only blend ---
fm_cols = [keys.index(fm) for fm in fm_names if fm in keys]
if len(fm_cols) > 1:
    w = fit_weights(fm_cols, y[fit_idx], M[fit_idx])
    hold = roc_auc_score(y[hold_idx], M[hold_idx][:, fm_cols] @ w)
    print(f"\n=== FM-ONLY BLEND ({len(fm_cols)} members) ===")
    print(f"  Held-out AUC: {hold:.6f}")
    for c, wi in zip(fm_cols, w):
        if wi > 0.01:
            print(f"  {keys[c]}: {wi:.4f}")

# --- Correlation matrix (key pairs) ---
print(f"\n=== CORRELATION MATRIX (key pairs) ===")
key_pairs = [("lookup", "fmplr"), ("tabm_seed3", "fmnum"), ("XGB", "fmplr"),
             ("HGB", "fmwide"), ("lookup", "fmwide")]
for a, b in key_pairs:
    if a in keys and b in keys:
        r = np.corrcoef(V[a], V[b])[0, 1]
        print(f"  {a} ~ {b}: {r:.5f}")

print("\n=== VERDICT ===")
print("Provenance: external-unverified (FM lattice) + owned (tier OOFs) + public (library)")
print("Report numbers to forum even if negative per Sisyphus instructions.")
