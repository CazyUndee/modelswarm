"""Band-Specific Blend — test whether different model weights in different
screen-time bands improves over a global blend.

Hypothesis: NN helps in low screen-time [0,7h] but hurts in high [9,24h].
A band-specific blend should outperform a global blend.
"""
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

np.random.seed(42)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)
screen = tr["daily_screen_time_hours"].values.astype(float)
screen = np.nan_to_num(screen, nan=0.0)

# Load vectors
V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[n] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
V["EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")

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
fit_idx = np.arange(400000, dtype=int)
hold_idx = np.arange(400000, N, dtype=int)

# Define bands
bands = [
    ("low [0,5)", (screen >= 0) & (screen < 5)),
    ("mid [5,9)", (screen >= 5) & (screen < 9)),
    ("high [9,24]", (screen >= 9) & (screen <= 24)),
]

# Test configurations
configs = [
    ("Global: naji03+lookup+tabm+EXP130", ["naji03", "lookup", "tabm_seed3", "EXP130"]),
    ("Global: naji03+lookup+tabm", ["naji03", "lookup", "tabm_seed3"]),
    ("Global: lookup+tabm+EXP122+EXP130", ["lookup", "tabm_seed3", "EXP122", "EXP130"]),
]

print("=" * 70)
print("BAND-SPECIFIC BLEND ANALYSIS")
print("=" * 70)

# First: global blend performance per band
print("\n--- Global blend performance per band ---")
for cfg_name, members in configs:
    cols = [keys.index(m) for m in members]
    w = fit_weights(cols, y[fit_idx], M[fit_idx])
    pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
    full_auc = roc_auc_score(y, pred)
    print(f"\n{cfg_name}")
    print(f"  Weights: {', '.join(f'{m}={w[i]:.3f}' for i, m in enumerate(members))}")
    for band_name, mask in bands:
        band_auc = roc_auc_score(y[mask], pred[mask])
        print(f"  {band_name}: AUC={band_auc:.4f} (n={np.sum(mask):,})")

# Now: band-specific blends
print("\n\n--- Band-specific blend (per-band weight optimization) ---")
test_members = ["naji03", "lookup", "tabm_seed3", "EXP130"]
cols = [keys.index(m) for m in test_members]

# Fit separate weights for each band
band_weights = {}
band_preds = np.zeros(N)
for band_name, mask in bands:
    band_y = y[mask]
    band_M = M[mask]
    w = fit_weights(cols, band_y, band_M)
    band_weights[band_name] = w
    band_preds[mask] = np.clip(band_M[:, cols] @ w, 1e-9, 1 - 1e-9)

# Evaluate
full_auc = roc_auc_score(y, band_preds)
print(f"\nBand-specific blend: OOF={full_auc:.6f}")
for band_name, mask in bands:
    band_auc = roc_auc_score(y[mask], band_preds[mask])
    w = band_weights[band_name]
    w_str = ', '.join(f'{m}={w[i]:.3f}' for i, m in enumerate(test_members))
    print(f"  {band_name}: AUC={band_auc:.4f} weights=[{w_str}]")

# Compare with global
global_cols = [keys.index(m) for m in test_members]
global_w = fit_weights(global_cols, y[fit_idx], M[fit_idx])
global_pred = np.clip(M[:, global_cols] @ global_w, 1e-9, 1 - 1e-9)
global_auc = roc_auc_score(y, global_pred)
print(f"\nGlobal blend:      OOF={global_auc:.6f}")
print(f"Band-specific:     OOF={full_auc:.6f}")
print(f"Δ (band - global): {full_auc - global_auc:+.6f}")

# Per-band comparison
print(f"\n--- Per-band comparison ---")
for band_name, mask in bands:
    g_auc = roc_auc_score(y[mask], global_pred[mask])
    b_auc = roc_auc_score(y[mask], band_preds[mask])
    print(f"  {band_name}: global={g_auc:.4f} band-specific={b_auc:.4f} Δ={b_auc-g_auc:+.4f}")

# Also test: 2-band split (low vs high)
print("\n\n--- 2-band split (low [0,7) vs high [7,24]) ---")
low_mask = (screen >= 0) & (screen < 7)
high_mask = (screen >= 7) & (screen <= 24)

band2_preds = np.zeros(N)
for bm, bm_name in [(low_mask, "low"), (high_mask, "high")]:
    w = fit_weights(cols, y[bm], M[bm])
    band2_preds[bm] = np.clip(M[bm][:, cols] @ w, 1e-9, 1 - 1e-9)
    w_str = ', '.join(f'{m}={w[i]:.3f}' for i, m in enumerate(test_members))
    print(f"  {bm_name}: weights=[{w_str}]")

auc2 = roc_auc_score(y, band2_preds)
print(f"  2-band OOF: {auc2:.6f} (global: {global_auc:.6f}, Δ={auc2-global_auc:+.6f})")

# Robustness check
print("\n--- Robustness (5 seeds) ---")
for seed in [42, 123, 7, 2024, 99]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    fi, hi = idx[:400000], idx[400000:]
    
    # Global
    gw = fit_weights(cols, y[fi], M[fi])
    g_pred = np.clip(M[:, cols] @ gw, 1e-9, 1 - 1e-9)
    g_auc = roc_auc_score(y[hi], g_pred[hi])
    
    # Band-specific: fit per-band on training fold, predict on all
    bs_preds = np.zeros(N)
    for bm, _ in bands:
        bm_arr = np.asarray(bm, dtype=bool)
        band_fi = fi[bm_arr[fi]]  # integer indices of training rows in this band
        if len(band_fi) < 100:
            continue
        bw = fit_weights(cols, y[band_fi], M[band_fi])
        band_all = np.nonzero(bm_arr)[0]  # all row indices in this band
        bs_preds[band_all] = np.clip(M[band_all][:, cols] @ bw, 1e-9, 1 - 1e-9)
    bs_auc = roc_auc_score(y[hi], bs_preds[hi])
    
    print(f"  seed={seed}: global={g_auc:.6f} band-specific={bs_auc:.6f} Δ={bs_auc-g_auc:+.6f}")

print("\n--- DONE ---")
