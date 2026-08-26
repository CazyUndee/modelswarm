"""Frontier Attack v2 — comprehensive analysis of whether owned signal improves
the external frontier (Naji Ama 0.97128).

Loads all available vectors, measures complementarity with the frontier,
and tests controlled blends.

Protocol: all evaluation via OOF predictions. No public LB weight tuning.
"""
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("FRONTIER ATTACK v2 — Loading vectors")
print("=" * 70)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)
print(f"Training rows: {N}")

V, T = {}, {}  # V = OOF, T = test predictions

# --- Library vectors ---
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
print(f"Library: {lib_dir}")

# Load key library members
lib_members = ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb"]
for n in lib_members:
    oof_path = os.path.join(lib_dir, "oof", f"oof_{n}.npy")
    test_path = os.path.join(lib_dir, "oof", f"test_{n}.npy")
    V[n] = np.load(oof_path)
    T[n] = np.load(test_path)
    print(f"  {n}: OOF {roc_auc_score(y, V[n]):.6f}")

# --- EXP-122 (owned, 10-fold champion) ---
e122_path = "shared/artifacts/stacking_vectors/exp122_oof.npy"
if os.path.exists(e122_path):
    V["EXP122"] = np.load(e122_path)
    print(f"  EXP122: OOF {roc_auc_score(y, V['EXP122']):.6f} (owned)")

# --- EXP-130 (owned, raphdraft block) ---
e130_path = "shared/artifacts/stacking_vectors/exp130_oof.npy"
if os.path.exists(e130_path):
    V["EXP130"] = np.load(e130_path)
    print(f"  EXP130: OOF {roc_auc_score(y, V['EXP130']):.6f} (owned)")

# --- EXP-132 (lookup-transformer, if available) ---
e132_path = "competitions/s6e8/experiments/output/EXP-132/oof_predictions.csv"
if os.path.exists(e132_path):
    V["lookup_xform"] = pd.read_csv(e132_path)["prediction"].values
    print(f"  lookup_xform: OOF {roc_auc_score(y, V['lookup_xform']):.6f} (owned)")
else:
    print("  lookup_xform: NOT YET AVAILABLE (EXP-132 still running)")

# --- Tier decomposition OOFs ---
tier_files = {"FAST": "oof_FAST.csv", "MED-B": "oof_MED-B.csv",
              "NUC-REF": "oof_NUC-REF.csv", "XGB": "oof_XGB.csv",
              "HGB": "oof_HGB.csv", "CAT": "oof_CAT.csv"}
for name, path in tier_files.items():
    if os.path.exists(path):
        V[name] = pd.read_csv(path)["pred"].values
        print(f"  {name}: OOF {roc_auc_score(y, V[name]):.6f} (owned)")

# ============================================================
# FRONTIER REFERENCE
# ============================================================
print("\n" + "=" * 70)
print("FRIER REFERENCE ANALYSIS")
print("=" * 70)

# The frontier = Naji Ama blend. Its test predictions are in the library.
# We don't have its OOF directly, but naji03 is a component of it.
# The vault submission.csv IS the frontier (test predictions only).

# Load vault submission if available (test predictions of the frontier)
vault_path = None
for p in ["competitions/s6e8/data/vault_submission.csv",
          "/tmp/s6e8_vault_submission.csv"]:
    if os.path.exists(p):
        vault_path = p
        break

if vault_path:
    vault = pd.read_csv(vault_path)
    T["frontier"] = vault["addicted_label"].values
    print(f"Frontier test predictions loaded from {vault_path}")
else:
    print("Frontier test predictions NOT available locally")
    print("Using naji03 as frontier proxy (they are the same blend)")

# Use naji03 as the frontier reference (it IS the same blend)
# The library's naji03 OOF is available
if "naji03" in V:
    frontier_auc = roc_auc_score(y, V["naji03"])
    print(f"\nFrontier (naji03) OOF: {frontier_auc:.6f}")
    print(f"Frontier LB: 0.97128 (Naji Ama Ensemble-of-Ensembles)")
    print(f"OOF→LB transfer: ~+0.00129")

# ============================================================
# PHASE 1: COMPLEMENTARITY WITH FRONTIER
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: COMPLEMENTARITY WITH FRONTIER")
print("=" * 70)

focus = ["naji03", "lookup", "tabm_seed3", "EXP122", "EXP130", "XGB", "HGB"]
if "lookup_xform" in V:
    focus.append("lookup_xform")
focus = [k for k in focus if k in V]

print(f"\n{'Pair':<30} {'Spearman':>10} {'Pearson':>10} {'Blend AUC':>10}")
print("-" * 65)

for i in range(len(focus)):
    for j in range(i + 1, len(focus)):
        ki, kj = focus[i], focus[j]
        sp = spearmanr(V[ki], V[kj]).correlation
        pe = pearsonr(V[ki], V[kj])[0]
        blend = np.clip((V[ki] + V[kj]) / 2, 1e-9, 1 - 1e-9)
        ba = roc_auc_score(y, blend)
        print(f"{ki:<15} ~ {kj:<13} {sp:>10.5f} {pe:>10.5f} {ba:>10.6f}")

# Error disagreement with frontier
print(f"\n--- Error disagreement with frontier (naji03) ---")
frontier_pred = (V["naji03"] >= 0.5).astype(int)
frontier_err = (frontier_pred != y)
for k in focus:
    if k == "naji03":
        continue
    pred_k = (V[k] >= 0.5).astype(int)
    err_k = (pred_k != y)
    agree = np.mean(frontier_err == err_k) * 100
    frontier_only = np.sum(frontier_err & ~err_k)
    model_only = np.sum(~frontier_err & err_k)
    print(f"  {k}: agree {agree:.1f}%, frontier-only-err {frontier_only}, model-only-err {model_only}")

# ============================================================
# PHASE 2: BLEND STRATEGIES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: BLEND STRATEGIES")
print("=" * 70)

# Helper
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

blend_configs = [
    ("Frontier alone (naji03)", ["naji03"]),
    ("greedy74-like (5 lib)", ["lookup", "tabm_seed3", "naji03", "XGB", "HGB"]),
    ("NN-only (lookup+tabm)", ["lookup", "tabm_seed3"]),
    ("Owned-only (EXP122+EXP130+XGB+HGB)", ["EXP122", "EXP130", "XGB", "HGB"]),
    ("NN + Owned", ["lookup", "tabm_seed3", "EXP122", "EXP130"]),
    ("Frontier + NN", ["naji03", "lookup", "tabm_seed3"]),
    ("Frontier + NN + EXP122", ["naji03", "lookup", "tabm_seed3", "EXP122"]),
    ("Frontier + NN + EXP130", ["naji03", "lookup", "tabm_seed3", "EXP130"]),
    ("Frontier + NN + Owned", ["naji03", "lookup", "tabm_seed3", "EXP122", "EXP130"]),
    ("Frontier + NN + greedy74", ["naji03", "lookup", "tabm_seed3", "XGB", "HGB"]),
    ("Full pool", ["naji03", "lookup", "tabm_seed3", "EXP122", "EXP130", "XGB", "HGB"]),
]
if "lookup_xform" in V:
    blend_configs.extend([
        ("Frontier + lookup_xform", ["naji03", "lookup_xform"]),
        ("Frontier + NN + lookup_xform", ["naji03", "lookup", "tabm_seed3", "lookup_xform"]),
        ("Full + lookup_xform", ["naji03", "lookup", "tabm_seed3", "EXP122", "EXP130", "lookup_xform"]),
    ])

blend_configs = [(n, [k for k in m if k in keys]) for n, m in blend_configs]

print(f"\n{'Config':<50} {'OOF AUC':>10} {'Held-out':>10} {'Δ vs frontier':>13}")
print("-" * 85)

results = {}
for name, members in blend_configs:
    if not members:
        continue
    cols = [keys.index(m) for m in members]
    if len(members) == 1:
        full_auc = roc_auc_score(y, V[members[0]])
        held_auc = roc_auc_score(y[hold_idx], V[members[0]][hold_idx])
        w = np.array([1.0])
    else:
        w = fit_weights(cols, y[fit_idx], M[fit_idx])
        full_pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
        full_auc = roc_auc_score(y, full_pred)
        held_pred = np.clip(M[hold_idx][:, cols] @ w, 1e-9, 1 - 1e-9)
        held_auc = roc_auc_score(y[hold_idx], held_pred)

    delta = full_auc - roc_auc_score(y, V["naji03"])
    results[name] = {"auc": full_auc, "held": held_auc, "members": members, "weights": w}
    marker = " ***" if delta > 0.0002 else (" *" if delta > 0.00005 else "")
    w_str = ""
    if len(members) > 1:
        top = [(members[i], w[i]) for i in range(len(members)) if w[i] > 0.01]
        w_str = " | " + ", ".join(f"{m}={ww:.3f}" for m, ww in top)
    print(f"{name:<50} {full_auc:>10.6f} {held_auc:>10.6f} {delta:>+13.6f}{marker}{w_str}")

# ============================================================
# PHASE 3: ROBUSTNESS CHECK ON BEST FRONTIER BLEND
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: ROBUSTNESS (best frontier+owned blend)")
print("=" * 70)

# Find the best blend that includes the frontier
frontier_blends = {k: v for k, v in results.items() if "Frontier" in k and v["auc"] > roc_auc_score(y, V["naji03"]) + 0.00005}
if frontier_blends:
    best_name = max(frontier_blends, key=lambda k: frontier_blends[k]["auc"])
    best = frontier_blends[best_name]
    print(f"\nBest improving blend: {best_name}")
    print(f"  Members: {best['members']}")
    print(f"  OOF: {best['auc']:.6f} (Δ {best['auc'] - roc_auc_score(y, V['naji03']):+.6f})")
    print(f"  Held-out: {best['held']:.6f}")

    cols = [keys.index(m) for m in best["members"]]
    print(f"\n  Testing across 5 OOS splits:")
    for seed in [42, 123, 7, 2024, 99]:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        fi, hi = idx[:400000], idx[400000:]
        w = fit_weights(cols, y[fi], M[fi])
        pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, pred)
        held = roc_auc_score(y[hi], pred[hi])
        delta = auc - roc_auc_score(y, V["naji03"])
        print(f"    seed={seed}: OOF={auc:.6f} held={held:.6f} Δ_vs_frontier={delta:+.6f}")
else:
    print("\nNo frontier blend improved over frontier alone.")
    print("The frontier (0.97128) may already be near the ceiling for this feature space.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
frontier_auc = roc_auc_score(y, V["naji03"])
print(f"\nFrontier (naji03) OOF:        {frontier_auc:.6f}  (LB 0.97128)")
print(f"Live LB top:                  0.97186 (Chris Deotte)")
print(f"Best owned blend (greedy74):  {results.get('greedy74-like (5 lib)', {}).get('auc', 0):.6f}")
print(f"Best NN+owned:                {results.get('NN + Owned', {}).get('auc', 0):.6f}")
if frontier_blends:
    print(f"Best frontier+owned:          {best['auc']:.6f} (Δ {best['auc'] - frontier_auc:+.6f})")
else:
    print(f"Best frontier+owned:          NO IMPROVEMENT over frontier alone")
print(f"\nProvenance: external (naji03/library) + owned (EXP122/130/tier OOFs)")
