"""NN + Owned-Model Stacking Investigation — 5-phase comprehensive analysis.

Phase 1: Pairwise complementarity metrics
Phase 2: Simple controlled blends
Phase 3: Meta-learning (logistic, ridge, optional tree)
Phase 4: Ablation
Phase 5: Robustness checks

All evaluation uses OOF predictions only. No public LB tuning.
Provenance: external (library NN) + owned (EXP-122/130, tier OOFs).
"""
import os
import json
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
print("LOADING VECTORS")
print("=" * 70)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)
print(f"Training rows: {N}")

# --- Library NN vectors (external) ---
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
print(f"Library: {lib_dir}")

V, T = {}, {}  # V = OOF, T = test predictions
nn_members = ["lookup", "tabm_seed3"]
for n in nn_members:
    oof_path = os.path.join(lib_dir, "oof", f"oof_{n}.npy")
    test_path = os.path.join(lib_dir, "oof", f"test_{n}.npy")
    V[n] = np.load(oof_path)
    T[n] = np.load(test_path)
    print(f"  {n}: OOF AUC {roc_auc_score(y, V[n]):.6f}")

# Also load naji03 (the real Naji blend, NOT vault) for reference
V["naji03_ref"] = np.load(os.path.join(lib_dir, "oof", "oof_naji03.npy"))
T["naji03_ref"] = np.load(os.path.join(lib_dir, "oof", "test_naji03.npy"))
print(f"  naji03_ref: OOF AUC {roc_auc_score(y, V['naji03_ref']):.6f} (external reference)")

# --- EXP-122 (owned, 10-fold champion) ---
e122_path = "shared/artifacts/stacking_vectors/exp122_oof.npy"
if os.path.exists(e122_path):
    V["EXP122"] = np.load(e122_path)
else:
    e122 = pd.read_csv("/tmp/stacking_vectors/exp122/oof_predictions.csv")
    V["EXP122"] = e122["prediction"].values
print(f"  EXP122: OOF AUC {roc_auc_score(y, V['EXP122']):.6f} (owned)")

# --- EXP-130 (owned, raphdraft block) ---
e130_path = "shared/artifacts/stacking_vectors/exp130_oof.npy"
if os.path.exists(e130_path):
    V["EXP130"] = np.load(e130_path)
else:
    e130 = pd.read_csv("/tmp/stacking_vectors/exp130/oof_predictions.csv")
    V["EXP130"] = e130["prediction"].values
print(f"  EXP130: OOF AUC {roc_auc_score(y, V['EXP130']):.6f} (owned)")

# --- Tier decomposition OOFs (owned, single-seed local) ---
tier_files = {
    "FAST": "oof_FAST.csv",
    "MED-B": "oof_MED-B.csv",
    "NUC-REF": "oof_NUC-REF.csv",
    "XGB": "oof_XGB.csv",
    "HGB": "oof_HGB.csv",
    "CAT": "oof_CAT.csv",
}
for name, path in tier_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        V[name] = df["pred"].values
        print(f"  {name}: OOF AUC {roc_auc_score(y, V[name]):.6f} (owned, local)")

# --- FM lattice (external, for reference only) ---
fm_dir = kagglehub.dataset_download("raykkretzschmar/s6e8-fm-lattice-blend-members")
for n in ["fmplr", "fmnum"]:
    oof_path = os.path.join(fm_dir, f"oof_{n}.npy")
    if os.path.exists(oof_path):
        V[n] = np.load(oof_path)
        print(f"  {n}: OOF AUC {roc_auc_score(y, V[n]):.6f} (external)")

# ============================================================
# PHASE 1: PAIRWISE COMPLEMENTARITY
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: PAIRWISE COMPLEMENTARITY METRICS")
print("=" * 70)

keys = list(V.keys())
n_models = len(keys)
auc_vec = np.array([roc_auc_score(y, V[k]) for k in keys])

print(f"\n{'Model':<15} {'OOF AUC':>10}  Provenance")
print("-" * 50)
provenance = {}
for i, k in enumerate(keys):
    if k in nn_members:
        prov = "external-NN"
    elif k == "naji03_ref":
        prov = "external-ref"
    elif k in ["fmplr", "fmnum"]:
        prov = "external-FM"
    else:
        prov = "owned"
    provenance[k] = prov
    print(f"{k:<15} {auc_vec[i]:>10.6f}  {prov}")

# Pairwise metrics for key pairs
focus_keys = ["EXP122", "EXP130", "lookup", "tabm_seed3", "FAST", "MED-B", "NUC-REF", "XGB", "HGB", "CAT"]
focus_keys = [k for k in focus_keys if k in keys]

print(f"\n{'Pair':<30} {'Spearman':>10} {'Pearson':>10} {'AUC Δ':>10} {'Blend AUC':>10}")
print("-" * 70)

pair_results = {}
for i in range(len(focus_keys)):
    for j in range(i + 1, len(focus_keys)):
        ki, kj = focus_keys[i], focus_keys[j]
        sp = spearmanr(V[ki], V[kj]).correlation
        pe = pearsonr(V[ki], V[kj])[0]
        auc_diff = auc_vec[keys.index(kj)] - auc_vec[keys.index(ki)]
        # Simple 50/50 blend
        blend_50 = np.clip((V[ki] + V[kj]) / 2, 1e-9, 1 - 1e-9)
        blend_auc = roc_auc_score(y, blend_50)
        pair_results[(ki, kj)] = {
            "spearman": sp, "pearson": pe, "auc_diff": auc_diff,
            "blend_50_auc": blend_auc, "auc_i": auc_vec[keys.index(ki)],
            "auc_j": auc_vec[keys.index(kj)]
        }
        print(f"{ki} ~ {kj:<15} {sp:>10.5f} {pe:>10.5f} {auc_diff:>+10.6f} {blend_auc:>10.6f}")

# Error disagreement analysis for key pairs
print(f"\n{'Pair':<30} {'Err agree %':>12} {'NN-only err':>12} {'Own-only err':>12}")
print("-" * 70)
for i in range(len(focus_keys)):
    for j in range(i + 1, len(focus_keys)):
        ki, kj = focus_keys[i], focus_keys[j]
        pred_i = (V[ki] >= 0.5).astype(int)
        pred_j = (V[kj] >= 0.5).astype(int)
        err_i = (pred_i != y)
        err_j = (pred_j != y)
        agree_pct = np.mean(err_i == err_j) * 100
        nn_only_err = np.sum(err_i & ~err_j)
        own_only_err = np.sum(~err_i & err_j)
        print(f"{ki} ~ {kj:<15} {agree_pct:>11.1f}% {nn_only_err:>12d} {own_only_err:>12d}")

# ============================================================
# HELPER: weight optimization
# ============================================================
def fit_blend_weights(cols, y_fit, M_fit, method="nelder-mead"):
    """Fit non-negative weights summing to 1 via logloss minimization."""
    Mf = M_fit[:, cols]
    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y_fit * np.log(p) + (1 - y_fit) * np.log(1 - p))
    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 1000, "maxfev": 2000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()

# OOS split: fit on first 400k, validate on last 291k
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

# ============================================================
# PHASE 2: SIMPLE BLENDS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: SIMPLE CONTROLLED BLENDS")
print("=" * 70)

blend_configs = [
    ("NN only (lookup)", ["lookup"]),
    ("NN only (tabm)", ["tabm_seed3"]),
    ("NN avg (lookup+tabm)", ["lookup", "tabm_seed3"]),
    ("EXP122 alone", ["EXP122"]),
    ("EXP130 alone", ["EXP130"]),
    ("greedy74-like (5 lib)", ["lookup", "tabm_seed3", "naji03_ref", "XGB", "HGB"]),
    ("NN + EXP122 (50/50)", ["lookup", "EXP122"]),
    ("NN + EXP130 (50/50)", ["lookup", "EXP130"]),
    ("NN + greedy74-like", ["lookup", "tabm_seed3", "naji03_ref", "XGB", "HGB", "EXP122"]),
    ("NN + EXP122 + EXP130", ["lookup", "tabm_seed3", "EXP122", "EXP130"]),
    ("NN + EXP122 + EXP130 + greedy74", ["lookup", "tabm_seed3", "naji03_ref", "XGB", "HGB", "EXP122", "EXP130"]),
    ("Full pool (NN+tiers+EXP)", ["lookup", "tabm_seed3", "FAST", "MED-B", "NUC-REF", "XGB", "HGB", "CAT", "EXP122", "EXP130"]),
]

# Filter to only keys that exist
blend_configs = [(name, [k for k in members if k in keys]) for name, members in blend_configs]

M_all = np.column_stack([V[k] for k in keys])

print(f"\n{'Config':<45} {'OOF AUC':>10} {'Held-out':>10} {'Δ vs EXP122':>12}")
print("-" * 80)

blend_results = {}
for name, members in blend_configs:
    if not members:
        continue
    cols = [keys.index(m) for m in members]

    # Full OOF
    if len(members) == 1:
        full_auc = roc_auc_score(y, V[members[0]])
        held_auc = roc_auc_score(y[hold_idx], V[members[0]][hold_idx])
    else:
        # Optimized weights
        w = fit_blend_weights(cols, y[fit_idx], M_all[fit_idx])
        full_pred = np.clip(M_all[:, cols] @ w, 1e-9, 1 - 1e-9)
        full_auc = roc_auc_score(y, full_pred)
        held_pred = np.clip(M_all[hold_idx][:, cols] @ w, 1e-9, 1 - 1e-9)
        held_auc = roc_auc_score(y[hold_idx], held_pred)

    delta = full_auc - roc_auc_score(y, V["EXP122"])
    blend_results[name] = {"full_auc": full_auc, "held_auc": held_auc, "members": members}

    weights_str = ""
    if len(members) > 1:
        w_final = fit_blend_weights(cols, y[fit_idx], M_all[fit_idx])
        top_w = [(members[i], w_final[i]) for i in range(len(members)) if w_final[i] > 0.01]
        weights_str = " | " + ", ".join(f"{m}={w:.3f}" for m, w in top_w)

    print(f"{name:<45} {full_auc:>10.6f} {held_auc:>10.6f} {delta:>+12.6f}{weights_str}")

# ============================================================
# PHASE 3: META-LEARNING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: META-LEARNING (strictly OOF)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold

# Use a curated pool for meta-learning
meta_pool = ["lookup", "tabm_seed3", "EXP122", "EXP130", "XGB", "HGB"]
meta_pool = [k for k in meta_pool if k in keys]
meta_cols = [keys.index(k) for k in meta_pool]

print(f"Meta-learner base pool: {meta_pool}")

# Generate OOF meta-features via inner CV (5-fold)
n_inner = 5
skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
meta_oof = np.zeros((N, len(meta_pool)))

for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
    for j, k in enumerate(meta_pool):
        meta_oof[va_idx, j] = V[k][va_idx]

print(f"\nBase OOF AUCs:")
for j, k in enumerate(meta_pool):
    print(f"  {k}: {roc_auc_score(y, meta_oof[:, j]):.6f}")

# --- Meta-learner 1: Logistic Regression ---
lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
lr.fit(meta_oof[fit_idx], y[fit_idx])
lr_pred = lr.predict_proba(meta_oof)[:, 1]
lr_auc = roc_auc_score(y, lr_pred)
lr_held = roc_auc_score(y[hold_idx], lr_pred[hold_idx])
print(f"\nLogistic Regression meta-learner:")
print(f"  Full OOF: {lr_auc:.6f}")
print(f"  Held-out: {lr_held:.6f}")
print(f"  Δ vs EXP122: {lr_auc - roc_auc_score(y, V['EXP122']):+.6f}")
print(f"  Coefficients:")
for j, k in enumerate(meta_pool):
    print(f"    {k}: {lr.coef_[0][j]:+.4f}")

# --- Meta-learner 2: Ridge (regularized linear) ---
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge = RidgeCV(alphas=alphas, scoring="neg_log_loss", cv=5)
ridge.fit(meta_oof[fit_idx], y[fit_idx])
ridge_pred = ridge.predict(meta_oof)
ridge_pred = np.clip(ridge_pred, 1e-9, 1 - 1e-9)
ridge_auc = roc_auc_score(y, ridge_pred)
ridge_held = roc_auc_score(y[hold_idx], ridge_pred[hold_idx])
print(f"\nRidge meta-learner (alpha={ridge.alpha_:.4f}):")
print(f"  Full OOF: {ridge_auc:.6f}")
print(f"  Held-out: {ridge_held:.6f}")
print(f"  Δ vs EXP122: {ridge_auc - roc_auc_score(y, V['EXP122']):+.6f}")
print(f"  Coefficients:")
for j, k in enumerate(meta_pool):
    print(f"    {k}: {ridge.coef_[j]:+.4f}")

# --- Meta-learner 3: LightGBM (small tree, if justified) ---
# Only if linear meta-learners show improvement
if lr_auc > roc_auc_score(y, V["EXP122"]) + 0.0002 or ridge_auc > roc_auc_score(y, V["EXP122"]) + 0.0002:
    print(f"\nLinear meta-learners show improvement — testing LightGBM meta-learner")
    try:
        import lightgbm as lgb
        lgb_meta = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
        )
        lgb_meta.fit(meta_oof[fit_idx], y[fit_idx])
        lgb_pred = lgb_meta.predict_proba(meta_oof)[:, 1]
        lgb_auc = roc_auc_score(y, lgb_pred)
        lgb_held = roc_auc_score(y[hold_idx], lgb_pred[hold_idx])
        print(f"  Full OOF: {lgb_auc:.6f}")
        print(f"  Held-out: {lgb_held:.6f}")
        print(f"  Δ vs EXP122: {lgb_auc - roc_auc_score(y, V['EXP122']):+.6f}")
        print(f"  Feature importances:")
        for j, k in enumerate(meta_pool):
            print(f"    {k}: {lgb_meta.feature_importances_[j]}")
    except Exception as e:
        print(f"  LightGBM meta-learner failed: {e}")
else:
    print(f"\nLinear meta-learners did not show meaningful improvement — skipping tree meta-learner")

# ============================================================
# PHASE 4: ABLATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: ABLATION (remove one base model at a time)")
print("=" * 70)

# Start with the best meta-learner or best simple blend
# Use the full pool and test leave-one-out
full_pool = ["lookup", "tabm_seed3", "EXP122", "EXP130", "XGB", "HGB"]
full_pool = [k for k in full_pool if k in keys]
full_cols = [keys.index(k) for k in full_pool]
full_w = fit_blend_weights(full_cols, y[fit_idx], M_all[fit_idx])
full_pred = np.clip(M_all[:, full_cols] @ full_w, 1e-9, 1 - 1e-9)
full_auc = roc_auc_score(y, full_pred)
print(f"\nFull blend ({len(full_pool)} members): OOF {full_auc:.6f}")
for i, k in enumerate(full_pool):
    if full_w[i] > 0.005:
        print(f"  {k}: weight {full_w[i]:.4f}")

print(f"\nLeave-one-out ablation:")
print(f"{'Removed':<20} {'OOF AUC':>10} {'Δ vs full':>10} {'NN contrib':>12}")
print("-" * 55)

for remove_k in full_pool:
    ablation_pool = [k for k in full_pool if k != remove_k]
    ablation_cols = [keys.index(k) for k in ablation_pool]
    ablation_w = fit_blend_weights(ablation_cols, y[fit_idx], M_all[fit_idx])
    ablation_pred = np.clip(M_all[:, ablation_cols] @ ablation_w, 1e-9, 1 - 1e-9)
    ablation_auc = roc_auc_score(y, ablation_pred)
    delta = ablation_auc - full_auc
    marker = " **HURTS**" if delta < -0.0002 else (" HELPFUL" if delta > 0.0002 else "")
    print(f"-{remove_k:<19} {ablation_auc:>10.6f} {delta:>+10.6f}{marker}")

# Test: owned-only vs owned+NN
print(f"\n--- NN contribution test ---")
owned_only = ["EXP122", "EXP130", "XGB", "HGB"]
owned_only = [k for k in owned_only if k in keys]
owned_cols = [keys.index(k) for k in owned_only]
owned_w = fit_blend_weights(owned_cols, y[fit_idx], M_all[fit_idx])
owned_pred = np.clip(M_all[:, owned_cols] @ owned_w, 1e-9, 1 - 1e-9)
owned_auc = roc_auc_score(y, owned_pred)
print(f"Owned-only blend: {owned_auc:.6f}")

# Add each NN one at a time
for nn_k in ["lookup", "tabm_seed3"]:
    if nn_k not in keys:
        continue
    combo = owned_only + [nn_k]
    combo_cols = [keys.index(k) for k in combo]
    combo_w = fit_blend_weights(combo_cols, y[fit_idx], M_all[fit_idx])
    combo_pred = np.clip(M_all[:, combo_cols] @ combo_w, 1e-9, 1 - 1e-9)
    combo_auc = roc_auc_score(y, combo_pred)
    nn_weight = combo_w[-1]
    print(f"  +{nn_k}: {combo_auc:.6f} (Δ {combo_auc - owned_auc:+.6f}, nn_weight={nn_weight:.4f})")

# ============================================================
# PHASE 5: ROBUSTNESS CHECKS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: ROBUSTNESS CHECKS")
print("=" * 70)

# Test the best configuration across multiple random seeds for the OOS split
best_config = ["lookup", "tabm_seed3", "EXP122", "EXP130"]
best_config = [k for k in best_config if k in keys]
best_cols = [keys.index(k) for k in best_config]

print(f"\nBest NN+owned config: {best_config}")
print(f"Testing across 5 random OOS splits:")

seed_results = []
for seed in [42, 123, 7, 2024, 99]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    fi, hi = idx[:400000], idx[400000:]
    w = fit_blend_weights(best_cols, y[fi], M_all[fi])
    pred = np.clip(M_all[:, best_cols] @ w, 1e-9, 1 - 1e-9)
    full_auc = roc_auc_score(y, pred)
    held_auc = roc_auc_score(y[hi], pred[hi])
    nn_w = [w[best_config.index(k)] for k in ["lookup", "tabm_seed3"] if k in best_config]
    seed_results.append({"seed": seed, "full": full_auc, "held": held_auc, "nn_weights": nn_w})
    nn_w_str = ", ".join(f"{k}={w:.3f}" for k, w in zip(["lookup", "tabm_seed3"], nn_w) if k in best_config)
    print(f"  seed={seed}: full={full_auc:.6f} held={held_auc:.6f} nn_w=[{nn_w_str}]")

# Stability metrics
full_aucs = [r["full"] for r in seed_results]
held_aucs = [r["held"] for r in seed_results]
print(f"\n  Full OOF: mean={np.mean(full_aucs):.6f} std={np.std(full_aucs):.6f} range={np.ptp(full_aucs):.6f}")
print(f"  Held-out: mean={np.mean(held_aucs):.6f} std={np.std(held_aucs):.6f} range={np.ptp(held_aucs):.6f}")

# NN weight stability
nn_ws = [r["nn_weights"] for r in seed_results]
for i, nn_k in enumerate(["lookup", "tabm_seed3"]):
    if nn_k in best_config:
        ws = [nw[i] for nw in nn_ws]
        print(f"  {nn_k} weight: mean={np.mean(ws):.4f} std={np.std(ws):.4f} range={np.ptp(ws):.4f}")

# Also test: does NN help at ALL across every split?
exp122_auc = roc_auc_score(y, V["EXP122"])
nn_improves_count = sum(1 for r in seed_results if r["full"] > exp122_auc + 0.0001)
print(f"\n  NN improves over EXP122 alone in {nn_improves_count}/{len(seed_results)} splits")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

baseline_auc = roc_auc_score(y, V["EXP122"])
print(f"\nOwned baseline (EXP122):        {baseline_auc:.6f}")
print(f"Best simple NN blend:           {max(r['full_auc'] for r in blend_results.values() if 'NN' in r.get('name', '') or 'lookup' in str(r.get('members', []))):.6f}" if any('NN' in r.get('name', '') or 'lookup' in str(r.get('members', [])) for r in blend_results.values()) else "")

# Best overall
best_name = max(blend_results, key=lambda k: blend_results[k]["full_auc"])
best = blend_results[best_name]
print(f"Best simple blend: {best_name}")
print(f"  Full OOF: {best['full_auc']:.6f}")
print(f"  Held-out: {best['held_auc']:.6f}")
print(f"  Δ vs EXP122: {best['full_auc'] - baseline_auc:+.6f}")

print(f"\nMeta-learner results:")
print(f"  Logistic: {lr_auc:.6f} (held {lr_held:.6f})")
print(f"  Ridge:    {ridge_auc:.6f} (held {ridge_held:.6f})")

print(f"\nRobustness: NN weight stability across seeds")
for nn_k in ["lookup", "tabm_seed3"]:
    if nn_k in best_config:
        i = best_config.index(nn_k)
        ws = [r["nn_weights"][i] for r in seed_results]
        print(f"  {nn_k}: mean={np.mean(ws):.4f} std={np.std(ws):.4f}")

print(f"\nProvenance: external-NN (lookup, tabm_seed3) + owned (EXP122, EXP130, tier OOFs)")
print(f"Method: OOF-only evaluation, no public LB tuning")
