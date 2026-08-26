"""Meta-Feature Stacking — polynomial interactions on OOF predictions.

Hypothesis: A learned meta-model on OOF predictions with polynomial
interactions captures nonlinear complementarity that simple weighted
average misses.

Community evidence: discussion 733023 reported CV 0.96947 / LB 0.97059
using OOF predictions + polynomial interactions as a single model.

Tests:
1. Simple OOF stacking (baseline — matches our earlier investigation)
2. Quadratic interaction features (pairwise OOF products)
3. Ridge/Logistic on interactions
4. LightGBM on interactions
5. Comparison vs simple weighted average
6. Robustness across seeds
"""
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("META-FEATURE STACKING INVESTIGATION")
print("=" * 70)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)
print(f"Training rows: {N}")

V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb"]:
    V[n] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  {n}: OOF {roc_auc_score(y, V[n]):.6f}")

# Load owned vectors
V["EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
print(f"  EXP122: OOF {roc_auc_score(y, V['EXP122']):.6f}")
print(f"  EXP130: OOF {roc_auc_score(y, V['EXP130']):.6f}")

# Load EXP-132 if available
e132_path = "competitions/s6e8/experiments/output/EXP-132/oof_predictions.csv"
if os.path.exists(e132_path):
    V["lookup_xform"] = pd.read_csv(e132_path)["prediction"].values
    print(f"  lookup_xform: OOF {roc_auc_score(y, V['lookup_xform']):.6f}")
else:
    print("  lookup_xform: NOT YET AVAILABLE")

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
# BASE POOLS
# ============================================================
# Pool 1: Core (NN + owned, matches our earlier best)
core_keys = ["lookup", "tabm_seed3", "EXP122", "EXP130"]
# Pool 2: Extended (adds library members)
ext_keys = ["lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb", "EXP122", "EXP130"]
# Pool 3: With lookup-transformer if available
if "lookup_xform" in V:
    ext_xform_keys = ext_keys + ["lookup_xform"]
else:
    ext_xform_keys = ext_keys

core_keys = [k for k in core_keys if k in V]
ext_keys = [k for k in ext_keys if k in V]
ext_xform_keys = [k for k in ext_xform_keys if k in V]

keys = list(V.keys())
M = np.column_stack([V[k] for k in keys])

# OOS split
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

print(f"\nCore pool ({len(core_keys)}): {core_keys}")
print(f"Extended pool ({len(ext_keys)}): {ext_keys}")
if "lookup_xform" in V:
    print(f"Extended+XF pool ({len(ext_xform_keys)}): {ext_xform_keys}")

# ============================================================
# PHASE 1: BASELINE — simple weighted average
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: BASELINE (simple weighted average)")
print("=" * 70)

for pool_name, pool_keys in [("core", core_keys), ("extended", ext_keys), ("extended+xform", ext_xform_keys)]:
    cols = [keys.index(k) for k in pool_keys]
    w = fit_weights(cols, y[fit_idx], M[fit_idx])
    pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    held = roc_auc_score(y[hold_idx], pred[hold_idx])
    w_str = ", ".join(f"{pool_keys[i]}={w[i]:.3f}" for i in range(len(pool_keys)) if w[i] > 0.01)
    print(f"  {pool_name:15s}: OOF={auc:.6f} held={held:.6f} [{w_str}]")

# ============================================================
# PHASE 2: META-FEATURE GENERATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: META-FEATURE STACKING")
print("=" * 70)

# Generate OOF meta-features via inner CV (5-fold) to avoid leakage
n_inner = 5
skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)

# For each pool, generate OOF meta-features
for pool_name, pool_keys in [("core", core_keys), ("extended", ext_keys)]:
    print(f"\n--- Pool: {pool_name} ({pool_keys}) ---")
    cols = [keys.index(k) for k in pool_keys]
    
    # Generate OOF meta-features (inner CV)
    meta_oof = np.zeros((N, len(pool_keys)))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
        for j, k in enumerate(pool_keys):
            meta_oof[va_idx, j] = V[k][va_idx]
    
    # Generate interaction features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    meta_interactions = poly.fit_transform(meta_oof)
    interaction_names = poly.get_feature_names_out(pool_keys)
    
    n_base = len(pool_keys)
    n_interactions = meta_interactions.shape[1] - n_base
    print(f"  Base features: {n_base}, Interaction features: {n_interactions}")
    
    # --- Test 1: Ridge on base OOF ---
    ridge_base = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
    ridge_base.fit(meta_oof[fit_idx], y[fit_idx])
    ridge_pred = np.clip(ridge_base.predict(meta_oof), 1e-9, 1 - 1e-9)
    ridge_auc = roc_auc_score(y, ridge_pred)
    ridge_held = roc_auc_score(y[hold_idx], ridge_pred[hold_idx])
    print(f"  Ridge (base OOF):      OOF={ridge_auc:.6f} held={ridge_held:.6f}")
    
    # --- Test 2: Ridge on base + interactions ---
    ridge_inter = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
    ridge_inter.fit(meta_interactions[fit_idx], y[fit_idx])
    ridge_inter_pred = np.clip(ridge_inter.predict(meta_interactions), 1e-9, 1 - 1e-9)
    ridge_inter_auc = roc_auc_score(y, ridge_inter_pred)
    ridge_inter_held = roc_auc_score(y[hold_idx], ridge_inter_pred[hold_idx])
    delta = ridge_inter_auc - ridge_auc
    print(f"  Ridge (interactions):  OOF={ridge_inter_auc:.6f} held={ridge_inter_held:.6f} Δ={delta:+.6f}")
    
    # Show top interaction coefficients
    top_idx = np.argsort(np.abs(ridge_inter.coef_))[-10:][::-1]
    print(f"  Top interaction coeffs:")
    for idx in top_idx:
        if idx < len(interaction_names):
            print(f"    {interaction_names[idx]}: {ridge_inter.coef_[idx]:+.4f}")
    
    # --- Test 3: Logistic on base OOF ---
    lr_base = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr_base.fit(meta_oof[fit_idx], y[fit_idx])
    lr_pred = lr_base.predict_proba(meta_oof)[:, 1]
    lr_auc = roc_auc_score(y, lr_pred)
    lr_held = roc_auc_score(y[hold_idx], lr_pred[hold_idx])
    print(f"  Logistic (base OOF):   OOF={lr_auc:.6f} held={lr_held:.6f}")
    
    # --- Test 4: Logistic on base + interactions ---
    lr_inter = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    lr_inter.fit(meta_interactions[fit_idx], y[fit_idx])
    lr_inter_pred = lr_inter.predict_proba(meta_interactions)[:, 1]
    lr_inter_auc = roc_auc_score(y, lr_inter_pred)
    lr_inter_held = roc_auc_score(y[hold_idx], lr_inter_pred[hold_idx])
    delta = lr_inter_auc - lr_auc
    print(f"  Logistic (interactions): OOF={lr_inter_auc:.6f} held={lr_inter_held:.6f} Δ={delta:+.6f}")
    
    # --- Test 5: LightGBM on interactions (if justified) ---
    if ridge_inter_auc > ridge_auc + 0.0001 or lr_inter_auc > lr_auc + 0.0001:
        try:
            import lightgbm as lgb
            lgb_meta = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
            )
            lgb_meta.fit(meta_interactions[fit_idx], y[fit_idx])
            lgb_pred = lgb_meta.predict_proba(meta_interactions)[:, 1]
            lgb_auc = roc_auc_score(y, lgb_pred)
            lgb_held = roc_auc_score(y[hold_idx], lgb_pred[hold_idx])
            delta_lgb = lgb_auc - ridge_auc
            print(f"  LightGBM (interactions): OOF={lgb_auc:.6f} held={lgb_held:.6f} Δ_vs_ridge_base={delta_lgb:+.6f}")
            
            # Feature importances — which interactions matter most?
            fi = lgb_meta.feature_importances_
            top_fi = np.argsort(fi)[-5:][::-1]
            print(f"  Top LightGBM features:")
            for idx in top_fi:
                if idx < len(interaction_names):
                    print(f"    {interaction_names[idx]}: importance={fi[idx]}")
        except Exception as e:
            print(f"  LightGBM failed: {e}")
    
    # --- Test 6: Simple quadratic features (pairwise products only) ---
    n_pairs = n_interactions
    pair_features = meta_interactions[:, n_base:]  # interaction columns only
    # Concatenate base + pairs
    meta_pairs = np.hstack([meta_oof, pair_features])
    
    ridge_pairs = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
    ridge_pairs.fit(meta_pairs[fit_idx], y[fit_idx])
    ridge_pairs_pred = np.clip(ridge_pairs.predict(meta_pairs), 1e-9, 1 - 1e-9)
    ridge_pairs_auc = roc_auc_score(y, ridge_pairs_pred)
    ridge_pairs_held = roc_auc_score(y[hold_idx], ridge_pairs_pred[hold_idx])
    print(f"  Ridge (base+pairs):    OOF={ridge_pairs_auc:.6f} held={ridge_pairs_held:.6f}")

# ============================================================
# PHASE 3: ROBUSTNESS CHECK
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: ROBUSTNESS (best meta-learner across seeds)")
print("=" * 70)

# Use the best configuration from Phase 2
# Test across 5 seeds
cols_core = [keys.index(k) for k in core_keys]

print(f"\nTesting core pool ({core_keys}) meta-learners across 5 seeds:")
for seed in [42, 123, 7, 2024, 99]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    fi, hi = idx[:400000], idx[400000:]
    
    # Generate OOF meta-features for this fold
    meta_fi = np.column_stack([V[k][fi] for k in core_keys])
    meta_hi = np.column_stack([V[k][hi] for k in core_keys])
    y_fi, y_hi = y[fi], y[hi]
    
    # Ridge on base
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
    ridge.fit(meta_fi, y_fi)
    ridge_pred = np.clip(ridge.predict(meta_hi), 1e-9, 1 - 1e-9)
    ridge_auc = roc_auc_score(y_hi, ridge_pred)
    
    # Ridge on interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    meta_fi_inter = poly.fit_transform(meta_fi)
    meta_hi_inter = poly.transform(meta_hi)
    ridge_inter = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
    ridge_inter.fit(meta_fi_inter, y_fi)
    ridge_inter_pred = np.clip(ridge_inter.predict(meta_hi_inter), 1e-9, 1 - 1e-9)
    ridge_inter_auc = roc_auc_score(y_hi, ridge_inter_pred)
    
    # Simple weighted average for comparison
    w = fit_weights(cols_core, y_fi, M[fi])
    simple_pred = np.clip(M[hi][:, cols_core] @ w, 1e-9, 1 - 1e-9)
    simple_auc = roc_auc_score(y_hi, simple_pred)
    
    print(f"  seed={seed}: simple={simple_auc:.6f} ridge_base={ridge_auc:.6f} ridge_inter={ridge_inter_auc:.6f} Δ_inter={ridge_inter_auc-simple_auc:+.6f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey question: do polynomial interactions on OOF predictions add value")
print("over simple weighted average stacking?")
print("\nIf interactions help: this is a genuine new mechanism for frontier attack.")
print("If not: the complementarity is already captured by linear combination.")

baseline = roc_auc_score(y, V["naji03"])
print(f"\nFrontier (naji03) OOF:     {baseline:.6f}")
print(f"Live LB top:               0.97186 (Chris Deotte)")
print(f"Best owned blend:          0.96942 (NN+owned)")
print(f"\nProvenance: external (library) + owned (EXP122/130)")
print(f"Method: inner-CV OOF meta-features, no LB tuning")
