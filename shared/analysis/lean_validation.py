#!/usr/bin/env python3
"""Lean validation: residual correction + incremental bolt member testing.

Uses only bolt library (fast, cached). Tests whether a GBDT trained on
rank-gauss stack features adds signal to the logistic baseline.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

N = 691369
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test_df = pd.read_csv("competitions/s6e8/data/test.csv")
N_TEST = len(test_df)

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

# Load bolt
import kagglehub, os
bolt_path = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
bolt_files = [f for f in os.listdir(bolt_path) if f.endswith('.parquet')]
bolt_oof = pd.read_parquet(os.path.join(bolt_path, [f for f in bolt_files if 'oof' in f.lower()][0]))
bolt_tst = pd.read_parquet(os.path.join(bolt_path, [f for f in bolt_files if 'test' in f.lower()][0]))
bolt_cols = [c for c in bolt_oof.columns if c != 'id']
print(f"bolt: {len(bolt_cols)} members")

# Build member matrices
OOF = np.column_stack([bolt_oof[c].to_numpy(float) for c in bolt_cols])
TST = np.column_stack([bolt_tst[c].to_numpy(float) for c in bolt_cols])

# Rank-gauss transform
R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)
Rt = np.column_stack([pct_rank(TST[:, j]) for j in range(TST.shape[1])]).astype(np.float32)
G = norm.ppf(np.clip(R, 1e-7, 1-1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt, 1e-7, 1-1e-7)).astype(np.float32)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N), y))

def fit_logistic(X_fit, y_fit, X_pred):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=1.0, max_iter=3000)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

# ============================================================
# BASELINE: Linear logistic on rank-gauss features
# ============================================================
print("\n" + "="*60)
print("BASELINE: Linear Logistic on bolt rank-gauss")
print("="*60)
oof_base = np.zeros(N)
for fi, vi in folds:
    oof_base[vi] = fit_logistic(G[fi], y[fi], G[vi])
base_auc = roc_auc_score(y, oof_base)
test_base = fit_logistic(G, y, Gt)
print(f"OOF: {base_auc:.6f}")

fold_base = [roc_auc_score(y[vi], oof_base[vi]) for _, vi in folds]
print(f"Fold AUCs: {[f'{x:.6f}' for x in fold_base]}")
print(f"Fold mean: {np.mean(fold_base):.6f}, std: {np.std(fold_base):.6f}")

# ============================================================
# CANDIDATE 1: GBDT residual correction on rank-gauss features
# ============================================================
print("\n" + "="*60)
print("CANDIDATE 1: GradientBoosting residual correction")
print("="*60)

for n_est, lr, md in [(200, 0.05, 3), (500, 0.03, 4), (1000, 0.02, 3)]:
    oof_gbdt = np.zeros(N)
    for fi, vi in folds:
        gbdt = GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=md,
            subsample=0.8, random_state=42
        )
        gbdt.fit(G[fi], y[fi])
        oof_gbdt[vi] = gbdt.predict_proba(G[vi])[:, 1]
    gbdt_auc = roc_auc_score(y, oof_gbdt)
    
    # Blend with linear logistic
    best_blend_auc = 0
    best_w = 0
    for w in np.arange(0.0, 1.05, 0.1):
        blend = pct_rank(w * pct_rank(oof_base) + (1-w) * pct_rank(oof_gbdt))
        ba = roc_auc_score(y, blend)
        if ba > best_blend_auc:
            best_blend_auc = ba
            best_w = w
    
    corr = spearmanr(oof_base, oof_gbdt).correlation
    print(f"  n_est={n_est:4d}, lr={lr:.2f}, md={md}: OOF={gbdt_auc:.6f}, "
          f"corr={corr:.4f}, best_blend_W={best_w:.1f}, blend_OOF={best_blend_auc:.6f}")
    
    # Fold stability of blend
    oof_blend = pct_rank(best_w * pct_rank(oof_base) + (1-best_w) * pct_rank(oof_gbdt))
    fold_blend = [roc_auc_score(y[vi], oof_blend[vi]) for _, vi in folds]
    print(f"    Blend folds: {[f'{x:.6f}' for x in fold_blend]}")

# ============================================================
# CANDIDATE 2: Per-member OOF AUC and correlation analysis
# ============================================================
print("\n" + "="*60)
print("CANDIDATE 2: Per-member analysis for bolt library")
print("="*60)

member_auc = {c: roc_auc_score(y, OOF[:, j]) for j, c in enumerate(bolt_cols)}
print("\n--- Top 10 by individual AUC ---")
for c, a in sorted(member_auc.items(), key=lambda x: -x[1])[:10]:
    print(f"  {c:50s} {a:.6f}")

print("\n--- Bottom 5 by individual AUC ---")
for c, a in sorted(member_auc.items(), key=lambda x: x[1])[:5]:
    print(f"  {c:50s} {a:.6f}")

# ============================================================
# CANDIDATE 3: Feature interactions (top-5 x top-5 pairs)
# ============================================================
print("\n" + "="*60)
print("CANDIDATE 3: Feature interactions (top-5 rank-gauss pairs)")
print("="*60)

top5_names = sorted(member_auc, key=member_auc.get, reverse=True)[:5]
top5_idx = [bolt_cols.index(c) for c in top5_names]
R_top5 = R[:, top5_idx]

# Build interaction features
interactions = []
for i in range(len(top5_idx)):
    for j in range(i+1, len(top5_idx)):
        inter = R_top5[:, i] * R_top5[:, j]
        interactions.append(inter)
        
if interactions:
    R_inter = np.column_stack(interactions).astype(np.float32)
    G_inter = norm.ppf(np.clip(R_inter, 1e-7, 1-1e-7)).astype(np.float32)
    
    # Baseline + interactions
    R_both = np.column_stack([R, R_inter]).astype(np.float32)
    G_both = norm.ppf(np.clip(R_both, 1e-7, 1-1e-7)).astype(np.float32)
    
    oof_both = np.zeros(N)
    for fi, vi in folds:
        oof_both[vi] = fit_logistic(G_both[fi], y[fi], G_both[vi])
    both_auc = roc_auc_score(y, oof_both)
    print(f"  Baseline + interactions: OOF={both_auc:.6f} (delta={both_auc-base_auc:+.6f})")
    
    # Interaction-only
    oof_inter = np.zeros(N)
    for fi, vi in folds:
        oof_inter[vi] = fit_logistic(G_inter[fi], y[fi], G_inter[vi])
    inter_auc = roc_auc_score(y, oof_inter)
    print(f"  Interactions only: OOF={inter_auc:.6f}")

# ============================================================
# CANDIDATE 4: Disagreement-based weighting
# ============================================================
print("\n" + "="*60)
print("CANDIDATE 4: Disagreement-weighted average")
print("="*60)

# Compute per-member rank predictions
Ranks = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])])

# For each row, compute: weight = 1 / (std of member predictions)
# Members that agree get lower weight, members that disagree get higher weight
row_std = Ranks.std(axis=1)
# Inverse weight: rows with high disagreement get averaged more carefully
inv_weight = 1.0 / (row_std + 1e-6)
inv_weight_norm = inv_weight / inv_weight.mean()

# Weighted average
weighted_oof = (Ranks * inv_weight_norm[:, None]).mean(axis=1)
weighted_auc = roc_auc_score(y, weighted_oof)
print(f"  Disagreement-weighted avg: OOF={weighted_auc:.6f} (delta={weighted_auc-base_auc:+.6f})")

# Simple average for comparison
simple_oof = Ranks.mean(axis=1)
simple_auc = roc_auc_score(y, simple_oof)
print(f"  Simple avg: OOF={simple_auc:.6f} (delta={simple_auc-base_auc:+.6f})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline (linear logistic on rank-gauss): {base_auc:.6f}")
print(f"Target to beat LB: 0.97131")
print(f"Gap: {0.97131 - base_auc:+.6f}")
print()
print("Each candidate needs OOF delta >0.0005 AND low correlation to justify a submission")
print()
print("DONE")
