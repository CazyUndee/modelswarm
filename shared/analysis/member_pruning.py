#!/usr/bin/env python3
"""Member pruning experiment: find which members hurt the stack.

The 177-member stack by dariushafshar/adarsh1077 uses distribution-drift
quarantine to remove harmful members. We have 252+ members but haven't
pruned them. This experiment:
1. Computes per-member contribution (OOF AUC, correlation with errors)
2. Tests dropping weakest members
3. Tests dropping most-correlated clusters
4. Reports whether pruning improves the stack
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import kagglehub
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test_df = pd.read_csv("competitions/s6e8/data/test.csv")

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

def load_vectors(oof_dir, prefix, test_prefixes=("test_", "testpred_", "tep_")):
    out = {}
    seen = set()
    for path in glob.glob(os.path.join(oof_dir, "**", "oof_*.npy"), recursive=True):
        name = os.path.basename(path)[4:-4]
        mate = next((c for tp in test_prefixes
                     for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
                     if os.path.exists(c)), None)
        if mate is None: continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    for path in glob.glob(os.path.join(oof_dir, "**", "*_oof.npy"), recursive=True):
        bname = os.path.basename(path)
        name = bname[:-8]
        mate = os.path.join(os.path.dirname(path), name + "_test.npy")
        if not os.path.exists(mate):
            mate = next((c for tp in test_prefixes
                         for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
                         if os.path.exists(c)), None)
        if mate is None: continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    return out

# Load all members (same sources as rank_gauss_stack.py)
SOURCES = [
    ("szymonkapiski/s6e8-oof-library-47-models", "sz_"),
    ("paiky1995/s6e8-oof-library-11-members", "nn_"),
    ("aadijoshi19/s6e8-mask-augmented-oof-library", "ma_"),
    ("tamerlanomralinov/s6e8-full-best-blend-npy", "tam_"),
    ("adarsh1077/s6e8-adarsh-oof-library", "a_"),
    ("dariushafshar/s6e8-golem-oof-library", "golem_"),
    ("raykkretzschmar/s6e8-fm-lattice-blend-members", "fm_"),
    ("hboyang/s6e8-catstrall-member", "x_"),
    ("hboyang/s6e8-150-fusion-local-members", "hb_"),
    ("masayakawamata/s6e8-catstr-aug16", "mk_"),
    ("beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts", "bc_"),
    ("beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts", "bd_"),
    ("beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts", "be_"),
    ("beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts", "bf_"),
    ("beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts", "bg_"),
    ("beicicc/s6e8-fixed900-structural-lgbm-artifacts", "bh_"),
]

members = {}
for dataset, prefix in SOURCES:
    try:
        root = kagglehub.dataset_download(dataset)
        got = load_vectors(root, prefix)
        members.update(got)
        print(f"{dataset:44s} {len(got):3d}")
    except Exception as e:
        print(f"{dataset:44s} SKIP ({e})")

# boltuzamaki
try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    bolt_files = [f for f in os.listdir(bolt) if f.endswith('.parquet')]
    oof_df = pd.read_parquet(os.path.join(bolt, [f for f in bolt_files if 'oof' in f.lower()][0]))
    tst_df = pd.read_parquet(os.path.join(bolt, [f for f in bolt_files if 'test' in f.lower()][0]))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (oof_df[col].to_numpy(float), tst_df[col].to_numpy(float))
    print(f"boltuzamaki parquet: {oof_df.shape[1]-1} cols")
except Exception as e:
    print(f"boltuzamaki SKIP ({e})")

# weak50
try:
    weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
    WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
    WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
    for j in range(WO.shape[1]):
        members[f"weak_{j:02d}"] = (np.asarray(WO[:, j], float), np.asarray(WT[:, j], float))
    print(f"weak-50: {WO.shape[1]} cols")
except Exception as e:
    print(f"weak-50 SKIP ({e})")

# RealMLP
try:
    members["rmlp_realmlp"] = (
        np.load("shared/analysis/data/oof_realmlp.npy").astype(np.float64),
        np.load("shared/analysis/data/pred_realmlp.npy").astype(np.float64),
    )
    print("rmlp_realmlp: OK")
except Exception as e:
    print(f"rmlp_realmlp SKIP ({e})")

# Build matrices
names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)
print(f"\n{len(names)} members total, AUC range: {auc.min():.5f} .. {auc.max():.5f}")

# ---- SECTION 1: Per-member analysis ----
print("\n" + "="*60)
print("SECTION 1: MEMBER ANALYSIS")
print("="*60)

# Individual AUC
print("\n--- Top 15 by individual AUC ---")
for name, a in auc.sort_values(ascending=False).head(15).items():
    print(f"  {name:50s} {a:.6f}")

print("\n--- Bottom 15 by individual AUC ---")
for name, a in auc.sort_values(ascending=True).head(15).items():
    print(f"  {name:50s} {a:.6f}")

# ---- SECTION 2: Correlation matrix (sampled for speed) ----
print("\n" + "="*60)
print("SECTION 2: CORRELATION ANALYSIS")
print("="*60)

# Rank-transform
R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)

# Compute mean pairwise correlation for each member
n_members = R.shape[1]
mean_corr = np.zeros(n_members)
max_corr = np.zeros(n_members)
most_similar = [''] * n_members

# Sample for speed: only compute for top-50 AUC members
top_idx = auc.sort_values(ascending=False).head(50).index
top_positions = [names.index(n) for n in top_idx]

Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
corr_matrix = (Z.T @ Z) / len(Z)

for i in range(n_members):
    corrs = corr_matrix[i, :]
    corrs_no_self = np.delete(corrs, i)
    mean_corr[i] = corrs_no_self.mean()
    max_corr[i] = corrs_no_self.max()
    most_similar[i] = names[np.argmax(np.delete(corrs, i)) if i > 0 else 1]

# Members with highest max correlation (most redundant)
print("\n--- Most redundant members (highest max corr with any other) ---")
sorted_by_max = np.argsort(-max_corr)[:20]
for idx in sorted_by_max:
    print(f"  {names[idx]:50s} max_corr={max_corr[idx]:.4f} (most similar: {most_similar[idx]})")

# Members with lowest mean correlation (most unique)
print("\n--- Most unique members (lowest mean corr with others) ---")
sorted_by_mean = np.argsort(mean_corr)[:20]
for idx in sorted_by_mean:
    print(f"  {names[idx]:50s} mean_corr={mean_corr[idx]:.4f}")

# ---- SECTION 3: Baseline stack ----
print("\n" + "="*60)
print("SECTION 3: BASELINE STACK (all members)")
print("="*60)

# Dedup near-duplicates
drop = set()
for i in range(n_members):
    if names[i] in drop: continue
    for j in range(i + 1, n_members):
        if names[j] in drop or corr_matrix[i, j] <= 0.9995: continue
        drop.add(names[j] if auc[names[i]] >= auc[names[j]] else names[i])
print(f"{len(drop)} near-duplicates dropped (0.9995 threshold)")

keep = [i for i in range(n_members) if names[i] not in drop]
names_keep = [names[i] for i in keep]
R_keep = R[:, keep]
G = norm.ppf(np.clip(R_keep, 1e-7, 1 - 1e-7)).astype(np.float32)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

oof_base = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_base[vi] = fit_logistic(G[fi], y[fi], G[vi])
base_auc = roc_auc_score(y, oof_base)
print(f"Baseline: {len(names_keep)} members, OOF AUC = {base_auc:.6f}")

# ---- SECTION 4: Prune low-AUC members ----
print("\n" + "="*60)
print("SECTION 4: PRUNE LOW-AUC MEMBERS")
print("="*60)

# Keep only members above various AUC thresholds
for threshold in [0.960, 0.963, 0.965, 0.967, 0.969]:
    pruned_keep = [i for i in keep if auc[names[i]] >= threshold]
    if len(pruned_keep) < 10:
        continue
    R_pruned = R[:, pruned_keep]
    G_pruned = norm.ppf(np.clip(R_pruned, 1e-7, 1 - 1e-7)).astype(np.float32)
    oof_pruned = np.zeros(N_TRAIN)
    for fi, vi in folds:
        oof_pruned[vi] = fit_logistic(G_pruned[fi], y[fi], G_pruned[vi])
    pruned_auc = roc_auc_score(y, oof_pruned)
    delta = pruned_auc - base_auc
    print(f"  AUC >={threshold:.3f}: {len(pruned_keep):3d} members, OOF={pruned_auc:.6f} (delta={delta:+.6f})")

# ---- SECTION 5: Drop most-correlated members ----
print("\n" + "="*60)
print("SECTION 5: DROP MOST-CORRELATED MEMBERS")
print("="*60)

# Greedy: iteratively drop the member that hurts least
current_keep = list(range(len(names_keep)))
R_current = R_keep.copy()

for n_drop in [5, 10, 20, 30, 50]:
    if n_drop >= len(current_keep):
        break
    # Find the member whose removal hurts least
    best_remove = -1
    best_oof = -1
    for idx in range(len(current_keep)):
        trial_keep = [i for j, i in enumerate(current_keep) if j != idx]
        R_trial = R_current[:, trial_keep]
        G_trial = norm.ppf(np.clip(R_trial, 1e-7, 1 - 1e-7)).astype(np.float32)
        oof_trial = np.zeros(N_TRAIN)
        for fi, vi in folds:
            oof_trial[vi] = fit_logistic(G_trial[fi], y[fi], G_trial[vi])
        trial_auc = roc_auc_score(y, oof_trial)
        if trial_auc > best_oof:
            best_oof = trial_auc
            best_remove = idx
    removed_name = names_keep[current_keep[best_remove]]
    current_keep.pop(best_remove)
    R_current = R_current[:, [i for i in range(R_current.shape[1]) if i != best_remove]]
    # Recompute after full prune
    R_final = R[:, [keep[i] for i in current_keep]]
    G_final = norm.ppf(np.clip(R_final, 1e-7, 1 - 1e-7)).astype(np.float32)
    oof_final = np.zeros(N_TRAIN)
    for fi, vi in folds:
        oof_final[vi] = fit_logistic(G_final[fi], y[fi], G_final[vi])
    final_auc = roc_auc_score(y, oof_final)
    print(f"  Drop #{n_drop:3d}: removed {removed_name}, {len(current_keep)} remain, OOF={final_auc:.6f} (delta={final_auc-base_auc:+.6f})")

print("\nDONE")
