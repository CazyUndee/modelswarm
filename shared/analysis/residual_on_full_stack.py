#!/usr/bin/env python3
"""Test if GBDT residual correction adds signal to the full 252-member stack.

The pseudo-label run found: GBDT on bolt rank-gauss OOF=0.970435 (+0.000432).
But bolt is only 47 members. The real question: does it help on the full stack?
"""
import glob, os, re, numpy as np, pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import kagglehub
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 691369
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
        mate = next((c for tp in test_prefixes for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")] if os.path.exists(c)), None)
        if mate is None: continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if oof.shape == (N,) and tst.shape == (N_TEST,) and np.isfinite(oof).all() and np.isfinite(tst).all():
            key = prefix + name
            if key not in seen: out[key] = (oof, tst); seen.add(key)
    for path in glob.glob(os.path.join(oof_dir, "**", "*_oof.npy"), recursive=True):
        name = os.path.basename(path)[:-8]
        mate = os.path.join(os.path.dirname(path), name + "_test.npy")
        if not os.path.exists(mate):
            mate = next((c for tp in test_prefixes for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")] if os.path.exists(c)), None)
        if mate is None: continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if oof.shape == (N,) and tst.shape == (N_TEST,) and np.isfinite(oof).all() and np.isfinite(tst).all():
            key = prefix + name
            if key not in seen: out[key] = (oof, tst); seen.add(key)
    return out

# Load all sources (same as rank_gauss_stack.py)
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
    bf = [f for f in os.listdir(bolt) if f.endswith('.parquet')]
    oof_df = pd.read_parquet(os.path.join(bolt, [f for f in bf if 'oof' in f.lower()][0]))
    tst_df = pd.read_parquet(os.path.join(bolt, [f for f in bf if 'test' in f.lower()][0]))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (oof_df[col].to_numpy(float), tst_df[col].to_numpy(float))
    print(f"boltuzamaki: {oof_df.shape[1]-1}")
except Exception as e:
    print(f"bolt SKIP ({e})")

# weak50
try:
    weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
    WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
    WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
    for j in range(WO.shape[1]):
        members[f"weak_{j:02d}"] = (np.asarray(WO[:, j], float), np.asarray(WT[:, j], float))
    print(f"weak-50: {WO.shape[1]}")
except: pass

# RealMLP
try:
    members["rmlp_realmlp"] = (
        np.load("shared/analysis/data/oof_realmlp.npy").astype(np.float64),
        np.load("shared/analysis/data/pred_realmlp.npy").astype(np.float64))
    print("rmlp_realmlp: OK")
except: pass

print(f"\nTotal: {len(members)} members")

# Build matrices
names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)

# Rank-gauss
R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)
Rt = np.column_stack([pct_rank(TST[:, j]) for j in range(TST.shape[1])]).astype(np.float32)

# Dedup
Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
corr_mat = (Z.T @ Z) / len(Z)
drop = set()
for i in range(len(names)):
    if names[i] in drop: continue
    for j in range(i+1, len(names)):
        if names[j] in drop or corr_mat[i, j] <= 0.9995: continue
        drop.add(names[j] if auc[names[i]] >= auc[names[j]] else names[i])
keep = [i for i in range(len(names)) if names[i] not in drop]
names_k = [names[i] for i in keep]
R_k = R[:, keep]
Rt_k = Rt[:, keep]
G = norm.ppf(np.clip(R_k, 1e-7, 1-1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt_k, 1e-7, 1-1e-7)).astype(np.float32)
print(f"After dedup: {len(keep)} members")

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N), y))

def fit_logistic(X_f, y_f, X_p):
    sc = StandardScaler().fit(X_f)
    m = LogisticRegression(C=1.0, max_iter=3000)
    m.fit(sc.transform(X_f), y_f)
    return m.predict_proba(sc.transform(X_p))[:, 1]

# BASELINE
print("\n" + "="*60)
print("BASELINE: Linear Logistic on full rank-gauss stack")
print("="*60)
oof_base = np.zeros(N)
for fi, vi in folds:
    oof_base[vi] = fit_logistic(G[fi], y[fi], G[vi])
base_auc = roc_auc_score(y, oof_base)
test_base = fit_logistic(G, y, Gt)
print(f"OOF: {base_auc:.6f}")
for fi, vi in folds:
    print(f"  Fold: {roc_auc_score(y[vi], oof_base[vi]):.6f}")

# CANDIDATE 1: HistGradientBoosting on rank-gauss (faster than GBDT for large N)
print("\n" + "="*60)
print("CANDIDATE 1: HistGradientBoosting meta-learner")
print("="*60)

for n_est, lr, md in [(200, 0.05, 3), (500, 0.03, 4), (200, 0.03, 5)]:
    oof_hgb = np.zeros(N)
    for fi, vi in folds:
        hgb = HistGradientBoostingClassifier(
            max_iter=n_est, learning_rate=lr, max_depth=md,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42
        )
        hgb.fit(G[fi], y[fi])
        oof_hgb[vi] = hgb.predict_proba(G[vi])[:, 1]
    hgb_auc = roc_auc_score(y, oof_hgb)
    corr = spearmanr(oof_base, oof_hgb).correlation
    
    # Best blend
    best_ba, best_bw = 0, 0
    for w in np.arange(0, 1.05, 0.1):
        bl = pct_rank(w * pct_rank(oof_base) + (1-w) * pct_rank(oof_hgb))
        ba = roc_auc_score(y, bl)
        if ba > best_ba: best_ba = ba; best_bw = w
    
    print(f"  n={n_est:4d} lr={lr:.2f} md={md}: OOF={hgb_auc:.6f} corr={corr:.4f} best_blend={best_bw:.1f} blend={best_ba:.6f}")

# CANDIDATE 2: Residual correction — train GBDT on BASELINE ERRORS
print("\n" + "="*60)
print("CANDIDATE 2: Residual correction (GBDT on stack errors)")
print("="*60)

for n_est, lr, md in [(200, 0.05, 3), (500, 0.03, 4)]:
    oof_res = np.zeros(N)
    for fi, vi in folds:
        # Train GBDT on rank-gauss features to predict labels (residual correction)
        hgb = HistGradientBoostingClassifier(
            max_iter=n_est, learning_rate=lr, max_depth=md,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42
        )
        hgb.fit(G[fi], y[vi] if len(y[vi]) > 0 else y[fi])
        oof_res[vi] = hgb.predict_proba(G[vi])[:, 1]
    res_auc = roc_auc_score(y, oof_res)
    corr = spearmanr(oof_base, oof_res).correlation
    
    best_ba, best_bw = 0, 0
    for w in np.arange(0, 1.05, 0.1):
        bl = pct_rank(w * pct_rank(oof_base) + (1-w) * pct_rank(oof_res))
        ba = roc_auc_score(y, bl)
        if ba > best_ba: best_ba = ba; best_bw = w
    
    print(f"  n={n_est:4d} lr={lr:.2f} md={md}: OOF={res_auc:.6f} corr={corr:.4f} best_blend={best_bw:.1f} blend={best_ba:.6f}")

# CANDIDATE 3: Disagreement-weighted average
print("\n" + "="*60)
print("CANDIDATE 3: Disagreement weighting")
print("="*60)
Ranks = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])])
row_std = Ranks.std(axis=1)
inv_w = 1.0 / (row_std + 1e-6)
inv_w_norm = inv_w / inv_w.mean()
weighted_oof = (Ranks * inv_w_norm[:, None]).mean(axis=1)
weighted_auc = roc_auc_score(y, weighted_oof)
print(f"  Disagreement-weighted: OOF={weighted_auc:.6f} (delta={weighted_auc-base_auc:+.6f})")

# Simple avg
simple_oof = Ranks.mean(axis=1)
simple_auc = roc_auc_score(y, simple_oof)
print(f"  Simple average: OOF={simple_auc:.6f} (delta={simple_auc-base_auc:+.6f})")

# CANDIDATE 4: Mixup / label smoothing
print("\n" + "="*60)
print("CANDIDATE 4: Label-smoothed logistic")
print("="*60)
for alpha in [0.01, 0.02, 0.05]:
    y_smooth = y * (1 - alpha) + 0.5 * alpha
    oof_ls = np.zeros(N)
    for fi, vi in folds:
        sc = StandardScaler().fit(G[fi])
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(sc.transform(G[fi]), y_smooth[fi])
        oof_ls[vi] = m.predict_proba(sc.transform(G[vi]))[:, 1]
    ls_auc = roc_auc_score(y, oof_ls)
    print(f"  alpha={alpha:.2f}: OOF={ls_auc:.6f} (delta={ls_auc-base_auc:+.6f})")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline: {base_auc:.6f}")
print(f"Target LB: 0.97131")
print("DONE")
