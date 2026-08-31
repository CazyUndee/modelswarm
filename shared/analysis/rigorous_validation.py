#!/usr/bin/env python3
"""Rigorous validation framework for S6E8 stack.

This script:
1. Loads all public OOF members
2. Computes per-member OOF AUC
3. Computes pairwise Spearman correlations
4. Builds baseline stack OOF
5. Tests incremental member additions
6. Measures residual diversity (which models make different errors)
7. Analyzes the OOF-to-LB generalization gap
8. Produces a decision report for each candidate
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import kagglehub
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ============================================================
# SECTION 1: Load all members
# ============================================================
print("="*60)
print("SECTION 1: LOADING ALL MEMBERS")
print("="*60)

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

# naji 19
try:
    naji_path = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
    for f in sorted(os.listdir(naji_path)):
        if '19' in f and '_oof_predictions.csv' in f:
            df = pd.read_csv(os.path.join(naji_path, f))
            pc = [c for c in df.columns if c != 'id'][0]
            oof = df[pc].values
            for tf in os.listdir(naji_path):
                if '19' in tf and 'submission' in tf:
                    try:
                        tdf = pd.read_csv(os.path.join(naji_path, tf))
                        tc = [c for c in tdf.columns if c != 'id'][0]
                        if len(tdf) == N_TEST and len(oof) == N_TRAIN:
                            members["naj_19"] = (oof, tdf[tc].values)
                            print(f"naj_19: AUC={roc_auc_score(y, oof):.6f}")
                            break
                    except: pass
            break
except Exception as e:
    print(f"naj_19 SKIP ({e})")

print(f"\nTotal: {len(members)} members")

# ============================================================
# SECTION 2: Per-member OOF AUC
# ============================================================
print("\n" + "="*60)
print("SECTION 2: PER-MEMBER OOF AUC")
print("="*60)

names = sorted(members.keys())
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])

auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)
print(f"\nAUC range: {auc.min():.6f} .. {auc.max():.6f}")
print(f"Mean AUC: {auc.mean():.6f}")
print(f"Median AUC: {auc.median():.6f}")

print("\n--- Top 20 by AUC ---")
for name, a in auc.sort_values(ascending=False).head(20).items():
    print(f"  {name:50s} {a:.6f}")

print("\n--- Bottom 10 by AUC ---")
for name, a in auc.sort_values(ascending=True).head(10).items():
    print(f"  {name:50s} {a:.6f}")

# ============================================================
# SECTION 3: Correlation analysis (sampled)
# ============================================================
print("\n" + "="*60)
print("SECTION 3: CORRELATION ANALYSIS")
print("="*60)

R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)
Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
n = R.shape[1]

# Compute correlation matrix for top-60 AUC members
top_names = auc.sort_values(ascending=False).head(60).index.tolist()
top_idx = [names.index(n) for n in top_names]
R_top = R[:, top_idx]
Z_top = (R_top - R_top.mean(0)) / (R_top.std(0) + 1e-12)
corr_top = (Z_top.T @ Z_top) / len(Z_top)

# Most correlated pairs
print("\n--- Most correlated pairs (top-60) ---")
pairs = []
for i in range(len(top_idx)):
    for j in range(i+1, len(top_idx)):
        pairs.append((top_names[i], top_names[j], corr_top[i, j]))
pairs.sort(key=lambda x: -x[2])
for n1, n2, c in pairs[:15]:
    print(f"  {n1:40s} <-> {n2:40s} corr={c:.4f}")

# Least correlated pairs
print("\n--- Least correlated pairs (top-60) ---")
for n1, n2, c in pairs[-15:]:
    print(f"  {n1:40s} <-> {n2:40s} corr={c:.4f}")

# Average correlation for each member
mean_corr = corr_top.mean(axis=1)
print("\n--- Members with lowest mean correlation (most unique) ---")
for i in np.argsort(mean_corr)[:10]:
    print(f"  {top_names[i]:50s} mean_corr={mean_corr[i]:.4f} AUC={auc[top_names[i]]:.6f}")

# ============================================================
# SECTION 4: Baseline stack
# ============================================================
print("\n" + "="*60)
print("SECTION 4: BASELINE STACK")
print("="*60)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

def run_stack(oof_matrix, test_matrix, folds, y):
    G = norm.ppf(np.clip(oof_matrix, 1e-7, 1 - 1e-7)).astype(np.float32)
    Gt = norm.ppf(np.clip(test_matrix, 1e-7, 1 - 1e-7)).astype(np.float32)
    oof = np.zeros(N_TRAIN)
    for fi, vi in folds:
        oof[vi] = fit_logistic(G[fi], y[fi], G[vi])
    test_pred = fit_logistic(G, y, Gt)
    return oof, test_pred

# Dedup near-duplicates
drop = set()
for i in range(n):
    if names[i] in drop: continue
    for j in range(i+1, n):
        if names[j] in drop: continue
        c = np.corrcoef(R[:, i], R[:, j])[0, 1]
        if c > 0.9995:
            drop.add(names[j] if auc[names[i]] >= auc[names[j]] else names[i])
print(f"Dropped {len(drop)} near-duplicates (corr>0.9995)")

keep = [i for i in range(n) if names[i] not in drop]
names_keep = [names[i] for i in keep]
R_keep = R[:, keep]
TST_keep = TST[:, keep]
print(f"Kept {len(keep)} members")

# Baseline
oof_base, test_base = run_stack(R_keep, TST_keep, folds, y)
base_auc = roc_auc_score(y, oof_base)
print(f"\nBASELINE: {len(keep)} members, OOF={base_auc:.6f}")

# ============================================================
# SECTION 5: Fold stability analysis
# ============================================================
print("\n" + "="*60)
print("SECTION 5: FOLD STABILITY")
print("="*60)

fold_aucs = []
for fi, vi in folds:
    fold_auc = roc_auc_score(y[vi], oof_base[vi])
    fold_aucs.append(fold_auc)
    print(f"  Fold {len(fold_aucs)}: AUC={fold_auc:.6f}, n={len(vi)}")

print(f"  Mean: {np.mean(fold_aucs):.6f}, Std: {np.std(fold_aucs):.6f}")
print(f"  Range: {min(fold_aucs):.6f} .. {max(fold_aucs):.6f}")

# ============================================================
# SECTION 6: OOF-to-LB generalization gap
# ============================================================
print("\n" + "="*60)
print("SECTION 6: OOF-TO-LB GENERALIZATION GAP")
print("="*60)

print(f"\nStack OOF: {base_auc:.6f}")
print(f"Best LB: 0.97131")
print(f"Gap: {0.97131 - base_auc:+.6f}")

# The vault base contributes to the LB
# vault OOF (from train labels) vs vault test predictions
# We can't compute vault OOF directly, but we can analyze the gap

# Load vault test predictions
vault_path = kagglehub.dataset_download("anthonytherrien/predicting-smartphone-addiction-vault")
for f in os.listdir(vault_path):
    if f.endswith('.csv'):
        try:
            vdf = pd.read_csv(os.path.join(vault_path, f))
            if 'addicted_label' in vdf.columns and len(vdf) >= N_TRAIN:
                vault_oof = vdf['addicted_label'].values[:N_TRAIN]
                vault_test = vdf['addicted_label'].values[N_TRAIN:N_TRAIN+N_TEST]
                vault_auc = roc_auc_score(y, vault_oof)
                print(f"Vault: OOF={vault_auc:.6f}")
                
                # What W gives best LB?
                # LB = W * stack_test + (1-W) * vault_test, then rank
                # Since we can't measure LB locally, analyze the OOF blend
                for W in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                    blend = pct_rank(W * pct_rank(oof_base) + (1-W) * pct_rank(vault_oof))
                    auc_b = roc_auc_score(y, blend)
                    print(f"  W={W:.2f}: blend OOF={auc_b:.6f}")
                break
        except: pass

# ============================================================
# SECTION 7: Residual diversity analysis
# ============================================================
print("\n" + "="*60)
print("SECTION 7: RESIDUAL DIVERSITY")
print("="*60)

# Compute stack residuals (where stack is wrong)
stack_errors = np.abs(y - oof_base)
print(f"Stack error distribution: mean={stack_errors.mean():.4f}, p95={np.percentile(stack_errors, 95):.4f}")

# For each candidate member, compute:
# 1. How often it agrees with the stack when stack is wrong
# 2. How often it corrects stack errors
# 3. Residual correlation with stack errors

# Use top-20 members by AUC
top20 = auc.sort_values(ascending=False).head(20).index.tolist()
print(f"\n--- Residual diversity of top-20 members vs stack ---")
print(f"{'Member':50s} 'corr_with_stack_err' 'corrects_when_wrong' 'agrees_when_wrong'")
for name in top20:
    member_oof = members[name][0]
    member_pred = (member_oof > 0.5).astype(float)
    stack_pred = (oof_base > 0.5).astype(float)
    
    # When stack is wrong
    wrong_mask = stack_pred != y
    if wrong_mask.sum() > 0:
        corrects = (member_pred[wrong_mask] == y[wrong_mask]).mean()
    else:
        corrects = 0
    
    # When stack is right
    right_mask = stack_pred == y
    if right_mask.sum() > 0:
        agrees = (member_pred[right_mask] == stack_pred[right_mask]).mean()
    else:
        agrees = 0
    
    # Correlation of OOF errors
    member_errors = np.abs(y - member_oof)
    err_corr = np.corrcoef(stack_errors, member_errors)[0, 1]
    
    print(f"  {name:50s} {err_corr:8.4f} {corrects:8.4f} {agrees:8.4f}")

# ============================================================
# SECTION 8: Incremental addition test
# ============================================================
print("\n" + "="*60)
print("SECTION 8: INCREMENTAL ADDITION TEST")
print("="*60)

# For each of the top-20 members, test if adding it improves the baseline
R_baseline = R[:, keep].copy()
G_baseline = norm.ppf(np.clip(R_baseline, 1e-7, 1 - 1e-7)).astype(np.float32)

# First, test adding naji19
naj19_idx = names.index("naj_19") if "naj_19" in names else -1
if naj19_idx >= 0 and naj19_idx not in drop:
    R_with_naj19 = np.column_stack([R_baseline, R[:, naj19_idx:naj19_idx+1]])
    G_with_naj19 = norm.ppf(np.clip(R_with_naj19, 1e-7, 1 - 1e-7)).astype(np.float32)
    oof_with_naj19, _ = run_stack(R_with_naj19, np.column_stack([TST_keep, TST[:, naj19_idx:naj19_idx+1]]), folds, y)
    auc_naj19 = roc_auc_score(y, oof_with_naj19)
    print(f"\nAdding naji19: OOF={auc_naj19:.6f} (delta={auc_naj19-base_auc:+.6f})")
    
    # Correlation with stack
    corr_naj19 = np.corrcoef(R_baseline.mean(axis=1), R[:, naj19_idx])[0, 1]
    print(f"  Correlation with stack: {corr_naj19:.4f}")

# Test each top-20 member individually
print(f"\n--- Incremental addition of top-20 members ---")
results = []
for name in top20:
    idx = names.index(name)
    if idx in drop:
        continue
    R_trial = np.column_stack([R_baseline, R[:, idx:idx+1]])
    oof_trial, _ = run_stack(R_trial, np.column_stack([TST_keep, TST[:, idx:idx+1]]), folds, y)
    trial_auc = roc_auc_score(y, oof_trial)
    delta = trial_auc - base_auc
    corr = np.corrcoef(R_baseline.mean(axis=1), R[:, idx])[0, 1]
    results.append((name, trial_auc, delta, corr))
    print(f"  {name:50s} OOF={trial_auc:.6f} delta={delta:+.6f} corr={corr:.4f}")

# Sort by delta
print("\n--- Ranked by OOF improvement ---")
for name, a, d, c in sorted(results, key=lambda x: -x[2]):
    marker = " *** POSITIVE" if d > 0 else ""
    print(f"  {name:50s} delta={d:+.6f} corr={c:.4f}{marker}")

print("\nDONE")
