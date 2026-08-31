#!/usr/bin/env python3
"""Add naji OOF members to the existing rank-gauss stack and evaluate.

Key finding: naji 19_blend individual AUC=0.970099, close to our stack's 0.970221.
If decorrelated enough, it could push the stack higher.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ---- Load all existing members from rank_gauss_stack.py ----
# (reuses the same sources list)
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
        if mate is None:
            continue
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
        if mate is None:
            continue
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

# ---- NEW: Add naji OOF members ----
print("\n=== LOADING NAJI OOF MEMBERS ===")
naji_path = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
naji_oof_count = 0
for f in sorted(os.listdir(naji_path)):
    if f.endswith("_oof_predictions.csv"):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        oof = df[pred_col].values
        name = f.replace("_oof_predictions.csv", "")
        # Find matching test submission
        test_file = os.path.join(naji_path, f"{name}_submission.csv")
        if os.path.exists(test_file) and len(oof) == N_TRAIN:
            tdf = pd.read_csv(test_file)
            tpred_col = [c for c in tdf.columns if c != 'id'][0]
            test_pred = tdf[tpred_col].values
            if len(test_pred) == N_TEST:
                auc = roc_auc_score(y, oof)
                members[f"naj_{name}"] = (oof, test_pred)
                naji_oof_count += 1
                print(f"  naj_{name}: AUC={auc:.6f}")
    elif f.endswith("_blend_oof_predictions.csv"):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        oof = df[pred_col].values
        name = f.replace("_blend_oof_predictions.csv", "")
        test_file = os.path.join(naji_path, f"{name}_blend_submission.csv")
        if os.path.exists(test_file) and len(oof) == N_TRAIN:
            tdf = pd.read_csv(test_file)
            tpred_col = [c for c in tdf.columns if c != 'id'][0]
            test_pred = tdf[tpred_col].values
            if len(test_pred) == N_TEST:
                auc = roc_auc_score(y, oof)
                members[f"naj_{name}"] = (oof, test_pred)
                naji_oof_count += 1
                print(f"  naj_{name}: AUC={auc:.6f}")
print(f"Loaded {naji_oof_count} naji OOF members")

# ---- Build matrices ----
names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)
print(f"\n{len(names)} members total")

# ---- Rank + dedup ----
R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)
Rt = np.column_stack([pct_rank(TST[:, j]) for j in range(TST.shape[1])]).astype(np.float32)

Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
corr = (Z.T @ Z) / len(Z)
del Z
drop = set()
for i in range(len(names)):
    if names[i] in drop:
        continue
    for j in range(i + 1, len(names)):
        if names[j] in drop or corr[i, j] <= 0.9995:
            continue
        drop.add(names[j] if auc[names[i]] >= auc[names[j]] else names[i])
print(f"{len(drop)} near-duplicates dropped")

# Keep ALL members including naji (no SELF_REFERENTIAL filter)
keep = [i for i, n in enumerate(names) if n not in drop]
names_kept = [names[i] for i in keep]
R, Rt = R[:, keep], Rt[:, keep]
print(f"{len(names_kept)} members kept (including naji)")

# Check how many naji survived
naji_kept = [n for n in names_kept if n.startswith("naj_")]
print(f"Naji members kept: {len(naji_kept)}: {naji_kept}")

# ---- Logistic stack ----
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt, 1e-7, 1 - 1e-7)).astype(np.float32)

oof_meta = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_meta[vi] = fit_logistic(G[fi], y[fi], G[vi])
stack_auc = roc_auc_score(y, oof_meta)
print(f"\n*** STACK OOF AUC (with naji): {stack_auc:.6f} ***")
print(f"*** Previous best (without naji): 0.970221 ***")
print(f"*** Delta: {stack_auc - 0.970221:+.6f} ***")

# ---- Compare: stack WITHOUT naji ----
print("\n=== CONTROL: Stack WITHOUT naji ===")
keep_no_naji = [i for i, n in enumerate(names) if n not in drop and not n.startswith("naj_")]
names_no_naji = [names[i] for i in keep_no_naji]
R_no = np.column_stack([pct_rank(OOF[:, j]) for j in keep_no_naji]).astype(np.float32)
G_no = norm.ppf(np.clip(R_no, 1e-7, 1 - 1e-7)).astype(np.float32)
oof_no = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_no[vi] = fit_logistic(G_no[fi], y[fi], G_no[vi])
no_naji_auc = roc_auc_score(y, oof_no)
print(f"Stack without naji: {no_naji_auc:.6f} ({len(names_no_naji)} members)")
print(f"Stack with naji:    {stack_auc:.6f} ({len(names_kept)} members)")
print(f"Improvement from naji: {stack_auc - no_naji_auc:+.6f}")

# ---- Blend with vault ----
try:
    vault_path = kagglehub.dataset_download("anthonytherrien/predicting-smartphone-addiction-vault")
    vault_files = [f for f in os.listdir(vault_path) if f.endswith('.csv')]
    if vault_files:
        vault_df = pd.read_csv(os.path.join(vault_path, vault_files[0]))
        vault_pred = vault_df['addicted_label'].values[:N_TRAIN]
        if len(vault_pred) == N_TRAIN:
            vault_auc = roc_auc_score(y, vault_pred)
            print(f"\nVault OOF: {vault_auc:.6f}")
            best_w, best_blend_auc = 0, 0
            for w in np.arange(0, 1.05, 0.05):
                blend = w * oof_meta + (1-w) * vault_pred
                bl_auc = roc_auc_score(y, blend)
                if bl_auc > best_blend_auc:
                    best_blend_auc = bl_auc
                    best_w = w
            print(f"Best blend: W={best_w:.2f}, AUC={best_blend_auc:.6f}")
except Exception as e:
    print(f"Vault: {e}")

# ---- Generate submission ----
test_meta = fit_logistic(G, y, Gt)
sub = pd.DataFrame({"id": test_df["id"], "addicted_label": pct_rank(test_meta)})
sub.to_csv("submission_naji_stack.csv", index=False)
print(f"\nSaved submission_naji_stack.csv")
print("DONE")
