"""Missingness-as-features experiment for S6E8.

The adversarial ladder (dariushafshar) shows missingness patterns carry
most of the predictive signal. Test whether explicitly encoding missingness
as features in the rank-gauss logistic stack improves OOF.

Run: gh workflow run analysis.yml -f script_path=shared/analysis/missingness_features.py
"""
import os
import sys
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
print("=== MISSINGNESS FEATURES EXPERIMENT ===", flush=True)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test = pd.read_csv("competitions/s6e8/data/test.csv")

# ---- Missingness features ----
feature_cols = [c for c in tr.columns if c not in ("id", "addicted_label")]
miss_train = tr[feature_cols].isna().astype(np.float32).values
miss_test = test[feature_cols].isna().astype(np.float32).values
print(f"Missingness flags: {miss_train.shape[1]} columns", flush=True)

# Also create pairwise missingness interactions for top features
n_feat = miss_train.shape[1]
interact_pairs = []
for i in range(n_feat):
    for j in range(i + 1, n_feat):
        interact_pairs.append((i, j))
miss_interact_train = np.column_stack([miss_train[:, i] * miss_train[:, j] for i, j in interact_pairs]).astype(np.float32)
miss_interact_test = np.column_stack([miss_test[:, i] * miss_test[:, j] for i, j in interact_pairs]).astype(np.float32)
print(f"Pairwise interactions: {miss_interact_train.shape[1]}", flush=True)

# Count how many rows have each missingness pattern
n_missing_per_row_train = miss_train.sum(axis=1)
n_missing_per_row_test = miss_test.sum(axis=1)
print(f"Missing per row (train): min={n_missing_per_row_train.min()}, max={n_missing_per_row_train.max()}, mean={n_missing_per_row_train.mean():.2f}", flush=True)

# Load the full rank-gauss stack members
import kagglehub

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

def load_vectors(oof_dir, prefix, test_prefixes=("test_", "testpred_", "tep_")):
    out = {}
    seen = set()
    for path in glob.glob(os.path.join(oof_dir, "**", "oof_*.npy"), recursive=True):
        name = os.path.basename(path)[4:-4]
        mate = next(
            (c for tp in test_prefixes
             for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
             if os.path.exists(c)),
            None,
        )
        if mate is None:
            continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if (oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,)
                and np.isfinite(oof).all() and np.isfinite(tst).all()):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    for path in glob.glob(os.path.join(oof_dir, "**", "*_oof.npy"), recursive=True):
        bname = os.path.basename(path)
        name = bname[:-8]
        mate = os.path.join(os.path.dirname(path), name + "_test.npy")
        if not os.path.exists(mate):
            mate = next(
                (c for tp in test_prefixes
                 for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
                 if os.path.exists(c)),
                None,
            )
        if mate is None:
            continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if (oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,)
                and np.isfinite(oof).all() and np.isfinite(tst).all()):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    return out

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
        print(f"{dataset:44s} {len(got):3d}", flush=True)
    except Exception as e:
        print(f"{dataset:44s} SKIP ({e})", flush=True)

# boltuzamaki
try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    oof_df = pd.read_parquet(os.path.join(bolt, "oof_predictions.parquet"))
    tst_df = pd.read_parquet(os.path.join(bolt, "test_predictions.parquet"))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (oof_df[col].to_numpy(float), tst_df[col].to_numpy(float))
    print(f"boltuzamaki parquet: {oof_df.shape[1]-1} cols", flush=True)
except Exception as e:
    print(f"boltuzamaki SKIP ({e})", flush=True)

# weak-50
try:
    weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
    WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
    WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
    for j in range(WO.shape[1]):
        members[f"weak_{j:02d}"] = (np.asarray(WO[:, j], float), np.asarray(WT[:, j], float))
    print(f"weak-50: {WO.shape[1]} cols", flush=True)
except Exception as e:
    print(f"weak-50 SKIP ({e})", flush=True)

# owned + RealMLP + screen-time
for tag, oof_path, test_path in [
    ("own_champ_m10", "shared/analysis/data/own_champ_m10_oof.npy", "shared/analysis/data/own_champ_m10_test.npy"),
    ("rmlp_realmlp", "shared/analysis/data/oof_realmlp.npy", "shared/analysis/data/pred_realmlp.npy"),
]:
    try:
        members[tag] = (np.load(oof_path).astype(np.float64), np.load(test_path).astype(np.float64))
        print(f"{tag}: loaded", flush=True)
    except Exception as e:
        print(f"{tag}: SKIP ({e})", flush=True)

for tag in ["catboost", "lgb3seed"]:
    try:
        members[f"fresh_{tag}"] = (
            np.load(f"shared/analysis/data/oof_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
            np.load(f"shared/analysis/data/test_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
        )
        print(f"fresh_{tag}: loaded", flush=True)
    except Exception as e:
        print(f"fresh_{tag}: SKIP ({e})", flush=True)

# ---- rank + dedup ----
names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)

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
print(f"{len(drop)} near-duplicate dropped", flush=True)

SELF_REFERENTIAL = re.compile(r"^(naji|sz_naji|v13_anchor|hb_candidate)")
keep = [i for i, n in enumerate(names) if n not in drop and not SELF_REFERENTIAL.match(n)]
names = [names[i] for i in keep]
R, Rt = R[:, keep], Rt[:, keep]
print(f"{len(names)} members kept", flush=True)

G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt, 1e-7, 1 - 1e-7)).astype(np.float32)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

# ============================================================
# EXPERIMENT 1: Baseline (rank-gauss only)
# ============================================================
print("\n=== EXPERIMENT 1: Baseline (rank-gauss logistic) ===", flush=True)
oof_base = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_base[vi] = fit_logistic(G[fi], y[fi], G[vi])
base_auc = roc_auc_score(y, oof_base)
print(f"Baseline OOF: {base_auc:.6f}", flush=True)

# ============================================================
# EXPERIMENT 2: Rank-gauss + missingness flags
# ============================================================
print("\n=== EXPERIMENT 2: RG + missingness flags ===", flush=True)
G_miss = np.column_stack([G, miss_train])
oof_miss = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_miss[vi] = fit_logistic(G_miss[fi], y[fi], G_miss[vi])
miss_auc = roc_auc_score(y, oof_miss)
print(f"RG + miss flags OOF: {miss_auc:.6f} (delta={miss_auc - base_auc:+.6f})", flush=True)

# ============================================================
# EXPERIMENT 3: Rank-gauss + missingness + interactions
# ============================================================
print("\n=== EXPERIMENT 3: RG + miss + interactions ===", flush=True)
G_miss_int = np.column_stack([G, miss_train, miss_interact_train])
oof_miss_int = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_miss_int[vi] = fit_logistic(G_miss_int[fi], y[fi], G_miss_int[vi])
miss_int_auc = roc_auc_score(y, oof_miss_int)
print(f"RG + miss + interact OOF: {miss_int_auc:.6f} (delta={miss_int_auc - base_auc:+.6f})", flush=True)

# ============================================================
# EXPERIMENT 4: Missingness features only (baseline comparison)
# ============================================================
print("\n=== EXPERIMENT 4: Missingness only ===", flush=True)
oof_miss_only = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_miss_only[vi] = fit_logistic(miss_train[fi], y[fi], miss_train[vi])
miss_only_auc = roc_auc_score(y, oof_miss_only)
print(f"Miss-only OOF: {miss_only_auc:.6f}", flush=True)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}", flush=True)
print("MISSINGNESS FEATURES SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Baseline (RG only):          {base_auc:.6f}", flush=True)
print(f"  RG + miss flags:             {miss_auc:.6f}  ({miss_auc - base_auc:+.6f})", flush=True)
print(f"  RG + miss + interactions:    {miss_int_auc:.6f}  ({miss_int_auc - base_auc:+.6f})", flush=True)
print(f"  Miss-only:                   {miss_only_auc:.6f}", flush=True)

best_delta = max(miss_auc - base_auc, miss_int_auc - base_auc)
if best_delta > 0.0001:
    print(f"\n*** MISSINGNESS FEATURES SHOW IMPROVEMENT: +{best_delta:.6f} ***", flush=True)
    print("This could be a genuinely new signal source for the stack.", flush=True)
    # If RG + miss is best, use it for submissions
    if miss_int_auc > miss_auc:
        best_test = fit_logistic(G_miss_int, y, np.column_stack([Gt, miss_test, miss_interact_test]))
        best_name = "RG + miss + interactions"
        best_oof = miss_int_auc
    else:
        best_test = fit_logistic(G_miss, y, np.column_stack([Gt, miss_test]))
        best_name = "RG + miss flags"
        best_oof = miss_auc
else:
    print(f"\nNo significant improvement from missingness features.", flush=True)
    best_test = fit_logistic(G, y, Gt)
    best_name = "Baseline (RG only)"
    best_oof = base_auc

# ---- Emit submissions ----
try:
    naji = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
    cand = None
    for f in sorted(glob.glob(os.path.join(naji, "*.csv"))):
        df = pd.read_csv(f)
        if "id" in df.columns and "addicted_label" in df.columns and len(df) == N_TEST:
            preds = df["addicted_label"].to_numpy(float)
            if (preds >= 0).all() and (preds <= 1).all():
                cand = preds
                break
    public = pct_rank(cand) if cand is not None else None
except:
    public = None

if best_delta > 0.0001 and public is not None:
    for W in [0.25, 0.30, 0.35, 0.40]:
        final = pct_rank(W * pct_rank(best_test) + (1 - W) * public)
        sub = pd.DataFrame({"id": test["id"], "addicted_label": final})
        fn = f"submission_miss_w{W:.2f}.csv"
        sub.to_csv(fn, index=False)
        print(f"  Emitted: {fn}", flush=True)

print(f"\n=== DONE ===", flush=True)
