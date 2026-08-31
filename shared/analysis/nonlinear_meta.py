"""Nonlinear meta-ensemble: LightGBM + XGBoost on rank-gauss OOF features.

Tests whether GBDT meta-learners capture nonlinear interactions between
member predictions that the linear logistic misses. Falls back to linear
if no genuine OOF improvement.

Run on GHA via: gh workflow run analysis.yml -f script_path=shared/analysis/nonlinear_meta.py
"""
import sys
import os
import glob
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
print("=== NONLINEAR META-ENSEMBLE SCRIPT STARTED ===", flush=True)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test = pd.read_csv("competitions/s6e8/data/test.csv")
print(f"Data loaded: {N_TRAIN} train, {N_TEST} test", flush=True)

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

import kagglehub

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

# boltuzamaki parquet
try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    oof_df = pd.read_parquet(os.path.join(bolt, "oof_predictions.parquet"))
    tst_df = pd.read_parquet(os.path.join(bolt, "test_predictions.parquet"))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (
                oof_df[col].to_numpy(float), tst_df[col].to_numpy(float),
            )
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

# owned champion
try:
    members["own_champ_m10"] = (
        np.load("shared/analysis/data/own_champ_m10_oof.npy").astype(np.float64),
        np.load("shared/analysis/data/own_champ_m10_test.npy").astype(np.float64),
    )
    print("own_champ_m10: OWNED", flush=True)
except Exception as e:
    print(f"own_champ_m10 SKIP ({e})", flush=True)

# RealMLP
try:
    members["rmlp_realmlp"] = (
        np.load("shared/analysis/data/oof_realmlp.npy").astype(np.float64),
        np.load("shared/analysis/data/pred_realmlp.npy").astype(np.float64),
    )
    print("rmlp_realmlp: EXTERNAL", flush=True)
except Exception as e:
    print(f"rmlp_realmlp SKIP ({e})", flush=True)

# screen-time
for tag in ["catboost", "lgb3seed"]:
    try:
        members[f"fresh_{tag}"] = (
            np.load(f"shared/analysis/data/oof_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
            np.load(f"shared/analysis/data/test_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
        )
        print(f"fresh_{tag}: EXTERNAL", flush=True)
    except Exception as e:
        print(f"fresh_{tag} SKIP ({e})", flush=True)

# ---- rank + dedup ----
names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)
print(f"\n{len(names)} members, OOF AUC {auc.min():.5f} .. {auc.max():.5f}", flush=True)

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
print(f"{len(drop)} near-duplicate members dropped", flush=True)

SELF_REFERENTIAL = re.compile(r"^(naji|sz_naji|v13_anchor|hb_candidate)")
keep = [i for i, n in enumerate(names) if n not in drop and not SELF_REFERENTIAL.match(n)]
names = [names[i] for i in keep]
R, Rt = R[:, keep], Rt[:, keep]
print(f"{len(names)} members kept", flush=True)

# ---- rank-gauss transform ----
G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt, 1e-7, 1 - 1e-7)).astype(np.float32)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

# ============================================================
# BASELINE: linear logistic stack (cross-fitted)
# ============================================================
print("\n=== LINEAR LOGISTIC STACK ===", flush=True)

def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    return model.predict_proba(scaler.transform(X_pred))[:, 1]

oof_logistic = np.zeros(N_TRAIN)
for fi, vi in folds:
    oof_logistic[vi] = fit_logistic(G[fi], y[fi], G[vi])
logistic_auc = roc_auc_score(y, oof_logistic)
print(f"Linear logistic OOF: {logistic_auc:.6f}", flush=True)

# Full-data fit for test predictions
print("Fitting full-data logistic for test predictions...", flush=True)
test_logistic = fit_logistic(G, y, Gt)
print("Done.", flush=True)

# ============================================================
# NONLINEAR: LightGBM meta-learner (cross-fitted)
# ============================================================
print("\n=== LIGHTGBM META-STACK ===", flush=True)
lgb_oof_auc = 0
lgb_test = None
try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}", flush=True)
    oof_lgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/5...", end=" ", flush=True)
        dtrain = lgb.Dataset(G[fi], label=y[fi])
        dval = lgb.Dataset(G[vi], label=y[vi], reference=dtrain)
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 500, "feature_fraction": 0.5,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        }
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[vi] = model.predict(G[vi])
        print(f"AUC={roc_auc_score(y[vi], oof_lgb[vi]):.6f} iters={model.best_iteration}", flush=True)
    lgb_oof_auc = roc_auc_score(y, oof_lgb)
    print(f"LightGBM OOF: {lgb_oof_auc:.6f} (delta={lgb_oof_auc - logistic_auc:+.6f})", flush=True)
    # Full-data LightGBM for test
    print("Training full LightGBM for test...", flush=True)
    dtrain_full = lgb.Dataset(G, label=y)
    lgb_full = lgb.train(params, dtrain_full, num_boost_round=1000)
    lgb_test = lgb_full.predict(Gt)
    print("Done.", flush=True)
except Exception as e:
    import traceback
    print(f"LightGBM FAILED:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()

# ============================================================
# NONLINEAR: XGBoost meta-learner (cross-fitted)
# ============================================================
print("\n=== XGBOOST META-STACK ===", flush=True)
xgb_oof_auc = 0
xgb_test = None
try:
    import xgboost as xgb_lib
    print(f"XGBoost version: {xgb_lib.__version__}", flush=True)
    oof_xgb = np.zeros(N_TRAIN)
    for fold_idx, (fi, vi) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/5...", end=" ", flush=True)
        dtrain = xgb_lib.DMatrix(G[fi], label=y[fi])
        dval = xgb_lib.DMatrix(G[vi], label=y[vi])
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.03, "max_depth": 4,
            "min_child_weight": 500, "subsample": 0.8,
            "colsample_bytree": 0.5, "reg_alpha": 1.0,
            "reg_lambda": 1.0, "verbosity": 0, "seed": 42,
        }
        model = xgb_lib.train(
            params, dtrain, num_boost_round=2000,
            evals=[(dval, "val")], early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_xgb[vi] = model.predict(dval)
        print(f"AUC={roc_auc_score(y[vi], oof_xgb[vi]):.6f}", flush=True)
    xgb_oof_auc = roc_auc_score(y, oof_xgb)
    print(f"XGBoost OOF: {xgb_oof_auc:.6f} (delta={xgb_oof_auc - logistic_auc:+.6f})", flush=True)
    # Full-data XGBoost for test
    print("Training full XGBoost for test...", flush=True)
    dtrain_full = xgb_lib.DMatrix(G, label=y)
    xgb_full = xgb_lib.train(params, dtrain_full, num_boost_round=1000)
    xgb_test = xgb_full.predict(xgb_lib.DMatrix(Gt))
    print("Done.", flush=True)
except Exception as e:
    import traceback
    print(f"XGBoost FAILED:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}", flush=True)
print("META-ENSEMBLE COMPARISON", flush=True)
print(f"{'='*60}", flush=True)

results = [("Linear Logistic", logistic_auc, test_logistic)]
if lgb_oof_auc > 0:
    results.append(("LightGBM", lgb_oof_auc, lgb_test))
if xgb_oof_auc > 0:
    results.append(("XGBoost", xgb_oof_auc, xgb_test))

for name, a, _ in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {name:25s} OOF={a:.6f}  (delta={a - logistic_auc:+.6f})", flush=True)

best_name, best_auc, best_test = max(results, key=lambda x: x[1])
print(f"\nBest: {best_name} ({best_auc:.6f})", flush=True)

# ============================================================
# EMIT SUBMISSIONS
# ============================================================
# Always emit linear logistic submissions (safe fallback)
# Also emit the best nonlinear if it beats linear

try:
    naji = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
    cand = None
    for f in sorted(glob.glob(os.path.join(naji, "*.csv"))):
        df = pd.read_csv(f)
        if "id" in df.columns and "addicted_label" in df.columns and len(df) == N_TEST:
            preds = df["addicted_label"].to_numpy(float)
            if (preds >= 0).all() and (preds <= 1).all():
                cand = preds
                print(f"Public base: {os.path.basename(f)}", flush=True)
                break
    if cand is not None:
        public = pct_rank(cand)
    else:
        public = None
except Exception as e:
    print(f"No public base: {e}", flush=True)
    public = None

for W in [0.25, 0.30, 0.35, 0.40]:
    if public is not None:
        final = pct_rank(W * pct_rank(best_test) + (1 - W) * public)
    else:
        final = pct_rank(best_test)
    sub = pd.DataFrame({"id": test["id"], "addicted_label": final})
    assert len(sub) == N_TEST and sub["id"].is_unique
    fn = f"submission_nl_w{W:.2f}.csv"
    sub.to_csv(fn, index=False)
    print(f"  Emitted: {fn} (W={W:.2f}, best={best_name})", flush=True)

# Also emit pure stack and pure base for reference
pd.DataFrame({"id": test["id"], "addicted_label": pct_rank(best_test)}).to_csv(
    "submission_nl_w1.00.csv", index=False)
if public is not None:
    pd.DataFrame({"id": test["id"], "addicted_label": public}).to_csv(
        "submission_nl_w0.00.csv", index=False)

print(f"\n=== DONE ===", flush=True)
print(f"Best meta-learner: {best_name} (OOF={best_auc:.6f})", flush=True)
print(f"Linear logistic OOF: {logistic_auc:.6f}", flush=True)
if lgb_oof_auc > 0:
    print(f"LightGBM OOF: {lgb_oof_auc:.6f}", flush=True)
if xgb_oof_auc > 0:
    print(f"XGBoost OOF: {xgb_oof_auc:.6f}", flush=True)
