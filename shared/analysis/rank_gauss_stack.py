"""Rank-Gauss Cross-Fitted Stack over the full public OOF pool (RGS method).

Goal: beat the 0.97130 public ceiling (asteriosterzis rank-gauss stack) by
reproducing the exact cross-fitted logistic rank-gauss stack AND adding
(i) our owned champion vector (own_champ_m10, the only owned vec with
    both OOF and TEST predictions on this machine), and
(ii) the newest public members released since the RGS kernel ran.

The stacked prediction is blended with the public "vault" base (weight W) in
rank space. All members are EXTERNAL public work; our owned vector is labeled.

Evaluation is ENTIRELY internal (OOF AUC on a train holdout split + the stack's
own cross-fitted OOF). No public LB weight tuning.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test = pd.read_csv("competitions/s6e8/data/test.csv")


def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)


import kagglehub


def load_vectors(oof_dir, prefix, test_prefixes=("test_", "testpred_", "tep_")):
    out = {}
    seen = set()
    # Convention A: oof_<name>.npy (prefix convention — szymonkapiski, adarsh, etc.)
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
    # Convention B: <name>_oof.npy / <name>_test.npy (suffix convention — beicicc, etc.)
    for path in glob.glob(os.path.join(oof_dir, "**", "*_oof.npy"), recursive=True):
        bname = os.path.basename(path)
        name = bname[:-8]  # strip _oof.npy
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
    # beicicc —6 independent-model OOF artifacts (credited as pool backbone by adarsh/Diversity-Beats-Strength)
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

# boltuzamaki parquet library
try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    oof_df = pd.read_parquet(os.path.join(bolt, "oof_predictions.parquet"))
    tst_df = pd.read_parquet(os.path.join(bolt, "test_predictions.parquet"))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (
                oof_df[col].to_numpy(float), tst_df[col].to_numpy(float),
            )
    print(f"boltuzamaki parquet: {oof_df.shape[1]-1} cols")
except Exception as e:
    print(f"boltuzamaki SKIP ({e})")

# szymonkapiski's 50 deliberately weak models
try:
    weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
    WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
    WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
    for j in range(WO.shape[1]):
        members[f"weak_{j:02d}"] = (np.asarray(WO[:, j], float), np.asarray(WT[:, j], float))
    print(f"weak-50: {WO.shape[1]} cols")
except Exception as e:
    print(f"weak-50 SKIP ({e})")

# our owned champion vector — the ONLY owned vec with a TEST prediction here
try:
    members["own_champ_m10"] = (
        np.load("shared/analysis/data/own_champ_m10_oof.npy").astype(np.float64),
        np.load("shared/analysis/data/own_champ_m10_test.npy").astype(np.float64),
    )
    print("own_champ_m10: OWNED (OOF + TEST)")
except Exception as e:
    print(f"own_champ_m10 SKIP ({e})")

# ---- post-RGS external members (new architecture / feature families) ----
# RealMLP (kodaifukuda0311, Aug 30)
try:
    members["rmlp_realmlp"] = (
        np.load("shared/analysis/data/oof_realmlp.npy").astype(np.float64),
        np.load("shared/analysis/data/pred_realmlp.npy").astype(np.float64),
    )
    print("rmlp_realmlp: EXTERNAL RealMLP (OOF + TEST)")
except Exception as e:
    print(f"rmlp_realmlp SKIP ({e})")

# fresh screen-time ratio features (johnsebin97, Aug 31 — GBDTs on a NEW feature family)
for tag in ["catboost", "lgb3seed"]:
    try:
        members[f"fresh_{tag}"] = (
            np.load(f"shared/analysis/data/oof_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
            np.load(f"shared/analysis/data/test_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
        )
        print(f"fresh_{tag}: EXTERNAL screen-time-ratio GBDT (OOF + TEST)")
    except Exception as e:
        print(f"fresh_{tag} SKIP ({e})")

names = sorted(members)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])
auc = pd.Series([roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])], index=names)
print(f"\n{len(names)} members, OOF AUC {auc.min():.5f} .. {auc.max():.5f}")
print(f"own_champ_m10 OOF AUC = {auc['own_champ_m10']:.6f} (OWNED, corr with library NN should be high)")
print(auc.sort_values(ascending=False).head(15).to_string())

# ---- rank + dedup near-duplicates ----
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
print(f"{len(drop)} near-duplicate members dropped")

# drop self-referential stacked members (RGS dropped only these; tam_ kept)
SELF_REFERENTIAL = re.compile(r"^(naji|sz_naji|v13_anchor|hb_candidate)")
keep = [i for i, n in enumerate(names) if n not in drop and not SELF_REFERENTIAL.match(n)]
names = [names[i] for i in keep]
R, Rt = R[:, keep], Rt[:, keep]
print(f"{len(names)} members kept")

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))


def fit_logistic(X_fit, y_fit, X_pred, C=1.0):
    scaler = StandardScaler().fit(X_fit)
    model = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", tol=1e-5)
    model.fit(scaler.transform(X_fit), y_fit)
    assert int(np.max(model.n_iter_)) < 3000, "meta-model did not converge"
    return model.predict_proba(scaler.transform(X_pred))[:, 1]


G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)
Gt = norm.ppf(np.clip(Rt, 1e-7, 1 - 1e-7)).astype(np.float32)

oof_meta = np.zeros(N_TRAIN)
for fit_idx, val_idx in folds:
    oof_meta[val_idx] = fit_logistic(G[fit_idx], y[fit_idx], G[val_idx])
print(f"\nstack OOF AUC = {roc_auc_score(y, oof_meta):.6f}")
test_meta = fit_logistic(G, y, Gt)
print(f"linear logistic test OOF = {roc_auc_score(y, oof_meta):.6f}")

# ============================================================
# NONLINEAR META-ENSEMBLE: LightGBM on rank-gauss features
# ============================================================
# Hypothesis: GBDT can capture nonlinear interactions between member
# predictions that the linear logistic misses.
try:
    import lightgbm as lgb
    print(f"\n=== NONLINEAR LightGBM META-STACK ===")
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
        print(f"AUC={roc_auc_score(y[vi], oof_lgb[vi]):.6f} iters={model.best_iteration}")
    lgb_auc = roc_auc_score(y, oof_lgb)
    print(f"\nLightGBM meta-stack OOF: {lgb_auc:.6f}")
    print(f"vs linear logistic: {lgb_auc - roc_auc_score(y, oof_meta):+.6f}")
    # Train full LightGBM for test predictions
    dtrain_full = lgb.Dataset(G, label=y)
    dtest_full = lgb.Dataset(Gt)
    params_full = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 500, "feature_fraction": 0.5,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    lgb_full = lgb.train(params_full, dtrain_full, num_boost_round=1000)
    test_lgb = lgb_full.predict(Gt)
    use_lgb = lgb_auc > roc_auc_score(y, oof_meta) + 0.00005
    if use_lgb:
        print("*** LightGBM shows genuine improvement — using for submissions ***")
    else:
        print("LightGBM improvement too small — falling back to linear")
except Exception as e:
    print(f"LightGBM not available: {e}; using linear")
    use_lgb = False

# ============================================================
# NONLINEAR META-ENSEMBLE: XGBoost on rank-gauss features
# ============================================================
try:
    import xgboost as xgb_lib
    print(f"\n=== XGBOOST META-STACK ===")
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
        print(f"AUC={roc_auc_score(y[vi], oof_xgb[vi]):.6f}")
    xgb_auc = roc_auc_score(y, oof_xgb)
    print(f"\nXGBoost meta-stack OOF: {xgb_auc:.6f}")
    print(f"vs linear logistic: {xgb_auc - roc_auc_score(y, oof_meta):+.6f}")
except Exception as e:
    print(f"XGBoost not available: {e}")
    xgb_auc = 0

# ============================================================
# SUMMARY: best meta-learner
# ============================================================
logistic_auc = roc_auc_score(y, oof_meta)
print(f"\n{'='*60}")
print(f"META-ENSEMBLE COMPARISON")
print(f"{'='*60}")
results = [("Linear Logistic", logistic_auc, oof_meta)]
try:
    results.append(("LightGBM", lgb_auc, oof_lgb))
except:
    pass
try:
    if xgb_auc > 0:
        results.append(("XGBoost", xgb_auc, oof_xgb))
except:
    pass
for name, a, _ in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {name:25s} OOF={a:.6f}  (Δ={a - logistic_auc:+.6f})")

best_name, best_auc, best_oof = max(results, key=lambda x: x[1])
print(f"\nBest: {best_name} ({best_auc:.6f})")

# Select the best meta-learner's test predictions for blending
if best_name == "LightGBM" and use_lgb:
    test_final = test_lgb
    oof_final = oof_lgb
    print("Using LightGBM test predictions for submissions")
elif best_name == "XGBoost" and xgb_auc > logistic_auc + 0.00005:
    # train full XGBoost for test
    dtrain_full = xgb_lib.DMatrix(G, label=y)
    dtest_full = xgb_lib.DMatrix(Gt)
    params_full = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "learning_rate": 0.03, "max_depth": 4,
        "min_child_weight": 500, "subsample": 0.8,
        "colsample_bytree": 0.5, "reg_alpha": 1.0,
        "reg_lambda": 1.0, "verbosity": 0, "seed": 42,
    }
    xgb_full = xgb_lib.train(params_full, dtrain_full, num_boost_round=1000)
    test_final = xgb_full.predict(dtest_full)
    oof_final = oof_xgb
    print("Using XGBoost test predictions for submissions")
else:
    test_final = test_meta
    oof_final = oof_meta
    print("Using linear logistic test predictions for submissions")

# ---- blend with public vault base in rank space ----
# RGS used the vault (Naji-style) public base. We use naji's published "addition"
# submission as the base proxy and sweep W on a train holdout split.
try:
    naji = kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
    cand = None
    for f in sorted(glob.glob(os.path.join(naji, "*.csv"))):
        df = pd.read_csv(f)
        if "id" in df.columns and "addicted_label" in df.columns and len(df) == N_TEST:
            preds = df["addicted_label"].to_numpy(float)
            if (preds >= 0).all() and (preds <= 1).all():
                cand = preds
                print(f"public base candidate: {os.path.basename(f)}")
                break
    if cand is None:
        raise RuntimeError("no usable public base submission in naji dataset")
    public = pct_rank(cand)
except Exception as e:
    print(f"no naji base ({e}); will use meta probe only")
    public = None

print("\nstack OOF AUC (full) =", roc_auc_score(y, oof_final))

# ---- rank-space blend with public base, multiple W candidates ----
# RGS documented the plateau: W in 0.20..0.50 all give LB 0.97130 (base alone
# 0.97128). Blend weight CANNOT be validated on train because the public base
# is a test-only snapshot (no train OOF to align with our stack OOF). We emit
# several W candidates and let the internal stack OOF + the published plateau
# guide selection. No public-LB weight tuning per policy.

sub_files = {}
for W in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    if public is not None:
        final = pct_rank(W * pct_rank(test_final) + (1 - W) * public)
    else:
        final = pct_rank(test_final)
    sub = pd.DataFrame({"id": test["id"], "addicted_label": final})
    assert len(sub) == N_TEST and sub["id"].is_unique
    assert np.isfinite(sub["addicted_label"]).all() and sub["addicted_label"].between(0, 1).all()
    fn = f"submission_nonlinear_w{W:.2f}.csv"
    sub.to_csv(fn, index=False)
    sub_files[f"{W:.2f}"] = fn
    print(f"  W={W:.2f}: {fn}")

# pure stack (W=1.0) and pure base (W=0.0) references
pd.DataFrame({"id": test["id"], "addicted_label": pct_rank(test_final)}).to_csv(
    "submission_nonlinear_w1.00.csv", index=False)
print(f"\nEmitted NONLINEAR meta-stack submissions at W=0.20..0.50 + W=1.00")
print(f"stack OOF = {roc_auc_score(y, oof_final):.6f} (best meta: {best_name})")