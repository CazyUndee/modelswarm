"""Shared member loader for S6E8 rank-gauss stacking experiments.

Centralizes all member loading logic so scripts don't each download
16 Kaggle libraries independently (~7 min overhead per script).

Usage:
    from shared.analysis.load_members import load_all_members, pct_rank, N_TRAIN, N_TEST

    members = load_all_members()
    names = sorted(members)
    OOF = np.column_stack([members[n][0] for n in names])
    TST = np.column_stack([members[n][1] for n in names])
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_TRAIN = 691369
N_TEST = 296302


def pct_rank(v):
    """Percentile rank transform: maps values to (0, 1) via fractional rank."""
    return (rankdata(v) - 0.5) / len(v)


def _load_vectors(oof_dir, prefix, test_prefixes=("test_", "testpred_", "tep_")):
    """Load OOF/test vector pairs from a dataset directory.

    Handles two naming conventions:
    - Convention A: oof_<name>.npy / test_<name>.npy (prefix)
    - Convention B: <name>_oof.npy / <name>_test.npy (suffix)
    """
    out = {}
    seen = set()
    # Convention A: oof_<name>.npy
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
    # Convention B: <name>_oof.npy / <name>_test.npy
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


# Dataset sources: (kaggle_dataset, prefix)
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


def load_all_members():
    """Load all member OOF/test vectors from all known sources.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: name -> (oof, test) vectors
    """
    import kagglehub

    members = {}

    # ---- Kaggle library sources ----
    for dataset, prefix in SOURCES:
        try:
            root = kagglehub.dataset_download(dataset)
            got = _load_vectors(root, prefix)
            members.update(got)
            print(f"{dataset:44s} {len(got):3d}", flush=True)
        except Exception as e:
            print(f"{dataset:44s} SKIP ({e})", flush=True)

    # ---- boltuzamaki parquet library ----
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

    # ---- szymonkapiski weak-50 models ----
    try:
        weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
        WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
        WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
        for j in range(WO.shape[1]):
            members[f"weak_{j:02d}"] = (
                np.asarray(WO[:, j], float), np.asarray(WT[:, j], float),
            )
        print(f"weak-50: {WO.shape[1]} cols", flush=True)
    except Exception as e:
        print(f"weak-50 SKIP ({e})", flush=True)

    # ---- owned champion vector ----
    try:
        members["own_champ_m10"] = (
            np.load("shared/analysis/data/own_champ_m10_oof.npy").astype(np.float64),
            np.load("shared/analysis/data/own_champ_m10_test.npy").astype(np.float64),
        )
        print("own_champ_m10: OWNED", flush=True)
    except Exception as e:
        print(f"own_champ_m10 SKIP ({e})", flush=True)

    # ---- RealMLP (post-RGS external) ----
    try:
        members["rmlp_realmlp"] = (
            np.load("shared/analysis/data/oof_realmlp.npy").astype(np.float64),
            np.load("shared/analysis/data/pred_realmlp.npy").astype(np.float64),
        )
        print("rmlp_realmlp: EXTERNAL", flush=True)
    except Exception as e:
        print(f"rmlp_realmlp SKIP ({e})", flush=True)

    # ---- fresh screen-time ratio features ----
    for tag in ["catboost", "lgb3seed"]:
        try:
            members[f"fresh_{tag}"] = (
                np.load(f"shared/analysis/data/oof_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
                np.load(f"shared/analysis/data/test_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
            )
            print(f"fresh_{tag}: EXTERNAL", flush=True)
        except Exception as e:
            print(f"fresh_{tag} SKIP ({e})", flush=True)

    return members


def dedup_members(members, corr_threshold=0.9995):
    """Rank-gauss transform, deduplicate near-duplicates, drop self-referential.

    Returns:
        (names, G, Gt, y, folds) where:
        - names: list of member names (kept)
        - G: rank-gauss transformed OOF matrix (N_TRAIN x len(names))
        - Gt: rank-gauss transformed test matrix (N_TEST x len(names))
        - y: labels
        - folds: list of (train_idx, val_idx) from StratifiedKFold(5)
    """
    from scipy.stats import norm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    tr = pd.read_csv("competitions/s6e8/data/train.csv")
    y = tr["addicted_label"].values

    names = sorted(members)
    OOF = np.column_stack([members[n][0] for n in names])
    TST = np.column_stack([members[n][1] for n in names])
    auc = pd.Series(
        [roc_auc_score(y, OOF[:, j]) for j in range(OOF.shape[1])],
        index=names,
    )
    print(f"\n{len(names)} members, OOF AUC {auc.min():.5f} .. {auc.max():.5f}", flush=True)

    R = np.column_stack([pct_rank(OOF[:, j]) for j in range(OOF.shape[1])]).astype(np.float32)
    Rt = np.column_stack([pct_rank(TST[:, j]) for j in range(TST.shape[1])]).astype(np.float32)

    # Dedup near-duplicates
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    corr = (Z.T @ Z) / len(Z)
    del Z
    drop = set()
    for i in range(len(names)):
        if names[i] in drop:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in drop or corr[i, j] <= corr_threshold:
                continue
            drop.add(names[j] if auc[names[i]] >= auc[names[j]] else names[i])
    print(f"{len(drop)} near-duplicate members dropped", flush=True)

    # Drop self-referential members
    SELF_REFERENTIAL = re.compile(r"^(naji|sz_naji|v13_anchor|hb_candidate)")
    keep = [i for i, n in enumerate(names)
            if n not in drop and not SELF_REFERENTIAL.match(n)]
    names = [names[i] for i in keep]
    R, Rt = R[:, keep], Rt[:, keep]
    print(f"{len(names)} members kept", flush=True)

    # Rank-gauss transform
    G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)
    Gt = norm.ppf(np.clip(Rt, 1e-7, 1 - 1e-7)).astype(np.float32)

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(N_TRAIN), y))

    return names, G, Gt, y, folds
