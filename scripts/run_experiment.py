#!/usr/bin/env python3
"""
Experiment runner — reads experiment config, trains model(s), writes results.

This script is designed to run inside GitHub Actions.
It reads an experiment config YAML, trains the configured model or ensemble,
and writes results back.

Supported model configs:
    # Single model
    model:
      name: lightgbm            # lightgbm | xgboost | catboost
      parameters: {...}

    # Ensemble (blended OOF is the primary metric)
    model:
      name: ensemble
      blend: probability_average   # probability_average | rank_average
      members:
        - {name: lightgbm, parameters: {...}}
        - {name: xgboost,  parameters: {...}}
        - {name: catboost, parameters: {...}}

Usage:
    python scripts/run_experiment.py --config competitions/s6e8/experiments/EXP-008.yaml \
        --output-dir experiments/output/EXP-008/
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(competition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition data."""
    data_dir = Path(f"competitions/{competition}/data")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def apply_feature_engineering(df: pd.DataFrame, config: dict,
                              fit_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply feature engineering from config.

    fit_frame: reference frame for statistics (medians etc.). Pass TRAIN here
    when transforming test so imputation stats never come from the wrong split.

    Supported ops (structured list): ratio | product | sum | diff | indicator |
    impute_median | budget_constraint | pair_grid. Legacy dict format
    {feature_engineering: {ratios: [...]}} is still accepted.
    """
    df = df.copy()
    fe = config.get("feature_engineering", []) or []

    ops: list[dict] = []
    if isinstance(fe, list):
        ops = [op for op in fe if isinstance(op, dict)]
    elif isinstance(fe, dict):
        for ratio in fe.get("ratios", []):
            ops.append({"op": "ratio", **ratio})

    for op in ops:
        kind = op.get("op", "ratio")
        if kind == "impute_median":
            # Unsupervised median fill; medians come from fit_frame (TRAIN) when
            # provided so test rows are never filled with test-derived stats.
            ref = df if fit_frame is None else fit_frame
            for c in op.get("columns", []):
                if c in df.columns:
                    df[c] = df[c].fillna(ref[c].median())
            continue
        if kind == "budget_constraint":
            # Budget identity as a CONSTRAINT (raphdraft 0.96943-LB pipeline):
            # daily >= sum(components); missing components are bounded by
            # daily - observed_others. Produces bound/slack/violation views.
            daily_c = op["daily"]
            comps = list(op["components"])
            day_h = float(op.get("day_hours", 0.0))
            comp_raw = df[comps]
            observed = comp_raw.sum(axis=1, skipna=True)
            n_missing = comp_raw.isna().sum(axis=1)
            daily_raw = df[daily_c]
            prefix = op.get("prefix", "bud_")
            df[f"{prefix}implied_daily_lower"] = observed.astype("float64")
            df[f"{prefix}n_observed"] = (
                len(comps) + 1 - df[[daily_c] + comps].isna().sum(axis=1)).astype("int8")
            df[f"{prefix}n_missing"] = n_missing.astype("int8")
            slack = daily_raw - observed
            df[f"{prefix}slack"] = slack.astype("float64")
            df[f"{prefix}slack_per_missing"] = (
                slack / n_missing.replace(0, np.nan)).astype("float64")
            for c in comps:
                room = daily_raw - comp_raw.drop(columns=[c]).sum(axis=1, skipna=True)
                med = df[c].median()
                df[f"{prefix}viol_{c}"] = np.where(
                    df[c].isna(), room - med, 0.0).astype("float64")
            if day_h > 0:
                acc = df[comps].fillna(df[comps].median()).sum(axis=1)
                df[f"{prefix}unaccounted"] = (day_h - acc - df[daily_c].fillna(0)).astype("float64")
            continue
        name = op.get("name")
        if kind == "pair_grid":
            # generator op: expands to many named columns, no single name needed.
            # IMPORTANT: only meaningful on coarse-grid columns (0.01/0.1 quantized
            # screen-time cols). Continuous derived features create near-unique
            # levels -> self-referential TE -> collapse (EXP-126 incident).
            _forbidden = {"id", "addicted_label"} | set(op.get("exclude", []))
            if op.get("columns"):
                nums = [c for c in op["columns"] if c in df.columns]
            else:
                nums = [c for c in df.columns
                        if c not in _forbidden and pd.api.types.is_numeric_dtype(df[c])]
            res = float(op.get("resolution", 100))  # cells per unit; 100 = 0.01 grid
            limit = int(op.get("max_pairs", 40))
            made = 0
            done = False
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    if made >= limit:
                        done = True
                        break
                    a, b = nums[i], nums[j]
                    ca = (df[a] * res).round()
                    cb = (df[b] * res).round()
                    df[f"pair_{a}__{b}"] = (ca.fillna(-1) * int(1e7)
                                            + cb.fillna(-1)).astype("int64")
                    made += 1
                if done:
                    break
            print(f"  FE pair_grid: {made} pair columns")
            continue
        if kind == "digit_split":
            # donmarch14 quantisation family: values sit on a 0.01 grid, so the
            # FIRST DECIMAL DIGIT carries artefactual signal (positive-rate swing
            # 0.65->0.74 across digits). Hours are non-negative -> int-cast safe
            # (the floor(x*10)%10 trap only bites negatives, which cannot occur).
            for c in op.get("columns", []):
                if c not in df.columns:
                    continue
                x10 = (df[c] * 10)
                df[f"d1_{c}"] = np.where(df[c].isna(), np.nan,
                                         np.floor(x10) % 10).astype("float64")
                df[f"frac_{c}"] = (df[c] - np.floor(df[c].fillna(0))).where(
                    df[c].notna()).astype("float64")
            continue
        name = op.get("name")
        if not name:
            print(f"⚠️  Skipping FE op without name: {op}")
            continue
        try:
            if kind == "ratio":
                df[name] = df[op["numerator"]] / (df[op["denominator"]] + 1e-8)
            elif kind == "indicator":
                df[name] = df[op["column"]].isna().astype(int)
            elif kind in ("product", "sum", "diff"):
                terms = op["terms"]
                df[name] = df[terms[0]]
                for t in terms[1:]:
                    if kind == "product":
                        df[name] = df[name] * df[t]
                    elif kind == "sum":
                        df[name] = df[name] + df[t]
                    else:
                        df[name] = df[name] - df[t]
            else:
                print(f"⚠️  Unknown FE op '{kind}' — skipping")
        except KeyError as e:
            print(f"⚠️  FE op '{name}' references missing column {e} — skipping")

    return df


def _target_encode_fit(df_fit: pd.DataFrame, y_fit: pd.Series,
                       cols: list[str], smoothing: float) -> dict:
    """Fit leak-free target-encoding maps from (df_fit, y_fit) ONLY.

    Encoded value for group v: (sum_y[v] + m * prior) / (count[v] + m).
    NaN values form an explicit group (pandas>=3 groupby silently DROPS NaN
    keys by default -- handled manually here, see discussion 735861 trap).
    Unseen-at-apply values resolve to the prior (equivalent to count=0).
    """
    prior = float(pd.Series(y_fit).mean())
    m = float(smoothing)
    maps: dict[str, pd.Series] = {}
    freq: dict[str, pd.Series] = {}
    for c in cols:
        tmp = pd.DataFrame({"v": df_fit[c].to_numpy(), "y": np.asarray(y_fit)})
        g = tmp.groupby("v", dropna=False)["y"].agg(["sum", "count"])
        enc = (g["sum"] + m * prior) / (g["count"] + m)
        maps[c] = enc.astype("float64")
        freq[c] = g["count"].astype("float64")
    return {"prior": prior, "m": m, "maps": maps, "freq": freq}


def _target_encode_apply(df_apply: pd.DataFrame, stats: dict,
                         cols: list[str]) -> pd.DataFrame:
    """Add te_<col> columns using maps fitted on a disjoint frame."""
    out = df_apply.copy()
    prior = stats["prior"]
    freq = stats.get("freq", {})
    for c in cols:
        enc = stats["maps"][c]
        col = out[c]
        vals = col.map(enc)
        if col.isna().any():
            nan_group = enc.index[enc.index.isna()]
            nan_val = float(enc.iloc[enc.index.get_indexer([np.nan])[0]]) \
                if len(nan_group) else prior
            vals[col.isna()] = nan_val
        out[f"te_{c}"] = vals.fillna(prior).astype("float64")
        if c in freq:
            cnt = col.map(freq[c])
            out[f"freq_{c}"] = np.log1p(cnt.fillna(0.0)).astype("float64")
    return out


def _nested_target_encode_train(X_fit: pd.DataFrame, y_fit: pd.Series,
                                cols: list[str], smoothing: float,
                                n_inner: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Leak-free TE for TRAINING rows via inner-CV out-of-fold encodings.

    Each training row is encoded using maps fitted WITHOUT its own label
    (inner-split), so no row's target contributes to its own encoding.
    Returns (encoded frame, full-fit stats for transforming val/test).
    """
    from sklearn.model_selection import StratifiedKFold

    out = X_fit.copy()
    for c in cols:
        out[f"te_{c}"] = np.nan
        out[f"freq_{c}"] = np.nan
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    y_arr = np.asarray(y_fit)
    for tr_i, va_i in skf.split(out, y_arr):
        stats_i = _target_encode_fit(out.iloc[tr_i][cols], pd.Series(y_arr[tr_i]),
                                     cols, smoothing)
        # reuse apply logic minus freq duplication: compute manually
        enc_frames = _target_encode_apply(out.iloc[va_i][cols], stats_i, cols)
        for c in cols:
            out.iloc[va_i, out.columns.get_loc(f"te_{c}")] = enc_frames[f"te_{c}"].to_numpy()
            out.iloc[va_i, out.columns.get_loc(f"freq_{c}")] = enc_frames[f"freq_{c}"].to_numpy()
    # full-fit stats for validation/test transforms
    stats_full = _target_encode_fit(X_fit, y_fit, cols, smoothing)
    return out, stats_full


def _get_members(model_config: dict) -> list[dict]:
    """Normalize model config into a list of member specs.

    Single-model configs become a one-element list; ensemble configs
    (model.name == 'ensemble') return model.members unchanged.
    """
    if model_config.get("name") == "ensemble":
        members = model_config.get("members", [])
        if not members:
            raise ValueError("ensemble config requires model.members")
        return members
    return [{"name": model_config.get("name", "lightgbm"),
             "parameters": model_config.get("parameters", {})}]


def _blend(member_preds: dict[str, np.ndarray], method: str,
           weights: dict[str, float] | None = None) -> np.ndarray:
    """Blend member prediction vectors into one.

    weights may be keyed by member key ("lightgbm[0]") or base member name
    ("lightgbm"); name-level weights apply to every member of that family.
    """
    names = list(member_preds)
    arrays = [member_preds[n] for n in names]

    def _w(n: str) -> float:
        if not weights:
            return 1.0
        return float(weights.get(n, weights.get(n.split("[")[0], 1.0)))

    if len(arrays) == 1:
        return arrays[0]
    if method == "probability_average":
        if weights:
            w = np.array([_w(n) for n in names])
            w = w / w.sum()
            return np.average(arrays, axis=0, weights=w)
        return np.mean(arrays, axis=0)
    if method == "rank_average":
        ranks = [pd.Series(a).rank(pct=True).to_numpy() for a in arrays]
        if weights:
            w = np.array([_w(n) for n in names])
            w = w / w.sum()
            return np.average(ranks, axis=0, weights=w)
        return np.mean(ranks, axis=0)
    raise ValueError(f"Unknown blend method: {method}")


def _align_categoricals(df_fit: pd.DataFrame, df_apply: pd.DataFrame,
                        cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy frames and pin categorical vocabularies of df_apply to df_fit's."""
    fit, apply_ = df_fit.copy(), df_apply.copy()
    for c in cat_cols:
        fit[c] = fit[c].astype("category")
        apply_[c] = pd.Categorical(apply_[c], categories=fit[c].cat.categories)
    return fit, apply_


def _catboost_frame(df_target: pd.DataFrame, medians: dict[str, float],
                    codes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a NaN-free numeric frame for CatBoost from prepared columns."""
    out = df_target.copy()
    for c in out.columns:
        if c in codes:
            out[c] = codes[c].astype("float64")
            out.loc[out[c] < 0, c] = np.nan
        else:
            out[c] = out[c].fillna(medians[c])
    return out


def _fit_member_predict(model_name: str, params: dict,
                        X_train: pd.DataFrame, y_train: pd.Series,
                        X_pred: pd.DataFrame, cat_cols: list[str],
                        eval_set: tuple | None = None,
                        sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Fit one model member; return positive-class probabilities for X_pred.

    Categorical handling per family:
      - lightgbm: native category dtype + native NaN handling
      - xgboost:  category dtype with enable_categorical=True (hist), native NaN
      - catboost: ordinal codes (NaN → separate bucket via NaN code) +
                  numerics median-filled with TRAINING statistics only
      - logistic: median-imputed numerics + one-hot categoricals, standardized
    eval_set=(X_val, y_val) enables early stopping where supported.
    sample_weight flows to fit() where supported (all families).
    """
    if model_name == "lightgbm":
        import lightgbm as lgb
        X_tr, X_pr = _align_categoricals(X_train, X_pred, cat_cols)
        ctor = {k: v for k, v in params.items()
                if k not in ("eval_metric", "early_stopping_rounds",
                             "monotone_constraints")}
        # monotone_constraints: accept a name-keyed dict and align to column order
        mc = params.get("monotone_constraints")
        if isinstance(mc, dict):
            ctor["monotone_constraints"] = [int(mc.get(c, 0)) for c in X_tr.columns]
        elif mc is not None:
            ctor["monotone_constraints"] = mc
        model = lgb.LGBMClassifier(**ctor)
        es = params.get("early_stopping_rounds")
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        if es and eval_set is not None:
            X_ev = _align_categoricals(eval_set[0], X_train, cat_cols)[0]
            model.fit(X_tr, y_train, eval_set=[(X_ev, eval_set[1])],
                      callbacks=[lgb.early_stopping(es, verbose=False)], **fit_kwargs)
        else:
            model.fit(X_tr, y_train, callbacks=[], **fit_kwargs)
        return model.predict_proba(X_pr)[:, 1]

    if model_name == "xgboost":
        import xgboost as xgb
        X_tr, X_pr = _align_categoricals(X_train, X_pred, cat_cols)
        ctor = {k: v for k, v in params.items()
                if k not in ("eval_metric", "early_stopping_rounds")}
        ctor.setdefault("tree_method", "hist")
        ctor.setdefault("enable_categorical", True)
        ctor.setdefault("eval_metric", "auc")
        model = xgb.XGBClassifier(**ctor)
        es = params.get("early_stopping_rounds")
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        if es and eval_set is not None:
            X_ev = _align_categoricals(eval_set[0], X_train, cat_cols)[0]
            model.set_params(early_stopping_rounds=es)
            model.fit(X_tr, y_train, eval_set=[(X_ev, eval_set[1])], verbose=False, **fit_kwargs)
        else:
            model.fit(X_tr, y_train, verbose=False, **fit_kwargs)
        return model.predict_proba(X_pr)[:, 1]

    if model_name == "catboost":
        from catboost import CatBoostClassifier
        medians = {c: (X_train[c].median() if c not in cat_cols else 0.0)
                   for c in X_train.columns}
        codes_tr = {c: pd.Categorical(X_train[c]).codes for c in cat_cols}
        vocab = {c: pd.Categorical(X_train[c]).categories for c in cat_cols}
        X_tr = _catboost_frame(X_train, medians, codes_tr)

        def encode(df: pd.DataFrame) -> pd.DataFrame:
            codes = {c: pd.Categorical(df[c], categories=vocab[c]).codes for c in cat_cols}
            return _catboost_frame(df, medians, codes)

        ctor = {("thread_count" if k == "n_jobs" else k): v
                for k, v in params.items() if k != "eval_metric"}
        es = params.get("early_stopping_rounds")
        model = CatBoostClassifier(**ctor)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if es and eval_set is not None:
            model.fit(X_tr, y_train, eval_set=[(encode(eval_set[0]), eval_set[1])],
                      early_stopping_rounds=es, verbose=False, **fit_kwargs)
        else:
            model.fit(X_tr, y_train, verbose=False, **fit_kwargs)
        return model.predict_proba(encode(X_pred))[:, 1]

    if model_name == "tabm":
        # NN member via pytabkit. Preprocessing: median-impute numerics with
        # TRAIN-fold stats + explicit missing-indicator columns (NaN patterns
        # carry signal; pytabkit rejects NaN in continuous cols), native
        # categorical handling via category dtype.
        from pytabkit import TabM_D_Classifier
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        medians = {c: X_train[c].median() for c in num_cols}
        # Pin categorical vocabularies to fold-train so train/val/test frames
        # share identical dtypes and category sets.
        vocab = {c: pd.Categorical(X_train[c]).categories for c in cat_cols}

        def _tabm_frame(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            # Indicator columns are UNCONDITIONAL so every frame gets an
            # identical column set regardless of where missingness falls.
            for c in num_cols:
                out[f"{c}_imputed"] = out[c].isna().astype("float64")
                out[c] = out[c].fillna(medians[c])
            for c in cat_cols:
                out[c] = pd.Categorical(out[c], categories=vocab[c])
            return out

        ctor = {k: v for k, v in params.items()
                if k not in ("eval_metric", "early_stopping_rounds", "n_jobs",
                             "use_eval")}
        ctor.setdefault("device", "cpu")
        t_budget = ctor.pop("time_to_fit_in_seconds", None)
        model = TabM_D_Classifier(**ctor)
        if eval_set is not None and params.get("use_eval", True):
            model.fit(_tabm_frame(X_train), y_train,
                      X_val=_tabm_frame(eval_set[0]), y_val=eval_set[1],
                      time_to_fit_in_seconds=t_budget)
        else:
            model.fit(_tabm_frame(X_train), y_train,
                      time_to_fit_in_seconds=t_budget)
        return model.predict_proba(_tabm_frame(X_pred))[:, 1]

    if model_name == "logistic":
        # Linear member: median-imputed numerics (train-fold stats), one-hot
        # categoricals, standardized. Genuinely decorrelated from tree splits.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        num_cols = [c for c in X_train.columns if c not in cat_cols]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols),
        ])
        ctor = {"C": params.get("C", 1.0),
                "max_iter": params.get("max_iter", 1000),
                "random_state": params.get("random_state", 42)}
        model = make_pipeline(pre, LogisticRegression(**ctor))
        fit_kwargs = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        return model.predict_proba(X_pred)[:, 1]

    if model_name in ("mlp", "knn"):
        # Non-tree families for diversity screening: same preprocessing as
        # logistic (median-imputed scaled numerics + one-hot categoricals).
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        num_cols = [c for c in X_train.columns if c not in cat_cols]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols),
        ])
        if model_name == "mlp":
            from sklearn.neural_network import MLPClassifier
            ctor = {"hidden_layer_sizes": tuple(params.get("hidden_layer_sizes", (64, 32))),
                    "max_iter": params.get("max_iter", 200),
                    "random_state": params.get("random_state", 42),
                    "early_stopping": True}
            model = make_pipeline(pre, MLPClassifier(**ctor))
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["mlpclassifier__sample_weight"] = sample_weight
            model.fit(X_train, y_train, **fit_kwargs)
        else:  # knn
            from sklearn.neighbors import KNeighborsClassifier
            ctor = {"n_neighbors": params.get("n_neighbors", 25),
                    "weights": params.get("weights", "distance")}
            model = make_pipeline(pre, KNeighborsClassifier(**ctor))
            model.fit(X_train, y_train)
        return model.predict_proba(X_pred)[:, 1]

    raise ValueError(f"Unknown member model: {model_name}")


def train_model(train: pd.DataFrame, config: dict, test: pd.DataFrame | None = None) -> dict:
    """Train the configured model/ensemble with stratified CV. Returns results dict.

    If config.training.pseudo_label is set and `test` is provided, uses leak-free
    per-fold self-training: a base model fit ONLY on the fold's training rows
    selects confident test rows, then final models are fit on
    fold-train + weighted pseudo-labeled rows and produce the recorded OOF.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    model_config = config.get("model", {})
    members = _get_members(model_config)
    blend_method = model_config.get("blend", "probability_average")
    blend_weights = model_config.get("weights")

    features = config.get("features", [])
    target = "addicted_label"  # Should come from competition config

    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]
    if missing:
        print(f"⚠️  Missing features (skipping): {missing}")

    X = train[available].copy()
    y = train[target]
    cat_cols = [c for c in available if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"  Members: {[m['name'] for m in members]} | blend: {blend_method}")
    print(f"  Features: {len(available)} ({len(cat_cols)} categorical)")

    pl_cfg = config.get("training", {}).get("pseudo_label")
    use_pseudo = bool(pl_cfg) and test is not None
    if use_pseudo:
        X_test_al = test[available].copy()
        for c in cat_cols:
            X_test_al[c] = X_test_al[c].astype(X[c].dtype) if X[c].dtype.name != 'category' else X[c].dtype
        print(f"  Pseudo-labeling ENABLED: low={pl_cfg.get('low')} high={pl_cfg.get('high')} "
              f"weight={pl_cfg.get('weight', 0.5)}")

    val_config = config.get("validation", {})
    skf = StratifiedKFold(n_splits=val_config.get("n_folds", 5),
                          shuffle=True, random_state=val_config.get("random_state", 42))

    te_cfg = config.get("target_encoding") or {}
    _te_requested = te_cfg.get("columns", [])
    import fnmatch
    te_cols = []
    for spec in _te_requested:
        if any(ch in spec for ch in "*?[]"):
            te_cols.extend(c for c in X.columns if fnmatch.fnmatch(c, spec))
        elif spec in X.columns:
            te_cols.append(spec)
    te_cols = list(dict.fromkeys(te_cols))
    te_smoothing = float(te_cfg.get("smoothing", 50))
    te_nested = bool(te_cfg.get("nested", False))
    if te_nested:
        print("  TE mode: NESTED (inner-CV out-of-fold encodings for train rows)")
    if te_cols:
        print(f"  Target encoding: {len(te_cols)} cols, smoothing={te_smoothing} "
              f"(fold-train fit only)")
    missing_te = [c for c in te_cfg.get("columns", []) if c not in X.columns]
    if missing_te:
        print(f"⚠️  TE columns not found (skipping): {missing_te}")

    member_keys = [f"{m['name']}[{i}]" for i, m in enumerate(members)]
    oof_by_member = {k: np.zeros(len(X)) for k in member_keys}
    member_fold_aucs = {k: [] for k in member_keys}
    blend_fold_aucs = []
    pseudo_counts = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- Leak-free target encoding (fit on fold-train ONLY) ---
        if te_cols:
            te_stats = _target_encode_fit(X_train, y_train, te_cols, te_smoothing)
            if te_nested:
                X_train, te_stats = _nested_target_encode_train(
                    X_train, y_train, te_cols, te_smoothing,
                    int(te_cfg.get("inner_folds", 5)), int(te_cfg.get("inner_seed", 42)))
            else:
                X_train = _target_encode_apply(X_train, te_stats, te_cols)
            X_val = _target_encode_apply(X_val, te_stats, te_cols)
            if use_pseudo:
                X_test_al = _target_encode_apply(X_test_al, te_stats, te_cols)

        # --- Optional pseudo-label generation (base models see fold-train ONLY) ---
        sw_map = None
        if use_pseudo:
            base_preds = {}
            for key, m in zip(member_keys, members):
                base_preds[key] = _fit_member_predict(
                    m["name"], m.get("parameters", {}), X_train, y_train,
                    X_test_al, cat_cols)
            base_avg = _blend(base_preds, blend_method, blend_weights)
            lo, hi = pl_cfg.get("low", 0.02), pl_cfg.get("high", 0.98)
            w = float(pl_cfg.get("weight", 0.5))
            conf_mask = (base_avg <= lo) | (base_avg >= hi)
            n_conf = int(conf_mask.sum())
            pseudo_counts.append(n_conf)
            if n_conf > 0:
                X_conf = X_test_al.loc[conf_mask].copy()
                y_conf = pd.Series((base_avg[conf_mask] >= hi).astype(int),
                                   index=X_conf.index)
                X_aug = pd.concat([X_train, X_conf], ignore_index=True)
                y_aug = pd.concat([y_train, y_conf], ignore_index=True)
                sw_map = np.concatenate([np.ones(len(X_train)), np.full(n_conf, w)])
                print(f"    fold {fold}: {n_conf} pseudo rows "
                      f"({n_conf / len(X_test_al):.1%} of test)")
            else:
                X_aug, y_aug = X_train, y_train
        else:
            X_aug, y_aug = X_train, y_train

        fold_preds = {}
        for key, m in zip(member_keys, members):
            preds = _fit_member_predict(m["name"], m.get("parameters", {}),
                                        X_aug, y_aug, X_val, cat_cols,
                                        eval_set=(X_val, y_val) if not use_pseudo else None,
                                        sample_weight=sw_map)
            oof_by_member[key][val_idx] = preds
            fold_preds[key] = preds
            auc = roc_auc_score(y_val, preds)
            member_fold_aucs[key].append(float(auc))

        blended = _blend(fold_preds, blend_method, blend_weights)
        blend_fold_aucs.append(float(roc_auc_score(y_val, blended)))

        fold_str = " ".join(f"{n}={member_fold_aucs[n][-1]:.5f}" for n in member_keys)
        print(f"  Fold {fold}: {fold_str} | blend={blend_fold_aucs[-1]:.5f}")

    if use_pseudo:
        print(f"  Pseudo rows/fold: {pseudo_counts}")

    member_oof_aucs = {}
    for k in member_keys:
        auc = roc_auc_score(y, oof_by_member[k])
        member_oof_aucs[k] = float(auc)
        print(f"  OOF {k}: {auc:.5f}")

    correlations = {
        f"{a}~{b}": float(np.corrcoef(oof_by_member[a], oof_by_member[b])[0, 1])
        for a, b in combinations(member_keys, 2)
    }
    for pair, r in correlations.items():
        print(f"  OOF corr {pair}: {r:.4f}")

    blended_oof = _blend(oof_by_member, blend_method, blend_weights)
    oof_auc = roc_auc_score(y, blended_oof)
    rank_diag = None
    if len(member_keys) > 1 and blend_method != "rank_average":
        rank_oof = _blend(oof_by_member, "rank_average", blend_weights)
        rank_diag = float(roc_auc_score(y, rank_oof))
        print(f"  OOF blend({blend_method}): {oof_auc:.5f} | rank_average diagnostic: {rank_diag:.5f}")
    else:
        print(f"  OOF blend({blend_method}): {oof_auc:.5f}")

    return {
        "oof_metric": float(oof_auc),
        "fold_metrics": blend_fold_aucs,
        "oof_predictions": blended_oof.tolist(),
        "member_oofs": {k: oof_by_member[k].tolist() for k in member_keys},
        "features_used": available,
        "categorical_features": cat_cols,
        "model_name": ("ensemble:" + "+".join(m["name"] for m in members)) if len(member_keys) > 1 else members[0]["name"],
        "blend_method": blend_method,
        "members": [{"member_index": i, "name": m["name"], "key": k,
                     "oof_auc": member_oof_aucs[k],
                     "fold_metrics": member_fold_aucs[k]}
                    for i, (k, m) in enumerate(zip(member_keys, members))],
        "member_correlations": correlations,
        "rank_average_diagnostic": rank_diag,
    }


def predict_test(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> np.ndarray:
    """Fit every member on the full dataset and predict the test set (submission).

    Mirrors the CV pseudo-label procedure when configured: base full-data fit
    selects confident test rows, final fit is on train + weighted pseudo rows.
    """
    model_config = config.get("model", {})
    members = _get_members(model_config)
    blend_method = model_config.get("blend", "probability_average")
    blend_weights = model_config.get("weights")

    target = "addicted_label"
    features = [f for f in config.get("features", []) if f in train.columns]
    X, y = train[features].copy(), train[target]
    X_test = test[features].copy()

    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    X, X_test = _align_categoricals(X, X_test, cat_cols)

    # Target encoding: fit on FULL train, apply to train + test (submission path
    # has no holdout, mirroring the fold-level fit-on-train-only discipline).
    te_cfg = config.get("target_encoding") or {}
    import fnmatch as _fnm
    _te_requested = te_cfg.get("columns", [])
    te_cols = []
    for spec in _te_requested:
        if any(ch in spec for ch in "*?[]"):
            te_cols.extend(c for c in X.columns if _fnm.fnmatch(c, spec))
        elif spec in X.columns:
            te_cols.append(spec)
    te_cols = list(dict.fromkeys(te_cols))
    if te_cols:
        if te_cfg.get("nested", False):
            # submission models train on FULL data: their own training rows get
            # inner-OOF encodings (leak-free), test rows get full-fit encodings.
            X, _ = _nested_target_encode_train(
                X, y, te_cols, float(te_cfg.get("smoothing", 50)),
                int(te_cfg.get("inner_folds", 5)), int(te_cfg.get("inner_seed", 42)))
        te_stats = _target_encode_fit(X, y, te_cols, float(te_cfg.get("smoothing", 50)))
        X_test = _target_encode_apply(X_test, te_stats, te_cols)
        if not te_cfg.get("nested", False):
            X = _target_encode_apply(X, te_stats, te_cols)
        print(f"  Target encoding (submission): {len(te_cols)} cols")

    pl_cfg = config.get("training", {}).get("pseudo_label")

    if pl_cfg:
        base_preds = {}
        for key, m in zip((f"{m['name']}[{i}]" for i, m in enumerate(members)), members):
            base_preds[key] = _fit_member_predict(
                m["name"], m.get("parameters", {}), X, y, X_test, cat_cols)
        base_avg = _blend(base_preds, blend_method, blend_weights)
        lo, hi = pl_cfg.get("low", 0.02), pl_cfg.get("high", 0.98)
        w = float(pl_cfg.get("weight", 0.5))
        conf_mask = (base_avg <= lo) | (base_avg >= hi)
        n_conf = int(conf_mask.sum())
        print(f"  pseudo-labeling submission: {n_conf} confident test rows")
        if n_conf > 0:
            X_conf = X_test.loc[conf_mask].copy()
            y_conf = pd.Series((base_avg[conf_mask] >= hi).astype(int), index=X_conf.index)
            X_aug = pd.concat([X, X_conf], ignore_index=True)
            y_aug = pd.concat([y, y_conf], ignore_index=True)
            sw = np.concatenate([np.ones(len(X)), np.full(n_conf, w)])
            final_preds = {}
            for key, m in zip((f"{m['name']}[{i}]" for i, m in enumerate(members)), members):
                # Predict the SAME test rows the model was partially trained on;
                # standard self-training practice — recorded transparently.
                final_preds[key] = _fit_member_predict(
                    m["name"], m.get("parameters", {}), X_aug, y_aug,
                    X_test, cat_cols, sample_weight=sw)
            return _blend(final_preds, blend_method, blend_weights)

    member_preds = {}
    for key, m in zip((f"{m['name']}[{i}]" for i, m in enumerate(members)), members):
        member_preds[key] = _fit_member_predict(
            m["name"], m.get("parameters", {}), X, y, X_test, cat_cols)
        print(f"  test predictions ready: {key}")

    return _blend(member_preds, blend_method, blend_weights)


def main():
    # Windows consoles may default to cp1252 and crash on emoji markers.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Run a ModelSwarm experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Smoke-test only: cap training rows (results NOT valid for research)")
    parser.add_argument("--no-submission", action="store_true",
                        help="Skip full-data test prediction / submission.csv output")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_id = config.get("experiment_id", "unknown")
    competition = config.get("competition", "s6e8")

    print(f"{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"Competition: {competition}")
    print(f"Hypothesis: {config.get('hypothesis', 'N/A')}")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading data...")
    try:
        train, test = load_data(competition)
        print(f"  Train: {train.shape}")
        print(f"  Test: {test.shape}")
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("Data must be committed under competitions/<id>/data/ — GitHub Actions has no other source.")
        sys.exit(1)

    if args.max_rows:
        print(f"⚠️  SMOKE TEST MODE: capping train to {args.max_rows} rows (invalid for research)")
        train = train.sample(n=min(args.max_rows, len(train)), random_state=42).reset_index(drop=True)

    # Screening mode: reduced rows for cheap relative-ranking runs.
    # Screening OOF scores are NOT comparable to full-data results.
    screening = None
    scr = config.get("training", {}).get("screening")
    if isinstance(scr, dict) and float(scr.get("row_fraction", 1.0)) < 1.0:
        frac = float(scr["row_fraction"])
        seed = int(scr.get("seed", 42))
        n_folds_eff = int(config.get("validation", {}).get("n_folds", 5))
        train = train.sample(frac=frac, random_state=seed).reset_index(drop=True)
        screening = {"row_fraction": frac, "seed": seed, "rows": int(len(train))}
        print(f"⚠️  SCREENING MODE: row_fraction={frac} (seed {seed}) -> {train.shape[0]} rows, "
              f"{n_folds_eff} folds. Scores are for RELATIVE ranking ONLY — "
              "not comparable to full-data OOF results.")

    print("\n[2/5] Applying feature engineering...")
    train = apply_feature_engineering(train, config)
    test = apply_feature_engineering(test, config, fit_frame=train)
    print(f"  Features after engineering: {train.shape[1]}")

    # Generator FE ops (pair_grid etc.) create columns not listed in
    # config['features'] — register them so train_model actually uses them.
    _known = set(config.get("features", [])) | {"id", "addicted_label"}
    generated = [c for c in train.columns if c not in _known]
    if generated:
        config["features"] = list(config.get("features", [])) + generated
        test = test.reindex(columns=config["features"] + ["id"], fill_value=np.nan) \
            if "id" in test.columns else test.reindex(columns=config["features"], fill_value=np.nan)
        print(f"  Registered {len(generated)} generated features: {generated[:6]}...")

    print("\n[3/5] Training (stratified CV)...")
    start_time = time.time()
    pl_enabled = bool(config.get("training", {}).get("pseudo_label"))
    results = train_model(train, config, test=test if pl_enabled else None)
    runtime = time.time() - start_time
    results["runtime_seconds"] = runtime
    if screening:
        results["screening"] = screening

    print("\n[4/5] Saving results...")
    oof_df = pd.DataFrame({
        "id": train["id"],
        "target": train["addicted_label"],
        "prediction": results["oof_predictions"],
    })
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF predictions: {oof_path}")

    member_oofs = results.pop("member_oofs", None)
    if member_oofs:
        for key, vec in member_oofs.items():
            m_path = output_dir / f"oof_member_{key}.csv"
            pd.DataFrame({"id": train["id"], "target": train["addicted_label"],
                          "prediction": vec}).to_csv(m_path, index=False)
        print(f"  Member OOFs: {len(member_oofs)} files ({', '.join(member_oofs)})")

    if not args.no_submission and config.get("training", {}).get("predict_test", True):
        print("  Predicting test set on full data...")
        test_preds = predict_test(train, test, config)
        submission = pd.DataFrame({"id": test["id"], "addicted_label": test_preds})
        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"  Submission: {sub_path}")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k not in ("oof_predictions", "member_oofs")}, f, indent=2)
    print(f"  Results: {results_path}")

    print(f"\n{'='*60}")
    smoke_tag = " [SMOKE TEST]" if args.max_rows else ""
    print(f"✅ Experiment {experiment_id} complete{smoke_tag}")
    print(f"   OOF AUC: {results['oof_metric']:.5f}")
    print(f"   Blend folds: {[f'{m:.5f}' for m in results['fold_metrics']]}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"{'='*60}")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"oof_metric={results['oof_metric']}\n")
            f.write(f"runtime={runtime}\n")


if __name__ == "__main__":
    main()
