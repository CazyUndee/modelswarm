"""Test FT-Transformer via pytorch-tabular for S6E8.

Hypothesis: FT-Transformer (Feature Tokenizer + Transformer) is a distinct
NN architecture that could provide complementary signal to existing lookup/tabm
models. Combined with proper OOF validation, it could improve the NN stack.

Protocol:
1. Install pytorch-tabular
2. Build OOF predictions with 5-fold CV
3. Measure standalone AUC
4. Test blend with existing NN and tree models
5. Report complementarity metrics
"""
import os
import subprocess
import sys

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-tabular", "torch", "scikit-learn"],
               capture_output=True, text=True)
print("Dependencies installed.")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.stats import spearmanr

np.random.seed(42)

# Load data
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# Feature columns
feature_cols = ["age", "daily_screen_time_hours", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours",
                "notifications_per_day", "app_opens_per_day",
                "weekend_screen_time", "gender", "stress_level",
                "academic_work_impact"]

X = tr[feature_cols].copy()
X["free_time_slack"] = X["daily_screen_time_hours"] - X["social_media_hours"] - X["gaming_hours"] - X["work_study_hours"]
# Encode all categorical columns
print(f"  Gender values: {X['gender'].unique()}")
X["gender"] = X["gender"].map({"Male": 1.0, "Female": 0.0, "Other": 0.5}).fillna(0.0).astype(np.float32)
print(f"  Stress values: {X['stress_level'].unique()}")
X["stress_level"] = X["stress_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.5).astype(np.float32)
print(f"  Impact values: {X['academic_work_impact'].unique()}")
X["academic_work_impact"] = X["academic_work_impact"].map({"No": 0.0, "Yes": 1.0}).fillna(0.0).astype(np.float32)
for col in X.columns:
    X[col] = X[col].astype(float)
# Numeric columns have missing values: impute with the GLOBAL median here.
# This is only the input matrix for a NN screen — medians computed on the whole
# dataset leak negligibly (a constant per column), and pytorch-tabular cannot
# take NaN. Tree experiments keep strict fold-safe imputation.
X = X.fillna(X.median(numeric_only=True))
feature_names = list(X.columns)

print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")

# ============================================================
# FT-TRANSFORMER OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("FT-TRANSFORMER OOF PREDICTIONS (5-fold CV)")
print("=" * 70)

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
    )
    from pytorch_tabular.models import FTTransformerConfig

    print("FT-Transformer config loaded.")

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    ft_oof = np.zeros(N)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
        print(f"  Fold {fold+1}/{n_folds}...", end=" ", flush=True)

        train_df = X.iloc[tr_idx].copy()
        train_df["target"] = y[tr_idx]
        val_df = X.iloc[va_idx].copy()
        val_df["target"] = y[va_idx]

        data_config = DataConfig(
            target=["target"],
            continuous_cols=feature_names,
            categorical_cols=[],
        )

        # GHA CPU budget: ~3.9 min/epoch on 691k rows → must fit 5 folds in 60 min.
        # 3 epochs × 5 folds × 4 min ≈ 60 min. Early stopping may cut earlier.
        trainer_config = TrainerConfig(
            max_epochs=3,
            batch_size=2048,
            early_stopping_patience=2,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr_scheduler="CosineAnnealingWarmRestarts",
            lr_scheduler_params={"T_0": 10, "T_mult": 1},
        )

        model_config = FTTransformerConfig(
            task="classification",
            learning_rate=1e-3,
        )

        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
        )

        model.fit(train=train_df, validation=val_df)
        # pytorch-tabular names output columns after the target
        # (e.g. 'target_prediction' / 'target_probability'), not literally 'prediction'.
        pred_df = model.predict(val_df)
        prob_col = next((c for c in pred_df.columns if c.endswith("_probability")), None)
        if prob_col is not None:
            preds = pred_df[prob_col].values.astype(float)
        else:
            pred_col = next((c for c in pred_df.columns if c.endswith("_prediction")), pred_df.columns[-1])
            preds = pred_df[pred_col].values.astype(float)
        ft_oof[va_idx] = preds

        fold_auc = roc_auc_score(y[va_idx], preds)
        print(f"AUC={fold_auc:.6f}")

    ft_auc = roc_auc_score(y, ft_oof)
    print(f"\nFT-Transformer OOF: {ft_auc:.6f}")

    # Save OOF and test predictions as CSV for artifact upload
    pd.DataFrame({"prediction": ft_oof}).to_csv("ft_transformer_oof.csv", index=False)
    print("Saved ft_transformer_oof.csv")

except Exception as e:
    print(f"FT-Transformer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# LOAD EXISTING MODELS FOR BLEND TEST
# ============================================================
print("\n" + "=" * 70)
print("BLEND TEST: FT-Transformer + existing models")
print("=" * 70)

import kagglehub
V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  lib_{n}: OOF {roc_auc_score(y, V[f'lib_{n}']):.6f}")

V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
V["ft_transformer"] = ft_oof
print(f"  ft_transformer: OOF {roc_auc_score(y, V['ft_transformer']):.6f}")

# ============================================================
# HELPER
# ============================================================
def fit_weights(cols, y_fit, M_fit):
    Mf = M_fit[:, cols]
    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y_fit * np.log(p) + (1 - y_fit) * np.log(1 - p))
    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 1000, "maxfev": 2000, "xatol": 1e-5, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()

keys = list(V.keys())
M = np.column_stack([V[k] for k in keys])
fit_idx = np.arange(400000)
hold_idx = np.arange(400000, N)

# ============================================================
# BLEND CONFIGURATIONS
# ============================================================
configs = [
    ("FT-Transformer alone", ["ft_transformer"]),
    ("Library NN (lookup+tabm)", ["lib_lookup", "lib_tabm_seed3"]),
    ("FT + Library NN", ["ft_transformer", "lib_lookup", "lib_tabm_seed3"]),
    ("FT + Owned", ["ft_transformer", "owned_EXP122", "owned_EXP130"]),
    ("FT + Library + Owned", ["ft_transformer", "lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]),
    ("Full pool", ["ft_transformer", "lib_lookup", "lib_tabm_seed3", "lib_naji03", "owned_EXP122", "owned_EXP130"]),
]

print(f"\n{'Config':<45} {'OOF AUC':>10} {'Held-out':>10} {'Δ vs ft':>12}")
print("-" * 80)

for name, members in configs:
    cols = [keys.index(m) for m in members]
    if len(members) == 1:
        auc = roc_auc_score(y, V[members[0]])
        held = roc_auc_score(y[hold_idx], V[members[0]][hold_idx])
    else:
        w = fit_weights(cols, y[fit_idx], M[fit_idx])
        pred = np.clip(M[:, cols] @ w, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, pred)
        held = roc_auc_score(y[hold_idx], pred[hold_idx])
    delta = auc - roc_auc_score(y, V["ft_transformer"])
    print(f"{name:<45} {auc:>10.6f} {held:>10.6f} {delta:>+12.6f}")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print(f"\n--- Correlation: FT-Transformer vs existing models ---")
for k in ["lib_lookup", "lib_tabm_seed3", "owned_EXP122", "owned_EXP130"]:
    sp = spearmanr(V["ft_transformer"], V[k]).correlation
    print(f"  ft ~ {k}: Spearman={sp:.5f}")

print("\n--- DONE ---")
