"""Error Structure Analysis — understand where models disagree and why.

Analyzes:
1. Disagreement regions between NN and tree models
2. Error concentration by feature value bands
3. Residual correlation with features
4. Subgroup performance analysis
"""
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

np.random.seed(42)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# Load vectors
V = {}
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[n] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))

V["EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")

# Load tier OOFs
for name, path in [("XGB", "oof_XGB.csv"), ("HGB", "oof_HGB.csv")]:
    if os.path.exists(path):
        V[name] = pd.read_csv(path)["pred"].values

print("=" * 70)
print("ERROR STRUCTURE ANALYSIS")
print("=" * 70)

# ============================================================
# 1. DISAGREEMENT REGIONS
# ============================================================
print("\n--- 1. NN vs Tree disagreement regions ---")

# Binary predictions
nn_pred = (V["lookup"] >= 0.5).astype(int)
tree_pred = (V["EXP122"] >= 0.5).astype(int)
nn_err = (nn_pred != y)
tree_err = (tree_pred != y)

agree_correct = ~(nn_err | tree_err)
agree_wrong = nn_err & tree_err
nn_only_err = nn_err & ~tree_err
tree_only_err = ~nn_err & tree_err
disagree = nn_err | tree_err

print(f"Both correct: {np.sum(agree_correct):,} ({np.mean(agree_correct)*100:.1f}%)")
print(f"Both wrong:   {np.sum(agree_wrong):,} ({np.mean(agree_wrong)*100:.1f}%)")
print(f"NN-only err:  {np.sum(nn_only_err):,} ({np.mean(nn_only_err)*100:.1f}%)")
print(f"Tree-only err:{np.sum(tree_only_err):,} ({np.mean(tree_only_err)*100:.1f}%)")

# Where NN is wrong but tree is right — these are the rows NN could fix
print(f"\n--- NN error regions (where NN is wrong but tree is right) ---")
nn_fixable = nn_only_err
print(f"Rows: {np.sum(nn_fixable):,}")
if np.sum(nn_fixable) > 0:
    for feat in ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
                  "work_study_hours", "sleep_hours", "notifications_per_day",
                  "app_opens_per_day", "weekend_screen_time"]:
        vals = tr[feat].values
        mean_val = np.mean(vals[nn_fixable])
        overall_mean = np.mean(vals)
        print(f"  {feat}: mean in nn-error={mean_val:.2f} vs overall={overall_mean:.2f} (Δ {mean_val-overall_mean:+.2f})")

# ============================================================
# 2. CONFIDENCE BAND ANALYSIS
# ============================================================
print("\n--- 2. Confidence band analysis ---")

# Where models are most/least confident
for model_name in ["lookup", "EXP122", "naji03"]:
    pred = V[model_name]
    low_conf = (pred < 0.3) | (pred > 0.7)
    mid_conf = (pred >= 0.3) & (pred <= 0.7)
    high_conf = (pred < 0.15) | (pred > 0.85)

    for band, mask in [("low-conf [0.3,0.7]", mid_conf), ("high-conf [0,0.15]∪[0.85,1]", high_conf)]:
        if np.sum(mask) > 0:
            auc = roc_auc_score(y[mask], pred[mask])
            n = np.sum(mask)
            acc = np.mean((pred[mask] >= 0.5) == y[mask])
            print(f"  {model_name:12s} {band:35s}: n={n:>7,} AUC={auc:.4f} acc={acc:.4f}")

# ============================================================
# 3. ERROR CONCENTRATION BY SCREEN TIME BANDS
# ============================================================
print("\n--- 3. Error concentration by daily_screen_time_hours ---")

# The key feature — error is concentrated in the mid-band
for model_name in ["lookup", "EXP122", "naji03", "EXP130"]:
    pred = (V[model_name] >= 0.5).astype(int)
    err = (pred != y)
    for lo, hi, label in [(0, 3, "[0,3)"), (3, 5, "[3,5)"), (5, 7, "[5,7)"),
                           (7, 9, "[7,9)"), (9, 12, "[9,12)"), (12, 24, "[12,24]")]:
        mask = (tr["daily_screen_time_hours"].values >= lo) & (tr["daily_screen_time_hours"].values < hi)
        if np.sum(mask) > 100:
            band_err = np.mean(err[mask]) * 100
            band_n = np.sum(mask)
            print(f"  {model_name:12s} {label:8s}: n={band_n:>7,} error={band_err:.1f}%")

# ============================================================
# 4. FEATURE-RESIDUAL CORRELATIONS
# ============================================================
print("\n--- 4. Feature-residual correlations (where model is wrong) ---")

# For the best blend, compute residuals and correlate with features
best_blend = np.clip((V["lookup"] * 0.4 + V["tabm_seed3"] * 0.3 + V["EXP122"] * 0.1 + V["EXP130"] * 0.2), 1e-9, 1-1e-9)
residuals = y - best_blend

# Correlate residuals with features (signed error)
for feat in ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "notifications_per_day",
              "app_opens_per_day", "weekend_screen_time", "stress_level"]:
    vals = tr[feat].values
    # Handle categorical
    if tr[feat].dtype == object or tr[feat].nunique() < 20:
        continue
    corr, pval = spearmanr(vals, residuals)
    print(f"  {feat:30s}: corr={corr:+.4f} (p={pval:.2e})")

# ============================================================
# 5. SUBGROUP PERFORMANCE — where does NN help most?
# ============================================================
print("\n--- 5. Where does NN help most? (per screen-time band) ---")

for lo, hi, label in [(0, 3, "[0,3)"), (3, 5, "[3,5)"), (5, 7, "[5,7)"),
                       (7, 9, "[7,9)"), (9, 12, "[9,12)"), (12, 24, "[12,24]")]:
    mask = (tr["daily_screen_time_hours"].values >= lo) & (tr["daily_screen_time_hours"].values < hi)
    if np.sum(mask) < 100:
        continue
    y_band = y[mask]
    nn_auc = roc_auc_score(y_band, V["lookup"][mask])
    tree_auc = roc_auc_score(y_band, V["EXP122"][mask])
    frontier_auc = roc_auc_score(y_band, V["naji03"][mask])
    blend_auc = roc_auc_score(y_band, best_blend[mask])
    print(f"  {label:8s}: nn={nn_auc:.4f} tree={tree_auc:.4f} frontier={frontier_auc:.4f} blend={blend_auc:.4f} nn-tree={nn_auc-tree_auc:+.4f}")

# ============================================================
# 6. PREDICTION VALUE DISTRIBUTION BY ERROR TYPE
# ============================================================
print("\n--- 6. Prediction value distribution for error types ---")

# Where the blend is most uncertain vs most certain
blend_pred = best_blend
for model_name in ["lookup", "EXP122", "naji03"]:
    pred = V[model_name]
    err = ((pred >= 0.5) != y)

    # Mean prediction for correct vs wrong
    mean_correct = np.mean(pred[~err])
    mean_wrong = np.mean(pred[err])
    print(f"  {model_name:12s}: mean_pred(correct)={mean_correct:.4f} mean_pred(wrong)={mean_wrong:.4f} gap={abs(mean_correct-mean_wrong):.4f}")

print("\n--- DONE ---")
