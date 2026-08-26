"""DL Model Investigation — analyze Tamerlan's DL model predictions
to infer architecture and understand their contribution.

The dl_s7 and dl_s23 models are unpublished deep learning models.
By analyzing their prediction patterns, correlations, and error structure,
we can infer what type of architecture they likely are.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
import scipy.special

np.random.seed(42)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)
screen = tr["daily_screen_time_hours"].values.astype(float)

# Load all Tamerlan models
tamerlan_dir = "/tmp/blend_npy/blend_data"
V = {}
for name in ["lookup_transformer", "catboost", "dl_s23", "dl_s7", "mlp",
             "xgb_te", "lgb_cat", "lightgbm", "lgb_te_a1"]:
    path = os.path.join(tamerlan_dir, f"oof_{name}.npy")
    if os.path.exists(path):
        raw = np.load(path)
        # Sigmoid-transform logits to probabilities
        if raw.min() < 0:
            V[name] = scipy.special.expit(raw)
        else:
            V[name] = raw

# Load our models
V["EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")

# ============================================================
# ANALYSIS
# ============================================================
print("=" * 70)
print("DL MODEL INVESTIGATION")
print("=" * 70)

# 1. Individual AUCs
print("\n--- Individual Model AUCs ---")
for name in sorted(V.keys(), key=lambda x: roc_auc_score(y, V[x]), reverse=True):
    auc = roc_auc_score(y, V[name])
    raw_range = f"min={V[name].min():.4f} max={V[name].max():.4f}"
    print(f"  {name:25s}: AUC={auc:.6f}  {raw_range}")

# 2. Pairwise correlations
print("\n--- Pairwise Spearman Correlations ---")
focus = ["dl_s7", "dl_s23", "mlp", "lookup_transformer", "EXP122", "EXP130"]
focus = [k for k in focus if k in V]
print(f"\n{'Model':<25}", end="")
for k in focus:
    print(f" {k:>12}", end="")
print()
print("-" * (25 + 13 * len(focus)))
for i, ki in enumerate(focus):
    print(f"{ki:<25}", end="")
    for j, kj in enumerate(focus):
        if i == j:
            print(f" {'---':>12}", end="")
        else:
            sp = spearmanr(V[ki], V[kj]).correlation
            print(f" {sp:>12.5f}", end="")
    print()

# 3. DL models: are they the same architecture with different seeds?
print("\n--- DL Model Comparison ---")
dl7 = V["dl_s7"]
dl23 = V["dl_s23"]
sp = spearmanr(dl7, dl23).correlation
pe = pearsonr(dl7, dl23)[0]
diff = np.abs(dl7 - dl23)
print(f"  dl_s7 vs dl_s23:")
print(f"    Spearman: {sp:.5f}")
print(f"    Pearson:  {pe:.5f}")
print(f"    Mean |diff|: {np.mean(diff):.6f}")
print(f"    Max |diff|: {np.max(diff):.6f}")
print(f"    dl_s7 AUC: {roc_auc_score(y, dl7):.6f}")
print(f"    dl_s23 AUC: {roc_auc_score(y, dl23):.6f}")

# 4. DL vs MLP: are they different architectures?
print(f"\n  dl_s7 vs mlp:")
sp = spearmanr(dl7, V["mlp"]).correlation
print(f"    Spearman: {sp:.5f}")
print(f"    Mean |diff|: {np.mean(np.abs(dl7 - V['mlp'])):.6f}")

# 5. DL vs lookup_transformer
print(f"\n  dl_s7 vs lookup_transformer:")
sp = spearmanr(dl7, V["lookup_transformer"]).correlation
print(f"    Spearman: {sp:.5f}")

# 6. Error analysis: where do DL models disagree with trees?
print("\n--- Error Disagreement Analysis ---")
dl_pred = (V["dl_s7"] >= 0.5).astype(int)
tree_pred = (V["EXP122"] >= 0.5).astype(int)
dl_err = (dl_pred != y)
tree_err = (tree_pred != y)

agree_correct = ~(dl_err | tree_err)
agree_wrong = dl_err & tree_err
dl_only = dl_err & ~tree_err
tree_only = ~dl_err & tree_err

print(f"  Both correct: {np.sum(agree_correct):,} ({np.mean(agree_correct)*100:.1f}%)")
print(f"  Both wrong:   {np.sum(agree_wrong):,} ({np.mean(agree_wrong)*100:.1f}%)")
print(f"  DL-only err:  {np.sum(dl_only):,} ({np.mean(dl_only)*100:.1f}%)")
print(f"  Tree-only err:{np.sum(tree_only):,} ({np.mean(tree_only)*100:.1f}%)")

# 7. DL performance by screen-time band
print("\n--- DL vs Tree Performance by Screen-Time Band ---")
for lo, hi, label in [(0, 3, "[0,3)"), (3, 5, "[3,5)"), (5, 7, "[5,7)"),
                       (7, 9, "[7,9)"), (9, 12, "[9,12)"), (12, 24, "[12,24]")]:
    mask = (screen >= lo) & (screen < hi)
    if np.sum(mask) < 100:
        continue
    dl_auc = roc_auc_score(y[mask], V["dl_s7"][mask])
    tree_auc = roc_auc_score(y[mask], V["EXP122"][mask])
    lt_auc = roc_auc_score(y[mask], V["lookup_transformer"][mask])
    print(f"  {label:8s}: dl={dl_auc:.4f} tree={tree_auc:.4f} lookup_xform={lt_auc:.4f} dl-tree={dl_auc-tree_auc:+.4f}")

# 8. Prediction value distribution
print("\n--- Prediction Value Distribution ---")
for name in ["dl_s7", "dl_s23", "mlp", "lookup_transformer"]:
    pred = V[name]
    mean_all = np.mean(pred)
    std_all = np.std(pred)
    median_all = np.median(pred)
    print(f"  {name:25s}: mean={mean_all:.4f} std={std_all:.4f} median={median_all:.4f}")

# 9. Blend: DL models vs tree-only
print("\n--- DL vs Tree-Only Blend ---")
from scipy.optimize import minimize

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

# Tree-only blend
tree_keys = ["catboost", "xgb_te", "lgb_cat", "lightgbm", "EXP122", "EXP130"]
tree_keys = [k for k in tree_keys if k in keys]
tree_cols = [keys.index(k) for k in tree_keys]
tree_w = fit_weights(tree_cols, y[fit_idx], M[fit_idx])
tree_pred = np.clip(M[:, tree_cols] @ tree_w, 1e-9, 1 - 1e-9)
tree_auc = roc_auc_score(y, tree_pred)
print(f"  Tree-only blend: OOF={tree_auc:.6f}")

# DL-only blend
dl_keys = ["dl_s7", "dl_s23", "mlp", "lookup_transformer"]
dl_keys = [k for k in dl_keys if k in keys]
dl_cols = [keys.index(k) for k in dl_keys]
dl_w = fit_weights(dl_cols, y[fit_idx], M[fit_idx])
dl_pred = np.clip(M[:, dl_cols] @ dl_w, 1e-9, 1 - 1e-9)
dl_auc = roc_auc_score(y, dl_pred)
print(f"  DL-only blend:   OOF={dl_auc:.6f}")

# Combined
all_keys = tree_keys + dl_keys
all_cols = [keys.index(k) for k in all_keys]
all_w = fit_weights(all_cols, y[fit_idx], M[fit_idx])
all_pred = np.clip(M[:, all_cols] @ all_w, 1e-9, 1 - 1e-9)
all_auc = roc_auc_score(y, all_pred)
print(f"  Combined blend:  OOF={all_auc:.6f}")
print(f"  DL adds:         {all_auc - tree_auc:+.6f}")

# Show weights
print(f"\n  Combined weights:")
for i, k in enumerate(all_keys):
    if all_w[i] > 0.01:
        print(f"    {k}: {all_w[i]:.3f}")

print("\n--- DONE ---")
