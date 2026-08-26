"""Tamerlan + Owned Models — test whether our owned models add value
to Tamerlan's published 9-model blend (OOF 0.969476).

This is the most actionable intelligence we have: a verified strong blend
with published OOF vectors that we can test additions against.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import spearmanr

np.random.seed(42)

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
N = len(y)

# ============================================================
# LOAD TARMELAN'S MODELS
# ============================================================
print("=" * 70)
print("TARMERLAN + OWNED MODELS ANALYSIS")
print("=" * 70)

tamerlan_dir = "/tmp/blend_npy/blend_data"
V = {}
for name in ["lookup_transformer", "catboost", "dl_s23", "dl_s7", "mlp",
             "xgb_te", "lgb_cat", "lightgbm", "lgb_te_a1"]:
    path = os.path.join(tamerlan_dir, f"oof_{name}.npy")
    if os.path.exists(path):
        V[f"t_{name}"] = np.load(path)
        print(f"  t_{name}: OOF {roc_auc_score(y, V[f't_{name}']):.6f}")

# ============================================================
# LOAD OUR OWNED MODELS
# ============================================================
V["owned_EXP122"] = np.load("shared/artifacts/stacking_vectors/exp122_oof.npy")
V["owned_EXP130"] = np.load("shared/artifacts/stacking_vectors/exp130_oof.npy")
print(f"  owned_EXP122: OOF {roc_auc_score(y, V['owned_EXP122']):.6f}")
print(f"  owned_EXP130: OOF {roc_auc_score(y, V['owned_EXP130']):.6f}")

# Load library NN for comparison
import kagglehub
lib_dir = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
for n in ["lookup", "tabm_seed3", "naji03"]:
    V[f"lib_{n}"] = np.load(os.path.join(lib_dir, "oof", f"oof_{n}.npy"))
    print(f"  lib_{n}: OOF {roc_auc_score(y, V[f'lib_{n}']):.6f}")

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
# BASELINE: TARMERLAN'S BLEND
# ============================================================
print("\n" + "=" * 70)
print("BASELINE: TARMERLAN'S 9-MODEL BLEND")
print("=" * 70)

tamerlan_keys = [k for k in keys if k.startswith("t_")]
t_cols = [keys.index(k) for k in tamerlan_keys]
t_w = fit_weights(t_cols, y[fit_idx], M[fit_idx])
t_pred = np.clip(M[:, t_cols] @ t_w, 1e-9, 1 - 1e-9)
t_auc = roc_auc_score(y, t_pred)
t_held = roc_auc_score(y[hold_idx], t_pred[hold_idx])
print(f"Tamerlan blend (optimized): OOF={t_auc:.6f} held={t_held:.6f}")
print(f"  Weights: {', '.join(f'{tamerlan_keys[i]}={t_w[i]:.3f}' for i in range(len(tamerlan_keys)) if t_w[i] > 0.01)}")

# ============================================================
# PHASE 1: ADD OUR MODELS ONE AT A TIME
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: ADD OWNED MODELS TO TAMERLAN'S BLEND")
print("=" * 70)

our_models = ["owned_EXP122", "owned_EXP130", "lib_lookup", "lib_tabm_seed3", "lib_naji03"]
for model in our_models:
    if model not in keys:
        continue
    model_col = keys.index(model)
    test_cols = t_cols + [model_col]
    w = fit_weights(test_cols, y[fit_idx], M[fit_idx])
    pred = np.clip(M[:, test_cols] @ w, 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, pred)
    held = roc_auc_score(y[hold_idx], pred[hold_idx])
    delta = auc - t_auc
    our_weight = w[-1]
    marker = " ***" if delta > 0.0002 else (" *" if delta > 0.00005 else "")
    print(f"  +{model:25s}: OOF={auc:.6f} held={held:.6f} Δ={delta:+.6f} (w={our_weight:.3f}){marker}")

# ============================================================
# PHASE 2: FORWARD SELECTION (ADD BEST OWNED MODELS)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: FORWARD SELECTION — ADD BEST OWNED MODELS")
print("=" * 70)

remaining_our = list(range(len(our_models)))
selected_our = []
best_auc = t_auc

for step in range(5):
    step_best = (-1, -1, None)
    for candidate_idx in remaining_our:
        model = our_models[candidate_idx]
        if model not in keys:
            continue
        model_col = keys.index(model)
        test_cols = t_cols + [keys.index(our_models[c]) for c in selected_our] + [model_col]
        w = fit_weights(test_cols, y[fit_idx], M[fit_idx])
        pred = np.clip(M[:, test_cols] @ w, 1e-9, 1 - 1e-9)
        auc = roc_auc_score(y, pred)
        if auc > step_best[1]:
            step_best = (candidate_idx, auc, w)
    
    if step_best[0] < 0 or step_best[1] <= best_auc + 0.00001:
        break
    
    selected_our.append(step_best[0])
    remaining_our.remove(step_best[0])
    best_auc = step_best[1]
    
    model = our_models[step_best[0]]
    w = step_best[2]
    our_total_w = sum(w[len(t_cols) + c] for c in range(len(selected_our)))
    held_pred = np.clip(M[hold_idx][:, t_cols + [keys.index(our_models[c]) for c in selected_our]] @ w[:len(t_cols) + len(selected_our)], 1e-9, 1 - 1e-9)
    held_auc = roc_auc_score(y[hold_idx], held_pred)
    print(f"  step {step+1}: +{model:25s} → OOF={best_auc:.6f} held={held_auc:.6f} (our_total_weight={our_total_w:.3f})")

# ============================================================
# PHASE 3: RANK-SPACE vs PROBABILITY-SPACE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: RANK-SPACE vs PROBABILITY-SPACE")
print("=" * 70)

# Tamerlan uses rank-space. Test if probability-space is better with our models.
from scipy.stats import rankdata

# Rank-space version of Tamerlan's blend
M_rank = np.column_stack([rankdata(M[:, k]) / N for k in range(M.shape[1])])
t_rank_pred = np.clip(M_rank[:, t_cols] @ t_w, 1e-9, 1 - 1e-9)
t_rank_auc = roc_auc_score(y, t_rank_pred)
print(f"  Tamerlan (prob-space): OOF={t_auc:.6f}")
print(f"  Tamerlan (rank-space): OOF={t_rank_auc:.6f}")

# ============================================================
# PHASE 4: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: CORRELATION — WHICH MODELS ARE MOST DECORRELATED?")
print("=" * 70)

focus = tamerlan_keys + ["owned_EXP122", "owned_EXP130", "lib_lookup", "lib_tabm_seed3"]
focus = [k for k in focus if k in keys]

print(f"\n{'Model':<30} {'vs best tamerlan':>18} {'vs owned_EXP122':>18}")
print("-" * 70)
best_t = "t_lookup_transformer"
for k in focus:
    if k == best_t:
        continue
    sp_t = spearmanr(V[best_t], V[k]).correlation
    sp_e = spearmanr(V["owned_EXP122"], V[k]).correlation
    print(f"{k:<30} {sp_t:>18.5f} {sp_e:>18.5f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nTamerlan baseline: OOF {t_auc:.6f} (reported 0.969476)")
print(f"Live LB frontier: 0.97186 (Chris Deotte)")
print(f"Tamerlan LB: ~0.97041 (lookup-transformer solo)")
print(f"\nIf our models add >+0.0002 OOF: they have genuine complementary signal")
print(f"If not: Tamerlan's blend already captures what we know")
print(f"\nKey question: can we find the missing 0.0024 between OOF 0.9695 and LB 0.9719?")
