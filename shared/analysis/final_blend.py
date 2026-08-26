"""FINAL blend composer — runs after EXP-126/127 artifacts land.

Protocol identical to greedy-74 (OOS half-split, logloss-fit inner weights,
AUC-greedy forward selection over: own singles + FMs + library core).
Submit only if held-out beats greedy74's 0.969654.

Usage: python final_blend.py <exp126_art_dir> [exp127_art_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import kagglehub
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

lib = kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
fm = r"C:\Users\ROONE~1.DES\AppData\Local\Temp\s6e8_fm"
exp126 = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ROONE~1.DES\AppData\Local\Temp\exp126_art"
exp127 = sys.argv[2] if len(sys.argv) > 2 else None

tr = pd.read_csv(r"competitions\s6e8\data\train.csv")
y = tr["addicted_label"].values

V, T = {}, {}
own = pd.read_csv(os.path.join(exp126, "oof_predictions.csv"))
s = pd.read_csv(os.path.join(exp126, "submission.csv"))
V["own_126"] = own["prediction"].values
T["own_126"] = s["addicted_label"].values
if exp127 and os.path.exists(os.path.join(exp127, "oof_predictions.csv")):
    o = pd.read_csv(os.path.join(exp127, "oof_predictions.csv"))
    t = pd.read_csv(os.path.join(exp127, "submission.csv"))
    V["own_tabm"] = o["prediction"].values
    T["own_tabm"] = t["addicted_label"].values
for n in ("fmplr", "fmnum", "fmpure", "fmwide", "fmdeep"):
    V[n] = np.load(os.path.join(fm, f"oof_{n}.npy"))
    T[n] = np.load(os.path.join(fm, f"test_{n}.npy"))
for n in ("lookup", "tabm_seed3", "naji03", "latr1_xgb", "digit_xgb"):
    V[n] = np.load(os.path.join(lib, "oof", f"oof_{n}.npy"))
    T[n] = np.load(os.path.join(lib, "oof", f"test_{n}.npy"))

keys = list(V)
Mall = np.column_stack([V[k] for k in keys])
print(f"pool {len(keys)}: {keys}")

rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
fit_m, hold_m = idx[:200000], idx[200000:500000]


def fit_weights(cols):
    Mf = Mall[fit_m][:, cols]

    def neg(a):
        p = np.clip(Mf @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y[fit_m] * np.log(p) + (1 - y[fit_m]) * np.log(1 - p))

    res = minimize(neg, np.ones(len(cols)) / len(cols), method="Nelder-Mead",
                   options={"maxiter": 800, "maxfev": 1200, "xatol": 1e-4, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


selected = ["own_126"]
best_hold = roc_auc_score(y[hold_m], Mall[hold_m][:, 0])
print(f"start own_126 hold {best_hold:.6f}")
for step in range(10):
    step_best = (None, None, -1)
    for k in keys:
        if k in selected:
            continue
        cols = [keys.index(x) for x in selected + [k]]
        w = fit_weights(cols)
        a = roc_auc_score(y[hold_m], Mall[hold_m][:, cols] @ w)
        if a > step_best[2]:
            step_best = (k, (cols, w), a)
    if step_best[0] is None:
        break
    selected.append(step_best[0])
    print(f"step {step+1}: +{step_best[0]} -> held-out {step_best[2]:.6f}", flush=True)

cols, w = step_best[1]
final_hold = roc_auc_score(y[hold_m], Mall[hold_m][:, cols] @ w)
print("\ncomposition:")
for ci, wi in zip(cols, w):
    if wi > 0.004:
        print(f"  {keys[ci]}: {wi:.4f}")
print(f"FINAL held-out: {final_hold:.6f}  (greedy74 baseline 0.969654)")
if final_hold > 0.969654:
    blend = np.clip(T[:, cols] @ w, 0, 1)
    out = pd.DataFrame({"id": s["id"].values, "addicted_label": blend})
    path = r"C:\Users\ROONE~1.DES\AppData\Local\Temp\opencode\submission_final.csv"
    out.to_csv(path, index=False)
    print("SUBMITTABLE:", path)
else:
    print("does not beat baseline -> no submission")
