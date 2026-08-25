"""Greedy forward blend selection over champion + all 74 library members.

Time-boxed, OOS-honest: weights refit on fit-half each step, scored on held-out
half (fixed split, seed 0). Reports curve and final composition.
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

t0 = time.time()
lib = r"C:\Users\ROONE~1.DES\AppData\Local\Temp\s6e8_ooflib"
tr = pd.read_csv(r"competitions\s6e8\data\train.csv")
y = tr["addicted_label"].values
champ = pd.read_csv(os.path.join(os.environ["TEMP"], "exp119_art", "oof_predictions.csv"))["prediction"].values

names = ["champ"] + [f[:-4] for f in os.listdir(os.path.join(lib, "oof")) if f.startswith("oof_")]
V = {"champ": champ}
for n in names[1:]:
    try:
        v = np.load(os.path.join(lib, "oof", f"{n}.npy"))
        if len(v) == len(y):
            V[n] = v
    except Exception:
        pass
keys = list(V)
Mall = np.column_stack([V[k] for k in keys])
print(f"pool: {len(keys)} members ({time.time()-t0:.0f}s)")

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


selected = ["champ"]
history = []
for step in range(16):
    best_gain, best_k, best_w = -1, None, None
    for k in keys:
        if k in selected:
            continue
        cols = [keys.index(x) for x in selected + [k]]
        w = fit_weights(cols)
        a = roc_auc_score(y[hold_m], Mall[hold_m][:, cols] @ w)
        if a > best_gain:
            best_gain, best_k, best_w = a, k, (cols, w)
    selected.append(best_k)
    history.append((step + 2, best_k, best_gain))
    print(f"step {step+2}: +{best_k} -> held-out {best_gain:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    if time.time() - t0 > 1500:
        print("time box hit"); break

final_cols, final_w = best_w
print("\nfinal composition:")
for c_i, w_i in zip(final_cols, final_w):
    if w_i > 0.004:
        print(f"  {keys[c_i]}: {w_i:.4f}")
print(f"FINAL held-out AUC: {history[-1][2]:.6f}")
