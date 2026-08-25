"""AUC-direct hill-climbing blend (Deotte-style) with overfit guard.

Greedy forward selection over the member pool; at each step try every
remaining member and every mixing weight on a grid; accept the move only if
fit-half AUC improves by >= tol (guards noise-fitting). Final composition is
scored on the untouched held-out half.
"""
import os

import numpy as np
import pandas as pd
import kagglehub
from sklearn.metrics import roc_auc_score

lib = os.environ.get("S6E8_LIB") or kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
OWN = os.path.join(os.path.dirname(__file__), "data")
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values

champ = np.load(os.path.join(OWN, "own_champ_m10_oof.npy")).astype(np.float64)
names = ["own_m10"] + [f[:-4] for f in os.listdir(os.path.join(lib, "oof")) if f.startswith("oof_")]
V = {"own_m10": champ}
for n in names[1:]:
    p = os.path.join(lib, "oof", f"{n}.npy")
    if os.path.exists(p):
        v = np.load(p)
        if len(v) == len(y):
            V[n] = v

rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
fit_m, hold_m = idx[:250000], idx[250000:500000]
yf, yh = y[fit_m], y[hold_m]

pool = sorted(V.keys(), key=lambda k: -roc_auc_score(yf, V[k][fit_m]))
print(f"pool {len(pool)}")

TOL = 1e-5
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

blend_f = V[pool[0]][fit_m].copy()
used = [(pool[0], 1.0)]
base_auc = roc_auc_score(yf, blend_f)
remaining = set(pool[1:])

improved = True
while improved:
    improved = False
    best = (TOL, None, None)
    for k in list(remaining):
        for g in GRID:
            cand = (1 - g) * blend_f + g * V[k][fit_m]
            a = roc_auc_score(yf, cand)
            if a - base_auc > best[0]:
                best = (a - base_auc, k, g)
    if best[1] is not None:
        k, g = best[1], best[2]
        blend_f = (1 - g) * blend_f + g * V[k][fit_m]
        base_auc += best[0]
        used.append((k, g))
        remaining.discard(k)
        improved = True
        print(f"+{k} @ mix{g}: fit-half {base_auc:.6f}", flush=True)
    else:
        break

# rebuild on hold-half with same sequence/weights for honest estimate
blend_h = V[used[0][0]][hold_m].copy()
for k, g in used[1:]:
    blend_h = (1 - g) * blend_h + g * V[k][hold_m]
print("\ncomposition:", used)
print(f"held-out AUC: {roc_auc_score(yh, blend_h):.6f}")
