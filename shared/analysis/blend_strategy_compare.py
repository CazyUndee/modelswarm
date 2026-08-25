"""Blend space comparison: probability vs rank vs logit weighting (OOS half-split).

Runs on GHA (analysis workflow). Expects own-champion vectors staged into the
library folder as own_champ_m10_oof.npy / own_champ_m10_test.npy (upload step).
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

lib = os.environ.get("S6E8_LIB", "/tmp/s6e8_ooflib")
tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values

MEMBERS = ["oof_naji05", "oof_naji03", "oof_tabm_seed3", "oof_lookup",
           "oof_tabm_x12", "oof_pub_rmlp", "oof_latwide_xgb"]
champ_oof = np.load(os.path.join(lib, "own_champ_m10_oof.npy"))
champ_test = np.load(os.path.join(lib, "own_champ_m10_test.npy"))

M = np.column_stack([champ_oof] + [np.load(os.path.join(lib, "oof", f"{n}.npy")) for n in MEMBERS])
T = np.column_stack([champ_test] + [np.load(os.path.join(lib, "oof", f"test_{n[4:]}.npy")) for n in MEMBERS])


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
h1, h2 = idx[:150000], idx[150000:450000]


def opt_weights(Xfit):
    def neg(a):
        p = np.clip(Xfit @ a, 1e-9, 1 - 1e-9)
        return -np.mean(y[h1] * np.log(p) + (1 - y[h1]) * np.log(1 - p))
    res = minimize(neg, np.ones(Xfit.shape[1]) / Xfit.shape[1], method="Nelder-Mead",
                   options={"maxiter": 1200, "maxfev": 2000, "xatol": 1e-4, "fatol": 1e-7})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


for tag, X in (("probability", M), ("logit", logit(M))):
    w = opt_weights(X[h1])
    print(f"{tag}-space weighted blend held-out AUC: {roc_auc_score(y[h2], X[h2] @ w):.6f}")

R = np.column_stack([pd.Series(M[:, j]).rank(pct=True).to_numpy() for j in range(M.shape[1])])
w = opt_weights(R[h1])
print(f"rank-space weighted blend held-out AUC: {roc_auc_score(y[h2], R[h2] @ w):.6f}")
equal = roc_auc_score(y[h2], M[h2].mean(axis=1))
print(f"equal-weight prob blend (reference): {equal:.6f}")
