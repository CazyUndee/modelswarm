"""Band-local conditional model test (raykkretzschmar 'fix the weak bands' method).

Question: does splicing band-local LightGBM predictions into the base blend
improve within-band ranking enough to matter, without breaking global AUC?

Protocol:
- Base blend: greedy74 5-member composition OOF (reconstructed here from library).
- Bands on daily_screen_time_hours: [3,6) and [6,7.8) per the public method.
- Band LGBMs trained ONLY on band rows (vocabulary rebuilt), same 5-fold scheme,
  plain TE m=10 on canonical cols. Splice = replace blend predictions for band rows.
- Report per-band AUC: base blend vs base+splice, plus full-length AUC after splice.

Runs on GHA via analysis workflow (heavy: several full LGBM trainings).
"""
import os

import kagglehub
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

lib = os.environ.get("S6E8_LIB") or kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
OWN = os.path.join(os.path.dirname(__file__), "data")

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values

# base blend = greedy74 composition weights applied to library vectors
SEL = {"oof_lookup": 0.3139, "oof_tabm_seed3": 0.2758, "oof_naji03": 0.2139,
       "oof_latr1_xgb": 0.1178, "oof_digit_xgb": 0.0786}
w = np.array(list(SEL.values())); w = w / w.sum()
base_oof = np.zeros(len(y))
for k, wk in zip(SEL, w):
    base_oof += wk * np.load(os.path.join(lib, "oof", f"{k}.npy"))
print(f"base blend full AUC {roc_auc_score(y, base_oof):.6f}")

FEATS = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
         "work_study_hours", "sleep_hours", "notifications_per_day",
         "app_opens_per_day", "weekend_screen_time"]
CATS = ["gender", "stress_level", "academic_work_impact"]
BANDS = [("band_3_6h", 3.0, 6.0), ("band_6_78h", 6.0, 7.8)]

daily = tr["daily_screen_time_hours"]
for name, lo, hi in BANDS:
    mask = ((daily >= lo) & (daily < hi)).values
    Xb = tr.loc[mask, FEATS + CATS + ["id"]].copy()
    yb = y[mask]
    base_b = base_oof[mask]
    print(f"\n=== {name}: {mask.sum()} rows | base blend in-band AUC {roc_auc_score(yb, base_b):.6f} ===")
    import lightgbm as lgbm
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_band = np.zeros(len(yb))
    Xb_num = Xb[FEATS]
    for c in CATS:
        Xb[c] = Xb[c].astype("category")
    for f, (tri, vai) in enumerate(skf.split(Xb, yb)):
        m = lgbm.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=63,
            min_child_samples=40, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=2.0, random_state=42+f,
            verbose=-1, n_jobs=-1)
        m.fit(Xb.iloc[tri][FEATS + CATS], yb[tri])
        oof_band[vai] = m.predict_proba(Xb.iloc[vai][FEATS + CATS])[:, 1]
    auc_band = roc_auc_score(yb, oof_band)
    print(f"band-local LGBM in-band OOF AUC {auc_band:.6f}")
    spliced = base_oof.copy()
    spliced[mask] = oof_band
    print(f"global AUC after splice: {roc_auc_score(y, spliced):.6f} "
          f"(delta vs base {roc_auc_score(y, spliced)-roc_auc_score(y, base_oof):+.6f})")
