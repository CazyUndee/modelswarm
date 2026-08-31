"""Member-Pool Forensics: analyze all 252 members for diversity gaps and opportunities.

Questions this answers:
1. Which members are highly correlated (redundant)?
2. Which members are genuinely unique (low correlation with all others)?
3. What model families / feature families are represented vs missing?
4. Where does the ensemble make systematic errors?
5. What predictive information is the pool MISSING?

Run: gh workflow run analysis.yml -f script_path=shared/analysis/member_forensics.py
"""
import sys
import os
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
print("=== MEMBER-POOL FORENSICS ===", flush=True)

N_TRAIN = 691369
N_TEST = 296302

tr = pd.read_csv("competitions/s6e8/data/train.csv")
y = tr["addicted_label"].values
test_df = pd.read_csv("competitions/s6e8/data/test.csv")

def pct_rank(v):
    return (rankdata(v) - 0.5) / len(v)

import kagglehub

def load_vectors(oof_dir, prefix, test_prefixes=("test_", "testpred_", "tep_")):
    out = {}
    seen = set()
    for path in glob.glob(os.path.join(oof_dir, "**", "oof_*.npy"), recursive=True):
        name = os.path.basename(path)[4:-4]
        mate = next(
            (c for tp in test_prefixes
             for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
             if os.path.exists(c)),
            None,
        )
        if mate is None:
            continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if (oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,)
                and np.isfinite(oof).all() and np.isfinite(tst).all()):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    for path in glob.glob(os.path.join(oof_dir, "**", "*_oof.npy"), recursive=True):
        bname = os.path.basename(path)
        name = bname[:-8]
        mate = os.path.join(os.path.dirname(path), name + "_test.npy")
        if not os.path.exists(mate):
            mate = next(
                (c for tp in test_prefixes
                 for c in [os.path.join(os.path.dirname(path), tp + name + ".npy")]
                 if os.path.exists(c)),
                None,
            )
        if mate is None:
            continue
        oof = np.load(path).astype(np.float64)
        tst = np.load(mate).astype(np.float64)
        if (oof.shape == (N_TRAIN,) and tst.shape == (N_TEST,)
                and np.isfinite(oof).all() and np.isfinite(tst).all()):
            key = prefix + name
            if key not in seen:
                out[key] = (oof, tst)
                seen.add(key)
    return out

SOURCES = [
    ("szymonkapiski/s6e8-oof-library-47-models", "sz_"),
    ("paiky1995/s6e8-oof-library-11-members", "nn_"),
    ("aadijoshi19/s6e8-mask-augmented-oof-library", "ma_"),
    ("tamerlanomralinov/s6e8-full-best-blend-npy", "tam_"),
    ("adarsh1077/s6e8-adarsh-oof-library", "a_"),
    ("dariushafshar/s6e8-golem-oof-library", "golem_"),
    ("raykkretzschmar/s6e8-fm-lattice-blend-members", "fm_"),
    ("hboyang/s6e8-catstrall-member", "x_"),
    ("hboyang/s6e8-150-fusion-local-members", "hb_"),
    ("masayakawamata/s6e8-catstr-aug16", "mk_"),
    ("beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts", "bc_"),
    ("beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts", "bd_"),
    ("beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts", "be_"),
    ("beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts", "bf_"),
    ("beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts", "bg_"),
    ("beicicc/s6e8-fixed900-structural-lgbm-artifacts", "bh_"),
]

members = {}
for dataset, prefix in SOURCES:
    try:
        root = kagglehub.dataset_download(dataset)
        got = load_vectors(root, prefix)
        members.update(got)
        print(f"{dataset:44s} {len(got):3d}", flush=True)
    except Exception as e:
        print(f"{dataset:44s} SKIP ({e})", flush=True)

try:
    bolt = kagglehub.dataset_download("boltuzamaki/s6e8-oof-prediction-library")
    oof_df = pd.read_parquet(os.path.join(bolt, "oof_predictions.parquet"))
    tst_df = pd.read_parquet(os.path.join(bolt, "test_predictions.parquet"))
    for col in oof_df.columns:
        if col != "id" and col in tst_df:
            members[f"bolt_{col}"] = (oof_df[col].to_numpy(float), tst_df[col].to_numpy(float))
    print(f"boltuzamaki parquet: {oof_df.shape[1]-1} cols", flush=True)
except Exception as e:
    print(f"boltuzamaki SKIP ({e})", flush=True)

try:
    weak = kagglehub.dataset_download("szymonkapiski/s6e8-50-weakest-oof-models")
    WO = np.load(os.path.join(weak, "oof.npy"), mmap_mode="r")
    WT = np.load(os.path.join(weak, "test.npy"), mmap_mode="r")
    for j in range(WO.shape[1]):
        members[f"weak_{j:02d}"] = (np.asarray(WO[:, j], float), np.asarray(WT[:, j], float))
    print(f"weak-50: {WO.shape[1]} cols", flush=True)
except Exception as e:
    print(f"weak-50 SKIP ({e})", flush=True)

for tag, oof_p, test_p in [
    ("own_champ_m10", "shared/analysis/data/own_champ_m10_oof.npy", "shared/analysis/data/own_champ_m10_test.npy"),
    ("rmlp_realmlp", "shared/analysis/data/oof_realmlp.npy", "shared/analysis/data/pred_realmlp.npy"),
]:
    try:
        members[tag] = (np.load(oof_p).astype(np.float64), np.load(test_p).astype(np.float64))
        print(f"{tag}: loaded", flush=True)
    except Exception as e:
        print(f"{tag}: SKIP ({e})", flush=True)

for tag in ["catboost", "lgb3seed"]:
    try:
        members[f"fresh_{tag}"] = (
            np.load(f"shared/analysis/data/oof_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
            np.load(f"shared/analysis/data/test_fresh_screentime_ratios_{tag}.npy").astype(np.float64),
        )
        print(f"fresh_{tag}: loaded", flush=True)
    except Exception as e:
        print(f"fresh_{tag}: SKIP ({e})", flush=True)

# ============================================================
# 1. MEMBER AUC AND FAMILY CLASSIFICATION
# ============================================================
print(f"\n{'='*70}", flush=True)
print("SECTION 1: MEMBER AUC + FAMILY CLASSIFICATION", flush=True)
print(f"{'='*70}", flush=True)

names = sorted(members)
N = len(names)
OOF = np.column_stack([members[n][0] for n in names])
TST = np.column_stack([members[n][1] for n in names])

auc_vals = np.array([roc_auc_score(y, OOF[:, j]) for j in range(N)])
auc_df = pd.DataFrame({"name": names, "auc": auc_vals})

# Classify families by prefix/name patterns
def classify_family(name):
    if name.startswith("bolt_"):
        if "lookup" in name or "deepfm" in name or "tabm" in name:
            return "bolt_NN"
        elif "xgb" in name:
            return "bolt_XGB"
        elif "lgb" in name:
            return "bolt_LGB"
        elif "cat" in name:
            return "bolt_CatBoost"
        return "bolt_other"
    if name.startswith("weak_"):
        return "weak50"
    if name.startswith("sz_"):
        base = name[3:]
        if "lookup" in base or "tabm" in base or "naji" in base:
            return "sz_NN"
        elif "xgb" in base or "imp_xgb" in base or "lat_xgb" in base:
            return "sz_XGB"
        elif "lgb" in base or "lat_lgb" in base or "imp_lgb" in base:
            return "sz_LGB"
        elif "cat" in base or "imp_cat" in base or "lat_cat" in base:
            return "sz_CatBoost"
        return "sz_other"
    if name.startswith("nn_"):
        return "paiky_NN"
    if name.startswith("ma_"):
        return "mask_aug"
    if name.startswith("tam_"):
        return "tamerlan"
    if name.startswith("a_"):
        return "adarsh"
    if name.startswith("golem_"):
        return "golem"
    if name.startswith("fm_"):
        return "fm_lattice"
    if name.startswith("x_"):
        return "hboyang_catstr"
    if name.startswith("hb_"):
        if "tabm" in name:
            return "hb_TABM"
        elif "bolt" in name and ("lookup" in name or "deepfm" in name):
            return "hb_NN_blend"
        return "hb_other"
    if name.startswith("mk_"):
        return "mk_catstr"
    if name.startswith("bc_") or name.startswith("bd_") or name.startswith("be_") or name.startswith("bf_") or name.startswith("bg_") or name.startswith("bh_"):
        return "beicicc"
    if name == "own_champ_m10":
        return "OWNED"
    if name == "rmlp_realmlp":
        return "RealMLP"
    if name.startswith("fresh_"):
        return "screentime"
    return "unknown"

auc_df["family"] = auc_df["name"].apply(classify_family)

# Per-family summary
print("\n--- Per-Family AUC Summary ---", flush=True)
family_stats = auc_df.groupby("family")["auc"].agg(["count", "mean", "min", "max", "std"]).sort_values("mean", ascending=False)
print(family_stats.to_string(), flush=True)

# Top 20 and bottom 20 members
print("\n--- Top 20 Members by AUC ---", flush=True)
print(auc_df.nlargest(20, "auc")[["name", "auc", "family"]].to_string(index=False), flush=True)
print("\n--- Bottom 20 Members by AUC ---", flush=True)
print(auc_df.nsmallest(20, "auc")[["name", "auc", "family"]].to_string(index=False), flush=True)

# ============================================================
# 2. CORRELATION ANALYSIS — find redundancy and uniqueness
# ============================================================
print(f"\n{'='*70}", flush=True)
print("SECTION 2: CORRELATION ANALYSIS", flush=True)
print(f"{'='*70}", flush=True)

# Spearman correlation on raw OOF (not rank-gauss)
print("Computing Spearman correlation matrix...", flush=True)
corr_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        sp = spearmanr(OOF[:, i], OOF[:, j]).correlation
        corr_matrix[i, j] = sp
        corr_matrix[j, i] = sp

# Most correlated pairs
print("\n--- Top 20 Most Correlated Pairs (redundancy) ---", flush=True)
pairs = []
for i in range(N):
    for j in range(i+1, N):
        pairs.append((corr_matrix[i, j], names[i], names[j]))
pairs.sort(reverse=True)
for sp, n1, n2 in pairs[:20]:
    print(f"  {sp:.6f}  {n1} <-> {n2}", flush=True)

# Least correlated pairs (potential diversity)
print("\n--- Top 20 Least Correlated Pairs (diversity) ---", flush=True)
pairs.sort()
for sp, n1, n2 in pairs[:20]:
    print(f"  {sp:.6f}  {n1} <-> {n2}", flush=True)

# Each member's max correlation with any other member
print("\n--- Members with LOWEST max-correlation (most unique) ---", flush=True)
max_corr = np.zeros(N)
for i in range(N):
    max_corr[i] = max(corr_matrix[i, j] for j in range(N) if j != i)
unique_df = pd.DataFrame({"name": names, "max_corr": max_corr, "auc": auc_vals, "family": [classify_family(n) for n in names]})
print(unique_df.nsmallest(20, "max_corr")[["name", "max_corr", "auc", "family"]].to_string(index=False), flush=True)

print("\n--- Members with HIGHEST max-correlation (most redundant) ---", flush=True)
print(unique_df.nlargest(20, "max_corr")[["name", "max_corr", "auc", "family"]].to_string(index=False), flush=True)

# ============================================================
# 3. STACK OOF AND ERROR ANALYSIS
# ============================================================
print(f"\n{'='*70}", flush=True)
print("SECTION 3: STACK OOF + ERROR ANALYSIS", flush=True)
print(f"{'='*70}", flush=True)

# Rank-gauss transform
R = np.column_stack([pct_rank(OOF[:, j]) for j in range(N)]).astype(np.float32)
G = norm.ppf(np.clip(R, 1e-7, 1 - 1e-7)).astype(np.float32)

# Cross-fitted logistic
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(N_TRAIN), y))

scaler = StandardScaler().fit(G)
lr = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", tol=1e-5)
lr.fit(scaler.transform(G), y)
stack_pred = lr.predict_proba(scaler.transform(G))[:, 1]
stack_auc = roc_auc_score(y, stack_pred)
print(f"Stack OOF (full-data fit proxy): {stack_auc:.6f}", flush=True)

# Cross-fitted OOF for honest error analysis
oof_meta = np.zeros(N_TRAIN)
for fi, vi in folds:
    s = StandardScaler().fit(G[fi])
    m = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", tol=1e-5)
    m.fit(s.transform(G[fi]), y[fi])
    oof_meta[vi] = m.predict_proba(s.transform(G[vi]))[:, 1]
oof_auc = roc_auc_score(y, oof_meta)
print(f"Stack OOF (cross-fitted): {oof_auc:.6f}", flush=True)

# Error regions
errors = np.abs(y - oof_meta)
threshold_90 = np.percentile(errors, 90)
hard_mask = errors >= threshold_90
easy_mask = errors < np.percentile(errors, 50)

print(f"\nRows in top-10% hardest: {hard_mask.sum()} ({hard_mask.mean()*100:.1f}%)", flush=True)
print(f"  True label rate: {y[hard_mask].mean():.4f} (overall: {y.mean():.4f})", flush=True)
print(f"  Mean predicted: {oof_meta[hard_mask].mean():.4f}", flush=True)

# Analyze feature patterns in hard rows
feature_cols = [c for c in tr.columns if c not in ("id", "addicted_label")]
print("\n--- Feature distributions in hard vs easy rows ---", flush=True)
for col in feature_cols:
    hard_vals = tr.loc[hard_mask, col]
    easy_vals = tr.loc[easy_mask, col]
    # For numeric: compare means. For categorical: compare mode.
    if hard_vals.dtype in (np.float64, np.int64):
        hard_mean = hard_vals.mean()
        easy_mean = easy_vals.mean()
        overall_mean = tr[col].mean()
        print(f"  {col:30s} hard={hard_mean:.4f} easy={easy_mean:.4f} overall={overall_mean:.4f}", flush=True)
    else:
        hard_mode = hard_vals.mode().iloc[0] if len(hard_vals.mode()) > 0 else "N/A"
        easy_mode = easy_vals.mode().iloc[0] if len(easy_vals.mode()) > 0 else "N/A"
        print(f"  {col:30s} hard_mode={hard_mode} easy_mode={easy_mode}", flush=True)

# Missing-value analysis in hard rows
print("\n--- Missing-value rates in hard vs easy rows ---", flush=True)
for col in feature_cols:
    hard_miss = tr.loc[hard_mask, col].isna().mean()
    easy_miss = tr.loc[easy_mask, col].isna().mean()
    overall_miss = tr[col].isna().mean()
    print(f"  {col:30s} hard={hard_miss:.4f} easy={easy_miss:.4f} overall={overall_miss:.4f}", flush=True)

# ============================================================
# 4. DIVERSITY GAPS — what's MISSING from the pool?
# ============================================================
print(f"\n{'='*70}", flush=True)
print("SECTION 4: DIVERSITY GAPS — WHAT THE POOL IS MISSING", flush=True)
print(f"{'='*70}", flush=True)

family_counts = auc_df["family"].value_counts()
print("\n--- Family representation ---", flush=True)
print(family_counts.to_string(), flush=True)

# Count NN vs tree models
nn_families = {"bolt_NN", "sz_NN", "paiky_NN", "hb_NN_blend", "hb_TABM", "RealMLP"}
tree_families = {"bolt_XGB", "bolt_LGB", "bolt_CatBoost", "sz_XGB", "sz_LGB", "sz_CatBoost", "weak50", "screentime"}
other_families = {"mask_aug", "tamerlan", "adarsh", "golem", "fm_lattice", "hboyang_catstr", "mk_catstr", "beicicc", "OWNED", "bolt_other", "sz_other", "hb_other"}

nn_count = auc_df[auc_df["family"].isin(nn_families)].shape[0]
tree_count = auc_df[auc_df["family"].isin(tree_families)].shape[0]
other_count = auc_df[auc_df["family"].isin(other_families)].shape[0]

print(f"\n--- Architecture balance ---", flush=True)
print(f"  Neural/NN members: {nn_count} ({nn_count/N*100:.1f}%)", flush=True)
print(f"  Tree-based members: {tree_count} ({tree_count/N*100:.1f}%)", flush=True)
print(f"  Other/blended: {other_count} ({other_count/N*100:.1f}%)", flush=True)

# Average correlation by family
print("\n--- Average within-family vs cross-family correlation ---", flush=True)
for fam in sorted(auc_df["family"].unique()):
    fam_idx = [i for i in range(N) if classify_family(names[i]) == fam]
    if len(fam_idx) < 2:
        continue
    within_corrs = [corr_matrix[i, j] for i in fam_idx for j in fam_idx if i < j]
    outside_corrs = [corr_matrix[i, j] for i in fam_idx for j in range(N) if j not in fam_idx]
    if within_corrs and outside_corrs:
        print(f"  {fam:25s} within={np.mean(within_corrs):.6f}  cross={np.mean(outside_corrs):.6f}  gap={np.mean(within_corrs)-np.mean(outside_corrs):+.6f}", flush=True)

# ============================================================
# 5. MARGINAL CONTRIBUTION — which members actually help?
# ============================================================
print(f"\n{'='*70}", flush=True)
print("SECTION 5: MARGINAL CONTRIBUTION (greedy forward selection, top 50)", flush=True)
print(f"{'='*70}", flush=True)

# Rank-gauss on all members
R_full = np.column_stack([pct_rank(OOF[:, j]) for j in range(N)]).astype(np.float32)
G_full = norm.ppf(np.clip(R_full, 1e-7, 1 - 1e-7)).astype(np.float32)

# Quick greedy: use just 2 folds for speed
fold_sub = folds[:2]

def quick_oof(indices):
    """2-fold CV for speed."""
    oof = np.zeros(N_TRAIN)
    for fi, vi in fold_sub:
        s = StandardScaler().fit(G_full[fi][:, indices])
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(s.transform(G_full[fi][:, indices]), y[fi])
        oof[vi] = m.predict_proba(s.transform(G_full[vi][:, indices]))[:, 1]
    return roc_auc_score(y, oof)

# Start with the best single member
best_single = np.argmax(auc_vals)
selected = [best_single]
current_auc = quick_oof(selected)
print(f"Step 0: {names[best_single]} -> AUC={current_auc:.6f}", flush=True)

for step in range(30):
    best_j = None
    best_gain = 0
    for j in range(N):
        if j in selected:
            continue
        trial = selected + [j]
        a = quick_oof(trial)
        gain = a - current_auc
        if gain > best_gain:
            best_gain = gain
            best_j = j
    if best_j is not None and best_gain > 0.00001:
        selected.append(best_j)
        current_auc = quick_oof(selected)
        print(f"Step {step+1}: +{names[best_j]} (AUC={auc_vals[best_j]:.5f}, gain={best_gain:+.6f}) -> pool={current_auc:.6f}", flush=True)
    else:
        print(f"Step {step+1}: No beneficial addition found. Stopping.", flush=True)
        break

print(f"\nGreedy subset: {len(selected)} members, AUC={current_auc:.6f}", flush=True)
print(f"Full pool: {N} members, AUC={quick_oof(list(range(N))):.6f}", flush=True)

# Members NOT in the greedy set — these are potentially redundant
not_selected = [i for i in range(N) if i not in selected]
print(f"\n--- {len(not_selected)} members NOT selected by greedy (potentially redundant) ---", flush=True)
for i in not_selected[:30]:
    print(f"  {names[i]:50s} AUC={auc_vals[i]:.6f}  family={classify_family(names[i])}", flush=True)

print(f"\n=== FORENSICS COMPLETE ===", flush=True)
