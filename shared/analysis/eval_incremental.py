"""Eval incremental value of owned vectors vs 260-member naji stack — GHA only."""
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import kagglehub, json

y=pd.read_csv("competitions/s6e8/data/train.csv")["addicted_label"].values
print(f"y len {len(y)}", flush=True)

# Reconstruct 260 stack OOF exactly as analysis339 did — download all libraries
# Use same download list as analysis339 for fidelity
libs=[
 ("szymonkapiski/s6e8-oof-library-47-models",74),
 ("paiky1995/s6e8-oof-library-11-members",11),
 ("aadijoshi19/s6e8-mask-augmented-oof-library",9),
 ("tamerlanomralinov/s6e8-full-best-blend-npy",9),
 ("adarsh1077/s6e8-adarsh-oof-library",22),
 ("dariushafshar/s6e8-golem-oof-library",7),
 ("raykkretzschmar/s6e8-fm-lattice-blend-members",5),
 ("hboyang/s6e8-catstrall-member",5),
 ("hboyang/s6e8-150-fusion-local-members",17),
 ("masayakawamata/s6e8-catstr-aug16",1),
 ("beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts",1),
 ("beicicc/s6e8-fixed1500-xgb-identity-digit-artifacts",2),
 ("beicicc/s6e8-fixed1500-xgb-screen-relation-artifacts",2),
 ("beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts",1),
 ("beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts",1),
 ("beicicc/s6e8-fixed900-structural-lgbm-artifacts",2),
 ("boltuzamaki/s6e8-oof-prediction-library",47),
 ("szymonkapiski/s6e8-50-weakest-oof-models",50),
]
members={}
# naji OOF members
try:
    naji_dir=kagglehub.dataset_download("najiama/predicting-smartphone-addiction-oof-submission-csv")
    for f in glob.glob(os.path.join(naji_dir,"*.csv"))+glob.glob(os.path.join(naji_dir,"**/*.csv"),recursive=True):
        try:
            df=pd.read_csv(f)
            if "addicted_label" in df.columns or "prediction" in df.columns:
                col="addicted_label" if "addicted_label" in df.columns else "prediction"
                v=df[col].values
                if len(v)==691369:
                    members[os.path.basename(f)]=v
        except: pass
    print(f"naji csvs loaded {len(members)}", flush=True)
except Exception as e: print(f"naji dl fail {e}", flush=True)

# For incremental test we don't need full 260 reconstruction if we can just load the analysis339 stack OOF via downloading its artifact? Instead approximate stack OOF as mean of available members + owned.
# Simpler: load owned vectors first and test pairwise
owned={}
for path,name in [
 ("shared/artifacts/stacking_vectors/exp122_oof.npy","exp122"),
 ("shared/artifacts/stacking_vectors/exp130_oof.npy","exp130"),
 ("shared/artifacts/exp137/exp-EXP-137-artifacts/oof_member_xgboost[0].csv","exp137_xgb"),
 ("shared/artifacts/exp138/exp-EXP-138-artifacts/oof_member_catboost[0].csv","exp138_cat"),
 ("shared/artifacts/exp136/exp-EXP-136-artifacts/oof_member_knn[1].csv","exp136_knn"),
]:
    try:
        if path.endswith(".npy"):
            v=np.load(path)
        else:
            df=pd.read_csv(path)
            col="prediction" if "prediction" in df.columns else df.columns[-1]
            v=df[col].values
        owned[name]=v
        print(f"{name} OOF {roc_auc_score(y,v):.6f} len {len(v)}", flush=True)
    except Exception as e: print(f"fail {name} {e}", flush=True)

# Load 8 lib members as proxy stack (from earlier analysis) to compute correlations
lib_proxy={}
try:
    d=kagglehub.dataset_download("szymonkapiski/s6e8-oof-library-47-models")
    import os as _os
    # load same 8 as in analysis352
    for n in ["lookup","tabm_seed3","naji03","latr1_xgb","digit_xgb","digit_cat","latr1_lgbm","digit_lgbm"]:
        p=os.path.join(d,"oof",f"oof_{n}.npy")
        if os.path.exists(p):
            v=np.load(p)
            lib_proxy[n]=v
            print(f"lib {n} OOF {roc_auc_score(y,v):.6f}", flush=True)
except Exception as e: print(f"lib dl fail {e}", flush=True)

stack_proxy=None
if lib_proxy:
    M=np.column_stack(list(lib_proxy.values()))
    # simple rank-average proxy for stack (weights from analysis352)
    w=np.array([0.2905,0.2743,0.2534,0.1688,0.0827,0,0,0]) # truncated to 8, but need 5 active: lookup,tabm,naji03,latr1_xgb,digit_xgb
    # use uniform for proxy
    stack_proxy=M.mean(axis=1)
    print(f"proxy stack OOF {roc_auc_score(y, stack_proxy):.6f}", flush=True)
    for name,v in owned.items():
        sp=spearmanr(stack_proxy, v).correlation
        pe=np.corrcoef(stack_proxy, v)[0,1]
        # incremental: mean blend
        inc=np.mean(np.column_stack([stack_proxy, v]), axis=1)
        inc_auc=roc_auc_score(y, inc)
        print(f"{name} vs proxy stack: spearman {sp:.5f} pearson {pe:.5f} inc_mean_OOF {inc_auc:.6f} delta {inc_auc - roc_auc_score(y, stack_proxy):+.6f}", flush=True)
        # disagreement
        # rank disagreement rate at 0.5 threshold proxy
        b_stack=(stack_proxy>0.5)
        b_v=(v>0.5)
        dis=(b_stack!=b_v).mean()
        # conditional correctness on disagreements
        if dis>0:
            mask=b_stack!=b_v
            correct_v=((v[mask]>0.5)==(y[mask]==1)).mean()
            correct_s=((stack_proxy[mask]>0.5)==(y[mask]==1)).mean()
            print(f"  disagree {dis:.3%} v_correct {correct_v:.3%} stack_correct {correct_s:.3%}", flush=True)

print("DONE eval_incremental", flush=True)
