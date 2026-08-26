"""
Frontier attack harness: test whether owned signal improves the 0.97128 external frontier.

External frontier = Naji Ama Ensemble of Ensembles (vault submission.csv) = 0.97128 LB
Owned pool = greedy74 members + EXP-130 + lookup-transformer (when available) + FM etc.

Protocol:
- For each candidate combination, optimize weights on OOF where available (library + owned OOFs)
- Frontier itself has no OOF in our store, so:
  * For OOF evaluation, use proxy: the frontier's test correlation structure is validated via LB
  * Weights are fit on OOF of owned components, then applied to test predictions including frontier
- Track: standalone OOF, correlation vs frontier/greedy74, error complementarity via residual analysis
- Submit only if OOF improvement is robust and LB confirms (per overfitting policy)

Usage:
  python shared/analysis/frontier_attack.py --lookup-oof experiments/output/EXP-132/oof_predictions.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def load_oof(path):
    return pd.read_csv(path)["prediction"].values if path.endswith(".csv") else np.load(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup-oof", default=None)
    parser.add_argument("--lookup-test", default=None)
    args = parser.parse_args()

    # Load frontier test predictions (external reference)
    frontier_test = pd.read_csv("/tmp/s6e8_vault_submission.csv")["addicted_label"].values if False else None
    # In GHA, use committed vault path or downloaded
    print("Frontier attack harness ready.")
    print("External frontier: Naji Ama 0.97128 (vault submission.csv)")
    print("Owned pool: greedy74 (lookup tabm_seed3 naji03 latr1_xgb digit_xgb), EXP-130, lookup-transformer")
    print("Next: when EXP-132 OOF lands, run full complementarity analysis:")
    print("  - Spearman vs frontier (if <0.99, potentially complementary)")
    print("  - Residual disagreement regions (where lookup disagrees strongly with frontier)")
    print("  - Blend gains: frontier, frontier+lookup, frontier+greedy74, etc. with OOF-validated weights")
