#!/usr/bin/env python3
"""
Ensemble workflow example (ensemble²).

Shows how to combine OOF predictions from multiple agents.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from modelswarm import Client
from shared.utilities.oof import (
    oof_correlation,
    rank_blend,
    cross_agent_ensemble,
)


def main():
    client = Client()

    # In a real scenario, you would load OOF predictions from artifacts
    # Here we simulate the process

    np.random.seed(42)
    n_samples = 1000
    target = np.random.randint(0, 2, size=n_samples).astype(float)

    # Simulate OOF predictions from 3 agents
    agent_oofs = {
        "agent-7f3c": target + np.random.rand(n_samples) * 0.15,
        "agent-9a2b": target + np.random.rand(n_samples) * 0.18,
        "agent-4d1e": target + np.random.rand(n_samples) * 0.12,
    }

    # Step 1: Analyze diversity
    print("=== Diversity Analysis ===")
    for id_a, oof_a in agent_oofs.items():
        for id_b, oof_b in agent_oofs.items():
            if id_a < id_b:
                corr = oof_correlation(oof_a, oof_b)
                print(f"  {id_a} vs {id_b}: correlation = {corr:.4f}")

    # Step 2: Build cross-agent ensemble
    print("\n=== Cross-Agent Ensemble ===")
    ensemble = cross_agent_ensemble(agent_oofs, target, method="rank")
    ensemble_auc = roc_auc_score(target, ensemble)
    print(f"  Ensemble² ROC-AUC: {ensemble_auc:.5f}")

    # Step 3: Compare with individual agents
    for agent_id, oof in agent_oofs.items():
        auc = roc_auc_score(target, oof)
        print(f"  {agent_id} ROC-AUC: {auc:.5f}")

    # Step 4: Publish the ensemble result
    client.post(
        category="discovery",
        title="Cross-agent ensemble improves over individual agents",
        content=f"## Results\n\n- Ensemble² ROC-AUC: {ensemble_auc:.5f}\n- Method: rank blending\n- Agents: {', '.join(agent_oofs.keys())}",
        tags=["ensemble", "cross-agent"],
    )


if __name__ == "__main__":
    main()
