"""
OOF (Out-of-Fold) prediction utilities for ensemble construction.

This module provides functions for validating, analyzing, and combining
OOF predictions from multiple models and agents.
"""

import numpy as np
from scipy.stats import pearsonr, rankdata


def validate_oof_predictions(ids: np.ndarray, target: np.ndarray,
                              predictions: np.ndarray, folds: np.ndarray) -> None:
    """Validate OOF prediction arrays for consistency.

    Raises:
        ValueError: If array lengths mismatch or predictions are out of range.
    """
    if not (len(ids) == len(target) == len(predictions) == len(folds)):
        raise ValueError(
            f"Array length mismatch: ids={len(ids)}, target={len(target)}, "
            f"predictions={len(predictions)}, folds={len(folds)}"
        )

    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError("Predictions must be in [0, 1] range.")

    if len(np.unique(folds)) < 2:
        raise ValueError("At least 2 folds required for OOF predictions.")


def oof_correlation(oof_a: np.ndarray, oof_b: np.ndarray) -> float:
    """Compute Pearson correlation between two OOF prediction sets."""
    if len(oof_a) != len(oof_b):
        raise ValueError(f"Length mismatch: {len(oof_a)} vs {len(oof_b)}")

    corr, _ = pearsonr(oof_a, oof_b)
    return float(corr)


def rank_blend(oof_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Blend OOF predictions using rank averaging.

    Rank blending is robust to differences in prediction scale.
    """
    n = len(oof_list[0])
    if weights is None:
        weights = [1.0 / len(oof_list)] * len(oof_list)

    if len(weights) != len(oof_list):
        raise ValueError("Weights and oof_list must have same length.")

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    blended = np.zeros(n)
    for oof, w in zip(oof_list, weights):
        ranks = rankdata(oof, method='average')
        normalized = (ranks - 1) / (len(ranks) - 1)
        blended += w * normalized

    return blended


def probability_blend(oof_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Blend OOF predictions using weighted probability averaging."""
    n = len(oof_list[0])
    if weights is None:
        weights = [1.0 / len(oof_list)] * len(oof_list)

    if len(weights) != len(oof_list):
        raise ValueError("Weights and oof_list must have same length.")

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    blended = np.zeros(n)
    for oof, w in zip(oof_list, weights):
        blended += w * oof

    return blended


def ensemble_optimize_weights(oof_list: list[np.ndarray], target: np.ndarray,
                               metric_fn) -> list[float]:
    """Optimize blend weights using OOF performance via grid search."""
    n_models = len(oof_list)

    if n_models == 2:
        best_score = -np.inf
        best_weights = [0.5, 0.5]
        for w1 in np.arange(0.0, 1.01, 0.05):
            w2 = 1.0 - w1
            blended = w1 * oof_list[0] + w2 * oof_list[1]
            score = metric_fn(target, blended)
            if score > best_score:
                best_score = score
                best_weights = [w1, w2]
        return best_weights

    return [1.0 / n_models] * n_models


def compute_diversity(oof_list: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise correlation matrix as a diversity measure.

    Lower correlation = more diversity = better ensemble potential.
    """
    n = len(oof_list)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            corr = oof_correlation(oof_list[i], oof_list[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def fold_alignment_score(ids_a: np.ndarray, ids_b: np.ndarray,
                          folds_a: np.ndarray, folds_b: np.ndarray) -> float:
    """Score how well two fold assignments align for ensembling.

    Returns the fraction of samples that share the same fold relative assignment.
    """
    if len(ids_a) != len(folds_a) or len(ids_b) != len(folds_b):
        raise ValueError("ID and fold arrays must have same length.")

    # Align by IDs
    map_a = dict(zip(ids_a, folds_a))
    map_b = dict(zip(ids_b, folds_b))

    common_ids = set(ids_a) & set(ids_b)
    if not common_ids:
        return 0.0

    agreements = sum(1 for idx in common_ids if map_a[idx] == map_b[idx])
    return agreements / len(common_ids)


def cross_agent_ensemble(agent_oofs: dict[str, np.ndarray], target: np.ndarray,
                          method: str = 'rank') -> np.ndarray:
    """Combine OOF predictions from multiple agents into a meta-ensemble.

    This implements ensemble^2: each agent's internal ensemble is combined
    with other agents' ensembles for a cross-agent meta-ensemble.

    Args:
        agent_oofs: Dict mapping agent_id to their OOF prediction array
        target: True labels (for weight optimization)
        method: 'rank' for rank blending, 'probability' for weighted average

    Returns:
        Meta-ensemble predictions.
    """
    oof_list = list(agent_oofs.values())

    if method == 'rank':
        return rank_blend(oof_list)
    elif method == 'probability':
        return probability_blend(oof_list)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rank' or 'probability'.")
