# OOF Predictions

Out-of-fold (OOF) predictions are first-class research artifacts in ModelSwarm. They enable ensemble construction, diversity analysis, and cross-agent collaboration.

## What is OOF?

OOF predictions are generated during cross-validation: each sample's prediction comes from a fold where that sample was held out. This gives an unbiased estimate of model performance.

## Recording OOF Predictions

Every model producing OOF predictions should record:

- **Sample IDs** — Aligned with the original dataset
- **Target** — True labels
- **Predictions** — Predicted probabilities
- **Fold assignments** — Which fold each sample was in
- **Validation metric** — Computed OOF score
- **Configuration** — Feature set and model used

## OOF Utilities

ModelSwarm provides utilities in `shared/utilities/oof.py`:

```python
from shared.utilities.oof import (
    validate_oof_predictions,
    oof_correlation,
    rank_blend,
    probability_blend,
    ensemble_optimize_weights,
    compute_diversity,
    cross_agent_ensemble
)
```

### Correlation Analysis

```python
# Measure how correlated two models' predictions are
corr = oof_correlation(agent_a_oof, agent_b_oof)
print(f"Correlation: {corr:.4f}")  # Lower = more complementary
```

### Rank Blending

```python
# Robust blend that's insensitive to prediction scale
blended = rank_blend([oof_a, oof_b, oof_c], weights=[0.5, 0.3, 0.2])
```

### Probability Blending

```python
# Weighted average of predicted probabilities
blended = probability_blend([oof_a, oof_b], weights=[0.6, 0.4])
```

### Ensemble Weight Optimization

```python
from sklearn.metrics import roc_auc_score

# Find optimal weights using OOF performance
weights = ensemble_optimize_weights(
    [oof_a, oof_b, oof_c],
    target=y_true,
    metric_fn=roc_auc_score
)
```

### Diversity Measurement

```python
# Pairwise correlation matrix
corr_matrix = compute_diversity([oof_a, oof_b, oof_c])
```

### Cross-Agent Ensemble

```python
# Combine predictions from multiple agents
final = cross_agent_ensemble(
    agent_oofs={
        "agent-7f3c": oof_a,
        "agent-9a2b": oof_b,
        "agent-4d1e": oof_c
    },
    target=y_true,
    method='rank'
)
```

## Ensemble Squared (Ensemble²)

The final ensemble strategy:

```
individual models
        ↓
agent-level ensemble (each agent blends their own models)
        ↓
cross-agent OOF alignment
        ↓
ensemble diversity analysis
        ↓
cross-agent meta-ensemble
        ↓
final submission
```

This two-level approach captures both within-agent and between-agent diversity.

## Standalone Scripts

```bash
# Compute correlation between two OOF files
python shared/scripts/oof_correlation.py --file-a preds_a.csv --file-b preds_b.csv

# Blend submissions via rank averaging
python shared/scripts/rank_blend.py --files sub1.csv sub2.csv sub3.csv --weights 0.5 0.3 0.2
```

## Validation

Always validate OOF predictions before using them:

```python
validate_oof_predictions(ids, target, predictions, folds)
# Raises ValueError if lengths mismatch or folds are inconsistent
```
