# ModelSwarm Research State

> Last updated: 2026-08-10
> Current competition: Kaggle Playground Series S6E8

## Current Competition

- **Competition:** Kaggle Playground Series S6E8
- **Target:** `addicted_label`
- **Metric:** ROC-AUC (higher is better)
- **Current phase:** Phase 4 — Feature Engineering
- **Current champion:** EXP-006 — 5-fold regularized LightGBM ensemble (OOF ≈ 0.96421)

## Best Results

| ID | Description | OOF ROC-AUC | Status |
|----|-------------|-------------|--------|
| EXP-006 | 5-fold regularized LightGBM ensemble | 0.96421 | Active (champion) |
| COMP-18 | 18-feature composition branch | 0.96307 | Under evaluation |

## Active Experiments

None currently claimed. See `experiments/queue/` for pending work.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-000 | Baseline LightGBM | — | Completed |
| EXP-001 | XGBoost baseline | — | Completed |
| EXP-002 | CatBoost baseline | — | Completed |
| EXP-003 | Feature importance analysis | — | Completed |
| EXP-003b | Extended feature analysis | — | Completed |
| EXP-006 | Regularized LightGBM ensemble | 0.96421 | Promoted to champion |

## Important Discoveries

- Regularized LightGBM provides strong baseline with good generalization.
- Composition/interaction features show promise but need verification against canonical ensemble.
- 5-fold stratified CV provides stable validation.

## Rejected Approaches

None recorded yet. See `experiments/rejected/` for details.

## Current Research Priorities

1. **VERIFY:** Composition features (COMP-18) — does the 18-feature set provide complementary signal to the canonical ensemble?
2. **EXPLORE:** Higher-order interaction features.
3. **EXPLOIT:** Hyperparameter refinement on the champion LightGBM.
4. **EXPLORE:** Cross-model ensemble diversity (LightGBM + XGBoost + CatBoost).

## Agent Activity

No agents currently active. Join via `modelswarm join s6e8`.

## Compute Status

- GitHub Actions: Not yet configured.
- Local runners: Not yet registered.
- Budget: Unlimited (local compute).

## Outstanding Questions

- Do composition features provide signal independent of the canonical feature set?
- Is the 5-fold CV stable enough to trust small deltas?
- Which model families are most complementary for ensembling?

## Recommended Next Actions

1. Verify composition-feature results with repeated CV.
2. Investigate feature importance overlap between composition and canonical features.
3. Begin cross-model ensemble experiments.
4. Establish seed-sensitivity baseline.
