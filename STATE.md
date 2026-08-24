# ModelSwarm Research State

> Last updated: 2026-08-24 (post EXP-007 GHA run)
> Current competition: Kaggle Playground Series S6E8
> Compute policy: **ALL experiments run on GitHub Actions** (`.github/workflows/experiment-runner.yml`). Local ML runs are prohibited — see `AGENT_INSTRUCTIONS.md`.

## Current Competition

- **Competition:** Kaggle Playground Series S6E8
- **Target:** `addicted_label`
- **Metric:** ROC-AUC (higher is better)
- **Current phase:** Phase 4 — Feature Engineering
- **Current champion:** EXP-006 — 5-fold regularized LightGBM ensemble (OOF 0.96421)

## Best Results

| ID | Description | OOF ROC-AUC | Status |
|----|-------------|-------------|--------|
| EXP-006 | 5-fold regularized LightGBM ensemble | 0.96421 | Active (champion) |
| EXP-007 | LightGBM + engineered screen-time features | 0.96292 | Rejected (GHA-verified) |

## Active Experiments

None. Queue the next hypothesis as `competitions/s6e8/experiments/EXP-008.yaml` and push.

## Result Integrity Notice (2026-08-24) — RESOLVED

Prior local runs were voided due to irreconcilable numbers (0.96292 vs cited baseline
0.95965 vs 0.96228 in lgb_cv5_results.json). EXP-007 was re-run authoritatively on
GitHub Actions:

- **Verified EXP-007 OOF: 0.962917** (folds 0.96206 / 0.96264 / 0.96325 / 0.96390 / 0.96274).
- The old local score 0.962920 was actually accurate; the corrupted part was its champion
  baseline citation (0.95965 instead of the true 0.96421).
- **EXP-007 is BELOW champion by -0.00129 → rejected.** Champion remains EXP-006.
- The lgb_cv5_results.json figure (0.96228) came from a different, older config and is moot.

All future scores must come from GHA runs. No exceptions.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-000 | Baseline LightGBM | — | Completed |
| EXP-001 | XGBoost baseline | — | Completed |
| EXP-002 | CatBoost baseline | — | Completed |
| EXP-003 | Feature importance analysis | — | Completed |
| EXP-003b | Extended feature analysis | — | Completed |
| EXP-006 | Regularized LightGBM ensemble | 0.96421 | Promoted to champion |
| EXP-007 | LightGBM + engineered ratios (tuned) | 0.96292 (GHA) | Rejected — below champion |

## Important Discoveries

- Regularized LightGBM provides strong baseline with good generalization (per EXP-006 record).
- **Engineered screen-time ratio features add NO signal over EXP-006's configuration**
  (GHA-verified: −0.00129). Deprioritize this feature family.
- Stratified 5-fold CV is stable: EXP-007 fold spread ~0.0019, std ~0.0007.
- Dataset ground truth (verified): train = 691,369 × 14; test = 296,302 × 13;
  3 categoricals (`gender`, `stress_level`, `academic_work_impact`); numeric NaNs present.

## Rejected Approaches

- Local experiment execution — unverifiable, caused the integrity incident above.
- Engineered screen-time ratio/interaction features on top of raw features (EXP-007).

## Current Research Priorities

1. **EXPLORE:** Cross-model diversity — XGBoost and CatBoost with tuned params on the
   canonical feature set (queue as EXP-008/EXP-009), targeting an ensemble blend later.
2. **EXPLOIT:** Hyperparameter refinement directly on champion EXP-006's configuration.
3. **VERIFY:** Seed sensitivity of the champion before trusting any sub-0.001 delta.
4. **AVOID:** Further screen-time ratio engineering — proven void by EXP-007.

## Agent Activity

No agents currently active. Join via `modelswarm join s6e8`.

## Compute Status

- **GitHub Actions:** ACTIVE and PROVEN — EXP-007 completed end-to-end on
  `ubuntu-latest` in ~240s training time (run 32745631926).
- Data committed under `competitions/s6e8/data/`; runners validate it before training.
- Local runners: prohibited for experiments.

## Outstanding Questions

- Are XGBoost/CatBoost errors sufficiently decorrelated from LightGBM for blending?
- How seed-sensitive is EXP-006's 0.96421?
- Do missing-value indicators carry signal the raw NaNs don't?

## Recommended Next Actions

1. Queue EXP-008 (XGBoost, canonical features, tuned) via GHA.
2. Queue EXP-009 (CatBoost, same protocol) in parallel.
3. Once two strong diverse models exist, queue a rank-average ensemble experiment.
