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
| EXP-006 | 5-fold regularized LightGBM ensemble | 0.96421 | Champion — **UNDER RE-VERIFICATION (EXP-010)** |
| EXP-008 | LGBM member, champion-style config (GHA) | 0.96322 | Best GHA-verified single model |

## Active Experiments

| ID | Description | Status |
|----|-------------|--------|
| EXP-009 | 5-seed LightGBM ensemble (seed diversity) | Running on GitHub Actions |
| EXP-010 | Faithful legacy-champion config reproduction | Running on GitHub Actions |

## Result Integrity Notice (2026-08-24) — RESOLVED

Prior local runs were voided due to irreconcilable numbers (0.96292 vs cited baseline
0.95965 vs 0.96228 in lgb_cv5_results.json). EXP-007 was re-run authoritatively on
GitHub Actions:

- **Verified EXP-007 OOF: 0.962917** (folds 0.96206 / 0.96264 / 0.96325 / 0.96390 / 0.96274).
- The old local score 0.962920 was actually accurate; the corrupted part was its champion
  baseline citation (0.95965 instead of the true 0.96421).
- **EXP-007 is BELOW champion by -0.00129 → rejected.**
- The lgb_cv5_results.json figure (0.96228) came from a different, older config and is moot.

## CHAMPION UNDER REVIEW (2026-08-24, post EXP-008)

EXP-008's champion-style LGBM member scored **0.96322**, not 0.96421. The EXP-006 record
is a legacy local-era artifact: empty fold metrics, no artifacts. Its config lists
`subsample: 0.8` without `subsample_freq` (sklearn default 0 = row bagging DISABLED).
**EXP-010 reproduces that configuration exactly on GHA.** Until it lands, treat both
0.96421 and 0.96322 as candidate baselines; no promotion decisions.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-000–003b | Baselines + feature analysis (local era) | — | Unverified |
| EXP-006 | Regularized LightGBM ensemble (local era) | 0.96421? | Under re-verification |
| EXP-007 | LightGBM + engineered ratios (tuned) | 0.96292 (GHA) | Rejected — below champion |
| EXP-008 | LGBM+XGB+CatBoost probability blend | 0.96227 (GHA) | Rejected — members too correlated, blend drags |

## Important Discoveries

- **Cross-family OOF correlations are near-saturated**: LGBM~XGB 0.9952,
  XGB~CatBoost 0.9867, LGBM~CatBoost 0.9828. Family-level blending is dead on this
  dataset unless a genuinely different model class (linear/NN/kNN) enters the mix.
- CatBoost substantially weaker here (0.95681) — likely ordinal-encoded categoricals
  underperform native handling; deprioritize.
- **Missing-value indicators are a dead hypothesis** (EDA): target rate among missing
  ≈ known for every feature (max Δ +0.004). No experiment needed.
- Signal concentrates in `daily_screen_time_hours`, `weekend_screen_time`,
  `social_media_hours` — strongly monotone, saturating near p=1.0 in top deciles.
  Categoricals carry minimal signal (gender Δ≈0.02 at most).
- Stratified 5-fold CV is stable: fold std ~0.0007 across GHA runs.
- All scores must come from GHA runs. Local results void.

## Rejected Approaches

- Local experiment execution — unverifiable, caused the integrity incident above.
- Engineered screen-time ratio/interaction features on raw features (EXP-007).
- Cross-family GBDT probability blending (EXP-008): correlation too high, members weaker.
- Missingness indicators (EDA-eliminated, no run needed).

## Current Research Priorities

1. **VERIFY:** EXP-010 — is the champion 0.96421 reproducible, or is the true baseline ~0.96322?
2. **MEASURE:** EXP-009 — seed sensitivity of the champion-family config; does seed averaging help?
3. Then (evidence-dependent):
   - If true baseline ≈ 0.9632: hyperparameter refinement becomes the main lever (num_leaves/lr/min_child sweep via parallel matrix runs).
   - If blend candidates remain viable: fixed-weight blends toward the strongest member.
   - Consider a decorrelated non-GBDT member (regularized logistic regression on saturated features) for stacking diversity.

## Agent Activity

Orchestrator active (this session): queued EXP-007..010; runner now supports ensembles/blends.

## Compute Status

- **GitHub Actions:** ACTIVE and PROVEN — matrix parallelism works (EXP-009+010 concurrently);
  result commits are rebase-safe; sync-results skips cleanly without API creds.
- Data committed under `competitions/s6e8/data/`; runners validate before training.
- Local runners: prohibited for experiments; EDA/EDA-scripts allowed.

## Outstanding Questions

- Is EXP-006's 0.96421 real? (EXP-010)
- How much does seed variance contribute vs family variance? (EXP-009 vs EXP-008 correlations)
- Would a non-tree model class provide usable decorrelation for stacking?

## Recommended Next Actions

1. Analyze EXP-009/010 on completion; re-baseline champion per evidence.
2. If re-baselined to ~0.9632: queue HP sweep (EXP-011+) as parallel matrix experiments.
3. Maintain the loop: analyze → hypothesize → implement → queue → monitor → record.
