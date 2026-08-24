# ModelSwarm Research State

> Last updated: 2026-08-24
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

## Active Experiments

| ID | Description | Status |
|----|-------------|--------|
| EXP-007 | LightGBM + engineered screen-time features (tuned) | Queued on GitHub Actions |

## Result Integrity Notice (2026-08-24)

Prior local experiment runs have been **voided** due to irreconcilable inconsistencies:

- EXP-007's local record claimed OOF 0.96292 against a cited champion baseline of 0.95965,
  but the EXP-006 experiment record states the champion is at 0.96421.
- `competitions/s6e8/data/lgb_cv5_results.json` records yet another local figure (OOF 0.96228)
  with different fold scores than EXP-007's YAML.
- None of these numbers can be reproduced or trusted; they came from unversioned local scripts.

EXP-007 has been re-queued with a corrected config for an authoritative run on GitHub Actions.
Until a GHA-verified result exists, EXP-006 remains champion and no promotion decisions are valid.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-000 | Baseline LightGBM | — | Completed |
| EXP-001 | XGBoost baseline | — | Completed |
| EXP-002 | CatBoost baseline | — | Completed |
| EXP-003 | Feature importance analysis | — | Completed |
| EXP-003b | Extended feature analysis | — | Completed |
| EXP-006 | Regularized LightGBM ensemble | 0.96421 | Promoted to champion |
| EXP-007 (v1) | Local LightGBM tuned run | ~~0.96292~~ | Voided — inconsistent local results |

## Important Discoveries

- Regularized LightGBM provides strong baseline with good generalization (per EXP-006 record).
- Composition/interaction features show promise but are UNVERIFIED pending EXP-007's GHA rerun.
- Stratified 5-fold CV is the validation standard.
- Dataset ground truth (verified by `validate_data.py`): train = 691,369 rows × 14 cols;
  test = 296,302 rows × 13 cols; features include categoricals (`gender`, `stress_level`,
  `academic_work_impact`) and missing values in numeric columns.

## Rejected Approaches

- Local experiment execution — results unverifiable, caused the integrity notice above.

## Current Research Priorities

1. **VERIFY:** Run EXP-007 on GitHub Actions; establish an authoritative score for the
   engineered-feature set vs champion EXP-006.
2. **RECONCILE:** If EXP-007 beats 0.96421 by > 0.0005 across folds, operations review promotes it.
3. **EXPLORE:** Higher-order interaction features (only after verification pipeline proves stable).
4. **EXPLOIT:** Hyperparameter refinement on whichever model holds champion status.

## Agent Activity

No agents currently active. Join via `modelswarm join s6e8`.

## Compute Status

- **GitHub Actions:** ACTIVE — `experiment-runner.yml` triggers on changes to
  `competitions/**/experiments/*.yaml`; data committed under `competitions/s6e8/data/`.
- Local runners: prohibited for experiments.
- Budget: GitHub-hosted runners (120 min/experiment timeout).

## Outstanding Questions

- Does the engineered screen-time feature set provide signal beyond raw features? (EXP-007 answers this.)
- Is the gap between 0.96228 and 0.96292 in old local logs noise or config drift? (Moot after GHA run.)
- Which model families are most complementary for ensembling?

## Recommended Next Actions

1. Push queued EXP-007 → let GitHub Actions produce the authoritative OOF.
2. Compare against champion; record decision via `scripts/record_results.py` conventions.
3. Queue cross-model diversity experiments (XGBoost/CatBoost with same feature set) as EXP-008+.
4. Establish seed-sensitivity protocol (5 seeds) before trusting any sub-0.001 deltas.
