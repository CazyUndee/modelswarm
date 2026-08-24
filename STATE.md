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
| **EXP-010** | Faithful legacy-champion config reproduction | **0.963127** | **Verified reference baseline** |
| EXP-009 | 5-seed LightGBM ensemble | 0.963303 | Best verified score (+0.00018, inconclusive) |
| EXP-008 | LGBM member, bagged champion-style | 0.963223 | Verified |

## Active Experiments

| ID | Description | Status |
|----|-------------|--------|
| EXP-011 | 5-seed ensemble + row bagging (A/B vs EXP-009) | Running on GitHub Actions |
| EXP-012 | Capacity probe: num_leaves 127, lr 0.03, 5 seeds | Running on GitHub Actions |

## CHAMPION RE-BASELINE (2026-08-24, EXP-010)

Faithful GHA reproduction of the legacy EXP-006 configuration yields
**OOF 0.963127**, not the claimed 0.96421 (legacy record: local-era, empty folds,
no artifacts — unreliable). All comparisons now key off 0.963127.
Recomputed field deltas: everything clusters at **0.9631 ± 0.0002 — a plateau**:

| Config | OOF | Δ vs 0.963127 |
|---|---|---|
| EXP-010 exact legacy champion | 0.963127 | baseline |
| EXP-008 LGBM (+subsample_freq 1) | 0.963223 | +0.00010 |
| EXP-009 5-seed average | 0.963303 | +0.00018 |
| EXP-007 ratios+tuned | 0.962917 | −0.00021 |
| EXP-008 blend (LGBM+XGB+CatBoost) | 0.962273 | −0.00085 |

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-006 (legacy) | Local-era champion claim | ~~0.96421~~ | Not reproducible — superseded by EXP-010 |
| EXP-007 | LightGBM + engineered ratios (tuned) | 0.96292 (GHA) | Rejected |
| EXP-008 | LGBM+XGB+CatBoost probability blend | 0.96227 (GHA) | Rejected |
| EXP-009 | 5-seed LightGBM ensemble | 0.96330 (GHA) | Inconclusive+ (below margin, best score) |
| EXP-010 | Legacy champion faithful reproduction | 0.96313 (GHA) | Promoted as verified baseline |

## Important Discoveries

- **Legacy champion number was inflated**: true reproducible value 0.96313, −0.00108 off.
- **Same-family seed diversity decorrelates MORE than family switching**:
  seed~seed corr ≈ 0.988 < LGBM~XGB 0.995. This is why seed averaging helped
  (+0.0002) while cross-family blending hurt (−0.0009).
- CatBoost substantially weaker here (0.95681); XGBoost redundant vs LightGBM.
- **Missing-value indicators are dead** (EDA): target|missing ≈ target|known everywhere.
- Signal concentrates in `daily_screen_time_hours`, `weekend_screen_time`,
  `social_media_hours` — strongly monotone, saturating near p=1.0 in top deciles.
- Stratified 5-fold CV stable: fold std ~0.0006–0.0008 across runs.
- All scores must come from GHA runs. Local results void.

## Rejected Approaches

- Local experiment execution (unverifiable).
- Engineered screen-time ratio/interaction features (EXP-007).
- Cross-family GBDT blending (EXP-008): correlations saturated, members weaker.
- Missingness indicators (EDA-eliminated).

## Current Research Priorities

1. **EXP-011:** do bagging and seed-averaging combine additively?
2. **EXP-012:** does doubling capacity break the plateau?
3. Next levers (evidence-dependent):
   - Capacity works → probe num_leaves 255 / min_child_samples 80.
   - Plateau holds → add a decorrelated **linear member** (logistic regression)
     to the stack; trees all correlate ≥0.98, a linear boundary may differ.
   - Seed count scaling: 10-seed average if seed lever keeps paying.

## Agent Activity

Orchestrator active: EXP-007..012 queued/executed this session; runner supports
ensembles, blends, per-member diagnostics; matrix parallelism + rebase-safe commits.

## Compute Status

- **GitHub Actions:** ACTIVE — parallel matrix runs proven; ~4–12 min per experiment;
  results auto-committed with rebase safety; sync-results skips without creds.
- Local: EDA/smoke only.

## Outstanding Questions

- Additivity of bagging × seeds (EXP-011)?
- Is the plateau capacity-limited or information-limited? (EXP-012 answers directionally.)
- Can a linear model decorrelate enough from trees to earn a stack slot?

## Recommended Next Actions

1. On EXP-011/012 completion: record decisions, update best-score table.
2. If plateau persists: implement `logistic` member in runner; queue stacked run.
3. If capacity pays: queue num_leaves 255 probe.
