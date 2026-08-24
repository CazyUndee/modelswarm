# ModelSwarm Research State

> Last updated: 2026-08-24 (EXP-012 champion; EXP-013/014 in flight)
> Current competition: Kaggle Playground Series S6E8
> Compute policy: **ALL experiments run on GitHub Actions** (`.github/workflows/experiment-runner.yml`). Local ML runs are prohibited — see `AGENT_INSTRUCTIONS.md`.

## Best Results

| ID | Description | OOF ROC-AUC | Status |
|----|-------------|-------------|--------|
| **EXP-022** | 5-seed LGBM nl255 + **max_bin 511** | **0.965203** | **CHAMPION** |
| EXP-013 | 5-seed LGBM nl255 (max_bin default 255) | 0.964109 | Verified (superseded) |
| EXP-015 | 10-seed nl127 | 0.964092 | Verified |
| EXP-012 | 5-seed nl127 | 0.964038 | Verified |
| EXP-010 | Legacy champion config reproduction | 0.963127 | Verified reference baseline |

## Active Experiments

| ID | Description | Status |
|----|-------------|--------|
| EXP-024 | max_bin 511 → 1024 continuation | Running on GitHub Actions |
| EXP-025 | Capacity re-probe nl511/mcs120 @ max_bin 511 | Running on GitHub Actions |

## Champion Lineage (all GHA-verified)

| Stage | ID | Config delta | OOF | Gain |
|-------|----|--------------|-----|------|
| Baseline | EXP-010 | legacy config reproduced (nl64, bagging off) | 0.963127 | — |
| +seeds | EXP-009 | 5-seed average | 0.963303 | +0.00018 |
| +bagging | EXP-011 | subsample_freq 1 | 0.963664 | +0.00036 |
| +capacity | EXP-012 | num_leaves 127, lr 0.03, 2000t | 0.964038 | +0.00037 |
| +capacity² | EXP-013 | num_leaves 255, lr 0.02, 3000t | 0.964109 | +0.00007 |
| +**bin resolution** | **EXP-022** | **max_bin 511** | **0.965203** | **+0.00109** |

Total verified progress this session: **+0.00208 over the re-baselined start.**
max_bin was the single largest lever discovered — the continuous screen-time
features need finer split granularity than LightGBM's default 255 bins.
EXP-024 (1024 bins) and EXP-025 (nl511 at high bins) probe whether the
(bin-resolution × capacity) surface has more gradient left.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-007 | LightGBM + engineered ratios (tuned) | 0.962917 | Rejected |
| EXP-008 | LGBM+XGB+CatBoost probability blend | 0.962273 | Rejected |
| EXP-009 | 5-seed ensemble nl64 unbagged | 0.963303 | Inconclusive+ |
| EXP-010 | Legacy champion reproduction nl64 | 0.963127 | Promoted as verified baseline |
| EXP-011 | Bagged 5-seed ensemble nl64 | 0.963664 | Promoted (superseded) |
| EXP-012 | Capacity probe nl127 | 0.964038 | Promoted (superseded) |
| EXP-013 | Capacity probe nl255 | 0.964109 | Promoted (superseded by 022) |
| EXP-014 | Capacity midpoint nl96 | 0.963986 | Rejected |
| EXP-015 | 10-seed nl127 | 0.964092 | Rejected (seed scaling exhausted) |
| EXP-016 | Weighted linear stack | 0.961869 | Rejected |
| EXP-017 | DART boosting nl127 | 0.961464 | Rejected |
| EXP-018 | GOSS boosting nl255 | 0.963781 | Rejected |
| EXP-019 | micro-grid mcs→20 @nl255 | 0.964064 | Rejected |
| EXP-020 | micro-grid colsample→0.8 @nl255 | 0.963921 | Rejected |
| EXP-021 | micro-grid reg_lambda→1.0 @nl255 | 0.964115 | Rejected (statistical tie) |
| EXP-022 | **max_bin 511 @nl255** | **0.965203** | **Promoted — current champion** |
| EXP-023 | Pseudo-labeling (self-training) | 0.963847 | Rejected |

## Levers Status

| Lever | Verdict | Evidence |
|-------|---------|----------|
| **Histogram resolution (max_bin)** | **STRONG POSITIVE (+0.0011)** | EXP-022 vs 013 |
| Seed averaging (→5) | POSITIVE (+0.0002) | EXP-009 vs 010 |
| Row bagging | POSITIVE (+0.0004) | EXP-011 vs 009 |
| Leaf capacity to 255 | POSITIVE, saturating (+0.0004) | EXP-012/013 curve |
| Micro-grid around optimum (mcs/col/L2) | FLAT (±0.0002) | EXP-019/020/021 |
| Seed scaling (→10) | EXHAUSTED (+0.00005) | EXP-015 |
| GOSS boosting | DEAD (−0.0003) | EXP-018 |
| DART boosting | DEAD (−0.0026) | EXP-017 |
| Pseudo-labeling | DEAD (−0.0003) | EXP-023 |
| Cross-family GBDT blend | DEAD (−0.0009) | EXP-008 |
| Linear/logistic stack | DEAD (−0.0022) | EXP-016 |
| Engineered ratios | DEAD (−0.0013) | EXP-007 |
| Missingness indicators | DEAD (EDA) | no run needed |

Boosting-mode ranking: **gbdt > goss > dart**.
Train/test distributions verified clean (KS p≥0.14 all numerics) — CV tracks LB.

## Champion Lineage (all GHA-verified)

| Stage | ID | Config delta | OOF | Gain |
|-------|----|--------------|-----|------|
| Baseline | EXP-010 | legacy config reproduced (nl64, bagging off) | 0.963127 | — |
| +seeds | EXP-009 | 5-seed average | 0.963303 | +0.00018 |
| +bagging | EXP-011 | subsample_freq 1 | 0.963664 | +0.00036 |
| +capacity | **EXP-012** | num_leaves 127, lr 0.03, 2000t | **0.964038** | +0.00037 |

Total verified progress this session: **+0.00091 over the re-baselined start.**
The early plateau was capacity-limited: each lever measured separately, then composed.
EXP-013/014 map the capacity curve's peak (64 → 96 → 127 → 255 leaves).

## Result Integrity Notices

1. **Legacy era (pre-2026-08-24) void**: EXP-000..006 records came from unverifiable
   local runs; the claimed champion 0.96421 is not reproducible (true value 0.96313,
   verified by EXP-010). All legacy numbers are superseded.
2. **All authoritative scores must come from GitHub Actions runs.** Local results void.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-007 | LightGBM + engineered ratios (tuned) | 0.962917 | Rejected |
| EXP-008 | LGBM+XGB+CatBoost probability blend | 0.962273 | Rejected |
| EXP-009 | 5-seed ensemble nl64 unbagged | 0.963303 | Inconclusive+ |
| EXP-010 | Legacy champion reproduction nl64 | 0.963127 | Promoted as verified baseline |
| EXP-011 | Bagged 5-seed ensemble nl64 | 0.963664 | Promoted (superseded) |
| EXP-012 | Capacity probe nl127 | 0.964038 | Promoted (superseded) |
| EXP-013 | Capacity probe nl255 | **0.964109** | **Promoted — current champion** |
| EXP-014 | Capacity midpoint nl96 | 0.963986 | Rejected (below champion) |
| EXP-015 | 10-seed nl127 | 0.964092 | Rejected (seed scaling exhausted) |
| EXP-016 | Weighted linear stack (LGBM + logistic) | 0.961869 | Rejected — quality gap > decorrelation |
| EXP-017 | DART boosting nl127 | 0.961464 | Rejected — high variance, no accuracy |
| EXP-018 | GOSS boosting nl255 | 0.963781 | Rejected — systematically below GBDT |

## Capacity Curve (all bagged, 5-seed blends)

| num_leaves | OOF | lr | trees |
|-----------|------|-----|-------|
| 64 | 0.963664 | 0.05 | 1000 |
| 96 | 0.963986 | 0.04 | 1500 |
| 127 | 0.964038 | 0.03 | 2000 |
| 255 | **0.964109** | 0.02 | 3000 |

Rising but flattened (+0.00007 last step). Deeper leaf-doubling deprioritized;
remaining capacity upside < remaining lever upside elsewhere.

## Levers Status

| Lever | Verdict | Evidence |
|-------|---------|----------|
| Seed averaging (→5) | POSITIVE (+0.0002) | EXP-009 vs 010 |
| Row bagging | POSITIVE (+0.0004) | EXP-011 vs 009 |
| Leaf capacity to 255 | POSITIVE, saturating (+0.0004 total) | EXP-012/013 curve |
| Seed scaling (→10) | EXHAUSTED (+0.00005) | EXP-015 |
| GOSS boosting | DEAD (−0.0003) | EXP-018 |
| DART boosting | DEAD (−0.0026) | EXP-017 |
| Cross-family GBDT blend | DEAD (−0.0009) | EXP-008 |
| Linear/logistic stack | DEAD (−0.0022) | EXP-016 (gap 0.05 AUC unbeatable by corr 0.84) |
| Engineered ratios | DEAD (−0.0013) | EXP-007 |
| Missingness indicators | DEAD (EDA) | no run needed |

Boosting-mode ranking: **gbdt > goss > dart**.
Train/test distributions verified clean (KS p≥0.14 all numerics; categorical
drift ≤2.3% on a zero-signal feature) — CV OOF should track LB faithfully.

## Micro-grid In Flight

Single-variable A/Bs vs champion EXP-013 (nl255/lr0.02/3000t/mcs80/col0.7/L2 0.1):
- EXP-019: min_child_samples 80→20 (relax leaf regularization)
- EXP-020: colsample_bytree 0.7→0.8 (isolate EXP-013's colsample change)
- EXP-021: reg_lambda 0.1→1.0 (stronger L2)

## Important Discoveries

- **Levers are additive and measurable**: seeds (+0.0002) × bagging (+0.0004) × capacity (+0.0004).
- Same-family seed diversity decorrelates more than family switching at small capacity
  (~0.988 vs LGBM~XGB 0.995); at nl127 bagged, within-family corr ~0.997.
- CatBoost weak here (0.95681); XGBoost redundant vs LightGBM (corr 0.995).
- Missing-value indicators dead (EDA): target|missing ≈ target|known for every feature.
- Signal concentrates in `daily_screen_time_hours`, `weekend_screen_time`,
  `social_media_hours` — strongly monotone, saturating near p=1.0 in top deciles.
- Logistic member implemented and smoke-tested: corr vs LGBM only ~0.92 — real
  decorrelation; needs weighting/stacking design to overcome the quality gap.
- Stratified 5-fold CV stable: fold std ~0.0006 across runs.

## Rejected Approaches

- Local experiment execution; engineered screen-time ratios (EXP-007);
  cross-family GBDT blending (EXP-008); missingness indicators (EDA-eliminated).

## Current Research Priorities

1. **MAP the capacity curve**: EXP-013 (nl255) + EXP-014 (nl96) in flight.
2. If nl255 ≥ nl127: continue doubling with matched regularization.
3. If peaked: micro-grid around the optimum via parallel matrix experiments.
4. Seed-count scaling: 10-seed variant of the winning config.
5. Stacking track: weighted logistic member to exploit the 0.92 correlation.

## Agent Activity

Orchestrator active: EXP-007..014 queued/executed this session.
Runner capabilities built along the way: ensembles, blends (+weights), keyed per-member
diagnostics, logistic member type, matrix parallelism, rebase-safe result commits.

## Compute Status

- GitHub Actions ACTIVE: 2–3 parallel experiment jobs per push; ~5–15 min each.
- Local: EDA/smoke tests only (`catboost_info/` gitignored).

## Outstanding Questions

- Where does the leaf-capacity curve peak? (EXP-013/014)
- Do gains persist when averaging MORE seeds (10) at the optimum?
- Can a weighted linear member add stack value despite the AUC gap?

## Recommended Next Actions

1. On EXP-013/014 landing: record decisions; locate curve peak; queue next step.
2. Queue 10-seed variant of the best config.
3. Design weighted-stack run if the capacity curve flattens.
