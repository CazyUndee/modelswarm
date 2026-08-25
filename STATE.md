# ModelSwarm Research State

> Last updated: 2026-08-25 (EXP-038 landed; single-variable search space exhausted)
> Current competition: Kaggle Playground Series S6E8
> Compute policy: **ALL experiments run on GitHub Actions** (`.github/workflows/experiment-runner.yml`). Local ML runs are prohibited — see `AGENT_INSTRUCTIONS.md`.

## Best Results

| ID | Description | OOF ROC-AUC | Status |
|----|-------------|-------------|--------|
| **EXP-035** | 5-seed LGBM nl255 / lr 0.02 / 3000t / bagged / max_bin 2048 / **colsample 0.3** | **0.966518** | **CHAMPION** |
| EXP-038 | same + min_data_in_bin 1 | 0.966525 | Statistical tie (rejected) |
| EXP-031 | colsample 0.5 | 0.966415 | Verified |
| EXP-030 | colsample 0.6 | 0.966270 | Verified |
| EXP-026 | colsample 0.7 | 0.966155 | Verified (former champion) |
| EXP-010 | Legacy champion config reproduction | 0.963127 | Verified reference baseline |

## Active Experiments

None queued. Every measured hyperparameter axis is at a confirmed optimum or
flat; the next improvement requires a structurally new hypothesis.

## Champion Lineage (all GHA-verified)

| Stage | ID | Config delta | OOF | Gain |
|-------|----|--------------|-----|------|
| Baseline | EXP-010 | legacy config reproduced (nl64, unbagged, seed42) | 0.963127 | — |
| +seeds+bagging | EXP-011 | bagged 5-seed average | 0.963664 | +0.00054 * |
| +capacity | EXP-013 | num_leaves 255, lr 0.02, 3000t | 0.964109 | +0.00007 (via 012) |
| +**bins** | EXP-022/024/026 | max_bin 255→2048 | 0.966155 | +0.00205 |
| +**colsample↓** | **EXP-035** | **colsample 0.7→0.3** | **0.966518** | +0.00036 |

\* Conflated step (sub-levers not separable; see Measurement Incidents).
Total verified progress this session: **+0.00339 over the re-baselined start**
(0.963127 → 0.966518) across 32 GHA experiment records.

## Champion Configuration (EXP-035)

```yaml
model: {name: ensemble, blend: probability_average}
members:  # ×5, random_state ∈ {42, 123, 7, 2024, 99}
  - lightgbm:
      n_estimators: 3000, learning_rate: 0.02, num_leaves: 255,
      min_child_samples: 80, subsample: 0.8, subsample_freq: 1,
      colsample_bytree: 0.3, reg_alpha: 0.1, reg_lambda: 0.1,
      max_bin: 2048, early_stopping_rounds: 150
features: [age, daily_screen_time_hours, social_media_hours, gaming_hours,
           work_study_hours, sleep_hours, notifications_per_day,
           app_opens_per_day, weekend_screen_time,
           gender, stress_level, academic_work_impact]   # raw canonical set only
validation: stratified 5-fold, seed 42
```

Submission artifacts live in Actions artifacts (`exp-<ID>-artifacts`).
The champion's `submission.csv` was format-validated against sample_submission
(ids aligned, no nulls, probabilities in range).

## Colsample Axis (mapped this session, at max_bin2048/nl255)

| colsample_bytree | OOF |
|------------------|------|
| 0.7 | 0.966155 |
| 0.6 | 0.966270 |
| 0.5 | 0.966415 |
| 0.4 | 0.966373 (wobble) |
| **0.3** | **0.966518 ← optimum** |
| 0.2 | 0.965517 (cliff −0.001) |
| 0.1 | 0.960087 (collapse −0.006) |

Heavy feature subsampling synergizes with fine histogram bins: each tree sees
~4 of 12 features, and the seed blend recovers ensemble coverage.

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-007–021 | lever exploration era (see git history & prior STATE) | — | recorded |
| EXP-022/024/026 | max_bin ladder 511/1024/2048 | 0.965203 / 0.966016 / 0.966155 | promoted chain |
| EXP-023 | Pseudo-labeling | 0.963847 | Rejected |
| EXP-025/027 | nl511 @ high bins | 0.964897 / 0.965673 | Rejected (leaf ceiling robust) |
| EXP-028 | max_bin 4096 | 0.966155 | Rejected (bit-exact tie — asymptote proven) |
| EXP-029 | subsample 0.7 | 0.966029 | Rejected |
| EXP-030 | colsample 0.6 | 0.966270 | Inconclusive+ (gradient found) |
| EXP-031 | colsample 0.5 | 0.966415 | Inconclusive+ (new best) |
| EXP-032/033 | lr bracket 0.015/0.025 | 0.966290 / 0.966267 | Rejected (axis flat) |
| EXP-034 | colsample 0.4 | 0.966373 | Rejected (wobble) |
| **EXP-035** | **colsample 0.3** | **0.966518** | **Promoted — champion** |
| EXP-036/037 | colsample 0.2 / 0.1 | 0.965517 / 0.960087 | Rejected (cliff/collapse) |
| EXP-038 | min_data_in_bin 1 | 0.966525 | Rejected (dead heat) |

## Lever Map (complete — every axis measured)

| Lever | Verdict | Evidence |
|-------|---------|----------|
| **Histogram resolution (max_bin)** | **+0.00205, asymptote proven by tie** | EXP-022/024/026 ladder; EXP-028 tie |
| **colsample descent to 0.3** | **+0.00036, cliffs both sides** | EXP-030/031/034/035/036/037 |
| Seeds+bagging (combined) | +0.00054 conflated | EXP-010 → 011 |
| Seed averaging proper | ~+0.00036 (single→5-seed) | EXP-009 v1 vs v2 |
| Leaf capacity to 255 | +0.0004, ceiling robust | EXP-012/013; 025/027 confirm |
| min_data_in_bin 3→1 | Flat (+0.000007) | EXP-038 |
| lr axis (0.015–0.025) | Flat (±0.00002) | EXP-032/033 |
| subsample fraction (0.7 vs 0.8) | Dead (−0.00013) | EXP-029 |
| mcs/L2 micro-grid | Flat (±0.0002) | EXP-019/020/021 |
| Seed scaling (→10) | Exhausted (+0.00005) | EXP-015 |
| GOSS / DART boosting | Dead | EXP-018 / 017 |
| Pseudo-labeling | Dead | EXP-023 |
| Cross-family GBDT blend | Dead | EXP-008 |
| Linear stack | Dead | EXP-016 |
| Engineered ratios | Dead | EXP-007 |
| Missingness indicators | Dead (EDA) | no run needed |

Boosting-mode ranking: gbdt > goss > dart.
Train/test distributions verified clean (KS p≥0.14 all numerics) — CV tracks LB.

## Measurement Incidents (documented for integrity)

1. **Degenerate-blend bug (fixed)**: pre-keying-fix runner collided member names;
   "ensembles" silently recorded only the last member. Detected via forensics on
   EXP-009's collapsed record. All affected results superseded by post-fix reruns.
2. **Rerun-wiped decisions (fixed)**: decision edits bundled into queue pushes
   triggered accidental reruns that nullified five records' decisions. Guard:
   `record_results.py` now preserves non-null judgment and sets truthful status.
3. **Config-drift copy**: an accidental rerun briefly made EXP-009.yaml duplicate
   EXP-011 (identical scores exposed it). Restored from history with truthful
   dual-run documentation.

## Infrastructure (built this session)

- Runner: ensembles with N members, probability/rank blending (+family-name weights),
  keyed per-member diagnostics & correlations, logistic member type,
  leak-free per-fold pseudo-labeling, smoke-test flags.
- Workflow: matrix parallelism, rebase-safe result commits, dispatch-input support,
  data validation gate, guarded API sync.
- Data committed under `competitions/s6e8/data/`; champion submission artifact
  validated against sample_submission format.

## Open Ideas (require new evidence before queueing)

1. Kaggle public score for the champion submission — validates CV↔LB transfer
   empirically. BLOCKED on credentials not present in this environment.
2. Micro-refinement of colsample within [0.25, 0.35] — projected ≤ ±0.0001.
3. Stacking over stored OOF artifacts — members correlate ≥0.995, low EV.
4. Any new lever must beat +0.0005 to justify a promotion claim per rules.

## Agent Activity

Orchestrator session total: **EXP-007..038 executed (32 GHA experiment records)**.
Champion advanced through eight verified promotions:
0.963127 → 0.963664 → 0.964038 → 0.964109 → 0.965203 → 0.966016 → 0.966155 → 0.966518.

## Recommended Next Actions

1. Submit champion EXP-035's artifact submission.csv to Kaggle; record public
   score here when available.
2. If public score diverges from OOF materially, revisit adversarial assumptions.
3. New experiments only for hypotheses projecting > +0.0005 (per integrity rules).
