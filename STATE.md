# ModelSwarm Research State

> Last updated: 2026-08-25 (post EXP-027; configuration frontier reached)
> Current competition: Kaggle Playground Series S6E8
> Compute policy: **ALL experiments run on GitHub Actions** (`.github/workflows/experiment-runner.yml`). Local ML runs are prohibited — see `AGENT_INSTRUCTIONS.md`.

## Best Results

| ID | Description | OOF ROC-AUC | Status |
|----|-------------|-------------|--------|
| **EXP-026** | 5-seed LGBM nl255 / lr 0.02 / 3000t / bagged / **max_bin 2048** | **0.966155** | **CHAMPION** |
| EXP-024 | same, max_bin 1024 | 0.966016 | Verified (superseded) |
| EXP-022 | same, max_bin 511 | 0.965203 | Verified (superseded) |
| EXP-013 | max_bin default (255) | 0.964109 | Verified (superseded) |
| EXP-010 | Legacy champion config reproduction nl64 | 0.963127 | Verified reference baseline |

## Active Experiments

None queued. The measured configuration space is exhausted at current margins;
new work requires a genuinely new hypothesis (see "Open Ideas").

## Champion Lineage (all GHA-verified)

| Stage | ID | Config delta | OOF | Gain |
|-------|----|--------------|-----|------|
| Baseline | EXP-010 | legacy config reproduced (nl64, unbagged, seed42) | 0.963127 | — |
| +seeds+bagging | EXP-011 | bagged 5-seed average (subsample_freq 1) | 0.963664 | +0.00054 * |
| +capacity | EXP-013 | num_leaves 255, lr 0.02, 3000t | 0.964109 | +0.00007 |
| +**bins** | EXP-022 | max_bin 511 | 0.965203 | +0.00109 |
| +**bins²** | EXP-024 | max_bin 1024 | 0.966016 | +0.00081 |
| +**bins³** | **EXP-026** | **max_bin 2048** | **0.966155** | +0.00014 |

\* Conflated step: bagging and seed-averaging were introduced together. Sub-attribution
from the fixed-runner measurement pair: single bagged seed ≈ 0.963303 (EXP-009 v1,
see Measurement Incidents) → 5-seed average 0.963664 ⇒ seed averaging ≈ +0.00036;
residual +0.00018 attributable to bagging, though no clean isolated A/B exists.
(EXP-012's nl127 run also sits in this ladder at 0.964038.)

Total verified progress this session: **+0.00303 over the re-baselined start**
(0.963127 → 0.966155). The single largest lever was histogram resolution
(max_bin), worth more than everything else combined.

## Measurement Incidents (documented for integrity)

1. **Degenerate-blend bug (fixed)**: before member-keying was added, ensemble
   members sharing a name collided in the blend dict, so a "5-seed average"
   silently recorded only the last member. Detected by forensics on EXP-009's
   collapsed record (blend folds == last member's folds). All affected results
   superseded by post-fix reruns.
2. **Rerun-wiped decisions (fixed)**: bundling decision edits into queue pushes
   triggered accidental matrix reruns whose `record_results` nullified five
   records' decisions. Guard added: `record_results.py` now preserves existing
   non-null decision/reasoning and sets truthful top-level status.
3. **Config-drift copy**: an accidental rerun briefly made EXP-009.yaml a
   duplicate of EXP-011 (identical scores exposed it). Restored from git history
   with a truthful dual-run record.

## Champion Configuration (EXP-026)

```yaml
model: {name: ensemble, blend: probability_average}
members:  # ×5, random_state ∈ {42, 123, 7, 2024, 99}
  - lightgbm:
      n_estimators: 3000, learning_rate: 0.02, num_leaves: 255,
      min_child_samples: 80, subsample: 0.8, subsample_freq: 1,
      colsample_bytree: 0.7, reg_alpha: 0.1, reg_lambda: 0.1,
      max_bin: 2048, early_stopping_rounds: 150
features: [age, daily_screen_time_hours, social_media_hours, gaming_hours,
           work_study_hours, sleep_hours, notifications_per_day,
           app_opens_per_day, weekend_screen_time,
           gender, stress_level, academic_work_impact]   # raw canonical set only
validation: stratified 5-fold, seed 42
```

Submission artifacts per experiment live in Actions artifacts
(`exp-<ID>-artifacts`: `submission.csv`, `oof_predictions.csv`, `results.json`).

## Recently Completed Experiments

| ID | Description | OOF ROC-AUC | Decision |
|----|-------------|-------------|----------|
| EXP-007 | LightGBM + engineered ratios | 0.962917 | Rejected |
| EXP-008 | LGBM+XGB+CatBoost blend | 0.962273 | Rejected |
| EXP-009 | Bagged 5-seed nl64 (authoritative v2 run) | 0.963664 | Inconclusive+ |
| EXP-010 | Legacy champion reproduction | 0.963127 | Promoted as verified baseline |
| EXP-011 | Bagged 5-seed nl64 | 0.963664 | Promoted (superseded) |
| EXP-012 | nl127 | 0.964038 | Promoted (superseded) |
| EXP-013 | nl255 | 0.964109 | Promoted (superseded) |
| EXP-014 | nl96 midpoint | 0.963986 | Rejected |
| EXP-015 | 10-seed nl127 | 0.964092 | Rejected (seed scaling exhausted) |
| EXP-016 | Weighted linear stack | 0.961869 | Rejected |
| EXP-017 | DART boosting | 0.961464 | Rejected |
| EXP-018 | GOSS boosting | 0.963781 | Rejected |
| EXP-019/20/21 | HP micro-grid @nl255 | 0.96392–0.96412 | Rejected (flat optimum) |
| EXP-022 | **max_bin 511** | 0.965203 | Promoted (superseded) |
| EXP-023 | Pseudo-labeling | 0.963847 | Rejected |
| EXP-024 | max_bin 1024 | 0.966016 | Promoted (superseded) |
| EXP-025 | nl511 @ 511 bins | 0.964897 | Rejected |
| EXP-026 | **max_bin 2048** | **0.966155** | **Promoted — champion** |
| EXP-027 | nl511 @ 1024 bins | 0.965673 | Rejected |

## Lever Map (complete)

| Lever | Verdict | Evidence |
|-------|---------|----------|
| **Histogram resolution (max_bin)** | **STRONGEST POSITIVE (+0.0020 total)** | EXP-022/024/026 ladder |
| Seed averaging (→5) | Positive (~+0.00036) | single bagged seed 0.963303 → 5-seed avg 0.963664 |
| Seeds+bagging combined | Positive (+0.00054 vs baseline) | EXP-010 → EXP-011 (sub-levers not separable) |
| Leaf capacity to 255 | Positive (+0.0004), ceiling robust | EXP-012/013; 025/027 confirm ceiling |
| Micro-grid mcs/col/L2 | FLAT (±0.0002) | EXP-019/020/021 |
| Seed scaling (→10) | Exhausted (+0.00005) | EXP-015 |
| Bin scaling (→4096) | Projected ≤ +0.00005; not run-worthiness | curve asymptote |
| GOSS / DART boosting | Dead (−0.0003 / −0.0026) | EXP-018 / 017 |
| Pseudo-labeling | Dead (−0.0003) | EXP-023 |
| Cross-family GBDT blend | Dead (−0.0009) | EXP-008 |
| Linear stack | Dead (−0.0022) | EXP-016 |
| Engineered ratios | Dead (−0.0013) | EXP-007 |
| Missingness indicators | Dead (EDA) | no run needed |

Key structural findings:
- LightGBM's default `max_bin=255` was severely underspecifying split granularity
  for these continuous features — worth more than everything else combined.
- The leaf-capacity ceiling (~255) is robust across bin resolutions.
- Boosting-mode ranking: gbdt > goss > dart.
- Train/test distributions verified clean (KS p≥0.14 all numerics) — CV tracks LB.
- All scores GHA-verified; legacy-era numbers void.

## Infrastructure (built this session)

- Runner: ensembles with N members, probability/rank blending (+family-name weights),
  keyed per-member diagnostics & correlations, logistic member type,
  leak-free per-fold pseudo-labeling, smoke-test flags.
- Workflow: matrix parallelism, rebase-safe result commits, dispatch-input support,
  data validation gate, guarded API sync.
- Data committed under `competitions/s6e8/data/`.

## Open Ideas (require new evidence before queueing)

1. max_bin 4096 — projected ≤ +0.00005 by asymptote; only if compute is free.
2. Stacking meta-model over stored OOF artifacts of diverse configs — members
   correlate ≥0.995, low expected value; requires artifact-collection plumbing.
3. External original-dataset merge — PROHIBITED by competition constraints.
4. Any new lever must beat +0.0005 to justify a promotion claim per rules.

## Agent Activity

Orchestrator session: EXP-007..027 executed (21 GHA experiment runs), champion advanced
0.963127 → 0.966155 through seven verified promotions, every result recorded with
decision + reasoning in its experiment YAML.

## Recommended Next Actions

1. Submit champion EXP-026's artifact submission.csv to Kaggle; record public score
   in STATE.md when available (validates CV↔LB transfer prediction).
2. If public score diverges from OOF materially, revisit adversarial assumptions.
3. New experiments only for hypotheses projecting > +0.0005 (per integrity rules).
