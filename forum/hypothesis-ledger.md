# ModelSwarm Hypothesis Ledger — S6E8

> Living document. Every hypothesis carries: motivation, evidence, expected effect,
> experiment, result, conclusion, and follow-up hypotheses. Rejected hypotheses are
> kept as evidence — re-test only if new evidence changes their assumptions.
>
> Champion at last update: **EXP-035 @ OOF 0.966518** (nl255 / max_bin 2048 / lr 0.02 /
> 3000t / bagged 5-seed / colsample 0.3). Baseline: EXP-010 @ 0.963127.

---

## L1. Histogram resolution drives split quality
- **Hypothesis:** LightGBM's default `max_bin=255` under-resolves splits on continuous features; raising it improves AUC.
- **Motivation:** Decile EDA showed continuous screen-time features with strong monotone-saturating signal; coarse bins blur split points.
- **Evidence:** Bin ladder monotone across four steps.
- **Expected effect:** +0.001–0.002 (uncertain at outset).
- **Experiment:** EXP-022 (511) → EXP-024 (1024) → EXP-026 (2048) → EXP-028 (4096).
- **Result:** 0.964109 → 0.965203 → 0.966016 → 0.966155 → 0.966155 (bit-exact tie).
- **Conclusion:** CONFIRMED, +0.00205 total; hard asymptote at 2048 bins (4096 tie proves it).
- **Follow-ups:** closed. Re-open only if feature set changes.

## L2. Feature subsampling (colsample_bytree) optimum far below default
- **Hypothesis:** At high bin resolution, aggressive per-tree feature restriction regularizes better than colsample 0.7–0.8.
- **Motivation:** EXP-020 showed 0.8 worse than 0.7 at nl255/bins255; gradient direction suggested probing lower once bins improved.
- **Evidence:** Monotone descent with mapped cliffs.
- **Expected effect:** +0.0001–0.0005.
- **Experiment:** EXP-030 (0.6) → EXP-031 (0.5) → EXP-034 (0.4) → EXP-035 (0.3) → EXP-036 (0.2) → EXP-037 (0.1).
- **Result:** 0.966270 → 0.966415 → 0.966373 → **0.966518** → 0.965517 → 0.960087.
- **Conclusion:** CONFIRMED, optimum at col 0.3 (+0.00036 over 0.7); cliff −0.001 below 0.3, collapse −0.006 at 0.1. Heavy feature subsampling synergizes with fine bins.
- **Follow-ups:** micro-refinement [0.25, 0.35] projected ≤ ±0.0001 — low priority.

## L3. Capacity (num_leaves) ceiling near 255
- **Hypothesis:** More leaves improve fit until overfitting dominates.
- **Motivation:** Standard GBDT scaling path.
- **Evidence:** 64→96→127 rising; 255→+0.00007 flattening; nl511 tested twice at high bins, both worse.
- **Expected effect:** +0.0005 hoped; delivered less.
- **Experiment:** EXP-012/013 (ladder); EXP-025/027 (nl511 re-probe at high bins, pre-colsample-descent).
- **Result:** 0.964038 (nl127) → 0.964109 (nl255); nl511 = 0.964897@511bins / 0.965673@1024bins — both below their nl255 comparators.
- **Conclusion:** CONFIRMED with ceiling at ~255. CAVEAT: both nl511 tests ran at col 0.7 (pre-descent); interaction with col 0.3 untested until EXP-040.
- **Follow-ups:** EXP-040 re-tests nl511 at the current optimum config.

## L4. Seed averaging + row bagging help (conflated step)
- **Hypothesis:** Bagged 5-seed averaging beats single-seed training.
- **Motivation:** Variance reduction is the cheapest ensemble gain.
- **Evidence:** EXP-010 (single seed42 unbagged) 0.963127 vs EXP-011 (bagged 5-seed avg) 0.963664.
- **Expected effect:** +0.0002–0.0005.
- **Experiment:** EXP-009 v1/v2 pair + EXP-011.
- **Result:** +0.00054 combined. Sub-levers: seed averaging alone ≈ +0.00036 (single bagged seed 0.963303 → 5-seed avg, measured via fixed-runner rerun pair); bagging-in-isolation never cleanly isolated (EXP-009 was created bagged by mistake).
- **Conclusion:** CONFIRMED as a conflated +0.00054 step.
- **Follow-ups:** none — decomposition would need a purpose-built A/B of marginal value.

## L5. Seed count beyond 5
- **Hypothesis:** Doubling seeds keeps paying.
- **Experiment:** EXP-015 (10-seed nl127).
- **Result:** +0.00005 vs 5-seed identical config.
- **Conclusion:** REJECTED — member std ~0.00004; averaging already converged at 5 seeds.

## L6. Boosting-mode alternatives
- **Hypothesis:** DART or GOSS beating standard gbdt.
- **Experiment:** EXP-017 (DART nl127), EXP-018 (GOSS nl255).
- **Result:** DART 0.961464 (−0.0026, high seed variance); GOSS 0.963781 (−0.0003).
- **Conclusion:** BOTH REJECTED. Ranking: gbdt > goss > dart.

## L7. Cross-model blending / stacking
- **Hypothesis:** Blending XGBoost/CatBoost with LightGBM exploits family decorrelation; linear meta-members exploit class decorrelation.
- **Experiment:** EXP-008 (prob blend of 3 families), EXP-016 (weighted stack with logistic members).
- **Result:** Blend 0.962273 (−0.0009); stack 0.961869 (−0.0022). Correlations: LGBM~XGB 0.995, logistic~LGBM 0.84 but quality gap ~0.05 unbeatable.
- **Conclusion:** REJECTED for this dataset/model mix. CatBoost structurally weak here (~0.9568).
- **Follow-ups:** only revisit if a genuinely diverse member reaches ≥0.965 standalone.

## L8. Engineered ratio/interaction features
- **Hypothesis:** Derived ratios (social/screen, gaming/screen etc.) add signal over raw features.
- **Experiment:** EXP-007 (9 engineered features, tuned LGBM).
- **Result:** 0.962917 vs champion-era 0.964109 — no gain.
- **Conclusion:** REJECTED. Trees capture these interactions natively; monotone transforms of raw features add nothing.
- **Follow-ups:** none unless a non-tree model enters the pool.

## L9. Missing-value indicators
- **Hypothesis:** Missingness patterns carry target signal.
- **Evidence:** EDA — target|missing ≈ target|known for every feature (max Δ +0.004).
- **Conclusion:** REJECTED without a run. No experiment spent.

## L10. Pseudo-labeling (self-training)
- **Hypothesis:** Confident test predictions (p≤0.01 / p≥0.99, w=0.3) add real signal from 296k unlabeled rows.
- **Experiment:** EXP-023 (leak-free per-fold design).
- **Result:** 0.963847 (−0.0003 vs then-champion era).
- **Conclusion:** REJECTED — consistent with near-saturating features where confident predictions are already correct. Infrastructure retained.

## L11. Micro-grid around optimum (mcs / reg_lambda)
- **Hypothesis:** Regularization sensitivity exists near the capacity optimum.
- **Experiment:** EXP-019 (mcs→20), EXP-021 (reg_lambda→1.0).
- **Result:** −0.00005 / +0.000006 — statistical zeros.
- **Conclusion:** FLAT. CAVEAT: ran at bins255-era config; interaction with max_bin2048 untested. Low priority to re-test.

## L12. lr sensitivity at fixed capacity
- **Hypothesis:** lr ±25% moves the result.
- **Experiment:** EXP-032 (0.015/4000t), EXP-033 (0.025/2500t) @ col 0.6.
- **Result:** 0.966290 / 0.966267 vs 0.966270 — ties.
- **Conclusion:** REJECTED — axis flat. Same caveat as L11 regarding config era.

## L13. CV-split variance
- **Hypothesis:** Split-choice noise sigma is large enough (~±0.0007 fold spread observed) that sub-margin "flat" verdicts may be split artifacts.
- **Motivation:** Champion folds span 0.96584–0.96727; micro-grid flatness was never controlled for split choice.
- **Expected effect:** quantifies sigma; recalibrates all margin judgments.
- **Experiment:** EXP-041 (cv_seed 123 → 0.966519), EXP-042 (cv_seed 2024 → 0.966498) — identical model config to champion (0.966518), different splits. Run e0ad07e, 3h wall.
- **Result:** σ ≈ 1e-5 (Δ +0.000001 / -0.00002 vs champion). Fold-spread dominates, split-choice noise is negligible.
- **Conclusion:** REJECTED as confounder — margins are trustworthy as-is. “Flat” calls survive; no re-audit needed.
- **Follow-up:** none. Use σ ≈ 1e-5 to calibrate future promotion thresholds.

## L14. Monotonicity constraints
- **Hypothesis:** Injecting the observed monotone-saturating DGP prior (constraints on daily_screen/social_media/gaming/work_study/weekend_screen) reduces variance and beats the unconstrained champion.
- **Motivation:** Domain logic + decile monotonicity; constraints are structural regularization, not tuning.
- **Expected effect:** unknown — genuinely open at queue time.
- **Experiment:** EXP-039 (monotone_constraints dict on 5 features → position list, max_bin2048/col0.3/nl255, GHA).
- **Result:** 0.953856 vs champion 0.966518 (−0.01266). Catastrophic.
- **Conclusion:** REJECTED — prior conflicts with data; DGP is not strictly monotone despite decile averages. Informs L17/L18 search away from hard priors.
- **Follow-up:** none for hard constraints. Soft penalty variant (L15 in batch-2) also failed at screen.

## L15. Capacity × colsample interaction
- **Hypothesis:** At col 0.3 each tree sees ~half the features, so the nl255 leaf ceiling may shift upward.
- **Motivation:** Ceiling verdicts (L3) were established at col 0.7.
- **Experiment:** EXP-040 (nl511/mcs120/lr0.015/4000t @ col 0.3/max_bin2048 → 0.966185).
- **Result:** −0.00033 vs champion 0.966518.
- **Conclusion:** REJECTED — ceiling confirmed robust across feature-diversity settings (col 0.7 and col 0.3 both). No climb.
- **Follow-up:** none.

---

## L16. Duplicate structure / label-conflict noise ceiling
- **Hypothesis:** Near-duplicate rows with conflicting labels create a measurable Bayes-error ceiling.
- **Motivation:** Residual analysis showed errors are contradiction-shaped (features contradict labels), suggesting intrinsic noise; duplicates would quantify it.
- **Experiment:** Local analysis — exact-duplicate detection + near-duplicates at 0.1 rounding tolerance over all 12 features, 691,369 rows.
- **Result:** 0 exact duplicates; 2 rows in near-dup groups (0.0003%).
- **Conclusion:** REJECTED — dataset is fully continuous/unique; no duplicate-based noise ceiling exists. The apparent error structure is continuous feature overlap (contradictory feature combinations), not duplicated observations. Noise floor must be inferred from cross-model convergence (~0.9665) instead.
- **Follow-ups:** none unless new evidence contradicts.

## L17. Compute economics: FAST/MEDIUM/NUCLEAR tier decomposition
- **Hypothesis:** The champion's gain over a fast config decomposes into components (bins/trees/leaves/colsample/ensemble) with different cost curves; a MEDIUM tier captures most of the gain at a fraction of runtime, enabling cheap hypothesis screening.
- **Motivation:** Champion costs ~2500s GHA vs ~120s for fast configs (21×) for −0.0034 AUC. Screening hypotheses at nuclear cost wastes capacity.
- **Method:** Local single-seed 5-fold sweep measuring (config → OOF, runtime): FAST (nl64/lr0.05/800t/bins255/col0.8), MED-A (nl127/lr0.03/1500t/bins1024/col0.5), MED-B (nl255/lr0.02/2000t/bins2048/col0.5), MED-C (=champion but 2000t), NUC-ref (full champion single seed). Plus champion-vs-fast per-row disagreement mapping.
- **Expected effect:** identify knee point; target MEDIUM ≥ champion −0.001 at ≤25% runtime.
- **Experiment:** local sweep (screening measurements only — GHA verification required before any research conclusion).
- **Result:** PENDING (sweep running).
- **Follow-up:** winning MEDIUM becomes the standard screening tier; nuclear reserved for final verification.

## L18. Ensemble diversity value at aggressive colsample (from EXP-035 evidence)
- **Hypothesis:** At col 0.3 the champion's blend exceeds its best member by more than at conservative colsample (diversity-value hypothesis).
- **Evidence:** EXP-035 members spanned 0.965742–0.965820 while blend hit 0.966518 (+0.0007 over best member) — compare against nl64-era where averaging added less.
- **Experiment:** analysis of existing records (no new run).
- **Result:** pending formal comparison vs EXP-011-era member/blend gap (+0.00036).
- **Conclusion:** PENDING — if confirmed, diversity-per-tree is a first-class lever alongside colsample itself.

---

## Open questions derived from evidence (not yet experiments)

- Q1: What is the Bayes/noise floor? → partially answered by L16: no duplicates; residual profile shows contradiction-shaped overlap. Floor inferred from model convergence ≈ 0.9665+.
- Q2: Where does the champion lose rank? → ANSWERED (residual analysis): 67% of errors in mid-band [0.25,0.9] with within-band AUC 0.682; missing-feature rows −0.03 AUC; contradiction-shaped FN/FP profiles.
- Q3: Is seed-99 systematically stronger at aggressive configs? (pattern hunt across records — pending)
- Q4: Do train/test generation-order artifacts exist? (id-vs-feature correlation check — pending)
