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

## L18. Ensemble diversity value at aggressive colsample
- **Hypothesis:** At col 0.3 the champion's blend exceeds its best member by more than at conservative colsample (diversity-value hypothesis).
- **Evidence:** EXP-035 members spanned 0.965742–0.965820 while blend hit 0.966518 (+0.00070 over best member) — vs EXP-011-era member/blend gap (+0.00036 at col 0.7).
- **Experiment:** analysis of existing records (no new run).
- **Result:** diversity gain at col 0.3 ≈ +0.00070, roughly 2× the +0.00036 seen at col 0.7. Cross-config agreement analysis (L19) shows this intra-ensemble signal is already fully captured by the probability-average — ensembles over EXTERNAL configs cannot add further.
- **Conclusion:** CONFIRMED — per-tree feature restriction increases member diversity value; the probability-average captures all of it. Diversity-per-tree is real but already harvested by the champion design.
- **Follow-ups:** none — any new diversity must come from a different model family or new features.

---

## L22. Leaderboard-intelligence pivot: DGP-constraint slack + 1-D target encoding
- **Trigger:** User reports S6E8 public LB top ≈ 0.9712. Verified context: community blend OOF 0.96943 → LB 0.97062 (+0.0012 OOF→LB transfer), so champion EXP-035 OOF 0.966518 ≈ LB ~0.9675 — a REAL gap of ~+0.004 sits above us. The "no headroom" verdicts (L19/L21) were correct for the closed feature×family space but the space itself had untested levers visible only from community evidence.
- **Lever 1 — free_time_slack (+0.0008 community-wide):** `daily_screen_time_hours ≥ social + gaming + work_study` holds in 100.00000% of all 987,671 train+test rows (verified locally on committed data: 0 violations, 331+128 exact-boundary rows). The slack is a latent generator input. Our EXP-110 diff falsification tested sleep_debt/screen_minus_social/social_minus_gaming — never this algebraic constraint. Queued as EXP-118 (champion ensemble + slack, one variable).
- **Lever 2 — 1-D target encoding (+0.0030 community-measured, largest known):** exact-value target rates wildly non-monotonic but replicate r=0.9975 across independent halves (digit-1 of daily_screen swings positive rate 0.6513→0.7365). Implemented leak-free fold-wise TE in the runner (`_target_encode_fit/_apply`, smoothing m=50 default, NaN kept as explicit group against the pandas≥3 silent-drop trap; 4 unit tests). Queued as EXP-119 (= EXP-118 + te_<col> for the 12 canonical features).
- **Also confirmed by community (matches our falsifications exactly):** ratios/indicators/linear-combos null, 2-way TE null (+0.00003), pseudo-labeling −0.00004, stacking over weighted average +0.00003, hyperparameter search transfers for LightGBM only. max_bin high = +0.0022 (we have bins2048 ✓).
- **Next-phase candidate:** architecture-diverse NN members (lookup-transformer LB 0.97041 solo; PLR+lookup embeddings with attention, rank corr vs trees 0.963–0.969 → largest blend weight 0.223 despite weaker solo score). This is the only direction that satisfies "decorrelated AND competitive" per independent community measurement — consistent with our L21 finding that tree-family diversity produces nothing. Requires new NN member type in runner; decision gated on EXP-118/119 results.
- **Status:** EXP-118 + EXP-119 running in parallel on GHA. Promotion gate unchanged (>+0.0005).

---

## L19. Cross-config error agreement: irreducible vs recoverable error
- **Hypothesis:** If different configs make DIFFERENT mistakes, diversity/stacking has recoverable headroom; if they share errors, the champion sits near the information ceiling.
- **Motivation:** 67% of errors concentrate in mid-band [0.25,0.9] with within-band AUC 0.682; needed to know whether that band is config-limited or data-limited.
- **Method:** Local analysis over 7 prediction vectors spanning the full config spectrum (FAST nl64/bins255 → champion nl255/bins2048/col0.3 5-seed blend): pairwise correlations, per-row error-agreement across configs, realistic ensembles (mean/rank/logistic-stack) over the 6 non-champion vectors.
- **Result:**
  - Correlations 0.986–0.9997 (MED-C~NUC-REF 0.99972; champion~NUC 0.99907).
  - Error agreement: **50,885 rows (7.4%) err under ALL 7 configs**; 607,520 err under none; only 32,964 disputed.
  - **79% of champion errors are invariant across the entire config spectrum**; only 339 unique.
  - Ensembles over all 6 non-champion vectors: mean 0.965844, rank 0.965820, logistic-stack 0.966346 (in-sample, optimistic) — ALL below champion 0.966518.
  - MID band: better configs rank MID monotonically better (FAST 0.718 → champ 0.751); 36,681 all-config errors remain in MID.
- **Conclusion:** RECOVERABLE DIVERSITY EXHAUSTED. ~79% of error mass is invariant across a 10× compute/config spectrum; no combination of existing-config predictions beats the champion; the seed-averaged blend already captures all available inter-config signal. Remaining error is dominated by intrinsic feature-label overlap (contradiction-shaped FN/FP), not model deficiency.
- **Follow-ups:** only new *information* changes this — LB public score, or features not derivable from the current 12. Config-space research on this feature set is complete.

## L20. Compute economics tier decomposition
- **Hypothesis:** A MEDIUM tier captures most of the champion's gain at ≤25% runtime.
- **Evidence:** Concurrent-session local sweep (single-seed, 5-fold): FAST 0.963017@200s → MED-A 0.965803@2757s → MED-B 0.966165@2435s → MED-C 0.966064@1599s → NUC-REF 0.966099@2113s; champion blend (GHA, 5-seed) 0.966518.
- **Result:** MED-B (nl255/bins2048/col0.5/2000t, single seed) reaches −0.00035 of champion at ~85% of its runtime — NO cheap knee exists; most of the gain comes from bins2048+nl255 which ARE the expensive components.
- **Conclusion:** REJECTED — no free lunch tier. Screening at FAST level (−0.0035) preserves ranking but underestimates absolute scores; acceptable for relative comparisons only (as used in EXP-100..108).
- **Follow-ups:** screening funnel stays as-is (12% rows/3 folds/600t for hypothesis elimination; full-data confirmation only for survivors).

## L21. Sibling-GBM family diversity (XGBoost / HGB / CatBoost) vs champion
- **Hypothesis:** A different GBM implementation makes decorrelated errors of comparable quality; blending harvests the disputed rows L19 left open.
- **Method (local, single-seed, same protocol StratifiedKFold5/seed42):** tuned siblings mirroring champion capacity — XGBoost `hist/lossguide/max_leaves255/lr0.02/3000r/sub0.8/col0.3`; sklearn HistGradientBoosting `1500i/lr0.03/leaves127/mcs80`; CatBoost `2500i/lr0.03/d6/Bernoulli0.8/rsm0.3` (codes-encoded cats, mirrors runner). Blend scan w∈{0.5..0.95}, gated rules, in-sample stacker upper bounds.
- **Evidence & results:**
  - XGBoost full OOF **0.964021** (−0.00250); champ~XGB r=**0.99303**; best blend (w=0.95) −0.000032.
  - HistGradientBoosting full OOF **0.963387** (−0.00313); champ~HGB r=**0.98857**; best blend −0.000003 (exactly flat).
  - CatBoost full OOF **0.956467** (−0.01005); champ~CAT r=**0.97053** (most decorrelated); best blend −0.000106; MID-band AUC collapses to 0.594.
  - Mechanism: champ-only-error sets are boundary coin-flips — e.g. vs HGB, 8,349 champ-only errors have 77% predictions in [0.4,0.6] and ~50/50 labels (pop 71%); sibling "wins" there carry >0.25 margin only 2% of the time.
  - Upper bounds: in-sample logit-stack 0.966468 (< champion); in-sample GBM-stack over {champ,hgb,medb} 0.966912 (+0.0004 optimistic bound ⇒ ~0 true). All gated mix rules ≤ champion (+0.000015 max via medb gate).
- **Conclusion: FAMILY AXIS CLOSED — recoverable diversity = ZERO across XGBoost, CatBoost, HGB.** Decorrelation exists (r 0.971–0.993) but is dominated by the quality gap in every combiner: global blends peak at w=1.0, gated rules ≤ +0.00002, in-sample stackers ≤ optimistic +0.0004. Remaining error mass is symmetric aleatoric boundary noise shared by all implementations, not model-specific deficiency. Combined with L19 (config pool) the entire accessible model space converges on the same ~50k-row invariant error core; EXP-035 stands as the effective Bayes projection for this feature set.

## L23. Slack + target encoding CONFIRMED on GHA; blend ceiling mapped
- **Results:** EXP-118 (+free_time_slack) OOF **0.967227** = +0.00071 vs EXP-035 (community predicted ~+0.0008). EXP-119 (+1-D TE m=50) OOF **0.967959** = **+0.00144 vs EXP-035** -> PROMOTED CHAMPION. LB transfer ~+0.0015 projects solo ~0.9695.
- **Blend v2** (EXP-119 + public library members, OOS half-split weights): held-out **0.969387**, statistically identical to blend v1 over the WEAKER old champion (0.969393). Conclusion: blend ceiling is set by the library's NN direction diversity (lookup w=0.334, tabm_seed3 w=0.243 dominate; our GBDT weight fell to 0.022), NOT by our champion's strength. Further blend gains require NEW directions we own.
- **Frontier:** (a) TE smoothing sweep m in {10,200} running as EXP-120a/b; (b) self-built NN members (lookup-transformer / TabM via pytabkit) to add owned decorrelated directions - local GPU is MX250/CPU-only-torch so GHA-CPU or time-boxed local CPU are the compute paths; (c) adversarial audit of under-tested families in flight.
- **Method note:** per-member OOF export (runner patch 5e87ab0) worked in production - member vectors now available for every future GHA run.