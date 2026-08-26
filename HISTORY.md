# SESSION HISTORY — S6E8 Research Program (rolling)

Last updated: 2026-08-25 ~20:10 UTC. Maintained by orchestrator for compaction survival.
Read this + STATE.md + competitions/s6e8/experiments/REGISTRY.md before assuming anything.

## Where we are RIGHT NOW

- **Champion: EXP-119 @ OOF 0.967959** = EXP-035 base + free_time_slack feature + 1-D target
  encoding (m=50) on all 12 canonical features. Promoted +0.00144 over EXP-035 (OOF 0.966518).
  Runner: scripts/run_experiment.py; config: competitions/s6e8/experiments/EXP-119.yaml.
- **Best LB: 0.97053** (blend v1) / 0.97050 (blend v2, statistically tied) = our champion +
  public OOF-library members (szymonkapiski/s6e8-oof-library-47-models, CC0, same CV scheme
  StratifiedKFold(5,seed42), verified aligned to published AUCs to 5dp).
  Top of LB: **0.97172** (Chris Deotte); dense pack 0.97129–0.97149. Gap ≈ +0.0012.
- Kaggle auth: OAuth cached as user **crazybro** (`python -m kaggle ...` works; kaggle.json absent).
  7 submissions left today. Submissions must attribute public-member sources.

## In flight (as of last update)

| Item | Where | ETA/Status |
|---|---|---|
| EXP-120a TE smoothing m=10 | GHA run 32891009744 | started 19:41 UTC (~40 min) |
| EXP-120b TE smoothing m=200 | GHA same matrix run | same |
| EXP-121 self-owned TabM ×3 seeds | GHA (pushed 867150a) | just queued |
| Library audit agent (74 members → direction clusters) | bg_c550b7d0 / ses_fc5c6709... | running |
| Librarian mining (Tamerlan lookup arch, RealMLP, TabM cfg, naji*, Deotte posts) | bg_ef681026 / ses_fc5c6427... | running |
| Adversarial audit (under-tested hypothesis families) | bg_f5c01bfc | running |

## How we got here (arc)

1. Sessions 1–2: built runner/GHA infra, 54 experiments, mapped 16 hyperparameter axes,
   promoted EXP-035 (nl255/lr0.02/3000t/bins2048/sub0.8f1/mcs80/col0.3, 5-seed blend).
   Declared config space exhausted (L19: 79% errors invariant across 7 configs).
2. Sibling-GBM probe (L21): tuned XGB/HGB/CatBoost all −0.0025..−0.010 with corr
   0.971–0.993; every blend flat-or-worse. Family axis closed — CORRECT for tree libraries.
3. User revealed top LB ≈ 0.9712 → intelligence pivot (L22): community discussion showed
   two missed levers: free_time_slack constraint (+0.0008) and 1-D TE (+0.0030).
4. EXP-118/119 confirmed both on GHA same day. Blend-with-library reached 0.97053 LB.
5. Now: closing the last +0.0012 via self-owned NN directions (EXP-121 TabM; lookup-transformer
   planned), TE smoothing sweep, adversarial hypothesis mining.

## Load-bearing assumptions (do not re-derive)

- LB runs ~+0.0011–0.0015 ABOVE OOF on this competition's public split (measured twice:
  champ 0.96652→0.96801; blends 0.96939→0.97050/0.97053).
- Error mass is boundary coin-flips: sibling-vs-champ disputed rows are 77% in [0.4,0.6]
  with ~50/50 labels. No combiner extracts them (proven in-sample AND out-of-sample).
- DGP exact constraint: daily ≥ social+gaming+work in 100.00000% rows (459 on boundary).
  weekend analog holds only 99.69% (weaker prior, unqueued beyond registry note).
- Exact-value target rates replicate r=0.9975 across train halves → 1-D TE is real signal;
  2-way TE is NOT (trees already do interactions). Ratios/diffs/indicators/pseudo-labeling/
  stacking-over-weighted-average are ALL falsified by us AND community.
- The library's NN members (lookup w≈0.33, tabm w≈0.24 in OOS-fitted blends) dominate because
  rank-corr vs trees is 0.963–0.98 while trees correlate 0.98–0.99 among themselves.
  Our own GBDT weight fell to 0.02 in v2 — champion strength alone cannot lift the blend;
  only NEW DIRECTIONS can.

## Compute policy (ABSOLUTE, user-mandated)

- **ZERO model execution locally.** No training/inference/predict_proba/"tiny smoke"/runtime
  estimation. Local = editing, static analysis, YAML validation, non-model unit tests,
  artifact analysis, orchestration, GHA dispatch/log reading ONLY.
- Model validation path: static inspection → mocked unit tests (no real library) → commit &
  push → debug from GHA logs.
- Local hardware reality: MX250 4GB with CPU-only torch build (cuda unavailable); CPU thermal-
  throttles under TabM even at 40k rows. pytabkit installed locally for API inspection only.
- GHA matrix runs multiple experiment yamls concurrently (fail-fast false); push to master
  touching competitions/*/experiments/*.yaml triggers runs automatically.

## Key artifacts

- %TEMP%\exp119_art\ — EXP-119 oof_predictions.csv + submission.csv + per-member OOFs
  (runner exports oof_member_<key>.csv since commit 5e87ab0).
- %TEMP%\exp035\ — old champion vectors (superseded).
- %TEMP%\s6e8_ooflib\ — public 74-model library (oof/*.npy + test_*.npy + manifest.csv +
  hyperparameters.json + src/). float64. Aligned to train.csv row order.
- %TEMP%\opencode\ — analysis scripts (lib_blend_scan.py, make_blend_v1.py, sibling_blend_analysis.py).
- forum/hypothesis-ledger.md L1–L23; REGISTRY.md = experiment lock board (update before queueing!).

## Incident log / hard rules learned

- SHALLOW-COPY YAML BUG (hit 3x on 2026-08-25): cloning an experiment yaml via dict(base)
  carries the source results block verbatim -> stale numbers look like real ones and mask
  GHA truth. RULE: when creating experiments from a template ALWAYS c['results'] = {} and
  c['status'] = chr(39)queuedchr(39). Verify by grepping the new file for oof_metric before commit.
- Result-commit races: matrix jobs push results with retries; orchestrator rapid doc-pushes
  caused 6-retry exhaustion under fetch-depth:2 (now fixed to fetch-depth:0). When yaml values
  look wrong, TRUST THE JOB LOGS first: gh api repos/CazyUndee/modelswarm/actions/jobs/<id>/logs.
- TE smoothing TRUE verdict (from logs): m=10 OOF 0.968041 > m=50 0.967959 > m=200 0.96778.
  m=10 wins slightly; champion formally stays EXP-119 (<+0.0005 bar). m=10 vectors downloaded
  at %TEMP%/exp120a_art for blend v3.

## Overnight session additions (2026-08-25 evening)

- LB submissions: champ anchor 0.96801 -> blends 0.97050/0.97053 -> greedy-74 5-member 0.97054.
  Public-LB plateau ~0.9705 with library blends; honest OOS fitting gains do NOT transfer to
  public split. Rank 534/2913 (top 18%); r500=0.97068, r100=0.97125, #1=0.97181.
- Blend-space axis CLOSED: probability-space weighting beats rank/logit (GHA analysis).
- Fold-count axis CONFIRMED: 10-fold +0.0003 over 5-fold (EXP-122 0.968223 best single).
- Nested TE REJECTED in our context: -0.00062 (EXP-123) despite raphdraft success - their gain
  came from their pipeline context (XGB depth-wise, lambda grid, imputation interplay).
- EXP-121 cancelled at old 120-min job cap -> cap raised to 200; requeued as EXP-127 (x2 @420s).
- New mechanisms implemented+queued: pair_grid FE op (Deotte/latwide), budget_constraint block,
  impute_median pinned to train stats (raphdraft block) - EXP-124/125 (nested-handicapped) and
  EXP-126 (clean max-strength combo, strongest candidate).

- PROVENANCE CORRECTION (00:55Z): the "vault NN" claim was FAKE — community exposé proved the
  vault author zero-weighted his own NN (weight 1e-6) and shipped byte-for-byte copies of public
  blends: vault submission.csv = **Naji Ama "Ensemble of Ensembles"** (0.97101 post, now scoring
  0.97128 as Naji iterated it), vault submission(1).csv = Krasnov Daniil 0.97099. Our verified
  0.97128 IS Naji's blend. Any final pick using it MUST credit "Naji Ama — Ensemble of
  Ensembles" (and note it is public-work-derived, not ours).

- Band-local mechanism INDEPENDENTLY REJECTED (GHA analysis 10:30Z): band LGBMs [3,6h)
  0.9028 / [6,7.8h) 0.9191 vs base-blend in-band 0.9176/0.9341; splice costs -0.0018..-0.0020
  global. Confirms raykkretzschmar caution on our own composition. Axis closed.

- Batch II (2026-08-26 12:15Z): EXP-129 0.968225 REJECTED (weekend_slack +0.000001); EXP-130 0.968583 KEEP (+0.00036 vs EXP-122, new best owned single); EXP-131 0.968236 REJECTED (digit family +0.00001). Raph block is only incremental gain.

## Frontier additions (00:50Z mining pass)

1. raykkretzschmar/s6e8-fm-lattice-blend-members: 7 Factorization Machines, aligned OOF+test
   on frozen CV. Bilinear lookup class absent from library; author-measured blend gain only
   +0.000006 (decorrelation without strength). USE: candidate members in OOF-validated blends.
2. anhadmahajan06/ps-s6e8...-submission: LB-scored prediction files up to 0.97092.
   USE: extra test-side ensemble ingredients (no OOF -> weight via LB probes ONLY if a
   hypothesis justifies; else skip per overfit policy).
3. Band-local conditional models ('mix meta-models, fix weak bands'): train per
   daily_screen_time band; author says pays when within 0.015 of blend on-band; two bands
   measured negative. HYPOTHESIS: our mid-band weakness (AUC 0.68) matches this method -est
   candidate after EXP-126/127 verdicts.
4. donmarch14 digit/float categorical families: partially covered by our TE-on-exact-values;
   digit-specific columns (d1/frac) still untested standalone (EXP candidate).

## OVERNIGHT FINAL STATE (03:15Z, mission hour ~5.3 of 8-10)

- ALL RUNS TERMINAL. No experiments in flight.
- Best LB: **vault NN 0.97128** (~top-100). Runner-up: greedy74 blend 0.97054.
- Best owned single OOF: **EXP-122 (10-fold m=10 TE) 0.968223**.
- Rejected tonight: nested TE (-0.00062), grid-pairs clean (-0.00088), pair-leak voids x2,
  max-strength combo collapse 0.9074 (RCA: continuous features in pair grid + pre-fix checkout),
  TabM-on-GHA-CPU x2 (timeout infrastructure).
- Final-selection plan (deadline Aug 31 23:59 UTC): select vault NN + greedy74 as the two
  distinct submissions. MANUAL STEP REQUIRED on Kaggle before deadline.
- Morning options: FM members sit ready for OOF-validated blends; band-local method unexplored;
  hamzah meta-feature single (LB 0.97059) replicable on GHA if desired.

- CROSS-HARNESS CONTACT 12:20Z: Buffy (second agent) onboarded from our docs; corrected its
  stale snapshot (best LB now 0.97128=Naji public blend provenance; best owned single
  EXP-122 0.968223); assigned: batch-II consumption + FM blend verification + final-selection
  draft. Thread: forum/discussions/reply-to-buffy.md.

## Next frontier (FROZEN 2026-08-26: Frontier Attack Mission)

1. **ATTACK 0.97128 EXTERNAL FRONTIER** — verified Naji Ama Ensemble-of-Ensembles (1.0000
   Spearman vs vault). Goal: frontier + owned-signal > frontier. Blend frontier with
   lookup-transformer, TabM, EXP-130, etc. only via leakage-safe OOF weights; keep
   external vs owned provenance distinct in every submission that touches the frontier.
2. Priority 1 remains **lookup-transformer** (exact-value embeddings + transformer) — strongest
   untested family (+0.00095 blend evidence in library). Build first, measure complementarity.
3. If lookup-transformer helps: explore its disagreement regions (bands, rare values, quantized
   structure) to generate the next mechanism. If it fails: use its error structure to hypothesize
   the next architecture family.
4. Stack only when justified: constrained linear/logistic meta-model on OOF; test pipeline must
   reproduce without LB feedback.
5. Final-submission hygiene near deadline: keep best-2 distinct picks (frontier + diverse owned).

Prior candidates closed since this was written: fold-count axis (EXP-122), nested TE, pairs,
band-local — see overnight final state above.
