# GHA Status Board — canonical Actions tracking

RULE: every pushed experiment/analysis gets a row here immediately. Update on every check.
An experiment is NOT finished until its GHA status has been explicitly verified in this file.
Read this file at the START of every research cycle. Do not trust memory/notifications.

Last full check: 2026-08-31 11:20 UTC (Buffy — rank-gauss stack reproduced the 0.97130 plateau)

## RUNNING / IN FLIGHT

| ID | Workflow | Run ID | Status | Result/OOF | Next action | Checked |
|---|---|---|---|---|---|---|
| analysis: rank_gauss_stack | analysis.yml | 33384855096 | **SUCCESS** | 241 members, stack OOF 0.970167 (matches asteriosterzis RGS) | consumed -> 0.97130 sub | 11:20Z |
| analysis: tamerlan_plus_owned | analysis.yml | 33337849843 | SUCCESS | Tamerlan baseline 0.969487; owned adds only +0.00008 (tabm_seed3/naji03) | VALUE: marginal, axis mostly closed | 11:20Z |

## RECENTLY COMPLETED — OVERNIGHT BATCH II

| ID | Run ID | Config | Result |
|---|---|---|---|
| EXP-129 | 32957981684 | EXP-122 + weekend_slack | **OOF 0.968225 REJECTED** (+0.000002) |
| EXP-130 | same push | EXP-122 + clean raphdraft block (no pairs) | **OOF 0.968583 KEEP** (+0.00036, below bar) |
| EXP-131 | same push | EXP-122 + digit family (d1/frac x8) | **OOF 0.968236 REJECTED** (+0.000013) |


## QUEUED

| ID | Workflow | Run ID | Notes |
|---|---|---|---|
| EXP-123 nested TE + freq (m=10) | (this push) | queued | 5x LGBM + nested/freq TE | when done: compare vs EXP-120a 0.968041 |

| ID | Workflow | Run ID | Notes |
|---|---|---|---|
| EXP-133 cross-family GBDT (LGBM+XGB+CatBoost) | experiment-runner.yml | (push f22f1e4) | QUEUED: first cross-family test, TE m=10, 5-fold | compare member correlations |


| EXP-123 nested TE | 32899563848 | **OOF 0.967339 REJECTED** (-0.00062 vs plain TE; inner-OOF hurt our context) | exp-EXP-123-artifacts | pairs/budget must be retested on plain TE -> EXP-126 |
| EXP-124 pair-lattice (nested) | 32901353003 | in progress (~37m) — confounded by nested knob | pending | read with EXP-123 caveat |
| EXP-125 budget block (nested) | 32901893862 | in progress (~24m) — confounded by nested knob | pending | same |
| EXP-126 max-strength combo | (push a404859) | QUEUED: plain TE m=10 + pairs + budget @ 10-fold | — | strongest candidate |
| EXP-122 10-fold | 32897449661 | **OOF 0.968223 KEEP** (+0.0003 vs 5-fold; best single) | exp-EXP-122-artifacts | fold-count axis confirmed |

## SUBMISSION TALLY + OVERFIT AUDIT (2026-08-26)

| # | submission | LB | classification |
|---|---|---|---|
| 1 | EXP-035 anchor | 0.96801 | genuine baseline |
| 2 | blend v1 | 0.97053 | OOS-validated blend |
| 3 | blend v2 | 0.97050 | OOS-validated blend (same lineage) |
| 4 | greedy74 | 0.97054 | OOS-validated selection |
| 5 | vault NN verify | **0.97128** | GENUINE standalone (no tuning) — highest confidence |
| 6-9 | n50/n65/a85/a75 mixes | 0.97084-0.97106 | PUBLIC-LB PROBES, same hypothesis x4, all negative -> STOPPED per policy |

Verdict: vault NN is the only high-confidence result among today's submissions.
Final-selection intent: vault NN (public-fit-free) + one structurally distinct
independently-validated alternative for private robustness.


## BEST SUBMISSION

| rank | submission | LB | notes |
|---|---|---|---|
| 1 | vault submission.csv | **0.97128** | PROVENANCE: byte-for-byte Naji Ama Ensemble-of-Ensembles (fake-NN cover exposed by community); spearman 0.990 vs greedy74; admixtures all hurt |
| 2 | greedy74 | 0.97054 | superseded as blend base |
| 3 | blend v1/v2 | 0.97053/0.97050 | historical |


## COMPLETED (analysis + older experiments)

| ID | Run ID | Result / OOF | Artifacts | Follow-up |
|---|---|---|---|---|
| EXP-129 weekend_slack | 32957981684 | **OOF 0.968225 REJECTED** (+0.000002 vs EXP-122) | GHA artifact | weekend axis CLOSED |
| EXP-130 raphdraft block | 32957981684 | **OOF 0.968583 KEEP** (+0.00036 vs EXP-122, below +0.0005 bar) | GHA artifact | retain as blend ingredient candidate |
| EXP-131 digit family | 32957981684 | **OOF 0.968236 REJECTED** (+0.000013 vs EXP-122) | GHA artifact | digit/float axis CLOSED |
| analysis: fm_blend_check | 32959461339 | **NEGATIVE** FM adds zero value; axis closed | analysis-output-32959461339 | FM lattice members too weak + correlated |
| EXP-118 slack | 32881022807 | OOF 0.967227 (+0.00071) | exp-EXP-118-artifacts | superseded by 119/120a |
| EXP-119 TE m=50 | 32882085543 | **OOF 0.967959 CHAMPION** | %TEMP%\exp119_art | promoted |
| EXP-120a m=10 | 32891009744 / job 97942744828 | **OOF 0.968041** best smoothing | %TEMP%\exp120a_art + GHA artifact | blend v3 ingredient; formal champ stays 119 |
| EXP-120b m=200 | 32891009744 / job 97942744721 | ~0.96778 REJECTED | GHA artifact | none |
| analysis: hillclimb_blend | 32906611244 | DONE: held-out 0.969550 (< greedy 0.969654); same core members | artifact done | recorded |
| analysis: blend_strategy_compare | 32898312819 -> REDISPATCH 32899786716 | v1/v2 failed inside; v4 DONE: prob 0.969383 > rank 0.969310 > logit 0.969065 -> probability-space confirmed optimal; axis closed | recorded in HISTORY |
| analysis: greedy_blend_74 | 32898316603 -> 32898911375 (failed: vectors untracked) -> REDISPATCH 32899790301 | artifact pending | consume selection curve |
| earlier session runs (EXP-007..117) | various | see experiments/*.yaml | mostly expired | historical |

| EXP-121 TabM x3 @600s | 32894173771 | CANCELLED at old 120-min job cap (~2h in) | requeued as EXP-127 x2 @420s under new 200-min cap |
| EXP-127 TabM x2 @420s | 32905414796 | FAILED: 3h18m silent (buffered stdout), cancelled at 200-min cap; TabM-CPU-on-GHA not viable | ABANDONED — library already holds tabm_seed3 (0.96867, best NN single) |

| EXP-124 REDISPATCH | 32906270220 | DONE: OOF 1.0 (leak, pre-fix checkout) -> VOIDED | superseded |
| EXP-125 REDISPATCH | 32906272823 | DONE: OOF 1.0 (same leak) -> VOIDED | superseded |
| EXP-126 REDISPATCH | 32911212785 | DONE: OOF 0.9074 REJECTED (continuous-feature pair TE collapse; RCA in yaml) | closed |

| EXP-125 v1/v2 | 32906272823 | OOF 1.0 - inherited same pair_grid leak; budget/ratio cols dropped by features-filter bug | — | voided; covered by EXP-126 |
| LEAK INCIDENT: EXP-124 v1/v2 | 32906270220 | OOF 1.0 - pair_grid encoded addicted_label; FE-generated cols were also silently dropped by config.features filter (both fixed, regression tests) | — | superseded by EXP-126 redispatch |

| EXP-126 root cause | — | OOF 0.9074 REJECTED: pair_grid over continuous derived features -> near-unique levels -> self-referential plain-TE collapse; ran pre-leak-fix checkout too. latwide recipe avoids by grid-only columns @0.1 res | — |
| EXP-128 clean pair-lattice | (push 80c3b80) | QUEUED: grid6 @0.1 res + plain TE m10 @10-fold on EXP-122 base | final candidate |

## FAILED / CANCELLED

| ID | Run ID | Reason | Resolution |
|---|---|---|---|
| analysis greedy (pre-fix) | 32898316603 | local paths + no kagglehub + tee-masked failure | fixed script+workflow; re-dispatched as 32898400579-lineage |

## SUBMISSION TALLY UPDATE (2026-08-31 deadline day)

| # | submission | LB | classification |
|---|---|---|---|
| A | **rank-gauss 241-stack w0.35 + VAULT base** | **0.97130** | reproduces public ceiling (asteriosterzis plateau); stack OOF 0.970167 |
| B | rank-gauss stack w0.35 + weak naji base (BROKEN base) | 0.96863 | ERROR — waste; blended vs a weak single model, not vault |
| C | r0tor cluster-aware logit-rank (50% stack+25%vault+25%aman) | in flight | structural Pick-B insurance; ~0.999 corr to vault |
| 0-9 | prior (vault 0.97128 etc.) | | |

**CURRENT BEST: 0.97130** (rank-gauss stack, exactly the public wall). Deadline 2026-08-31 23:59Z.

Gap to frontier: Chris Deotte moved to 0.97207. Exceeding 0.97130 needs NEW OOF members beyond
the RGS Aug-30 pool; all owned NN training on GHA still dead (EXP-132 killed @3h20m, FT-Transformer
60-min cap, TabFM OOM). 7 submissions left today as of 11:20Z.

## INTELLIGENCE SUMMARY (2026-08-31)

- **Public LB frontier**: 0.97207 (Chris Deotte, Aug 30) — up from 0.97186
- **Public-pool stack wall**: asteriosterzis rank-gauss = EXACTLY 0.97130 for any W in 0.20..0.50 (documented plateau; stack OOF 0.970167); pure stack alone = 0.97125 < vault
- **Blend base matters hugely**: same 0.970167 stack w0.35 vs weak naji base = 0.96863; vs vault base = 0.97130
- **Tamerlan**: owned models add only +0.00008 to his blend — our GBDT ladder cannot exceed the wall
- **Our NN diversity**: lookup + tabm_seed3 + EXP122 + EXP130 → OOF 0.96942 (away from the 0.9702 wall)

## Re-dispatch queue
- [x] analysis.yml kagglehub dep + pipefail fix
- [x] rank_gauss_stack.py (+hb_/mk_ pools, pyarrow, owned champ) -> 0.97130
- [x] tamerlan_plus_owned.py -> consumed (owned adds 0.00008)
- [x] EXP-132 consume -> ZERO artifacts (killed @3h20m runner cap); not viable
- [x] FT-Transformer -> infeasible on GHA (58min in fold1 @60-min cap); screen abandoned
- [x] TabFM -> OOM (exit 143); fixed safetensors but reruns not yet requeued
- [ ] explore new OOF members (post Aug-30) for the stack
- [ ] decide if C (r0tor cluster) or more submissions are worth burning for private robustness
