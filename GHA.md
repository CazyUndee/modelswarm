# GHA Status Board — canonical Actions tracking

RULE: every pushed experiment/analysis gets a row here immediately. Update on every check.
An experiment is NOT finished until its GHA status has been explicitly verified in this file.
Read this file at the START of every research cycle. Do not trust memory/notifications.

Last full check: 2026-08-25 22:35 UTC

## RUNNING

| ID | Workflow | Run ID | Commit | Started | Job status | Exp status | Result/OOF | Artifacts | Next action | Checked |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-121 | experiment-runner | 32894173771 | 867150a | 20:14Z | in_progress (1h16m+) | pending | TabM x3, ETA ~22:15Z | not yet | when done: pull results, blend v3 IMMEDIATELY | 21:31Z |
| EXP-124 | experiment-runner | 32901353003 | 6882f18 | 21:30Z | in_progress (12m) | pending | pair-lattice nested TE | not yet | compare vs EXP-123 | 21:43Z |
| EXP-125 | experiment-runner | 32901893862 | 486416c | 21:36Z | in_progress (6m) | pending | raphdraft budget/ratio block on pairs base | not yet | compare vs EXP-124 | 21:43Z |
| EXP-122 | experiment-runner | 32897449661 | d113e0b | 20:48Z | in_progress (13m) | pending | pending (10-fold, ETA ~22:20Z) | not yet | when done: compare vs EXP-120a | 21:10Z |

## QUEUED

| ID | Workflow | Run ID | Notes |
|---|---|---|---|
| EXP-123 nested TE + freq (m=10) | (this push) | queued | 5x LGBM + nested/freq TE | when done: compare vs EXP-120a 0.968041 |

| ID | Workflow | Run ID | Notes |
|---|---|---|---|
| (none) | | | |


| EXP-123 nested TE | 32899563848 | **OOF 0.967339 REJECTED** (-0.00062 vs plain TE; inner-OOF hurt our context) | exp-EXP-123-artifacts | pairs/budget must be retested on plain TE -> EXP-126 |
| EXP-124 pair-lattice (nested) | 32901353003 | in progress (~37m) — confounded by nested knob | pending | read with EXP-123 caveat |
| EXP-125 budget block (nested) | 32901893862 | in progress (~24m) — confounded by nested knob | pending | same |
| EXP-126 max-strength combo | (push a404859) | QUEUED: plain TE m=10 + pairs + budget @ 10-fold | — | strongest candidate |
| EXP-122 10-fold | 32897449661 | **OOF 0.968223 KEEP** (+0.0003 vs 5-fold; best single) | exp-EXP-122-artifacts | fold-count axis confirmed |

## COMPLETED

| ID | Run ID | Result / OOF | Artifacts | Follow-up |
|---|---|---|---|---|
| EXP-118 slack | 32881022807 | OOF 0.967227 (+0.00071) | exp-EXP-118-artifacts | superseded by 119/120a |
| EXP-119 TE m=50 | 32882085543 | **OOF 0.967959 CHAMPION** | %TEMP%\exp119_art | promoted |
| EXP-120a m=10 | 32891009744 / job 97942744828 | **OOF 0.968041** best smoothing | %TEMP%\exp120a_art + GHA artifact | blend v3 ingredient; formal champ stays 119 |
| EXP-120b m=200 | 32891009744 / job 97942744721 | ~0.96778 REJECTED | GHA artifact | none |
| analysis: hillclimb_blend | 32906611244 | DONE: held-out 0.969550 (< greedy 0.969654); same core members | artifact done | recorded |
| analysis: blend_strategy_compare | 32898312819 -> REDISPATCH 32899786716 | v1/v2 failed inside; v4 DONE: prob 0.969383 > rank 0.969310 > logit 0.969065 -> probability-space confirmed optimal; axis closed | recorded in HISTORY |
| analysis: greedy_blend_74 | 32898316603 -> 32898911375 (failed: vectors untracked) -> REDISPATCH 32899790301 | artifact pending | consume selection curve |
| earlier session runs (EXP-007..117) | various | see experiments/*.yaml | mostly expired | historical |

| EXP-121 TabM x3 @600s | 32894173771 | CANCELLED at old 120-min job cap (~2h in) | requeued as EXP-127 x2 @420s under new 200-min cap |

| EXP-124 REDISPATCH | 32906270220 | running (FE-registration fix landed) | — |
| EXP-125 REDISPATCH | 32906272823 | running | — |
| EXP-126 | 32905105822 CANCELLED (same bug) -> REDISPATCH 32906275742 | running | — |

## FAILED / CANCELLED

| ID | Run ID | Reason | Resolution |
|---|---|---|---|
| analysis greedy (pre-fix) | 32898316603 | local paths + no kagglehub + tee-masked failure | fixed script+workflow; re-dispatched as 32898400579-lineage |

## Re-dispatch queue
- [x] analysis.yml kagglehub dep + pipefail fix (this commit)
- [ ] re-dispatch blend_strategy_compare.py after push
- [ ] re-dispatch greedy_blend_74.py after push
- [ ] EXP-121 lands -> run blend_v3.py inputs (needs artifacts download)
- [ ] EXP-122 lands -> fold-count verdict -> possibly promote config for final
