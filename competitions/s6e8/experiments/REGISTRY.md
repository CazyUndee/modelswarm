# S6E8 Experiment Registry — Orchestrator Lock Board

Purpose: prevent duplicate experiments across parallel agents. Before queueing anything,
check LOCKED/RUNNING rows here. Update status as evidence lands.

| ID | Hypothesis | Parent | Compute | Status | Result | Owner |
|---|---|---|---|---|---|---|
| EXP-118 | +free_time_slack DGP constraint | EXP-035 | GHA | DONE | **OOF 0.967227 (+0.00071)** | orchestrator |
| EXP-119 | EXP-118 + 1-D TE (m=50) | EXP-118 | GHA | DONE **CHAMPION** | **OOF 0.967959 (+0.00144)** | orchestrator |
| BLEND-v2 | EXP-119 + library members, OOS weights | — | local analysis | SUBMITTED | held-out 0.969387 (LB pending) | orchestrator |
| BLEND-v1 | champ + public OOF library members (OOS-fitted weights) | — | local analysis | SUBMITTED | **LB 0.97053** (held-out est 0.96939) | orchestrator |
| CHAMP-LB | EXP-035 anchor submission | — | — | DONE | **LB 0.96801** | orchestrator |
| EXP-120a | TE smoothing m=10 | EXP-119 | GHA | DONE | **OOF 0.968041 (best m)** keep-as-blend-ingredient | orchestrator |
| EXP-120b | TE smoothing m=200 | EXP-119 | GHA | DONE | ~0.96778 REJECTED | orchestrator |
| EXP-121 | weekend_slack secondary constraint | EXP-118 | GHA | PLANNED (low prior: constraint only 99.69%) | — | unassigned |
| NN-TABM | Reproduce TabM (pytabkit); local CPU too slow for full OOF -> evaluate GHA-CPU member type or time-boxed local | — | TBD | TOOLING READY (pytabkit installed; API mapped; needs NaN-impute recipe) | smoke interrupted on CPU | orchestrator |
| NN-LOOKUP | Reproduce lookup-transformer (per-value embeddings + attention) | — | LOCAL | PLANNED | — | unassigned |

## Rules
1. Check this board before queueing; claim by writing your name/agent id.
2. One variable per experiment vs its parent unless explicitly batched.
3. Authoritative OOF numbers come from GHA runs only; local numbers are directional.
4. Blend submissions must attribute member sources in the submission message.
| EXP-122 | 10-fold champion variant (fold-count axis) | EXP-120a | GHA | RUNNING run 32897449661 | pending | orchestrator |
| EXP-123 | NESTED TE + frequency columns (m=10) | EXP-120a | GHA | RUNNING (queued push f32f50d) | pending | orchestrator |
| EXP-122b | 10-fold champion variant | EXP-120a | GHA | DONE | **OOF 0.968223 best single** (+0.0003 vs 5-fold) | orchestrator |
| EXP-123b | nested TE + freq (m=50) | EXP-119 | GHA | DONE | **0.967339 REJECTED** (-0.00062; inner-OOF hurt) | orchestrator |
| EXP-124 | pair-lattice nested TE (36 pairs) | EXP-123-lineage | GHA | RUNNING | pending (nested-confounded) | orchestrator |
| EXP-125 | raphdraft budget/ratio block (nested) | EXP-124 | GHA | RUNNING | pending (nested-confounded) | orchestrator |
| EXP-126 | pairs + budget on PLAIN TE @ 10-fold | EXP-122 | GHA | RUNNING (push a404859) | strongest candidate | orchestrator |
| EXP-127 | TabM x2 @420s retry | EXP-121(cancelled at old cap) | GHA | QUEUED (cap raised 200m) | pending | orchestrator |
| EXP-129 | +weekend_slack | EXP-122 | GHA | DONE | **OOF 0.968225 REJECTED** (+0.000002, tied) | buffy |
| EXP-130 | +clean raphdraft block (no pairs) | EXP-122 | GHA | DONE | **OOF 0.968583 KEEP** (+0.00036, below bar) | buffy |
| EXP-131 | +digit family standalone | EXP-122 | GHA | DONE | **OOF 0.968236 REJECTED** (+0.000013, tied) | buffy |
| EXP-132 | lookup-transformer (exact-value embeddings + transformer, 3 seeds) | EXP-130 | GHA | CANCELLED (32980487428) @3h20m runner cap | **ZERO artifacts** — not viable on GHA | buffy |
| ANALYSIS | tamerlan_plus_owned (public blend seed) | — | GHA | DONE 33337849843 | Tamerlan 0.969487; owned adds only +0.00008 | buffy |
| ANALYSIS | rank_gauss_stack (reproduce RGS) | — | GHA | DONE 33384855096 | 241 members, stack OOF 0.970167 -> **LB 0.97130** (public ceiling) | buffy |
| ANALYSIS | FT-Transformer screen | — | GHA | INFEASIBLE | 58 min in fold 1 @60-min cap; abandoned | buffy |
| ANALYSIS | TabFM screen | — | GHA | OOM (exit 143) | fixed safetensors; rerun unqueued | buffy |
| ANALYSIS | mlp_baseline | — | GHA | DONE 33333005011 | MLP OOF 0.9378, adds 0 to blends — axis closed | buffy |
| EXP-133 | cross-family GBDT ensemble (LGBM+XGB+CatBoost) with TE m=10 | EXP-119 | GHA | QUEUED (push 97bbc2e) | pending — first cross-family test | agent-81d2 |
