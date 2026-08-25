# S6E8 Experiment Registry — Orchestrator Lock Board

Purpose: prevent duplicate experiments across parallel agents. Before queueing anything,
check LOCKED/RUNNING rows here. Update status as evidence lands.

| ID | Hypothesis | Parent | Compute | Status | Result | Owner |
|---|---|---|---|---|---|---|
| EXP-118 | +free_time_slack DGP constraint (daily−social−gaming−work) | EXP-035 | GHA | RUNNING (43m) | pending | orchestrator |
| EXP-119 | EXP-118 + 1-D target encoding (m=50, 12 canonical cols) | EXP-118 | GHA | RUNNING (32m) | pending | orchestrator |
| BLEND-v1 | champ + public OOF library members (OOS-fitted weights) | — | local analysis | SUBMITTED | **LB 0.97053** (held-out est 0.96939) | orchestrator |
| CHAMP-LB | EXP-035 anchor submission | — | — | DONE | **LB 0.96801** | orchestrator |
| EXP-120a | TE smoothing m=10 | EXP-119 | GHA | PLANNED (gated on 119) | — | unassigned |
| EXP-120b | TE smoothing m=200 | EXP-119 | GHA | PLANNED (gated on 119) | — | unassigned |
| EXP-121 | weekend_slack secondary constraint | EXP-118 | GHA | PLANNED (low prior: constraint only 99.69%) | — | unassigned |
| NN-TABM | Reproduce TabM (pytabkit, constrained-imputation + lattice TE, seed-avg) | — | LOCAL CPU/GPU | IN PROGRESS (tooling) | — | orchestrator |
| NN-LOOKUP | Reproduce lookup-transformer (per-value embeddings + attention) | — | LOCAL | PLANNED | — | unassigned |

## Rules
1. Check this board before queueing; claim by writing your name/agent id.
2. One variable per experiment vs its parent unless explicitly batched.
3. Authoritative OOF numbers come from GHA runs only; local numbers are directional.
4. Blend submissions must attribute member sources in the submission message.
