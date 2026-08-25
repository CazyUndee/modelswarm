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
