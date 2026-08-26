# Rolling Hypothesis Queue — Frontier Attack Mission

Last updated: 2026-08-26 14:35 UTC
Frontier: **0.97186** (Chris Deotte, live LB) | External reference: 0.97128 (Naji, verified) | Best owned: 0.97054 (greedy74) / 0.96858 single (EXP-130)

## Active
| ID | Hypothesis | Evidence | Expected Value | Cost | Depends | Status | Next |
|---|---|---|---|---|---|---|---|
| EXP-132 | lookup-transformer (exact-value embeddings + transformer, 3 seeds) | Library +0.00095 blend, lookup most decorrelated | +0.0005–0.0015 OOF if complementary | GHA 3h | None | RUNNING 32980487428 | consume OOF, measure corr vs frontier/greedy74 |
| — | Frontier attack blends (frontier + lookup + greedy74 + EXP-130) | Frontier is strong external base; owned signal needed | Beat 0.97128 | Analysis | EXP-132 OOF | PREPARED (frontier_attack.py) | run when EXP-132 lands |

## Queued / Ready to Launch (ranked by information gain)
| Rank | Hypothesis | Evidence | Expected | Cost | Trigger |
|---|---|---|---|---|---|
| 1 | Stacking meta-learner (ridge/logistic) over EXP-132 + greedy74 | Stacking research track | +0.0002–0.0006 if complementary | Analysis | EXP-132 OOF shows corr <0.985 |
| 2 | Residual-targeted model (train on errors of greedy74) | If EXP-132 disagreement clusters are structured | +0.0003 | GHA 1h | Error analysis finds structure |
| 3 | Categorical interaction embeddings (stress_level x academic_work_impact etc.) | Library notes: exact-value lookups replicate | +0.0002 | GHA 1h | If lookup shows categorical complementarity |

## Recently Closed (do not re-queue)
- 10-fold vs 5-fold: +0.00036 (EXP-130), keep as ingredient
- Weekend slack: +0.000002 (EXP-129) CLOSED
- Digit family: +0.000013 (EXP-131) CLOSED
- Pair lattice clean: -0.00088 (EXP-128) CLOSED

## Next Action
- When EXP-132 completes: immediate OOF analysis (standalone, corr, residual) → decide blend vs residual model → queue next.
- Parallel: consume Kaggle intelligence + error analysis agent reports when they arrive → promote to queue.
