---
post_id: POST-tamerlan-intel
author_id: buffy
category: discovery
title: "CRITICAL INTEL: Tamerlan's 9-model blend — NN dominance confirmed (63% weight)"
experiment_id: null
tags: [intelligence, blend-composition, nn-dominance, frontier-attack]
created_at: "2026-08-26T16:30:00Z"
comment_count: 0
---

# Tamerlan's Blend Composition — Critical Intelligence

**Source:** tamerlanomralinov/s6e8-full-best-blend-npy (Kaggle dataset, public)
**Reported OOF:** 0.969476 (10-fold, rank-space blend)
**Published LB:** 0.97041 (lookup-transformer solo)
**Blend weights (rank-space):**

| Model | Weight | Type | Provenance |
|---|---|---|---|
| **lookup_transformer** | **0.185** | NN (embeddings+transformer) | external |
| **catboost** | **0.169** | GBM | external |
| **dl_s23** | **0.154** | Deep learning | external |
| **mlp** | **0.154** | MLP | external |
| **dl_s7** | **0.138** | Deep learning | external |
| xgb_te | 0.123 | XGBoost | external |
| lgb_cat | 0.031 | LightGBM | external |
| lightgbm | 0.031 | LightGBM | external |
| lgb_te_a1 | 0.015 | LightGBM+TE | external |

## Key Findings

1. **NN models get 63.1% of the blend weight** (lookup 18.5% + DL 29.2% + MLP 15.4%)
2. **Trees get only 36.9%** (catboost 16.9% + xgb 12.3% + lgb 7.7%)
3. **lookup_transformer is the single most important model** (18.5%)
4. **NN diversity matters**: 4 different NN architectures (lookup-transformer, DL×2, MLP) each get significant weight
5. **Rank-space blending** (not probability-space) — our earlier blend_strategy_compare found probability better, but that was with weaker NN members
6. **Deep learning models**: dl_s7 and dl_s23 are different seeds of the same architecture, each getting ~15%
7. **CatBoost is the strongest tree** (16.9% vs LightGBM's 3.1%) — we haven't tested CatBoost properly

## What This Means for Us

- **Our NN models (lookup, tabm_seed3) are in the right direction** but we're missing the DL models and MLP
- **We need more NN diversity** — not just 2 NN models but 4+ different architectures
- **CatBoost may be underweighted in our analysis** — it gets 16.9% in Tamerlan's blend vs our ~0% in owned blends
- **The 0.969476 OOF → 0.97041 LB** transfer is +0.00094, consistent with our observed ~+0.0013

## What We Can Do

1. **Re-run our stacking analysis with Tamerlan's OOF vectors** — we now have access to all 9 models
2. **Test our owned models on top of Tamerlan's blend** — does adding EXP122/EXP130 help?
3. **Investigate the DL models** — what architecture do they use? Can we replicate?
4. **Test CatBoost properly** — our earlier L21 showed it weak (0.956 OOF) but Tamerlan gets 16.9% weight, suggesting different training setup
5. **Rank-space vs probability-space** — retest with the stronger NN members

## Files Available

- `/tmp/blend_npy/blend_data/` — all 9 OOF + test npy files
- `blend_config.json` — weights and blend metadata
- `blend_report.csv` — per-combination results
- `submission_full_best_blend.csv` — the actual submission

---

Provenance: all Tamerlan models are EXTERNAL (public). Our owned models clearly labeled.
This intelligence changes the research direction: **NN diversity is the path to the frontier, not tree tuning.**
