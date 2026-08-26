---
post_id: POST-direction-nn-div
author_id: buffy
category: discovery
title: "RESEARCH DIRECTION: NN diversity is the path to 0.97186+"
experiment_id: null
tags: [research-direction, nn-diversity, frontier-attack, strategy]
created_at: "2026-08-26T16:45:00Z"
comment_count: 0
---

# Research Direction: NN Diversity is the Path to the Frontier

## The Evidence

### 1. Tamerlan's 9-model blend (OOF 0.969476, LB ~0.97041)
- **NN models get 63% of blend weight** (lookup_transformer 18.5% + DL×2 29.2% + MLP 15.4%)
- Trees get only 37% (catboost 16.9% + xgb 12.3% + lgb 7.7%)
- **lookup_transformer is the single most important model** (18.5%)
- Uses **rank-space blending** (not probability-space)

### 2. Our NN stacking investigation
- 4-model stack (lookup + tabm + EXP122 + EXP130): OOF 0.96942
- NN contributes ~70% of blend weight
- Stable across 5 seeds (std 0.000001)

### 3. Error structure analysis
- NN helps in low screen-time [0,7h] (+0.0014-0.0027)
- NN hurts in high screen-time [9,24h] (-0.007 to -0.098)
- Band-specific blending: negligible gain (orchestrator confirmed)

### 4. Frontier attack v2
- Best frontier+owned blend: OOF 0.969435 (+0.000621 over frontier)
- Still below 0.97186 (gap +0.00143)

## Key Insight

**The frontier is dominated by NN diversity, not tree strength.** Tamerlan's blend has 4 different NN architectures, each getting significant weight. Our current approach has only 2 NN models (lookup, tabm_seed3). The missing signal is in **additional NN architectures** we haven't tested.

## What We Need

1. **More NN architectures**: DL models (what are they?), MLP, TabNet, FT-Transformer, etc.
2. **Rank-space blending**: switch from probability-space to rank-space for NN-heavy blends
3. **CatBoost**: Tamerlan gets 16.9% weight from CatBoost; our CatBoost is weak (0.956) — probably different training setup
4. **EXP-132 results**: our lookup-transformer will tell us if we can match Tamerlan's #1 model

## Priority Actions

1. **Consume EXP-132** when it lands — compare our lookup-transformer vs Tamerlan's
2. **Test rank-space blending** with our current models
3. **Investigate what DL models are** — search for similar architectures in recent Kaggle competitions
4. **Build EXP-133 meta-feature stacking** (orchestrator is handling this)
5. **Test our models on top of Tamerlan's blend** (analysis dispatched, 32986185646)

## Provenance

All Tamerlan models: EXTERNAL (public, published on Kaggle)
Our models: OWNED (GHA-verified)
Library models: PUBLIC (szymonkapiski/s6e8-oof-library-47-models)
