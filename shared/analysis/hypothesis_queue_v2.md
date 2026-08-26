# Hypothesis Queue v2 — Updated 2026-08-26 17:15 UTC

## Current Frontier
- **Live LB top:** 0.97186 (Chris Deotte)
- **Naji_19_blend:** OOF 0.970099 (projected LB ~0.97138)
- **Tamerlan full blend:** OOF 0.969487 (reported 0.969476)
- **Best owned blend:** OOF 0.969587 (tabm_seed3 + naji03 on top of Tamerlan)
- **Gap to frontier:** +0.00127 (from best owned OOF to 0.97186 LB)

## Key Intelligence (This Session)

### Tamerlan's 9-model blend
- NN gets 63% weight (lookup_transformer 20.2%, DL 28.8%, MLP 16.1%)
- Trees get 37% (catboost 12.8%, xgb 8.9%, lgb 7.1%)
- **Probability-space beats rank-space** (0.969487 vs 0.969470)
- CatBoost most decorrelated with lookup_transformer (0.967 Spearman)

### Naji's 14-model collection
- naji_19_blend alone scores 0.970099 — already a complete blend
- Our models HURT when added (-0.0014 to -0.0018)
- All Naji models highly correlated (0.982-0.990)

### DL Model Investigation
- dl_s7 vs dl_s23: Spearman 0.983, different seeds of same architecture
- DL helps in low screen-time (+0.0035 at [0,3h)), hurts in high (-0.127 at [12,24h])
- DL-only blend (0.969217) beats tree-only (0.968820)
- Combined blend: 0.969507

## Active Hypotheses

| ID | Hypothesis | Evidence | Status | Owner |
|---|---|---|---|---|
| EXP-132 | lookup-transformer (exact-value embeddings + transformer) | Tamerlan #1 model (20.2% weight) | RUNNING | orchestrator |
| EXP-133 | Meta-feature polynomial interaction stack | Community evidence CV 0.96947 | PREPARED | orchestrator |
| TabFM | Google tabular foundation model (in-context learning) | Released June 2026, scikit-learn API | RUNNING | buffy |
| H3 | Rank-space blending beats prob-space | Tamerlan uses rank-space | READY | orchestrator |
| H4 | CatBoost training setup matters | 16.9% weight in Tamerlan vs our 0% | READY | orchestrator |
| H5 | DL architecture inference | dl_s7/dl_s23 unpublished | READY | orchestrator |
| H6 | Missing mechanism for 0.97186 gap | No blend reaches it | RESEARCH | orchestrator |

## Recently Closed (do not re-queue)
- Band-specific blending: negligible gain (orchestrator confirmed)
- FM lattice blend: negative (all FM models too weak)
- Weekend slack: tied (+0.000002)
- Digit family standalone: tied (+0.000013)
- Pair lattice clean: -0.00088
- Nested TE: -0.00062

## Priority Next Actions
1. **Consume EXP-132** when it lands — compare our lookup-transformer vs Tamerlan's
2. **Consume TabFM results** — test if foundation model adds value
3. **Build EXP-133** with lookup-transformer OOF included (orchestrator handling)
4. **Investigate what DL models are** — search for similar architectures
5. **Test our models on top of Naji_19_blend** — can we improve the strongest blend?

## Files Available
- Tamerlan: `/tmp/blend_npy/blend_data/` (9 OOF + test npy, blend_config.json)
- Naji: `/tmp/naji_oof/naji_data/` (14 OOF + submission CSVs)
- Our vectors: `shared/artifacts/stacking_vectors/` (EXP-122, EXP-130)
- Library: kagglehub (lookup, tabm_seed3, naji03, etc.)
