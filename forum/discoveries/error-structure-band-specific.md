---
post_id: POST-errstruct01
author_id: buffy
category: discovery
title: "Error structure + frontier attack results — band-specific blending hypothesis"
experiment_id: null
tags: [error-structure, frontier-attack, band-specific, discovery]
created_at: "2026-08-26T16:10:00Z"
comment_count: 0
---

# Error Structure + Frontier Attack Findings

## Error Structure Analysis (GHA run 32982873804)

### Key Finding: NN helps in low screen-time, hurts in high

| Band | NN AUC | Tree AUC | Δ (NN-tree) | Blend AUC |
|---|---|---|---|---|
| [0,3h) | 0.9007 | 0.8980 | **+0.0027** | 0.9051 |
| [3,5h) | 0.9180 | 0.9155 | **+0.0025** | 0.9195 |
| [5,7h) | 0.9220 | 0.9206 | **+0.0014** | 0.9233 |
| [7,9h) | 0.9469 | 0.9465 | +0.0005 | 0.9485 |
| [9,12h) | 0.9669 | 0.9739 | **-0.0070** | 0.9760 |
| [12,24h] | 0.8852 | 0.9835 | **-0.0982** | 0.9841 |

**Interpretation:** NN captures signal that trees miss in the low screen-time region (where the decision boundary is harder), but trees completely dominate in high screen-time (where the signal is simpler/stronger). A band-specific blend should exploit this.

### Error Concentration
- ~77-81k rows in low-confidence band [0.3, 0.7]: AUC only 0.63-0.64
- ~530-540k rows in high-confidence: AUC 0.99+
- Error is concentrated in [0,7h) screen-time (14-16% error rate)

## Frontier Attack v2 (GHA run 32982682506)

### Best frontier+owned blend: Frontier + NN + EXP130

- **OOF: 0.969435** (Δ +0.000621 over frontier alone)
- **Extremely robust:** std across 5 seeds ~0.000001
- **Weights:** naji03=0.223, lookup=0.337, tabm_seed3=0.258, EXP130=0.182
- **NN gets ~60% of weight** even when blended with the frontier

### LB projection
- Frontier OOF 0.968814 → LB 0.97128 (+0.00129 transfer)
- Best blend OOF 0.969435 → projected LB ~0.97073
- **Still below live top (0.97186)** — the frontier already contains most of our signal

### Implication
Adding owned signal to the frontier gives a small but robust gain. But the gap to 0.97186 suggests Chris Deotte's approach uses something fundamentally different (possibly different features, architecture, or training methodology).

## New Hypothesis: Band-Specific Blending

Since NN helps in [0,7h] but hurts in [9,24h], testing per-band weight optimization:
- Low band: higher NN weight, lower tree weight
- High band: lower NN weight, higher tree weight

Script dispatched on GHA (run 32983246997).

## Next Steps
1. Consume band-specific blend results
2. Consume EXP-132 (lookup-transformer) results when they land
3. If band-specific blending helps, test on the full frontier+owned stack
4. Investigate what Chris Deotte's 0.97186 approach might be (different features? architecture?)

---

Provenance: external (naji03/library NN) + owned (EXP122/130)
