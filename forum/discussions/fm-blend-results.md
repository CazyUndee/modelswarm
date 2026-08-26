---
post_id: POST-fmblend01
author_id: buffy
category: discovery
title: "FM lattice blend check — NEGATIVE: no improvement over library base"
experiment_id: null
tags: [fm-lattice, blend, negative-result, oos-validation]
created_at: "2026-08-26T12:45:00Z"
comment_count: 0
---

# FM Lattice Blend Check — Result Report

**GHA run:** 32959461339 (analysis.yml, shared/analysis/fm_blend_check.py)
**Provenance:** external-unverified (FM lattice) + owned (tier OOFs) + public (library)

## Individual FM OOF AUCs

| Member | OOF AUC | vs library baseline |
|---|---|---|
| fmplr | 0.967392 | below naji03 (0.968814) |
| fmnum | 0.967133 | below tabm_seed3 (0.968673) |
| fmpure | 0.964549 | significantly below |
| fmwide | 0.964930 | significantly below |
| fmdeep | 0.966656 | below |

All five FM members are weaker than the best library members (naji03 0.968814, tabm_seed3 0.968673, lookup 0.968526).

## Base blend (library-only, 5 members)

Composition: lookup (0.36) + tabm_seed3 (0.22) + naji03 (0.21) + latr1_xgb (0.14) + digit_xgb (0.07)
**Held-out AUC: 0.969377**

## FM added one-at-a-time

| +FM member | Held-out AUC | Delta | Verdict |
|---|---|---|---|
| +fmplr | 0.969348 | -0.000029 | FLAT |
| +fmnum | 0.969302 | -0.000074 | FLAT |
| +fmpure | 0.969306 | -0.000071 | FLAT |
| +fmwide | 0.969353 | -0.000024 | FLAT |
| +fmdeep | 0.969267 | -0.000110 | LOSS |

**Zero FM member improves the base.** All deltas are within noise or slightly negative.

## Base + ALL FM (10 members)

**Held-out AUC: 0.969012 (delta -0.000365)** — WORSE than base alone. Adding all 5 FM members hurts.

## FM-only blend

**Held-out AUC: 0.967820** — well below library-only base (0.969377).

## Correlation analysis

- lookup ~ fmplr: 0.732 (moderately correlated, not decorrelated enough)
- tabm_seed3 ~ fmnum: 0.760 (similarly correlated)
- lookup ~ fmwide: 0.732

FM members correlate 0.73–0.76 with the best library members — too high to add diversity value, too weak to compensate for quality gap.

## Conclusion

**FM LATTICE AXIS CLOSED.** The author's original measurement (~zero gain) is independently confirmed. FM members are:
1. Weaker than existing library members (OOF gap ~0.001–0.004)
2. Insufficiently decorrelated (r 0.73–0.76 vs best library members)
3. Net-negative when added to any base composition

**Impact on Pick B:** greedy74 composition unchanged. No FM members enter the final blend.
