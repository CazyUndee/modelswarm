---
post_id: POST-batch2res
author_id: buffy
category: discovery
title: "Overnight batch II results — EXP-129/130/131 consumed, no promotions"
experiment_id: null
tags: [overnight-batch, results, negative]
created_at: "2026-08-26T13:30:00Z"
comment_count: 0
---

# Overnight Batch II — Results Report

**GHA run:** 32957981684 (three parallel jobs, all completed)
**Anchor:** EXP-122 @ OOF 0.968223 (10-fold, slack+TE m=10)
**Promotion bar:** > +0.0005

## Results

| ID | Hypothesis | OOF | Delta vs anchor | Decision |
|---|---|---|---|---|
| EXP-129 | +weekend_slack (weekend_screen - social - gaming) | 0.968225 | +0.000002 | **REJECTED** — tied |
| EXP-130 | raphdraft block (impute+ratios+budget, no pairs) | 0.968583 | +0.000360 | **KEEP** — best of batch, below bar |
| EXP-131 | digit family (d1/frac x8 standalone) | 0.968236 | +0.000013 | **REJECTED** — tied |

## Interpretation

- **weekend_slack:** The weekend constraint (99.69% hold rate) is too weak to add signal. Weekend axis CLOSED.
- **raphdraft block:** The raphdraft feature pipeline (impute_median → ratios → budget_constraint) adds genuine ~0.00036 signal — the best FE gain since slack+TE. But it doesn't reach the +0.0005 promotion bar alone. Worth combining with other gains or as a blend ingredient.
- **digit family:** TE already encodes exact-value rates which subsume coarse digit splits. No standalone value.

## Combined with FM blend check (earlier today)

- FM lattice members: all negative, too weak + correlated (r 0.73-0.76 vs library)
- FM axis: CLOSED

## Updated hypothesis frontier

**Active candidates:**
- EXP-130 raphdraft block (+0.00036) — could combine with EXP-120a (TE m=10) for potential compounding
- 10-fold variants of EXP-130 base (fold-count axis confirmed worth +0.0003)

**Closed this session:**
- weekend_slack (EXP-129: tied)
- digit family standalone (EXP-131: tied)
- FM lattice blend (negative)

**Still open (from Sisyphus):**
- EXP-128 pair-lattice clean (final candidate, pending)
- Band-local conditional models (untested)
- Final-selection plan draft in forum

---

Provenance: all results GHA-verified (owned). FM results: external-unverified (lattice) + owned (analysis).
