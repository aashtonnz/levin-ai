# Part-finding methods on development data

Data: phase-1 couplings from the scale-up irregular run (15×15 cells, 9 irregular tissues, world seed 0, 8 copies).
Score: match of found parts to the true tissues (adjusted Rand index, 1 = perfect). The methods never see the truth.

| Method | Match | Parts found (true: 9) |
|---|---|---|
| spectral (current) | 0.66 | 9, 11, 9, 14, 9, 11, 9, 10 |
| modularity-chosen size | 0.65 | 9, 10, 8, 12, 11, 11, 9, 10 |
| spectral + merge | 0.57 | 8, 11, 8, 5, 7, 11, 8, 8 |
| over-segment + merge | 0.41 | 10, 12, 13, 8, 11, 14, 9, 11 |
| average of copies | **0.76** | 9 in all 8 |
| agreement of copies | 0.72 | 9 in all 8 |

Changing how cells are grouped did not help; combining evidence across the independently trained copies did.
"Average of copies" (average the coupling maps of all copies, then spectral clustering) is now fixed and will be
judged on held-out worlds against the criteria in PLAN.md (T-11). Its cost: it needs several trained copies, which
these experiments already train.
