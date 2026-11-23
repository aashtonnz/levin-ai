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

## Held-out test (method fixed beforehand), 2026-11-24

| World (not used in development) | Spectral (current) | Average of copies | Change |
|---|---|---|---|
| Irregular, world seed 1 | 0.63 (parts 25, 10, 10, 9, 10, 9, 9, 9) | 0.75 (9 in 7 of 8 copies) | +0.12 |
| Irregular, world seed 2 | 0.69 (parts 14, 9, 9, 9, 9, 9, 9, 9) | 0.75 (9 in all 8) | +0.06 |
| Square, world seed 0 | 0.90 (9 in all 8) | **1.00** (9 in all 8) | +0.10 |
| Large, 25 square tissues | 0.83 (parts 18–25) | 0.89 (24 in all 8; true 25) | +0.06 |
| Combined: 25 irregular, 8 options, shift | 0.49 (parts 8–40) | 0.47 (13 in all 8) | −0.02 |

**Verdict against the criteria fixed in advance (PLAN.md, T-11): FAIL.**
- Mean improvement +0.06 (needed +0.10). Without the combined world, where training itself failed (see the
  8-option result), it would be +0.085: still short.
- No condition got worse by more than 0.03: met.

Reading: averaging copies helps consistently (four of five held-out worlds, never clearly worse) and is exact on
the square world, but the gain is smaller than the bar set. It is a useful improvement, not the fix. The combined
world shows a separate problem: when the model cannot learn the task (8 options), its couplings carry little
information about the parts, and no part-finder can recover them.
