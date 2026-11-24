# T-16: 8 options with 16 codes (2026-11-24, T4, 8 seeds)

Question: does training with 8 options fail because codes = options = 8 leaves no spare codes?
Test: rerun the 8-option world (15x15, 9 square tissues, independent rule) with 16 codes. Validity bar, fixed
before: true parts reach at least 80% with the best search.

## Result: NOT VALID again (true parts 7%, was 2% with 8 codes)

| Goal panel | Parts | Encoder | Search (own cells) | Search (probe influence) | Gradient-tuned codes |
|---|---|---|---|---|---|
| discovered parts | 15 | 51% (51%) | 17% (18%) | 16% (15%) | 16% (16%) |
| true tissues | 9 | 8% (4%) | 7% (5%) | 6% (6%) | 9% (3%) |

Phase 1: match 0.71 (per seed 0.33–1.00); parts after merging [26, 9, 26, 9, 25, 9, 9, 9]; phase-1 model's encoder
accuracy 95%.

Per seed:

| Seed | Parts found | Match | Discovered: search | Discovered: encoder | True tissues: search |
|---|---|---|---|---|---|
| 0 | 26 | 0.33 | 3% | 100% | 6% |
| 1 | 9 | 0.88 | 0% | 0% | 3% |
| 2 | 26 | 0.33 | 6% | 100% | 0% |
| 3 | 9 | 1.00 | 0% | 0% | 28% |
| 4 | 25 | 0.34 | 6% | 100% | 6% |
| 5 | 9 | 0.85 | 0% | 0% | 3% |
| 6 | 9 | 0.99 | 9% | 9% | 3% |
| 7 | 9 | 0.93 | 100% | 100% | 0% |

## Reading

- Spare codes are not the main cause. Hypothesis rejected.
- Clue 1: when parts are small (about 25 parts, or the phase-1 model's 3x3 grid parts), the model learns 8 options
  well: encoder 95–100%. With 9 big parts (5x5 cells) it mostly fails, though seed 7 shows it is possible.
  So the difficulty is one code having to drive 8 different patterns across a big part.
- Clue 2: where the encoder reaches 100% with 25–26 parts, search gets only 3–6%. The codes work but search cannot
  find them, perhaps because codes are used in combination across parts rather than one part, one choice.
- Training error with true parts ended at 0.014–0.02 per seed, near the 0.02 pass line per tissue: underfitting.
- Combined world with 16 codes (all_c16) stopped in phase 2 to save GPU: held-out error 0.008–0.043, the same
  pattern as the failed run, and it includes 8 options, so it could not be valid.

Next ideas (not yet tried): longer phase-2 training or a larger network for 8 options; train true-part models
with a curriculum (4 options, then 8).
