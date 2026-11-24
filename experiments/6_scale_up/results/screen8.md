# T-16 screen: why does training fail with 8 options? (2026-11-24, T4)

True parts only (9 square 5x5 tissues, 15x15 world, independent rule), 2 seeds per variant, run side by side.
Rule fixed before running: a variant passes if best search >= 80% averaged over its 2 seeds and the control stays low;
a passing variant then gets a full 8-seed run before any claim.

| Variant | Change | Held-out error per seed | Encoder | Best search | Screen |
|---|---|---|---|---|---|
| A control | 8 options, as before | 0.020, 0.025 | 0% | 0% | control (fails, as expected) |
| B longer | 6000 steps instead of 2000 | 0.0044, 0.0153 | 22% [38, 6] | 23% [34, 6] | FAIL |
| C bigger | 128 hidden units instead of 64 | 0.025–0.032 | 0% | 0% | FAIL |
| D curriculum | 4 options for first half, then 8 | 0.015, 0.036 | 5% [9, 0] | 5% [9, 0] | FAIL |
| E 4 options | positive control | 0.0002, 0.0002 | 100% | 100% | control (passes, as expected) |

## Reading

- None of the three fixes pass. Longer training helps a little (one seed reaches 34–38%) but is far from 80%.
  A follow-up 20000-step run ([long8.md](long8.md)) peaked at step 4000 and did not improve after: not a training-time limit.
- A bigger per-cell network does not help, so capacity of the cell network is not the limit.
- Mean error can be low while patterns still fail: a pattern counts only if all 9 tissues are under 0.02 at once.
- Next hypothesis (untested): too few update steps for one code to draw 8 different patterns across a 5x5 part, or
  the encoder/code bottleneck itself. Options: more rollout steps; several codes per part; smaller parts.
- Lesson for screens: running 5 variants at once slowed each about 4-5x; put the long variant on its own.
