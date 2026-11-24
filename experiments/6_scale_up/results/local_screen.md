# T-18 screen: random codes during training (2026-11-24, T4, 2 seeds, irregular world 0, 4 options)

Rule fixed before running: pass if best search >= 80% (mean of 2 seeds) and encoder >= 90%.

| Variant | Code noise | Held-out error | Encoder | Search (probe influence) | Search (own cells) | Gradient codes | Screen |
|---|---|---|---|---|---|---|---|
| (T-17) small squares | 0 | 0.0003 | 100% | 4% | 0% | 1% | |
| small squares | 0.3 | 0.0006, 0.0004 | 98% [100, 97] | 28% [28, 28] | 27% [22, 31] | 58% | FAIL |
| small squares | 0.6 | 0.0027, 0.0014 | 83% [75, 91] | 53% [62, 44] | 75% [62, 88] | 33% | FAIL (close) |
| (T-17) nested 3,6 | 0 | 0.0003 | 100% | 9% | 2% | 7% | |
| nested 3,6 | 0.3 | 0.0005, 0.0009 | 91% [100, 81] | 11% [3, 19] | 28% [31, 25] | 17% | FAIL |
| nested 3,6 | 0.6 | 0.006, 0.008 | 22% [44, 0] | 8% [0, 16] | 22% [22, 22] | 11% | FAIL |

## Reading

- None pass, but the idea is confirmed in direction: with small squares, search rises steadily with code noise:
  4% -> 27% -> 75% (one seed 88%). Random codes in training do make each square's code mean something locally.
- The cost: the one-shot encoder drops (100% -> 98% -> 83%) and training error rises at high noise.
- Nested squares do worse than small squares alone here; the big squares' codes do not become local (a random big
  code wipes out the loss over a large area, so there is little signal).
- Follow-up screen (run_local2.sh): small squares with more noise and longer training, to see if search passes 80%
  while the encoder recovers.

## Follow-up screen (small squares only, 2 seeds, world 0)

| Variant | Held-out error | Encoder | Search (probe influence) | Search (own cells) | Screen |
|---|---|---|---|---|---|
| noise 0.6, 5000 steps | 0.0015, 0.0010 | 88% [91, 84] | 47% [72, 22] | 56% [44, 69] | FAIL |
| noise 0.8, 2000 steps | (not read) | 2% [3, 0] | 2% [3, 0] | 6% [12, 0] | FAIL |
| noise 0.8, 5000 steps | 0.008, 0.0041 | 28% [28, 28] | 19% [6, 31] | 20% [3, 38] | FAIL |

Reading: no variant passes. Noise around 0.6 is the best tried (search 56–75% across two runs, single seeds from
22% to 88%); 0.8 is too much (the model barely learns). Longer training at 0.6 did not help search. Seed-to-seed
spread is large, so 2-seed screens are only a rough guide here. Random codes move search from 4% to roughly
50–75% but not reliably to 80%. Next ideas: randomise only one or two squares per example (gentler than a fixed
rate), or train with the search procedure itself in the loop.

## Gentler version: randomise exactly k squares per example (small squares, 4 seeds, world 0)

| Variant | Encoder | Search (probe influence) | Search (own cells) | Gradient codes | Screen |
|---|---|---|---|---|---|
| k = 1 | 100% [100, 100, 100, 100] | 4% [0, 0, 3, 12] | 1% [0, 3, 0, 0] | 18% | FAIL |
| k = 2 | 99% [100, 97, 100, 100] | 7% [9, 6, 3, 9] | 2% [3, 0, 0, 3] | 31% | FAIL |
| k = 4 | 99% [100, 100, 97, 100] | 28% [22, 38, 6, 47] | 13% [0, 31, 9, 12] | 66% | FAIL |
| 60% of squares | 71% [59, 50, 88, 78] | 62% [59, 28, 72, 88] | 66% [69, 31, 75, 91] | 55% | FAIL |

Reading: no variant passes. Across all random-code runs there is a clear trade-off: the more squares are randomised
in training, the better search works and the worse the one-shot encoder works (k=1: 100% / 4%; k=4: 99% / 28%;
60%: 71% / 66%). No setting gets both above the bar, so this approach on its own does not close the gap.
Gradient-tuned codes rise with noise too (18% -> 66% at k=4). Next idea: train with search in the loop.
