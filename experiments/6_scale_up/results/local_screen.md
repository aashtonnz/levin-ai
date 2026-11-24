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
