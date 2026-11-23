# Spectral parts + per-seed best-checkpoint (early stopping), 8 seeds, T4

## Part discovery
- Spectral: 9 parts in 8/8 seeds; match to true tissues (ARI) 0.92 (old method 0.72)

## Steering to new patterns (% fully correct)

| Goal panel | Parts | Encoder | Search | Gradient-tuned codes |
|---|---|---|---|---|
| discovered parts (spectral) | 9 | 88% | 88% | 70% |
| true tissues (best case) | 9 | 88% | 88% | 88% |

| Seed | Match (ARI) | Discovered: search | True tissues: search |
|---|---|---|---|
| 0 | 1.00 | 100% | 100% |
| 1 | 0.90 | 100% | 100% |
| 2 | 0.86 | 100% | 100% |
| 3 | 0.85 | 6% | 100% |
| 4 | 0.99 | 100% | 100% |
| 5 | 0.99 | 100% | 100% |
| 6 | 0.87 | 100% | 0% |
| 7 | 0.92 | 100% | 100% |

Early stopping fixed the late blow-ups (seed 0 true panel now 100%).
Remaining failures are different seeds in each panel (discovered: seed 3, true: seed 6) and their
held-out error stays high from early on (e.g. 0.01 vs 0.0003) -> occasional failure to learn at all,
not a parts problem. Next fix: detect via held-out error and restart that seed.
