# Spectral part-finder: discovered parts as the goal panel (8 seeds, T4)

Part-finder: spectral clustering of the learned coupling graph (affinity = coupling^2, k from eigengap),
replacing threshold + fragment merging.

## Part discovery
- Match to true tissues (ARI): spectral 0.88 vs old method 0.60
- Parts per seed: spectral [9, 9, 9, 9, 9, 9, 8, 9]; old method [8, 9, 7, 1, 7, 7, 5, 9] (true answer: 9)

## Steering to new patterns (% fully correct)

| Goal panel | Parts | Encoder | Search | Gradient-tuned codes |
|---|---|---|---|---|
| discovered parts (spectral) | 9 | 75% | 76% | 65% |
| true tissues (best case) | 9 | 75% | 75% | 75% |

| Seed | Parts | Match (ARI) | Discovered: search | True tissues: search |
|---|---|---|---|---|
| 0 | 9 | 0.98 | 100% | 0% |
| 1 | 9 | 0.90 | 100% | 100% |
| 2 | 9 | 0.67 | 9% | 100% |
| 3 | 9 | 0.99 | 100% | 100% |
| 4 | 9 | 0.92 | 100% | 100% |
| 5 | 9 | 0.93 | 100% | 100% |
| 6 | 8 | 0.74 | 0% | 0% |
| 7 | 9 | 0.89 | 100% | 100% |

Steering through discovered parts now matches the hand-given best case on average (76% vs 75%).
Failures: seed 2 (9 parts but boundaries misplaced, ARI 0.67), seed 6 (lumped + phase-2 training failure,
which also breaks the true-tissue panel), seed 0's true-tissue panel (training failure).
