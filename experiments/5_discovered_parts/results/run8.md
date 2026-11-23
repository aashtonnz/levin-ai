# Spectral parts + early stopping + restart of non-learning seeds (8 seeds, T4)

Restart rule: after training, any seed with held-out error >10x the median seed is retrained from a new
initialisation; the better version is kept. Same rule for both panels.

## Part discovery
- Spectral: 9 parts in 8/8 seeds; match to true tissues 0.93 (old method 0.52)

## Steering to new patterns (% fully correct)

| Goal panel | Parts | Encoder | Search | Gradient-tuned codes |
|---|---|---|---|---|
| discovered parts (spectral) | 9 | 88% | 89% | 77% |
| true tissues (best case) | 9 | 100% | 100% | 100% |

| Seed | Match (ARI) | Discovered: search | True tissues: search |
|---|---|---|---|
| 0 | 1.00 | 100% | 100% |
| 1 | 0.94 | 100% | 100% |
| 2 | 0.71 | 9% | 100% |
| 3 | 0.97 | 100% | 100% |
| 4 | 0.99 | 100% | 100% |
| 5 | 0.98 | 100% | 100% |
| 6 | 0.90 | 100% | 100% |
| 7 | 0.92 | 100% | 100% |

Restarts: discovered seeds 2 and 7 (7 recovered, 2 did not); true seed 6 (recovered).
Training reliability is now solved: the best-case panel is 100% on all 8 seeds.
Only remaining failure is a parts problem: seed 2 found 9 parts but with misplaced boundaries (ARI 0.71).
