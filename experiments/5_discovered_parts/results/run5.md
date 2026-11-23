# Per-seed check: does part-finding quality predict steering? (8 seeds, T4)

| Goal panel | Parts | Encoder | Search | Gradient-tuned codes |
|---|---|---|---|---|
| discovered parts | 7 | 39% | 39% | 29% |
| true tissues (best case) | 9 | 75% | 75% | 75% |

| Seed | Parts found | Match (ARI) | Discovered: search | Discovered: encoder | True tissues: search |
|---|---|---|---|---|---|
| 0 | 9 | 0.87 | 100% | 100% | 0% |
| 1 | 7 | 0.71 | 6% | 6% | 100% |
| 2 | 9 | 0.84 | 100% | 100% | 100% |
| 3 | 2 | 0.21 | 0% | 0% | 100% |
| 4 | 7 | 0.67 | 3% | 3% | 100% |
| 5 | 7 | 0.73 | 3% | 3% | 100% |
| 6 | 7 | 0.63 | 0% | 0% | 0% |
| 7 | 9 | 0.70 | 100% | 100% | 100% |

Correlation of ARI with discovered-parts search (6 seeds where true tissues work): 0.47.
Clearer predictor: exactly 9 parts -> 100% (3/3 seeds); any lumping -> 0-6% (5/5 seeds).
True-tissue failures (seeds 0, 6) are phase-2 training failures, not part problems.
