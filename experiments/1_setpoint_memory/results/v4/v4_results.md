# v4 results: no coordinate input, edge marker instead (3 seeds each, values per seed)

| Test | NCA | NCA + pool | Hybrid setpoint + pool |
|---|---|---|---|
| Reach goal (MSE) | 0.012, 0.013, 0.014 | 0.066, 0.059, 0.098 | 0.084, 0.083, 0.078 |
| After damage (MSE) | 0.066, 0.134, 0.071 | 0.090, 0.056, 0.110 | 0.086, 0.087, 0.079 |
| Goal swap, held out (MSE) | 0.134, 0.239, 0.114 | 0.092, 0.056, 0.111 | 0.092, 0.086, 0.081 |
| Long run 400 (MSE) | 0.242, 0.433, 0.262 | 0.092, 0.053, 0.109 | 0.096, 0.085, 0.082 |
| Commits to its own side, 64 steps (%) | 100, 100, 100 | 50, 50, 50 | 0, 0, 0 |
| ...still committed after 400 steps (%) | 0, 0, 0 | 0, 50, 50 | 0, 0, 0 |
| ...committed after damage (%) | 0, 0, 0 | 25, 50, 50 | 0, 0, 0 |
| Rewrite holds 'right' after reset (%) | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |

Strict commitment = output within 0.03 MSE of the correct disk (hedge = 0.069, blank = 0.139).
