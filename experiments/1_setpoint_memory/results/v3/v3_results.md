# v3 results (bistable share of training = 35%, 3 seeds each)

| Test | NCA + pool | Hybrid setpoint + pool |
|---|---|---|
| Reach goal (MSE) | 0.006, 0.013, 0.007 | 0.040, 0.063, 0.063 |
| After damage (MSE) | 0.014, 0.026, 0.011 | 0.041, 0.063, 0.063 |
| Goal swap, held out (MSE) | 0.018, 0.037, 0.015 | 0.059, 0.072, 0.074 |
| Long run 400 (MSE) | 0.019, 0.040, 0.017 | 0.044, 0.062, 0.063 |
| Bistable 400 steps (MSE) | 0.138, 0.137, 0.133 | 0.130, 0.127, 0.132 |
| Commits to its own side, 400 steps (%) | 50, 50, 50 | 50, 50, 50 |
| Rewrite test: started committed left (%) | 100, 100, 100 | 0, 0, 0 |
| Rewrite test: committed right after reset (%) | 0, 0, 0 | 100, 100, 100 |

Reference MSE levels on the bistable task: committed correctly ~0, even hedge 0.069, always one side 0.139.
Every run committed to a single fixed side regardless of start: NCA + pool always left, hybrid always right.
