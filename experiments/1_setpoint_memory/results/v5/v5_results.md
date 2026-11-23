# v5: plain NCA, no coordinates, trained on long runs (4 segments, ~160 steps), 5 seeds

| Test | Seeds 0-4 |
|---|---|
| Reach goal (MSE) | 0.067, 0.029, 0.080, 0.064, 0.048 |
| After damage (MSE) | 0.064, 0.034, 0.079, 0.065, 0.076 |
| Goal swap, held out (MSE) | 0.064, 0.031, 0.079, 0.060, 0.081 |
| Long run 400 (MSE) | 0.068, 0.032, 0.080, 0.063, 0.088 |
| Commits to own side, 64 steps (%) | 0, 0, 0, 0, 50 |
| ...after 400 steps (%) | 0, 0, 0, 0, 0 |
| ...after 1000 steps (%) | 0, 0, 0, 0, 0 |
| ...after damage (%) | 0, 0, 0, 0, 38 |
| Rewrite holds 'right' (%) | 0, 0, 0, 0, 0 |

Compare v4 plain NCA (short training): reach ~0.013, commits to own side at 64 steps 100% (3/3 seeds), 0% at 400 steps.
Reference: even hedge between the two disks = 0.069 MSE.
