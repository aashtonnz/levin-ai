# v6: built-in switch channels (fixed double-well), no coordinates, 3 seeds each

| Test | Short training + switches | Long training + switches |
|---|---|---|
| Reach goal (MSE) | 0.045, 0.027, 0.034 | 0.069, 0.074, 0.088 |
| After damage (MSE) | 0.072, 0.055, 0.041 | 0.068, 0.068, 0.093 |
| Long run 400 (MSE) | 0.132, 0.082, 0.047 | 0.069, 0.066, 0.093 |
| Commits to own side, 64 steps (%) | 100, 100, 100 | 0, 0, 0 |
| ...after 400 steps (%) | 0, 0, 0 | 0, 0, 0 |
| ...after 1000 steps (%) | 0, 0, 0 | 0, 0, 0 |
| ...after damage (%) | 50, 0, 0 | 0, 0, 0 |
| Rewrite holds 'right' (%) | 0, 0, 0 | 0, 0, 0 |

Without switches (v4 short / v5 long): short reach ~0.013, own side 100% at 64 steps, 0% at 400; long reach 0.03-0.08, 0% commitment.
