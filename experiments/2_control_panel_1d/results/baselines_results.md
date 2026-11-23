# Stronger comparisons, interacting parts (8 tissues x 4 options, 8 seeds)

Retraining all weights on the same 8 examples (earlier run): 1% / 1%, 400 gradient steps.

| Method | Correct patterns | After damage | Cost | Settings correct |
|---|---|---|---|---|
| Correct settings given by hand (ceiling) | 77% | 66% | - |  |
| Gradient-tuned goal inputs ('prompt tuning'), 100 steps | 0% | 0% | ~300 rollout-equivalents, 69 s |  |
| Gradient-tuned goal inputs ('prompt tuning'), 300 steps | 0% | 0% | ~900 rollout-equivalents, 197 s |  |
| Local-error search, random halves (not told the layout), 8 rounds | 0% | 0% | 32 rollouts, 8 s | 53% |
| Local-error search, random halves (not told the layout), 16 rounds | 5% | 4% | 64 rollouts, 13 s | 79% |
| Local-error search, alternating neighbours (told the layout), 8 rounds | 77% | 66% | 32 rollouts, 8 s | 100% |
| Search judged by TOTAL error only (no local errors), 1 sweep | 3% | 3% | 128 rollouts, 25 s | 54% |
| Gradient-tuned goal inputs, 300 steps, step size 0.3 | 0% | 0% | ~900 rollout-equivalents, 208 s |  |
| Gradient-tuned goal inputs, 300 steps, step size 1.0 | 0% | 0% | ~900 rollout-equivalents, 209 s |  |
| Local-error search, random halves (not told the layout), 32 rounds | 0% | 0% | 128 rollouts, 24 s | 47% |
| Local-error search, random halves (not told the layout), 64 rounds | 0% | 0% | 256 rollouts, 45 s | 46% |
