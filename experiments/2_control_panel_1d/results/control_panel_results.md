# Control panel vs retraining (8 seeds each)

| Setup | Model | Original: patterns / after damage | Reshuffled meaning: patterns / after damage | Cost to retarget |
|---|---|---|---|---|
| 4 tissues x 4 options (256 combos) | Flat (retrain weights) | 100% / 100% | 88% / 85% | 400 gradient steps, 181 s |
| 4 tissues x 4 options (256 combos) | Control panel (search) | 100% / 100% | 100% / 100% | 4 rollouts (no gradients), 2 s |
| 8 tissues x 4 options (65,536 combos) | Flat (retrain weights) | 100% / 95% | 73% / 57% | 400 gradient steps, 312 s |
| 8 tissues x 4 options (65,536 combos) | Control panel (search) | 100% / 96% | 100% / 96% | 4 rollouts (no gradients), 3 s |
