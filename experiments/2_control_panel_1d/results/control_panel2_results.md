# Control panel with interacting parts (8 tissues x 4 options, 8 seeds)

Tissue n shows motif (goal_n + goal_{n-1}) mod 4, so each tissue's right setting depends on its left neighbour.
Flat model and panel lower level are the same trained network (identical training); they differ only in how they adapt.

| Model / adaptation | Correct patterns | After damage | Cost |
|---|---|---|---|
| Network on original task (ceiling for this lower level) | 77-80% | 66-69% | - |
| Flat: retrain weights on 8 examples of new meaning | 1% | 1% | 400 gradient steps, 320 s |
| Panel: simple search (tissues judged alone) | 0% | 0% | 4 rollouts, 3 s |
| Panel: neighbourhood search (own + neighbour region) | 5% | 5% | 28 rollouts, 7 s |
| Panel: own-region search, 2 sweeps | 1% | 1% | 16 rollouts, 4 s |
| Panel: own-region search, 4 sweeps | 77% (settings 100% correct) | 66% | 32 rollouts, 8 s |
| Panel: own-region search, 8 sweeps | 77% (settings 98% correct) | 66% | 64 rollouts, 13 s |
