# 2D test on a T4 GPU: 3x3 tissues of 5x5 cells, 4 options, 8 seeds

Steering to a new meaning of the goal inputs, from 8 examples. Pass criteria were fixed before running.

## Boundary rule (each tissue shows its own setting and marks edges where neighbours differ)

| Method | Patterns correct | After damage | Settings correct | Cost |
|---|---|---|---|---|
| original task | 98% | 92% | - | - |
| ceiling (correct settings by hand) | 98% | 93% | - | - |
| discover + search | 98% | 93% | 100% | 10 probe + 456 search trial runs, 7s |
| independent search (ignores interactions) | 98% | 93% | 100% | 4 trial runs, 1s |
| retrain all weights | 0% | 0% | - | 400 gradient steps, 65s |
| prompt-tune goal inputs | 0% | 0% | - | 400 gradient steps, 48s |

Influence map discovered exactly for 100% of seeds. Validity OK (98%). P1 PASS (98% vs 98%). P2 PASS (98% vs 0%).

## Shift rule (each tissue shows its left neighbour's setting), with the learning-rate schedule fix

| Method | Patterns correct | After damage | Settings correct | Cost |
|---|---|---|---|---|
| original task | 100% | 87% | - | - |
| ceiling (correct settings by hand) | 100% | 90% | - | - |
| discover + search | 100% | 93% | 75% | 10 probe + 24 search trial runs, 1s |
| independent search (ignores interactions) | 2% | 2% | 50% | 4 trial runs, 1s |
| retrain all weights | 2% | 2% | - | 400 gradient steps, 67s |
| prompt-tune goal inputs | 0% | 0% | - | 400 gradient steps, 50s |

Influence map discovered exactly for 100% of seeds. Validity OK (100%). P1 PASS (100% vs 100%). P2 PASS (100% vs 2%).

"Settings correct" is below 100% for the shift rule because the tissues on the right edge have no visible
effect, so their setting cannot be known from the examples; all patterns are still correct.

The first shift-rule run, without the schedule fix, became unstable late in training and reached only 62%
(validity check not met).
