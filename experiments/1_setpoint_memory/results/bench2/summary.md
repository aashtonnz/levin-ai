| Mechanism | Commits to start side (t=100) | Still holds it (t=1000) | Holds it after damage | Rewrite took (t=300) | Rewrite still holds (t=1000) | Params | Train s |
|---|---|---|---|---|---|---|---|
| Plain | 100% (8/8 seeds >= 90%) | 81% (5/8 seeds >= 90%) | 99% (8/8 seeds >= 90%) | 96% (7/8 seeds >= 90%) | 71% (5/8 seeds >= 90%) | 1128 | 625 |
| Hidden switches | 100% (8/8 seeds >= 90%) | 59% (3/8 seeds >= 90%) | 86% (4/8 seeds >= 90%) | 67% (4/8 seeds >= 90%) | 38% (1/8 seeds >= 90%) | 1128 | 704 |
| Switch = output | 88% (7/8 seeds >= 90%) | 75% (5/8 seeds >= 90%) | 74% (1/8 seeds >= 90%) | 35% (2/8 seeds >= 90%) | 22% (0/8 seeds >= 90%) | 1128 | 684 |
| Learned gate | 100% (8/8 seeds >= 90%) | 74% (4/8 seeds >= 90%) | 97% (7/8 seeds >= 90%) | 85% (6/8 seeds >= 90%) | 71% (4/8 seeds >= 90%) | 1392 | 739 |

Seeds reliable on ALL five tests (>= 90% each): Plain 5/8, Hidden switches 0/8, Switch = output 0/8, Learned gate 3/8
