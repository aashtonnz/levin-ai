# Natural two-neighbour interaction (boundary marking) + improved solver
8 tissues x 4 options, 8 seeds. Same discovery code for every layout; solver now settles a tissue only when the
evidence is unambiguous.

| Layout / lower level | Ceiling (correct settings by hand): patterns / after damage | Discover + solve (not told layout) | Cost (trial runs) |
|---|---|---|---|
| Boundary marking, 32 hidden units | 99% / 83% | 99% / 83%, settings 100% correct | 9 + 142 |
| Boundary marking, 64 hidden units | 100% / 100% | 100% / 100%, settings 100% correct | 9 + 144 |
| Left neighbour (modular rule) | 77% / 66% | 75% / 64%, settings 84% correct | 9 + 688 |
| Right neighbour (modular rule) | 73% / 59% | 72% / 59%, settings 93% correct | 9 + 48 |

Earlier two-neighbour rule (sum of three settings mod 4): ceiling 0% (middle tissues never learned it).
First version of the solver on boundary marking: 0% (it locked in arbitrary settings when several looked equally good).
