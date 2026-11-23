# Making the parts more competent: neighbour-goal sensing (8 tissues x 4 options, 8 seeds)

## Layout: left (tissue n depends on n[-1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 69% / 57% | 100% | - |
| Discover map, then solve (not told the layout) | 68% / 56% | 84% | 9 probe + 34 solve (14 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 68% / 56% | 89% | 32 (8 s) |

Discovered influence map exactly right for 88% of models.

## Layout: both (tissue n depends on n[-1, 1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 0% / 0% | 100% | - |
| Discover map, then solve (not told the layout) | 0% / 0% | 28% | 9 probe + 38 solve (14 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 0% / 0% | 32% | 32 (9 s) |

Discovered influence map exactly right for 25% of models.


Per-tissue accuracy with correct settings (all 8 tissues must be right for a pattern to count):

| Lower level | Tissues correct | By position (0..7) | All 8 correct |
|---|---|---|---|
| Original, left layout | 96% | 98 94 96 93 95 96 95 94 | 79% |
| Original, both layout | 2% | 11 0 0 0 0 0 0 3 | 0% |
| Sensing, left layout | 87% | 82 88 86 84 91 87 89 87 | 70% |
| Sensing, both layout | 19% | 74 0 0 0 0 0 0 76 | 0% |
