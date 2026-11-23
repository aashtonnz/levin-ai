# Structure discovery + search (8 tissues x 4 options, 8 seeds; same code for every layout)

## Layout: left (tissue n depends on n[-1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 77% / 66% | 100% | - |
| Discover map, then solve (not told the layout) | 76% / 66% | 92% | 9 probe + 34 solve (19 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 77% / 66% | 100% | 32 (11 s) |

Discovered influence map exactly right for 88% of models.

## Layout: right (tissue n depends on n[1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 73% / 59% | 100% | - |
| Discover map, then solve (not told the layout) | 72% / 59% | 90% | 9 probe + 33 solve (13 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 21% / 18% | 88% | 32 (8 s) |

Discovered influence map exactly right for 88% of models.

## Layout: both (tissue n depends on n[-1, 1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 0% / 0% | 100% | - |
| Discover map, then solve (not told the layout) | 0% / 0% | 28% | 9 probe + 20 solve (9 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 0% / 0% | 25% | 32 (7 s) |

Discovered influence map exactly right for 0% of models.

Note: in the 'both neighbours' layout the lower level never learned the task (ceiling 0%), so that row says nothing about the search.
