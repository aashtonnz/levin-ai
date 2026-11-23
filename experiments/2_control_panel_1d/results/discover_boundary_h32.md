## Layout: boundary (tissue n depends on n[-1, 1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 99% / 83% | 100% | - |
| Discover map, then solve (not told the layout) | 99% / 83% | 100% | 9 probe + 142 solve (40 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 99% / 83% | 100% | 32 (8 s) |

Discovered influence map exactly right for 88% of models.

