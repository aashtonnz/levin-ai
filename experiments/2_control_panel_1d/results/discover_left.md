## Layout: left (tissue n depends on n[-1])

| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |
|---|---|---|---|
| Correct settings by hand (ceiling) | 77% / 66% | 100% | - |
| Discover map, then solve (not told the layout) | 75% / 64% | 84% | 9 probe + 688 solve (176 s) |
| Reference: 'told the layout' search (assumes left-neighbour chain) | 77% / 66% | 100% | 32 (7 s) |

Discovered influence map exactly right for 88% of models.

