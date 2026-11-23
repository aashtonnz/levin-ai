# CPU check before the full run (2 seeds, 600 training iterations, 60 baseline steps - reduced, not the final test)

## 2D test (shift rule): 3x3 tissues of 5x5 cells, 4 options, 2 seeds (cpu)

| Method | Patterns correct | After damage | Settings correct | Cost |
|---|---|---|---|---|
| original task | 100% | 83% | - | - |
| ceiling (correct settings by hand) | 100% | 80% | - | - |
| discover + search | 100% | 77% | 75% | 10 probe + 24 search trial runs, 3s |
| independent search (ignores interactions) | 0% | 0% | 47% | 4 trial runs, 2s |
| retrain all weights | 2% | 1% | - | 60 gradient steps, 39s |
| prompt-tune goal inputs | 0% | 0% | - | 60 gradient steps, 32s |

Influence map discovered exactly for 100% of seeds.
Validity check (ceiling >= 80%, i.e. the parts are competent): OK (100%)
Pre-registered P1 (within 2 points of ceiling): PASS (100% vs 100%)
Pre-registered P2 (>= 30 points above best baseline): PASS (100% vs 2%)
Total time 454s

## 2D test: 3x3 tissues of 5x5 cells, 4 options, 2 seeds (cpu)

| Method | Patterns correct | After damage | Settings correct | Cost |
|---|---|---|---|---|
| original task | 93% | 62% | - | - |
| ceiling (correct settings by hand) | 95% | 59% | - | - |
| discover + search | 95% | 59% | 100% | 10 probe + 530 search trial runs, 24s |
| independent search (ignores interactions) | 95% | 59% | 100% | 4 trial runs, 2s |
| retrain all weights | 0% | 0% | - | 60 gradient steps, 37s |
| prompt-tune goal inputs | 4% | 2% | - | 60 gradient steps, 32s |

Influence map discovered exactly for 0% of seeds.
Validity check (ceiling >= 80%, i.e. the parts are competent): OK (95%)
Pre-registered P1 (within 2 points of ceiling): PASS (95% vs 95%)
Pre-registered P2 (>= 30 points above best baseline): PASS (95% vs 4%)
Total time 454s
