# Reduced CPU check of idea 4 (2 seeds, 600 training iterations; boundary rule, coupling cost 0.02)

## learned coupling

| Steering method | New patterns correct | After damage | Cost |
|---|---|---|---|
| encoder (one-shot guess) | 44% | 38% | 1 encoder pass |
| local search on panel | 38% | 38% | 128 batched trial runs, 287s |
| gradient-tuned codes | 6% | 6% | 200 gradient steps, 495s |

Mean coupling on TRUE (hidden) tissue boundaries: 0.81; inside tissues: 0.84
Strongly coupled groups vs true tissues (adjusted Rand index, 1 = perfect, 0 = chance): 0.00 (per seed [0.0, 0.0]); groups found per seed [1, 1]

Total time 2735s

## fixed coupling

| Steering method | New patterns correct | After damage | Cost |
|---|---|---|---|
| encoder (one-shot guess) | 28% | 17% | 1 encoder pass |
| local search on panel | 42% | 42% | 128 batched trial runs, 140s |
| gradient-tuned codes | 17% | 14% | 200 gradient steps, 698s |

Total time 2716s
