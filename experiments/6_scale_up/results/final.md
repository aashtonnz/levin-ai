# Scale-up results (final, 2026-11-24)

Steering to new patterns, % fully correct. Pass criterion, fixed in advance: discovered parts within 10 points of
true parts. Validity: true parts reach at least 80%.

| Condition | True parts | Discovered parts | Parts found / match | Verdict |
|---|---|---|---|---|
| Interacting parts (shift rule), 9 tissues | 100% | 98% | 9 in 8 of 8 / 0.91 | PASS (see note) |
| Irregular tissue shapes, 9 tissues | 99% | 51% | 9–14 / 0.66 | FAIL (part-finding) |
| 8 options per part | 2% | 5% | / | NOT VALID (training failed even with true parts) |
| Bigger world, 25 square tissues | 100% | 16% | 18–22 / 0.83 | FAIL (part-finding) |
| Combined (25 irregular, 8 options, shift) | 0% | 0% | 8–40 / 0.49 | NOT VALID (includes 8 options, so training failed) |
| Retraining comparison, 25 known tissues, shift rule | 100% (ceiling 100%) | 100% (discover influence + search) | influence map exact in 8 of 8 | PASS: retrain all weights 0%, prompt-tune 0%, independent search 0% |

Note on the shift rule: with learned codes, the model learned to treat each part's code as "what to show here",
which sidesteps the dependency between neighbours, so this is not a real test of interacting parts. The retraining
comparison (fixed meaning of each input) is the real one.

Main reading: with true parts, steering holds up in larger and irregular worlds; part-finding is the bottleneck.
8 options broke training itself (likely too few spare codes: codes = options = 8).

## Retraining comparison on the big world (shift rule, 5x5 tissues of 5x5 cells, 4 options, 8 seeds)

| Method | Patterns correct | After damage | Settings correct | Cost |
|---|---|---|---|---|
| original task | 100% | 77% | - | - |
| ceiling (correct settings by hand) | 100% | 78% | - | - |
| discover influence + search | 100% | 77% | 85% | 26 probe + 80 search trial runs, 23s |
| independent search (ignores interactions) | 0% | 0% | 40% | 4 trial runs, 9s |
| retrain all weights | 0% | 0% | - | 400 gradient steps, 543s |
| prompt-tune goal inputs | 0% | 0% | - | 400 gradient steps, 437s |

P1 (within 2 points of ceiling): PASS. P2 (at least 30 points above best baseline): PASS. Here each part's input has a
fixed meaning, so neighbours really do interact; searching while ignoring interactions gets 0%.

## Combined world (25 irregular tissues, 8 options, shift rule)

Phase 1 match 0.49 (per seed 0.30–0.62; 8–40 parts). Phase 2: every method 0%, with true and discovered parts alike.
Validity not met, so no conclusion about part-finding. Cause is the same as the 8-option run; retested with 16 codes:
still invalid (true parts 7%), see [codes16.md](codes16.md).

## Overall

- Steering by search scales: 99–100% with true parts on irregular shapes, 25 tissues, and real interactions between
  parts, against 0% for retraining and prompt tuning.
- Part-finding is the bottleneck: discovered parts give 51% (irregular) and 16% (big).
- 8 options breaks training. Spare codes (16) did not fix it (true parts 7%); big parts with many options look
  hard to learn. Both runs with 8 options remain invalid.
