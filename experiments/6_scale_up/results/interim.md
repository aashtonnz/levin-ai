# Scale-up results so far (2026-11-24; full tables to follow from each run's summary.md)

Steering to new patterns, % fully correct. Pass criterion, fixed in advance: discovered parts within 10 points of
true parts. Validity: true parts reach at least 80%.

| Condition | True parts | Discovered parts | Parts found / match | Verdict |
|---|---|---|---|---|
| Interacting parts (shift rule), 9 tissues | 100% | 98% | 9 in 8 of 8 / 0.91 | PASS (see note) |
| Irregular tissue shapes, 9 tissues | 99% | 51% | 9–14 / 0.66 | FAIL (part-finding) |
| 8 options per part | 2% | 5% | / | NOT VALID (training failed even with true parts) |
| Bigger world, 25 square tissues | 100% | 16% | 18–22 / 0.83 | FAIL (part-finding) |
| Combined (25 irregular, 8 options, shift) | running | running | 24–28 / 0.49 | |
| Retraining comparison, 25 known tissues | running | | | |

Note on the shift rule: with learned codes, the model learned to treat each part's code as "what to show here",
which sidesteps the dependency between neighbours, so this is not a real test of interacting parts. The retraining
comparison (fixed meaning of each input) is the real one.

Main reading: with true parts, steering holds up in larger and irregular worlds; part-finding is the bottleneck.
8 options broke training itself (likely too few spare codes: codes = options = 8).
