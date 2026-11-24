# T-17: nested squares as parts (2026-11-24, T4, 8 seeds, 4 options, irregular worlds)

Idea (from the user): instead of discovering irregular parts, use squares within squares, like pixel resolution, so
any region shape can be approximated. Every cell gets one code from each level; search sets coarse codes first.
Criteria fixed before running (run_hier.sh): H1 nested 3,6 on world 0 best search >= 89%; H2 nested 3,6 on world 1
within 10 points of true parts on world 1.

| Run | Parts | Encoder (one-shot) | Search (probe influence) | Search (own cells) | Gradient-tuned codes | Best search |
|---|---|---|---|---|---|---|
| Nested 3,6, world 0 | 9 + 36 | 100% | 9% | 2% | 7% | 9% |
| Nested 3,6, world 1 | 9 + 36 | 100% | 5% | 1% | 2% | 5% |
| Big squares only (3), world 0 | 9 | 30% | 51% | 36% | 25% | 51% |
| Small squares only (6), world 0 | 36 | 100% | 4% | 0% | 1% | 4% |
| True parts, world 1 | 9 | 99% | 96% | 100% | 61% | 100% |
| (earlier) true parts, world 0 | 9 | | | | | 99% |
| (earlier) discovered parts, world 0 | 9–14 | | | | | 51% |

Training: nested runs reached held-out error about 0.0003 on every seed, as low as true parts.

## Verdict

H1 FAIL (9% vs 89% needed). H2 FAIL (5% vs 90% needed).

## Reading

- Nested squares carry enough information: the model learns the task fully, and the one-shot encoder picks correct
  codes for new patterns 100% of the time on both worlds, with no part-finding at all.
- But the codes are not searchable. Trying one square's code at a time almost never reaches a working combination,
  with or without the coarse level. The same happened with small squares alone and with over-split discovered parts.
- Likely cause: training only ever sees codes chosen by the encoder, so nothing makes each square's code mean
  something by itself; the model learns a joint, distributed use of all the codes. With true parts, each part's
  code maps to its own tissue's pattern, so search works.
- So the obstacle has moved: from "find the right parts" to "make each part's code mean something locally".
- Ideas to test next: train with codes that are sometimes set at random or one at a time (so each code's local
  effect must be learned), or add a per-square loss; then search should work. Also: report the one-shot encoder as a
  steering method in its own right (100% here), with its limit (only for targets of the kind it was trained on).
