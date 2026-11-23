# Plan

The question: can a network built from competent parts, with a small goal panel per part, be retargeted by searching the panel instead of retraining, and can it find its own parts? Toy grids first; a real task only if the method survives scaling.

## T-3: Scale-up test
Change one thing at a time from the current setup (3×3 tissues of 5×5 cells, 4 options, independent rule):
- 5×5 grid of tissues (25 parts).
- Irregular tissues: random shapes of varied size (for example a Voronoi partition), so spectral clustering cannot rely on a regular layout.
- 8 options per part.
- Shift rule (each tissue shows a neighbour's setting), so search must handle interactions.
Measure for each: parts found and match to true tissues, steering success through discovered and true parts, retraining and prompt-tuning on the same examples, and trial runs used by search. Pass criteria to fix before running: steering through discovered parts within 10 points of true parts, and at least 30 points above the best baseline.

## T-11: Better part-finding (started 2026-11-24)
Scale-up so far: with irregular tissue shapes, steering through true parts reached 99% but through discovered parts
only 51%. Part-finding is the bottleneck. Approach, chosen to avoid tuning to one dataset:
- Methods may not use the true tissues; those are only for scoring. Prefer self-checks the system could really do.
- Development data: phase-1 couplings from the irregular run (world seed 0, 8 seeds).
- Held-out test data, used once, without changes to the method: new irregular worlds (world seeds 1 and 2), the
  square base world, and the large and combined worlds from the scale-up.
- Criteria, fixed before testing: on held-out data, the mean match to true tissues (adjusted Rand index) improves by
  at least 0.10 over the current spectral method, and no condition gets worse by more than 0.03. Then one full run
  on a held-out irregular world: steering through discovered parts within 10 points of true parts.

## T-4: Gate after scale-up
If discovered-part steering holds up, move to the part-quality check and the real-task pilot. If it does not, find which change broke it (size, shape, options or interaction) and rethink part-finding or search for that case before going further.

## T-2: Part-quality check
After finding parts, run search on the training examples. If a part's own error stays high, its boundaries are likely wrong: split it, or rerun clustering with a different number of parts, and keep the version whose search fits the examples best. Uses only the training examples, no true labels.

## T-5: Literature check and real-task pilot choice
Compare with neural cellular automata, modular and mixture-of-experts networks, steering vectors and controllable generation. Choose one real task where parts are plausible, such as image generation with controllable regions or a simulated body with controllable limbs.

## Long-term milestones (agreed 2026-11-24)
Each builds on the last and is attempted only if the one before holds up. Same rules for all: criteria fixed in
advance, strong baselines with equal budgets, held-out tests, several seeds, open code and data, an outside check.

- M-1, biology testbed: reproduce a documented bioelectric finding in simulation (blocking communication across a
  part boundary vs inside a part, then rescue by resetting the part's goal), make predictions in advance, and ask a
  developmental biologist (for example Levin's lab) for comment. Cost: a few dollars of GPU.
- M-2, self-rearranging robot skills: a simulated many-legged robot (MuJoCo "Ant" or similar) discovers its parts
  and is redirected to new tasks (a disabled leg, walking backwards, turning, carrying) by searching per-part codes,
  compared with fine-tuning, LoRA adapters and steering-vector search on equal budgets. Cost: tens of dollars.
- M-3, adjustable and inspectable models: a small model trained on several distinct skills forms discoverable
  modules; test whether modules match skills, redirection through module codes alone, whether other skills stay
  unchanged, and whether a simple refusal behaviour survives adaptation better than with fine-tuning. Cost: tens
  of dollars.
- M-4, regenerative medicine link: share the simulation's predictions about tissue control points with a lab
  that already runs bioelectric experiments.

## T-6, T-7: Review and write-up
A fresh-eyes review of code and results before the write-up, then a short report with methods, results, failures and comparisons.
