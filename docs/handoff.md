# Handoff (2026-11-24): state of the project for a new session

Read this, then README.md, AGENTS.md, TASKS.md, PLAN.md, DECISIONS.md and docs/risks.md.

## What the project tests
Ideas from Michael Levin's work as AI methods: a neural cellular automaton made of parts, each part steered by one
goal code; redirecting to new targets by searching codes instead of retraining; and parts that the system finds
itself.

## What holds up (with true parts, 4 options per part)
- Steering by searching codes: 98-100% of new targets, against 0-2% for retraining and prompt tuning
  (experiments/3_grid_2d, experiments/6_scale_up/results/final.md).
- Holds with irregular part shapes, 25 parts, and parts that interact (shift rule with fixed input meaning).

## Open problems, in priority order
1. Parts the system can use without being told them.
   - Discovered parts (spectral clustering of learned coupling): fine on squares, 51% on irregular, 16% on 25 parts.
     Averaging copies failed its fixed criterion (experiments/7_part_finding).
   - Nested squares, "pixel resolution" (user's idea, T-17, results/hier.md): the model learns fully and the one-shot
     encoder steers 100% on two irregular worlds, but square-by-square search fails (5-9%). The codes are used jointly,
     not locally.
   - Random codes during training (T-18, results/local_screen.md): clear trade-off; search 4% -> 66% as more squares
     are randomised, encoder 100% -> 71%. No setting meets both bars (search >= 80%, encoder >= 90%).
   - Next (T-19): train with search in the loop.
2. 8 options per part fails even with true parts (T-16, results/screen8.md, long8.md). Not fixed by more codes,
   longer training, a bigger network or a curriculum. Parked.

## How runs are done
- Main script: experiments/6_scale_up/levin_scale.py. Useful flags: --true-only (skip part-finding), --hier 3,6
  (nested squares), --code-noise / --code-noise-k, --seeds, --iters2, --eval-every, --lr-drops, --quick (smoke test).
- Each experiment has a run_*.sh that fixes its pass criteria in the header before running. Keep doing this.
- GPU so far: Google Colab T4 via the user's browser (clone the public repo, run the script, read logs/*.log and
  */summary.md, then disconnect the runtime). About 1.15 compute units an hour, roughly US$0.10 a unit.
  T4s are sometimes unavailable in the evening; never fall back to a CPU runtime for real runs.
- Smoke-test on CPU first with --quick (a few minutes; search with many parts is slow on CPU).

## Lessons
- Screens with 2 seeds are noisy here; use 4.
- Running more than 3-4 variants at once on one T4 slows each a lot.
- Write criteria before running and report failures plainly; the user values honest readings and pushes back on
  overstatements.
- The user prefers simple, literal explanations (no analogies unless asked) and LaTeX for equations.
