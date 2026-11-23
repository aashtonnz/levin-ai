# Tasks

Last updated: 2026-11-24 (Pacific/Auckland)
Follows: task board v1.12, risk list v1.1, fresh-eyes review v1.2, style guide v1.1 (ai-working-practices), changelog to entry 40

Standing priorities:
- Report results as they are, including failures; fix comparisons before running where possible.
- Keep GPU spend low: smoke-test on CPU first, disconnect the Colab runtime after every run.

## Doing (at most 3)
- T-11: Better part-finding for irregular tissues, developed on one dataset and tested on held-out worlds (see PLAN.md). Covers R-1 and R-3.
- T-3: Scale-up test. Code ready, smoke-tested and pushed; GPU run blocked on 2026-11-24 because Colab had no free GPU. Next: start `run_all.sh` when one is available.

## Waiting on you

## Next (in order, at most about 10)
- T-4: Gate: does steering through discovered parts still beat retraining at scale? If no: rethink part-finding before anything else.
- T-2: Part-quality check: redo part-finding when search through the parts cannot reach the examples. Covers R-3.
- T-5: Literature check (first pass done: docs/literature.md); the real-task pilot is now M-2. Covers R-4.
- T-6: Fresh-eyes review of the results before any write-up.
- T-7: Short write-up: method, results, failures, comparison with existing work.

## Planned (after Next, by phase)
Long-term milestones (see PLAN.md; each only if the previous holds up):
- T-12: M-1 biology testbed: reproduce a bioelectric finding in simulation; ask a developmental biologist for comment.
- T-13: M-2 simulated robot that discovers its parts and is redirected by search, against fine-tuning, LoRA and steering vectors.
- T-14: M-3 small multi-skill model: discoverable modules, redirection without forgetting, refusal behaviour kept.
- T-15: M-4 share tissue control-point predictions with a bioelectric lab.

## Later / ideas
- Revisit memory and switch-like units with the discovered-parts setup.
- Add a minimal Colab notebook to the repository for one-click runs.
- Hierarchical parts (parts of parts) with goals at each level.

## Done (last two weeks, newest first)
- T-9: MIT licence added. 2026-11-24.
- T-8: Next step chosen: scale-up first; repository made public so Colab can clone it. 2026-11-24.
- T-1: Training reliability fixed: best-version keeping plus restarts; true-part steering 100% on 8 of 8 seeds, discovered parts 88%. [run 8](experiments/5_discovered_parts/results/run8.md) 2026-11-24.
- Spectral part-finding: 9 parts in 7–8 of 8 seeds, steering through discovered parts matches true parts. [run 6](experiments/5_discovered_parts/results/run6.md) 2026-11-24.
- 2D test passed its fixed criteria for both rules (search 98% and 100%, retraining 0–2%). [results](experiments/3_grid_2d/results/gpu_runs.md) 2026-11-23.

Retired: (none)
