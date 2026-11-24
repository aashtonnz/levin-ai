# Tasks

Last updated: 2026-11-24 (Pacific/Auckland)
Follows: task board v1.12, risk list v1.1, fresh-eyes review v1.2, style guide v1.1 (ai-working-practices), changelog to entry 40

Standing priorities:
- Report results as they are, including failures; fix comparisons before running where possible.
- Keep GPU spend low: smoke-test on CPU first, disconnect the Colab runtime after every run.

## Doing (at most 3)
- T-18: Make each square's code mean something locally (random codes during training). Screen: none pass, but small-square search rises 4% -> 27% -> 75% with noise 0 -> 0.3 -> 0.6 (encoder 100% -> 83%); nested squares worse. [local_screen](experiments/6_scale_up/results/local_screen.md). Follow-up screen (more noise, longer training) running.
- T-17: Nested squares as parts (user's idea). Learns fully (error 0.0003) and one-shot encoder steers 100% on both worlds, but search fails (5–9%; criteria H1/H2 FAIL): codes are used jointly, not locally. [hier](experiments/6_scale_up/results/hier.md). Next: train so each square's code has a local meaning (random or one-at-a-time codes in training), then retest search.
- T-11: Better part-finding. Averaging copies failed its fixed criterion on held-out worlds (+0.06 vs +0.10 needed) though it helped in 4 of 5; see experiments/7_part_finding/results_development.md. Next idea: a self-check using steering on the training examples.
- T-16: Fix training with 8 options. Not fixed by 16 codes (7%), longer training (23% at 6000 steps; 20000 steps with a high learning rate peaked at step 4000 and ended at 8%), a bigger network (0%) or a 4-then-8 curriculum (5%); [screen8](experiments/6_scale_up/results/screen8.md), [long8](experiments/6_scale_up/results/long8.md). Points to a design limit. Next to try: more update steps, or several codes per part. Lower priority than T-11.

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
- T-3: Scale-up finished. With true parts, steering holds at 99–100% (irregular shapes, 25 tissues, real interactions; retraining and prompt tuning 0%). Discovered parts fail on irregular (51%) and big (16%) worlds; 8-option and combined runs invalid (training failed). [final](experiments/6_scale_up/results/final.md) 2026-11-24.
- T-9: MIT licence added. 2026-11-24.
- T-8: Next step chosen: scale-up first; repository made public so Colab can clone it. 2026-11-24.
- T-1: Training reliability fixed: best-version keeping plus restarts; true-part steering 100% on 8 of 8 seeds, discovered parts 88%. [run 8](experiments/5_discovered_parts/results/run8.md) 2026-11-24.
- Spectral part-finding: 9 parts in 7–8 of 8 seeds, steering through discovered parts matches true parts. [run 6](experiments/5_discovered_parts/results/run6.md) 2026-11-24.
- 2D test passed its fixed criteria for both rules (search 98% and 100%, retraining 0–2%). [results](experiments/3_grid_2d/results/gpu_runs.md) 2026-11-23.

Retired: (none)
