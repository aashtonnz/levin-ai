# levin-ai

Small experiments testing whether an idea from Michael Levin's work on how bodies control their shape gives a useful way to build and steer neural networks.

**Status: early research on toy problems.** Results so far are from small simulated grids, not real tasks.

## The idea

Levin's work suggests that bodies are built from *competent parts*: groups of cells that each reliably reach a target pattern and repair damage on their own. The body changes what a part builds by changing its goal, not by rewiring its cells.

Translated to machine learning:

1. Train a system made of parts that each settle into a target pattern given a goal code.
2. Give the whole system a small *goal panel*: one code per part.
3. To make it do something new, search the panel instead of retraining the weights.
4. Ideally, have the system find its own parts, instead of being told where they are.

Each cell runs the same small network (a neural cellular automaton). With state $s_i$, goal code $u_{p(i)}$ for the part $p(i)$ containing cell $i$, and learned coupling $c_{ij}$ to neighbour $j$, one update step is

$$
s_i \leftarrow s_i + f_\theta\big(s_i,\ \{c_{ij}\, s_j\}_{j \in N(i)},\ u_{p(i)}\big) + \varepsilon .
$$

Parts are found by learning the couplings $c_{ij}$ with a cost on communication, then clustering the coupling graph.

## What has been found

All results are on simulated grids of 3×3 "tissues" of 5×5 cells (or 1D strips of 8 tissues), with 4 options per tissue and 8 random seeds per test.

**Steering the goal panel beats retraining.** Given 8 examples of a new target, searching the panel reached the new patterns in seconds. Retraining all weights and tuning the goal inputs by gradient descent (prompt tuning) on the same examples both failed.

| 2D test, new targets | Search on goal panel | Retrain weights | Prompt-tune |
|---|---|---|---|
| Each tissue shows its own setting | 98% | 0% | 0% |
| Each tissue shows its neighbour's setting | 100% | 2% | 0% |

When parts affect each other, the search first probes which part influences which region; search that ignored this got 2%.

**Part boundaries must match the real structure.** With parts that line up with the true tissues, search steered the system. With a grid of parts that cut across tissues, search failed (1%), although a trained encoder could still steer it (90%).

**The system can find its own parts.** Couplings between neighbouring cells were learned with a gradually increasing cost on communication, then grouped by spectral clustering. In the latest run:

| | Result |
|---|---|
| Runs that found the true number of parts (9) | 8 of 8 |
| Match to true tissues (adjusted Rand index, 1 = perfect) | 0.93 |
| Steering through discovered parts | 88% (7 of 8 runs at 100%) |
| Steering through true parts, given by hand | 100% (8 of 8 runs) |

The one failure found 9 parts but with some boundaries misplaced.

**What did not work.** Earlier experiments on memory (keeping a pattern after the goal is removed) and on switch-like units gave mixed results that did not beat standard training once the comparison was fair. They are kept in `experiments/1_setpoint_memory` for the record.

Full tables are in each experiment's `results` folder. `experiments/5_discovered_parts/results/run8.md` has the latest.

## Limits

- Every test world is small and regular: equal square tissues, 4 options each.
- Parts in these worlds really are separate units. Many real problems have fuzzy or overlapping structure.
- The method combines known pieces (neural cellular automata, modular networks, search over control inputs). Whether the combination is useful beyond toy problems is not yet known.

## Planned next

These are plans, not results:

- Test on larger grids, irregular tissue shapes, more options per part, and rules where parts depend on their neighbours.
- Have the system check its own parts and redo part-finding when steering through them fails.

## Running the code

Each script in `experiments` is self-contained and needs Python 3 with PyTorch and NumPy.

```bash
pip install torch numpy matplotlib
cd experiments/5_discovered_parts
python levin_parts.py --quick --skip-grid    # smoke test on CPU, about 2 minutes
python levin_parts.py --skip-grid            # full run; use a GPU (about 30 minutes on a T4)
```

Full runs were done on a T4 GPU in Google Colab.

## Layout

- `experiments/1_setpoint_memory`: early tests of memory and switch-like units.
- `experiments/2_control_panel_1d`: goal panels on a 1D strip, search methods and baselines.
- `experiments/3_grid_2d`: the 2D test against retraining and prompt tuning.
- `experiments/4_learned_coupling`: first attempts to learn part boundaries.
- `experiments/5_discovered_parts`: discovered parts used as the goal panel (current work).
- `TASKS.md`, `PLAN.md`, `DECISIONS.md`, `docs/risks.md`: how the project is run.
