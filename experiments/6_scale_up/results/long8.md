# T-16 scaling check: 8 options, 20000 training steps (2026-11-24, T4)

True parts only, 4 seeds, learning rate held high until 80% (drops at 80% and 95%), encoder accuracy of the
current weights on 64 fixed held-out patterns logged every 2000 steps. Reading fixed before running: still rising by
5+ points over the last two checkpoints before step 16000 means longer training is a live fix; levelled off well
below 80% means it is not.

| Step | Accuracy per seed (current weights) | Best held-out error so far per seed |
|---|---|---|
| 2000 | 8, 0, 0, 0 | 0.0118, 0.0131, 0.0271, 0.0271 |
| 4000 | 28, 0, 8, 0 | 0.0051, 0.0131, 0.0099, 0.0271 |
| 6000 | 0, 0, 8, 0 | unchanged |
| 8000 | 2, 0, 2, 0 | unchanged |
| 10000 | 2, 0, 2, 0 | unchanged |
| 12000 | 2, 0, 0, 0 | unchanged |
| 14000 | 2, 0, 0, 0 | 0.0051, 0.0131, 0.0099, 0.0268 |
| 16000 | 2, 0, 0, 0 | unchanged |
| 18000 | 0, 0, 2, 0 | unchanged |
| 19999 | 0, 0, 2, 0 | unchanged |

Final (best held-out version of each seed): encoder 9% [31, 0, 6, 0]; search 8% [22, 3, 6, 0]. Total 4007 s.

## Reading

- By the fixed rule: NOT a live fix. Accuracy peaked at step 4000 and fell back; the best version of every seed is
  from about step 4000 and nothing after improved on it.
- Holding the learning rate high did not help: training stalled and the current weights got worse. In the earlier
  6000-step run (B), dropping the rate at 50% and 80% gave 23%, better than this.
- What this does not rule out: a slowly decaying learning rate over a long run, or more seeds (2 of 4 seeds never
  learned at all, here and in earlier runs). The evidence so far points to a limit of the design (one code per big
  part, fixed number of update steps), not to too little training.
