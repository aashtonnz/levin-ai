import time, numpy as np, torch
from baselines_lib import *
rows = []
for lr in (0.3, 1.0):
    t0 = time.time(); soft = soft_goals_grad(300, lr=lr)
    rows.append(report(f"Gradient-tuned goal inputs, 300 steps, step size {lr}", evaluate_soft(soft),
                       f"~900 rollout-equivalents, {time.time()-t0:.0f} s"))
rng = np.random.default_rng(0)
for rounds in (32, 64):
    t0 = time.time(); tab, n = subset_search(rounds, lambda i: [t for t in range(N) if rng.random() < 0.5])
    rows.append(report(f"Local-error search, random halves (not told the layout), {rounds} rounds", evaluate_table(tab),
                       f"{n} rollouts, {time.time()-t0:.0f} s", f"{(tab == new).float().mean()*100:.0f}%"))
open("baselines_results.md", "a").write("\n".join(rows) + "\n")
