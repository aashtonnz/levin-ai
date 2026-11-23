"""Plot results.json from setpoint_experiment.py -> figure PNG + summary markdown table."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.json"))
R = d["results"]; T = np.array(d["targets"])
import os
if os.path.exists("results_soft/results.json"):
    ds = json.load(open("results_soft/results.json")); R.update(ds["results"]); d["strips"].update(ds["strips"])
KINDS = [k for k in ["nca", "soft_setpoint", "setpoint"] if k in R]
CONST = float(T.var())   # MSE of the best constant output (trivial baseline)
COL = {"setpoint": "#2a78d6", "nca": "#eb6834", "soft_setpoint": "#1baf7a"}
NAME = {"setpoint": "Setpoint, hard contraction", "nca": "Unrolled NCA (baseline)",
        "soft_setpoint": "Setpoint, soft stability penalty"}
SHORT = {"setpoint": "Setpoint\n(hard)", "nca": "NCA", "soft_setpoint": "Setpoint\n(soft)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

tests = ["reach", "damage", "swap_seen", "swap_heldout", "long_run_400"]
labels = ["Reach goal", "After damage", "Goal swap\n(seen)", "Goal swap\n(held out)", "Long run\n(400 steps)"]

def stat(kind, key):
    v = np.array([r[key] for r in R[kind]]); return v.mean(), v.std(), v

fig = plt.figure(figsize=(12, 9.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.62, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(tests)); w = 0.26
for i, k in enumerate(KINDS):
    off = (i - (len(KINDS) - 1) / 2) * (w + 0.02)
    m = [stat(k, t)[0] for t in tests]
    ax.bar(x + off, m, w, color=COL[k], label=NAME[k] + " (bar = mean)")
    for j, t in enumerate(tests):
        v = stat(k, t)[2]
        ax.scatter(np.full(len(v), x[j] + off), v, s=14, color=INK, zorder=3, lw=0)
ax.axhline(CONST, ls="--", lw=1.2, color=MUTED)
ax.text(len(tests) - 0.5, CONST, "best constant output", ha="right", va="bottom", color=MUTED, fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("Error vs target (MSE, lower is better)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Robustness tests (3 seeds, dots = individual seeds, frozen weights)", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

ax = fig.add_subplot(gs[0, 1])
for k in KINDS:
    steps = sorted(int(n) for n in R[k][0]["steps_curve"])
    arr = np.array([[r["steps_curve"][str(n)] for n in steps] for r in R[k]])
    ax.plot(steps, arr.mean(0), lw=2, color=COL[k], marker="o", ms=5, label=NAME[k])
    ax.fill_between(steps, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=COL[k], alpha=0.15, lw=0)
ax.axvspan(32, 48, color=GRID, alpha=0.6, lw=0); ax.text(39, ax.get_ylim()[1] * 0.97, "training\nhorizon", ha="center", va="top", fontsize=8, color=MUTED)
ax.axhline(CONST, ls="--", lw=1.2, color=MUTED)
ax.set_xscale("log", base=2); ax.set_xlabel("Iterations run at test time", labelpad=2); ax.set_ylabel("Error vs target (MSE)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("More test-time iterations: help or hurt?", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

sub = gs[1, :].subgridspec(len(KINDS) + 1, 7, wspace=0.08, hspace=0.25)
heads = ["8 steps", "24 steps", "48 steps", "damaged", "+48 steps", "goal → cross\n+96 steps", "target"]
for row, k in enumerate(KINDS):
    frames = d["strips"][k] + [T[2]]
    for j, f in enumerate(frames):
        a = fig.add_subplot(sub[row, j]); a.imshow(np.array(f), cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(True); sp.set_color(COL[k] if j < 6 else MUTED)
        if row == 0: a.set_title(heads[j], fontsize=8.5, color=INK)
        if j == 0: a.set_ylabel(SHORT[k], color=COL[k], fontsize=9)
a = fig.add_subplot(sub[len(KINDS), :]); a.axis("off")
a.text(0, 0.6, "Seed-0 models. Grow disk → erase a disc → keep running → switch goal register to cross "
       "(a held-out transition). No weight updates at any point.", fontsize=8.5, color=MUTED)
fig.savefig("results/setpoint_vs_nca.png", dpi=150, bbox_inches="tight")

rows = ["| Test | " + " | ".join(NAME[k] for k in KINDS) + " |", "|---" * (len(KINDS) + 1) + "|"]
for t, l in zip(tests + ["train_seconds"], [s.replace("\n", " ") for s in labels] + ["Training time (s, mean)"]):
    f = (lambda k: "%.3f (seeds: %s)" % (stat(k, t)[0], ", ".join("%.3f" % v for v in stat(k, t)[2]))) \
        if t != "train_seconds" else (lambda k: "%.0f" % stat(k, t)[0])
    rows.append(f"| {l} | " + " | ".join(f(k) for k in KINDS) + " |")
rows.append("| Best constant output | " + " | ".join(f"{CONST:.3f}" for k in KINDS) + " |")
rows.append("| Parameters | " + " | ".join(str(R[k][0]["params"]) for k in KINDS) + " |")
open("results/summary.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
