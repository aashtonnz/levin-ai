"""Plot memory_bench results: python plot_bench.py bench"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "bench"
ORDER = ["plain", "switches", "switch_output", "gated", "discrete"]
NAME = {"plain": "Plain", "switches": "Hidden switches", "switch_output": "Switch = output",
        "gated": "Learned gate", "discrete": "Discrete output"}
COL = {"plain": "#eb6834", "switches": "#eda100", "switch_output": "#1baf7a", "gated": "#2a78d6", "discrete": "#8a5cd1"}
R = {v: json.load(open(f"{d}/{v}.json")) for v in ORDER if os.path.exists(f"{d}/{v}.json")}
V = [v for v in ORDER if v in R]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
keys = ["commit_100", "hold_1000", "after_damage", "rewrite_300", "rewrite_1000"]
labels = ["Commits to start\nside (t=100)", "Still holds it\n(t=1000)", "Holds it after\ndamage", "Rewrite took\n(t=300)", "Rewrite still\nholds (t=1000)"]

fig = plt.figure(figsize=(12.5, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.42)
ax = fig.add_subplot(gs[0])
x = np.arange(len(keys)); w = 0.8 / len(V)
for i, v in enumerate(V):
    off = (i - (len(V) - 1) / 2) * w
    for j, k in enumerate(keys):
        vals = 100 * np.array(R[v][k])
        ax.bar(x[j] + off, vals.mean(), w * 0.9, color=COL[v], label=NAME[v] if j == 0 else None)
        ax.scatter(np.full(len(vals), x[j] + off) + np.random.uniform(-w * 0.2, w * 0.2, len(vals)), vals,
                   s=8, color=INK, lw=0, zorder=3, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.set_ylabel("% of trials committed to the correct side")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title(f"Memory benchmark ({os.path.basename(d)}): {len(R[V[0]]['hold_1000'])} seeds per mechanism (bar = mean, dots = seeds)",
             loc="left", fontsize=11, color=INK, pad=30)
ax.legend(frameon=False, ncol=len(V), loc="lower left", bbox_to_anchor=(0, 1.06), fontsize=8.5)

sub = gs[1].subgridspec(1, len(V), wspace=0.15)
for i, v in enumerate(V):
    a = fig.add_subplot(sub[i]); img = np.array(R[v]["raster"])
    a.imshow(img, aspect="auto", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    for t, lab in [(150, "damage"), (300, "pulse 'right'")]:
        a.axhline(t, color=COL[v], lw=1.2, ls="--"); a.text(31, t - 8, lab, ha="right", fontsize=7.5, color=INK,
                                                           bbox=dict(fc="white", ec="none", pad=0.5))
    a.set_title(NAME[v], fontsize=9.5, color=INK); a.set_xticks([0, 8, 24, 31]); a.set_xticklabels(["0", "L", "R", "31"])
    a.set_yticks([0, 200, 400, 600, 799]) if i == 0 else a.set_yticks([])
    if i == 0: a.set_ylabel("time step (down) - seeded LEFT")
fig.text(0.125, 0.02, "Bottom: space-time view of each mechanism's best seed (cells across, time down). "
         "Correct behaviour: bump on the left, repaired after damage, moves right after the pulse and stays.",
         fontsize=8.5, color=MUTED)
fig.savefig(f"{d}/bench_figure.png", dpi=150, bbox_inches="tight")

rows = ["| Mechanism | " + " | ".join(l.replace("\n", " ") for l in labels) + " | Params | Train s |", "|---" * 8 + "|"]
for v in V:
    rows.append(f"| {NAME[v]} | " + " | ".join("%.0f%% (%d/%d seeds >= 90%%)" % (100 * np.mean(R[v][k]),
                sum(np.array(R[v][k]) >= 0.9), len(R[v][k])) for k in keys) +
                f" | {R[v]['params_per_model']} | {R[v]['train_seconds']:.0f} |")
rows += ["", "Seeds reliable on ALL five tests (>= 90% each): " +
         ", ".join(f"{NAME[v]} {int(np.all(np.array([R[v][k] for k in keys]) >= 0.9, axis=0).sum())}/{len(R[v][keys[0]])}" for v in V)]
open(f"{d}/summary.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
