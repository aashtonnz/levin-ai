"""Merge per-seed v2 results and plot. Usage: python plot_results_v2.py results_v2_s0 results_v2_s1 ..."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dirs = sys.argv[1:] or ["results_v2_s0", "results_v2_s1", "results_v2_s2"]
R, strips, T = {}, None, None
for d in dirs:
    p = f"{d}/results.json"
    if not os.path.exists(p): continue
    j = json.load(open(p)); T = np.array(j["targets"])
    for k, v in j["results"].items(): R.setdefault(k, []).extend(v)
    if strips is None or "s0" in d: strips = j["strips"] or strips
KINDS = [k for k in ["nca", "nca_pool", "soft_setpoint", "setpoint"] if k in R]
CONST = float(T[:4].var(axis=(1, 2)).mean())
COL = {"nca": "#eb6834", "nca_pool": "#eda100", "soft_setpoint": "#1baf7a", "setpoint": "#2a78d6"}
NAME = {"nca": "NCA", "nca_pool": "NCA + sample pool", "soft_setpoint": "Setpoint, soft stability",
        "setpoint": "Setpoint, hard contraction"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

def vals(k, key): return np.array([r[key] for r in R[k]])

def bars(ax, keys, labels, title, ylabel, pct=False, ref=None, reflabel=None):
    x = np.arange(len(keys)); w = 0.8 / len(KINDS)
    for i, k in enumerate(KINDS):
        off = (i - (len(KINDS) - 1) / 2) * w
        for j, key in enumerate(keys):
            v = vals(k, key) * (100 if pct else 1)
            ax.bar(x[j] + off, v.mean(), w * 0.92, color=COL[k], label=NAME[k] if j == 0 else None)
            ax.scatter(np.full(len(v), x[j] + off), v, s=11, color=INK, zorder=3, lw=0)
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1.1, color=MUTED)
        ax.text(len(keys) - 0.55, ref, reflabel, ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)

fig = plt.figure(figsize=(12.5, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
bars(ax, ["reach", "damage", "swap_seen", "swap_heldout", "long_run_400"],
     ["Reach goal", "After\ndamage", "Goal swap\n(seen)", "Goal swap\n(held out)", "Long run\n(400 steps)"],
     "A. Robustness on single-outcome goals", "Error (MSE, lower is better)", ref=CONST,
     reflabel="best constant output")
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

ax = fig.add_subplot(gs[0, 1])
for k in KINDS:
    steps = sorted(int(n) for n in R[k][0]["steps_curve"])
    arr = np.array([[r["steps_curve"][str(n)] for n in steps] for r in R[k]])
    ax.plot(steps, arr.mean(0), lw=2, color=COL[k], marker="o", ms=4.5, label=NAME[k])
    ax.fill_between(steps, arr.min(0), arr.max(0), color=COL[k], alpha=0.13, lw=0)
ax.axvspan(32, 48, color=GRID, alpha=0.6, lw=0)
ax.set_xscale("log", base=2); ax.set_xlabel("Iterations run at test time", labelpad=2)
ax.set_ylabel("Error (MSE)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("B. More test-time iterations (band = seed range; grey = training horizon)", loc="left", color=INK, fontsize=10.5)

ax = fig.add_subplot(gs[1, 0])
bars(ax, ["explicit_lr", "bistable_reach", "bistable_damage", "bistable_long", "rewrite_err_vs_right"],
     ["Explicit goal\n(left / right)", "Bistable:\nreach", "Bistable:\nafter damage", "Bistable:\n400 steps",
      "Rewrite: error vs\n'right', 200 steps\nafter goal reset"],
     "C. Two valid outcomes, one goal register", "Error (MSE, lower is better)")
for y, lab in [(0.0694, "hedge: faint blob on both sides"), (0.1389, "always one side (or blank)")]:
    ax.axhline(y, ls="--", lw=1.1, color=MUTED); ax.text(-0.42, y + 0.003, lab, ha="left", va="bottom", color=MUTED, fontsize=8, bbox=dict(fc="white", ec="none", pad=1))

ax = fig.add_subplot(gs[1, 1])
bars(ax, ["bistable_correct_side", "bistable_long_correct_side"],
     ["Output nearer the side it\nwas seeded on (64 steps)", "...after 400 steps"],
     "D. Does it commit to the side it started on? (% of trials)", "% of trials", pct=True, ref=50, reflabel="chance")
ax.set_ylim(0, 105)

sub = gs[2, :].subgridspec(len(KINDS) + 1, 7, wspace=0.08, hspace=0.3)
heads = ["bistable, seeded\nleft, 64 steps", "damaged", "+48 steps", "goal = 'right'\n48 steps",
         "goal = bistable\nagain, 200 steps", "left target", "right target"]
for row, k in enumerate(KINDS):
    frames = strips[k] + [T[4].tolist(), T[5].tolist()]
    for j, f in enumerate(frames):
        a = fig.add_subplot(sub[row, j]); a.imshow(np.array(f), cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(COL[k] if j < 5 else MUTED)
        if row == 0: a.set_title(heads[j], fontsize=8.3, color=INK)
        if j == 0: a.set_ylabel({"nca": "NCA", "nca_pool": "NCA\n+ pool", "soft_setpoint": "Setpoint\n(soft)", "setpoint": "Setpoint\n(hard)"}[k], color=INK, fontsize=8)
a = fig.add_subplot(sub[len(KINDS), :]); a.axis("off")
a.text(0, 0.55, "E. Seed-0 models. The rewrite (last two frames) was never part of training. "
       "No weight updates at any point.", fontsize=8.5, color=MUTED)
os.makedirs("results_v2", exist_ok=True)
fig.savefig("results_v2/v2_figure.png", dpi=150, bbox_inches="tight")

keys = [("reach", "Reach goal"), ("damage", "After damage"), ("swap_seen", "Goal swap (seen)"),
        ("swap_heldout", "Goal swap (held out)"), ("long_run_400", "Long run (400)"),
        ("explicit_lr", "Explicit left/right"), ("bistable_reach", "Bistable reach"),
        ("bistable_damage", "Bistable after damage"), ("bistable_long", "Bistable 400 steps"),
        ("bistable_long_correct_side", "Keeps own side, 400 steps (%)"),
        ("rewrite_err_vs_right", "Rewrite: error vs right"), ("train_seconds", "Train time (s)")]
rows = ["| Test | " + " | ".join(NAME[k] for k in KINDS) + " |", "|---" * (len(KINDS) + 1) + "|"]
for key, lab in keys:
    def f(k):
        v = vals(k, key)
        if "%" in lab: return "%.0f (%s)" % (100 * v.mean(), ", ".join("%.0f" % (100 * x) for x in v))
        if key == "train_seconds": return "%.0f" % v.mean()
        return "%.3f (%s)" % (v.mean(), ", ".join("%.3f" % x for x in v))
    rows.append(f"| {lab} | " + " | ".join(f(k) for k in KINDS) + " |")
rows.append(f"| n seeds | " + " | ".join(str(len(R[k])) for k in KINDS) + " |")
open("results_v2/summary.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
