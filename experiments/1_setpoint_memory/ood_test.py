"""Out-of-distribution tests on saved bench2 models (no retraining).
Training conditions: noise 0.02, damage segments 4-10 cells, <=2 damages, episodes of 140 steps."""
import json, torch, numpy as np, memory_bench2 as mb
torch.manual_seed(0)

def big_damage(s, lo, hi):
    M, B = s.shape[:2]
    start = torch.randint(0, mb.L - hi, (M, B)); length = torch.randint(lo, hi + 1, (M, B))
    seg = (mb.xs >= start[..., None]) & (mb.xs < (start + length)[..., None])
    return s * (~seg)[:, :, None, :].float()

@torch.no_grad()
def scenario(m, noise=0.02, steps=1000, dmg=None, n_dmg=0, B=64):
    mb.NOISE = noise
    side = torch.arange(B)[None].expand(m.M, -1) % 2
    s = mb.init_state(m.M, B, side); none = torch.zeros(m.M, B, 2)
    dmg_times = set(np.linspace(150, steps - 150, n_dmg).astype(int)) if n_dmg else set()
    for t in range(steps):
        if t in dmg_times: s = big_damage(s, *dmg)
        s = m.step(s, none)
    mb.NOISE = 0.02
    return (mb.mse_to(m.output(s), side) < mb.COMMIT).float().mean(1).numpy()

tests = {
    "in-distribution (noise .02, 1000 steps)": dict(),
    "noise x2.5 (0.05)": dict(noise=0.05),
    "noise x4 (0.08)": dict(noise=0.08),
    "big damage 12-16 cells": dict(dmg=(12, 16), n_dmg=1),
    "5 damages (4-10 cells)": dict(dmg=(4, 10), n_dmg=5),
    "very long run (5000 steps)": dict(steps=5000),
}
out = {}
for v in ["plain", "switches", "switch_output", "gated"]:
    m = mb.Ensemble(v, 8, 0); m.load_state_dict(torch.load(f"bench2/{v}.pt")); out[v] = {}
    for name, kw in tests.items():
        r = scenario(m, **kw); out[v][name] = r.tolist()
json.dump(out, open("bench2/ood.json", "w"))
names = {"plain": "Plain", "switches": "Hidden switches", "switch_output": "Switch = output", "gated": "Learned gate"}
print("| Condition | " + " | ".join(names.values()) + " |\n|---|---|---|---|---|")
for name in tests:
    print(f"| {name} | " + " | ".join("%.0f%% (%d/8)" % (100 * np.mean(out[v][name]), sum(np.array(out[v][name]) >= .9)) for v in names) + " |")
