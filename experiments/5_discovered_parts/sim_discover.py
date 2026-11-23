import numpy as np, sys, re, types
src = open("levin_parts.py").read()
ns = {}
exec("import numpy as np\nfrom math import comb\nH = W = 15\n" + src[src.index("def ari("):src.index("# ------------------------------------------------------------------ steering tests")], ns)
from discover2 import discover2
ari, discover = ns["ari"], ns["discover"]
H = W = 15; TS = 5
tissue = (np.arange(H)[:, None] // TS) * 3 + (np.arange(W)[None, :] // TS)
bh = ((np.arange(W - 1) + 1) % TS == 0)[None, :].repeat(H, 0); bv = ((np.arange(H - 1) + 1) % TS == 0)[:, None].repeat(W, 1)
def lumped(lab):  # does any found part contain >=3 cells from each of two different tissues?
    for p in np.unique(lab):
        t, c = np.unique(tissue[lab == p], return_counts=True)
        if (c >= 3).sum() >= 2: return True
    return False
rng = np.random.default_rng(0)
for name, mi, mb, sd in [("like GPU run", 0.55, 0.34, 0.15), ("noisier", 0.55, 0.40, 0.18), ("cleaner", 0.55, 0.30, 0.10)]:
    R = {"old": [], "new": []}
    for _ in range(60):
        ch = np.clip(rng.normal(mi, sd, (H, W - 1)), 0, 1); cv = np.clip(rng.normal(mi, sd, (H - 1, W)), 0, 1)
        ch[bh] = np.clip(rng.normal(mb, sd, bh.sum()), 0, 1); cv[bv] = np.clip(rng.normal(mb, sd, bv.sum()), 0, 1)
        for k, lab in [("old", discover(ch, cv)[0]), ("new", discover2(ch, cv))]:
            R[k].append((lumped(lab), lab.max() + 1, ari(lab.ravel(), tissue.ravel())))
    for k in R:
        a = np.array(R[k], dtype=float)
        print(f"{name:13s} {k}: lumping in {100*a[:,0].mean():3.0f}% of maps | parts median {np.median(a[:,1]):.0f} | ARI {a[:,2].mean():.2f}")
