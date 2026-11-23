exec(open("sim_discover.py").read().split("rng = np.random")[0])
from discover2 import spectral_parts
rng = np.random.default_rng(0)
for name, mi, mb, sd in [("like GPU run", 0.55, 0.34, 0.15), ("cleaner", 0.55, 0.30, 0.10)]:
    R = {}
    for _ in range(40):
        ch = np.clip(rng.normal(mi, sd, (H, W - 1)), 0, 1); cv = np.clip(rng.normal(mi, sd, (H - 1, W)), 0, 1)
        ch[bh] = np.clip(rng.normal(mb, sd, bh.sum()), 0, 1); cv[bv] = np.clip(rng.normal(mb, sd, bv.sum()), 0, 1)
        outs = [("old", discover(ch, cv)[0])]
        for e in (0, 3, 6):
            for pw in (2, 4):
                outs.append((f"spectral +{e} pow{pw}", spectral_parts(ch, cv, extra=e, power=pw)[0]))
        for k, lab in outs:
            R.setdefault(k, []).append((lumped(lab), lab.max() + 1, ari(lab.ravel(), tissue.ravel())))
    for k in R:
        a = np.array(R[k], dtype=float)
        print(f"{name:13s} {k:20s}: lumping {100*a[:,0].mean():3.0f}% | parts median {np.median(a[:,1]):.0f} | ARI {a[:,2].mean():.2f}")
