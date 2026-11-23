"""
Compare part-finding methods on saved phase-1 couplings (couplings.npz from levin_scale.py).
The methods never see the true tissues; the true tissues are used only to score them.

Usage: python try_partfinders.py RUN_DIR [levin_scale.py world options, e.g. --irregular --world-seed 0]
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
run_dir, world_args = sys.argv[1], sys.argv[2:]
sys.argv = ["levin_scale.py", "--quick"] + world_args
src = open(os.path.join(HERE, "../6_scale_up/levin_scale.py")).read().split('if __name__ == "__main__":')[0]
exec(src)                                                   # world (TIS, H, W), ari(), spectral_parts()
d = np.load(os.path.join(run_dir, "couplings.npz")); CH, CVV = d["ch"], d["cv"]
true_lab = TIS.flatten()


def adjacency(ch, cv, power=2):
    n = H * W; A = np.zeros((n, n))
    for y in range(H):
        for x in range(W - 1): i = y * W + x; A[i, i + 1] = A[i + 1, i] = ch[y, x] ** power
    for y in range(H - 1):
        for x in range(W): i = y * W + x; A[i, i + W] = A[i + W, i] = cv[y, x] ** power
    return A


def links(lab, ch, cv):
    """(a, b, w) for every neighbouring pair of cells."""
    L = lab.reshape(H, W); out = []
    for y in range(H):
        for x in range(W - 1): out.append((L[y, x], L[y, x + 1], ch[y, x]))
    for y in range(H - 1):
        for x in range(W): out.append((L[y, x], L[y + 1, x], cv[y, x]))
    return out


def relabel(lab):
    _, inv = np.unique(lab, return_inverse=True); return inv


def modularity(lab, A):
    m2 = A.sum(); k = A.sum(1); Q = 0.0
    for c in np.unique(lab):
        idx = lab == c; Q += A[np.ix_(idx, idx)].sum() / m2 - (k[idx].sum() / m2) ** 2
    return Q


def merge_by_borders(lab, ch, cv, rule):
    """Repeatedly merge the adjacent pair of parts with the strongest MEAN border coupling, while rule(...) allows."""
    lab = lab.copy()
    while True:
        lk = links(lab, ch, cv)
        within = np.mean([w for a, b, w in lk if a == b]); bd = {}
        for a, b, w in lk:
            if a != b:
                k_ = (min(a, b), max(a, b)); s_ = bd.setdefault(k_, [0.0, 0]); s_[0] += w; s_[1] += 1
        if not bd: break
        (a, b), (s_, n_) = max(bd.items(), key=lambda kv: kv[1][0] / kv[1][1])
        allb = np.mean([v[0] / v[1] for v in bd.values()])
        if not rule(s_ / n_, within, allb): break
        lab[lab == b] = a
    return relabel(lab)


def m_spectral(ch, cv, k):         # current method
    return spectral_parts(ch, cv, power=2, seed=k, kmax=min(H * W // 8, 60))[0].flatten()


def m_modularity(ch, cv, k):       # spectral at several sizes, keep the most modular partition
    A = adjacency(ch, cv); best = None
    for extra in (-2, -1, 0, 1, 2, 3, 4):
        try: lab = spectral_parts(ch, cv, power=2, seed=k, extra=extra, kmax=min(H * W // 8, 60))[0].flatten()
        except Exception: continue
        q = modularity(lab, A)
        if best is None or q > best[0]: best = (q, lab)
    return best[1]


def m_over_merge(ch, cv, k):       # over-segment (twice the estimated parts), then merge by mean border strength
    base, k0 = spectral_parts(ch, cv, power=2, seed=k, kmax=min(H * W // 8, 60))
    lab = spectral_parts(ch, cv, power=2, seed=k, extra=k0, kmax=min(H * W // 8, 60))[0].flatten()
    return merge_by_borders(lab, ch, cv, lambda bm, within, allb: bm > (within + allb) / 2)


def m_spectral_merge(ch, cv, k):   # current method, then merge parts whose border is as strong as their insides
    return merge_by_borders(m_spectral(ch, cv, k), ch, cv, lambda bm, within, allb: bm > (within + allb) / 2)


METHODS = {"spectral (current)": m_spectral, "modularity-chosen size": m_modularity,
           "spectral + merge": m_spectral_merge, "over-segment + merge": m_over_merge}
print(f"world: side {H}, {N} tissues, {'irregular' if args.irregular else 'square'}, world seed {args.world_seed}; "
      f"{CH.shape[0]} seeds from {run_dir}")
for name, f in METHODS.items():
    a_, p_ = [], []
    for k in range(CH.shape[0]):
        lab = f(CH[k], CVV[k], k); a_.append(ari(lab, true_lab)); p_.append(int(lab.max() + 1))
    print(f"{name:24s} match {np.mean(a_):.2f}  per seed {[round(x, 2) for x in a_]}  parts {p_}")
