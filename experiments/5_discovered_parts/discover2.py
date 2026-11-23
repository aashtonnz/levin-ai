import numpy as np

def otsu(v):
    v = np.sort(v.ravel()); best, thr = -1, v.mean()
    for t in np.quantile(v, np.linspace(0.05, 0.95, 91)):
        a, b = v[v <= t], v[v > t]
        if len(a) == 0 or len(b) == 0: continue
        s = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if s > best: best, thr = s, t
    return thr

def discover2(ch, cv, min_part=8, bias=0.0):
    """Region merging by MEAN coupling along whole shared borders (robust to single leaky links).
    Start: every cell its own region. Repeatedly merge the adjacent pair with the strongest mean border
    coupling, while it exceeds the Otsu split of all link strengths (+bias). Then absorb fragments
    smaller than min_part into the neighbour with the strongest mean border."""
    H, W = ch.shape[0], cv.shape[1]
    thr = otsu(np.concatenate([ch.ravel(), cv.ravel()])) + bias
    lab = np.arange(H * W)
    edges = [(y * W + x, y * W + x + 1, ch[y, x]) for y in range(H) for x in range(W - 1)] + \
            [(y * W + x, (y + 1) * W + x, cv[y, x]) for y in range(H - 1) for x in range(W)]
    def borders():
        b = {}
        for a, c, w in edges:
            la, lc = lab[a], lab[c]
            if la != lc:
                k = (min(la, lc), max(la, lc)); s = b.get(k, [0.0, 0]); s[0] += w; s[1] += 1; b[k] = s
        return b
    while True:
        b = borders()
        if not b: break
        k, (s, n) = max(b.items(), key=lambda kv: kv[1][0] / kv[1][1])
        if s / n <= thr: break
        lab[lab == k[1]] = k[0]
    while True:
        ids, cnt = np.unique(lab, return_counts=True)
        small = [i for i, c in zip(ids, cnt) if c < min_part]
        if not small or len(ids) == 1: break
        sm = min(small, key=lambda i: cnt[list(ids).index(i)])
        b = borders(); cand = {(k[0] if k[1] == sm else k[1]): s / n for k, (s, n) in b.items() if sm in k}
        lab[lab == sm] = max(cand, key=cand.get)
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(H, W)

def spectral_parts(ch, cv, kmax=20, power=4, extra=0, seed=0, min_part=8):
    """Spectral clustering of the coupling graph (judges regions as a whole, not single links).
    k = eigengap estimate + `extra` (bias toward over-segmenting, which is harmless)."""
    H, W = ch.shape[0], cv.shape[1]; n = H * W
    A = np.zeros((n, n))
    for y in range(H):
        for x in range(W - 1): i = y * W + x; A[i, i + 1] = A[i + 1, i] = ch[y, x] ** power
    for y in range(H - 1):
        for x in range(W): i = y * W + x; A[i, i + W] = A[i + W, i] = cv[y, x] ** power
    d = A.sum(1); Dm = 1 / np.sqrt(d + 1e-9)
    L = np.eye(n) - Dm[:, None] * A * Dm[None, :]
    ev, vec = np.linalg.eigh(L)
    gaps = np.diff(ev[:kmax + 1]); k = int(np.argmax(gaps[1:]) + 2) + extra   # k >= 2
    X = vec[:, :k]; X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed); best = None
    for _ in range(5):                                   # k-means, a few restarts
        C = X[rng.choice(n, k, replace=False)]
        for _ in range(50):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        sse = ((X - C[lab]) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, lab.copy())
    lab = best[1].reshape(H, W)
    # split disconnected pieces into separate parts (a part must be spatially connected)
    out = -np.ones((H, W), int); nid = 0
    for y in range(H):
        for x in range(W):
            if out[y, x] >= 0: continue
            st = [(y, x)]; out[y, x] = nid
            while st:
                a, b = st.pop()
                for aa, bb in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                    if 0 <= aa < H and 0 <= bb < W and out[aa, bb] < 0 and lab[aa, bb] == lab[y, x]:
                        out[aa, bb] = nid; st.append((aa, bb))
            nid += 1
    return out, k
