"""
Closing the loop: parts DISCOVERED by learnable coupling become the goal panel.

Hidden world: 3x3 tissues of 5x5 cells, 4 motif options each (independent rule: each tissue shows its own motif).
Phase 1: cells with learnable, costly gap-junction-like coupling are trained through a misaligned 5x5 code grid.
         The coupling cost is ANNEALED: zero for the first 30% of training, ramped to full by 70%.
         Parts = groups of strongly coupled cells; fragments smaller than MIN_PART cells are merged into the
         neighbouring part they are most strongly coupled to.
Phase 2: fresh cells + encoder are trained with ONE GOAL CODE PER PART, for three panels:
           discovered  parts found in phase 1 (coupling kept from phase 1)
           true        the hidden 3x3 tissues (best case; not available in reality)
           grid        a fixed 5x5 grid, misaligned with the tissues (previous design)
Steering test (new target patterns, weights frozen):
           encoder     one-shot guess of the codes
           search      local search over part codes; each part judged by the error in its OWN cells
           gradient    continuous codes tuned by gradient through the dynamics

Usage: python levin_parts.py [--quick] [--seeds 8] [--iters1 3000] [--iters2 2000]
"""
import argparse, json, time, os
from math import comb
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--iters1", type=int, default=3000)
ap.add_argument("--iters2", type=int, default=2000)
ap.add_argument("--codes", type=int, default=8)
ap.add_argument("--coupling-cost", type=float, default=0.15)
ap.add_argument("--min-part", type=int, default=8)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--skip-grid", action="store_true")
ap.add_argument("--out", default="parts_results")
args = ap.parse_args()
if args.quick: args.seeds, args.iters1, args.iters2 = 2, 6, 6

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R, TS, K = 3, 5, 4
N, H, W = R * R, R * TS, R * TS
C, HID, M, V, GRID = 12, 64, args.seeds, args.codes, 5
NOISE, COMMIT, T = 0.02, 0.02, 80

# ------------------------------------------------------------------ hidden world (independent rule)
tissue = torch.arange(N).reshape(R, R).repeat_interleave(TS, 0).repeat_interleave(TS, 1).to(DEV)
MOT = torch.zeros(K, TS, TS)
MOT[0, 2, 1:4] = 1; MOT[1, 1:4, 2] = 1
MOT[2, 1, 1] = MOT[2, 1, 3] = MOT[2, 3, 1] = MOT[2, 3, 3] = 1
MOT[3, 1:4, 1:4] = 1; MOT[3, 2, 2] = 0
MOT = MOT.to(DEV)


def target(g):
    loc = MOT[g]; sh = loc.shape[:-3]
    return loc.reshape(*sh, R, R, TS, TS).transpose(-3, -2).reshape(*sh, H, W)


def tissue_err(out, tgt):
    d = (out - tgt) ** 2; sh = d.shape[:-2]
    return d.reshape(*sh, R, TS, R, TS).transpose(-3, -2).reshape(*sh, N, TS * TS).mean(-1)


BORDER = torch.zeros(4, H, W, device=DEV)
BORDER[0, :, 0] = 1; BORDER[1, :, -1] = 1; BORDER[2, 0, :] = 1; BORDER[3, -1, :] = 1
PIN = 5 * C + V + 4 + 3          # +3: position cue within the cell's own part


# ------------------------------------------------------------------ part maps
class Parts:
    """Per-model partition of the cells: pm (M,H,W) part index, P parts max."""
    def __init__(self, pm):
        self.pm = pm.to(DEV).long(); self.P = int(pm.max()) + 1
        self.mask = F.one_hot(self.pm, self.P).permute(0, 3, 1, 2).float()         # (M,P,H,W)
        self.count = self.mask.sum((2, 3)).clamp(min=1)                             # (M,P)
        self.exists = self.mask.sum((2, 3)) > 0
        adj = torch.zeros(M, self.P, self.P, dtype=torch.bool)
        pmc = self.pm.cpu()
        for a, b in [(pmc[:, :, :-1], pmc[:, :, 1:]), (pmc[:, :-1, :], pmc[:, 1:, :])]:
            for k in range(M):
                aa, bb = a[k].flatten(), b[k].flatten(); d = aa != bb
                adj[k, aa[d], bb[d]] = True; adj[k, bb[d], aa[d]] = True
        colour = torch.full((M, self.P), -1, dtype=torch.long)                      # greedy graph colouring
        for k in range(M):
            for p_ in range(self.P):
                used = set(colour[k, adj[k, p_]].tolist())
                colour[k, p_] = next(c for c in range(self.P + 1) if c not in used)
        self.colour = colour.to(DEV); self.ncol = int(colour.max()) + 1
        # position cue within own part, computed identically for every panel:
        # offset from the part's centre (scaled by the part's half-size) + whether the cell is on the part's edge
        ys = torch.arange(H, device=DEV).float()[None, None, :, None]; xs = torch.arange(W, device=DEV).float()[None, None, None, :]
        cy = (self.mask * ys).sum((2, 3)) / self.count; cx = (self.mask * xs).sum((2, 3)) / self.count     # (M,P)
        sy = ((self.mask * (ys - cy[..., None, None]).abs()).amax((2, 3))).clamp(min=1)
        sx = ((self.mask * (xs - cx[..., None, None]).abs()).amax((2, 3))).clamp(min=1)
        g = lambda t: t.gather(1, self.pm.reshape(M, -1)).reshape(M, H, W)
        dy = (ys[0, 0] - g(cy)) / g(sy); dx = (xs[0, 0] - g(cx)) / g(sx)
        pmp = F.pad(self.pm.float()[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
        edge = ((pmp[:, 1:-1, :-2] != self.pm) | (pmp[:, 1:-1, 2:] != self.pm) |
                (pmp[:, :-2, 1:-1] != self.pm) | (pmp[:, 2:, 1:-1] != self.pm)).float()
        self.pos = torch.stack([dy, dx, edge], 1)                                   # (M,3,H,W)
        self.centre_idx = (cy.round().clamp(0, H - 1) * W + cx.round().clamp(0, W - 1)).long()   # (M,P)

    def field(self, codes):                       # codes (M,B,P,V) -> (M,B,V,H,W)
        Mn, B = codes.shape[:2]
        idx = self.pm.reshape(Mn, 1, H * W, 1).expand(Mn, B, H * W, V)
        f = codes.gather(2, idx).reshape(Mn, B, H, W, V).permute(0, 1, 4, 2, 3)
        return torch.cat([f, self.pos[:, None].expand(Mn, B, 3, H, W)], 2)

    def part_err(self, out, tgt):                 # (M,B,H,W) -> (M,B,P) mean error over each part's own cells
        return torch.einsum("mbhw,mphw->mbp", (out - tgt) ** 2, self.mask) / self.count[:, None]


def grid_parts():
    g = (torch.arange(H) * GRID // H)
    return Parts((g[:, None] * GRID + g[None, :]).expand(M, H, W).clone())


def true_parts():
    return Parts(tissue.cpu().expand(M, H, W).clone())


# ------------------------------------------------------------------ model
def init(seed, coupling):
    g = torch.Generator().manual_seed(seed)
    p = {"W1": torch.randn(M, HID, PIN, generator=g) / PIN ** 0.5, "b1": torch.zeros(M, HID),
         "W2": torch.zeros(M, C, HID), "b2": torch.zeros(M, C),
         "e1": torch.randn(M, 64, 49, generator=g) / 7, "e2b": torch.zeros(M, 64),
         "e3": torch.randn(M, V, 64, generator=g) / 8, "eb": torch.zeros(M, V)}
    p = {k: v.to(DEV).requires_grad_(True) for k, v in p.items()}
    if coupling == "learn":
        p["ch"] = torch.full((M, H, W - 1), 2.0, device=DEV, requires_grad=True)
        p["cv"] = torch.full((M, H - 1, W), 2.0, device=DEV, requires_grad=True)
    return p


def couplings(p, fixed=None):
    if fixed is not None: return fixed
    if "ch" not in p: return torch.ones(M, 4, H, W, device=DEV) * (1 - BORDER)
    ch, cv = torch.sigmoid(p["ch"]), torch.sigmoid(p["cv"]); c = torch.zeros(M, 4, H, W, device=DEV)
    c[:, 0, :, 1:] = ch; c[:, 1, :, :-1] = ch; c[:, 2, 1:, :] = cv; c[:, 3, :-1, :] = cv
    return c * (1 - BORDER)


def encode(p, parts, tgt, tau=1.0, noise=True, hard=True):
    """Each part reads a 7x7 window of the target centred on its own centre, then a small MLP picks its code.
    Identical for every panel."""
    Mn, B = tgt.shape[:2]
    win = F.unfold(F.pad(tgt.reshape(Mn * B, 1, H, W), (3, 3, 3, 3)), 7).reshape(Mn, B, 49, H * W)
    idx = parts.centre_idx[:, None, None, :].expand(Mn, B, 49, parts.P)              # (M,B,49,P)
    x = win.gather(3, idx).permute(0, 1, 3, 2)                                          # (M,B,P,49)
    h = F.relu(torch.einsum("mbpi,mji->mbpj", x, p["e1"]) + p["e2b"][:, None, None])
    logits = torch.einsum("mbpj,mvj->mbpv", h, p["e3"]) + p["eb"][:, None, None]
    if noise: logits = logits - torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1)))
    soft = F.softmax(logits / tau, -1)
    return soft + (F.one_hot(soft.argmax(-1), V).float() - soft).detach() if hard else soft


def step(p, s, u, c):
    Mn, B = s.shape[:2]
    sl = F.pad(s, (1, 0))[..., :, :-1]; sr = F.pad(s, (0, 1))[..., :, 1:]
    su = F.pad(s, (0, 0, 1, 0))[..., :-1, :]; sd = F.pad(s, (0, 0, 0, 1))[..., 1:, :]
    cc = c[:, None]
    x = torch.cat([s, cc[:, :, 0:1] * sl, cc[:, :, 1:2] * sr, cc[:, :, 2:3] * su, cc[:, :, 3:4] * sd, u,
                   BORDER.expand(Mn, B, 4, H, W)], 2)
    h = F.relu(torch.einsum("mhp,mbpyx->mbhyx", p["W1"], x) + p["b1"][:, None, :, None, None])
    d = torch.einsum("mch,mbhyx->mbcyx", p["W2"], h) + p["b2"][:, None, :, None, None]
    return (s + d + NOISE * torch.randn_like(s)).clamp(-2, 2)


def damage(s, mask):
    Mn, B = s.shape[:2]
    size = torch.randint(2, 6, (Mn, B), device=DEV)
    y0 = torch.randint(0, H - 2, (Mn, B), device=DEV); x0 = torch.randint(0, W - 2, (Mn, B), device=DEV)
    ys = torch.arange(H, device=DEV)[None, None, :, None]; xs = torch.arange(W, device=DEV)[None, None, None, :]
    m = ((ys >= y0[..., None, None]) & (ys < (y0 + size)[..., None, None]) &
         (xs >= x0[..., None, None]) & (xs < (x0 + size)[..., None, None])) & mask[..., None, None]
    return s * (~m)[:, :, None].float()


def rollout(p, parts, codes, steps=T, desired=None, dmg_prob=0.0, grad=False, cfix=None):
    Mn, B = codes.shape[:2]
    s = torch.zeros(Mn, B, C, H, W, device=DEV); u = parts.field(codes); c = couplings(p, cfix)
    t_d = int(np.random.randint(steps // 2, steps - 15)) if dmg_prob > 0 else -1
    do = torch.rand(Mn, B, device=DEV) < dmg_prob; loss, n = 0.0, 0
    with torch.set_grad_enabled(grad):
        for t in range(steps):
            if t == t_d: s = damage(s, do)
            s = step(p, s, u, c)
            if desired is not None and t >= 30 and t % 5 == 0:
                loss = loss + ((s[:, :, 0] - desired) ** 2).mean((-1, -2)).mean(1).sum(); n += 1
    return s, (loss / max(n, 1) if desired is not None else None)


@torch.no_grad()
def val_loss(p, parts, cfix=None):
    """Per-seed error on a fixed held-out batch (encoder codes, no noise)."""
    st = torch.get_rng_state(); torch.manual_seed(12345)
    g = torch.randint(0, K, (M, 32, N), device=DEV); tgt = target(g)
    s = torch.zeros(M, 32, C, H, W, device=DEV); u = parts.field(encode(p, parts, tgt, noise=False)); c = couplings(p, cfix)
    for t in range(T): s = step(p, s, u, c)
    torch.set_rng_state(st)
    return ((s[:, :, 0].clamp(0, 1) - tgt) ** 2).mean((1, 2, 3))           # (M,)


def train(p, parts, iters, anneal_cost=0.0, cfix=None, tag="", B=16, keep_best=False):
    opt = torch.optim.Adam(p.values(), lr=3e-3)
    best_v = torch.full((M,), float("inf"), device=DEV); best_p = {k: v.detach().clone() for k, v in p.items()}
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.5), int(iters * 0.8)], 0.3)
    t0 = time.time()
    for it in range(iters):
        g = torch.randint(0, K, (M, B, N), device=DEV); tgt = target(g)
        codes = encode(p, parts, tgt, tau=max(0.5, 1.0 - it / iters), noise=False)
        _, loss = rollout(p, parts, codes, T, tgt, dmg_prob=0.6, grad=True, cfix=cfix)
        ramp = min(1.0, max(0.0, (it / iters - 0.3) / 0.4))                 # 0 until 30%, full by 70%
        cost = anneal_cost * ramp * couplings(p).mean((1, 2, 3)).sum() if "ch" in p else 0.0
        opt.zero_grad(); (loss + cost).backward(); torch.nn.utils.clip_grad_norm_(p.values(), M); opt.step(); sched.step()
        if keep_best and (it % 100 == 0 or it == iters - 1) and it > 0:
            v = val_loss(p, parts, cfix); imp = v < best_v; best_v = torch.where(imp, v, best_v)
            for k_ in p: best_p[k_][imp] = p[k_].detach()[imp]
        if it % 200 == 0 or it == iters - 1:
            extra = f" | held-out per seed {[round(x, 4) for x in best_v.tolist()]}" if keep_best and it > 0 else ""
            print(f"  [{tag}] it {it:5d} recon/model {loss.item() / M:.4f} mean coupling "
                  f"{couplings(p, cfix).mean().item():.2f} ({time.time() - t0:.0f}s){extra}", flush=True)
    return best_p if keep_best else {k: v.detach() for k, v in p.items()}


# ------------------------------------------------------------------ discovering parts from coupling
def ari(a, b):
    ct = {}
    for x, y in zip(a, b): ct[(x, y)] = ct.get((x, y), 0) + 1
    sa, sb = {}, {}
    for (x, y), v in ct.items(): sa[x] = sa.get(x, 0) + v; sb[y] = sb.get(y, 0) + v
    idx = sum(comb(v, 2) for v in ct.values()); ia = sum(comb(v, 2) for v in sa.values()); ib = sum(comb(v, 2) for v in sb.values())
    tot = comb(len(a), 2); exp = ia * ib / tot; mx = (ia + ib) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def discover(ch, cv, thr=None, min_part=8):
    """Strongly coupled components, then merge fragments < min_part into the neighbour they couple to most.
    Default threshold: this seed's own average coupling (links weaker than average are cut)."""
    if thr is None: thr = float(np.concatenate([ch.flatten(), cv.flatten()]).mean())
    par = list(range(H * W))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for y in range(H):
        for x in range(W - 1):
            if ch[y, x] > thr: par[f(y * W + x)] = f(y * W + x + 1)
    for y in range(H - 1):
        for x in range(W):
            if cv[y, x] > thr: par[f(y * W + x)] = f((y + 1) * W + x)
    lab = np.array([f(i) for i in range(H * W)])
    raw = lab.copy()
    edges = [(y * W + x, y * W + x + 1, ch[y, x]) for y in range(H) for x in range(W - 1)] + \
            [(y * W + x, (y + 1) * W + x, cv[y, x]) for y in range(H - 1) for x in range(W)]
    while True:
        ids, counts = np.unique(lab, return_counts=True)
        small = [i for i, c in zip(ids, counts) if c < min_part]
        if not small or len(ids) == 1: break
        s = min(small, key=lambda i: counts[list(ids).index(i)])
        score = {}
        for a, b, w in edges:
            la, lb = lab[a], lab[b]
            if la == s and lb != s: score[lb] = score.get(lb, 0) + w
            if lb == s and la != s: score[la] = score.get(la, 0) + w
        lab[lab == s] = max(score, key=score.get)
    _, lab = np.unique(lab, return_inverse=True); _, raw = np.unique(raw, return_inverse=True)
    return lab.reshape(H, W), raw.reshape(H, W)


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


# ------------------------------------------------------------------ steering tests
@torch.no_grad()
def check(p, parts, codes, tgt, cfix=None):
    Mn, B = codes.shape[:2]
    s = torch.zeros(Mn, B, C, H, W, device=DEV); u = parts.field(codes); c = couplings(p, cfix)
    ok = lambda s: (tissue_err(s[:, :, 0].clamp(0, 1), tgt) < COMMIT).all(-1).float().mean(1)
    for t in range(T): s = step(p, s, u, c)
    a = ok(s); s = damage(s, torch.ones(Mn, B, dtype=torch.bool, device=DEV))
    for t in range(T): s = step(p, s, u, c)
    return a.cpu().numpy(), ok(s).cpu().numpy()


@torch.no_grad()
def local_search(p, parts, tgt, sweeps=4, cfix=None):
    Mn, B = tgt.shape[:2]
    codes = torch.zeros(Mn, B, parts.P, dtype=torch.long, device=DEV); runs = 0
    for _ in range(sweeps):
        for col in range(parts.ncol):
            sel = (parts.colour == col)[:, None, :].expand(Mn, B, parts.P)
            errs = []
            for v in range(V):
                trial = torch.where(sel, torch.full_like(codes, v), codes)
                s, _ = rollout(p, parts, F.one_hot(trial, V).float(), cfix=cfix); runs += 1
                errs.append(parts.part_err(s[:, :, 0].clamp(0, 1), tgt))
            best = torch.stack(errs).argmin(0)
            codes = torch.where(sel, best, codes)
    return codes, runs


def grad_codes(p, parts, tgt, steps=200, lr=0.3, cfix=None):
    Mn, B = tgt.shape[:2]
    z = torch.zeros(Mn, B, parts.P, V, device=DEV, requires_grad=True); opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        _, loss = rollout(p, parts, F.softmax(z, -1), T, tgt, grad=True, cfix=cfix)
        opt.zero_grad(); loss.backward(); opt.step()
    return F.one_hot(z.detach().argmax(-1), V).float()


def steering(p, parts, tag, cfix=None):
    torch.manual_seed(7); np.random.seed(7)
    g = torch.randint(0, K, (M, 32, N), device=DEV); tgt = target(g); res = {}
    with torch.no_grad(): enc = encode(p, parts, tgt, noise=False)
    res["encoder (one-shot guess)"] = (*check(p, parts, enc, tgt, cfix), "1 encoder pass")
    t = time.time(); codes, runs = local_search(p, parts, tgt, cfix=cfix)
    res["search (each part judged by its own cells)"] = (*check(p, parts, F.one_hot(codes, V).float(), tgt, cfix),
                                                         f"{runs} batched trial runs, {time.time() - t:.0f}s")
    t = time.time(); gs = 20 if args.quick else 200; gc = grad_codes(p, parts, tgt, steps=gs, cfix=cfix)
    res["gradient-tuned codes"] = (*check(p, parts, gc, tgt, cfix), f"{gs} gradient steps, {time.time() - t:.0f}s")
    print(f"  [{tag}] " + " | ".join(f"{k}: {100 * v[0].mean():.0f}%" for k, v in res.items()), flush=True)
    return res


if __name__ == "__main__":
    T0 = time.time(); os.makedirs(args.out, exist_ok=True)
    print(f"device {DEV} | hidden 3x3 tissues (independent rule) | {M} seeds", flush=True)
    torch.manual_seed(0); np.random.seed(0)

    print("PHASE 1: learn coupling (annealed cost) through a misaligned 5x5 grid", flush=True)
    grid = grid_parts()
    p1 = train(init(0, "learn"), grid, args.iters1, anneal_cost=args.coupling_cost, tag="phase1")
    ch, cv = torch.sigmoid(p1["ch"]).cpu().numpy(), torch.sigmoid(p1["cv"]).cpu().numpy()
    xs, ys = np.arange(W - 1), np.arange(H - 1)
    bh = ((xs + 1) % TS == 0)[None, :].repeat(H, 0); bv = ((ys + 1) % TS == 0)[:, None].repeat(W, 1)
    true_lab = tissue.cpu().numpy().flatten(); pm = np.zeros((M, H, W), dtype=np.int64); ph1 = []
    for k in range(M):
        old_lab, raw = discover(ch[k], cv[k], min_part=args.min_part)
        lab, _k = spectral_parts(ch[k], cv[k], power=2, seed=k); pm[k] = lab
        ph1.append({"boundary": float(np.concatenate([ch[k][bh], cv[k][bv]]).mean()),
                    "interior": float(np.concatenate([ch[k][~bh], cv[k][~bv]]).mean()),
                    "ari_raw": ari(raw.flatten(), true_lab), "parts_raw": int(raw.max() + 1),
                    "ari_merged": ari(lab.flatten(), true_lab), "parts_merged": int(lab.max() + 1),
                    "ari_old_method": ari(old_lab.flatten(), true_lab), "parts_old_method": int(old_lab.max() + 1)})
    with torch.no_grad():
        gg = torch.randint(0, K, (M, 32, N), device=DEV); tt = target(gg)
        a1, _ = check(p1, grid, encode(p1, grid, tt, noise=False), tt)
    np.savez(f"{args.out}/couplings.npz", ch=ch, cv=cv)
    print(f"  old part-finder: match {np.mean([x['ari_old_method'] for x in ph1]):.2f}, parts {[x['parts_old_method'] for x in ph1]}", flush=True)
    disc = Parts(torch.tensor(pm)); cfix1 = couplings(p1)
    print(f"  coupling on hidden boundaries {np.mean([x['boundary'] for x in ph1]):.2f}, inside "
          f"{np.mean([x['interior'] for x in ph1]):.2f}; match to true tissues raw {np.mean([x['ari_raw'] for x in ph1]):.2f}"
          f" -> merged {np.mean([x['ari_merged'] for x in ph1]):.2f}; parts per seed {[x['parts_merged'] for x in ph1]}", flush=True)

    print("PHASE 2: retrain with one goal code per part", flush=True)
    results = {}
    panels = [("discovered parts", disc, cfix1), ("true tissues (best case)", true_parts(), None)]
    if not args.skip_grid: panels.append(("fixed 5x5 grid (misaligned)", grid, None))
    for name, parts, cfix in panels:
        torch.manual_seed(1); np.random.seed(1)
        p2 = train(init(1, "fixed"), parts, args.iters2, cfix=cfix, tag=name, keep_best=True)
        # restart any seed that failed to learn (held-out error >10x the median seed), keep the better version
        v = val_loss(p2, parts, cfix); bad = v > 10 * v.median()
        if bad.any():
            print(f"  [{name}] restarting seeds {torch.nonzero(bad).flatten().tolist()} (held-out {[round(x, 4) for x in v[bad].tolist()]})", flush=True)
            torch.manual_seed(2); np.random.seed(2)
            p3 = train(init(2, "fixed"), parts, args.iters2, cfix=cfix, tag=name + " restart", keep_best=True)
            v3 = val_loss(p3, parts, cfix); use = bad & (v3 < v)
            for k_ in p2: p2[k_][use] = p3[k_][use]
            print(f"  [{name}] after restart: held-out {[round(x, 4) for x in torch.where(use, v3, v).tolist()]}", flush=True)
        results[name] = steering(p2, parts, name, cfix=cfix)
        results[name]["n_parts"] = [int(parts.exists[k].sum()) for k in range(M)]
        print(f"  [{name}] PER SEED search: {[round(100 * float(x)) for x in results[name]['search (each part judged by its own cells)'][0]]}"
              f" encoder: {[round(100 * float(x)) for x in results[name]['encoder (one-shot guess)'][0]]}", flush=True)
        json.dump({"phase1": ph1, "parts": pm.tolist(), "partial": {n: {k: (v if k == "n_parts" else
                   [float(v[0].mean()), float(v[1].mean()), v[2]]) for k, v in r.items()} for n, r in results.items()}},
                  open(f"{args.out}/partial.json", "w"))

    L = [f"# Discovered parts as the goal panel ({M} seeds, {DEV})", "",
         "## Phase 1: parts from annealed coupling", "",
         f"- Coupling on hidden tissue boundaries: {np.mean([x['boundary'] for x in ph1]):.2f}; inside tissues: "
         f"{np.mean([x['interior'] for x in ph1]):.2f}",
         f"- Match to true tissues (adjusted Rand index, 1 = perfect, 0 = chance): raw groups "
         f"{np.mean([x['ari_raw'] for x in ph1]):.2f}, after merging fragments {np.mean([x['ari_merged'] for x in ph1]):.2f} "
         f"(per seed {[round(x['ari_merged'], 2) for x in ph1]})",
         f"- (part-finder: spectral clustering of the coupling graph)\n- Parts per seed: raw {[x['parts_raw'] for x in ph1]}; after merging {[x['parts_merged'] for x in ph1]} (true answer: 9)",
         f"- Phase-1 model's own accuracy on new patterns (encoder): {100 * a1.mean():.0f}% (earlier, un-annealed run: 46%)", "",
         "## Phase 2: steering to new patterns (% fully correct; after damage in brackets)", "",
         "| Goal panel | Parts | Encoder | Search (own-part errors) | Gradient-tuned codes | Search cost |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        f = lambda k: f"{100 * r[k][0].mean():.0f}% ({100 * r[k][1].mean():.0f}%)"
        L.append(f"| {name} | {np.mean(r['n_parts']):.0f} | {f('encoder (one-shot guess)')} | "
                 f"{f('search (each part judged by its own cells)')} | {f('gradient-tuned codes')} | "
                 f"{r['search (each part judged by its own cells)'][2]} |")
    L += ["", "## Per seed", "", "| Seed | Parts found | Match to tissues (ARI) | Discovered: search | Discovered: encoder | True tissues: search |",
          "|---|---|---|---|---|---|"]
    rd, rt = results["discovered parts"], results["true tissues (best case)"]
    S = "search (each part judged by its own cells)"; E = "encoder (one-shot guess)"
    for k in range(M):
        L.append(f"| {k} | {ph1[k]['parts_merged']} | {ph1[k]['ari_merged']:.2f} | {100 * rd[S][0][k]:.0f}% | "
                 f"{100 * rd[E][0][k]:.0f}% | {100 * rt[S][0][k]:.0f}% |")
    a_ = np.array([x["ari_merged"] for x in ph1]); d_ = np.array(rd[S][0], dtype=float); t_ = np.array(rt[S][0], dtype=float)
    ok = t_ > 0.5
    if ok.sum() >= 3 and a_[ok].std() > 0 and d_[ok].std() > 0:
        L.append(f"\nCorrelation of match (ARI) with discovered-parts search, over seeds where true tissues work "
                 f"({int(ok.sum())} seeds): {np.corrcoef(a_[ok], d_[ok])[0, 1]:.2f}")
    L += ["", f"Total time {time.time() - T0:.0f}s"]
    print("\n".join(L), flush=True)
    open(f"{args.out}/summary.md", "w").write("\n".join(L) + "\n")
    json.dump({"phase1": ph1, "parts": pm.tolist()}, open(f"{args.out}/results.json", "w"))
    print("(position cue within each part given to all panels)", flush=True)
