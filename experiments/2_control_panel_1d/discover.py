"""
Structure discovery + search, with nothing about the interaction layout hard-coded.

Layouts (tissue n shows motif (goal_n + sum of neighbour goals) mod K; missing neighbours count as 0):
  left : n-1      right : n+1      both : n-1 and n+1
Usage:
  python discover.py train right|both     # train + save a lower level for that layout (left = panel2_N8_K4.pt)
  python discover.py solve left|right|both

Method (generic):
  1. PROBE: baseline run with all goals 0; then for each tissue, switch its goal and see which regions change
     -> map S_r = set of tissues affecting region r.                                   cost N+1 rollouts
  2. SOLVE: repeatedly pick the region with the fewest unsettled tissues; try every combination of goals for
     those tissues (K^u rollouts, applied to all their symbols at once); for each example keep the best
     combination for that region's own error; set table entries by majority vote; mark them settled.
Reference: the "told the layout" search (alternate even/odd tissues, judge by own region) from chain_search.py.
"""
import sys, time, json, itertools
import numpy as np
import torch
import torch.nn.functional as F
import control_panel as cp, control_panel2 as cp2
torch.set_num_threads(1)
N, K, M = 8, 4, 8
OFFSETS = {"left": [-1], "right": [1], "both": [-1, 1], "boundary": [-1, 1]}


class LayoutWorld(cp.World):
    def __init__(self, N, K, offsets):
        super().__init__(N, K); self.offsets = offsets
    def realized(self, g):
        tot = g.clone()
        for o in self.offsets:
            sh = torch.zeros_like(g)
            if o < 0: sh[..., -o:] = g[..., :o]
            else: sh[..., :-o] = g[..., o:]
            tot = tot + sh
        return tot % self.K
    def target(self, g):
        return self.motifs[self.realized(g)].flatten(-2)


def load(layout):
    import os
    w = LayoutWorld(N, K, OFFSETS[layout])
    if layout == "boundary":
        from boundary import BoundaryWorld
        hid = int(os.environ.get("HID", 32)); cp.HID = hid
        w = BoundaryWorld(N, K); m = cp.Model(w, "panel", M, 0)
        m.load_state_dict(torch.load(f"panel_boundary_h{hid}.pt"))
    elif os.environ.get("SENSE"):
        from sense import SenseModel
        m = SenseModel(w, "panel", M, 0); m.load_state_dict(torch.load(f"panel_sense_{layout}.pt"))
    else:
        m = cp.Model(w, "panel", M, 0)
        m.load_state_dict(torch.load("panel2_N8_K4.pt" if layout == "left" else f"panel_layout_{layout}.pt"))
    for p in m.parameters(): p.requires_grad_(False)
    return w, m


@torch.no_grad()
def run_out(m, table, ex_sym):
    gi = cp.meaning_apply(table, ex_sym); M_, B = gi.shape[:2]
    s = torch.zeros(M_, B, cp.C, m.w.L); u = m.u(F.one_hot(gi, K).float())
    for t in range(100): s = m.step(s, u)
    return s[:, :, 0].clamp(0, 1)                                         # (M,B,L)


@torch.no_grad()
def probe(m, ex_sym, thr=0.1):
    base = run_out(m, torch.zeros(M, N, K, dtype=torch.long), ex_sym)
    A = torch.zeros(M, N, N, dtype=torch.bool)                            # A[k, tissue, region]
    for n in range(N):
        t = torch.zeros(M, N, K, dtype=torch.long); t[:, n] = 1
        d = (run_out(m, t, ex_sym) - base).abs().reshape(M, -1, N, cp.TW).mean((1, 3))  # (M, regions)
        A[:, n] = d > thr
    return A, N + 1


@torch.no_grad()
def solve(m, ex_sym, desired, A, tol=0.03):
    """Settle a tissue only when the evidence is unambiguous: every near-best combination for the region
    agrees on that tissue's setting. Ambiguous tissues are left for later regions to decide."""
    table = torch.zeros(M, N, K, dtype=torch.long); rollouts = 0
    for k in range(M):
        settled = set(); S = [set(torch.nonzero(A[k, :, r]).flatten().tolist()) for r in range(N)]
        tried = set()
        while len(settled) < N:
            cand = [(len(S[r] - settled), r) for r in range(N) if S[r] - settled and (r, frozenset(settled)) not in tried]
            if not cand:
                break
            _, r = min(cand); tried.add((r, frozenset(settled))); U = sorted(S[r] - settled)
            combos = list(itertools.product(range(K), repeat=len(U))); errs = []
            for combo in combos:
                t = table[k:k + 1].clone()
                for u, g in zip(U, combo): t[0, u] = g
                out = run_out_single(m, k, t, ex_sym[k:k + 1])
                errs.append(((out - desired[k:k + 1]) ** 2).reshape(1, -1, N, cp.TW)[0, :, r].mean(-1))
                rollouts += 1
            E = torch.stack(errs)                                    # (combos, B)
            near = E <= E.min(0).values + tol                        # near-best combos per example
            newly = set()
            for i, u in enumerate(U):
                ok = True; choice = {}
                for b in range(E.shape[1]):
                    vals = {combos[c][i] for c in torch.nonzero(near[:, b]).flatten().tolist()}
                    if len(vals) != 1: ok = False; break
                    sym = int(ex_sym[k, b, u]); v = vals.pop()
                    if choice.setdefault(sym, v) != v: ok = False; break
                if ok:
                    for sym, v in choice.items(): table[k, u, sym] = v
                    newly.add(u)
            settled |= newly
    return table, rollouts / M


_single = {}
@torch.no_grad()
def run_out_single(m, k, table, ex_sym):
    """Run only model k (for per-model search)."""
    if k not in _single:
        mk = type(m)(m.w, "panel", 1, 0)
        mk.load_state_dict({n: p[k:k + 1].clone() for n, p in m.state_dict().items()}); _single[k] = mk
    return run_out(_single[k], table, ex_sym)


@torch.no_grad()
def parity_search(m, ex_sym, desired, sweeps=4):
    table = torch.zeros(M, N, K, dtype=torch.long); r = 0
    for _ in range(sweeps):
        for parity in (0, 1):
            ts = [n for n in range(N) if n % 2 == parity]; errs = []
            for g in range(K):
                t = table.clone(); t[:, ts] = g
                errs.append(cp2.region_err(m, cp.meaning_apply(t, ex_sym), desired)); r += 1
            for k in range(M):
                for n in ts:
                    for sym in range(K):
                        sel = ex_sym[k, :, n] == sym
                        if sel.any(): table[k, n, sym] = torch.stack([e[k, sel, n].mean() for e in errs]).argmin()
    return table, r


def evaluate(m, table, new):
    torch.manual_seed(11)
    return cp.evaluate(m, lambda s: F.one_hot(cp.meaning_apply(table, s), K).float(), new)


if __name__ == "__main__":
    mode, layout = sys.argv[1], sys.argv[2]
    if mode == "train":
        torch.manual_seed(0); np.random.seed(0)
        w = LayoutWorld(N, K, OFFSETS[layout]); m = cp.Model(w, "panel", M, 0); cp.train(m, 900)
        torch.save(m.state_dict(), f"panel_layout_{layout}.pt"); sys.exit()
    w, m = load(layout)
    torch.manual_seed(5)
    new = torch.randint(0, K, (M, N, K)); ex_sym = torch.randint(0, K, (M, 8, N))
    for j in range(K): ex_sym[:, j] = (j + torch.arange(N)) % K
    desired = w.target(cp.meaning_apply(new, ex_sym))
    f = lambda r: "%.0f%% / %.0f%%" % (100 * np.mean(r["patterns"]), 100 * np.mean(r["after_damage"]))
    rows = [f"## Layout: {layout} (tissue n depends on n{'' if False else ''}{OFFSETS[layout]})", "",
            "| Method | Correct patterns / after damage | Settings correct | Cost (rollouts) |", "|---|---|---|---|"]
    rows.append(f"| Correct settings by hand (ceiling) | {f(evaluate(m, new, new))} | 100% | - |")
    t0 = time.time(); A, pc = probe(m, ex_sym)
    true_A = torch.zeros(N, N, dtype=torch.bool)
    for n in range(N):
        true_A[n, n] = True
        for o in OFFSETS[layout]:
            if 0 <= n - o < N: true_A[n, n - o] = True                  # tissue n affects region n - o
    map_ok = (A == true_A).flatten(1).all(1).float().mean().item()
    table, sc = solve(m, ex_sym, desired, A)
    rows.append(f"| Discover map, then solve (not told the layout) | {f(evaluate(m, table, new))} | "
                f"{(table == new).float().mean()*100:.0f}% | {pc} probe + {sc:.0f} solve ({time.time()-t0:.0f} s) |")
    t0 = time.time(); tp, rp = parity_search(m, ex_sym, desired)
    rows.append(f"| Reference: 'told the layout' search (assumes left-neighbour chain) | {f(evaluate(m, tp, new))} | "
                f"{(tp == new).float().mean()*100:.0f}% | {rp} ({time.time()-t0:.0f} s) |")
    rows += ["", f"Discovered influence map exactly right for {map_ok*100:.0f}% of models.", ""]
    print("\n".join(rows), flush=True)
    import os
    open(f"discover_{layout}{'_sense' if os.environ.get('SENSE') else ''}{'_h' + os.environ.get('HID', '') if layout == 'boundary' else ''}.md", "w").write("\n".join(rows) + "\n")
