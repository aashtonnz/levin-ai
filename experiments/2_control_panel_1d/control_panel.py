"""
Scaling the "control panel": competent parts + a small goal interface found by search, vs retraining.

N tissues x 8 cells, K motif options per tissue. An instruction gives each tissue a symbol (K symbols).
Cells: no position within tissue, constant noise, damage. Same NCA cell network for both models.

  flat  cells receive their tissue's symbol (one-hot) directly; trained end-to-end on the original meaning.
  panel lower level trained ALONE to build any motif it is given (random goal per tissue) - 100% of budget.
        The "control panel" is a table (tissue, symbol) -> goal, found by SEARCH, never by gradient.

Search ("local"): try goal g on every tissue at once (K rollouts in total); each tissue's own error tells,
for each (tissue, symbol) seen in the examples, which goal works best. Cost = K rollouts, independent of N.

Retarget: every tissue gets its own random reshuffle of symbol -> motif. 8 example instructions.
  flat retrains all weights (400 steps); panel re-runs the search (K rollouts).
Evaluate on 64 random instructions: all tissues correct, and again after damage.
"""
import json, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F

TW, C, HID, NOISE, EDGE, COMMIT = 8, 8, 32, 0.02, 1.0, 0.05
LOC = torch.arange(TW)
MOTIF_SETS = {
    2: [[1, 2, 3], [4, 5, 6]],
    4: [[1, 2, 3], [4, 5, 6], [1, 2, 5, 6], [3, 4]],
}
KER = torch.stack([torch.tensor([0., 1, 0]), torch.tensor([-.5, 0, .5]), torch.tensor([1., -2, 1])])


class World:
    def __init__(self, N, K):
        self.N, self.K, self.L = N, K, N * TW
        self.tissue = torch.arange(self.L) // TW
        m = torch.zeros(K, TW)
        for k, cells in enumerate(MOTIF_SETS[K]): m[k, cells] = 1
        self.motifs = m                                                   # (K, TW)
        self.idch = F.one_hot(self.tissue, N).float().T                   # (N, L)
        self.ker = KER.repeat(C + N, 1).unsqueeze(1)

    def target(self, goal_idx):            # (M,B,N) -> (M,B,L)
        return self.motifs[goal_idx].flatten(-2)


class Model(torch.nn.Module):
    def __init__(self, w, kind, M, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w, self.kind, self.M = w, kind, M
        P = 3 * (C + w.N) + w.K
        self.W1 = torch.nn.Parameter(torch.randn(M, HID, P, generator=g) / P ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(M, HID))
        self.W2 = torch.nn.Parameter(torch.zeros(M, C, HID))
        self.b2 = torch.nn.Parameter(torch.zeros(M, C))

    def u(self, onehot_per_tissue):         # (M,B,N,K) -> (M,B,K,L)
        return onehot_per_tissue[:, :, self.w.tissue, :].permute(0, 1, 3, 2)

    def step(self, s, u):
        w = self.w; M, B = s.shape[:2]
        full = torch.cat([s, w.idch.expand(M, B, w.N, w.L)], 2)
        p = F.conv1d(F.pad(full.reshape(M * B, C + w.N, w.L), (1, 1), value=EDGE), w.ker, groups=C + w.N)
        x = torch.cat([p.reshape(M, B, -1, w.L), u], 2)
        h = F.relu(torch.einsum("mhp,mbpl->mbhl", self.W1, x) + self.b1[:, None, :, None])
        d = torch.einsum("mch,mbhl->mbcl", self.W2, h) + self.b2[:, None, :, None]
        return (s + d + NOISE * torch.randn_like(s)).clamp(-2, 2)


def damage(s, L, mask):
    M, B = s.shape[:2]
    start = torch.randint(0, L - 8, (M, B)); length = torch.randint(3, 8, (M, B))
    xs = torch.arange(L)
    seg = (xs >= start[..., None]) & (xs < (start + length)[..., None]) & mask[..., None]
    return s * (~seg)[:, :, None, :].float()


def episode(m, inp, goal_idx, T=100, dmg=True, grad=True):
    """inp: (M,B,N,K) one-hot fed to cells; goal_idx: (M,B,N) which motif each tissue should show."""
    w = m.w; M, B = goal_idx.shape[:2]
    s = torch.zeros(M, B, C, w.L); tgt = w.target(goal_idx); u = m.u(inp)
    t_d = int(np.random.randint(45, 75)); do = torch.rand(M, B) < 0.6; loss = 0.0; n = 0
    with torch.set_grad_enabled(grad):
        for t in range(T):
            if dmg and t == t_d: s = damage(s, w.L, do)
            s = m.step(s, u)
            if t >= 35 and t % 5 == 0:
                loss = loss + ((s[:, :, 0] - tgt) ** 2).mean(-1).mean(1).sum(); n += 1
    return loss / n, s


def meaning_apply(meaning, sym):
    """meaning: (M,N,K) table symbol->motif per tissue; sym: (M,B,N) -> motif idx (M,B,N)"""
    return torch.gather(meaning[:, None].expand(-1, sym.shape[1], -1, -1), 3, sym[..., None]).squeeze(-1)


def train(m, iters, B=16):
    w = m.w; M = m.M; opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    ident = torch.arange(w.K)[None, None].expand(M, w.N, -1)
    for it in range(iters):
        sym = torch.randint(0, w.K, (M, B, w.N))
        if m.kind == "flat":
            loss, _ = episode(m, F.one_hot(sym, w.K).float(), meaning_apply(ident, sym))
        else:   # lower-level competency: input IS the goal
            loss, _ = episode(m, F.one_hot(sym, w.K).float(), sym)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), M); opt.step()
        if it % 300 == 0 or it == iters - 1:
            print(f"[{m.kind} N{w.N} K{w.K}] it {it} loss/model {loss.item()/M:.4f} ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def local_search(m, ex_sym, ex_goal):
    """K rollouts: set every tissue to goal g; record per-(example, tissue) error; pick best g per (tissue, symbol)."""
    w = m.w; M, B = ex_sym.shape[:2]
    err = torch.zeros(w.K, M, B, w.N)
    for g in range(w.K):
        gi = torch.full((M, B, w.N), g)
        s = torch.zeros(M, B, C, w.L); u = m.u(F.one_hot(gi, w.K).float())
        for t in range(100): s = m.step(s, u)
        tgt = w.target(ex_goal)        # the DESIRED pattern for each example
        err[g] = ((s[:, :, 0].clamp(0, 1) - tgt) ** 2).reshape(M, B, w.N, TW).mean(-1)
    table = torch.randint(0, w.K, (M, w.N, w.K))                         # unseen entries stay random
    for k in range(M):
        for n in range(w.N):
            for sym in range(w.K):
                sel = ex_sym[k, :, n] == sym
                if sel.any(): table[k, n, sym] = err[:, k, sel, n].mean(-1).argmin()
    return table


@torch.no_grad()
def evaluate(m, inp_fn, meaning, n=64):
    w = m.w; M = m.M
    sym = torch.randint(0, w.K, (M, n, w.N)); goal = meaning_apply(meaning, sym)
    u = m.u(inp_fn(sym)); tgt = w.target(goal)
    def ok(s):
        e = ((s[:, :, 0].clamp(0, 1) - tgt) ** 2).reshape(M, n, w.N, TW).mean(-1)
        return (e < COMMIT).all(-1).float().mean(1)
    s = torch.zeros(M, n, C, w.L)
    for t in range(100): s = m.step(s, u)
    a = ok(s); s = damage(s, w.L, torch.ones(M, n, dtype=torch.bool))
    for t in range(100): s = m.step(s, u)
    return {"patterns": a.tolist(), "after_damage": ok(s).tolist()}


def run(N, K, kind, M=8, iters=int(__import__("os").environ.get("ITERS", 900)), n_ex=8, retrain_steps=400):
    torch.manual_seed(0); np.random.seed(0)
    w = World(N, K); m = Model(w, kind, M, 0); train(m, iters)
    ident = torch.arange(K)[None, None].expand(M, N, -1)
    torch.manual_seed(5)
    new = torch.stack([torch.stack([torch.randperm(K) for _ in range(N)]) for _ in range(M)])  # (M,N,K)
    ex_sym = torch.randint(0, K, (M, n_ex, N))
    for j in range(K):        # first K examples cover every (tissue, symbol) once; same examples for both models
        ex_sym[:, j] = (j + torch.arange(N)) % K
    res = {"N": N, "K": K, "kind": kind}
    if kind == "flat":
        res["orig"] = evaluate(m, lambda s: F.one_hot(s, K).float(), ident)
        t0 = time.time(); mm = copy.deepcopy(m); opt = torch.optim.Adam(mm.parameters(), lr=3e-3)
        for _ in range(retrain_steps):
            b = ex_sym.repeat(1, 2, 1)
            loss, _ = episode(mm, F.one_hot(b, K).float(), meaning_apply(new, b))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mm.parameters(), M); opt.step()
        res["new"] = evaluate(mm, lambda s: F.one_hot(s, K).float(), new); res["retarget_seconds"] = time.time() - t0
        res["retarget_cost"] = f"{retrain_steps} gradient steps"
    else:
        tab0 = local_search(m, ex_sym, meaning_apply(ident, ex_sym))
        res["orig"] = evaluate(m, lambda s: F.one_hot(meaning_apply(tab0, s), K).float(), ident)
        t0 = time.time(); tab = local_search(m, ex_sym, meaning_apply(new, ex_sym))
        res["new"] = evaluate(m, lambda s: F.one_hot(meaning_apply(tab, s), K).float(), new)
        res["retarget_seconds"] = time.time() - t0; res["retarget_cost"] = f"{K} rollouts (no gradients)"
        seen = torch.stack([torch.stack([torch.stack([(ex_sym[k, :, n] == s).any() for s in range(K)])
                                         for n in range(N)]) for k in range(M)])
        res["examples_covered_every_case"] = seen.flatten(1).all(1).tolist()
        torch.save(m.state_dict(), f"panel_N{N}_K{K}.pt")
    return res


if __name__ == "__main__":
    torch.set_num_threads(1)
    N, K, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    r = run(N, K, kind)
    json.dump(r, open(f"cp_N{N}_K{K}_{kind}.json", "w"))
    print("RESULT", json.dumps({k: (np.round(np.mean(v["patterns"]), 2), np.round(np.mean(v["after_damage"]), 2))
                                if isinstance(v, dict) else v for k, v in r.items()}), flush=True)
