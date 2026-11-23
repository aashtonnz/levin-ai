"""
Goals at every level, where the LOWER level does the hard work.

Strip of 32 cells = 4 tissues x 8 cells. Each tissue must grow one of two motifs (motif 0: cells 1-3 of the
tissue on; motif 1: cells 4-6 on), from a blank start, with NO position input inside the tissue (cells know
only their tissue id and can sense neighbouring tissue ids), under constant noise and random damage.
Instruction: one bit per tissue (which motif) - the top-level job is simple; the cell-level job is hard.

Models (same NCA cell network; all seeds trained at once as an ensemble):
  flat       cells receive their tissue's instruction bit directly; trained end-to-end on instructions.
  hier_goal  cells receive a goal vector T = h(bit, tissue id) from an upper level.
             Phase 1 (70% of budget): lower level alone learns to reach ANY goal it is given
             (random one-hot motif goal per tissue). Phase 2 (30%): lower frozen; h learns to set goals.

Tests:
  in-distribution: all 16 instructions correct at t=100; and after damage (+100 steps)
  retarget: bits now mean the OTHER motif. Adapt from 4 example instructions for 150 steps.
            flat adapts all weights, or only its instruction-input weights; hier adapts only h.
            Then re-test patterns AND damage repair under the new meaning.
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F

L, NT, TW, C, HID = 32, 4, 8, 8, 32
NOISE, EDGE, COMMIT = 0.02, 1.0, 0.05
xs = torch.arange(L)
TISSUE = xs // TW
LOCAL = xs % TW
MOTIFS = torch.stack([((LOCAL >= 1) & (LOCAL <= 3)).float(), ((LOCAL >= 4) & (LOCAL <= 6)).float()])  # (2, L)
IDCH = F.one_hot(TISSUE, NT).float().T                                                         # (4, L)
CODES = torch.tensor([[(i >> k) & 1 for k in range(NT)] for i in range(16)])                   # (16, 4)
KER = torch.stack([torch.tensor([0., 1, 0]), torch.tensor([-.5, 0, .5]), torch.tensor([1., -2, 1])])
KER = KER.repeat(C + NT, 1).unsqueeze(1)


def target(motif_per_tissue):
    """(M,B,4) motif ids -> (M,B,L) target"""
    m = motif_per_tissue[..., TISSUE]                    # (M,B,L)
    return torch.where(m == 0, MOTIFS[0], MOTIFS[1])


class Model(torch.nn.Module):
    def __init__(self, kind, M, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind, self.M = kind, M
        self.u_dim = 1 if kind == "flat" else 2
        P = 3 * (C + NT) + self.u_dim
        self.W1 = torch.nn.Parameter(torch.randn(M, HID, P, generator=g) / P ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(M, HID))
        self.W2 = torch.nn.Parameter(torch.zeros(M, C, HID))
        self.b2 = torch.nn.Parameter(torch.zeros(M, C))
        if kind == "hier_goal":   # upper level: (bit, tissue id) -> goal (2)
            self.H1 = torch.nn.Parameter(torch.randn(M, 16, 1 + NT, generator=g) / 5 ** 0.5)
            self.hb1 = torch.nn.Parameter(torch.zeros(M, 16))
            self.H2 = torch.nn.Parameter(torch.randn(M, 2, 16, generator=g) / 4)
            self.hb2 = torch.nn.Parameter(torch.zeros(M, 2))

    def lower_params(self): return [self.W1, self.b1, self.W2, self.b2]
    def upper_params(self): return [self.H1, self.hb1, self.H2, self.hb2]

    def goals(self, bits):
        """bits (M,B,4) -> per-tissue goal (M,B,4,2)"""
        x = torch.cat([bits[..., None].float(), torch.eye(NT).expand(*bits.shape, NT)], -1)
        h = F.relu(torch.einsum("mhi,mbti->mbth", self.H1, x) + self.hb1[:, None, None])
        return torch.einsum("moh,mbth->mbto", self.H2, h) + self.hb2[:, None, None]

    def cell_input(self, bits=None, goal=None):
        """-> (M,B,u_dim,L)"""
        if self.kind == "flat":
            return bits.float()[..., TISSUE][:, :, None, :]
        if goal is None: goal = self.goals(bits)
        return goal[:, :, TISSUE, :].permute(0, 1, 3, 2)

    def step(self, s, u):
        M, B = s.shape[:2]
        full = torch.cat([s, IDCH.expand(M, B, NT, L)], 2)
        p = F.conv1d(F.pad(full.reshape(M * B, C + NT, L), (1, 1), value=EDGE), KER, groups=C + NT)
        x = torch.cat([p.reshape(M, B, -1, L), u], 2)
        h = F.relu(torch.einsum("mhp,mbpl->mbhl", self.W1, x) + self.b1[:, None, :, None])
        d = torch.einsum("mch,mbhl->mbcl", self.W2, h) + self.b2[:, None, :, None]
        return (s + d + NOISE * torch.randn_like(s)).clamp(-2, 2)


def damage(s, mask):
    M, B = s.shape[:2]
    start = torch.randint(0, L - 8, (M, B)); length = torch.randint(3, 8, (M, B))
    seg = (xs >= start[..., None]) & (xs < (start + length)[..., None]) & mask[..., None]
    return s * (~seg)[:, :, None, :].float()


def episode(model, u_fn, motif, T=100, dmg=True, grad=True):
    """Run from blank state; dense loss after t=35. u_fn() gives the cell input (recomputed: keeps graph)."""
    M, B = motif.shape[:2]
    s = torch.zeros(M, B, C, L); tgt = target(motif); loss = 0.0; n = 0
    t_d = int(np.random.randint(45, 75)); do = torch.rand(M, B) < 0.6
    u = u_fn()
    with torch.set_grad_enabled(grad):
        for t in range(T):
            if dmg and t == t_d: s = damage(s, do)
            s = model.step(s, u)
            if t >= 35 and t % 5 == 0:
                loss = loss + ((s[:, :, 0] - tgt) ** 2).mean(-1).mean(1).sum(); n += 1
    return loss / n, s


def train(model, iters, B=16, log=True):
    M = model.M; t0 = time.time()
    if model.kind == "flat":
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for it in range(iters):
            bits = CODES[torch.randint(0, 16, (M, B))]
            loss, _ = episode(model, lambda: model.cell_input(bits=bits), bits)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), M); opt.step()
            if log and it % 300 == 0: print(f"[flat] it {it} loss/model {loss.item()/M:.4f} ({time.time()-t0:.0f}s)", flush=True)
    else:
        n1 = int(iters * 0.7)
        opt = torch.optim.Adam(model.lower_params(), lr=3e-3)
        for it in range(n1):   # phase 1: lower-level competency on random goals
            motif = torch.randint(0, 2, (M, B, NT))
            goal = F.one_hot(motif, 2).float()
            loss, _ = episode(model, lambda: model.cell_input(goal=goal), motif)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.lower_params(), M); opt.step()
            if log and it % 300 == 0: print(f"[hier lower] it {it} loss/model {loss.item()/M:.4f} ({time.time()-t0:.0f}s)", flush=True)
        for p in model.lower_params(): p.requires_grad_(False)
        opt = torch.optim.Adam(model.upper_params(), lr=3e-3)
        for it in range(iters - n1):   # phase 2: upper level learns to set goals
            bits = CODES[torch.randint(0, 16, (M, B))]
            loss, _ = episode(model, lambda: model.cell_input(bits=bits), bits)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[hier upper] final loss/model {loss.item()/M:.4f} ({time.time()-t0:.0f}s)", flush=True)
        for p in model.lower_params(): p.requires_grad_(True)


@torch.no_grad()
def evaluate(model, meaning="orig", reps=4):
    """Fraction of trials with ALL 4 tissues correct; at t=100, and after damage at t=100 (+100 steps)."""
    M = model.M
    bits = CODES.repeat(reps, 1)[None].expand(M, -1, -1)
    motif = bits if meaning == "orig" else 1 - bits
    u = model.cell_input(bits=bits); tgt = target(motif)
    s = torch.zeros(M, len(CODES) * reps, C, L)
    def ok(s):
        e = ((s[:, :, 0].clamp(0, 1) - tgt) ** 2).reshape(M, -1, NT, TW).mean(-1)
        return (e < COMMIT).all(-1).float().mean(1)
    for t in range(100): s = model.step(s, u)
    a = ok(s)
    s = damage(s, torch.ones(s.shape[:2], dtype=torch.bool))
    for t in range(100): s = model.step(s, u)
    return {"patterns": a.tolist(), "after_damage": ok(s).tolist()}


RETARGET_STEPS = 150


def retarget(model, which, steps=None):
    steps = steps or RETARGET_STEPS
    """New meaning: each bit now selects the OTHER motif. Adapt from 4 example instructions."""
    M = model.M
    idx = torch.stack([torch.randperm(16)[:4] for _ in range(M)])            # 4 examples per model
    if which == "flat_all": ps = list(model.parameters())
    elif which == "flat_input": ps = [model.W1]                                # mask below: only input column
    else: ps = model.upper_params()
    for p in model.parameters(): p.requires_grad_(False)
    for p in ps: p.requires_grad_(True)
    opt = torch.optim.Adam(ps, lr=3e-3)
    for _ in range(steps):
        bits = torch.gather(CODES[None].expand(M, -1, -1), 1, idx[..., None].expand(-1, -1, NT))
        bits = bits.repeat(1, 4, 1)
        loss, _ = episode(model, lambda: model.cell_input(bits=bits), 1 - bits)
        opt.zero_grad(); loss.backward()
        if which == "flat_input": model.W1.grad[:, :, :-1] = 0                # only instruction-input weights
        opt.step()
    return evaluate(model, "new")


if __name__ == "__main__":
    import copy, sys
    M, iters = 8, int(sys.argv[1]) if len(sys.argv) > 1 else 900
    import os; RETARGET_STEPS = int(os.environ.get("RETARGET_STEPS", 150))
    torch.manual_seed(0); np.random.seed(0)
    res = {}
    kinds = sys.argv[2:] or ["flat", "hier_goal"]
    for kind in kinds:
        torch.manual_seed(0); np.random.seed(0)
        m = Model(kind, M, 0); train(m, iters)
        torch.save(m.state_dict(), f"hd_{kind}.pt")
        res[kind] = {}
        res[kind]["orig"] = evaluate(m)
        modes = ["flat_all", "flat_input"] if kind == "flat" else ["hier_upper"]
        for mode in modes:
            mm = copy.deepcopy(m); torch.manual_seed(1); np.random.seed(1)
            res[kind][mode] = retarget(mm, mode)
        print(kind, json.dumps({k: {kk: round(float(np.mean(vv)), 2) for kk, vv in v.items()} for k, v in res[kind].items()}), flush=True)
        json.dump(res, open(f"hier_dynamic_{"_".join(kinds)}.json", "w"))
