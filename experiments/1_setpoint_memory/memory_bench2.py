"""
v2 of the benchmark: longer training (1200 iters), longer episodes (T=140), up to two damage events
per episode (prob 0.7 each), and a fix for the discrete-output baseline (no clamp in its loss).

Fast memory benchmark (1D). Isolates the open problem from the 2D experiments:
COMMIT to a choice, HOLD it through noise and damage, and REWRITE it on command.

World: a strip of L=32 cells, C channels each. Channel 0 is the visible output.
  - Growth starts from a seed at x=8 (left) or x=24 (right).
  - Target: a bump of 11 "on" cells centred on the seed side. Same task signal either way:
    the choice can only be stored in the cells' own state.
  - Rewrite: a brief external pulse (10 steps, broadcast to all cells) saying "left" or "right";
    afterwards the strip should hold the new side indefinitely.
  - Small state noise every step; occasional damage (a segment of cells zeroed).
  - No coordinate input; strip ends padded with a marker value.

Variants (same local network; all seeds of a variant trained at once as an ensemble):
  plain           residual NCA update
  switches        + 3 hidden channels with fixed double-well physics (bistable, not learned)
  switch_output   + the visible channel itself is a double-well switch (cell 'fate' = its switch)
  gated           LSTM-style learned gate on the update (standard ML memory tool)
  discrete        visible output hard-thresholded with a straight-through gradient

Usage: python memory_bench.py --variants plain switches switch_output gated discrete --seeds 8
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

L, C, HID = 32, 8, 32
LEFT, RIGHT, HALF = 8, 24, 5
EDGE = 1.0
NOISE = 0.02
SW = [5, 6, 7]            # hidden switch channels
BETA = 0.1
COMMIT = 0.05             # MSE threshold for "committed" (hedge = 0.17, blank = 0.34, wrong = 0.69)

xs = torch.arange(L).float()
TGT = torch.stack([((xs - LEFT).abs() <= HALF).float(), ((xs - RIGHT).abs() <= HALF).float()])  # (2, L)
KER = torch.stack([torch.tensor([0., 1, 0]), torch.tensor([-.5, 0, .5]), torch.tensor([1., -2, 1])])
KER = KER.repeat(C, 1).unsqueeze(1)    # (3C, 1, 3)


class Ensemble(torch.nn.Module):
    """M independent models with stacked parameters, run in one batched computation."""
    def __init__(self, variant, M, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = 3 * C + 2                                   # perception + 2 pulse channels
        self.variant, self.M = variant, M
        self.W1 = torch.nn.Parameter(torch.randn(M, HID, P, generator=g) / P ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(M, HID))
        self.W2 = torch.nn.Parameter(torch.zeros(M, C, HID))
        self.b2 = torch.nn.Parameter(torch.zeros(M, C))
        if variant == "gated":
            self.Wg = torch.nn.Parameter(torch.randn(M, C, HID, generator=g) * 0.1)
            self.bg = torch.nn.Parameter(torch.zeros(M, C))

    def params(self):
        return sum(p[0].numel() for p in self.parameters())

    def step(self, s, pulse):
        """s: (M,B,C,L); pulse: (M,B,2) -> next state."""
        M, B = s.shape[:2]
        p = F.conv1d(F.pad(s.reshape(M * B, C, L), (1, 1), value=EDGE), KER, groups=C).reshape(M, B, 3 * C, L)
        x = torch.cat([p, pulse[..., None].expand(-1, -1, -1, L)], 2)
        h = F.relu(torch.einsum("mhp,mbpl->mbhl", self.W1, x) + self.b1[:, None, :, None])
        d = torch.einsum("mch,mbhl->mbcl", self.W2, h) + self.b2[:, None, :, None]
        if self.variant == "gated":
            gate = torch.sigmoid(torch.einsum("mch,mbhl->mbcl", self.Wg, h) + self.bg[:, None, :, None])
            s = s + gate * d
        else:
            s = s + d
        s = s + NOISE * torch.randn_like(s)
        if self.variant in ("switches", "switch_output"):
            idx = SW + ([0] if self.variant == "switch_output" else [])
            sw = s[:, :, idx]
            s = s.clone(); s[:, :, idx] = sw + BETA * (sw - sw ** 3)
        return s.clamp(-2, 2)

    def output(self, s):
        o = s[:, :, 0]
        if self.variant == "switch_output":
            return (o + 1) / 2                           # switch at -1/+1 -> off/on
        if self.variant == "discrete":
            hard = (o > 0.5).float()
            return o + (hard - o).detach()               # straight-through estimator
        return o


def init_state(M, B, side):
    """side: (M,B) in {0 left, 1 right}."""
    s = torch.zeros(M, B, C, L)
    pos = torch.where(side == 0, LEFT, RIGHT)
    s[torch.arange(M)[:, None], torch.arange(B)[None], 1:, pos] = 1.0
    return s


def damage(s, mask):
    """Zero a random segment (length 4-10) in the episodes where mask is True."""
    M, B = s.shape[:2]
    start = torch.randint(0, L - 10, (M, B)); length = torch.randint(4, 11, (M, B))
    seg = (xs[None, None] >= start[..., None]) & (xs[None, None] < (start + length)[..., None]) & mask[..., None]
    return s * (~seg)[:, :, None, :].float()


def mse_to(out, side, clamp=True):
    o = out.clamp(0, 1) if clamp else out   # v2: no clamp in the discrete loss - rounding put STE outputs
    return ((o - TGT[side]) ** 2).mean(-1)  # just outside [0,1], where clamp silently zeroed the gradient


def train(model, iters, B=16, T=140, dmg_prob=0.7, log=True):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    M = model.M; t0 = time.time()
    for it in range(iters):
        side = torch.randint(0, 2, (M, B))
        s = init_state(M, B, side)
        do_pulse = torch.rand(M, B) < 0.5; t_p = int(np.random.randint(45, 65))
        do_dmg = torch.rand(M, B) < dmg_prob; t_d = int(np.random.randint(40, 70))
        do_dmg2 = torch.rand(M, B) < dmg_prob; t_d2 = int(np.random.randint(90, 120))   # v2: second damage event
        cur = side.clone(); loss = 0.0; n = 0
        for t in range(T):
            pulse = torch.zeros(M, B, 2)
            if t_p <= t < t_p + 10:
                pulse[..., 0] = (do_pulse & (cur == 1)).float()   # pulse toward the OTHER side
                pulse[..., 1] = (do_pulse & (cur == 0)).float()
            if t == t_p + 10:
                cur = torch.where(do_pulse, 1 - side, side)
            if t == t_d:
                s = damage(s, do_dmg)
            if t == t_d2:
                s = damage(s, do_dmg2)
            s = model.step(s, pulse)
            grace = (t >= t_p) & (t < t_p + 30)
            if t >= 35 and t % 5 == 0:
                e = mse_to(model.output(s), cur, clamp=(model.variant != "discrete"))
                if grace: e = e * (~do_pulse).float()
                loss = loss + e.mean(1).sum(); n += 1
        loss = loss / n
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * M)
        opt.step(); sched.step()
        if log and (it % 200 == 0 or it == iters - 1):
            print(f"[{model.variant}] it {it:4d} loss/model {loss.item() / M:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return time.time() - t0


@torch.no_grad()
def evaluate(model, B=64):
    """All metrics: fraction of trials 'committed' to the correct side (MSE < COMMIT), per model."""
    M = model.M; res = {}
    none = torch.zeros(M, B, 2)
    side = torch.arange(B)[None].expand(M, -1) % 2
    s = init_state(M, B, side)
    traj = []
    for t in range(1000):
        s = model.step(s, none)
        if t + 1 == 100: res["commit_100"] = (mse_to(model.output(s), side) < COMMIT).float().mean(1); s100 = s.clone()
        if t + 1 == 1000: res["hold_1000"] = (mse_to(model.output(s), side) < COMMIT).float().mean(1)
    # damage at t=100, check at t=300
    s = damage(s100, torch.ones(M, B, dtype=torch.bool))
    for t in range(200): s = model.step(s, none)
    res["after_damage"] = (mse_to(model.output(s), side) < COMMIT).float().mean(1)
    # rewrite: 10-step pulse to the other side at t=100; check t=300 and t=1000
    new = 1 - side
    pulse = torch.zeros(M, B, 2); pulse[..., 0] = (new == 0).float(); pulse[..., 1] = (new == 1).float()
    s = s100.clone()
    for t in range(10): s = model.step(s, pulse)
    for t in range(190): s = model.step(s, none)
    res["rewrite_300"] = (mse_to(model.output(s), new) < COMMIT).float().mean(1)
    for t in range(700): s = model.step(s, none)
    res["rewrite_1000"] = (mse_to(model.output(s), new) < COMMIT).float().mean(1)
    return {k: v.tolist() for k, v in res.items()}


@torch.no_grad()
def raster(model, m=0):
    """Space-time picture for model m: seed LEFT, damage at 150, pulse RIGHT at 300, run to 800."""
    s = init_state(model.M, 1, torch.zeros(model.M, 1, dtype=torch.long)); rows = []
    none = torch.zeros(model.M, 1, 2); right = none.clone(); right[..., 1] = 1
    for t in range(800):
        if t == 150: s = damage(s, torch.ones(model.M, 1, dtype=torch.bool))
        s = model.step(s, right if 300 <= t < 310 else none)
        rows.append(model.output(s)[m, 0].clamp(0, 1).tolist())
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["plain", "switches", "switch_output", "gated", "discrete"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--out", default="bench2")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); os.makedirs(args.out, exist_ok=True)
    for v in args.variants:
        torch.manual_seed(0); np.random.seed(0)
        m = Ensemble(v, args.seeds, seed=0)
        t = train(m, args.iters)
        r = evaluate(m); r["train_seconds"] = t; r["params_per_model"] = m.params()
        torch.save(m.state_dict(), f"{args.out}/{v}.pt")
        r["raster"] = raster(m, int(np.argmax(r["hold_1000"])))   # show the best seed
        json.dump(r, open(f"{args.out}/{v}.json", "w"))
        print(v, {k: round(float(np.mean(x)), 2) for k, x in r.items() if isinstance(x, list) and k != "raster"},
              f"{t:.0f}s", flush=True)
