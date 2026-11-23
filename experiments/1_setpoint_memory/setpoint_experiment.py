"""
Setpoint network vs unrolled NCA: a minimal empirical test of Levin-derived principles.

Both models share the SAME local update network (same parameter count) and the SAME
training regime (damage + goal swaps). They differ only in how they are run and trained:

  NCA (baseline):  s <- s + delta(s, T)   stochastic cell updates, full backprop through time
  Setpoint (ours): s <- s + 0.5 * (tanh(net(s, T)) - s)   damped map whose fixed point is the
                   answer; solve with no gradient, backprop only through the last K steps
                   ("phantom gradient" approximation to implicit differentiation), plus a
                   fixed-point residual penalty.

Tests (all with frozen weights):
  reach        grow target from a seed
  damage       grow, erase a disc, keep running
  swap_seen    grow goal A, switch goal register to B (transition seen in training)
  swap_heldout same, for transitions never seen in training
  long_run     run 8x longer than training horizon
  steps curve  error vs number of iterations (test-time compute)

Usage: python setpoint_experiment.py --seeds 0 1 2 --iters 1200 --out results
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 24
C = 12          # channel 0 = visible output, rest hidden
N_GOALS = 4
HIDDEN = 64
TRAIN_STEPS = (32, 48)

# Transitions used during training; the rest are held out.
SEEN_SWAPS = [(0, 1), (1, 2), (2, 3), (3, 0)]
HELDOUT_SWAPS = [(a, b) for a in range(N_GOALS) for b in range(N_GOALS)
                 if a != b and (a, b) not in SEEN_SWAPS]


def make_targets():
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    cy, cx = (H - 1) / 2, (W - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = (r <= 8).float()
    ring = (((yy - cy).abs() <= 8) & ((xx - cx).abs() <= 8)).float() - \
           (((yy - cy).abs() <= 5) & ((xx - cx).abs() <= 5)).float()
    cross = (((yy - cy).abs() <= 2) & ((xx - cx).abs() <= 9)).float().clamp(max=1) + \
            (((xx - cx).abs() <= 2) & ((yy - cy).abs() <= 9)).float()
    cross = cross.clamp(max=1)
    tri = ((yy >= cy - 8) & (yy <= cy + 8) & ((xx - cx).abs() <= (yy - (cy - 8)) * 0.55)).float()
    return torch.stack([disk, ring, cross, tri])  # (4, H, W)


class UpdateNet(nn.Module):
    """Shared local rule: fixed identity/sobel perception -> goal-conditioned MLP per cell."""
    def __init__(self):
        super().__init__()
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float() / 8
        k = torch.stack([ident, sx, sx.T])                       # (3,3,3)
        self.register_buffer("kernel", k.repeat(C, 1, 1).unsqueeze(1))  # (3C,1,3,3)
        self.l1 = nn.Conv2d(3 * C, HIDDEN, 1, bias=False)          # state path
        self.l1x = nn.Conv2d(N_GOALS + 2, HIDDEN, 1)                # goal + position injection
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx])[None])  # fixed positional input (same for both models)
        self.l2 = nn.Conv2d(HIDDEN, C, 1)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)

    def forward(self, s, goal):
        p = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="constant", value=0.0), self.kernel, groups=C)
        g = goal[:, :, None, None].expand(-1, -1, s.shape[2], s.shape[3])
        xy = self.coords.expand(s.shape[0], -1, -1, -1)
        return self.l2(F.relu(self.l1(p) + self.l1x(torch.cat([g, xy], 1))))


class NCA(nn.Module):
    kind = "nca"
    def __init__(self):
        super().__init__(); self.net = UpdateNet()
    def step(self, s, goal):
        d = self.net(s, goal)
        mask = (torch.rand(s.shape[0], 1, H, W) < 0.5).float() if self.training else 1.0
        return s + d * mask
    def output(self, s):
        return s[:, :1]


class Setpoint(nn.Module):
    kind = "setpoint"
    # Contractive in the STATE by construction: perception norm <= sqrt(3), spectrally normalised
    # state-path convs (norm <= 1), relu/tanh 1-Lipschitz, scale c < 1/sqrt(3) => G is a contraction
    # in s, so each goal T has exactly one fixed point s*(T), reached from ANY starting state.
    # The goal/position injection (l1x) is unconstrained: contraction only limits dependence on s.
    C_SCALE = 0.97 / math.sqrt(3)
    def __init__(self):
        super().__init__(); self.net = UpdateNet()
        sn = torch.nn.utils.parametrizations.spectral_norm
        self.net.l1 = sn(self.net.l1); nn.init.normal_(self.net.l2.weight, std=0.01)
        self.net.l2 = sn(self.net.l2)
    def G(self, s, goal):
        return torch.tanh(self.C_SCALE * self.net(s, goal))
    def step(self, s, goal):
        return s + 0.5 * (self.G(s, goal) - s)
    def output(self, s):
        return (s[:, :1] + 1) / 2                  # map [-1,1] -> [0,1]


class SoftSetpoint(Setpoint):
    """Same fixed-point solver + implicit gradient, but NO hard contraction: stability is
    encouraged with a Jacobian penalty (Hutchinson estimate of ||J||_F^2) instead."""
    kind = "soft_setpoint"
    def __init__(self):
        nn.Module.__init__(self); self.net = UpdateNet()
        nn.init.normal_(self.net.l2.weight, std=0.01)
    def G(self, s, goal):
        return torch.tanh(self.net(s, goal))


def seed_state(b, model):
    s = torch.zeros(b, C, H, W)
    if model.kind != "nca":
        s[:] = -1.0; s[:, 1:, H // 2, W // 2] = 1.0
    else:
        s[:, 1:, H // 2, W // 2] = 1.0
    return s


def damage(s, model, rng):
    s = s.clone()
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    for i in range(s.shape[0]):
        cy, cx = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        rad = rng.uniform(4, 7)
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2
        s[i][:, m] = 0.0 if model.kind == "nca" else -1.0
    return s


def onehot(idx):
    return F.one_hot(torch.as_tensor(idx), N_GOALS).float()


def run(model, s, goal, n, grad=True):
    with torch.set_grad_enabled(grad):
        for _ in range(n):
            s = model.step(s, goal)
    return s


@torch.no_grad()
def solve(model, s, goal, max_steps=96, tol=1e-5):
    """Forward solver: iterate the damped map until the residual is small."""
    for _ in range(max_steps):
        s_new = model.step(s, goal)
        if ((s_new - s) ** 2).mean() < tol:
            return s_new
        s = s_new
    return s


def implicit_step(model, s_star, goal, n_back=30):
    """Implicit-function-theorem gradient at the fixed point (DEQ-style).
    Backward solves g = J^T g + dL/ds by fixed-point iteration, in constant memory."""
    s_star = s_star.detach()
    z = model.step(s_star, goal)                 # gradient path to parameters
    z0 = z.detach().requires_grad_()
    f0 = model.step(z0, goal)
    def hook(grad):
        g = grad
        for _ in range(n_back):
            g = torch.autograd.grad(f0, z0, g, retain_graph=True)[0] + grad
        return g
    z.register_hook(hook)
    return z


def train(model, targets, iters, seed, batch=8, k_grad=2, log=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    model.train(); t0 = time.time()
    for it in range(iters):
        g1 = rng.integers(0, N_GOALS, batch)
        mode = rng.choice(["plain", "damage", "swap"], p=[0.4, 0.3, 0.3])
        g2 = g1.copy()
        if mode == "swap":
            pairs = [SEEN_SWAPS[j] for j in rng.integers(0, len(SEEN_SWAPS), batch)]
            g1 = np.array([p[0] for p in pairs]); g2 = np.array([p[1] for p in pairs])
        n1 = int(rng.integers(*TRAIN_STEPS)); n2 = int(rng.integers(*TRAIN_STEPS)) if mode != "plain" else 0
        s = seed_state(batch, model)
        if model.kind == "nca":
            s = run(model, s, onehot(g1), n1, grad=True)
            if mode == "damage": s = damage(s, model, rng)
            s = run(model, s, onehot(g2), n2, grad=True)
            reg = torch.tensor(0.0)
        else:
            s = run(model, s, onehot(g1), n1, grad=False)
            if mode == "damage": s = damage(s, model, rng)
            s = solve(model, s, onehot(g2), max_steps=max(n2, 48) + 32)
            s = implicit_step(model, s, onehot(g2))
            reg = ((model.G(s, onehot(g2)) - s) ** 2).mean()   # fixed-point residual
            if model.kind == "soft_setpoint":                    # Jacobian penalty
                z = s.detach().requires_grad_(); v = torch.randn_like(z)
                jv = torch.autograd.grad(model.G(z, onehot(g2)), z, v, create_graph=True)[0]
                reg = reg + 0.5 * (jv ** 2).sum() / v.numel()
        out = model.output(s)
        loss = F.mse_loss(out, targets[g2].unsqueeze(1)) + 0.1 * reg
        if model.kind == "nca":   # keep hidden state bounded (standard NCA practice)
            loss = loss + 1e-3 * (s ** 2).mean()
        opt.zero_grad(); loss.backward()
        for p in model.parameters():
            if p.grad is not None: p.grad /= (p.grad.norm() + 1e-8)  # per-param grad norm (NCA standard)
        opt.step(); sched.step()
        if log is not None and (it % 200 == 0 or it == iters - 1):
            print(f"[{model.kind} seed{seed}] it {it:5d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return time.time() - t0


@torch.no_grad()
def mse(model, s, targets, g):
    out = model.output(s).clamp(0, 1)
    return F.mse_loss(out, targets[g].unsqueeze(1), reduction="none").mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(model, targets, seed, trials=8):
    model.eval(); rng = np.random.default_rng(1000 + seed); torch.manual_seed(1000 + seed)
    res = {}
    g = np.repeat(np.arange(N_GOALS), trials)
    s = run(model, seed_state(len(g), model), onehot(g), 48, grad=False)
    res["reach"] = mse(model, s, targets, g).mean().item()
    sd = run(model, damage(s, model, rng), onehot(g), 48, grad=False)
    res["damage"] = mse(model, sd, targets, g).mean().item()
    for name, pairs in [("swap_seen", SEEN_SWAPS), ("swap_heldout", HELDOUT_SWAPS)]:
        a = np.array([p[0] for p in pairs for _ in range(trials // 2)])
        b = np.array([p[1] for p in pairs for _ in range(trials // 2)])
        s1 = run(model, seed_state(len(a), model), onehot(a), 48, grad=False)
        s2 = run(model, s1, onehot(b), 96, grad=False)
        res[name] = mse(model, s2, targets, b).mean().item()
    s_long = run(model, seed_state(len(g), model), onehot(g), 400, grad=False)
    res["long_run_400"] = mse(model, s_long, targets, g).mean().item()
    curve = {}
    s = seed_state(len(g), model); done = 0
    for n in [8, 16, 32, 48, 64, 128, 256]:
        s = run(model, s, onehot(g), n - done, grad=False); done = n
        curve[n] = mse(model, s, targets, g).mean().item()
    res["steps_curve"] = curve
    if model.kind != "nca":
        res["fixed_point_residual"] = ((model.G(s_long, onehot(g)) - s_long) ** 2).mean().item()
    return res


@torch.no_grad()
def filmstrip(model, targets, seed):
    """Frames: grow disk(0) -> damage -> recover -> swap to cross(2) (held-out transition)."""
    model.eval(); rng = np.random.default_rng(7 + seed)
    frames = []
    s = seed_state(1, model); g = onehot([0])
    for n in [8, 16, 24]:   # cumulative steps 8, 24, 48
        s = run(model, s, g, n, grad=False)
        frames.append(model.output(s)[0, 0].clamp(0, 1).numpy())
    s = damage(s, model, rng); frames.append(model.output(s)[0, 0].clamp(0, 1).numpy())
    s = run(model, s, g, 48, grad=False); frames.append(model.output(s)[0, 0].clamp(0, 1).numpy())
    s = run(model, s, onehot([2]), 96, grad=False); frames.append(model.output(s)[0, 0].clamp(0, 1).numpy())
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["nca", "setpoint"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); torch.set_num_threads(os.cpu_count())
    targets = make_targets()
    CLS = {"nca": NCA, "setpoint": Setpoint, "soft_setpoint": SoftSetpoint}
    all_res = {k: [] for k in args.models}; strips = {}
    for seed in args.seeds:
        for cls in [CLS[k] for k in args.models]:
            torch.manual_seed(seed); m = cls()
            t = train(m, targets, args.iters, seed, log=True)
            r = evaluate(m, targets, seed); r["train_seconds"] = t; r["seed"] = seed
            r["params"] = sum(p.numel() for p in m.parameters())
            all_res[m.kind].append(r); print(m.kind, seed, json.dumps(r), flush=True)
            if seed == args.seeds[0]:
                strips[m.kind] = [f.tolist() for f in filmstrip(m, targets, seed)]
            json.dump({"results": all_res, "strips": strips, "targets": targets.tolist(),
                       "seen_swaps": SEEN_SWAPS, "heldout_swaps": HELDOUT_SWAPS},
                      open(f"{args.out}/results.json", "w"))
