"""
v4 = v3 + --no_coords: removes the coordinate input; grid edges padded with a marker value (1.0).

v3 = v2 + hybrid model "soft_setpoint_pool" (setpoint map trained along trajectories with a
sample pool), tunable share of bistable training tasks, and strict commitment metrics.

v2: Setpoint networks vs NCA, adding (1) a sample-pool NCA baseline and (2) a multistability /
pattern-memory test modelled on Levin's two-headed planaria.

Targets: 0 disk, 1 ring, 2 cross, 3 triangle, 4 left disk, 5 right disk
Goals:   0-3 -> same target; 4 "left"; 5 "right";
         6 "bistable": EITHER left or right disk is correct; which one is decided only by
           where growth starts (seed left -> left disk, seed right -> right disk).
           The choice must be stored in the system's own state, not in the goal register.

Models (identical local update network, identical parameter count):
  nca            unrolled NCA, full backprop through time
  nca_pool       NCA trained with a sample pool (Mordvintsev et al. 2020) - fairer baseline
  soft_setpoint  fixed-point solve + implicit gradient + Jacobian stability penalty
  setpoint       fixed-point solve + implicit gradient + HARD contraction (unique fixed point)

Untrained test of interest - REWRITE: settle in bistable-left, set goal "right" for 48 steps,
set goal back to "bistable" for 200 steps. Does the system now stably hold "right"?

Usage: python setpoint_experiment_v2.py --models nca nca_pool soft_setpoint setpoint --seeds 0 1 2
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 24
C = 12
N_GOALS = 7
BISTABLE = 6
HIDDEN = 64
TRAIN_STEPS = (32, 48)
SEEN_SWAPS = [(0, 1), (1, 2), (2, 3), (3, 0)]
HELDOUT_SWAPS = [(a, b) for a in range(4) for b in range(4) if a != b and (a, b) not in SEEN_SWAPS]
SIDE_OFFSET = 6
USE_COORDS = True    # v4: set False -> no free position input; edges padded with a marker value
EDGE_VALUE = 0.0


def make_targets():
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    cy, cx = (H - 1) / 2, (W - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = (r <= 8).float()
    ring = (((yy - cy).abs() <= 8) & ((xx - cx).abs() <= 8)).float() - \
           (((yy - cy).abs() <= 5) & ((xx - cx).abs() <= 5)).float()
    cross = ((((yy - cy).abs() <= 2) & ((xx - cx).abs() <= 9)) |
             (((xx - cx).abs() <= 2) & ((yy - cy).abs() <= 9))).float()
    tri = ((yy >= cy - 8) & (yy <= cy + 8) & ((xx - cx).abs() <= (yy - (cy - 8)) * 0.55)).float()
    left = (torch.sqrt((yy - cy) ** 2 + (xx - (cx - SIDE_OFFSET)) ** 2) <= 5).float()
    right = (torch.sqrt((yy - cy) ** 2 + (xx - (cx + SIDE_OFFSET)) ** 2) <= 5).float()
    return torch.stack([disk, ring, cross, tri, left, right])


def target_index(goal, side):
    """goal, side: int arrays. Bistable goal's correct target depends on the seed side."""
    goal = np.asarray(goal); side = np.asarray(side)
    return np.where(goal == BISTABLE, np.where(side < 0, 4, 5), np.minimum(goal, 5))


class UpdateNet(nn.Module):
    def __init__(self):
        super().__init__()
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float() / 8
        k = torch.stack([ident, sx, sx.T])
        self.register_buffer("kernel", k.repeat(C, 1, 1).unsqueeze(1))
        self.l1 = nn.Conv2d(3 * C, HIDDEN, 1, bias=False)
        self.l1x = nn.Conv2d(N_GOALS + 2, HIDDEN, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx])[None])
        self.l2 = nn.Conv2d(HIDDEN, C, 1)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)

    def forward(self, s, goal):
        p = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="constant", value=EDGE_VALUE), self.kernel, groups=C)
        g = goal[:, :, None, None].expand(-1, -1, s.shape[2], s.shape[3])
        xy = self.coords.expand(s.shape[0], -1, -1, -1) * (1.0 if USE_COORDS else 0.0)
        return self.l2(F.relu(self.l1(p) + self.l1x(torch.cat([g, xy], 1))))


class NCA(nn.Module):
    kind = "nca"; fill = 0.0
    def __init__(self):
        super().__init__(); self.net = UpdateNet()
    def step(self, s, goal):
        d = self.net(s, goal)
        mask = (torch.rand(s.shape[0], 1, H, W) < 0.5).float() if self.training else 1.0
        return s + d * mask
    def output(self, s):
        return s[:, :1]


class NCAPool(NCA):
    kind = "nca_pool"


class Setpoint(nn.Module):
    kind = "setpoint"; fill = -1.0
    C_SCALE = 0.97 / math.sqrt(3)   # contraction in s by construction (see v1 notes)
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
        return (s[:, :1] + 1) / 2


class SoftSetpoint(Setpoint):
    kind = "soft_setpoint"
    def __init__(self):
        nn.Module.__init__(self); self.net = UpdateNet()
        nn.init.normal_(self.net.l2.weight, std=0.01)
    def G(self, s, goal):
        return torch.tanh(self.net(s, goal))



class SoftSetpointPool(SoftSetpoint):
    """Hybrid: soft-stability setpoint map, but trained along trajectories with a sample pool
    (backprop through the steps), plus the fixed-point residual and Jacobian penalties."""
    kind = "soft_setpoint_pool"


def seed_state(b, model, side):
    side = np.broadcast_to(np.asarray(side), (b,))
    s = torch.full((b, C, H, W), model.fill)
    for i in range(b):
        s[i, 1:, H // 2, W // 2 + int(side[i]) * SIDE_OFFSET] = 1.0
    return s


def damage(s, model, rng, idx=None):
    s = s.clone()
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    for i in (range(s.shape[0]) if idx is None else idx):
        cy, cx = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        rad = rng.uniform(4, 7)
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2
        s[i][:, m] = model.fill
    return s


def onehot(idx):
    return F.one_hot(torch.as_tensor(np.asarray(idx)), N_GOALS).float()


def run(model, s, goal, n, grad=True):
    with torch.set_grad_enabled(grad):
        for _ in range(n):
            s = model.step(s, goal)
    return s


@torch.no_grad()
def solve(model, s, goal, max_steps=96, tol=1e-5):
    for _ in range(max_steps):
        s_new = model.step(s, goal)
        if ((s_new - s) ** 2).mean() < tol:
            return s_new
        s = s_new
    return s


def implicit_step(model, s_star, goal, n_back=30):
    s_star = s_star.detach()
    z = model.step(s_star, goal)
    z0 = z.detach().requires_grad_()
    f0 = model.step(z0, goal)
    def hook(grad):
        g = grad
        for _ in range(n_back):
            g = torch.autograd.grad(f0, z0, g, retain_graph=True)[0] + grad
        return g
    z.register_hook(hook)
    return z


BISTABLE_FRAC = 1 / N_GOALS


def sample_task(rng, b):
    goal = rng.integers(0, N_GOALS - 1, b)
    goal = np.where(rng.random(b) < BISTABLE_FRAC, BISTABLE, goal)
    side = np.where(goal == BISTABLE, rng.choice([-1, 1], b), 0)
    return goal, side


def apply_seen_swaps(goal, rng):
    g2 = goal.copy()
    for i in range(len(g2)):
        if g2[i] < 4:
            g2[i] = dict(SEEN_SWAPS)[g2[i]]
    return g2


def grad_norm_step(model, opt, loss):
    opt.zero_grad(); loss.backward()
    for p in model.parameters():
        if p.grad is not None: p.grad /= (p.grad.norm() + 1e-8)
    opt.step()


def train(model, targets, iters, seed, batch=8, log=True):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    model.train(); t0 = time.time()
    if model.kind.endswith("pool"):
        POOL = 256
        pg, ps = sample_task(rng, POOL)
        pool = seed_state(POOL, model, ps); ptgt = target_index(pg, ps)
    for it in range(iters):
        if model.kind.endswith("pool"):
            idx = rng.choice(POOL, batch, replace=False)
            s = pool[idx].clone(); g = pg[idx].copy(); tg = ptgt[idx].copy()
            with torch.no_grad():   # replace worst sample with a fresh seed
                err = ((model.output(s) - targets[tg].unsqueeze(1)) ** 2).mean(dim=(1, 2, 3))
            order = torch.argsort(err, descending=True).numpy()
            ng, ns = sample_task(rng, 1)
            s[order[0]] = seed_state(1, model, ns)[0]; g[order[0]] = ng[0]; tg[order[0]] = target_index(ng, ns)[0]
            s = damage(s, model, rng, idx=order[-2:])   # damage the two best
            for i in order[1:3]:                         # goal swaps (seen transitions only)
                if g[i] < 4 and rng.random() < 0.5:
                    g[i] = dict(SEEN_SWAPS)[g[i]]; tg[i] = g[i]
            s = run(model, s, onehot(g), int(rng.integers(*TRAIN_STEPS)), grad=True)
            loss = F.mse_loss(model.output(s), targets[tg].unsqueeze(1))
            if model.kind == "nca_pool":
                loss = loss + 1e-3 * (s ** 2).mean()
            else:   # same stability penalties as soft_setpoint, applied at the end of the trajectory
                gg = onehot(g)
                loss = loss + 0.1 * ((model.G(s, gg) - s) ** 2).mean()
                z = s.detach().requires_grad_(); v = torch.randn_like(z)
                jv = torch.autograd.grad(model.G(z, gg), z, v, create_graph=True)[0]
                loss = loss + 0.5 * (jv ** 2).sum() / v.numel()
            grad_norm_step(model, opt, loss); sched.step()
            pool[idx] = s.detach(); pg[idx] = g; ptgt[idx] = tg
        else:
            g1, side = sample_task(rng, batch)
            mode = rng.choice(["plain", "damage", "swap"], p=[0.4, 0.3, 0.3])
            g2 = apply_seen_swaps(g1, rng) if mode == "swap" else g1.copy()
            tg = target_index(g2, side)
            n1 = int(rng.integers(*TRAIN_STEPS)); n2 = int(rng.integers(*TRAIN_STEPS)) if mode != "plain" else 0
            s = seed_state(batch, model, side)
            if model.kind == "nca":
                s = run(model, s, onehot(g1), n1, grad=True)
                if mode == "damage": s = damage(s, model, rng)
                s = run(model, s, onehot(g2), n2, grad=True)
                reg = 1e-3 * (s ** 2).mean()
            else:
                s = run(model, s, onehot(g1), n1, grad=False)
                if mode == "damage": s = damage(s, model, rng)
                s = solve(model, s, onehot(g2), max_steps=max(n2, 48) + 32)
                s = implicit_step(model, s, onehot(g2))
                reg = 0.1 * ((model.G(s, onehot(g2)) - s) ** 2).mean()
                if model.kind == "soft_setpoint":
                    z = s.detach().requires_grad_(); v = torch.randn_like(z)
                    jv = torch.autograd.grad(model.G(z, onehot(g2)), z, v, create_graph=True)[0]
                    reg = reg + 0.5 * (jv ** 2).sum() / v.numel()   # same weight as v1
            loss = F.mse_loss(model.output(s), targets[tg].unsqueeze(1)) + reg
            grad_norm_step(model, opt, loss); sched.step()
        if log and (it % 300 == 0 or it == iters - 1):
            print(f"[{model.kind} seed{seed}] it {it:5d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return time.time() - t0


@torch.no_grad()
def err(model, s, targets, tg):
    out = model.output(s).clamp(0, 1)
    return ((out - targets[np.asarray(tg)].unsqueeze(1)) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def picks_right(model, s, targets):
    """True where the output is closer to the right disk than the left disk."""
    return (err(model, s, targets, [5] * len(s)) < err(model, s, targets, [4] * len(s))).float()


@torch.no_grad()
def evaluate(model, targets, seed, trials=8):
    model.eval(); rng = np.random.default_rng(1000 + seed); torch.manual_seed(1000 + seed)
    res = {}
    g = np.repeat(np.arange(4), trials)
    s = run(model, seed_state(len(g), model, 0), onehot(g), 48, grad=False)
    res["reach"] = err(model, s, targets, g).mean().item()
    res["damage"] = err(model, run(model, damage(s, model, rng), onehot(g), 48, False), targets, g).mean().item()
    for name, pairs in [("swap_seen", SEEN_SWAPS), ("swap_heldout", HELDOUT_SWAPS)]:
        a = np.array([p[0] for p in pairs for _ in range(trials // 2)])
        b = np.array([p[1] for p in pairs for _ in range(trials // 2)])
        s1 = run(model, seed_state(len(a), model, 0), onehot(a), 48, False)
        res[name] = err(model, run(model, s1, onehot(b), 96, False), targets, b).mean().item()
    res["long_run_400"] = err(model, run(model, seed_state(len(g), model, 0), onehot(g), 400, False), targets, g).mean().item()
    curve = {}; s = seed_state(len(g), model, 0); done = 0
    for n in [8, 16, 32, 48, 64, 128, 256]:
        s = run(model, s, onehot(g), n - done, False); done = n
        curve[n] = err(model, s, targets, g).mean().item()
    res["steps_curve"] = curve

    # explicit left/right goals
    g_lr = np.repeat([4, 5], trials)
    s = run(model, seed_state(len(g_lr), model, 0), onehot(g_lr), 64, False)
    res["explicit_lr"] = err(model, s, targets, g_lr).mean().item()

    # bistable: same goal register, two valid outcomes decided by start position
    side = np.repeat([-1, 1], trials); gb = np.full(len(side), BISTABLE); tb = target_index(gb, side)
    s = run(model, seed_state(len(side), model, side), onehot(gb), 64, False)
    res["bistable_reach"] = err(model, s, targets, tb).mean().item()
    res["bistable_correct_side"] = (picks_right(model, s, targets).numpy() == (side > 0)).mean().item()
    sd = run(model, damage(s, model, rng), onehot(gb), 48, False)
    res["bistable_damage"] = err(model, sd, targets, tb).mean().item()
    s_long = run(model, s, onehot(gb), 400, False)
    res["bistable_long"] = err(model, s_long, targets, tb).mean().item()
    res["bistable_long_correct_side"] = (picks_right(model, s_long, targets).numpy() == (side > 0)).mean().item()

    # REWRITE (never trained): bistable-left -> goal "right" 48 steps -> back to bistable 200 steps
    n = 2 * trials
    s = run(model, seed_state(n, model, -1), onehot(np.full(n, BISTABLE)), 64, False)
    s = run(model, s, onehot(np.full(n, 5)), 48, False)
    res["rewrite_reached_right"] = picks_right(model, s, targets).mean().item()
    s = run(model, s, onehot(np.full(n, BISTABLE)), 200, False)
    res["rewrite_holds_right"] = picks_right(model, s, targets).mean().item()
    res["rewrite_err_vs_right"] = err(model, s, targets, [5] * n).mean().item()
    COMMIT = 0.03   # "committed" = within 0.03 MSE of one disk (hedge ~0.07, blank ~0.14)
    def committed(s, want):
        e = err(model, s, targets, want)
        return (e < COMMIT).float().mean().item()
    s0 = run(model, seed_state(len(side), model, side), onehot(gb), 64, False)
    res["strict_bistable_64"] = committed(s0, tb)
    res["strict_bistable_400"] = committed(run(model, s0, onehot(gb), 400, False), tb)
    s0 = damage(s0, model, rng)
    res["strict_bistable_damage"] = committed(run(model, s0, onehot(gb), 96, False), tb)
    sr = run(model, seed_state(n, model, -1), onehot(np.full(n, BISTABLE)), 64, False)
    res["strict_rewrite_start_left"] = committed(sr, [4] * n)
    sr = run(model, sr, onehot(np.full(n, 5)), 48, False)
    sr = run(model, sr, onehot(np.full(n, BISTABLE)), 200, False)
    res["strict_rewrite_holds_right"] = committed(sr, [5] * n)
    if model.kind not in ("nca", "nca_pool"):
        res["fixed_point_residual"] = ((model.G(s_long, onehot(gb)) - s_long) ** 2).mean().item()
    return res


@torch.no_grad()
def filmstrip(model, targets, seed):
    """Bistable story: seed LEFT -> settle -> damage -> recover -> goal 'right' -> back to bistable."""
    model.eval(); rng = np.random.default_rng(7 + seed)
    o = lambda s: model.output(s)[0, 0].clamp(0, 1).numpy().tolist()
    gb, gr = onehot([BISTABLE]), onehot([5]); fr = []
    s = run(model, seed_state(1, model, -1), gb, 64, False); fr.append(o(s))
    s = damage(s, model, rng); fr.append(o(s))
    s = run(model, s, gb, 48, False); fr.append(o(s))
    s = run(model, s, gr, 48, False); fr.append(o(s))
    s = run(model, s, gb, 200, False); fr.append(o(s))
    return fr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--models", nargs="+", default=["nca", "nca_pool", "soft_setpoint", "setpoint"])
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no_coords", action="store_true")
    ap.add_argument("--bistable_frac", type=float, default=1 / N_GOALS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); torch.set_num_threads(args.threads or os.cpu_count())
    targets = make_targets()
    CLS = {"nca": NCA, "nca_pool": NCAPool, "setpoint": Setpoint, "soft_setpoint": SoftSetpoint,
           "soft_setpoint_pool": SoftSetpointPool}
    BISTABLE_FRAC = args.bistable_frac
    if args.no_coords:
        USE_COORDS = False; EDGE_VALUE = 1.0   # edge marker distinct from both empty fills (0 / -1)
    all_res = {k: [] for k in args.models}; strips = {}
    for seed in args.seeds:
        for k in args.models:
            torch.manual_seed(seed); m = CLS[k]()
            t = train(m, targets, args.iters, seed)
            r = evaluate(m, targets, seed); r.update(train_seconds=t, seed=seed,
                                                    params=sum(p.numel() for p in m.parameters()))
            all_res[k].append(r); print(k, seed, json.dumps(r), flush=True)
            if seed == args.seeds[0]: strips[k] = filmstrip(m, targets, seed)
            json.dump({"results": all_res, "strips": strips, "targets": targets.tolist()},
                      open(f"{args.out}/results.json", "w"))
