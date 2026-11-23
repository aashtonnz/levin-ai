"""
2D test of "competent parts + small goal panel, steered by discovery + search" vs retraining.

World: 3x3 grid of tissues, each 5x5 cells (15x15 cells). Each tissue has 4 options (goals):
  interior motif (centre 3x3): 0 horizontal bar, 1 vertical bar, 2 four corners, 3 ring
  boundary marking: a tissue lights an edge (middle 3 cells of that side) where the neighbour on that side
  has a DIFFERENT goal (no neighbour -> off). So each tissue depends on up to 4 neighbours, and the influence
  map has loops (unlike the 1D chain).
Cells: no coordinates; they see their own + neighbouring states, fixed tissue-id channels, and their tissue's
goal input. Constant noise, random damage. All seeds are trained together as one batched ensemble.

Retargeting test: every (tissue, instruction symbol) gets a new random meaning; 8 example instructions.
  ceiling           correct settings given by hand
  discover+search   probe which tissues affect which regions, then ambiguity-aware search using each
                    region's own error (nothing about the layout is given)
  independent       search judging each tissue only by its own region (ignores interactions)
  retrain           retrain all weights on the 8 examples
  prompt-tune       learn continuous goal inputs by gradient, weights frozen
Pre-registered pass criteria (fixed before running):
  P1: discover+search is within 2 points of the ceiling (patterns correct)
  P2: discover+search beats the better of retrain / prompt-tune by >= 30 points

Usage:   python levin2d.py                 # full run (GPU strongly recommended)
         python levin2d.py --quick         # tiny smoke test
"""
import argparse, itertools, json, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--grid", type=int, default=3)          # tissues per side
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--retrain-steps", type=int, default=400)
ap.add_argument("--rule", choices=["boundary", "shift"], default="boundary",
                help="boundary: interior shows own goal + edges mark differing neighbours; "
                     "shift: interior shows the LEFT neighbour's goal (own region reveals nothing about own goal)")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="levin2d_results")
args = ap.parse_args()
if args.quick:
    args.seeds, args.iters, args.retrain_steps = 2, 6, 3

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
R, TS, K, C, HID, M = args.grid, 5, 4, 12, args.hidden, args.seeds
N, H, W = R * R, R * TS, R * TS
NOISE, EDGE, COMMIT, T_TRAIN = 0.02, 1.0, 0.02, 80
MAX_COMBOS, CHUNK = 256, 64          # search: at most 4 unsettled tissues per region; batch 64 combos at a time

# ---------------------------------------------------------------- world
tissue = torch.arange(N).reshape(R, R).repeat_interleave(TS, 0).repeat_interleave(TS, 1)        # (H,W)
IDCH = F.one_hot(tissue, N).permute(2, 0, 1).float().to(DEV)                                      # (N,H,W)
MOT = torch.zeros(K, TS, TS)
MOT[0, 2, 1:4] = 1; MOT[1, 1:4, 2] = 1
MOT[2, 1, 1] = MOT[2, 1, 3] = MOT[2, 3, 1] = MOT[2, 3, 3] = 1
MOT[3, 1:4, 1:4] = 1; MOT[3, 2, 2] = 0
MOT = MOT.to(DEV)
NBR = torch.full((N, 4), -1, dtype=torch.long)                                                   # left,right,up,down
for n in range(N):
    y, x = divmod(n, R)
    if x > 0: NBR[n, 0] = n - 1
    if x < R - 1: NBR[n, 1] = n + 1
    if y > 0: NBR[n, 2] = n - R
    if y < R - 1: NBR[n, 3] = n + R
NBR = NBR.to(DEV)


def target(g):
    """goals (..., N) -> target image (..., H, W)"""
    nb = g[..., NBR.clamp(min=0)]                                                                # (...,N,4)
    if args.rule == "shift":                                    # interior = left neighbour's motif (none -> own)
        src_g = torch.where(NBR[:, 0] >= 0, nb[..., 0], g)
        loc = MOT[src_g].clone()
        sh = loc.shape[:-3]
        return loc.reshape(*sh, R, R, TS, TS).transpose(-3, -2).reshape(*sh, H, W)
    loc = MOT[g].clone()                                                                         # (...,N,TS,TS)
    diff = (nb != g[..., None]) & (NBR >= 0)
    loc[..., 1:4, 0] = diff[..., 0, None].float(); loc[..., 1:4, TS - 1] = diff[..., 1, None].float()
    loc[..., 0, 1:4] = diff[..., 2, None].float(); loc[..., TS - 1, 1:4] = diff[..., 3, None].float()
    sh = loc.shape[:-3]
    return loc.reshape(*sh, R, R, TS, TS).transpose(-3, -2).reshape(*sh, H, W)


def region_err(out, desired):
    """(..., H, W) -> per-tissue MSE (..., N)"""
    d = (out - desired) ** 2
    sh = d.shape[:-2]
    return d.reshape(*sh, R, TS, R, TS).transpose(-3, -2).reshape(*sh, N, TS * TS).mean(-1)


# ---------------------------------------------------------------- model (ensemble of M seeds)
KER = torch.stack([torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]]),
                   torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8,
                   torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]) / 8,
                   torch.tensor([[1., 2, 1], [2, -12, 2], [1, 2, 1]]) / 16])
CH = C + N
KERN = KER.repeat(CH, 1, 1).unsqueeze(1).to(DEV)                                                 # (4CH,1,3,3)
P = 4 * CH + K


def init_params(Mn, seed):
    g = torch.Generator().manual_seed(seed)
    p = {"W1": torch.randn(Mn, HID, P, generator=g) / P ** 0.5, "b1": torch.zeros(Mn, HID),
         "W2": torch.zeros(Mn, C, HID), "b2": torch.zeros(Mn, C)}
    return {k: v.to(DEV).requires_grad_(True) for k, v in p.items()}


def step(p, s, u):
    """s (M,B,C,H,W); u (M,B,K,H,W) goal input per cell"""
    Mn, B = s.shape[:2]
    full = torch.cat([s, IDCH.expand(Mn, B, N, H, W)], 2).reshape(Mn * B, CH, H, W)
    per = F.conv2d(F.pad(full, (1, 1, 1, 1), value=EDGE), KERN, groups=CH).reshape(Mn, B, 4 * CH, H, W)
    x = torch.cat([per, u], 2)
    h = F.relu(torch.einsum("mhp,mbpyx->mbhyx", p["W1"], x) + p["b1"][:, None, :, None, None])
    d = torch.einsum("mch,mbhyx->mbcyx", p["W2"], h) + p["b2"][:, None, :, None, None]
    return (s + d + NOISE * torch.randn_like(s)).clamp(-2, 2)


def goal_field(onehot):            # (M,B,N,K) -> (M,B,K,H,W)
    return onehot[:, :, tissue.to(DEV)].permute(0, 1, 4, 2, 3)


def damage(s, mask):
    Mn, B = s.shape[:2]
    size = torch.randint(2, 6, (Mn, B), device=DEV)
    y0 = torch.randint(0, H - 2, (Mn, B), device=DEV); x0 = torch.randint(0, W - 2, (Mn, B), device=DEV)
    ys = torch.arange(H, device=DEV); xs = torch.arange(W, device=DEV)
    m = ((ys[None, None, :, None] >= y0[..., None, None]) & (ys[None, None, :, None] < (y0 + size)[..., None, None]) &
         (xs[None, None, None, :] >= x0[..., None, None]) & (xs[None, None, None, :] < (x0 + size)[..., None, None]))
    m = m & mask[..., None, None]
    return s * (~m)[:, :, None].float()


def rollout(p, onehot_goals, T, desired=None, dmg_prob=0.0, grad=False, dense_from=30):
    Mn, B = onehot_goals.shape[:2]
    s = torch.zeros(Mn, B, C, H, W, device=DEV); u = goal_field(onehot_goals)
    t_d = int(np.random.randint(T // 2, T - 15)) if dmg_prob > 0 else -1
    do = torch.rand(Mn, B, device=DEV) < dmg_prob
    loss, n = 0.0, 0
    with torch.set_grad_enabled(grad):
        for t in range(T):
            if t == t_d: s = damage(s, do)
            s = step(p, s, u)
            if desired is not None and t >= dense_from and t % 5 == 0:
                loss = loss + ((s[:, :, 0] - desired) ** 2).mean((-1, -2)).mean(1).sum(); n += 1
    return s, (loss / max(n, 1) if desired is not None else None)


def train(p, iters, B=16):
    opt = torch.optim.Adam(p.values(), lr=3e-3); t0 = time.time()
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.4), int(iters * 0.7)], 0.3)
    for it in range(iters):
        g = torch.randint(0, K, (M, B, N), device=DEV)
        _, loss = rollout(p, F.one_hot(g, K).float(), T_TRAIN, target(g), dmg_prob=0.6, grad=True)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(p.values(), M); opt.step(); sched.step()
        if it % 100 == 0 or it == iters - 1:
            print(f"  train it {it:5d}  loss/model {loss.item() / M:.4f}  ({time.time() - t0:.0f}s)", flush=True)


def frozen(p):
    return {k: v.detach() for k, v in p.items()}


def single(p, k):
    return {kk: v[k:k + 1] for kk, v in p.items()}


@torch.no_grad()
def settle_time(p):
    g = torch.randint(0, K, (M, 32, N), device=DEV); tgt = target(g)
    s = torch.zeros(M, 32, C, H, W, device=DEV); u = goal_field(F.one_hot(g, K).float()); errs = []
    for t in range(2 * T_TRAIN):
        s = step(p, s, u); errs.append(((s[:, :, 0].clamp(0, 1) - tgt) ** 2).mean().item())
    final = np.mean(errs[-10:])
    for t, e in enumerate(errs):
        if e <= final * 1.1 + 1e-3: return max(30, t + 1)
    return 2 * T_TRAIN


def apply_meaning(table, sym):       # table (M,N,K), sym (M,B,N) -> goals (M,B,N)
    return torch.gather(table[:, None].expand(-1, sym.shape[1], -1, -1), 3, sym[..., None]).squeeze(-1)


@torch.no_grad()
def evaluate(p, inp_fn, meaning, n=64, seed=11):
    torch.manual_seed(seed)
    sym = torch.randint(0, K, (M, n, N), device=DEV); tgt = target(apply_meaning(meaning, sym))
    s = torch.zeros(M, n, C, H, W, device=DEV); u = goal_field(inp_fn(sym))
    ok = lambda s: (region_err(s[:, :, 0].clamp(0, 1), tgt) < COMMIT).all(-1).float().mean(1)
    for t in range(T_TRAIN): s = step(p, s, u)
    a = ok(s); s = damage(s, torch.ones(M, n, dtype=torch.bool, device=DEV))
    for t in range(T_TRAIN): s = step(p, s, u)
    return {"patterns": a.tolist(), "after_damage": ok(s).tolist(), "example": s[0, 0, 0].clamp(0, 1).cpu().tolist()}


# ---------------------------------------------------------------- search methods
@torch.no_grad()
def probe(p, T, thr=0.15):
    """Base run (all goals 0) + one run per tissue with its goal switched; which regions change?"""
    cfg = torch.zeros(N + 1, N, dtype=torch.long, device=DEV)
    for n in range(N): cfg[n + 1, n] = 1
    s, _ = rollout(p, F.one_hot(cfg, K).float()[None].expand(M, -1, -1, -1), T)
    out = s[:, :, 0].clamp(0, 1)                                                                # (M,N+1,H,W)
    d = region_err(out[:, 1:], out[:, :1].expand(-1, N, -1, -1)).sqrt()                        # (M, tissue, region)
    return d > thr, N + 1


@torch.no_grad()
def discover_search(p, ex_sym, desired, A, T, tol=0.02):
    table = torch.zeros(M, N, K, dtype=torch.long, device=DEV); rollouts = 0
    B = ex_sym.shape[1]
    for k in range(M):
        pk = single(p, k); settled = set(); tried = set()
        S = [set(torch.nonzero(A[k, :, r]).flatten().tolist()) for r in range(N)]
        while len(settled) < N:
            cand = [(len(S[r] - settled), r) for r in range(N) if S[r] - settled and (r, frozenset(settled)) not in tried]
            if not cand: break
            _, r = min(cand); tried.add((r, frozenset(settled))); U = sorted(S[r] - settled)
            if K ** len(U) > MAX_COMBOS: continue                     # too many unknowns: try another region
            combos = torch.tensor(list(itertools.product(range(K), repeat=len(U))), device=DEV)     # (Cc, |U|)
            Cc = len(combos)
            tab = table[k][None].repeat(Cc, 1, 1)                                                     # (Cc,N,K)
            for i, u in enumerate(U): tab[:, u, :] = combos[:, i, None]
            goals = torch.gather(tab[:, None].expand(-1, B, -1, -1), 3,
                                 ex_sym[k][None, :, :, None].expand(Cc, -1, -1, 1)).squeeze(-1)      # (Cc,B,N)
            Es = []
            for c0 in range(0, Cc, CHUNK):                                # chunked to bound memory
                gch = goals[c0:c0 + CHUNK]; cc = len(gch)
                s, _ = rollout(pk, F.one_hot(gch.reshape(1, cc * B, N), K).float(), T)
                Es.append(region_err(s[0, :, 0].clamp(0, 1).reshape(cc, B, H, W), desired[k][None])[..., r])
            E = torch.cat(Es); rollouts += Cc                                                         # (Cc,B)
            near = E <= E.min(0).values + tol
            for i, u in enumerate(U):
                ok, choice = True, {}
                for b in range(B):
                    vals = set(combos[near[:, b], i].tolist())
                    if len(vals) != 1: ok = False; break
                    sym = int(ex_sym[k, b, u]); v = vals.pop()
                    if choice.setdefault(sym, v) != v: ok = False; break
                if ok:
                    for sym, v in choice.items(): table[k, u, sym] = v
                    settled.add(u)
    return table, rollouts / M


@torch.no_grad()
def independent_search(p, ex_sym, desired, T):
    B = ex_sym.shape[1]
    cfg = torch.arange(K, device=DEV)[:, None, None].expand(K, B, N)                              # all tissues = g
    s, _ = rollout(p, F.one_hot(cfg.reshape(1, K * B, N), K).float().expand(M, -1, -1, -1), T)
    E = region_err(s[:, :, 0].clamp(0, 1).reshape(M, K, B, H, W), desired[:, None])            # (M,K,B,N)
    table = torch.zeros(M, N, K, dtype=torch.long, device=DEV)
    for k in range(M):
        for n in range(N):
            for sym in range(K):
                sel = ex_sym[k, :, n] == sym
                if sel.any(): table[k, n, sym] = E[k, :, sel, n].mean(-1).argmin()
    return table, K


def retrain(p, ex_sym, new, steps):
    q = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
    opt = torch.optim.Adam(q.values(), lr=3e-3); b = ex_sym.repeat(1, 2, 1)
    for _ in range(steps):
        _, loss = rollout(q, F.one_hot(b, K).float(), T_TRAIN, target(apply_meaning(new, b)), dmg_prob=0.6, grad=True)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.values(), M); opt.step()
    return frozen(q)


def prompt_tune(p, ex_sym, new, steps, lr=0.3):
    soft = F.one_hot(torch.arange(K, device=DEV), K).float()[None, None].expand(M, N, K, K).clone().requires_grad_(True)
    opt = torch.optim.Adam([soft], lr=lr); b = ex_sym.repeat(1, 2, 1)
    pick = lambda sym: torch.gather(soft[:, None].expand(-1, sym.shape[1], -1, -1, -1), 3,
                                    sym[..., None, None].expand(-1, -1, -1, 1, K)).squeeze(3)
    for _ in range(steps):
        _, loss = rollout(p, pick(b), T_TRAIN, target(apply_meaning(new, b)), dmg_prob=0.6, grad=True)
        opt.zero_grad(); loss.backward(); opt.step()
    sd = soft.detach()
    return lambda sym: torch.gather(sd[:, None].expand(-1, sym.shape[1], -1, -1, -1), 3,
                                    sym[..., None, None].expand(-1, -1, -1, 1, K)).squeeze(3)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    T0 = time.time(); log = {"device": str(DEV), "config": vars(args)}
    print(f"device: {DEV} | grid {R}x{R} tissues of {TS}x{TS} cells | {K} options | {M} seeds", flush=True)
    print("1/4 training the competent parts (input = goal)", flush=True)
    p = init_params(M, 0); train(p, args.iters); p = frozen(p)
    T = settle_time(p); log["settle_steps"] = T
    print(f"  patterns settle in ~{T} steps; trial runs use {T} steps ({time.time() - T0:.0f}s so far)", flush=True)

    torch.manual_seed(5)
    ident = torch.arange(K, device=DEV)[None, None].expand(M, N, K)
    new = torch.randint(0, K, (M, N, K), device=DEV)
    ex_sym = torch.randint(0, K, (M, 8, N), device=DEV)
    for j in range(K): ex_sym[:, j] = (j + torch.arange(N, device=DEV)) % K                       # cover every case
    desired = target(apply_meaning(new, ex_sym))
    onehot = lambda t: (lambda sym: F.one_hot(apply_meaning(t, sym), K).float())
    res = {}
    print("2/4 original task + ceiling", flush=True)
    res["original task"] = evaluate(p, onehot(ident), ident)
    res["ceiling (correct settings by hand)"] = evaluate(p, onehot(new), new)

    print("3/4 steering by search", flush=True)
    t = time.time(); A, pc = probe(p, T)
    true_A = torch.zeros(N, N, dtype=torch.bool, device=DEV)
    for n in range(N):
        if args.rule == "boundary":
            true_A[n, n] = True
            for nb in NBR[n].tolist():
                if nb >= 0: true_A[n, nb] = True
        else:                                                   # tissue n drives its right neighbour's interior
            if NBR[n, 1] >= 0: true_A[n, NBR[n, 1]] = True
            if NBR[n, 0] < 0: true_A[n, n] = True
    log["map_exact_fraction"] = (A == true_A).flatten(1).all(1).float().mean().item()
    tab, sc = discover_search(p, ex_sym, desired, A, T)
    res["discover + search"] = evaluate(p, onehot(tab), new)
    res["discover + search"].update(cost=f"{pc} probe + {sc:.0f} search trial runs, {time.time() - t:.0f}s",
                                    settings_correct=(tab == new).float().mean().item())
    t = time.time(); tab2, c2 = independent_search(p, ex_sym, desired, T)
    res["independent search (ignores interactions)"] = evaluate(p, onehot(tab2), new)
    res["independent search (ignores interactions)"].update(cost=f"{c2} trial runs, {time.time() - t:.0f}s",
                                                            settings_correct=(tab2 == new).float().mean().item())

    print("4/4 baselines: retraining and prompt tuning", flush=True)
    t = time.time(); q = retrain(p, ex_sym, new, args.retrain_steps)
    res["retrain all weights"] = evaluate(q, lambda sym: F.one_hot(sym, K).float(), new)
    res["retrain all weights"]["cost"] = f"{args.retrain_steps} gradient steps, {time.time() - t:.0f}s"
    t = time.time(); inp = prompt_tune(p, ex_sym, new, args.retrain_steps)
    res["prompt-tune goal inputs"] = evaluate(p, inp, new)
    res["prompt-tune goal inputs"]["cost"] = f"{args.retrain_steps} gradient steps, {time.time() - t:.0f}s"

    pct = lambda r, k: 100 * float(np.mean(r[k]))
    ds, ceil = pct(res["discover + search"], "patterns"), pct(res["ceiling (correct settings by hand)"], "patterns")
    best_base = max(pct(res["retrain all weights"], "patterns"), pct(res["prompt-tune goal inputs"], "patterns"))
    VALID = ceil >= 80                                            # test only counts if the parts are competent
    P1, P2 = ds >= ceil - 2, ds >= best_base + 30
    lines = [f"# 2D test ({args.rule} rule): {R}x{R} tissues of {TS}x{TS} cells, {K} options, {M} seeds ({DEV})", "",
             "| Method | Patterns correct | After damage | Settings correct | Cost |", "|---|---|---|---|---|"]
    for name, r in res.items():
        sc_ = f"{100 * r['settings_correct']:.0f}%" if "settings_correct" in r else "-"
        lines.append(f"| {name} | {pct(r, 'patterns'):.0f}% | {pct(r, 'after_damage'):.0f}% | {sc_} | {r.get('cost', '-')} |")
    lines += ["", f"Influence map discovered exactly for {100 * log['map_exact_fraction']:.0f}% of seeds.",
              f"Validity check (ceiling >= 80%, i.e. the parts are competent): {'OK' if VALID else 'NOT MET - results not meaningful; train longer'} ({ceil:.0f}%)",
              f"Pre-registered P1 (within 2 points of ceiling): {'PASS' if P1 else 'FAIL'} ({ds:.0f}% vs {ceil:.0f}%)",
              f"Pre-registered P2 (>= 30 points above best baseline): {'PASS' if P2 else 'FAIL'} ({ds:.0f}% vs {best_base:.0f}%)",
              f"Total time {time.time() - T0:.0f}s"]
    print("\n".join(lines), flush=True)
    import os; os.makedirs(args.out, exist_ok=True)
    open(f"{args.out}/summary.md", "w").write("\n".join(lines) + "\n")
    json.dump({"log": log, "results": res}, open(f"{args.out}/results.json", "w"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = ["ceiling (correct settings by hand)", "discover + search", "retrain all weights", "prompt-tune goal inputs"]
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
        for a, nm in zip(ax, names):
            a.imshow(np.array(res[nm]["example"]), cmap="gray_r", vmin=0, vmax=1)
            for v in range(1, R): a.axhline(v * TS - .5, color="#2a78d6", lw=.8); a.axvline(v * TS - .5, color="#2a78d6", lw=.8)
            a.set_title(nm.split(" (")[0], fontsize=9); a.set_xticks([]); a.set_yticks([])
        fig.suptitle("One example under the NEW meaning, after damage and recovery (seed 0)", fontsize=10)
        fig.savefig(f"{args.out}/examples.png", dpi=140, bbox_inches="tight")
    except Exception as e:
        print("figure skipped:", e)
