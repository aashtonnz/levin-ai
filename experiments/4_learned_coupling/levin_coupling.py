"""
Idea 4: do PARTS EMERGE from learnable cell-cell coupling (gap-junction-like), instead of being hand-designed?

Hidden structure: a 15x15 grid made of 3x3 tissues (5x5 cells), each with 4 options (boundary rule: interior motif +
edges lit where the neighbouring tissue differs). The model is NOT told about tissues, their boundaries or options.

Model:
  * Cells (NCA) see their own state and each neighbour's state multiplied by a LEARNABLE COUPLING c in (0,1) for that
    edge. Coupling has a small cost, so links are only kept where useful.
  * Goal panel: a coarse 5x5 grid of discrete codes (8 options each), deliberately MISALIGNED with the 3x3 tissues.
    A small encoder learns which codes to use for each example (straight-through discrete codes).
  * Control condition: identical model with all couplings fixed at 1.

Checks:
  1. Reconstruction: held-out patterns correct (every tissue right), also after damage.
  2. Emergent parts: are couplings weaker on the true (hidden) tissue boundaries than inside tissues?
     Do strongly coupled groups of cells match the true tissues (adjusted Rand index; 1 = perfect, 0 = chance)?
  3. Steering: reach NEW target patterns by searching the panel with each panel cell's own local error,
     vs. gradient-tuning continuous codes; the encoder's one-shot guess is shown for reference.

Usage: python levin_coupling.py [--quick] [--seeds 8] [--iters 2000]
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--panel", type=int, default=5)          # panel grid side (5 = misaligned with the 3x3 tissues)
ap.add_argument("--codes", type=int, default=8)
ap.add_argument("--coupling-cost", type=float, default=0.05)
ap.add_argument("--conditions", nargs="+", default=["learned", "fixed"])
ap.add_argument("--rule", choices=["independent", "boundary"], default="independent",
                help="independent: each tissue just shows its own motif (cross-boundary chatter is only a nuisance); "
                     "boundary: tissues also mark edges where neighbours differ (needs cross-boundary communication)")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="coupling_results")
args = ap.parse_args()
if args.quick: args.seeds, args.iters = 2, 5

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R, TS, K = 3, 5, 4
N, H, W = R * R, R * TS, R * TS
C, HID, M, PN, V = 12, 64, args.seeds, args.panel, args.codes
NOISE, EDGE, COMMIT, T = 0.02, 1.0, 0.02, 80

# ------------------------------------------------------------------ hidden world (boundary rule)
tissue = torch.arange(N).reshape(R, R).repeat_interleave(TS, 0).repeat_interleave(TS, 1).to(DEV)
MOT = torch.zeros(K, TS, TS)
MOT[0, 2, 1:4] = 1; MOT[1, 1:4, 2] = 1
MOT[2, 1, 1] = MOT[2, 1, 3] = MOT[2, 3, 1] = MOT[2, 3, 3] = 1
MOT[3, 1:4, 1:4] = 1; MOT[3, 2, 2] = 0
MOT = MOT.to(DEV)
NBR = torch.full((N, 4), -1, dtype=torch.long)
for n in range(N):
    y, x = divmod(n, R)
    if x > 0: NBR[n, 0] = n - 1
    if x < R - 1: NBR[n, 1] = n + 1
    if y > 0: NBR[n, 2] = n - R
    if y < R - 1: NBR[n, 3] = n + R
NBR = NBR.to(DEV)


def target(g):
    loc = MOT[g].clone(); nb = g[..., NBR.clamp(min=0)]
    if args.rule == "independent":
        sh = loc.shape[:-3]
        return loc.reshape(*sh, R, R, TS, TS).transpose(-3, -2).reshape(*sh, H, W)
    diff = (nb != g[..., None]) & (NBR >= 0)
    loc[..., 1:4, 0] = diff[..., 0, None].float(); loc[..., 1:4, TS - 1] = diff[..., 1, None].float()
    loc[..., 0, 1:4] = diff[..., 2, None].float(); loc[..., TS - 1, 1:4] = diff[..., 3, None].float()
    sh = loc.shape[:-3]
    return loc.reshape(*sh, R, R, TS, TS).transpose(-3, -2).reshape(*sh, H, W)


def tissue_err(out, tgt):
    d = (out - tgt) ** 2; sh = d.shape[:-2]
    return d.reshape(*sh, R, TS, R, TS).transpose(-3, -2).reshape(*sh, N, TS * TS).mean(-1)


# ------------------------------------------------------------------ model (ensemble of M seeds)
PIN = 5 * C + V + 4                                       # own + 4 gated neighbours + code + 4 border flags
cell_of_panel = (torch.arange(H, device=DEV) * PN // H)  # which panel row/col a cell belongs to
PANEL_IDX = (cell_of_panel[:, None] * PN + cell_of_panel[None, :])          # (H,W) -> panel cell index
BORDER = torch.zeros(4, H, W, device=DEV)
BORDER[0, :, 0] = 1; BORDER[1, :, -1] = 1; BORDER[2, 0, :] = 1; BORDER[3, -1, :] = 1


def init(seed, learn_coupling):
    g = torch.Generator().manual_seed(seed)
    p = {"W1": torch.randn(M, HID, PIN, generator=g) / PIN ** 0.5, "b1": torch.zeros(M, HID),
         "W2": torch.zeros(M, C, HID), "b2": torch.zeros(M, C),
         "e1": torch.randn(M * 32, 1, 3, 3, generator=g) / 3, "e2": torch.randn(M * 32, 32, 3, 3, generator=g) / 17,
         "e3": torch.randn(M * V, 32, 1, 1, generator=g) / 6}
    if learn_coupling:
        p["ch"] = torch.full((M, H, W - 1), 2.0); p["cv"] = torch.full((M, H - 1, W), 2.0)
    return {k: v.to(DEV).requires_grad_(True) for k, v in p.items()}


def couplings(p):
    """-> (M,4,H,W) coupling to left,right,up,down neighbour (borders 0: the border flag carries that info)"""
    Mn = p["W1"].shape[0]
    if "ch" not in p:
        c = torch.ones(Mn, 4, H, W, device=DEV)
    else:
        ch, cv = torch.sigmoid(p["ch"]), torch.sigmoid(p["cv"]); c = torch.zeros(Mn, 4, H, W, device=DEV)
        c[:, 0, :, 1:] = ch; c[:, 1, :, :-1] = ch; c[:, 2, 1:, :] = cv; c[:, 3, :-1, :] = cv
    c = c * (1 - BORDER)
    return c


def encode(p, tgt, hard=True, tau=1.0, noise=True):
    """tgt (M,B,H,W) -> codes (M,B,PN*PN,V) one-hot (straight-through)"""
    Mn, B = tgt.shape[:2]
    x = tgt.permute(1, 0, 2, 3)                                                   # (B,M,H,W)
    h = F.relu(F.conv2d(x, p["e1"], padding=1, groups=Mn))
    h = F.relu(F.conv2d(h, p["e2"], padding=1, groups=Mn))
    h = F.adaptive_avg_pool2d(h, PN)
    logits = F.conv2d(h, p["e3"], groups=Mn).reshape(B, Mn, V, PN * PN).permute(1, 0, 3, 2)   # (M,B,P,V)
    if noise: logits = logits - torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1)))
    soft = F.softmax(logits / tau, -1)
    if not hard: return soft
    return soft + (F.one_hot(soft.argmax(-1), V).float() - soft).detach()


def code_field(codes):          # (M,B,P,V) -> (M,B,V,H,W)
    return codes[:, :, PANEL_IDX].permute(0, 1, 4, 2, 3)


def step(p, s, u, c):
    Mn, B = s.shape[:2]
    sl = F.pad(s, (1, 0))[..., :, :-1]; sr = F.pad(s, (0, 1))[..., :, 1:]
    su = F.pad(s, (0, 0, 1, 0))[..., :-1, :]; sd = F.pad(s, (0, 0, 0, 1))[..., 1:, :]
    cc = c[:, None]                                                               # (M,1,4,H,W)
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


def rollout(p, codes, steps, desired=None, dmg_prob=0.0, grad=False):
    Mn, B = codes.shape[:2]
    s = torch.zeros(Mn, B, C, H, W, device=DEV); u = code_field(codes); c = couplings(p)
    t_d = int(np.random.randint(steps // 2, steps - 15)) if dmg_prob > 0 else -1
    do = torch.rand(Mn, B, device=DEV) < dmg_prob; loss, n = 0.0, 0
    with torch.set_grad_enabled(grad):
        for t in range(steps):
            if t == t_d: s = damage(s, do)
            s = step(p, s, u, c)
            if desired is not None and t >= 30 and t % 5 == 0:
                loss = loss + ((s[:, :, 0] - desired) ** 2).mean((-1, -2)).mean(1).sum(); n += 1
    return s, (loss / max(n, 1) if desired is not None else None)


def train(p, iters, B=16):
    opt = torch.optim.Adam(p.values(), lr=3e-3); t0 = time.time()
    for it in range(iters):
        g = torch.randint(0, K, (M, B, N), device=DEV); tgt = target(g)
        codes = encode(p, tgt, tau=max(0.5, 1.0 - it / iters))
        _, loss = rollout(p, codes, T, tgt, dmg_prob=0.6, grad=True)
        cost = args.coupling_cost * couplings(p).mean((1, 2, 3)).sum() if "ch" in p else 0.0
        opt.zero_grad(); (loss + cost).backward(); torch.nn.utils.clip_grad_norm_(p.values(), M); opt.step()
        if it % 100 == 0 or it == iters - 1:
            cm = couplings(p).mean().item()
            print(f"  it {it:5d} recon/model {loss.item() / M:.4f} mean coupling {cm:.2f} ({time.time() - t0:.0f}s)", flush=True)


def frozen(p): return {k: v.detach() for k, v in p.items()}


@torch.no_grad()
def check_codes(p, codes, tgt):
    """fraction of examples with every tissue correct; then after damage"""
    Mn, B = codes.shape[:2]
    s = torch.zeros(Mn, B, C, H, W, device=DEV); u = code_field(codes); c = couplings(p)
    ok = lambda s: (tissue_err(s[:, :, 0].clamp(0, 1), tgt) < COMMIT).all(-1).float().mean(1)
    for t in range(T): s = step(p, s, u, c)
    a = ok(s); s = damage(s, torch.ones(Mn, B, dtype=torch.bool, device=DEV))
    for t in range(T): s = step(p, s, u, c)
    return a, ok(s), s[:, :, 0].clamp(0, 1)


def ari(a, b):
    """adjusted Rand index between two labelings (numpy int arrays)"""
    from math import comb
    ct = {}
    for x, y in zip(a, b): ct[(x, y)] = ct.get((x, y), 0) + 1
    sa, sb = {}, {}
    for (x, y), v in ct.items(): sa[x] = sa.get(x, 0) + v; sb[y] = sb.get(y, 0) + v
    idx = sum(comb(v, 2) for v in ct.values()); ia = sum(comb(v, 2) for v in sa.values()); ib = sum(comb(v, 2) for v in sb.values())
    tot = comb(len(a), 2); exp = ia * ib / tot; mx = (ia + ib) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def components(cmap, thr=0.5):
    """cells joined where coupling > thr -> labels (H*W)"""
    ch, cv = cmap
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
    return np.array([f(i) for i in range(H * W)])


def part_metrics(p):
    if "ch" not in p: return None
    ch, cv = torch.sigmoid(p["ch"]).cpu().numpy(), torch.sigmoid(p["cv"]).cpu().numpy()
    xs = np.arange(W - 1); ys = np.arange(H - 1)
    bh = ((xs + 1) % TS == 0)[None, :].repeat(H, 0); bv = ((ys + 1) % TS == 0)[:, None].repeat(W, 1)
    true = tissue.cpu().numpy().flatten(); out = []
    for k in range(M):
        boundary = np.concatenate([ch[k][bh], cv[k][bv]]).mean(); interior = np.concatenate([ch[k][~bh], cv[k][~bv]]).mean()
        out.append({"boundary": float(boundary), "interior": float(interior),
                    "ari": float(ari(components((ch[k], cv[k])), true)),
                    "n_parts": int(len(set(components((ch[k], cv[k])).tolist())))})
    return out


def region_err_panel(out, tgt):
    """per panel-cell MSE: (..., H, W) -> (..., PN*PN)"""
    d = (out - tgt) ** 2; idx = PANEL_IDX.flatten()
    flat = d.flatten(-2); res = torch.zeros(*flat.shape[:-1], PN * PN, device=DEV)
    res.index_add_(-1, idx, flat) if flat.dim() == 1 else res.scatter_add_(-1, idx.expand_as(flat), flat)
    return res / torch.bincount(idx, minlength=PN * PN).float()


@torch.no_grad()
def local_search(p, tgt, sweeps=4):
    """Coordinate search over the panel: each panel cell tries every code, judged by ITS OWN region's error."""
    Mn, B = tgt.shape[:2]; P = PN * PN
    codes = torch.zeros(Mn, B, P, dtype=torch.long, device=DEV); runs = 0
    for _ in range(sweeps):
        for parity in range(4):                                  # checkerboard-like groups: 4 colours on the panel
            cells = [i for i in range(P) if ((i // PN) % 2) * 2 + (i % PN) % 2 == parity]
            errs = []
            for v in range(V):
                trial = codes.clone(); trial[:, :, cells] = v
                s, _ = rollout(p, F.one_hot(trial, V).float(), T); runs += 1
                errs.append(region_err_panel(s[:, :, 0].clamp(0, 1), tgt))
            E = torch.stack(errs)                                 # (V,M,B,P)
            best = E.argmin(0)
            codes[:, :, cells] = best[:, :, cells]
    return codes, runs


def grad_codes(p, tgt, steps=200, lr=0.3):
    Mn, B = tgt.shape[:2]
    z = torch.zeros(Mn, B, PN * PN, V, device=DEV, requires_grad=True); opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        _, loss = rollout(p, F.softmax(z, -1), T, tgt, grad=True)
        opt.zero_grad(); loss.backward(); opt.step()
    return F.one_hot(z.detach().argmax(-1), V).float()


if __name__ == "__main__":
    T0 = time.time(); os.makedirs(args.out, exist_ok=True); report = {}
    print(f"device {DEV} | hidden 3x3 tissues | panel {PN}x{PN} codes x {V} | {M} seeds", flush=True)
    lines = [f"# Do parts emerge from learnable coupling? ({args.rule} rule, coupling cost {args.coupling_cost}, {M} seeds, {DEV})", ""]
    for cond in args.conditions:
        print(f"== condition: {cond} coupling", flush=True)
        torch.manual_seed(0); np.random.seed(0)
        p = init(0, cond == "learned"); train(p, args.iters); p = frozen(p)
        torch.manual_seed(7)
        g = torch.randint(0, K, (M, 32, N), device=DEV); tgt = target(g)
        r = {}
        a, d, _ = check_codes(p, encode(p, tgt, noise=False), tgt); r["encoder (one-shot guess)"] = (a, d, "1 encoder pass")
        t = time.time(); codes, runs = local_search(p, tgt); a, d, ex = check_codes(p, F.one_hot(codes, V).float(), tgt)
        r["local search on panel"] = (a, d, f"{runs} batched trial runs, {time.time() - t:.0f}s")
        t = time.time(); gc = grad_codes(p, tgt, steps=20 if args.quick else 200); a, d, _ = check_codes(p, gc, tgt)
        r["gradient-tuned codes"] = (a, d, f"{20 if args.quick else 200} gradient steps, {time.time() - t:.0f}s")
        pm = part_metrics(p)
        lines += [f"## {cond} coupling", "", "| Steering method | New patterns correct | After damage | Cost |", "|---|---|---|---|"]
        for k_, (a, d, cst) in r.items():
            lines.append(f"| {k_} | {100 * a.mean():.0f}% | {100 * d.mean():.0f}% | {cst} |")
        if pm:
            lines += ["", f"Mean coupling on TRUE (hidden) tissue boundaries: {np.mean([x['boundary'] for x in pm]):.2f}; "
                          f"inside tissues: {np.mean([x['interior'] for x in pm]):.2f}",
                      f"Strongly coupled groups vs true tissues (adjusted Rand index, 1 = perfect, 0 = chance): "
                      f"{np.mean([x['ari'] for x in pm]):.2f} (per seed {[round(x['ari'], 2) for x in pm]}); "
                      f"groups found per seed {[x['n_parts'] for x in pm]}"]
            ch = torch.sigmoid(p["ch"][0]).cpu().numpy(); cv = torch.sigmoid(p["cv"][0]).cpu().numpy()
            report[cond] = {"parts": pm, "ch": ch.tolist(), "cv": cv.tolist()}
        lines.append("")
        report.setdefault(cond, {})["steering"] = {k_: [v[0].tolist(), v[1].tolist(), v[2]] for k_, v in r.items()}
        print("\n".join(lines), flush=True)
    lines.append(f"Total time {time.time() - T0:.0f}s")
    open(f"{args.out}/summary.md", "w").write("\n".join(lines) + "\n")
    json.dump(report, open(f"{args.out}/results.json", "w"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        if "learned" in report and "ch" in report["learned"]:
            ch = np.array(report["learned"]["ch"]); cv = np.array(report["learned"]["cv"])
            fig, ax = plt.subplots(figsize=(5, 5))
            for y in range(H):
                for x in range(W - 1): ax.plot([x, x + 1], [y, y], color="#1f1f1e", lw=4 * ch[y, x], alpha=0.9)
            for y in range(H - 1):
                for x in range(W): ax.plot([x, x], [y, y + 1], color="#1f1f1e", lw=4 * cv[y, x], alpha=0.9)
            for b in range(1, R):
                ax.axvline(b * TS - .5, color="#eb6834", ls="--", lw=1); ax.axhline(b * TS - .5, color="#eb6834", ls="--", lw=1)
            ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Learned coupling (line width), seed 0\norange dashes = hidden true tissue boundaries", fontsize=9)
            fig.savefig(f"{args.out}/coupling_map.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        print("figure skipped:", e)
