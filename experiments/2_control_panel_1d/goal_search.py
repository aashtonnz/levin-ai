"""Upper level that PICKS goals instead of learning them by gradient: for each (tissue, bit value) try both
goals the lower level knows, keep whichever gives lower error on the 4 example instructions. No gradients."""
import time, json, torch, numpy as np, torch.nn.functional as F
import hier_dynamic as hd

@torch.no_grad()
def rollout_loss(m, table, bits, motif):
    goal_idx = torch.gather(table[:, None].expand(-1, bits.shape[1], -1, -1), 3, bits[..., None]).squeeze(-1)
    loss, _ = hd.episode(m, lambda: m.cell_input(goal=F.one_hot(goal_idx, 2).float()), motif, dmg=False, grad=False)
    return None, goal_idx

@torch.no_grad()
def per_model_loss(m, table, bits, motif):
    goal_idx = torch.gather(table[:, None].expand(-1, bits.shape[1], -1, -1), 3, bits[..., None]).squeeze(-1)
    u = m.cell_input(goal=F.one_hot(goal_idx, 2).float()); tgt = hd.target(motif)
    s = torch.zeros(m.M, bits.shape[1], hd.C, hd.L)
    for t in range(100): s = m.step(s, u)
    return ((s[:, :, 0] - tgt) ** 2).mean((1, 2))           # (M,)

@torch.no_grad()
def search(m, bits, motif):
    M = m.M; table = torch.randint(0, 2, (M, 4, 2)); n = 0
    for t in range(4):
        for b in range(2):
            losses = []
            for g in range(2):
                trial = table.clone(); trial[:, t, b] = g
                losses.append(per_model_loss(m, trial, bits, motif)); n += 1
            table[:, t, b] = torch.stack(losses).argmin(0)
    return table, n

@torch.no_grad()
def evaluate(m, table, meaning):
    bits = hd.CODES.repeat(4, 1)[None].expand(m.M, -1, -1)
    motif = bits if meaning == "orig" else 1 - bits
    goal_idx = torch.gather(table[:, None].expand(-1, bits.shape[1], -1, -1), 3, bits[..., None]).squeeze(-1)
    u = m.cell_input(goal=F.one_hot(goal_idx, 2).float()); tgt = hd.target(motif)
    def ok(s):
        e = ((s[:, :, 0].clamp(0, 1) - tgt) ** 2).reshape(m.M, -1, 4, 8).mean(-1)
        return (e < hd.COMMIT).all(-1).float().mean(1)
    s = torch.zeros(m.M, bits.shape[1], hd.C, hd.L)
    for t in range(100): s = m.step(s, u)
    a = ok(s); s = hd.damage(s, torch.ones(s.shape[:2], dtype=torch.bool))
    for t in range(100): s = m.step(s, u)
    return a.tolist(), ok(s).tolist()

torch.manual_seed(1)
m = hd.Model("hier_goal", 8, 0); m.load_state_dict(torch.load("hd_hier_goal.pt"))
idx = torch.stack([torch.randperm(16)[:4] for _ in range(8)])          # 4 examples per model
ex = torch.gather(hd.CODES[None].expand(8, -1, -1), 1, idx[..., None].expand(-1, -1, 4)).repeat(1, 4, 1)
out = {}
for meaning in ["orig", "new"]:
    t0 = time.time()
    table, n = search(m, ex, ex if meaning == "orig" else 1 - ex)
    pats, dmg = evaluate(m, table, meaning)
    out[meaning] = {"patterns": pats, "after_damage": dmg, "rollouts": n, "seconds": time.time() - t0}
    print(f"{meaning}: patterns {np.mean(pats):.2f} {[round(x,2) for x in pats]} | after damage {np.mean(dmg):.2f} "
          f"| {n} rollouts, {time.time()-t0:.1f}s")
json.dump(out, open("goal_search_results.json", "w"))
