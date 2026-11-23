"""
"Goals at every level" test: does an upper level that SETS GOALS for separately competent lower-level
units generalise and re-target better than end-to-end alternatives?

Task: 32 cells = 4 tissues x 8 cells. A 4-bit instruction says which of two local motifs each tissue shows
(motif 0: cells 1-3 of the tissue on; motif 1: cells 4-6 on). 16 global patterns.
Each cell receives static inputs: the instruction, its tissue id (one-hot), its position within the tissue.

Models (similar parameter counts):
  flat        one MLP: (instruction, tissue id, position) -> on/off
  hier_latent upper h(instruction, tissue id) -> 2-d signal T; lower f(T, position) -> on/off; end-to-end
  hier_goal   same architecture. Lower level f first trained ALONE to reach any local goal T in
              {motif 0, motif 1} (its own competency, its own errors). Then f is frozen and the upper level
              h learns only to set goals (gradient flows through f; no labels for T are ever given).

Test 1: train on 12 of 16 instructions, test on the 4 held-out ones.
Test 2: re-target: instructions get a new meaning (bit order reversed AND motifs swapped). Adapt from only
        4 example instructions for a fixed number of steps; evaluate on all 16. Two-level models adapt only h;
        flat adapts everything (also reported: flat adapting only its last layer).
"""
import json, itertools
import numpy as np
import torch
import torch.nn as nn

CODES = torch.tensor(list(itertools.product([0, 1], repeat=4))).float()          # (16, 4)
POS = torch.arange(8)
MOTIF = torch.stack([((POS >= 1) & (POS <= 3)).float(), ((POS >= 4) & (POS <= 6)).float()])  # (2, 8)


def target(codes, mapping="orig"):
    """codes (N,4) -> (N,4 tissues,8 cells)"""
    if mapping == "orig":
        m = codes.long()
    else:                                   # new meaning: reversed bit order, motifs swapped
        m = 1 - codes.flip(1).long()
    return MOTIF[m]


def cell_inputs(codes):
    """-> instruction (N,4,8,4), tissue one-hot (N,4,8,4), position (N,4,8,1)"""
    N = len(codes)
    instr = codes[:, None, None, :].expand(N, 4, 8, 4)
    tissue = torch.eye(4)[:, None, :].expand(4, 8, 4)[None].expand(N, 4, 8, 4)
    pos = ((POS.float() - 3.5) / 3.5)[None, None, :, None].expand(N, 4, 8, 1)
    return instr, tissue, pos


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Flat(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(9, 48, 1)
    def forward(self, codes):
        i, t, p = cell_inputs(codes)
        return self.net(torch.cat([i, t, p], -1)).squeeze(-1)


class Hier(nn.Module):
    def __init__(self):
        super().__init__(); self.h = mlp(8, 32, 2); self.f = mlp(3, 32, 1)
    def goals(self, codes):
        i, t, _ = cell_inputs(codes)
        return self.h(torch.cat([i[:, :, 0], t[:, :, 0]], -1))          # (N,4,2) one goal per tissue
    def lower(self, T, pos):
        return self.f(torch.cat([T, pos], -1)).squeeze(-1)
    def forward(self, codes):
        T = self.goals(codes)[:, :, None, :].expand(-1, -1, 8, -1)
        _, _, p = cell_inputs(codes)
        return self.lower(T, p)


def bce(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def fit(params, loss_fn, steps, lr=3e-3):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    return l.item()


@torch.no_grad()
def score(model, codes, mapping="orig"):
    pred = (model(codes) > 0).float(); y = target(codes, mapping)
    cell_acc = (pred == y).float().mean().item()
    exact = (pred == y).flatten(1).all(1).float().mean().item()
    return cell_acc, exact


def train_lower_competency(model, steps=1500):
    """Lower level learns to reach ANY local goal it is given (random goals, its own error)."""
    pos = ((POS.float() - 3.5) / 3.5)[:, None]
    def loss():
        g = torch.randint(0, 2, (64,))
        T = torch.eye(2)[g][:, None, :].expand(-1, 8, -1)
        return bce(model.lower(T, pos[None].expand(64, -1, -1)), MOTIF[g])
    fit(model.f.parameters(), loss, steps)


def run(seed, steps=2000, adapt_steps=200):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    perm = rng.permutation(16); train_idx, test_idx = perm[:12], perm[12:]
    adapt_idx = rng.permutation(16)[:4]
    res = {}
    for kind in ["flat", "hier_latent", "hier_goal"]:
        torch.manual_seed(seed)
        m = Flat() if kind == "flat" else Hier()
        tr = CODES[train_idx]
        if kind == "hier_goal":
            train_lower_competency(m)
            for p in m.f.parameters(): p.requires_grad_(False)
            fit(m.h.parameters(), lambda: bce(m(tr), target(tr)), steps)
        else:
            fit(m.parameters(), lambda: bce(m(tr), target(tr)), steps)
        r = {"params": sum(p.numel() for p in m.parameters()),
             "train_exact": score(m, tr)[1], "heldout_cell": score(m, CODES[test_idx])[0],
             "heldout_exact": score(m, CODES[test_idx])[1]}
        # Test 2: re-target from 4 examples
        ad = CODES[adapt_idx]
        variants = [("all", None)] if kind == "flat" else [("upper_only", None)]
        if kind == "flat": variants.append(("last_layer", None))
        state = {k: v.clone() for k, v in m.state_dict().items()}
        for name, _ in variants:
            m.load_state_dict(state)
            for p in m.parameters(): p.requires_grad_(False)
            if kind == "flat":
                ps = list(m.parameters()) if name == "all" else list(m.net[-1].parameters())
            else:
                ps = list(m.h.parameters())
            for p in ps: p.requires_grad_(True)
            curve = []
            opt = torch.optim.Adam(ps, lr=3e-3)
            for s in range(adapt_steps):
                opt.zero_grad(); bce(m(ad), target(ad, "new")).backward(); opt.step()
                if (s + 1) in (25, 50, 100, 200): curve.append(score(m, CODES, "new")[1])
            r[f"retarget_{name}"] = curve
        res[kind] = r
    return res


if __name__ == "__main__":
    out = [run(s) for s in range(10)]
    json.dump(out, open("hier_goals_results.json", "w"))
    def agg(kind, key, idx=None):
        v = np.array([o[kind][key] if idx is None else o[kind][key][idx] for o in out])
        return v.mean(), v.min(), v.max()
    print("params:", {k: out[0][k]["params"] for k in out[0]})
    print("\nTest 1 - held-out instructions (exact pattern correct, mean [min-max] over 10 seeds)")
    for k in ["flat", "hier_latent", "hier_goal"]:
        print(f"  {k:12s} train {agg(k,'train_exact')[0]:.2f}  held-out cells {agg(k,'heldout_cell')[0]:.2f}  "
              "held-out exact %.2f [%.2f-%.2f]" % agg(k, "heldout_exact"))
    print("\nTest 2 - re-target from 4 examples: exact patterns correct on all 16 after 25/50/100/200 steps")
    for k, key in [("flat", "retarget_all"), ("flat", "retarget_last_layer"),
                   ("hier_latent", "retarget_upper_only"), ("hier_goal", "retarget_upper_only")]:
        print(f"  {k:12s} {key:22s}", " ".join("%.2f" % agg(k, key, i)[0] for i in range(4)))
