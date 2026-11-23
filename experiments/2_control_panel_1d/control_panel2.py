"""
Control panel with INTERACTING parts.
Tissue n shows motif (goal_n + goal_{n-1}) mod K   (goal_{-1} = 0), so a tissue's correct setting depends on
its left neighbour, and the lower level must coordinate across tissue boundaries through its dynamics.

  flat   cells receive their tissue's symbol; original meaning goal = symbol; retrains weights to retarget.
  panel  lower level trained alone on random goals (it must render the interacting pattern).
         Retarget by search over the table (tissue, symbol) -> goal:
           "independent": last round's search (each tissue judged alone) - ignores interactions
           "neighbourhood": alternate even/odd tissues; for each candidate goal, judge a tissue by the error
                            in its own region + its right neighbour's region (the regions its goal affects).
                            Cost: 2*K rollouts per sweep, independent of the number of tissues.
New meaning: a random table (tissue, symbol) -> goal, so a correct setting always exists.
"""
import json, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
import control_panel as cp


class InteractingWorld(cp.World):
    def realized(self, goal_idx):                     # (M,B,N) -> motif shown per tissue
        left = torch.cat([torch.zeros_like(goal_idx[..., :1]), goal_idx[..., :-1]], -1)
        return (goal_idx + left) % self.K

    def target(self, goal_idx):
        return self.motifs[self.realized(goal_idx)].flatten(-2)


def region_err(m, goal_idx, desired_pattern):
    """Run 100 steps with these goals; per-(example, tissue) error vs the desired pattern. (M,B,N)"""
    w = m.w; M, B = goal_idx.shape[:2]
    s = torch.zeros(M, B, cp.C, w.L); u = m.u(F.one_hot(goal_idx, w.K).float())
    for t in range(100): s = m.step(s, u)
    return ((s[:, :, 0].clamp(0, 1) - desired_pattern) ** 2).reshape(M, B, w.N, cp.TW).mean(-1)


@torch.no_grad()
def search(m, ex_sym, desired, mode, sweeps=3):
    w = m.w; M, B, N, K = *ex_sym.shape[:2], w.N, w.K
    table = torch.zeros(M, N, K, dtype=torch.long); rollouts = 0
    def pick(err_by_g, tissues, score_fn):
        for k in range(M):
            for n in tissues:
                for sym in range(K):
                    sel = ex_sym[k, :, n] == sym
                    if sel.any(): table[k, n, sym] = torch.stack([score_fn(e, k, sel, n) for e in err_by_g]).argmin()
    # independent search (also used as the starting point for neighbourhood search)
    errs = [region_err(m, torch.full((M, B, N), g), desired) for g in range(K)]; rollouts += K
    pick(errs, range(N), lambda e, k, sel, n: e[k, sel, n].mean())
    if mode == "neighbourhood":
        for _ in range(sweeps):
            for parity in (0, 1):
                tissues = [n for n in range(N) if n % 2 == parity]
                errs = []
                for g in range(K):
                    trial = table.clone(); trial[:, tissues] = g
                    gi = cp.meaning_apply(trial, ex_sym)
                    errs.append(region_err(m, gi, desired)); rollouts += 1
                def score(e, k, sel, n):
                    r = e[k, sel, n]
                    if n + 1 < N: r = r + e[k, sel, n + 1]
                    return r.mean()
                pick(errs, tissues, score)
    return table, rollouts


def run(N, K, kind, M=8, iters=900, n_ex=8, retrain_steps=400):
    torch.manual_seed(0); np.random.seed(0)
    w = InteractingWorld(N, K); m = cp.Model(w, kind, M, 0); cp.train(m, iters)
    ident = torch.arange(K)[None, None].expand(M, N, -1)
    torch.manual_seed(5)
    new = torch.randint(0, K, (M, N, K))                               # new meaning: random goal table
    ex_sym = torch.randint(0, K, (M, n_ex, N))
    for j in range(K): ex_sym[:, j] = (j + torch.arange(N)) % K
    res = {"N": N, "K": K, "kind": kind}
    onehot = lambda s: F.one_hot(s, K).float()
    if kind == "flat":
        res["orig"] = cp.evaluate(m, onehot, ident)
        t0 = time.time(); mm = copy.deepcopy(m); opt = torch.optim.Adam(mm.parameters(), lr=3e-3)
        for _ in range(retrain_steps):
            b = ex_sym.repeat(1, 2, 1)
            loss, _ = cp.episode(mm, onehot(b), cp.meaning_apply(new, b))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mm.parameters(), M); opt.step()
        res["new"] = cp.evaluate(mm, onehot, new); res["retarget_seconds"] = time.time() - t0
        res["retarget_cost"] = f"{retrain_steps} gradient steps"
    else:
        torch.save(m.state_dict(), f"panel2_N{N}_K{K}.pt")
        res["lower_given_true_goals"] = cp.evaluate(m, lambda s: onehot(cp.meaning_apply(new, s)), new)
        for mode in ("independent", "neighbourhood"):
            t0 = time.time()
            desired = w.target(cp.meaning_apply(new, ex_sym))
            table, n_roll = search(m, ex_sym, desired, mode)
            res[f"new_{mode}"] = cp.evaluate(m, lambda s: onehot(cp.meaning_apply(table, s)), new)
            res[f"cost_{mode}"] = f"{n_roll} rollouts, {time.time() - t0:.0f} s"
    return res


if __name__ == "__main__":
    torch.set_num_threads(1)
    N, K, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    import os; iters = int(os.environ.get("ITERS", 900))
    r = run(N, K, kind, iters=iters, retrain_steps=int(os.environ.get("RETRAIN", 400)))
    json.dump(r, open(f"cp2_N{N}_K{K}_{kind}.json", "w"))
    print("RESULT", json.dumps({k: (round(float(np.mean(v["patterns"])), 2), round(float(np.mean(v["after_damage"])), 2))
                                if isinstance(v, dict) else v for k, v in r.items()}), flush=True)
