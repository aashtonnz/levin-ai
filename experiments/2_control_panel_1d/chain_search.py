"""Alternative searches on the saved interacting-parts panel model (no retraining)."""
import time, json, torch, numpy as np, torch.nn.functional as F
import control_panel as cp, control_panel2 as cp2
torch.set_num_threads(1)
N, K, M = 8, 4, 8
w = cp2.InteractingWorld(N, K); m = cp.Model(w, "panel", M, 0); m.load_state_dict(torch.load("panel2_N8_K4.pt"))
torch.manual_seed(5)                                   # same new meaning + examples as the main run
new = torch.randint(0, K, (M, N, K)); ex_sym = torch.randint(0, K, (M, 8, N))
for j in range(K): ex_sym[:, j] = (j + torch.arange(N)) % K
desired = w.target(cp.meaning_apply(new, ex_sym)); onehot = lambda s: F.one_hot(s, K).float()

@torch.no_grad()
def own_region_search(sweeps):
    """Alternate even/odd tissues; judge each candidate ONLY by the tissue's own region (given its left neighbour)."""
    table = torch.zeros(M, N, K, dtype=torch.long); rollouts = 0
    for _ in range(sweeps):
        for parity in (0, 1):
            tissues = [n for n in range(N) if n % 2 == parity]; errs = []
            for g in range(K):
                trial = table.clone(); trial[:, tissues] = g
                errs.append(cp2.region_err(m, cp.meaning_apply(trial, ex_sym), desired)); rollouts += 1
            for k in range(M):
                for n in tissues:
                    for sym in range(K):
                        sel = ex_sym[k, :, n] == sym
                        if sel.any(): table[k, n, sym] = torch.stack([e[k, sel, n].mean() for e in errs]).argmin()
    return table, rollouts

torch.manual_seed(11)
truth = cp.evaluate(m, lambda s: onehot(cp.meaning_apply(new, s)), new)
print("lower level given the TRUE goals: patterns %.2f, after damage %.2f" % (np.mean(truth["patterns"]), np.mean(truth["after_damage"])))
out = {"true_goals": truth}
for sweeps in (2, 4, 8):
    t0 = time.time(); table, n = own_region_search(sweeps)
    torch.manual_seed(11); r = cp.evaluate(m, lambda s: onehot(cp.meaning_apply(table, s)), new)
    acc = (table == new).float().mean().item()
    out[f"own_region_{sweeps}"] = r
    print(f"own-region search, {sweeps} sweeps ({n} rollouts, {time.time()-t0:.0f}s): patterns {np.mean(r['patterns']):.2f}, "
          f"after damage {np.mean(r['after_damage']):.2f}, table entries matching truth {acc:.2f}")
json.dump(out, open("chain_search_results.json", "w"))
