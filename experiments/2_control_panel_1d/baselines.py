"""
Stronger comparisons on the interacting-parts task, using the SAME saved network (panel2_N8_K4.pt),
the same new meaning and the same 8 examples as chain_search.py. Nothing is retrained except where stated.

  soft_goals_grad   "prompt tuning": learn continuous goal vectors per (tissue, symbol) by gradient through
                    the dynamics (weights frozen). Starts from the original meaning.
  global_search     discrete search WITHOUT local errors: try each option for each (tissue, symbol) entry,
                    judged by the TOTAL error over the whole strip.
  random_subsets    local-error search that is NOT told which tissues neighbour which: each round, a random
                    half of tissues try each option (others fixed); each tissue judged by its own region.
  parity (ref)      the search from chain_search.py (alternates even/odd tissues = knows the neighbour layout).
Cost is reported in rollouts (100-step forward runs); one gradient step costs roughly 3 rollouts.
"""
import time, json, sys, torch, numpy as np, torch.nn.functional as F
import control_panel as cp, control_panel2 as cp2
torch.set_num_threads(1)
N, K, M = 8, 4, 8
w = cp2.InteractingWorld(N, K); m = cp.Model(w, "panel", M, 0); m.load_state_dict(torch.load("panel2_N8_K4.pt"))
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(5)
new = torch.randint(0, K, (M, N, K)); ex_sym = torch.randint(0, K, (M, 8, N))
for j in range(K): ex_sym[:, j] = (j + torch.arange(N)) % K
desired = w.target(cp.meaning_apply(new, ex_sym)); onehot = lambda s: F.one_hot(s, K).float()


def evaluate_table(table):
    torch.manual_seed(11); return cp.evaluate(m, lambda s: onehot(cp.meaning_apply(table, s)), new)


def evaluate_soft(soft):
    def inp(s):
        return torch.gather(soft[:, None].expand(-1, s.shape[1], -1, -1, -1), 3,
                            s[..., None, None].expand(-1, -1, -1, 1, K)).squeeze(3)
    torch.manual_seed(11); return cp.evaluate(m, inp, new)


def soft_goals_grad(steps, lr=0.05):
    soft = F.one_hot(torch.arange(K), K).float()[None, None].expand(M, N, K, K).clone().requires_grad_(True)
    opt = torch.optim.Adam([soft], lr=lr); tgt_goal = cp.meaning_apply(new, ex_sym.repeat(1, 2, 1))
    b = ex_sym.repeat(1, 2, 1)
    for _ in range(steps):
        inp = torch.gather(soft[:, None].expand(-1, b.shape[1], -1, -1, -1), 3,
                           b[..., None, None].expand(-1, -1, -1, 1, K)).squeeze(3)
        loss, _ = cp.episode(m, inp, tgt_goal)
        opt.zero_grad(); loss.backward(); opt.step()
    return soft.detach()


@torch.no_grad()
def global_search(sweeps):
    table = torch.zeros(M, N, K, dtype=torch.long); r = 0
    for _ in range(sweeps):
        for n in range(N):
            for sym in range(K):
                tot = []
                for g in range(K):
                    trial = table.clone(); trial[:, n, sym] = g
                    tot.append(cp2.region_err(m, cp.meaning_apply(trial, ex_sym), desired).mean((1, 2))); r += 1
                table[:, n, sym] = torch.stack(tot).argmin(0)
    return table, r


@torch.no_grad()
def subset_search(rounds, choose):
    table = torch.zeros(M, N, K, dtype=torch.long); r = 0
    for i in range(rounds):
        tissues = choose(i); errs = []
        for g in range(K):
            trial = table.clone(); trial[:, tissues] = g
            errs.append(cp2.region_err(m, cp.meaning_apply(trial, ex_sym), desired)); r += 1
        for k in range(M):
            for n in tissues:
                for sym in range(K):
                    sel = ex_sym[k, :, n] == sym
                    if sel.any(): table[k, n, sym] = torch.stack([e[k, sel, n].mean() for e in errs]).argmin()
    return table, r


def report(name, r, cost, extra=""):
    line = f"| {name} | {np.mean(r['patterns'])*100:.0f}% | {np.mean(r['after_damage'])*100:.0f}% | {cost} | {extra} |"
    print(line, flush=True); return line


rows = ["| Method | Correct patterns | After damage | Cost | Settings correct |", "|---|---|---|---|---|"]
t0 = time.time(); rows.append(report("Correct settings given by hand (ceiling)", evaluate_table(new), "-"))
for steps in (100, 300):
    t0 = time.time(); soft = soft_goals_grad(steps)
    rows.append(report(f"Gradient-tuned goal inputs ('prompt tuning'), {steps} steps", evaluate_soft(soft),
                       f"~{3*steps} rollout-equivalents, {time.time()-t0:.0f} s"))
rng = np.random.default_rng(0)
for rounds in (8, 16):
    t0 = time.time(); tab, n = subset_search(rounds, lambda i: [t for t in range(N) if rng.random() < 0.5])
    rows.append(report(f"Local-error search, random halves (not told the layout), {rounds} rounds", evaluate_table(tab),
                       f"{n} rollouts, {time.time()-t0:.0f} s", f"{(tab == new).float().mean()*100:.0f}%"))
t0 = time.time(); tab, n = subset_search(8, lambda i: [t for t in range(N) if t % 2 == i % 2])
rows.append(report("Local-error search, alternating neighbours (told the layout), 8 rounds", evaluate_table(tab),
                   f"{n} rollouts, {time.time()-t0:.0f} s", f"{(tab == new).float().mean()*100:.0f}%"))
t0 = time.time(); tab, n = global_search(1)
rows.append(report("Search judged by TOTAL error only (no local errors), 1 sweep", evaluate_table(tab),
                   f"{n} rollouts, {time.time()-t0:.0f} s", f"{(tab == new).float().mean()*100:.0f}%"))
open("baselines_results.md", "w").write("# Stronger comparisons, interacting parts (8 tissues x 4 options, 8 seeds)\n\n"
                                        "Retraining all weights on the same 8 examples (earlier run): 1% / 1%, 400 gradient steps.\n\n"
                                        + "\n".join(rows) + "\n")
