import sys; sys.argv=["x","--seeds","8"]
exec(open("levin_parts.py").read().split('if __name__ == "__main__":')[0])
import torch, time
parts = true_parts()
torch.manual_seed(1); np.random.seed(1)
p = train(init(1, "fixed"), parts, int(os.environ.get("IT", "600")), tag="true")
torch.manual_seed(7); np.random.seed(7)
g = torch.randint(0, K, (M, 32, N), device=DEV); tgt = target(g)
with torch.no_grad():
    enc = encode(p, parts, tgt, noise=False)
    a, _ = check(p, parts, enc, tgt)
    c = enc.argmax(-1)
    # per-seed: does the code identify the goal? (purity of goal given code)
    for k in range(M):
        tab = torch.zeros(K, V)
        for gi, ci in zip(g[k].flatten(), c[k].flatten()): tab[gi, ci] += 1
        purity = (tab.max(0).values.sum() / tab.sum()).item()
        # with correct codes given by an oracle mapping (best code per goal), does the NCA render?
        best = tab.argmax(1)                      # code most used for each goal
        oc = F.one_hot(best.to(DEV)[g[k:k+1]], V).float()
        s = torch.zeros(1, 32, C, H, W, device=DEV); u = parts.field(torch.cat([oc]*M)[:M]) if False else None
        print(f"seed {k}: steer {100*a[k]:.0f}% | codes used {int((tab.sum(0)>0).sum())} | goal purity {purity:.2f} | goal->code {best.tolist()}")
