"""Lower level whose cells also SENSE the goal signals of neighbouring cells (like paracrine signalling).
Same size of hidden layer and training budget as before; only the perception input is wider."""
import sys, numpy as np, torch, torch.nn.functional as F
import control_panel as cp
from discover import LayoutWorld, OFFSETS, N, K, M


class SenseModel(cp.Model):
    def __init__(self, w, kind, M, seed):
        super().__init__(w, kind, M, seed)
        g = torch.Generator().manual_seed(seed)
        P = 3 * (cp.C + w.N + w.K)
        self.W1 = torch.nn.Parameter(torch.randn(M, cp.HID, P, generator=g) / P ** 0.5)
        self.ker2 = cp.KER.repeat(cp.C + w.N + w.K, 1).unsqueeze(1)

    def step(self, s, u):
        w = self.w; M_, B = s.shape[:2]; ch = cp.C + w.N + w.K
        full = torch.cat([s, w.idch.expand(M_, B, w.N, w.L), u], 2)
        p = F.conv1d(F.pad(full.reshape(M_ * B, ch, w.L), (1, 1), value=cp.EDGE), self.ker2, groups=ch)
        h = F.relu(torch.einsum("mhp,mbpl->mbhl", self.W1, p.reshape(M_, B, -1, w.L)) + self.b1[:, None, :, None])
        d = torch.einsum("mch,mbhl->mbcl", self.W2, h) + self.b2[:, None, :, None]
        return (s + d + cp.NOISE * torch.randn_like(s)).clamp(-2, 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    layout = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    torch.manual_seed(0); np.random.seed(0)
    w = LayoutWorld(N, K, OFFSETS[layout]); m = SenseModel(w, "panel", M, 0); cp.train(m, iters)
    torch.save(m.state_dict(), f"panel_sense_{layout}.pt")
