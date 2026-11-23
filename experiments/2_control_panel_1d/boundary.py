"""Natural two-neighbour interaction: BOUNDARY MARKING.
Tissue n shows its own motif in cells 1-6 (set by goal_n alone), and lights its edge cell where the neighbour's
goal differs: cell 0 on iff goal_{n-1} != goal_n, cell 7 on iff goal_{n+1} != goal_n (no neighbour -> off).
Usage: python boundary.py <hidden units>"""
import sys, numpy as np, torch
import control_panel as cp
from discover import LayoutWorld, N, K, M


class BoundaryWorld(LayoutWorld):
    def __init__(self, N, K):
        super().__init__(N, K, [-1, 1])
    def target(self, g):
        t = self.motifs[g].clone()                                   # (..., N, TW); motifs never use cells 0, 7
        left_diff = torch.zeros_like(g, dtype=torch.bool); left_diff[..., 1:] = g[..., 1:] != g[..., :-1]
        right_diff = torch.zeros_like(g, dtype=torch.bool); right_diff[..., :-1] = g[..., :-1] != g[..., 1:]
        t[..., 0] = left_diff.float(); t[..., 7] = right_diff.float()
        return t.flatten(-2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    hid = int(sys.argv[1]); cp.HID = hid
    torch.manual_seed(0); np.random.seed(0)
    w = BoundaryWorld(N, K); m = cp.Model(w, "panel", M, 0); cp.train(m, 900)
    torch.save(m.state_dict(), f"panel_boundary_h{hid}.pt")
