"""One fibre, built exactly as the generator builds objects (curves_like + render_curves), curving in the viewed plane:
its material with the true centreline; its target, unblurred: at every voxel the fibre's own orientation there (its
per-voxel elliptical ODF) with the voxel's share of fibre material, nothing where there is no fibre; and the target's
axis error against the true tangent along the fibre. (A window-averaged target is shown for comparison if WINDOW > 0.)

    PYTHONPATH=src python scripts/figure_one_fibre.py OUT.png [RADIUS] [WINDOW]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.filters import gaussian
from shapetensor.learn.scene import Canvas, Rng, curves_like, render_curves
from odf_render import draw_tensors

out = sys.argv[1]; rad = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5; WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
S = 400; xs = torch.linspace(-6, 70, S, device=dev)
P = torch.stack([torch.full_like(xs, 32.0), 32 + 12 * torch.sin(xs / 11), xs], -1)[None]        # (z, y, x): a curve in the z = 32 plane
c = curves_like(P, torch.full((1, S), rad, device=dev), Rng(0, dev), obj=torch.tensor([1], device=dev), seg=torch.tensor([1], device=dev), contrast=80.0)
cv = Canvas((64, 64, 64), dev); render_curves(cv, c); r = cv.finish()
occ, ten = r["occupancy"].float(), r["tensor"].float()                                          # (Z, Y, X), (6, Z, Y, X) zz yy xx zy zx yx
spread = float(c.spread[0]) if c.spread is not None else float("nan")

z0 = 32; o2 = occ[z0].cpu().numpy()
tg = torch.nn.functional.normalize(torch.gradient(P[0], dim=0)[0], dim=-1)                       # true tangents along the centreline
pc = P[0].cpu().numpy()

def t6_axis(t6):
    M = torch.stack([torch.stack([t6[..., 0], t6[..., 3], t6[..., 4]], -1), torch.stack([t6[..., 3], t6[..., 1], t6[..., 5]], -1),
                     torch.stack([t6[..., 4], t6[..., 5], t6[..., 2]], -1)], -2)
    return torch.linalg.eigh(M.double())[1][..., :, 2].float()

# per-voxel: every 2nd voxel inside the fibre
yy, xx = torch.meshgrid(torch.arange(1, 64, 2, device=dev), torch.arange(1, 64, 2, device=dev), indexing="ij")
inside = occ[z0, yy, xx] > 0.5
pv = ten[:, z0, yy, xx].permute(1, 2, 0)[inside]
cen_v = torch.stack([xx[inside], yy[inside]], -1).cpu().numpy()
# windowed target: the occupancy-weighted mean of the per-voxel ellipses in a Gaussian window
W = gaussian(occ[None, None], WIN)[0, 0]; TW = torch.stack([gaussian((occ * ten[k])[None, None], WIN)[0, 0] for k in range(6)])
gy, gx = torch.meshgrid(torch.arange(2, 64, 4, device=dev), torch.arange(2, 64, 4, device=dev), indexing="ij")
m = W[z0, gy, gx]; keep = m > 0.03
tw = (TW[:, z0, gy, gx] / W[z0, gy, gx].clamp_min(1e-9)).permute(1, 2, 0)[keep]
tw = tw / tw[:, :3].sum(-1, keepdim=True)
cen_w = torch.stack([gx[keep], gy[keep]], -1).cpu().numpy()
# axis error of the windowed target at the centreline points
cp = P[0][(P[0][:, 2] > 4) & (P[0][:, 2] < 60)]; ti = tg[(P[0][:, 2] > 4) & (P[0][:, 2] < 60)]
idx = cp.round().long().clamp(0, 63)
tc = (TW[:, idx[:, 0], idx[:, 1], idx[:, 2]] / W[idx[:, 0], idx[:, 1], idx[:, 2]].clamp_min(1e-9)).T
err = torch.rad2deg(torch.acos((t6_axis(tc) * ti).sum(-1).abs().clamp(max=1))).cpu().numpy()
errv = torch.rad2deg(torch.acos((t6_axis(ten[:, idx[:, 0], idx[:, 1], idx[:, 2]].T) * ti).sum(-1).abs().clamp(max=1))).cpu().numpy()

fig, axs = plt.subplots(1, 4, figsize=(22, 5.6))
sh_v = occ[z0, yy, xx][inside].clamp(0, 1).cpu().numpy()
axs[0].imshow(o2, cmap="gray", vmin=0, vmax=1); axs[0].plot(pc[:, 2], pc[:, 1], color="yellow", lw=1)
axs[0].set_title(f"one fibre: its material (slice z = {z0}) and true centreline\nradius {rad:g} voxels, its own spread {spread:.1f} deg", fontsize=10)
axs[1].imshow(o2, cmap="gray", vmin=0, vmax=1, alpha=0.35); draw_tensors(axs[1], cen_v, pv.cpu().numpy(), 1.3, view="z")
axs[1].set_title("target, unblurred: the fibre's own orientation at each voxel\n(every 2nd voxel; none where there is no fibre)", fontsize=10)
axs[2].imshow(o2, cmap="gray", vmin=0, vmax=1, alpha=0.35); draw_tensors(axs[2], cen_w, tw.cpu().numpy(), 1.8, view="z")
axs[2].set_title(f"for comparison, the old target: window {WIN:g} average\n(spreads off the fibre, averages bends)", fontsize=10)
for a_ in axs[:3]:
    a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(63.5, -0.5)
axs[3].plot(cp[:, 2].cpu().numpy(), errv, label="unblurred target (per voxel)", lw=1.5)
axs[3].plot(cp[:, 2].cpu().numpy(), err, label=f"old target, window {WIN:g}", lw=1, alpha=0.6)
axs[3].set_xlabel("position along x (voxels)"); axs[3].set_ylabel("axis vs true tangent (deg)"); axs[3].legend(fontsize=9); axs[3].grid(alpha=0.3)
axs[3].set_title(f"axis vs true tangent: unblurred median {np.median(errv):.1f} deg (max {np.max(errv):.1f}); old window-{WIN:g} {np.median(err):.1f}", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=85); print("saved", out, "| spread", round(spread, 2), "| median errors", round(float(np.median(errv)), 2), round(float(np.median(err)), 2))
