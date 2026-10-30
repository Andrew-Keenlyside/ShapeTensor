"""One fibre's disorder varying along its length (synthetic, construction): for a few base disorders, its material (the
texture), its own ODF in each voxel, and along its length the construction's local spread beside the measured
line-ness of its ODF -- both from the same local value.

    PYTHONPATH=src python scripts/figure_along_length.py OUT.png [RADIUS]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn import scene as SC
from odf_render import draw_tensors

out = sys.argv[1]; rad = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0; dev = "cuda"
S = 300; xs = torch.linspace(-6, 70, S, device=dev)
P = torch.stack([torch.full_like(xs, 32.0), torch.full_like(xs, 32.0), xs], -1)[None]                # straight along x (z = y = 32)
bases = (0.2, 0.5, 0.8)
fig, axs = plt.subplots(3, len(bases), figsize=(6.4 * len(bases), 8.0), gridspec_kw={"height_ratios": (0.6, 0.6, 1.3)})
for j, d0 in enumerate(bases):
    c = SC.curves_like(P, torch.full((1, S), rad, device=dev), SC.Rng(5, dev), obj=torch.tensor([1], device=dev), seg=torch.tensor([1], device=dev),
                       contrast=80.0, wave=None, disorder=torch.tensor([d0], device=dev))
    cv = SC.Canvas((64, 64, 64), dev); SC.render_curves(cv, c); r = cv.finish()
    occ, ten = r["occupancy"].float(), r["tensor"].float()
    o2 = occ[32].cpu().numpy(); T2 = ten[:, 32].permute(1, 2, 0).cpu().numpy()
    axs[0, j].imshow(o2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axs[0, j].set_title(f"base disorder {d0:g}: its material (slice)", fontsize=12)
    yy, xx = np.meshgrid(np.arange(32, 33), np.arange(2, 64, 4), indexing="ij"); k = o2[yy, xx] > 0.3
    axs[1, j].imshow(o2, cmap="gray", vmin=0, vmax=1, alpha=0.3)
    draw_tensors(axs[1, j], np.stack([xx[k], yy[k]], -1), T2[yy, xx][k], 3.8, view="z")
    axs[1, j].set_title("its ODF in each voxel", fontsize=12)
    for a_ in axs[:2, j]:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(38.5, 25.5)
    # along its length: the construction's local spread and the measured line-ness of the per-voxel ODF (section mean)
    arcl = torch.cat([torch.zeros(1, 1, device=dev), torch.cumsum((P[:, 1:] - P[:, :-1]).norm(dim=-1), 1)], 1)
    dl = SC.local_disorder(c, arcl)[0]; sp = (SC.TEX_SPREAD[0] + dl * (SC.TEX_SPREAD[1] - SC.TEX_SPREAD[0])).cpu().numpy()
    m = occ > 0.3; lin = []
    for x in range(64):
        mm = m[:, :, x]
        if int(mm.sum()) == 0:
            lin.append(np.nan); continue
        t6 = (ten[:, :, :, x][:, mm] * occ[:, :, x][mm]).sum(1) / occ[:, :, x][mm].sum()
        M = torch.tensor([[t6[0], t6[3], t6[4]], [t6[3], t6[1], t6[5]], [t6[4], t6[5], t6[2]]]); ev = torch.linalg.eigvalsh(M.double()).flip(0)
        lin.append(float((ev[0] - ev[1]) / ev[0]))
    ax = axs[2, j]; ax2 = ax.twinx()
    ax.plot(P[0, :, 2].cpu().numpy(), sp, color="#c0392b", lw=2, label="construction: local spread (deg)")
    ax2.plot(np.arange(64), lin, color="#2c3e50", lw=2, label="measured: line-ness of its ODF")
    ax.set_xlim(0, 63); ax.set_ylim(0, 40); ax2.set_ylim(0.5, 1.0)
    ax.set_xlabel("position along the fibre (voxels)"); ax.set_ylabel("local spread (deg)", color="#c0392b"); ax2.set_ylabel("line-ness", color="#2c3e50")
    ax.set_title("along its length: spread up -> line-ness down", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=72); print("saved", out)
