"""A/B of one fibre's ODF target against its internal texture (synthetic, construction): A, the generator before 30 Oct
(no internal texture, a needle); B, the same fibre with its disorder low, middle and high -- its texture (the
material's density inside it: depth and streak misalignment) and its own ODF (each voxel's ellipse) from one parameter.

    PYTHONPATH=src python scripts/figure_ab_texture.py OUT.png [RADIUS]
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
S = 260; xs = torch.linspace(-4, 68, S, device=dev)
P = torch.stack([torch.full_like(xs, 32.0), 32 + 4 * torch.sin(xs / 14), xs], -1)[None]             # a gently curving fibre in the z = 32 plane
cases = [("A (before): no texture", False, None)] + [(f"B: disorder {d:g}", True, d) for d in (0.0, 0.5, 1.0)]
fig, axs = plt.subplots(2, len(cases), figsize=(5.4 * len(cases), 5.6))
for j, (lab, tex, dis) in enumerate(cases):
    SC.TEXTURE_ODF = tex
    kw = dict(wave=None) if tex else dict(wave=None, spread=torch.tensor([2.0], device=dev))
    if dis is not None:
        kw["disorder"] = torch.tensor([float(dis)], device=dev)
    c = SC.curves_like(P, torch.full((1, S), rad, device=dev), SC.Rng(3, dev), obj=torch.tensor([1], device=dev), seg=torch.tensor([1], device=dev), contrast=80.0, **kw)
    cv = SC.Canvas((64, 64, 64), dev); SC.render_curves(cv, c); r = cv.finish()
    occ, ten = r["occupancy"][32].float(), r["tensor"][:, 32].float()
    axs[0, j].imshow(occ.clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    sp = float(c.spread[0]); dep = float(c.fibril[0, 0]) if c.fibril is not None else 0.0
    axs[0, j].set_title(f"{lab}\nits material (slice z = 32)", fontsize=12)
    yy, xx = np.meshgrid(np.arange(0, 64, 2), np.arange(0, 64, 2), indexing="ij"); o = occ.cpu().numpy(); T = ten.permute(1, 2, 0).cpu().numpy()
    k = o[yy, xx] > 0.3
    axs[1, j].imshow(o, cmap="gray", vmin=0, vmax=1, alpha=0.3)
    draw_tensors(axs[1, j], np.stack([xx[k], yy[k]], -1), T[yy, xx][k], 1.9, view="z")
    t6 = torch.tensor(T[yy, xx][k]).mean(0); M = torch.tensor([[t6[0], t6[3], t6[4]], [t6[3], t6[1], t6[5]], [t6[4], t6[5], t6[2]]])
    ev = torch.linalg.eigvalsh(M.double()).flip(0)
    axs[1, j].set_title(f"its ODF in each voxel: spread {sp:.0f} deg\nline-ness {float((ev[0] - ev[1]) / ev[0]):.2f}", fontsize=12)
    for a_ in axs[:, j]:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(42.5, 21.5)
SC.TEXTURE_ODF = True
fig.tight_layout(); fig.savefig(out, dpi=72); print("saved", out)
