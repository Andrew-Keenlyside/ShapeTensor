"""A grid of ODFs in a small volume of interest (VOI), shown against the same VOI's image, with the structure tensor's
axis at the same points as the benchmark.

The model runs once on a context block around the VOI. A 4 x 4 x 4 grid of query points inside the VOI is read with
window ``--window`` (voxels). Each z-layer of the grid is drawn as the image slice through it, with the ODF glyphs on
top (DEC: R = x, G = y, B = z; radius = q / max q; grey = near flat), beside the structure tensor's line axis
(gradients at 1, the same window). A locator shows the VOI on a whole slice. The VOI image, points and ODFs are also
saved as .npz.

    PYTHONPATH=src python scripts/figure_voi_odfs.py VOLUME.ome.zarr OUT_PREFIX --centre Z Y X --ckpt MODEL.pt
        [--level 0] [--voi 32] [--context 128] [--window 4] [--label NAME]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor.learn import entity_infer as EI
from odf_render import draw, fit_sh
from shapetensor.learn.model import PopulationNet, fib_hemisphere, st_products, st_matrix, window_mean, CE_GRID

ap = argparse.ArgumentParser()
ap.add_argument("volume"); ap.add_argument("out"); ap.add_argument("--centre", type=int, nargs=3, required=True)
ap.add_argument("--ckpt", required=True); ap.add_argument("--level", default="0"); ap.add_argument("--voi", type=int, default=32)
ap.add_argument("--context", type=int, default=128); ap.add_argument("--window", type=float, default=4.0); ap.add_argument("--label", default="")
a = ap.parse_args(); dev = "cuda"

arr = zarr.open_group(a.volume, mode="r")[a.level]; shp = np.array(arr.shape); c = np.array(a.centre)
lo = np.clip(c - a.context // 2, 0, shp - a.context); hi = lo + a.context
block = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
q1, q99 = np.percentile(block, [0.5, 99.5]); x = torch.tensor(np.clip((block - q1) / max(q99 - q1, 1e-6), 0, 1), device=dev)
cl = c - lo                                                                                     # the VOI centre in the block
offs = (np.arange(4) - 1.5) * (a.voi / 4)                                                       # 4 points per axis, evenly inside the VOI
P = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3) + cl
pts = torch.tensor(P, dtype=torch.float32, device=dev); sig = torch.full((pts.shape[0],), a.window, device=dev)

ck = torch.load(a.ckpt, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
grid = fib_hemisphere(CE_GRID, dev)
with torch.no_grad():
    f = EI.features(net, x); r = EI.query(net, f, pts, sig, grid, vol_shape=x.shape)
q = torch.exp(r["logq"].float()).cpu().numpy()
j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
ev, V = torch.linalg.eigh(st_matrix(j6).double()); ev = ev.float()
st_ax = V[..., :, 0].float().cpu().numpy(); lin = ((ev[:, 1] - ev[:, 0]) / ev[:, 2].clamp_min(1e-12)).clamp(0, 1).cpu().numpy()

h = a.voi // 2; vlo = cl - h; voi = x[vlo[0]:vlo[0] + a.voi, vlo[1]:vlo[1] + a.voi, vlo[2]:vlo[2] + a.voi].cpu().numpy()
np.savez_compressed(a.out + ".npz", voi=(voi * 255).astype(np.uint8), voi_origin=(lo + vlo), points=(P + lo), window=a.window,
                    dirs=grid.cpu().numpy(), odf=q, st_axis=st_ax, st_lineness=lin)

U = np.concatenate([grid.cpu().numpy(), -grid.cpu().numpy()]); DEC = np.clip(np.abs(U[:, ::-1]), 0, 1); step = a.voi / 4
fig = plt.figure(figsize=(13, 17)); gs = fig.add_gridspec(5, 3, height_ratios=[1.25, 1, 1, 1, 1])
ax0 = fig.add_subplot(gs[0, :]); sl = np.asarray(arr[int(c[0])]).astype(np.float32); s1, s99 = np.percentile(sl, [0.5, 99.5])
ax0.imshow(np.clip((sl - s1) / max(s99 - s1, 1e-6), 0, 1), cmap="gray")
ax0.add_patch(plt.Rectangle((c[2] - h, c[1] - h), a.voi, a.voi, fill=False, ec="r", lw=1.5))
ax0.add_patch(plt.Rectangle((lo[2], lo[1]), a.context, a.context, fill=False, ec="y", lw=0.8, ls="--"))
ax0.set_title(f"{a.label}  slice z = {c[0]}: VOI (red, {a.voi}^3 voxels) and model context (yellow, {a.context}^3)", fontsize=9)
ax0.set_xticks([]); ax0.set_yticks([])
for k in range(4):                                                                               # one row per z-layer of the grid
    zl = int(round(P[16 * k, 0] - vlo[0])); sel = slice(16 * k, 16 * (k + 1))
    img = voi[zl]; Pk = P[sel] - vlo                                                             # VOI coordinates
    axs = [fig.add_subplot(gs[1 + k, j]) for j in range(3)]
    for ax in axs:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    axs[0].set_title(f"VOI slice z = {vlo[0] + lo[0] + zl}", fontsize=8)
    draw(axs[1], Pk[:, [2, 1]], fit_sh(q[sel], grid.cpu().numpy()), 0.46 * step, view="z")
    axs[1].set_title(f"model ODFs, SH order 8 (window {a.window:g} voxels; min-max scaled, grey = flat)", fontsize=8)
    for p, d, l in zip(Pk, st_ax[sel], lin[sel]):
        hh = 0.46 * step * max(l, 0.05); axs[2].plot([p[2] - hh * d[2], p[2] + hh * d[2]], [p[1] - hh * d[1], p[1] + hh * d[1]], color=np.clip(np.abs(d[::-1]), 0, 1), lw=2)
    axs[2].set_title("structure tensor axis (length = line-ness)", fontsize=8)
fig.suptitle("ODFs in a volume of interest, against the same region's image (DEC: R = x, G = y, B = z)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(a.out + ".png", dpi=100); print("saved", a.out + ".png")
