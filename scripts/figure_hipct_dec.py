"""One HiP-CT crop, its middle slice six ways: the image; the model's populations as stacked elliptical ODFs; DEC maps
(R = x, G = y, B = z; brightness = line-ness) of the predominant fibre population, of the first, second and third
populations by share, and of the structure tensor's line axis (the benchmark). Real data: viewing only.

    PYTHONPATH=src python scripts/figure_hipct_dec.py VOLUME.ome.zarr OUT.png --centre Z Y X --ckpt MODEL.pt
        [--level 0] [--crop 64] [--context 128] [--window 3] [--label NAME]
"""
import argparse
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
from shapetensor.learn.model import PopulationNet, st_products, st_matrix, window_mean
from odf_render import draw_tensors

ap = argparse.ArgumentParser()
ap.add_argument("volume"); ap.add_argument("out"); ap.add_argument("--centre", type=int, nargs=3, required=True)
ap.add_argument("--ckpt", required=True); ap.add_argument("--level", default="0"); ap.add_argument("--crop", type=int, default=64)
ap.add_argument("--context", type=int, default=128); ap.add_argument("--window", type=float, default=3.0); ap.add_argument("--label", default="")
a = ap.parse_args(); dev = "cuda"

arr = zarr.open_group(a.volume, mode="r")[a.level]; shp = np.array(arr.shape); c = np.array(a.centre)
lo = np.clip(c - a.context // 2, 0, shp - a.context); hi = lo + a.context
block = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
q1, q99 = np.percentile(block, [0.5, 99.5]); x = torch.tensor(np.clip((block - q1) / max(q99 - q1, 1e-6), 0, 1), device=dev)
cl = c - lo; h = a.crop // 2; y0, x0 = cl[1] - h, cl[2] - h; z = int(cl[0])

ck = torch.load(a.ckpt, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
yy, xx = np.meshgrid(np.arange(a.crop), np.arange(a.crop), indexing="ij")
P = np.stack([np.full(yy.size, z), yy.ravel() + y0, xx.ravel() + x0], -1).astype(np.float32)
pts = torch.tensor(P, device=dev); sig = torch.full((pts.shape[0],), a.window, device=dev)
with torch.no_grad():
    f = EI.features(net, x); r = EI.query(net, f, pts, sig, EI.sphere_dirs(8, dev), vol_shape=x.shape, chunk=1024)
ex, sh, T = r["pop_exist"].float(), r["pop_share"][:, :-1].float(), r["pop_T"].float()      # (N, K), (N, K), (N, K, 3, 3)
ev, V = torch.linalg.eigh(T.double()); ev = ev.float().flip(-1); ax = V[..., :, -1].float()     # descending eigenvalues; line axis (z, y, x)
lin = ((ev[..., 0] - ev[..., 1]) / ev[..., 0].clamp_min(1e-9)).clamp(0, 1)
on = ex > 0.5
dec = lambda v, b: (np.abs(v[..., [2, 1, 0]]) * b[..., None]).reshape(a.crop, a.crop, 3)          # (z, y, x) -> RGB = (x, y, z)

shs = torch.where(on, sh, torch.full_like(sh, -1.0)); order = shs.argsort(-1, descending=True)     # populations by share
ar = torch.arange(pts.shape[0], device=dev)
fib = torch.where(on & (lin > 0.3), sh, torch.full_like(sh, -1.0)); kf = fib.argmax(-1); has_f = fib.max(-1).values > 0
pre = dec(ax[ar, kf].cpu().numpy(), (lin[ar, kf] * has_f).cpu().numpy())
pops = [dec(ax[ar, order[:, j]].cpu().numpy(), (lin[ar, order[:, j]] * (shs[ar, order[:, j]] > 0)).cpu().numpy()) for j in range(3)]

j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
sev, sV = torch.linalg.eigh(st_matrix(j6).double()); sev = sev.float()
st_lin = ((sev[:, 1] - sev[:, 0]) / sev[:, 2].clamp_min(1e-12)).clamp(0, 1)
st_dec = dec(sV[..., :, 0].float().cpu().numpy(), st_lin.cpu().numpy())

img = x[z, y0:y0 + a.crop, x0:x0 + a.crop].cpu().numpy()
g = np.arange(2, a.crop, 4); sel = (yy.ravel() % 4 == 2) & (xx.ravel() % 4 == 2)
m6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
qs = torch.tensor(np.nonzero(sel)[0], device=dev); qq, kk = torch.nonzero(on[qs], as_tuple=True); qi = qs[qq]
cen = np.stack([xx.ravel()[qi.cpu().numpy()], yy.ravel()[qi.cpu().numpy()]], -1)

fig, axs = plt.subplots(2, 4, figsize=(20, 11.5))
axs[0, 0].imshow(img, cmap="gray"); axs[0, 0].set_title("HiP-CT image (middle slice of the crop)", fontsize=10)
axs[0, 1].imshow(img, cmap="gray", alpha=0.5)
draw_tensors(axs[0, 1], cen, m6(T[qi, kk]).cpu().numpy(), 1.9, view="z", scale=np.sqrt(sh[qi, kk].clamp(0, 1).cpu().numpy()))
axs[0, 1].set_xlim(-0.5, a.crop - 0.5); axs[0, 1].set_ylim(a.crop - 0.5, -0.5)
axs[0, 1].set_title("model ODFs: populations (elliptical, scaled by sqrt share)", fontsize=10)
axs[0, 2].imshow(np.clip(pre, 0, 1)); axs[0, 2].set_title("DEC: predominant fibre population (model)", fontsize=10)
axs[0, 3].imshow(np.clip(st_dec, 0, 1)); axs[0, 3].set_title("DEC: structure tensor line axis (benchmark)", fontsize=10)
for j in range(3):
    axs[1, j].imshow(np.clip(pops[j], 0, 1)); axs[1, j].set_title(f"DEC: population {j + 1} by share (model)", fontsize=10)
lg = np.zeros((a.crop, a.crop, 3)); u = np.linspace(-1, 1, a.crop); U, W = np.meshgrid(u, u)
inside = U ** 2 + W ** 2 <= 1; lg[..., 0], lg[..., 1] = np.abs(U), np.abs(W); lg[..., 2] = np.sqrt(np.clip(1 - U ** 2 - W ** 2, 0, 1)); lg[~inside] = 1
axs[1, 3].imshow(lg); axs[1, 3].set_title("DEC key: R = x (left-right), G = y (up-down), B = z (through)", fontsize=10)
for ax_ in axs.ravel():
    ax_.set_xticks([]); ax_.set_yticks([])
fig.suptitle(f"{a.label}  |  model step {ck.get('step')}, window {a.window:g} voxels; brightness = line-ness", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5); fig.savefig(a.out, dpi=80); print("saved", a.out)
