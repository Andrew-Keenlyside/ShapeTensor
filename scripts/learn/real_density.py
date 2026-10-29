"""Density glyphs of a density model on real HiP-CT (S-21-46 20 µm cortex VOI), at several description scales.

    python scripts/learn/real_density.py RUN_DIR [--region vessel|boundary|Y,X] [--zoom 64] [--out FILE]

The slab as in `real_glyphs.py` (96 middle slices, 8-bit, untouched). Panels: the zoom; the lattice SH head (sigma 5,
a glyph per 4-voxel cell); with a query decoder, the density at sigma 2, 5 and 10 voxels (40, 100, 200 µm) on a 2-voxel
grid of the same zoom -- one model, one input, several description scales. Glyphs as in `qc_density.py` (in-plane
cut, per-glyph shape, size ∝ √ order-0 mass, colour = top peak's DEC, white tick = a second peak).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from qc_density import glyphs                                                                          # noqa: E402
from real_glyphs import SRC, otsu                                                                      # noqa: E402

from shapetensor.learn.model import PopulationNet

DEV = "cuda"; HALF_Z = 48


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--region", default="vessel"); ap.add_argument("--zoom", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args(); run = Path(a.run); ZW = a.zoom
    ck = torch.load(run / "model.pt", map_location=DEV); args = ck["args"]; lmax = int(args.get("sh_lmax", 0)); assert lmax
    net = PopulationNet.from_args(args).to(DEV); net.load_state_dict(ck["model"]); net.eval()
    arr = zarr.open_group(SRC, mode="r")["0"]; c = arr.shape[0] // 2
    img = np.asarray(arr[c - HALF_Z:c + HALF_Z]).astype(np.uint8)
    tissue = img > otsu(img); win = tuple(float(q) for q in np.quantile(img[tissue], [0.005, 0.995]))
    k = HALF_Z; g = np.clip((img[k].astype(np.float32) - win[0]) / (win[1] - win[0]), 0, 1); tis = tissue[k]
    h2 = ZW // 2
    if "," in a.region:
        cy, cx = (int(t) for t in a.region.split(","))
    else:
        if a.region == "vessel":
            dark = ndimage.gaussian_filter(g, 8) < 0.2; score = ndimage.uniform_filter((dark & (g > 0.5)).astype(np.float32), ZW)
        else:
            score = ndimage.uniform_filter(np.hypot(*np.gradient(ndimage.gaussian_filter(g, 8))), ZW)
        score[:h2] = score[-h2:] = 0; score[:, :h2] = score[:, -h2:] = 0
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
    y0, x0 = (min(max(int(c_) - h2, 0), n_ - ZW) // 4 * 4 for c_, n_ in ((cy, g.shape[0]), (cx, g.shape[1])))
    x = torch.as_tensor(((img.astype(np.float32) - win[0]) / (win[1] - win[0])).clip(0, 1), device=DEV)[None, None]
    has_q = bool(args.get("query"))
    cols = 2 + (3 if has_q else 0)
    fig, axs = plt.subplots(1, cols, figsize=(5.2 * cols, 5.6), facecolor="white", squeeze=False); axs = axs[0]
    ext = (-0.5, ZW - 0.5, ZW - 0.5, -0.5); under = g[y0:y0 + ZW, x0:x0 + ZW]
    axs[0].imshow(under, cmap="gray", extent=ext); axs[0].set_title(f"HiP-CT zoom {ZW}² voxels ({ZW * 0.02:.2f} mm), axial", fontsize=10)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        P = net(x)
    kc = k // 4
    iy, ix = np.mgrid[y0 // 4:(y0 + ZW) // 4, x0 // 4:(x0 + ZW) // 4]
    cp = P["sh"][0].float()[:, kc, iy, ix].reshape(P["sh"].shape[1], -1)
    mref = float(torch.quantile(cp[0][cp[0] > 0], 0.9)) if bool((cp[0] > 0).any()) else 1.0
    axs[1].imshow(under, cmap="gray", alpha=0.5, extent=ext)
    n2, nk = glyphs(axs[1], cp, (4 * ix + 1.5 - x0).ravel(), (4 * iy + 1.5 - y0).ravel(), lmax, 1.9, mref)
    axs[1].set_title(f"lattice head (σ 5 vox = 100 µm): {n2}/{nk} with 2 peaks", fontsize=10)
    if has_q:
        gy, gx = np.mgrid[y0:y0 + ZW:2, x0:x0 + ZW:2] + 0.5
        pts = torch.tensor(np.stack([np.full(gy.size, k + 0.0), gy.ravel(), gx.ravel()], 1), device=DEV, dtype=torch.float32)
        for j, sig in enumerate((2.0, 5.0, 10.0)):
            sv = torch.full((pts.shape[0],), sig, device=DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                qp = net(x, pts[None], sv[None])["q_sh"][0].float().T
            mq = float(torch.quantile(qp[0][qp[0] > 0], 0.9)) if bool((qp[0] > 0).any()) else 1.0
            ax = axs[2 + j]; ax.imshow(under, cmap="gray", alpha=0.5, extent=ext)
            n2, nk = glyphs(ax, qp, gx.ravel() - x0, gy.ravel() - y0, lmax, 0.95, mq)
            ax.set_title(f"query decoder σ {sig:g} vox ({sig * 20:.0f} µm): {n2}/{nk} two peaks", fontsize=10)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, ZW - 0.5); ax.set_ylim(ZW - 0.5, -0.5)
    fig.suptitle(f"Population density on real HiP-CT from {run.name} (step {ck['step']}; synthetic training only)", fontsize=12)
    out = Path(a.out) if a.out else run / f"real_density_{a.region.replace(',', '_')}.png"
    plt.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=70); print("wrote", out)


if __name__ == "__main__":
    main()
