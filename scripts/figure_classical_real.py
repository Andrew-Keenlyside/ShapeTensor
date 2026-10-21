"""The classical pathway beside the standard structure tensor on the HiP-CT cortex VOI (S-21-46, 20 µm).

A slab of 96 axial slices through the VOI centre, the full 768 x 768 in-plane; the readers' signature dictionary
calibrated on the slab itself (as run_volume calibrates a chunk). Shown on the slab's middle slice: the image; the
standard (plain) structure tensor, sigma 1, rho 10, in DEC; the classical pathway's merged field; its second lobe's
direction where it has one; the number of lobes; the explained share (the presence statistic); and a zoom on the
grey-white boundary with sticks for the structure tensor and the classical lobes (in-plane components).

    python scripts/figure_classical_real.py [OUT_DIR]        ENGINE=anatomix_devvit (default) | anatomix_enc2 | anatomix16
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor import st
from shapetensor.classical import classical_field
from shapetensor.pipeline import calibrate
from shapetensor.viz import dec_rgb

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/classical_real")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; ENGINE = os.environ.get("ENGINE", "anatomix_devvit"); HALF_Z = 48
ZOOM = tuple(int(v) for v in os.environ.get("ZOOM", "-1,-1").split(","))                  # (y0, x0) of a 192² zoom; -1: the boundary


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def sticks(ax, d, w, step, color, scale=0.45, lw=0.8, thr=0.0):
    """In-plane sticks of directions ``d (3, H, W)`` (z, y, x): the (y, x) components, length ∝ in-plane share."""
    H, W = d.shape[1:]
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    dy, dx = d[1][ys, xs], d[2][ys, xs]; ww = w[ys, xs]
    keep = ww > thr
    L = scale * step
    for y, x, a, b in zip(ys[keep], xs[keep], dy[keep], dx[keep]):
        ax.plot([x - L * b, x + L * b], [y - L * a, y + L * a], color=color, lw=lw)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    a = zarr.open_group(SRC, mode="r")["0"]; c = a.shape[0] // 2
    img = np.asarray(a[c - HALF_Z:c + HALF_Z]).astype(np.uint8)                               # (96, 768, 768)
    v = torch.as_tensor(img, device=DEV).float(); tissue = v > otsu(img)
    log("slab", img.shape)
    d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=False, order="zyx", device=DEV)
    log("structure tensor done")
    cal = calibrate(v, tissue, sigma=1.0, rho=10.0, truncate=3.0, engine=ENGINE, engine_kwargs={}, smooth=3, seed=0, device=DEV)
    log(f"calibrated ({ENGINE}): orientation bins {cal.orientation_bins_occupied}")
    o = classical_field(cal, v, tissue, device=DEV)
    log("classical done")
    k = HALF_Z
    g = img[k].astype(np.float32); g = np.clip((g - np.percentile(g, 1)) / max(np.ptp(np.percentile(g, [1, 99])), 1e-6), 0, 1)
    tis = tissue[k].cpu().numpy()
    dst = d_st[:, k].cpu().numpy(); dcl = o["vec"][:, k].cpu().numpy()
    V = o["vecs"][:, :, k].cpu().numpy(); Wt = o["weights"][:, k].cpu().numpy()
    rel2 = Wt[1] / np.maximum(Wt[0], 1e-9)
    nlob = (Wt > 0.25 * Wt[0:1]).sum(0) * (Wt[0] > 0)
    expl = o["explained"][k].cpu().numpy()
    dec = lambda d: dec_rgb(d) * tis[..., None]
    second = dec_rgb(V[1]) * (rel2 >= 0.25)[..., None] * tis[..., None]
    panels = [("HiP-CT image (S-21-46, 20 µm)", np.stack([g] * 3, -1)), ("standard structure tensor (σ 1, ρ 10)", dec(dst)),
              (f"classical, merged ({ENGINE})", dec(dcl)), ("classical, second lobe (≥ 25% of the first)", second)]
    fig, axs = plt.subplots(2, 4, figsize=(19, 10.4), facecolor="white")
    for ax, (t, im) in zip(axs[0], panels):
        ax.imshow(im, interpolation="nearest"); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].plot([20, 20 + 250], [740, 740], color="yellow", lw=3); axs[0, 0].text(20, 725, "5 mm", color="yellow", fontsize=9)
    h = axs[1, 0].imshow(np.where(tis, nlob, -1), cmap=plt.matplotlib.colors.ListedColormap(["black", "#444444", "#2a78d6", "#eb6834", "#e34948"]),
                         vmin=-1.5, vmax=3.5, interpolation="nearest")
    axs[1, 0].set_title("classical: lobes per voxel (≥ 25% of the first)", fontsize=10); cb = plt.colorbar(h, ax=axs[1, 0], fraction=0.046, ticks=[0, 1, 2, 3])
    h = axs[1, 1].imshow(np.where(tis, expl, np.nan), cmap="magma", vmin=0, vmax=float(np.nanpercentile(np.where(tis, expl, np.nan), 99)))
    axs[1, 1].set_title("classical: explained share (presence statistic)", fontsize=10); plt.colorbar(h, ax=axs[1, 1], fraction=0.046)
    # zoom: the grey-white boundary (largest tissue-intensity gradient in a 192² window) unless given
    if ZOOM[0] < 0:
        from scipy import ndimage
        gm = ndimage.gaussian_filter(g, 8); gy, gx = np.gradient(gm); mag = ndimage.uniform_filter(np.hypot(gy, gx) * tis, 192)
        y0, x0 = np.unravel_index(np.argmax(mag[96:-96, 96:-96]), mag[96:-96, 96:-96].shape); y0, x0 = int(y0), int(x0)
    else:
        y0, x0 = ZOOM
    zs = (slice(y0, y0 + 192), slice(x0, x0 + 192))
    for ax, d, title in ((axs[1, 2], dst, "zoom: standard structure tensor"), (axs[1, 3], None, "zoom: classical lobes (1st white, 2nd orange)")):
        ax.imshow(g[zs], cmap="gray", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        if d is not None:
            sticks(ax, d[:, zs[0], zs[1]], tis[zs].astype(float), 8, "#00e0ff")
        else:
            sticks(ax, V[0][:, zs[0], zs[1]], (Wt[0][zs] > 0).astype(float), 8, "white")
            sticks(ax, V[1][:, zs[0], zs[1]], (rel2[zs] >= 0.25).astype(float), 8, "#ff8c1a")
    for ax in axs[0, 1:]:
        ax.add_patch(plt.Rectangle((x0, y0), 192, 192, fill=False, ec="white", lw=1.0))
    fig.suptitle(f"Classical pathway vs the standard structure tensor, S-21-46 20 µm cortex VOI, axial slice z = {c} (slab {c - HALF_Z}-{c + HALF_Z}); "
                 f"DEC R |x| G |y| B |z|, tissue only", fontsize=11)
    plt.tight_layout()
    path = OUT / f"classical_vs_st_{ENGINE}.png"; fig.savefig(path, dpi=100); log("figure", path)
    share2 = float(((rel2 >= 0.25) & tis & (Wt[0] > 0)).sum() / max(tis.sum(), 1))
    agree = np.degrees(np.arccos(np.clip(np.abs((dst * dcl).sum(0)), 0, 1)))[tis]
    json.dump({"engine": ENGINE, "slab": [c - HALF_Z, c + HALF_Z], "zoom_yx": [y0, x0], "share of tissue with a second lobe": round(share2, 3),
               "classical merged vs standard ST, median angle": round(float(np.median(agree)), 1)}, open(OUT / f"classical_vs_st_{ENGINE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
