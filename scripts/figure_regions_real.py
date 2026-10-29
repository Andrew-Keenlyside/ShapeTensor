"""The regions model beside the standard structure tensor on the HiP-CT cortex VOI (S-21-46, 20 µm).

A slab of 96 axial slices through the VOI centre, 768 x 768 in-plane. The regions model (``shapetensor.regions``) is
calibrated on the slab itself (its confident cells, as run_volume calibrates a chunk) and on synthetic fibre-free
texture; tissue = above the lower of two multi-Otsu thresholds (the background out, grey and white matter in). Shown on the slab's middle slice: the image; the standard (plain) structure tensor, sigma 1, rho 10, in DEC;
the regions model's direction where it assigns one (texture dark grey, unassigned black); its state map; the agreement
of its neighbours' directions; a zoom with sticks (structure tensor cyan, regions white).

    python scripts/figure_regions_real.py [OUT_DIR]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from matplotlib.colors import ListedColormap

from shapetensor import st
from shapetensor.features import make_engine
from shapetensor.fibre_sets import iso_textures
from shapetensor.regions import ORIENTED, TEXTURE, UNASSIGNED_FIBRE, UNASSIGNED_TEXTURE, RegionModel
from shapetensor.viz import dec_rgb

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/regions_real")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; HALF_Z = 48


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def sticks(ax, d, keep, step, color, scale=0.45, lw=0.8):
    H, W = d.shape[1:]
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    k = keep[ys, xs]; L = scale * step
    for y, x, a, b in zip(ys[k], xs[k], d[1][ys, xs][k], d[2][ys, xs][k]):
        ax.plot([x - L * b, x + L * b], [y - L * a, y + L * a], color=color, lw=lw)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    a = zarr.open_group(SRC, mode="r")["0"]; c = a.shape[0] // 2
    img = np.asarray(a[c - HALF_Z:c + HALF_Z]).astype(np.uint8)
    from skimage.filters import threshold_multiotsu
    v = torch.as_tensor(img, device=DEV).float(); tissue = v > float(threshold_multiotsu(img[::2, ::4, ::4], classes=3)[0])   # background out, grey and white matter in
    d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=False, order="zyx", device=DEV); log("structure tensor")
    eng = make_engine("anatomix_devvit", device=DEV)
    model = RegionModel(per_block=30000).fit(eng, [v], [tissue], iso=iso_textures("calibrated", DEV)); log("fitted", json.dumps(model.report))
    o = model.field(eng, v); a_ = o["lattice"]; log("regions")
    k = HALF_Z; kc = k // 4
    g = img[k].astype(np.float32); g = np.clip((g - np.percentile(g, 1)) / max(np.ptp(np.percentile(g, [1, 99])), 1e-6), 0, 1)
    tis = tissue[k].cpu().numpy()
    dst = d_st[:, k].cpu().numpy(); dreg = o["vec"][:, k].cpu().numpy()
    state = a_["state"][kc].cpu().numpy(); state_f = np.kron(state, np.ones((4, 4), dtype=state.dtype))[:768, :768]
    coh = a_["coherence"][kc].cpu().numpy(); coh_f = np.kron(coh, np.ones((4, 4)))[:768, :768]
    reg = dec_rgb(dreg) * (state_f == ORIENTED)[..., None] + 0.25 * (state_f == TEXTURE)[..., None]
    reg *= tis[..., None]
    fig, axs = plt.subplots(2, 3, figsize=(19.5, 13.2), facecolor="white")
    axs[0, 0].imshow(np.stack([g] * 3, -1)); axs[0, 0].set_title("HiP-CT image (S-21-46, 20 µm)", fontsize=11)
    axs[0, 0].plot([20, 270], [740, 740], color="yellow", lw=3); axs[0, 0].text(20, 725, "5 mm", color="yellow", fontsize=9)
    axs[0, 1].imshow(dec_rgb(dst) * tis[..., None]); axs[0, 1].set_title("standard structure tensor (σ 1, ρ 10)", fontsize=11)
    axs[0, 2].imshow(reg); axs[0, 2].set_title("regions: direction where assigned (texture dark grey, unassigned black)", fontsize=11)
    cmap = ListedColormap(["#555555", "#2a78d6", "#e34948", "#eda100"])
    h = axs[1, 0].imshow(np.where(tis, state_f, np.nan), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    cb = plt.colorbar(h, ax=axs[1, 0], fraction=0.046, ticks=[0, 1, 2, 3]); cb.ax.set_yticklabels(["texture", "oriented", "unassigned,\nfibre-like", "unassigned,\ntexture-like"])
    axs[1, 0].set_title("regions: state per 80 µm cell", fontsize=11)
    h = axs[1, 1].imshow(np.where(tis & (state_f == ORIENTED), coh_f, np.nan), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(h, ax=axs[1, 1], fraction=0.046); axs[1, 1].set_title("regions: agreement of the neighbours' directions", fontsize=11)
    from scipy import ndimage
    gm = ndimage.gaussian_filter(g, 8); gy, gx = np.gradient(gm); mag = ndimage.uniform_filter(np.hypot(gy, gx) * tis, 192)
    y0, x0 = np.unravel_index(np.argmax(mag[96:-96, 96:-96]), mag[96:-96, 96:-96].shape); y0, x0 = int(y0), int(x0)
    zs = (slice(y0, y0 + 192), slice(x0, x0 + 192))
    ax = axs[1, 2]; ax.imshow(g[zs], cmap="gray"); ax.set_title("zoom: structure tensor (cyan), regions (white, where assigned)", fontsize=11)
    sticks(ax, dst[:, zs[0], zs[1]], tis[zs], 8, "#00e0ff"); sticks(ax, dreg[:, zs[0], zs[1]], (state_f[zs] == ORIENTED) & tis[zs], 8, "white")
    for a2 in axs.flat:
        a2.set_xticks([]); a2.set_yticks([])
    for a2 in (axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]):
        a2.add_patch(plt.Rectangle((x0, y0), 192, 192, fill=False, ec="white", lw=1.0))
    fig.suptitle(f"Regions model vs the standard structure tensor, S-21-46 20 µm cortex VOI, axial slice z = {c}; DEC R |x| G |y| B |z|", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); path = OUT / "regions_vs_st.png"; fig.savefig(path, dpi=95); log("figure", path)
    ti = tis & np.isfinite(dreg[0])
    shares = {STATE: round(float(((state_f == s) & tis).sum() / max(tis.sum(), 1)), 3) for s, STATE in ((TEXTURE, "texture"), (ORIENTED, "oriented"),
                                                                                                     (UNASSIGNED_FIBRE, "unassigned fibre-like"), (UNASSIGNED_TEXTURE, "unassigned texture-like"))}
    agree = np.degrees(np.arccos(np.clip(np.abs((dst * dreg).sum(0)), 0, 1)))[ti & (state_f == ORIENTED)]
    json.dump({"model": model.report, "tissue shares": shares, "regions vs standard ST median angle (oriented)": round(float(np.median(agree)), 1) if agree.size else None},
              open(OUT / "regions_vs_st.json", "w"), indent=1)
    log(json.dumps(shares))


if __name__ == "__main__":
    main()
