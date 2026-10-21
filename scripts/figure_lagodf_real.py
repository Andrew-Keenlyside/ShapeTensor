"""The lag-ODF populations beside the standard structure tensor on the HiP-CT cortex VOI (S-21-46, 20 µm).

A slab of 96 axial slices through the VOI centre, 768 x 768 in-plane. The lag-ODF pathway (``shapetensor.lagodf``,
classical, intensity only) with its response estimated from the slab itself (its most anisotropic single-lobe cells)
and, for comparison, from the synthetic calibration bundles; tissue = above the lower of two multi-Otsu thresholds.
Shown on the slab's middle slice: the image; the standard structure tensor (sigma 1, rho 10) in DEC; the weighted
single field (the strongest population's direction, brightness its weight); the number of populations per 80 µm
cell; the second population's direction where there are two; a zoom with sticks (structure tensor cyan, first
population white, second magenta).

    python scripts/figure_lagodf_real.py [OUT_DIR] [--along 12 --sigma-w 4] [--momentum]
"""
from __future__ import annotations

import argparse
import json
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
from shapetensor.lagodf import LagODF, label_populations, self_response
from shapetensor.viz import dec_rgb, identity_palette, identity_rgb

SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; HALF_Z = 48


def sticks(ax, d, keep, step, color, scale=0.45, lw=0.8):
    H, W = d.shape[1:]
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    k = keep[ys, xs]; L = scale * step
    for y, x, a, b in zip(ys[k], xs[k], d[1][ys, xs][k], d[2][ys, xs][k]):
        ax.plot([x - L * b, x + L * b], [y - L * a, y + L * a], color=color, lw=lw)


def up(a, n=768):
    return np.kron(a, np.ones((4, 4), dtype=a.dtype))[:n, :n] if a.ndim == 2 else np.stack([up(x, n) for x in a])


def synthetic_response(model):
    from shapetensor.phantoms import Grid, build, direction
    from shapetensor.validate import ANGLES
    import importlib.util
    spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    tex, texture = P.tex_for("calibrated"); cg = Grid()
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES)]

    def truth(i, O):
        cs = O.shape[1:]; d = torch.as_tensor(direction(ANGLES[i]), device=DEV).float().view(3, 1, 1, 1).expand(3, *cs)
        m = torch.zeros(cs, dtype=torch.bool, device=DEV); zs, ys, xs = cg.inner
        m[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
        return d, m
    return model.fit_response(vols, truth).response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/lagodf_real")
    ap.add_argument("--sigma-w", type=float, default=10.0); ap.add_argument("--along", type=float, default=0.0)
    ap.add_argument("--momentum", action="store_true"); ap.add_argument("--response", default="self", choices=("self", "synthetic"))
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    z = zarr.open_group(SRC, mode="r")["0"]; c = z.shape[0] // 2
    img = np.asarray(z[c - HALF_Z:c + HALF_Z]).astype(np.uint8)
    from skimage.filters import threshold_multiotsu
    v = torch.as_tensor(img, device=DEV).float(); tissue = v > float(threshold_multiotsu(img[::2, ::4, ::4], classes=3)[0])
    d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=False, order="zyx", device=DEV); log("structure tensor")
    model = LagODF(sigma_w=a.sigma_w, window_along=a.along or None, momentum=a.momentum, device=DEV)
    O = model.odf(v); log("lag odf", tuple(O.shape))
    R_self, n_self = self_response(O, model.grid, bin_deg=model.bin_deg)
    R = R_self if a.response == "self" else synthetic_response(model)
    o = model.populations(v, O=O, response=R); log("populations")
    labels, table = label_populations(o["dirs_cells"], o["mass"], max_angle_deg=20.0, min_fixels=20); log("labels", len(table))
    k = HALF_Z; kc = (k - model.origin) // model.step
    g = img[k].astype(np.float32); g = np.clip((g - np.percentile(g, 1)) / max(np.ptp(np.percentile(g, [1, 99])), 1e-6), 0, 1)
    tis = tissue[k].cpu().numpy(); dst = d_st[:, k].cpu().numpy()
    D = o["dirs_cells"][:, :, kc].cpu().numpy(); M = o["mass"][:, kc].cpu().numpy(); N = o["N"][kc].cpu().numpy()
    W = o["weight"][kc].cpu().numpy(); SH = o["sheet"][kc].cpu().numpy()
    D1, D2 = up(D[0]), up(D[1]); M1, M2 = up(M[0]), up(M[1]); Nf = up(N); Wf = up(W); SHf = up(SH.astype(np.uint8)).astype(bool)
    LAB = labels[:, kc].cpu().numpy(); pal = identity_palette(max(len(table), 1))
    IDf = identity_rgb(LAB, 4, pal, shape=(768, 768)); L1, L2 = up(LAB[0]), up(LAB[1])
    fig, axs = plt.subplots(2, 3, figsize=(19.5, 13.2), facecolor="white")
    axs[0, 0].imshow(np.stack([g] * 3, -1)); axs[0, 0].set_title("HiP-CT image (S-21-46, 20 µm)", fontsize=11)
    axs[0, 0].plot([20, 270], [740, 740], color="yellow", lw=3); axs[0, 0].text(20, 725, "5 mm", color="yellow", fontsize=9)
    axs[0, 1].imshow(dec_rgb(dst) * tis[..., None]); axs[0, 1].set_title("standard structure tensor (σ 1, ρ 10)", fontsize=11)
    axs[0, 2].imshow(IDf * tis[..., None]); axs[0, 2].set_title("lag ODF: populations by identity (a cell with two split on the diagonal)", fontsize=11)
    cmap = ListedColormap(["#222222", "#2a78d6", "#e34948", "#eda100"])
    h = axs[1, 0].imshow(np.where(tis, np.where(SHf, 3, Nf), np.nan), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    cb = plt.colorbar(h, ax=axs[1, 0], fraction=0.046, ticks=[0, 1, 2, 3]); cb.ax.set_yticklabels(["none", "one", "two", "sheet"])
    axs[1, 0].set_title("lag ODF: populations per 80 µm cell", fontsize=11)
    top = table[:15]
    ax = axs[1, 1]; ax.set_facecolor("white")
    ys_ = np.arange(len(top))[::-1]
    ax.barh(ys_, [t["fixels"] for t in top], color=[pal[t["label"]] for t in top])
    for yy, t in zip(ys_, top):
        d = np.abs(np.asarray(t["direction"])); ax.add_patch(plt.Rectangle((-0.09, yy - 0.4), 0.05, 0.8, transform=ax.get_yaxis_transform(), clip_on=False,
                                                                       color=(d[2], d[1], d[0])))
    ax.set_yticks(ys_); ax.set_yticklabels([f"population {t['label']}" for t in top], fontsize=8)
    ax.set_xlabel("fixels in the slab (80 µm cells)"); ax.set_title(f"the 15 largest of {len(table)} populations (swatch: mean direction, DEC)", fontsize=11)
    from scipy import ndimage
    gm = ndimage.gaussian_filter(g, 8); gy, gx = np.gradient(gm); mag = ndimage.uniform_filter(np.hypot(gy, gx) * tis, 192)
    y0, x0 = np.unravel_index(np.argmax(mag[96:-96, 96:-96]), mag[96:-96, 96:-96].shape); y0, x0 = int(y0), int(x0)
    zs = (slice(y0, y0 + 192), slice(x0, x0 + 192))
    ax = axs[1, 2]; ax.imshow(g[zs], cmap="gray"); ax.set_title("zoom: structure tensor (thin white), populations (coloured by identity)", fontsize=11)
    sticks(ax, dst[:, zs[0], zs[1]], tis[zs], 8, "white", lw=0.4)
    for Dk, Lk in ((D1, L1), (D2, L2)):
        for lab in np.unique(Lk[zs][Lk[zs] > 0]):
            sticks(ax, Dk[:, zs[0], zs[1]], (Lk[zs] == lab) & tis[zs], 8, pal[lab], lw=1.4)
    for a2 in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 2]):
        a2.set_xticks([]); a2.set_yticks([])
    for a2 in (axs[0, 1], axs[0, 2], axs[1, 0]):
        a2.add_patch(plt.Rectangle((x0, y0), 192, 192, fill=False, ec="white", lw=1.0))
    tag = f"sw{a.sigma_w}_al{a.along}_{a.response}{'_mom' if a.momentum else ''}"
    fig.suptitle(f"Lag-ODF populations vs the standard structure tensor, S-21-46 20 µm cortex VOI, axial slice z = {c} "
                 f"(σw {a.sigma_w}, along {a.along or '-'}, {a.response} response{', momentum' if a.momentum else ''}); DEC R |x| G |y| B |z|", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); path = OUT / f"lagodf_vs_st_{tag}.png"; fig.savefig(path, dpi=95); log("figure", path)
    Nt = o["N"].cpu().numpy(); tc = tissue[model.origin::model.step, model.origin::model.step, model.origin::model.step].cpu().numpy()[:Nt.shape[0], :Nt.shape[1], :Nt.shape[2]]
    agree = np.degrees(np.arccos(np.clip(np.abs((dst * D1).sum(0)), 0, 1)))[tis & (Nf >= 1)]
    rep = {"settings": vars(a), "self response cells": n_self, "response": [round(float(x), 3) for x in R[::4]],
           "tissue cells by N": {int(n): round(float(((Nt == n) & tc).sum() / max(tc.sum(), 1)), 3) for n in range(3)},
           "tissue cells sheet": round(float((o["sheet"].cpu().numpy() & tc).sum() / max(tc.sum(), 1)), 3),
           "ring fraction q10/50/90 (weight >= min)": [round(float(x), 3) for x in np.quantile(o["ring"].cpu().numpy()[(o["weight"] >= model.min_weight).cpu().numpy() & tc], [0.1, 0.5, 0.9])],
           "first population vs ST median angle (slice, N>=1)": round(float(np.median(agree)), 1) if agree.size else None,
           "populations": len(table), "largest populations": table[:15], "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(OUT / f"lagodf_vs_st_{tag}.json", "w"), indent=1); log(json.dumps(rep))


if __name__ == "__main__":
    main()
