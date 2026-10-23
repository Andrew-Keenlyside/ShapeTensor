"""Glyphs of the pilot models' output on real HiP-CT (S-21-46 20 µm cortex VOI), one glyph per 4³ output cell.

    python scripts/learn/real_glyphs.py RUN_DIR [RUN_DIR ...] [--zoom 128] [--region boundary|vessel|Y,X] [--out FILE]

On a zoom of the middle axial slice, over the image, per model (one row), all read from the model output by the engine:

* **population sticks**: each line population present in the cell, its axis projected on the slice (a through-plane
  population is short), length ∝ its mass, DEC colour of its 3-D axis;
* **orientation density, in-plane cut**: the populations' Watson lobes (kappa 20, each normalised over the sphere,
  weighted by mass) evaluated on the directions in the slice -- a polar glyph per cell, its shape normalised to the
  cell's maximum and its size ∝ √(the cell's line mass);
* **mixture tensor**: the mixture head's tensor, its in-plane 2x2 block as an ellipse (semi-axes = its eigenvalues of
  the trace-1 tensor: an in-plane line is a bar, isotropic a small disc), opacity by the cell's total mass.

Top row: the slice with the zoom box, and the structure tensor's axis (gradient sigma 1, integration rho 5 voxels)
as sticks at the cell centres.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import torch
import zarr
from scipy import ndimage

from shapetensor import st
from shapetensor.learn.engine import PopulationEngine
from shapetensor.lift import hemisphere
from shapetensor.viz import dec_rgb

SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; HALF_Z = 48; KAPPA = 20.0; R = 1.9                                                  # glyph half-size (voxels; cells are 4)


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def watson_cut(T: torch.Tensor, m: torch.Tensor, phi: torch.Tensor, sphere: torch.Tensor) -> torch.Tensor:
    """In-plane cut of the populations' density: tensors ``T (P, 6, n)``, masses ``(P, n)``, slice directions at angles
    ``phi`` (u = (0, sin, cos) in z, y, x), each lobe ``exp(kappa (uᵀTu - 1))`` normalised over ``sphere``; ``(A, n)``."""
    def q(u):                                                                                      # uᵀTu for u (n_u, 3): (n_u, P, n)
        zz, yy, xx, zy, zx, yx = T.unbind(1)
        uz, uy, ux = (c[:, None, None] for c in u.unbind(1))
        return uz * uz * zz + uy * uy * yy + ux * ux * xx + 2 * (uz * uy * zy + uz * ux * zx + uy * ux * yx)
    norm = torch.exp(KAPPA * (q(sphere) - 1)).mean(0)                                              # (P, n)
    u = torch.stack([torch.zeros_like(phi), torch.sin(phi), torch.cos(phi)], 1)
    return (torch.exp(KAPPA * (q(u) - 1)) / norm.clamp_min(1e-12) * m).sum(1)                      # (A, n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--zoom", type=int, default=128); ap.add_argument("--out")
    ap.add_argument("--region", default="boundary", help="boundary (largest smoothed-intensity edge), vessel (most bright pixels inside the large dark region) or Y,X (zoom centre)")
    a = ap.parse_args(); ZW = a.zoom
    arr = zarr.open_group(SRC, mode="r")["0"]; c = arr.shape[0] // 2
    img = np.asarray(arr[c - HALF_Z:c + HALF_Z]).astype(np.uint8)
    v = torch.as_tensor(img, device=DEV).float()
    tissue = img > otsu(img); win = tuple(float(q) for q in np.quantile(img[tissue], [0.005, 0.995]))
    k = HALF_Z; kc = k // 4
    g = np.clip((img[k].astype(np.float32) - win[0]) / (win[1] - win[0]), 0, 1); tis = tissue[k]
    h2 = ZW // 2
    if "," in a.region:
        cy, cx = (int(t) for t in a.region.split(","))
    else:
        if a.region == "vessel":                                                                   # bright thin structures in the dark region
            dark = ndimage.gaussian_filter(g, 8) < 0.2                                             # the large dark region
            score = ndimage.uniform_filter((dark & (g > 0.5)).astype(np.float32), ZW)
        else:                                                                                      # the boundary between bright and dark tissue
            score = ndimage.uniform_filter(np.hypot(*np.gradient(ndimage.gaussian_filter(g, 8))), ZW)
        score[:h2] = score[-h2:] = 0; score[:, :h2] = score[:, -h2:] = 0
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
    y0, x0 = (min(max(int(c_) - h2, 0), n_ - ZW) // 4 * 4 for c_, n_ in ((cy, g.shape[0]), (cx, g.shape[1])))
    iy, ix = np.mgrid[y0 // 4:(y0 + ZW) // 4, x0 // 4:(x0 + ZW) // 4]                               # cells in the zoom
    cy0, cx0 = max(0, y0 - 64), max(0, x0 - 64)                                                      # a crop with 64 voxels of context
    vc = v[:, cy0:min(g.shape[0], y0 + ZW + 64), cx0:min(g.shape[1], x0 + ZW + 64)]
    vc = vc[:, :vc.shape[1] // 4 * 4, :vc.shape[2] // 4 * 4]
    iyc, ixc = iy - cy0 // 4, ix - cx0 // 4
    py, px = 4 * iy + 1.5 - y0, 4 * ix + 1.5 - x0                                                  # their centres in zoom pixels
    tis_c = ndimage.zoom(tis.astype(np.float32), 0.25, order=1)[iy, ix] > 0.5
    extent = (-0.5, ZW - 0.5, ZW - 0.5, -0.5)
    under = lambda ax: ax.imshow(g[y0:y0 + ZW, x0:x0 + ZW], cmap="gray", vmin=0, vmax=1, extent=extent, alpha=0.75)
    phi = torch.linspace(0, 2 * np.pi, 73, device=DEV)[:-1]; sphere = hemisphere(400, DEV).float()

    def sticks(ax, d, w, colour=None, lw=1.6):
        """``d (3, ...)`` in z, y, x; ``w`` the half-length factor in [0, 1]."""
        keep = w > 0.02
        dy, dx = d[1][keep], d[2][keep]; L = R * w[keep]
        seg = np.stack([np.stack([px[keep] - L * dx, py[keep] - L * dy], -1), np.stack([px[keep] + L * dx, py[keep] + L * dy], -1)], 1)
        col = dec_rgb(d[:, keep]) if colour is None else colour
        ax.add_collection(LineCollection(seg, colors=col, linewidths=lw))

    nr = len(a.runs) + 1
    fig, axs = plt.subplots(nr, 3, figsize=(21, 7.2 * nr), facecolor="white", squeeze=False)
    ax = axs[0, 0]; ax.imshow(g, cmap="gray", vmin=0, vmax=1); ax.add_patch(plt.Rectangle((x0, y0), ZW, ZW, fill=False, ec="yellow", lw=2))
    ax.set_title(f"HiP-CT S-21-46 cortex VOI, 20 µm, axial slice; zoom {ZW}² voxels ({ZW * 0.02:.2f} mm)", fontsize=10)
    ax = axs[0, 1]; ax.imshow(g[y0:y0 + ZW, x0:x0 + ZW], cmap="gray", vmin=0, vmax=1, extent=extent); ax.set_title("zoom: image", fontsize=10)
    d_st = st.structure_tensor(v, sigma=1.0, rho=5.0, truncate=3.0, excise=False, order="zyx", device=DEV)[:, k].cpu().numpy()
    dst = d_st[:, 4 * iy + 2, 4 * ix + 2]
    ax = axs[0, 2]; under(ax); sticks(ax, dst, tis_c.astype(np.float32) * 0.9); ax.set_title("structure tensor axis (σ 1, ρ 5) at the cell centres", fontsize=10)
    for r, run in enumerate(a.runs, start=1):
        eng = PopulationEngine(str(Path(run) / "model.pt"), device=DEV); step = torch.load(Path(run) / "model.pt", map_location="cpu")["step"]
        with torch.no_grad():
            net = eng.network(vc, win); pop = eng.populations(net, labels=False)                    # on the crop around the zoom
        D = pop["dirs_cells"][:, :, kc].cpu().numpy()[:, :, iyc, ixc]                                 # (P, 3, h, w)
        M = pop["mass"][:, kc].cpu().numpy()[:, iyc, ixc]                                             # (P, h, w)
        mref = float(np.quantile(M[M > 0], 0.95)) if (M > 0).any() else 1.0
        name = Path(run).name
        # population sticks
        ax = axs[r, 0]; under(ax)
        for p in range(M.shape[0]):
            sticks(ax, D[p], np.clip(M[p] / mref, 0, 1))
        npop = (M > 0).sum(0)
        ax.set_title(f"{name} (step {step}): line populations as sticks (length ∝ mass)\n"
                     f"cells in tissue with 0/1/2/3+ populations: {', '.join(f'{(npop[tis_c] == n).mean():.0%}' for n in (0, 1, 2))}, {(npop[tis_c] >= 3).mean():.0%}", fontsize=10)
        # density cut
        T = pop["line_tensors"][:, :, kc][:, :, iyc, ixc].reshape(M.shape[0], 6, -1)
        cut = watson_cut(T.to(DEV).float(), torch.as_tensor(M.reshape(M.shape[0], -1), device=DEV), phi, sphere).cpu().numpy()   # (A, n)
        mline = M.sum(0).reshape(-1)
        size = np.sqrt(np.clip(mline / mref, 0, 1))                                                  # glyph size ∝ √(line mass)
        rad = cut / np.maximum(cut.max(0, keepdims=True), 1e-12) * R * size[None]
        ph = phi.cpu().numpy()[:, None]
        polys = np.stack([px.reshape(1, -1) + rad * np.cos(ph), py.reshape(1, -1) + rad * np.sin(ph)], -1).transpose(1, 0, 2)   # (n, A, 2)
        keep = size > 0.05
        cols = dec_rgb(D[0].reshape(3, -1))
        ax = axs[r, 1]; under(ax)
        ax.add_collection(PolyCollection(polys[keep], facecolors=np.concatenate([cols[keep], np.full((int(keep.sum()), 1), 0.35)], 1), edgecolors=cols[keep], linewidths=1.2))
        ax.set_title("orientation density of the populations, cut in the slice plane\n(per-cell shape, size ∝ √ line mass)", fontsize=10)
        # mixture tensor ellipses
        mix = net["mixture"][:, kc].cpu().numpy()[:, iyc, ixc]                                         # (6, h, w): zz yy xx zy zx yx
        B = np.stack([np.stack([mix[1], mix[5]], -1), np.stack([mix[5], mix[2]], -1)], -2)            # in-plane (y, x) block
        lam, vec = np.linalg.eigh(B.reshape(-1, 2, 2))                                               # ascending
        lam = np.clip(lam, 0, None)
        Mfull = np.stack([np.stack([mix[0], mix[3], mix[4]], -1), np.stack([mix[3], mix[1], mix[5]], -1), np.stack([mix[4], mix[5], mix[2]], -1)], -2)
        _, V3 = np.linalg.eigh(Mfull.reshape(-1, 3, 3)); ax3 = V3[:, :, 2].T                          # principal 3-D axis (3, n)
        th = np.linspace(0, 2 * np.pi, 40)[None, :, None]
        e1 = vec[:, :, 1][:, None]; e2 = vec[:, :, 0][:, None]                                       # (n, 1, 2) in (y, x)
        pts = (R * lam[:, 1][:, None, None] * np.cos(th) * e1 + R * lam[:, 0][:, None, None] * np.sin(th) * e2)   # (n, 40, 2) (y, x)
        polys = np.stack([px.reshape(-1, 1) + pts[..., 1] * 2, py.reshape(-1, 1) + pts[..., 0] * 2], -1)   # ×2: semi-axis = 2R λ (λ ≤ 1)
        mtot = pop["slot_mass"][:, kc].sum(0).cpu().numpy()[iyc, ixc].reshape(-1)
        alpha = 0.25 + 0.75 * np.clip(mtot / max(float(np.quantile(mtot[mtot > 0], 0.95)) if (mtot > 0).any() else 1.0, 1e-9), 0, 1)
        cols = np.concatenate([dec_rgb(ax3), alpha[:, None]], 1)
        keep = tis_c.reshape(-1)
        ax = axs[r, 2]; under(ax); ax.add_collection(PolyCollection(polys[keep], facecolors=cols[keep], edgecolors="none"))
        ax.set_title("mixture tensor, in-plane ellipse (bar = in-plane line, disc = isotropic); opacity ∝ total mass", fontsize=10)
        del net, pop; torch.cuda.empty_cache()
        print(name, "done", flush=True)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[:, 1:].ravel():
        ax.set_xlim(-0.5, ZW - 0.5); ax.set_ylim(ZW - 0.5, -0.5)
    fig.suptitle("Glyphs from the pilot models' output on real HiP-CT (synthetic pilot training only); DEC colour: red x, green y, blue z (through-plane)", fontsize=13)
    tag = a.region.replace(",", "_")
    out = Path(a.out) if a.out else Path(f"/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_pilot/real_hipct_glyphs_{tag}.png")
    plt.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(out, dpi=70); print("wrote", out)


if __name__ == "__main__":
    main()
