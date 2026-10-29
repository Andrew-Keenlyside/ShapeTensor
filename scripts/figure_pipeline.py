"""The pipeline illustrated on a 90° band-crossing phantom, every panel from the chain's own intermediates.

    python scripts/figure_pipeline.py [--out docs/img/pipeline_overview_0.3.png]

Top: the five stages as boxes. Below, left to right: the image with the position read out; the lifted distribution
at that position (raw, deconvolved, the two lobes); the presence map against the null; the share of the crossed band's
direction along its own axis before and after the carry-forward; identities; the co-located vectors in the overlap;
the single field beside the structure tensor; the streamlines of both from the same seeds.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shapetensor import st
from shapetensor.field import Orientations
from shapetensor.phantoms import Grid, build, direction
from shapetensor.populations import peaks
from shapetensor.tracking import track
from shapetensor.validate import tex_for
from shapetensor.viz import dec_rgb, identity_palette

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/img/pipeline_overview_0.3.png")
    a = ap.parse_args()
    tex, texture = tex_for("calibrated"); grid = Grid(core=(192, 96, 192))
    img, tr, meta = build("band_crossing", 1, grid, tex=tex, texture=texture, device=DEV, cross_deg=90.0)
    v = torch.as_tensor(img, device=DEV).float()
    model = Orientations(device=DEV)
    pop_nomom = Orientations(device=DEV, momentum=False).populations(v, labels=False)
    out = model.run(v)
    d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=True, order="zyx", device=DEV)
    zs, ys, xs = grid.inner; ky = (ys.start + ys.stop) // 2                                     # the mid-y slice
    a1, a2 = meta["angles"]; d1, d2 = direction(a1), direction(a2)                              # band 1 along z, band 2 along x
    Z, Y, X = v.shape; step, org = model.step, model.origin
    cz, cx = (Z // 2 - org) // step, (X // 2 - org) // step; cy = (ky - org) // step             # the overlap's centre cell
    g = model.grid.cpu().numpy()
    # --- the lifted distribution at the centre: directions near the x-z plane, as a function of the in-plane angle
    O = out["lift"][:, cz, cy, cx].cpu().numpy(); f = out["fodf"][:, cz, cy, cx].cpu().numpy()
    phi_ip = np.arange(0, 180, 1.0)                                                             # the great circle of the crossing plane, 0 = along x, 90 = along z
    circ = np.stack([np.sin(np.radians(phi_ip)), np.zeros_like(phi_ip), np.cos(np.radians(phi_ip))], 1)    # (z, y, x)
    K = np.exp((np.abs(circ @ g.T) ** 2 - 1) / (2 * math.sin(math.radians(9)) ** 2)); K /= K.sum(1, keepdims=True)   # an axial kernel over the grid
    O_ip, f_ip = K @ O, K @ f
    dirs_c = out["dirs_cells"][:, :, cz, cy, cx].cpu().numpy(); mass_c = out["mass"][:, cz, cy, cx].cpu().numpy()
    angs = [math.degrees(math.atan2(dirs_c[r][0], dirs_c[r][2])) % 180.0 for r in range(dirs_c.shape[0]) if mass_c[r] > 0]
    # --- the share of band 2's direction along band 2's own axis (the line z = Z/2), before and after the carry-forward
    k2 = int(np.argmax(np.abs(g @ d2)))                                                       # the grid direction nearest band 2
    cos2 = np.abs(g @ d2); lobe2 = cos2 >= math.cos(math.radians(20))
    def share_along(fodf):
        F = fodf[:, cz, cy, :]                                                                  # (nd, cells along x)
        return (F[lobe2].sum(0) / F.sum(0).clamp_min(1e-12)).cpu().numpy()
    s_before, s_after = share_along(pop_nomom["fodf"]), share_along(out["fodf"])
    xcells = (org + step * np.arange(s_after.shape[0]) - X / 2) * grid.vox
    band1_half = meta["width_mm"] / 2
    # --- the figure
    fig = plt.figure(figsize=(22, 11.5), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 4.2, 4.2], hspace=0.32, wspace=0.18, left=0.03, right=0.99, top=0.95, bottom=0.05)
    # stage boxes
    axb = fig.add_subplot(gs[0, :]); axb.set_xlim(0, 10); axb.set_ylim(0, 1); axb.axis("off")
    stages = [("1  Lift", "the array as a field over\npositions × directions:\nits lagged autocorrelation on a shell"),
              ("2  Separate", "lobes = populations, N from the data;\nresponse and presence floor\nmeasured on the block"),
              ("3  Carry forward", "each direction smoothed along itself,\nbridged where it holds on both sides,\nnever past a free end"),
              ("4  Identify", "best-aligned neighbours joined,\nnever two lobes of one\nposition in one population"),
              ("5  Field", "every voxel: a weighted single\nfield, and co-located vectors\nwith weights and identities")]
    for i, (title, body) in enumerate(stages):
        x0 = 0.1 + i * 2.0
        axb.add_patch(FancyBboxPatch((x0, 0.05), 1.75, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#eef3fb", ec="#2a5db0", lw=1.2))
        axb.text(x0 + 0.08, 0.8, title, fontsize=12, weight="bold", va="center", color="#1c3f7a")
        axb.text(x0 + 0.08, 0.42, body, fontsize=8.6, va="center", color="#222222")
        if i < 4:
            axb.add_patch(FancyArrowPatch((x0 + 1.77, 0.5), (x0 + 1.98, 0.5), arrowstyle="-|>", mutation_scale=16, color="#2a5db0", lw=1.4))
    # A. the image
    ax = fig.add_subplot(gs[1, 0]); gimg = img[:, ky].astype(np.float32); ax.imshow(gimg, cmap="gray", vmin=40, vmax=220)
    ax.plot([X / 2], [Z / 2], "o", mfc="none", mec="#ffd400", mew=2, ms=14)
    ax.set_title("the array: two fascicles crossing at 90° in texture\n(a 20 µm phantom; the circle marks the position read out)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # B. the lift at the position
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(phi_ip, O_ip - O_ip.min(), color="#888888", lw=1.6, label="the lift ρ(u) − min (raw)")
    ax.plot(phi_ip, f_ip / max(f_ip.max(), 1e-9) * (O_ip - O_ip.min()).max(), color="#2a5db0", lw=2.2, label="deconvolved: the populations")
    for r in range(dirs_c.shape[0]):
        if mass_c[r] > 0:
            ang = math.degrees(math.atan2(dirs_c[r][0], dirs_c[r][2])) % 180.0
            ax.axvline(ang, color="#e34948", lw=1.2, ls="--")
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlabel("direction in the plane of the crossing (°; 0 = along x, 90 = along z)", fontsize=9)
    ax.set_ylabel("correlation above the median direction", fontsize=9); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("1. the lifted distribution at that position: a lobe per population\n(red: the directions read off, " + " and ".join(f"{x:.0f}°" for x in angs) + ")", fontsize=10)
    # C. presence
    ax = fig.add_subplot(gs[1, 2]); p = out["presence"][:, cy, :].cpu().numpy(); fl = out["floor"]
    ax.imshow(p, cmap="magma", vmin=0, vmax=max(float(np.percentile(p, 99)), 2 * fl), interpolation="nearest")
    ax.contour(p, levels=[fl], colors=["#4fd1ff"], linewidths=1.0)
    ax.set_title(f"2. presence per lattice cell (0 to {max(float(np.percentile(p, 99)), 2 * fl):.2f}) and its floor (cyan, {fl:.3f}),\nmeasured on an isotropic surrogate of the block", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # D. carry-forward along band 2
    ax = fig.add_subplot(gs[1, 3])
    ax.axvspan(-band1_half, band1_half, color="#f3d9d9", label="inside the other band")
    ax.plot(xcells, s_before, color="#999999", lw=1.8, label="before the carry-forward")
    ax.plot(xcells, s_after, color="#2a5db0", lw=2.4, label="after")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(0, 1.0); ax.set_xlabel("position along the crossed band (mm; 0 at the crossing)", fontsize=9)
    ax.set_ylabel("share of the band's direction in the distribution", fontsize=9); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("3. the crossed band's own direction along its axis: held\nthrough the other band by its two arms, not extended past an end", fontsize=10)
    # E. identities
    ax = fig.add_subplot(gs[2, 0]); lab = out["labels"][:, :, ky, :].cpu().numpy(); pal = identity_palette(max(int(lab.max()), 1))
    rgb = np.zeros(lab.shape[1:] + (3,), np.float32)
    for r in range(lab.shape[0] - 1, -1, -1):
        m = lab[r] > 0; rgb[m] = pal[lab[r][m]][:, :3]
    ax.imshow(rgb); ax.set_title("4. identities: one number per bundle, through the crossing\n(the second vector's label drawn where there are two)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # F. the co-located vectors in the overlap
    ax = fig.add_subplot(gs[2, 1]); half = 36; z0, x0 = Z // 2 - half, X // 2 - half
    ax.imshow(gimg[z0:z0 + 2 * half, x0:x0 + 2 * half], cmap="gray", vmin=40, vmax=220)
    vecs = out["vecs"][:, :, :, ky, :].cpu().numpy(); w = out["weights"][:, :, ky, :].cpu().numpy(); labs = lab; wvox = out["weight"][:, ky, :].cpu().numpy()
    stp = 6
    for zz in range(z0 + stp // 2, z0 + 2 * half, stp):
        for xx in range(x0 + stp // 2, x0 + 2 * half, stp):
            for r in range(vecs.shape[0]):
                if w[r, zz, xx] <= 0 or wvox[zz, xx] < 0.2:
                    continue
                dz_, dx_ = vecs[r, 0, zz, xx], vecs[r, 2, zz, xx]; L = 0.45 * stp
                col = pal[labs[r, zz, xx]][:3] if labs[r, zz, xx] > 0 else (1, 1, 1)
                ax.plot([xx - x0 - L * dx_, xx - x0 + L * dx_], [zz - z0 - L * dz_, zz - z0 + L * dz_], color=col, lw=1.6)
    ax.set_title("5. the co-located vectors in the overlap (presence weight ≥ 0.2),\ncoloured by identity: a tracker keeps its own bundle", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # G. the single field vs the structure tensor (DEC), side by side in one axis
    ax = fig.add_subplot(gs[2, 2])
    tis = (tr["region"] == 2)[:, :] if "region" in tr else np.ones((Z, X), bool)
    left = (dec_rgb(d_st[:, :, ky, :].cpu().numpy()) * tis[..., None])[zs, xs]
    right = (dec_rgb(out["vec"][:, :, ky, :].cpu().numpy(), out["weight"][:, ky, :].cpu().numpy()) * tis[..., None])[zs, xs]
    both = np.concatenate([left, np.ones((left.shape[0], 6, 3)), right], 1); ax.imshow(both)
    ax.set_title("the single field, DEC: structure tensor (left, the bisector in the overlap)\nand the chain's strongest population with its weight (right)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # H. streamlines
    ax = fig.add_subplot(gs[2, 3]); ax.imshow(gimg, cmap="gray", vmin=40, vmax=220)
    core = torch.zeros(v.shape, dtype=torch.bool, device=DEV); core[zs, ys, xs] = True
    b2 = torch.as_tensor(tr["band2"], device=DEV)[:, None, :].expand(Z, Y, X); b1 = torch.as_tensor(tr["band1"], device=DEV)[:, None, :].expand(Z, Y, X)
    Xm = torch.arange(X, device=DEV).float()[None, None, :].expand(Z, Y, X)
    near = b2 & ~b1 & core & (Xm < X / 2 - 30)
    idx = near.nonzero(); gsel = torch.Generator().manual_seed(1); seeds = idx[torch.randperm(idx.shape[0], generator=gsel)[:40].to(DEV)].float()
    for name, dirs, wts, col in (("structure tensor", d_st[None], None, "#ff5b5b"), ("shape", out["vecs"], out["weights"], "#5bff8a")):
        res = track(dirs, core, seeds, step=0.5, max_angle_deg=30.0, max_steps=2000, weights=wts, min_weight=1e-6 if wts is not None else 0.0)
        pts = res["points"].cpu().numpy()
        for i in range(pts.shape[0]):
            for h_ in (0, 1):
                q = pts[i, h_]; q = q[~np.isnan(q[:, 0])]
                ax.plot(q[:, 2], q[:, 0], color=col, lw=0.9, alpha=0.85)
        ax.plot([], [], color=col, lw=2, label=name)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, X); ax.set_ylim(Z, 0)
    ax.set_title("streamlines from the same seeds in the crossed band:\nthe chain's pass through, the structure tensor's switch band", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("ShapeTensor 0.3: the chain on a 90° band crossing, every panel from the chain's own intermediates", fontsize=14, y=0.985)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=100); print("wrote", a.out)


if __name__ == "__main__":
    main()
