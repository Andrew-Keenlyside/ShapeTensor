"""An image-based explainer: how TCA reads several fibre directions out of shape-space (multichannel) data.

A 60-degree crossing phantom patch -> anatomix layer-58 channels (8 PCs) -> the gradients of every channel inside a
window form a 3-way array X[voxel, channel, axis] -> CP fits it as four rank-1 pieces (spatial map x channel signature
x ONE direction) -> pieces are paired by geometry (most perpendicular pairs); each pair spans the plane of one
population's gradients, and the plane's normal is that population's fibre. CPU only (a small block).

    python scripts/fig_tca_multi.py        (shapetensors env)
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import numpy as np
import torch

from shapetensor.phantoms import make
from shapetensor.features.anatomix import AnatomixEngine
from shapetensor.tensor import gradients
from shapetensor.readout import multi, common

OUT = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # the four pieces
POP = ["#6250d6", "#e34948"]                                # the two populations
torch.set_num_threads(16)


def stick(ax, cx, cz, d, L, color, lw=2.4, ls="-", zorder=5):
    """An axial stick for a (z, y, x) direction, drawn in the x-z plane (length scaled by its in-plane part)."""
    dz, dx = d[0], d[2]
    inpl = np.hypot(dz, dx)
    if inpl < 1e-6:
        return
    ux, uz = dx / inpl, dz / inpl
    ax.plot([cx - ux * L * inpl / 2, cx + ux * L * inpl / 2], [cz - uz * L * inpl / 2, cz + uz * L * inpl / 2],
            color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)


def main():
    tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
    tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(128, 32, 128), margin=(32, 32, 32))
    img, tr, meta = make.build("P4-lines-c60-f1", 3, tex, g, device="cpu")
    vol = torch.as_tensor(img).float()
    eng = AnatomixEngine(layer="l58", device="cpu", amp=False, nyquist="none", tile=128, margin=32, n_pcs=8,
                         l2=True, out_dtype=torch.float32)
    F = eng(vol)                                                      # (8, Z, Y, X)
    G = gradients(F, 1.0, 3.0, out_dtype=torch.float32)               # (24, Z, Y, X)
    Z, Y, X = img.shape
    cz, cy, cx, r, rho = Z // 2, Y // 2, X // 2, 16, 10.0
    sl = (slice(cz - r, cz + r + 1), slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
    zz, yy, xx = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij")
    w = np.exp(-0.5 * (zz ** 2 + yy ** 2 + xx ** 2) / rho ** 2).astype(np.float32)
    Gw = G[(slice(None),) + sl].numpy() * np.sqrt(w)[None]            # (24, n, n, n)
    nC = F.shape[0]
    Xarr = Gw.reshape(nC, 3, -1).transpose(2, 0, 1)                   # (P, C, I)
    Xt = torch.as_tensor(np.ascontiguousarray(Xarr))[None]            # (1, P, C, I)
    Xt = Xt / Xt.norm()
    R = 4
    A, B, D, err = multi.cp_als(Xt, R, None, starts=6, iters=400, seed=1)
    W = multi.block_gram(A, B, D)
    fib, en, msine, best = multi.pair_geometric(D, W)
    pairs = multi.matchings(R)[int(best[0])]
    fib, en = fib[0].numpy(), en[0].numpy()
    Dn = D[0].numpy()                                                 # (3, R)
    An = A[0].numpy().reshape(2 * r + 1, 2 * r + 1, 2 * r + 1, R)
    Bn = B[0].numpy()                                                 # (C, R)
    share = np.diag(W[0].numpy()); share = share / share.sum()
    # the structure tensor on the same window (the single-direction baseline)
    T = np.einsum("cip,cjp->ij", Gw.reshape(nC, 3, -1), Gw.reshape(nC, 3, -1))
    e3 = np.linalg.eigh(T)[1][:, 0]
    tru = [tr["dir1"][cz, cx], tr["dir2"][cz, cx]]
    ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
    errs = [min(ang(fib[q], tru[0]), ang(fib[q], tru[1])) for q in range(2)]
    print(f"CP rel err {float(err[0]):.3f}, pairing min sine {float(msine[0]):.2f}, pairs {pairs}; fibre errors {errs[0]:.1f} / {errs[1]:.1f} deg; "
          f"ST e3 to truth {ang(e3, tru[0]):.0f} / {ang(e3, tru[1]):.0f} deg", flush=True)

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(17, 7.6), dpi=130)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 8, width_ratios=[1.25, 1.1, 0.9, 1, 1, 1, 1, 1.45], height_ratios=[1, 0.8, 1], wspace=0.35, hspace=0.5,
                          left=0.03, right=0.99, top=0.80, bottom=0.08)
    ext = (-r * 0.02, r * 0.02, -r * 0.02, r * 0.02)
    # (a) the image window
    ax = fig.add_subplot(gs[:, 0])
    pad = 24
    ax.imshow(img[cz - r - pad:cz + r + pad + 1, cy, cx - r - pad:cx + r + pad + 1], cmap="gray", origin="lower",
              extent=(-(r + pad) * 0.02, (r + pad) * 0.02, -(r + pad) * 0.02, (r + pad) * 0.02))
    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, ec=INK, lw=1.2, ls="--"))
    for q in range(2):
        stick(ax, 0, 0, tru[q], 0.55, POP[q], lw=2.2, ls=(0, (3, 2)))
    stick(ax, 0, 0, e3, 0.5, MUTED, lw=4, zorder=4)
    ax.set_title("a  image: two fibre populations crossing at 60°\n(dashed = the truth; grey = the structure tensor's e₃, the bisector)", fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    # (b) shape space channels
    sub = gs[:, 1].subgridspec(3, 2, wspace=0.08, hspace=0.12)
    b_axes = []
    for k in range(6):
        ax = fig.add_subplot(sub[k // 2, k % 2]); b_axes.append(ax)
        ax.imshow(F[k][sl][:, r, :].numpy(), cmap="gray", origin="lower", extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.04, 0.9, f"ch {k + 1}", transform=ax.transAxes, fontsize=7.5, color="white", fontweight="bold")
    pb = b_axes[0].get_position()
    fig.text(pb.x0, pb.y1 + 0.015, "b  shape space: anatomix layer 58,\n8 channels (6 shown)", fontsize=9, ha="left", va="bottom")
    # (c) the 3-way array
    ax = fig.add_subplot(gs[:, 2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for k in range(5):
        off = 0.07 * (4 - k)
        ax.add_patch(Rectangle((0.12 + off, 0.20 + off), 0.5, 0.40, fc="#e3eefb" if k else "#cde2fb", ec=C[0], lw=1.1))
    ax.text(0.37, 0.14, "voxels in the window", ha="center", fontsize=8)
    ax.text(0.03, 0.46, "axis (z, y, x)", rotation=90, va="center", fontsize=8)
    ax.text(0.86, 0.80, "channel", rotation=35, fontsize=8)
    ax.text(0.45, 1.0, "X[voxel, channel, axis]\n= gradient of channel c\nalong axis i, at voxel p\n(Gaussian window ρ)", ha="center", va="top", fontsize=8.2)
    pc = ax.get_position()
    fig.text(pc.x0, pc.y1 + 0.015, "c  gradients of every channel,\nkept as a 3-way array", fontsize=9, ha="left", va="bottom")
    # (d) the CP pieces
    for q in range(R):
        col = C[q]
        ax = fig.add_subplot(gs[0, 3 + q])
        amap = An[r, :, :, q] if False else An[:, r, :, q]
        ax.imshow(np.abs(amap), cmap="magma", origin="lower", extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"piece {q + 1}  ({share[q]:.0%} of the fit)", fontsize=8.5, color=col, loc="left")
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(1.6)
        ax = fig.add_subplot(gs[1, 3 + q])
        ax.bar(np.arange(nC) + 1, Bn[:, q], color=col, width=0.7)
        ax.axhline(0, color=GRID, lw=0.8); ax.set_xticks([1, 4, 8]); ax.set_yticks([]); ax.set_xlabel("channel", fontsize=7.5, labelpad=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax = fig.add_subplot(gs[2, 3 + q]); ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis("off")
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=GRID, lw=1))
        d = Dn[:, q]
        ax.add_patch(FancyArrowPatch((0, 0), (d[2], d[0]), arrowstyle="-|>", mutation_scale=14, lw=2.2, color=col))
        ax.add_patch(FancyArrowPatch((0, 0), (-d[2], -d[0]), arrowstyle="-", lw=2.2, color=col, alpha=0.4))
        ax.text(0, -1.18, f"direction d, x–z plane\n(y-component {abs(d[1]):.2f}{': mostly out of plane' if abs(d[1]) > 0.8 else ''})", ha="center", va="top", fontsize=7.2, color=MUTED)
    pd_ = fig.axes[-9].get_position() if False else None
    fig.text(0.475, 0.845, "d  CP (TCA): X ≈ Σ pieces, each = spatial map × channel signature × ONE direction (rank 4 here)", fontsize=9, ha="left", va="bottom")
    # (e) pairing and the fibres
    ax = fig.add_subplot(gs[:, 7]); ax.set_aspect("equal"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.9, 1.35); ax.axis("off")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=GRID, lw=1))
    for q in range(R):
        d = Dn[:, q]
        ax.plot([-d[2], d[2]], [-d[0], d[0]], color=C[q], lw=2.2)
        ax.text(d[2] * 1.14, d[0] * 1.14, f"{q + 1}", color=C[q], fontsize=8.5, ha="center", va="center", fontweight="bold")
    for k, (p1, p2) in enumerate(pairs):
        d1, d2 = Dn[:, p1], Dn[:, p2]
        mid = (d1 + d2) / 2
        ax.annotate("", xy=(d1[2], d1[0]), xytext=(d2[2], d2[0]), arrowprops=dict(arrowstyle="<->", color=POP[k], lw=1.2, ls=":"))
        stick(ax, 0, 0, fib[k], 2.1, POP[k], lw=4.5, zorder=6)
        e = min(ang(fib[k], tru[0]), ang(fib[k], tru[1]))
        ax.text(fib[k][2] * 1.2 + (0.02 if fib[k][2] >= 0 else -0.02), fib[k][0] * 1.2, f"fibre {k + 1}\n{e:.0f}° off", color=POP[k], fontsize=8,
                ha="left" if fib[k][2] >= 0 else "right", va="center")
    ax.text(0, -1.42, "pieces paired by geometry (the most\nperpendicular pairs); each pair spans one\npopulation's gradient plane, and the plane's\nnormal (d₁ × d₂) is that population's fibre",
            ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("e  pair the pieces → fibres", fontsize=9, loc="left")
    fig.suptitle("How TCA reads several fibre directions out of shape-space data\n"
                 "A fibre's gradients fill the plane perpendicular to it. With several channels, the 3-way array of gradients can be split into pieces with ONE direction each; "
                 "pairs of pieces span the planes, and each plane's normal is a fibre.",
                 fontsize=10.5, x=0.03, ha="left", y=0.985)
    fig.text(0.03, 0.015, "Phantom P4-lines-c60-f1 at 20 µm (calibrated texture, 8° dispersion), window ρ 10 voxels; CP rank 4 by ALS; the structure tensor on the same window "
             f"gives the bisector ({ang(e3, tru[0]):.0f}° / {ang(e3, tru[1]):.0f}° from the two truths). For N without a fixed rank the suite uses CSD of the gradient-direction ODF instead.",
             fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tca_multi_explainer.png", facecolor="white")
    fig.savefig(OUT / "tca_multi_explainer.svg", facecolor="white")
    print("wrote", OUT / "tca_multi_explainer.png")


if __name__ == "__main__":
    main()
