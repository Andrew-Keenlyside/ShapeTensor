"""Figures for the shape-tensor prototype (shape_tensor.py): shape space with and without the interface step, the tensor
glyphs, and angle to the true fibres against depth from the interface.

    python shape_tensor.py fig VARIANT ...        (chunkreg env; the variants are the glyph columns)
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import edge_aware_phantom_fig as EPF
import shape_tensor as ST
import surface_phantoms as SP

ROWS = ("D0", "D15", "D30")
B = 12                                                                     # voxels left out at the block faces
TITLES = {"I_r3": "intensity, rho 0.3 mm (BRIDGE now)", "I_r6": "intensity, rho 0.6 mm",
          "F_r6": "shape tensor (anatomix gradients), rho 0.6 mm", "I_k8_r6": "intensity, within 8 shape clusters, rho 0.6",
          "R8_k8_r6": "contrast removed by shape clusters,\nwithin clusters, rho 0.6", "Ro_r6": "contrast removed, oracle split, rho 0.6",
          "F_k8_r6": "shape tensor within 8 shape clusters, rho 0.6", "H_r6": "shape tensor of high-passed features, rho 0.6"}


def shape_space(out):
    F = {n: ST.features(n) for n in ROWS}
    C = F["D0"].shape[0]
    rng = np.random.default_rng(0)
    core = (slice(None), slice(B, -B), slice(B, -B), slice(B, -B))
    X = np.concatenate([f[core].reshape(C, -1)[:, rng.choice(f[core][0].size, 100_000, replace=False)] for f in F.values()], 1)
    mu = X.mean(1, keepdims=True); w, V = np.linalg.eigh(np.cov(X - mu)); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    Bs = V[:, :3].T; Bs *= np.sign(Bs[np.arange(3), np.abs(Bs).argmax(1)])[:, None]
    P = Bs @ (X - mu); lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
    expl = w[:3] / w.sum()

    def rgb(Fs):
        q = Bs @ (Fs.reshape(C, -1) - mu)
        return np.clip((q - lo[:, None]) / (hi - lo)[:, None], 0, 1).T.reshape(Fs.shape[1:] + (3,))

    tru, draw, _ = EPF.truth_and_bounds("D15")
    z = (np.arange(SP.NZ) + 0.5) * SP.VOX; dep = np.abs(z - SP.D_PLANE_Z)
    fig = plt.figure(figsize=(22, 21), dpi=100); gs = fig.add_gridspec(4, 4)
    for r, n in enumerate(ROWS):
        v = ST.volume(n)[:, ST.JY, :]
        ax = fig.add_subplot(gs[r, 0]); EPF.base_image(ax, v, draw); ax.set_ylabel(f"{n}: step {n[1:]}", fontsize=13)
        ax.set_title("intensity (x-z, mid y)" if r == 0 else "", fontsize=11)
        ax = fig.add_subplot(gs[r, 1]); ax.imshow(np.transpose(rgb(F[n][:, :, ST.JY, :]), (1, 0, 2)), origin="lower", extent=EPF.EXT)
        draw(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"anatomix RGB, one basis for D0/D15/D30 (PC1-3 {expl.sum():.0%})" if r == 0 else "", fontsize=11)
        d2, _ = ST.clusters(F[n], 8)
        ax = fig.add_subplot(gs[r, 2]); ax.imshow(d2[:, ST.JY, :].argmin(-1).T, cmap=ListedColormap(plt.get_cmap("tab10").colors[:8]),
                                                  origin="lower", extent=EPF.EXT, interpolation="nearest")
        draw(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("8 shape clusters (k-means per phantom; colours arbitrary)" if r == 0 else "", fontsize=11)
        ax = fig.add_subplot(gs[r, 3])
        if n == "D0":
            gm = np.sqrt(ST.t0_shape(F[n][:, :, ST.JY - 3:ST.JY + 4, :])[:3, :, 3, :].sum(0))
            ax.imshow(gm.T, cmap="magma", origin="lower", extent=EPF.EXT, vmin=0, vmax=np.percentile(gm, 99))
            ax.set_title("D0: shape-gradient magnitude (no step)", fontsize=11)
        else:
            dF = np.linalg.norm(F[n][:, :, ST.JY, :] - F["D0"][:, :, ST.JY, :], axis=0)
            ax.imshow(dF.T, cmap="magma", origin="lower", extent=EPF.EXT, vmin=0, vmax=1.0)
            ax.set_title(f"{n}: feature change from D0, |f - f_D0| (0-1)", fontsize=11)
        draw(ax); ax.set_xticks([]); ax.set_yticks([])
    # profile of the feature change against depth, and the shape space itself
    ax = fig.add_subplot(gs[3, 0]); ys = slice(ST.JY - 20, ST.JY + 21); xs = slice(10, SP.NX - 10)
    for n, c in (("D8", "C2"), ("D15", "C1"), ("D30", "C3")):
        f = ST.features(n) if n not in F else F[n]
        prof = np.linalg.norm(f[:, xs, ys, :] - F["D0"][:, xs, ys, :], axis=0).mean((0, 1))
        ax.plot(z - SP.D_PLANE_Z, prof, color=c, label=f"{n} vs D0")
    ax.axvline(0, color="c", lw=0.8); ax.set_xlabel("z from the interface (mm)"); ax.set_ylabel("mean |f - f_D0|")
    ax.set_title("how far the step reaches into shape space\n(floor = different noise draws)", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    labs = np.digitize(dep, [0.5, 1.0, 2.0]); cols = ("#e41a1c", "#ff7f00", "#377eb8", "#4daf4a")
    names = ("< 0.5 mm", "0.5-1 mm", "1-2 mm", "> 2 mm")
    for c_, n in enumerate(("D0", "D15", "D30"), start=1):
        f = F[n][:, B:-B:3, ST.JY - 20:ST.JY + 21:4, B:-B:2]
        q = Bs @ (f.reshape(C, -1) - mu); lz = np.broadcast_to(labs[None, None, B:-B:2], f.shape[1:]).ravel()
        ax = fig.add_subplot(gs[3, c_])
        for k in range(4):
            s = lz == k; ax.scatter(q[0, s], q[1, s], s=2, alpha=0.5, lw=0, c=cols[k], label=names[k])
        ax.set_xlim(lo[0] - 0.05, hi[0] + 0.05); ax.set_ylim(lo[1] - 0.05, hi[1] + 0.05)
        ax.set_xlabel(f"PC1 ({expl[0]:.0%})"); ax.set_ylabel(f"PC2 ({expl[1]:.0%})"); ax.grid(alpha=0.3)
        ax.set_title(f"{n}: shape space (shared basis), by distance from the interface", fontsize=11)
        if c_ == 1:
            ax.legend(markerscale=4, fontsize=9)
    fig.suptitle("Shape space with and without the tract-interface step (phantoms D0 / D15 / D30: identical fibres, "
                 "upper half brighter by 0 / 15 / 30)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(out / "shape_space_with_without_step.png"); plt.close(fig)


def glyph_fig(variants, out):
    tru, draw, near = EPF.truth_and_bounds("D15")
    fig, axs = plt.subplots(len(ROWS), len(variants), figsize=(4.3 * len(variants), 4.5 * len(ROWS)), dpi=100, squeeze=False)
    for r, n in enumerate(ROWS):
        cache = {}; v = ST.volume(n)[:, ST.JY, :]; mask = np.ones_like(v, bool)
        for c, var in enumerate(variants):
            e, _ = ST.tensor(n, var, cache); ax = axs[r, c]
            EPF.base_image(ax, v, draw); EPF.glyphs(ax, e, mask, tru)
            sc, _ = ST.score_e3(e)
            EPF.label(ax, f"within 0.5 mm: {sc['within_0.5mm']:.0f} deg\n3-8 mm away: {sc['deep_3-8mm']:.0f} deg")
            if r == 0:
                ax.set_title(TITLES.get(var, var), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{n}: step {n[1:]}", fontsize=12)
    fig.suptitle("Tensor e3 on the tract-interface phantoms; colour = angle to the true fibres (green 0, yellow 45, red 90 deg). "
                 "Cyan = interface; bands cross it at 90 / 60 / 30 deg", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "shape_tensor_glyphs.png"); plt.close(fig)


def depth_fig(variants, out):
    sc = json.loads((ST.OUT / "scores.json").read_text())
    ref = json.loads(Path("/home/andrew/bridge_wmcov/edge_aware/tensor_variants.json").read_text())
    mids = np.arange(len(ST.BINS))                                           # depth bins, evenly spaced
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=100, sharey=True)
    for r, n in enumerate(("D15", "D30")):
        for c, band in enumerate(("90deg", "60deg", "30deg")):
            ax = axs[r, c]
            for k, var in enumerate(variants):
                ax.plot(mids, sc[var][n][band], "-o", ms=4, label=TITLES.get(var, var).replace("\n", " "), color=f"C{k}")
            ax.plot(mids, ref["gm3"][n][band], "--", color="k", label="gradient excision (gm3, fix A), rho 0.3")
            ax.plot(mids, sc["I_r3"]["D0"][band], ":", color="grey", label="BRIDGE now on D0 (no step)")
            ax.set_xticks(mids); ax.set_xticklabels(["0-0.25", "0.25-0.5", "0.5-1", "1-2", "3-8"])
            ax.set_title(f"{n}, fibres at {band[:-3]} deg to the interface", fontsize=11); ax.grid(alpha=0.3)
            ax.set_xlabel("distance from the interface (mm)")
            if c == 0:
                ax.set_ylabel("median angle of e3 to the true fibres (deg)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Angle to the true fibres against distance from the tract interface", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "shape_tensor_depth.png"); plt.close(fig)


def main(variants):
    out = ST.FIGS; out.mkdir(parents=True, exist_ok=True)
    shape_space(out); print("shape space", flush=True)
    glyph_fig(variants, out); print("glyphs", flush=True)
    depth_fig(variants, out); print("wrote", out, flush=True)
