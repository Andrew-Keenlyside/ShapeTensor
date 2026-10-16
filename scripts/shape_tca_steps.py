"""PROTOTYPE: the shape-space orientation pipeline step by step, on phantoms D0 (no step) and D15 (tract interface,
step 15), grown by 9.6 mm per x and z face and cropped back to the original 20 x 20 mm before display and scoring.

  1 image           intensity, x-z plane (mid y)
  2a shape space    anatomix (whole block, 8 grid phases), RGB = top 3 principal components (D0 and D15 pooled)
  2b shape space    the 16-D feature vectors themselves (6000 voxels of the slice), on the boundary axis (step direction
                    1) and the fibre axis (fibre-component PC 1): the division is the pair of axes; colour = RGB of 2a
  3 boundary        the boundary component: coordinates on the top 3 step directions (PCA of f_D15 - f_D0 within 3 mm
                    of the interface, as shape_window.py)
  4 fibre           the fibre component: the features with the top 4 step directions removed, RGB = its top 3 PCs
  5 eigen           standard eigendecomposition: e3 of the fibre-component shape tensor (rho 16 vox)
  6 TCA all         CP rank 3 of (space x channel x gradient axis) per pixel, all 16 channels; fibre = the direction
                    least represented by the components (null direction of their gradient-axis Gram), nothing dropped
  7 TCA fibre       the same on the 12 fibre-component channels

    python shape_tca_steps.py [rho]     (shapetensors env; uses the padded whole-block features of shape_tca.py --pad)
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import edge_aware_phantom_fig as EPF
import shape_tca as T
import shape_window as SW

T.PAD = T.ST.SP.D_PAD
NAMES = ("D0", "D15")
JY = T.ST.JY


def rgb(P, lo, hi):
    return np.clip((P - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1)


def main(rho=16.0):
    F = {n: T.feats(n, None) for n in ("D0", "D15", "D30")}
    E, ov = SW.model({n: T.crop(f, (1, 3)) for n, f in F.items()})        # step directions (columns, by change energy)
    Qf = E[:, SW.K:].T.astype(np.float32)                                    # (12, 16) fibre component basis
    sl = {n: T.crop(F[n][:, :, JY, :], (1, 2)).reshape(16, -1) for n in NAMES}   # (16, NX * NZ) mid-y slice
    pool = np.concatenate(list(sl.values()), 1); mu = pool.mean(1, keepdims=True)

    def pcs(X, k=3):
        w, V = np.linalg.eigh(np.cov(X - X.mean(1, keepdims=True))); return V[:, np.argsort(w)[::-1][:k]].T
    Vs = pcs(pool); Ps = {n: Vs @ (x - mu) for n, x in sl.items()}
    Pb = {n: E[:, :3].T @ (x - mu) for n, x in sl.items()}
    fib = {n: Qf @ x for n, x in sl.items()}; fpool = np.concatenate(list(fib.values()), 1)
    Vf = pcs(fpool); fm = fpool.mean(1, keepdims=True); Pf = {n: Vf @ (x - fm) for n, x in fib.items()}
    scale = {}
    for nm, D in (("s", Ps), ("b", Pb), ("f", Pf)):
        q = np.concatenate(list(D.values()), 1); scale[nm] = (np.percentile(q, 1, axis=1), np.percentile(q, 99, axis=1))
    # orientations
    ori, sc = {}, {}
    for n in NAMES:
        for chans, Q in (("all", np.eye(16, dtype=np.float32)), ("fibre", Qf)):
            K = T.crop(T.k_slice(np.tensordot(Q, F[n], axes=(1, 0)), rho), (0, 1))
            e, fit, _ = T.readouts(K, np.ones(T.MASK.shape, bool), (3,))
            ori[n, f"eigen_{chans}"] = e["eigen"]; ori[n, f"tca_{chans}"] = e["tca_recon_R3"]
            print(n, chans, "CP rank-3 relative error", round(fit[3], 3), flush=True)
        for k in ("eigen_all", "eigen_fibre", "tca_all", "tca_fibre"):
            sc[n, k] = T.score(ori[n, k])
            print(n, k, sc[n, k]["within_0.5mm"], sc[n, k]["deep_3-8mm"], flush=True)
    # figure
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ); cmap.set_bad("#e1e0d9")
    cols = ("1  image", "2a  anatomix shape space\n(RGB = top 3 PCs)", "2b  shape space: boundary vs fibre axes\n(each dot = one voxel's 16-D feature)", "3  boundary component\n(top 3 step directions)",
            "4  fibre component\n(the rest; RGB = its top 3 PCs)", "5  standard eigendecomposition\n(fibre-component tensor, e3)",
            "6  TCA all\n(all 16 channels, CP rank 3)", "7  TCA fibre\n(fibre component, CP rank 3)")
    fig, axs = plt.subplots(2, 8, figsize=(38, 10.5), dpi=90)
    rng = np.random.default_rng(0); samp = rng.choice(T.ST.SP.NX * T.ST.SP.NZ, 6000, replace=False)
    dep = T.DEP.ravel()[samp]
    lim = [np.percentile(np.concatenate([D[n][0, samp] for n in NAMES]), (0.5, 99.5)) for D in (Pb, Pf)]
    tot = {n: ((sl[n] - sl[n].mean(1, keepdims=True)) ** 2).sum() for n in NAMES}
    share_b = {n: ((E[:, :SW.K].T @ (sl[n] - sl[n].mean(1, keepdims=True))) ** 2).sum() / tot[n] for n in NAMES}
    for r, n in enumerate(NAMES):
        tru, draw, _ = EPF.truth_and_bounds(n)
        EPF.base_image(axs[r, 0], T.volume(n)[:, JY, :], draw)
        ax = axs[r, 2]
        o = np.argsort(dep < 0.5)                                              # interface voxels drawn on top
        ax.scatter(Pb[n][0, samp][o], Pf[n][0, samp][o], c=rgb(Ps[n][:, samp], *scale["s"]).T[o], s=4, lw=0, alpha=0.8)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.grid(alpha=0.3); ax.axvline(0, color="#898781", lw=0.8, ls="--")
        ax.set_xlabel("boundary axis (step direction 1)", fontsize=9); ax.set_ylabel("fibre axis (fibre PC 1)", fontsize=9)
        ax.text(0.02, 0.98, f"variance in the 4 boundary directions: {share_b[n]:.0%}\nin the 12 fibre directions: {1 - share_b[n]:.0%}",
                transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(boxstyle="round", fc="w", alpha=0.85, lw=0))
        for c, (nm, D) in zip((1, 3, 4), (("s", Ps), ("b", Pb), ("f", Pf))):
            img = rgb(D[n], *scale[nm]).T.reshape(T.ST.SP.NX, T.ST.SP.NZ, 3)
            ax = axs[r, c]; ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=EPF.EXT); draw(ax)
            ax.set_xticks([]); ax.set_yticks([])
        for c, k in enumerate(("eigen_fibre", "tca_all", "tca_fibre"), 5):
            ax = axs[r, c]; e = ori[n, k]
            a = np.where((T.XD >= 1) & (T.XD < 19), EPF.angle(e, tru), np.nan)
            im = ax.imshow(np.ma.masked_invalid(a.T), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=EPF.EXT)
            xi, zi = np.meshgrid(np.arange(3, T.ST.SP.NX, 8), np.arange(3, T.ST.SP.NZ, 8), indexing="ij")
            keep = (T.XD[xi, zi] >= 1) & (T.XD[xi, zi] < 19); xi, zi = xi[keep], zi[keep]
            ee = e[xi, zi]; inpl = np.hypot(ee[:, 0], ee[:, 2]); d = ee[:, [0, 2]] / np.maximum(inpl[:, None], 1e-9)
            L = 0.6 * inpl; cx, cz = (xi + 0.5) * EPF.VOX, (zi + 0.5) * EPF.VOX
            for j in range(len(cx)):
                ax.plot([cx[j] - d[j, 0] * L[j] / 2, cx[j] + d[j, 0] * L[j] / 2], [cz[j] - d[j, 1] * L[j] / 2, cz[j] + d[j, 1] * L[j] / 2],
                        color="#0b0b0b", lw=1.1, path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
            draw(ax); ax.set_xlim(EPF.EXT[:2]); ax.set_ylim(EPF.EXT[2:]); ax.set_xticks([]); ax.set_yticks([])
            s = sc[n, k]; EPF.label(ax, f"median error {s['within_0.5mm']:.0f}° within 0.5 mm, {s['deep_3-8mm']:.0f}° at 3-8 mm")
        axs[r, 0].set_ylabel({"D0": "D0\nno step", "D15": "D15\nstep 15"}[n], fontsize=14, rotation=0, labelpad=40)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=11)
    fig.subplots_adjust(top=0.86, wspace=0.3, hspace=0.15, left=0.04, right=0.92)
    cb = fig.colorbar(im, cax=fig.add_axes([0.93, 0.2, 0.006, 0.55])); cb.set_label("angle to the true fibre direction (deg)")
    ea = {n: sc[n, "eigen_all"]["within_0.5mm"] for n in NAMES}
    fig.suptitle(f"PROTOTYPE. Shape-space fibre orientation step by step, D0 and D15 (whole-block anatomix, rho {rho:g} vox = "
                 f"{rho * 0.1:g} mm). Sticks = estimated fibre direction, colour = its error; cyan = interface; fibres cross it at "
                 f"90 / 60 / 30° (left to right).\nFor reference, the standard eigendecomposition on all 16 channels: "
                 f"D0 {ea['D0']:.0f}°, D15 {ea['D15']:.0f}° within 0.5 mm. Phantoms grown 9.6 mm per x and z face, cropped "
                 "before display and scoring. Step directions learned from D15 - D0.", fontsize=12, x=0.01, ha="left", y=0.995)
    out = T.ST.FIGS / "shape_tca_steps.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 16.0)
