"""Glyph figures from the suite's per-site results: for each phantom, the image with the true directions, then one
panel per arm with the estimated directions as sticks over the image, coloured by their error (light = right),
and for crossing phantoms the CSD fixels. Problem areas show up as dark sticks turning away from the truth.

    python scripts/fig_glyphs.py --tag main --pids P3-40-t0.5 P1-t0.5 --rho 16 --seed 0
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from shapetensor import DATA_ROOT
from shapetensor.phantoms import make
from shapetensor.validate.suite import Spec, result_path

OUT = Path(__file__).resolve().parents[1] / "figures" / "tca_field" / "glyphs"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, GRID = "#0b0b0b", "#e1e0d9"
ARMS = [("single", "intensity", "eigen", "intensity structure tensor"), ("single", "anatomix48", "tca", "anatomix48 TCA R3"),
        ("single", "anatomix16", "tca", "anatomix16 TCA R3"), ("single", "jet", "tca", "jet TCA R3"),
        ("hybrid", "excised-anatomix48", "excise_all", "excised tensor (fix A)")]
POP = ["#6250d6", "#e34948", "#eda100"]


def sticks(ax, cz, cx, d, L, colors, lw=1.3):
    """Axial sticks at (cz, cx) mm for (z, y, x) directions d (n, 3); length by the in-plane part."""
    inpl = np.hypot(d[:, 0], d[:, 2])
    ux, uz = d[:, 2] / np.maximum(inpl, 1e-9), d[:, 0] / np.maximum(inpl, 1e-9)
    h = L * inpl / 2
    segs = np.stack([np.stack([cx - ux * h, cz - uz * h], -1), np.stack([cx + ux * h, cz + uz * h], -1)], 1)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 1.2, capstyle="round"))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--pids", nargs="+", required=True)
    ap.add_argument("--rho", type=float, default=16.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    spec = Spec(pids=tuple(a.pids), tag=a.tag)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    OUT.mkdir(parents=True, exist_ok=True)
    for pid in a.pids:
        img, tr, meta, grid = make.load(pid, a.seed)
        (z0, z1), (x0, x1) = grid.core_zx
        sl_img = img[z0:z1, grid.y_mid, x0:x1]
        vox = grid.vox
        ext = (0, (x1 - x0) * vox, 0, (z1 - z0) * vox)
        panels = []
        for kind, eng, rd, label in ARMS:
            p = result_path(spec, kind, pid, a.seed, eng, a.rho).with_suffix(".npz")
            if not p.exists():
                continue
            with np.load(p) as z:
                if f"dir_{rd}" not in z.files:
                    continue
                panels.append((label, z[f"dir_{rd}"], z[f"err_{rd}"], z["z"], z["x"]))
        multi = None
        pm = result_path(spec, "multi", pid, a.seed, "anatomix48", a.rho).with_suffix(".npz")
        if pm.exists():
            with np.load(pm) as z:
                if "csd_fibres" in z.files:
                    multi = (z["csd_fibres"], z["csd_fracs"], z["csd_count"], z["z"], z["x"])
        if not panels:
            print("no results for", pid); continue
        n = 1 + len(panels) + (1 if multi is not None else 0)
        fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 5.6), dpi=110)
        axs = np.atleast_1d(axs)
        # truth
        ax = axs[0]; ax.imshow(sl_img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=ext)
        zs, xs = panels[0][3], panels[0][4]
        Z, X = np.meshgrid(zs, xs, indexing="ij")
        from shapetensor.validate.score import sample_truth
        t = sample_truth(tr, zs, xs)
        sel = np.zeros(len(zs) * len(xs), bool); sel.reshape(len(zs), len(xs))[::a.every, ::a.every] = True
        cz, cx = (Z.ravel() - z0 + 0.5) * vox, (X.ravel() - x0 + 0.5) * vox
        L = vox * float(np.diff(zs)[0]) * a.every * 0.9
        ok = np.isfinite(t["dir1"][:, 0]) & sel
        sticks(ax, cz[ok], cx[ok], t["dir1"][ok], L, INK, lw=1.1)
        if np.isfinite(t["dir2"][:, 0]).any():
            ok2 = np.isfinite(t["dir2"][:, 0]) & sel
            sticks(ax, cz[ok2], cx[ok2], t["dir2"][ok2], L, "#52514e", lw=1.1)
        ax.set_title(f"{pid}: image + true directions", fontsize=9.5, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        # arms
        for ax, (label, d, err, zs_, xs_) in zip(axs[1:], panels):
            ax.imshow(sl_img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=ext, alpha=0.55)
            e = np.where(np.isfinite(err), err, np.nan)
            ax.imshow(np.ma.masked_invalid(e.reshape(len(zs_), len(xs_))), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=ext, alpha=0.45, interpolation="nearest")
            ok = sel & np.isfinite(err)
            cols = cmap(np.clip(err[ok] / 90.0, 0, 1))
            sticks(ax, cz[ok], cx[ok], d[ok], L, cols)
            med = np.nanmedian(err[np.isfinite(t["dir1"][:, 0])])
            ax.set_title(f"{label}\nmedian {med:.1f}°", fontsize=9.5, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        if multi is not None:
            fib, frac, cnt, zs_, xs_ = multi
            ax = axs[-1]; ax.imshow(sl_img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=ext, alpha=0.55)
            Zm, Xm = np.meshgrid(zs_, xs_, indexing="ij")
            czm, cxm = (Zm.ravel() - z0 + 0.5) * vox, (Xm.ravel() - x0 + 0.5) * vox
            Lm = vox * float(np.diff(zs_)[0]) * 0.9
            for k in range(3):
                ok = frac[:, k] > 0
                sticks(ax, czm[ok], cxm[ok], fib[ok, k], Lm, POP[k], lw=1.0)
            share2 = float((cnt >= 2).mean())
            ax.set_title(f"anatomix48 CSD fixels\n2 populations at {share2:.0%} of sites", fontsize=9.5, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90)); sm.set_array([])
        cb = fig.colorbar(sm, ax=axs[1:1 + len(panels)].tolist(), fraction=0.02, pad=0.01); cb.set_label("angle to the true fibre (deg)")
        fig.suptitle(f"{pid}, seed {a.seed}, ρ {a.rho:g} voxels ({a.rho * vox:.2f} mm): sticks = estimated directions (in-plane part), colour = error; underlay = error map",
                     x=0.01, ha="left", fontsize=10.5)
        out = OUT / f"{pid}_s{a.seed}_rho{a.rho:g}_glyphs.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
