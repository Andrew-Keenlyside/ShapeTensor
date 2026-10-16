"""Renders of the surface phantoms: per phantom, the image with the true fibres, BRIDGE's tensor e3 (default, step
excision at rho 0.3 mm, excision at rho 0.6 mm) and the replayed pointer walks of the default and excision builds.
Glyphs and streamline segments are coloured by their angle to the truth (green 0 deg ... red 90 deg; grey = no truth).

    PYTHONPATH=<snap>:. python edge_aware_phantom_fig.py OUT.png
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import edge_aware_builds as EB
import phantom_shell as PS
import surface_phantoms as SP
import tensor_variants as TV

VOX, NX, NY, NZ = SP.VOX, SP.NX, SP.NY, SP.NZ
EXT = (0, NX * VOX, 0, NZ * VOX)
ROWS = (("A", "A: grey/white step, fan crossing it"), ("B", "B: control, no step"),
        ("C", "C: step + U-fibres along it (truth IS parallel)"), ("D15", "D15: tract interface (step 15) crossed at 90/60/30 deg"),
        ("D0", "D0: control, no interface step"))
CMAP = plt.get_cmap("RdYlGn_r")
SCORES = json.loads((EB.OUT / "builds_scores.json").read_text())


def orient(vol, mask, rho, f=None):
    """e3 of the gradient tensor at integration rho (voxels); f: excise gradients whose coarse (sigma 3 vox) or fine
    magnitude exceeds f x its in-mask median, normalising the integration by the surviving weight."""
    g = TV.grads(vol)
    w = np.ones(vol.shape, np.float32)
    if f is not None:
        gc = np.sqrt(sum(ndi.gaussian_filter(vol, 3.0, order=o, mode="nearest", truncate=TV.TR) ** 2
                         for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
        w *= gc <= f * np.median(gc[mask])
        if rho != 3.0:                                                            # the rho-0.6 variant is gmf
            gn = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
            w *= gn <= f * np.median(gn[mask])
    den = np.maximum(TV.integ(w, rho), 1e-6)
    T = TV.assemble([TV.integ(w * g[i] * g[j], rho) / den for i, j in TV.PAIRS], vol.shape)
    return np.linalg.eigh(T)[1][..., :, 0], w


def truth_and_bounds(name):
    """(truth (NX, NZ, 2) or nan, draw(ax), near-surface selector on the slice grid)."""
    X, Z = np.meshgrid((np.arange(NX) + 0.5) * VOX, (np.arange(NZ) + 0.5) * VOX, indexing="ij")
    if name in ("A", "B", "C"):
        t = np.load(PS.ROOT / "truth.npz") if name in ("A", "B") else np.load(SP.ROOT / "C_truth.npz")
        wm, sdf = t["wm"], t["sdf"]
        tru = t["tru"] if name == "C" else None
        if name in ("A", "B"):                                                     # fibres run in WM and GM only
            tru = np.where((wm | t["gm"])[..., None], PS.truth_dir(X, Z), np.nan)

        def draw(ax):
            ax.contour(X, Z, wm.astype(float), [0.5], colors="c", linewidths=0.7)
        near = (sdf >= 0.15) & (sdf <= 1.2) if name == "C" else (sdf >= 0) & (sdf < 0.5)
        return tru, draw, near
    td = np.load(SP.ROOT / "D_truth.npz")

    def draw(ax):
        ax.axhline(SP.D_PLANE_Z, color="c", lw=0.7)
        for x0, _, _ in SP.D_BANDS[1:]:
            ax.axvline(x0, color="w", lw=0.5, ls=":")
    return td["tru"], draw, td["depth"] < 0.5


def angle(d, tru):
    """Angle (deg) of 3D unit d (x, y, z) to an in-slice truth (tx, tz); nan where truth is nan."""
    return np.degrees(np.arccos(np.clip(np.abs(d[..., 0] * tru[..., 0] + d[..., 2] * tru[..., 1]), 0, 1)))


def colours(a):
    c = CMAP(np.nan_to_num(a) / 90.0); c[~np.isfinite(a)] = (0.6, 0.6, 0.6, 1.0)
    return c


def base_image(ax, img, draw):
    lo, hi = np.percentile(img, (0.5, 99.5))
    ax.imshow(img.T, cmap="gray", origin="lower", extent=EXT, vmin=lo, vmax=hi)
    draw(ax); ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_xticks([]); ax.set_yticks([])


def label(ax, text):
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8, va="bottom", color="k",
            bbox=dict(boxstyle="round", fc="w", alpha=0.85, lw=0))


def glyphs(ax, e, mask, tru, st=4, scale=0.36):
    xi, zi = np.meshgrid(np.arange(2, NX, st), np.arange(2, NZ, st), indexing="ij")
    ok = mask[xi, zi]; xi, zi = xi[ok], zi[ok]
    ee = e[xi, zi]; inpl = np.hypot(ee[:, 0], ee[:, 2])
    d = ee[:, [0, 2]] / np.maximum(inpl[:, None], 1e-9); Lg = scale * inpl
    cx, cz = (xi + 0.5) * VOX, (zi + 0.5) * VOX
    segs = np.stack([np.stack([cx - d[:, 0] * Lg / 2, cz - d[:, 1] * Lg / 2], 1),
                     np.stack([cx + d[:, 0] * Lg / 2, cz + d[:, 1] * Lg / 2], 1)], 1)
    ax.add_collection(LineCollection(segs, colors=colours(angle(ee, tru[xi, zi])), linewidths=1.0))


def streamlines(ax, name, variant, tru):
    import anat_lib as L
    rep = L.extract(EB.OUT / "builds" / f"{name}_{variant}", f"ea_{name}_{variant}", EB.AFF)
    P = L.world_of(rep); flat, offs = L.pointer_walk(rep); P = P[flat]
    sid = np.repeat(np.arange(len(offs) - 1), np.diff(offs)); same = sid[1:] == sid[:-1]
    a, b = P[:-1][same], P[1:][same]
    yc = NY * VOX / 2
    ky = (np.abs(a[:, 1] - yc) <= 1.0) & (np.abs(b[:, 1] - yc) <= 1.0)
    a, b = a[ky], b[ky]
    d = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-9)
    m = 0.5 * (a + b); qi = np.clip((m[:, [0, 2]] / VOX).astype(int), 0, [NX - 1, NZ - 1])
    ax.add_collection(LineCollection(np.stack([a[:, [0, 2]], b[:, [0, 2]]], 1), colors=colours(angle(d, tru[qi[:, 0], qi[:, 1]])),
                                     linewidths=0.45, alpha=0.9))
    return len(offs) - 1


def stream_text(name, variant):
    s = SCORES.get(f"{name}:{variant}")
    if s is None:
        return "not built"
    if name in ("A", "B"):
        return (f"enter GM (>1 mm): {s['streamlines_entering_gm_1mm']:.0%}\n"
                f"boundary runs >=5 mm: {s['arc_in_parallel_runs_ge5mm']:.0%} of arc\n"
                f"angle <=1 mm of boundary: {s['angle_median_within_1mm']:.0f} deg")
    if name.startswith("C"):
        return (f"angle in U-fibre band: {s['angle_median_ufibre_band']:.0f} deg\n"
                f"angle deep WM: {s['angle_median_beyond_2mm']:.0f} deg")
    return (f"cross the interface: {s['streamlines_crossing_interface']:.0%}\n"
            f"angle at it, 90/60/30: {s['angle_90deg_within_1mm']:.0f}/{s['angle_60deg_within_1mm']:.0f}/"
            f"{s['angle_30deg_within_1mm']:.0f} deg\nruns along it >=5 mm: {s['arc_in_parallel_runs_ge5mm']:.0%} of arc")


def main(out):
    cols = ("image, true fibres (yellow)", "tensor e3: default (rho 0.3 mm)", "tensor e3: surface cut, rho 0.3 mm",
            "tensor e3: surface cut, rho 0.6 mm", "streamlines: default build", "streamlines: surface-cut build (rho 0.3 mm)")
    fig, axs = plt.subplots(len(ROWS), 6, figsize=(6 * 3.9, len(ROWS) * 4.0), dpi=100)
    for r, (name, title) in enumerate(ROWS):
        root = PS.ROOT if name in ("A", "B") else SP.ROOT
        vol, H = TV.slab(name, root)
        thr = PS.THRESH[name] if name in PS.THRESH else SP.THRESH[name]
        m3 = vol >= thr; img = vol[:, H]; mask = m3[:, H]
        tru, draw, near = truth_and_bounds(name)
        # col 0: image + truth
        ax = axs[r, 0]; base_image(ax, img, draw)
        xi, zi = np.meshgrid(np.arange(4, NX, 10), np.arange(4, NZ, 10), indexing="ij")
        t = tru[xi, zi]; ok = np.isfinite(t).all(-1); cx, cz = (xi[ok] + 0.5) * VOX, (zi[ok] + 0.5) * VOX; t = t[ok]
        ax.add_collection(LineCollection(np.stack([np.stack([cx - t[:, 0] * 0.4, cz - t[:, 1] * 0.4], 1),
                                                   np.stack([cx + t[:, 0] * 0.4, cz + t[:, 1] * 0.4], 1)], 1),
                                         colors="yellow", linewidths=1.1))
        ax.set_ylabel(title, fontsize=10)
        # cols 1-3: tensor glyphs
        for c, (rho, f) in enumerate(((3.0, None), (3.0, 3.0), (6.0, 3.0)), start=1):
            e3, w = orient(vol, m3, rho, f)
            e = e3[:, H]; ax = axs[r, c]; base_image(ax, img, draw)
            if f is not None:
                cut = (w[:, H] == 0) & mask
                ax.imshow(np.ma.masked_where(~cut.T, cut.T.astype(float)), cmap="autumn", alpha=0.3, origin="lower", extent=EXT)
            glyphs(ax, e, mask, tru)
            a = angle(e, tru); sel = mask & near & np.isfinite(a)
            label(ax, f"median angle near surface: {np.median(a[sel]):.0f} deg\n"
                      f"elsewhere: {np.nanmedian(a[mask & ~near]):.0f} deg" + ("\norange = cut gradients" if f else ""))
        # cols 4-5: streamlines
        for c, v in ((4, "base"), (5, "gm3")):
            ax = axs[r, c]; base_image(ax, img, draw)
            try:
                n = streamlines(ax, name, v, tru)
                label(ax, f"{n:,} streamlines; slab y +-1 mm\n" + stream_text(name, v))
            except (IndexError, FileNotFoundError) as ex:
                label(ax, f"no build ({type(ex).__name__})")
        print("row", name, flush=True)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=11)
    fig.suptitle("Surface phantoms (0.1 mm voxels, 20 x 20 mm slice at mid y).  Glyphs and streamline segments coloured by "
                 "angle to the true fibres: green 0 deg, yellow 45, red 90; grey = no truth.  Cyan = the surface.",
                 fontsize=13)
    fig.tight_layout(rect=[0.01, 0, 1, 0.975]); fig.savefig(out); print("wrote", out)
    from PIL import Image
    im = Image.open(out).convert("RGB"); im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    small = str(out).rsplit(".", 1)[0] + "_small.jpg"; im.save(small, quality=88); print("wrote", small)


if __name__ == "__main__":
    main(sys.argv[1])
