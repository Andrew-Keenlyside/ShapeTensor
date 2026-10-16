"""PROTOTYPE: the orientation field on a sub-block of the real 20 um S-21-46 crop (internal capsule / thalamus), as a
qualitative demonstration (no truth; user decision 2026-10-16): the structure tensor, the excised tensor (fix A) and
the CSD N-population map on the mid planes, with the boundary strength, the sheet dominance and the orientation gap.

    python scripts/demo_real_crop.py [--rho 16] [--core 384] [--reduce 4] [--shape anatomix48] [--width 20]
        (shapetensors env, CUDA_VISIBLE_DEVICES=0)

Writes figures/tca_field/real_S2146_*.png and the field store under /home/andrew/bridge_wmcov/shapetensor/fields/.
"""
import argparse, json, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
import zarr

from shapetensor import DATA_ROOT
from shapetensor.field import compute_field, write_store

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def sticks(ax, d, ext, step, scale, color, plane, lw=1.2):
    nz, nx = d.shape[1:]
    zi, xi = np.meshgrid(np.arange(step // 2, nz, step), np.arange(step // 2, nx, step), indexing="ij")
    dz, dx = d[plane[0], zi, xi], d[plane[1], zi, xi]
    inpl = np.hypot(dz, dx)
    cz = ext[2] + (zi + 0.5) * (ext[3] - ext[2]) / nz
    cx = ext[0] + (xi + 0.5) * (ext[1] - ext[0]) / nx
    L = scale * inpl
    ux, uz = dx / np.maximum(inpl, 1e-9), dz / np.maximum(inpl, 1e-9)
    segs = np.stack([np.stack([cx - ux * L / 2, cz - uz * L / 2], -1), np.stack([cx + ux * L / 2, cz + uz * L / 2], -1)], 1).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 1.4))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=16.0); ap.add_argument("--core", type=int, default=384)
    ap.add_argument("--reduce", type=int, default=4); ap.add_argument("--shape", default=None)
    ap.add_argument("--width", type=float, default=20.0); ap.add_argument("--peak-rel", type=float, default=0.3)
    a = ap.parse_args()
    t0 = time.time()
    z0 = zarr.open(CROP, mode="r")["0"]
    n = z0.shape[0]
    m = int(3 * a.rho + 0.5) + (64 if a.shape else 0)
    m = int(np.ceil(m / a.reduce) * a.reduce)
    lo = (n - a.core) // 2 - m
    hi = lo + a.core + 2 * m
    vol = np.asarray(z0[lo:hi, lo:hi, lo:hi]).astype(np.float32)
    print(f"block {vol.shape} read in {time.time() - t0:.0f} s (core {a.core}, margin {m})", flush=True)
    field = compute_field(torch.as_tensor(vol, device="cuda"), rho=a.rho, reduce=a.reduce, margin=m, csd_width_deg=a.width,
                          peak_rel=a.peak_rel, shape_engine=a.shape, engine_kwargs={"nyquist": "none"} if a.shape else None)
    print(f"field {field['shape']} computed in {time.time() - t0:.0f} s; excised share {float(field['excised_share'].mean()):.3%}; "
          f"npop shares {[round(float((field['npop'] == k).mean()), 3) for k in range(4)]}", flush=True)
    store = Path(DATA_ROOT) / "fields" / f"S2146_core{a.core}_rho{a.rho:g}{'_' + a.shape if a.shape else ''}.zarr"
    write_store(store, field, 0.02, block_origin_vox=(lo, lo, lo))
    print("wrote", store, flush=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    ncm = ListedColormap(["#ffffff", "#cde2fb", "#6da7ec", "#184f95"])
    (cz0, cz1), (cy0, cy1), (cx0, cx1) = field["core"]
    for pname, axis in (("axial_zconst", 0), ("coronal_yconst", 1), ("sagittal_xconst", 2)):
        mid = field["shape"][axis] // 2
        sl = [slice(None)] * 3; sl[axis] = mid; sl = tuple(sl)
        core_sl = [slice(cz0, cz1), slice(cy0, cy1), slice(cx0, cx1)]
        core_sl[axis] = (cz0, cy0, cx0)[axis] + mid * a.reduce + a.reduce // 2
        img = vol[tuple(core_sl)]
        keep = [k for k in range(3) if k != axis]
        ext = (0, img.shape[1] * 0.02, 0, img.shape[0] * 0.02)
        step = max(1, 12 // a.reduce); scale = 0.02 * a.reduce * step * 0.75
        panels = [("dir_st", "intensity structure tensor", CAT[0]), ("dir", "excised tensor (fix A)", CAT[1])]
        if a.shape:
            panels.append(("dir_shape", f"{a.shape} TCA R3", CAT[2]))
        fig, axs = plt.subplots(2, 4, figsize=(21, 10.5), dpi=100)
        for ax, (key, title, col) in zip(axs[0], panels):
            ax.imshow(img, cmap="gray", origin="lower", extent=ext)
            sticks(ax, field[key][(slice(None),) + sl], ext, step, scale, col, (keep[0], keep[1]))
            ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax = axs[0, len(panels)]
        ax.imshow(field["npop"][sl], cmap=ncm, vmin=-0.5, vmax=3.5, origin="lower", extent=ext, interpolation="nearest")
        for k, col in ((0, CAT[0]), (1, CAT[1]), (2, CAT[2])):
            d = field["dirs"][k][(slice(None),) + sl]
            frac = field["fracs"][k][sl]
            d = d * (frac > 0)[None]
            sticks(ax, d, ext, step, scale, col, (keep[0], keep[1]), lw=1.0)
        ax.set_title("CSD N-population map (white 0, light 1, mid 2, dark 3) + fixels", fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
        for ax, (key, title, vmax) in zip(axs[1], (("gm", "boundary strength: coarse gradient magnitude", None), ("cl", "sheet dominance (Westin cl)", 0.6),
                                                   ("gap", "orientation gap (λ2−λ3)/λ2, excised tensor", 1.0), ("gap_st", "orientation gap, plain tensor", 1.0))):
            im = ax.imshow(field[key][sl], cmap=cmap, origin="lower", extent=ext, vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        for ax in axs.ravel():
            ax.grid(False)
        fig.suptitle(f"PROTOTYPE. S-21-46 20 um crop, {a.core}^3 core, {pname}: ρ {a.rho:g} vox ({a.rho * 0.02:.2f} mm), sites every {a.reduce} vox. "
                     "Sticks = in-plane direction (short = out of plane). No ground truth: a consistency demonstration only.",
                     fontsize=10.5, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIGS / f"real_S2146_rho{a.rho:g}_{pname}.png"
        fig.savefig(out); plt.close(fig)
        print("wrote", out, flush=True)
    json.dump({"rho": a.rho, "core": a.core, "reduce": a.reduce, "block": [lo, hi], "csd_width_deg": a.width, "peak_rel": a.peak_rel,
               "excised_share": float(field["excised_share"].mean()), "npop_shares": [float((field["npop"] == k).mean()) for k in range(4)],
               "elapsed_s": time.time() - t0}, open(FIGS / f"real_S2146_rho{a.rho:g}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
