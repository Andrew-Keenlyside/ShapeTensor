"""A visualisation test: per population, the layer's direction as DEC colour (R = |x|, G = |y|, B = |z|) with the
activation magnitude as the overlay's opacity, over the grey intensity image; and all populations together.
Reads an exported population field (readout.layers.export_population_field) and rebuilds / reloads the image.

    python scripts/fig_dec_overlay.py --field phantom_3pop   -> figures/tca_field/dec_overlay_phantom_3pop.png
    python scripts/fig_dec_overlay.py --field S2146_block192 -> figures/tca_field/dec_overlay_S2146_block192.png
"""
import argparse, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor import DATA_ROOT
from shapetensor.phantoms import make
from shapetensor.viz import dec_activation_overlay

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"


def dec(d):
    """``d (3, ...)`` in (z, y, x) -> RGB with R = |x|, G = |y|, B = |z|."""
    return np.clip(np.stack([np.abs(d[2]), np.abs(d[1]), np.abs(d[0])], -1), 0, 1)


def phantom_image(angles, seed):
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32))
    rng = np.random.default_rng(seed); sp = make.Splat(g, tex, rng, "cuda")
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    return make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--field", default="phantom_3pop"); ap.add_argument("--gamma", type=float, default=0.7); ap.add_argument("--mode", default="brightness", choices=["brightness", "saturation", "opacity"])
    a = ap.parse_args()
    if a.field == "phantom_3pop":
        g = zarr.open_group(str(Path(DATA_ROOT) / "fields" / "phantom_3pop_population_field.zarr"), mode="r")
        img = phantom_image([(0.0, 1 / 3), (60.0, 1 / 3), (120.0, 1 / 3)], 304); title = "3-population phantom (0 / 60 / 120°)"
        crop = (slice(36, 124), slice(36, 124)); y = 48
    else:
        g = zarr.open_group(str(Path(DATA_ROOT) / "fields" / "S2146_population_field_block192.zarr"), mode="r")
        z0 = zarr.open(CROP, mode="r")["0"]; o = (z0.shape[0] - 384) // 2 + (384 - 192) // 2
        img = np.asarray(z0[o:o + 192, o:o + 192, o:o + 192]).astype(np.float32); title = "S-21-46 crop, 192³ block"
        crop = (slice(24, 168), slice(24, 168)); y = 96
    A = np.asarray(g["activation"]).astype(np.float32); D = np.asarray(g["dirs"]).astype(np.float32); N = A.shape[0]
    masks = np.asarray(g["masks"])
    im = img[crop[0], y, crop[1]]
    fig, axs = plt.subplots(1, N + 2, figsize=(6.2 * (N + 2), 6.4), dpi=100)
    axs[0].imshow(im, cmap="gray", origin="lower"); axs[0].set_title(f"{title}: intensity"); axs[0].set_xticks([]); axs[0].set_yticks([])
    acts = [A[k][crop[0], y, crop[1]] for k in range(N)]; dds = [D[k][:, crop[0], y, crop[1]] for k in range(N)]
    for k in range(N):
        ax = axs[1 + k]; ax.imshow(dec_activation_overlay(dds[k], acts[k], im, a.mode, a.gamma), origin="lower", interpolation="nearest")
        ax.set_title(f"population {k + 1}: DEC of its layer's direction, brightness = activation" if a.mode == "brightness" else f"population {k + 1}: its layer's direction as hue everywhere, activation as {a.mode}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    if a.mode == "brightness":
        comb = sum(dec_activation_overlay(dds[k], acts[k], im, a.mode, a.gamma) for k in range(N))      # the DEC maps add: a voxel with two populations shows both
        ax = axs[N + 1]; ax.imshow(np.clip(comb, 0, 1), origin="lower", interpolation="nearest")
        ax.set_title("all populations: the per-population DEC maps summed (brightness = activation)", fontsize=10)
    else:
        best = np.argmax(np.stack(acts), 0); dboth = np.choose(best[None], dds) if N > 1 else dds[0]; aboth = np.max(np.stack(acts), 0)
        ax = axs[N + 1]; ax.imshow(dec_activation_overlay(dboth, aboth, im, a.mode, a.gamma), origin="lower", interpolation="nearest")
        ax.set_title(f"all populations: the most active layer's direction per voxel, activation as {a.mode}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DEC orientation of each population's layer with its activation as {a.mode} (R = |x|, G = |y|, B = |z|)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); out = FIGS / f"dec_overlay_{a.field}_{a.mode}.png"; fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
