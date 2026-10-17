"""Compression against smoothing on the real block: the activation images, their DEC-opacity overlays and the combined
field for the plain pipeline (smooth 3), the feature compressions (unit-norm, √) and the activation compressions
(√a, log1p(10a)), with the spikiness numbers (p99 / median inside the population's mask, the relative two-voxel
difference in tissue, layer vs image tensor). Reads the fields exported by fig_activation_real.py --nmax 2 --smooth 3
[--transform unit|sqrt] [--act-transform sqrt|log].

    python scripts/fig_transform_real.py   -> figures/tca_field/transform_real_S2146.png
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor import DATA_ROOT
from shapetensor.readout.layers import combined_field
from shapetensor.viz import dec_activation_overlay

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
ROWS = [("", "plain (smooth 3)"), ("_tunit", "unit-norm features (the pattern only)"), ("_tsqrt", "√ features"), ("_asqrt", "√ activation"), ("_alog", "log1p(10 · activation)")]


def main():
    z0 = zarr.open(CROP, mode="r")["0"]; o = (z0.shape[0] - 384) // 2 + (384 - 192) // 2
    img = np.asarray(z0[o:o + 192, o:o + 192, o:o + 192]).astype(np.float32)
    crop = (slice(24, 168), slice(24, 168)); y = 96; im = img[crop[0], y, crop[1]]
    tis = img >= 142.5
    fig, axs = plt.subplots(len(ROWS), 6, figsize=(33, 6.0 * len(ROWS)), dpi=90)
    for i, (sfx, label) in enumerate(ROWS):
        g = zarr.open_group(str(Path(DATA_ROOT) / "fields" / f"S2146_population_field_block192_nmax2_s3{sfx}.zarr"), mode="r")
        A = np.asarray(g["activation"]).astype(np.float32); D = np.asarray(g["dirs"]).astype(np.float32); masks = np.asarray(g["masks"]).astype(bool)
        res = json.load(open(f"{DATA_ROOT}/results/activation_real_S2146_nmax2_s3{sfx}.json")); L = res["layers"]
        vct, wct = combined_field(torch.as_tensor(D), torch.as_tensor(A)); vc, wc = vct.numpy(), wct.numpy()
        axs[i, 0].imshow(im, cmap="gray", origin="lower"); axs[i, 0].set_title(f"{label}: intensity", fontsize=11)
        for k in range(2):
            act = A[k][crop[0], y, crop[1]]
            mk = masks[k] & tis; spike = np.percentile(A[k][mk], 99) / max(np.median(A[k][mk]), 1e-6)
            d2 = np.abs(A[k][2:] - A[k][:-2])[tis[1:-1]].mean() / max(A[k][tis].mean(), 1e-6)
            axs[i, 1 + k].imshow(act, cmap="Blues", vmin=0, vmax=np.percentile(act, 99), origin="lower", interpolation="nearest")
            axs[i, 1 + k].set_title(f"{label}: activation {k + 1}\np99 / median in mask {spike:.1f}; 2-voxel difference / mean {d2:.2f}", fontsize=10)
            axs[i, 3 + k].imshow(dec_activation_overlay(D[k][:, crop[0], y, crop[1]], act, im, "opacity", floor=0.0), origin="lower", interpolation="nearest")
            axs[i, 3 + k].set_title(f"{label}: DEC of layer {k + 1}, opacity = activation\nvs image tensor in mask {L[k]['layer_vs_image_tensor_in_mask_deg']}°; mask {L[k]['mask_share']:.0%} of tissue", fontsize=10)
        axs[i, 5].imshow(dec_activation_overlay(vc[:, crop[0], y, crop[1]], wc[crop[0], y, crop[1]], im, "opacity", floor=0.0), origin="lower", interpolation="nearest")
        axs[i, 5].set_title(f"{label}: combined field, opacity = total activation\nresidual {res['residual_median']:.2f}, τ {res['tau']:.2f}", fontsize=10)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Compressing the features or the activations instead of smoothing, on the real block (S-21-46 192³, two populations, self-calibrated): "
                 "a pointwise compression flattens the peaks but keeps every spatial detail; the masks are unchanged by construction (monotone)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGS / "transform_real_S2146.png"); plt.close(fig); print("wrote", FIGS / "transform_real_S2146.png")


if __name__ == "__main__":
    main()
