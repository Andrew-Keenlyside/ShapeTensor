"""The feature average (smooth 1 / 3 / 5) on the real HiP-CT block: per setting, the two populations' activation
images, their DEC-opacity overlays, the combined (activation-weighted) vector field as DEC, and the numbers.
Reads the fields exported by fig_activation_real.py --nmax 2 --smooth {1,3,5}.

    python scripts/fig_smooth_real.py   -> figures/tca_field/smooth_real_S2146.png
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
from shapetensor.viz import dec_activation_overlay, dec_rgb

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"


def main():
    z0 = zarr.open(CROP, mode="r")["0"]; o = (z0.shape[0] - 384) // 2 + (384 - 192) // 2
    img = np.asarray(z0[o:o + 192, o:o + 192, o:o + 192]).astype(np.float32)
    crop = (slice(24, 168), slice(24, 168)); y = 96; im = img[crop[0], y, crop[1]]
    fig, axs = plt.subplots(3, 6, figsize=(33, 16.5), dpi=90)
    for i, s in enumerate((1, 3, 5)):
        g = zarr.open_group(str(Path(DATA_ROOT) / "fields" / f"S2146_population_field_block192_nmax2_s{s}.zarr"), mode="r")
        A = np.asarray(g["activation"]).astype(np.float32); D = np.asarray(g["dirs"]).astype(np.float32)
        vct, wct = combined_field(torch.as_tensor(D), torch.as_tensor(A))                        # the activation-weighted average vector, unmasked
        vc, wc = vct.numpy(), wct.numpy()
        res = json.load(open(f"{DATA_ROOT}/results/activation_real_S2146_nmax2_s{s}.json"))
        L = res["layers"]
        axs[i, 0].imshow(im, cmap="gray", origin="lower"); axs[i, 0].set_title(f"smooth {s}: intensity", fontsize=11)
        for k in range(2):
            act = A[k][crop[0], y, crop[1]]
            axs[i, 1 + k].imshow(act, cmap="Blues", vmin=0, vmax=np.percentile(act, 99), origin="lower", interpolation="nearest")
            axs[i, 1 + k].set_title(f"smooth {s}: activation, population {k + 1} (mask {L[k]['mask_share']:.0%} of tissue)", fontsize=11)
            axs[i, 3 + k].imshow(dec_activation_overlay(D[k][:, crop[0], y, crop[1]], act, im, "opacity", floor=0.0), origin="lower", interpolation="nearest")
            axs[i, 3 + k].set_title(f"smooth {s}: DEC of layer {k + 1}, opacity = activation; vs image tensor in mask {L[k]['layer_vs_image_tensor_in_mask_deg']}°", fontsize=11)
        axs[i, 5].imshow(dec_activation_overlay(vc[:, crop[0], y, crop[1]], wc[crop[0], y, crop[1]], im, "opacity", floor=0.0), origin="lower", interpolation="nearest")
        axs[i, 5].set_title(f"smooth {s}: combined field (activation-weighted average vector), opacity = total activation;\nresidual {res['residual_median']:.2f}, readout overlap {res.get('fit', '')}", fontsize=11)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The feature average on the real block (S-21-46, two populations, self-calibrated): activations, DEC-by-opacity per layer, and the combined vector field, for smooth 1 / 3 / 5", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / "smooth_real_S2146.png"); plt.close(fig); print("wrote", FIGS / "smooth_real_S2146.png")


if __name__ == "__main__":
    main()
