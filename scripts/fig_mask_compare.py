"""A shape-space mask against an intensity mask on the real block: the union of the population masks (and the
background activation) versus bright tissue (intensity >= 142.5); overlap, what each adds, and where the
differences sit (boundary strength, orientation gap).

    python scripts/fig_mask_compare.py [--smooth 3]   -> figures/tca_field/mask_compare_S2146.png, results/mask_compare_S2146.json
"""
import argparse, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import zarr

from shapetensor import DATA_ROOT, st
from shapetensor.filters import gaussian

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
BRIGHT = 142.5


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smooth", type=int, default=3); a = ap.parse_args()
    z0 = zarr.open(CROP, mode="r")["0"]; o = (z0.shape[0] - 384) // 2 + (384 - 192) // 2
    img = np.asarray(z0[o:o + 192, o:o + 192, o:o + 192]).astype(np.float32)
    g = zarr.open_group(str(Path(DATA_ROOT) / "fields" / f"S2146_population_field_block192_nmax2_s{a.smooth}.zarr"), mode="r")
    union = np.asarray(g["union_mask"]); masks = np.asarray(g["masks"]); A = np.asarray(g["activation"]).astype(np.float32)
    inten = img >= BRIGHT
    inner = (slice(24, 168),) * 3
    U, I = union[inner], inten[inner]
    v = torch.as_tensor(img, device="cuda")
    gm = sum(gaussian(v, 3.0, order=oo, truncate=4.0) ** 2 for oo in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    gm = gaussian(gm, 10.0, truncate=4.0)[inner].cpu().numpy()
    S = st.structure_tensor_3d(v, 1.0, 10.0, 4.0); val, _ = st.eig_special_3d(S); gap = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)[inner].cpu().numpy(); del S
    both, only_i, only_u = U & I, I & ~U, U & ~I
    res = {"smooth": a.smooth, "intensity_share": float(I.mean()), "union_share": float(U.mean()), "dice": dice(U, I),
           "union_covers_intensity": float(both.sum() / max(1, I.sum())), "union_outside_intensity_share_of_union": float(only_u.sum() / max(1, U.sum())),
           "gm_median_both_onlyI_onlyU": [float(np.median(gm[m])) for m in (both, only_i, only_u)],
           "gap_median_both_onlyI_onlyU": [float(np.median(gap[m])) for m in (both, only_i, only_u)],
           "intensity_median_both_onlyI_onlyU": [float(np.median(img[inner][m])) for m in (both, only_i, only_u)]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{DATA_ROOT}/results/mask_compare_S2146_s{a.smooth}.json", "w"), indent=1)
    y = 72; im = img[inner][:, y]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    axs[0, 0].imshow(im, cmap="gray", origin="lower"); axs[0, 0].set_title("intensity")
    axs[0, 1].imshow(im, cmap="gray", origin="lower", alpha=0.5); axs[0, 1].imshow(np.ma.masked_where(~I[:, y], I[:, y]), cmap=ListedColormap(["#6250d6"]), origin="lower", alpha=0.6, interpolation="nearest")
    axs[0, 1].set_title(f"intensity mask (≥ {BRIGHT:g}): {res['intensity_share']:.0%} of the block")
    axs[0, 2].imshow(im, cmap="gray", origin="lower", alpha=0.5); axs[0, 2].imshow(np.ma.masked_where(~U[:, y], U[:, y]), cmap=ListedColormap(["#2a9d8f"]), origin="lower", alpha=0.6, interpolation="nearest")
    axs[0, 2].set_title(f"shape mask (union of population masks): {res['union_share']:.0%} of the block")
    cls = np.zeros(I[:, y].shape, int); cls[both[:, y]] = 1; cls[only_i[:, y]] = 2; cls[only_u[:, y]] = 3
    axs[1, 0].imshow(im, cmap="gray", origin="lower", alpha=0.5); axs[1, 0].imshow(np.ma.masked_where(cls == 0, cls), cmap=ListedColormap(["#9ec5f4", "#e34948", "#eda100"]), vmin=1, vmax=3, origin="lower", alpha=0.75, interpolation="nearest")
    axs[1, 0].set_title(f"both (blue) · intensity only (red) · shape only (orange); Dice {res['dice']:.2f}")
    im1 = axs[1, 1].imshow(gm[:, y], cmap="Blues", origin="lower"); fig.colorbar(im1, ax=axs[1, 1], fraction=0.04); axs[1, 1].set_title(f"boundary strength gm: median both {res['gm_median_both_onlyI_onlyU'][0]:.2f}, intensity-only {res['gm_median_both_onlyI_onlyU'][1]:.2f}, shape-only {res['gm_median_both_onlyI_onlyU'][2]:.2f}", fontsize=10)
    im2 = axs[1, 2].imshow(gap[:, y], cmap="Blues", vmin=0, vmax=1, origin="lower"); fig.colorbar(im2, ax=axs[1, 2], fraction=0.04); axs[1, 2].set_title(f"orientation gap: median both {res['gap_median_both_onlyI_onlyU'][0]:.2f}, intensity-only {res['gap_median_both_onlyI_onlyU'][1]:.2f}, shape-only {res['gap_median_both_onlyI_onlyU'][2]:.2f}", fontsize=10)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Shape-space mask vs intensity mask on the real block (S-21-46 192³, smooth {a.smooth}): the shape mask covers {res['union_covers_intensity']:.0%} of the intensity mask; {res['union_outside_intensity_share_of_union']:.0%} of it lies outside the intensity mask", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / f"mask_compare_S2146_s{a.smooth}.png"); plt.close(fig); print("wrote", FIGS / f"mask_compare_S2146_s{a.smooth}.png")


if __name__ == "__main__":
    main()
