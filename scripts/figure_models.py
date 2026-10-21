"""QC figure: the old and new anatomix models' feature spaces on the HiP-CT VOI (S-21-46 20 µm cortex), one axial
slice: the image, then each model's features as RGB (the slice's first three principal components, each scaled to its
1-99th percentiles; the components' signs are arbitrary). Models: the anatomix UNet (5.9 M parameters) at its 16-channel
output and at encoder block 2 (the released pipeline's features), and anatomix-dev-vit (26 M, 32 channels).

    python scripts/figure_models.py [OUT_PNG]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor.features import make_engine

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/figures/models_feature_space.png")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; HALF_Z, HALF_XY, SHOW = 48, 160, 256
MODELS = (("anatomix UNet, 16-ch output", "anatomix16"), ("anatomix UNet, encoder block 2\n(released pipeline)", "anatomix_enc2"),
          ("anatomix-dev-vit, 32-ch output", "anatomix_devvit"))


def pca_rgb(F: np.ndarray):
    """``F (C, H, W)`` -> RGB of its first three principal components and their share of the variance."""
    C = F.shape[0]; X = F.reshape(C, -1).T.astype(np.float64); X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 60000), replace=False)], full_matrices=False)
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
    rgb = np.clip((Y - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(F.shape[1], F.shape[2], 3)
    return rgb, (S[:3] ** 2).sum() / (S ** 2).sum()


def main():
    a = zarr.open_group(SRC, mode="r")["0"]; c = [n // 2 for n in a.shape]
    blk = np.asarray(a[c[0] - HALF_Z:c[0] + HALF_Z, c[1] - HALF_XY:c[1] + HALF_XY, c[2] - HALF_XY:c[2] + HALF_XY]).astype(np.uint8)
    v = torch.as_tensor(blk, device=DEV).float()
    s0 = slice(HALF_XY - SHOW // 2, HALF_XY + SHOW // 2)
    img = blk[HALF_Z, s0, s0].astype(np.float32)
    panels = [("HiP-CT image (S-21-46, 20 µm)", np.stack([np.clip((img - np.percentile(img, 1)) / max(np.ptp(np.percentile(img, [1, 99])), 1e-6), 0, 1)] * 3, -1))]
    for title, eng in MODELS:
        F = make_engine(eng, device=DEV, nyquist="none")(v).float()[:, HALF_Z, s0, s0].cpu().numpy()
        rgb, share = pca_rgb(F)
        panels.append((f"{title}\nPCs 1-3 as RGB ({100 * share:.0f}% of variance)", rgb))
        del F; torch.cuda.empty_cache()
    fig, axs = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 5.0), facecolor="white")
    for ax, (t, im) in zip(axs, panels):
        ax.imshow(im, interpolation="nearest"); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axs[0].plot([10, 10 + 50], [SHOW - 12] * 2, color="yellow", lw=3); axs[0].text(10, SHOW - 18, "1 mm", color="yellow", fontsize=9)
    fig.suptitle(f"Feature spaces on the HiP-CT cortex VOI: axial slice z = {c[0]}, {SHOW} x {SHOW} voxels at the VOI centre", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.92)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=120); print(OUT)


if __name__ == "__main__":
    main()
