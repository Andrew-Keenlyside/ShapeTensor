"""Step 1: the anatomix shape space of the 20 um S-21-46 calibration crop's central VOI. Nothing else: no clustering,
no labels, no tensors.

VOI: the crop's central 256^3 (5.12 mm, deep left hemisphere), read with 32 voxels of real context per side.
Array axes as the crop's TIFFs: a0 inferior -> superior, a1 anterior -> posterior, a2 patient right -> left.
anatomix runs in ONE pass on the CPU (the block does not fit the GPU; any sliding window leaves its stride in the
features, seams at 0.25 overlap and a faint 40-voxel period at 0.75), averaged over the 8 one-voxel grid phases to
cancel the network's own period-2 checkerboard (anatomix_shape.anatomix_clean).

Writes the features of the VOI (16 x 256^3, float16) to OUT/voi20_features.npy for the next steps, and figures:
  voi20_step1_shape_space.png   intensity and shape-space RGB (PC1/2/3) in three planes; the PCA scatter
  voi20_step1_channels.png      the 16 channels, coronal plane

    python shape_20um_space.py FIGDIR         (chunkreg env)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
DATA = Path("/home/andrew/bridge_wmcov/shape_tensor")
VOX = 0.02
N, M = 256, 32
LO = 384 - N // 2 - M
E = (slice(M, M + N),) * 3


def main(figdir):
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    sl = slice(LO, LO + N + 2 * M)
    v = np.asarray(zarr.open(CROP, mode="r")["0"][sl, sl, sl]).astype(np.float32)
    F = AS.anatomix_clean(v, VOX, device="cpu", amp=False)[(slice(None),) + E]
    v = v[E]
    np.save(DATA / "voi20_features.npy", F.astype(np.float16))
    Fe = F.reshape(16, -1).T
    rng = np.random.default_rng(0)
    X = Fe[rng.choice(len(Fe), 300_000, replace=False)]
    mu = X.mean(0); w, V = np.linalg.eigh(np.cov((X - mu).T)); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    B = V[:, :3].T; B *= np.sign(B[np.arange(3), np.abs(B).argmax(1)])[:, None]
    P = (X - mu) @ B.T; plo, phi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    expl = w / w.sum()
    json.dump({"voi": {"crop": CROP, "box_start_voxels": LO + M, "size_voxels": N, "voxel_mm": VOX},
               "pca_explained": [round(float(x), 4) for x in expl], "pca_basis": B.tolist(), "pca_mean": mu.tolist()},
              open(DATA / "voi20_features.json", "w"), indent=1)

    def rgb(Fs):
        q = (Fs.reshape(16, -1).T - mu) @ B.T
        return np.clip((q - plo) / (phi - plo), 0, 1).reshape(Fs.shape[1:] + (3,))
    c = N // 2
    planes = (("coronal (superior up)", np.s_[:, c, :]), ("axial", np.s_[c, :, :]), ("sagittal (superior up)", np.s_[:, :, c]))
    lo_, hi_ = np.percentile(v, (0.5, 99.5))
    fig = plt.figure(figsize=(22, 22), dpi=100); gs = fig.add_gridspec(3, 3)
    for k, (pn, s) in enumerate(planes):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(v[s], cmap="gray", origin="lower", vmin=lo_, vmax=hi_)
        ax.set_title(f"intensity, {pn}", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1, k]); ax.imshow(rgb(F[(slice(None),) + s]), origin="lower")
        ax.set_title(f"anatomix shape space, RGB = PC1 / PC2 / PC3, {pn}", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    cols = np.clip((P - plo) / (phi - plo), 0, 1); sel = rng.choice(len(P), 50_000, replace=False)
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = fig.add_subplot(gs[2, k]); ax.scatter(P[sel, i], P[sel, j], c=cols[sel], s=2, alpha=0.5, lw=0); ax.grid(alpha=0.3)
        ax.set_xlabel(f"PC{i + 1} ({expl[i]:.0%})"); ax.set_ylabel(f"PC{j + 1} ({expl[j]:.0%})")
        ax.set_title(f"shape space, PC{i + 1} vs PC{j + 1} (each voxel coloured by its RGB)", fontsize=11)
    fig.suptitle(f"Step 1. anatomix shape space, S-21-46 20 um calibration crop, central 5.1 mm VOI. "
                 f"PC1-3 carry {expl[:3].sum():.0%} of the variance (16 channels)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(figdir / "voi20_step1_shape_space.png"); plt.close(fig)
    fig, axs = plt.subplots(4, 4, figsize=(18, 18), dpi=90)
    for k, ax in enumerate(axs.ravel()):
        ch = F[k][planes[0][1]]; a, b = np.percentile(ch, (1, 99))
        ax.imshow(ch, cmap="gray", origin="lower", vmin=a, vmax=b); ax.set_title(f"channel {k}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Step 1. the 16 anatomix channels, coronal plane (superior up)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(figdir / "voi20_step1_channels.png"); plt.close(fig)
    print("explained", np.round(expl[:6], 3), "| wrote", figdir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
