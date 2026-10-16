"""Step 2: what the anatomix shape space of the 20 um S-21-46 VOI is made of (reads step 1's features).

  axes        each PC is a fixed weighting of the 16 channels (its loadings); a voxel's coordinate on it is the
              projection of its L2-normalised 16-vector (unitless, -1..1); % = share of the variance over the VOI
  channels    the network's own widths: 64 channels at 1/4 resolution (80 um), 32 at 1/2 (40 um), the 16-channel
              output at full resolution; and the output cut to its top 3 or 8 PCs
  continuous  no channels, no classes: the cosine similarity of every voxel's 16-vector to one seed voxel's, and the
              shape-space gradient |df/dx|

    python shape_20um_explain.py FIGDIR        (chunkreg env; CPU, one pass for the internal layers)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import shape_20um_space as S

LAYERS = {50: "64 channels, 1/4 resolution (80 um)", 57: "32 channels, 1/2 resolution (40 um)"}


def internal(v):
    """Activations of the anatomix UNet at LAYERS for the VOI (one CPU pass, same padding as anatomix_shape)."""
    import torch
    from chunkreg.features.anatomix import load_backbone
    model = load_backbone("anatomix", "cpu")
    margin = 16
    pad = [(margin, margin + (-(n + 2 * margin)) % 32) for n in v.shape]
    x = torch.from_numpy(np.pad(np.clip(v / 255.0, 0, 1), pad, mode="reflect").astype(np.float32))[None, None]
    with torch.no_grad():
        _, acts = model(x, layers=list(LAYERS))
    out = {}
    for lid, a in zip(LAYERS, acts):
        a = a[0].numpy(); f = x.shape[2] // a.shape[1]                    # downsampling factor
        lo = (margin + S.M) // f; n = S.N // f
        a = a[:, lo:lo + n, lo:lo + n, lo:lo + n]
        out[lid] = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-6)
    return out


def pca_rgb(F, rng, k=3):
    X = F.reshape(F.shape[0], -1).T
    Xs = X[rng.choice(len(X), min(200_000, len(X)), replace=False)]
    mu = Xs.mean(0); w, V = np.linalg.eigh(np.cov((Xs - mu).T)); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    B = V[:, :k].T; B *= np.sign(B[np.arange(k), np.abs(B).argmax(1)])[:, None]
    P = (Xs - mu) @ B.T; lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)

    def rgb(Fs):
        q = (Fs.reshape(F.shape[0], -1).T - mu) @ B.T
        return np.clip((q - lo) / (hi - lo), 0, 1).reshape(Fs.shape[1:] + (k,))
    return rgb, w / w.sum(), B, mu


def main(figdir):
    figdir = Path(figdir)
    F = np.load(S.DATA / "voi20_features.npy").astype(np.float32)
    sl = slice(S.LO, S.LO + S.N + 2 * S.M)
    vb = np.asarray(zarr.open(S.CROP, mode="r")["0"][sl, sl, sl]).astype(np.float32)
    v = vb[S.E]
    rng = np.random.default_rng(0)
    rgb16, ex16, B16, mu16 = pca_rgb(F, rng, k=16)
    c = S.N // 2; cor = np.s_[:, c, :]
    lo_, hi_ = np.percentile(v, (0.5, 99.5))
    # --- figure A: the axes
    fig = plt.figure(figsize=(22, 17), dpi=100); gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(v[cor], cmap="gray", origin="lower", vmin=lo_, vmax=hi_)
    ax.set_title("intensity, coronal (superior up)", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    pcs = rgb16(F[(slice(None),) + cor])                                  # (H, W, 16) in 0..1 (1-99% range)
    for k in range(3):
        ax = fig.add_subplot(gs[0, k + 1]); ax.imshow(pcs[..., k], cmap="viridis", origin="lower")
        ax.set_title(f"PC{k + 1} alone ({ex16[k]:.0%} of variance): the {'RGB'[k]} of the RGB maps", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :2]); x = np.arange(16)
    for k, col in enumerate(("r", "g", "b")):
        ax.bar(x + (k - 1) * 0.27, B16[k], 0.27, color=col, label=f"PC{k + 1}")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xlabel("anatomix channel"); ax.set_ylabel("loading (weight)")
    ax.set_title("each shape-space axis = a fixed weighted sum of the 16 channels (PCA loadings, this VOI)", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax = fig.add_subplot(gs[1, 2]); ax.bar(np.arange(1, 17), ex16, color="grey"); ax.plot(np.arange(1, 17), np.cumsum(ex16), "k-o", ms=3)
    ax.set_xlabel("principal component"); ax.set_ylabel("share of variance (bars), cumulative (line)")
    ax.set_title("how many dimensions the 16 channels really span", fontsize=11); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 3]); Cm = np.corrcoef(F.reshape(16, -1)[:, rng.choice(S.N ** 3, 200_000, replace=False)])
    im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("channel-channel correlation over the VOI", fontsize=11); ax.set_xlabel("channel"); ax.set_ylabel("channel")
    for k in range(4):
        ax = fig.add_subplot(gs[2, k]); ch = F[k * 4][cor]; a, b = np.percentile(ch, (1, 99))
        ax.imshow(ch, cmap="gray", origin="lower", vmin=a, vmax=b); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"raw channel {k * 4} (no fixed meaning: any rotation of the 16 is equivalent)", fontsize=9)
    fig.suptitle("Step 2a. The shape-space axes: PC1/2/3 are directions in the 16-channel feature space, ranked by how much "
                 "the VOI's voxels vary along them", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(figdir / "voi20_step2a_axes.png"); plt.close(fig)
    # --- figure B: more and fewer channels
    acts = internal(vb)
    fig, axs = plt.subplots(2, 3, figsize=(21, 15.5), dpi=100)
    for col, (lid, title) in enumerate(LAYERS.items()):
        a = acts[lid]; rgbA, exA, _, _ = pca_rgb(a, rng)
        img = rgbA(a[(slice(None), slice(None), a.shape[2] // 2, slice(None))])
        axs[0, col].imshow(img, origin="lower", interpolation="nearest")
        axs[0, col].set_title(f"internal layer {lid}: {title}\nPC1-3 carry {exA[:3].sum():.0%}", fontsize=11)
    rgb3, ex3, _, _ = pca_rgb(F, rng)
    axs[0, 2].imshow(rgb3(F[(slice(None),) + cor]), origin="lower")
    axs[0, 2].set_title(f"output: 16 channels, full resolution (20 um)\nPC1-3 carry {ex3[:3].sum():.0%}", fontsize=11)
    # fewer: the output cut to k PCs, judged by cosine similarity to a seed
    X = F.reshape(16, -1).T
    seed = np.unravel_index(np.argmax(ndi.gaussian_filter(pcs[..., 0], 3)), pcs.shape[:2])
    sidx = np.ravel_multi_index((seed[0], c, seed[1]), (S.N,) * 3)
    for col, k in enumerate((3, 8, 16)):
        Bk = B16[:k]; Xk = (X - mu16) @ Bk.T + 0                            # coordinates in the top-k PCs
        Fk = Xk @ Bk + mu16                                                  # back in channel space
        Fk = Fk / np.maximum(np.linalg.norm(Fk, axis=1, keepdims=True), 1e-6)
        sim = (Fk @ Fk[sidx]).reshape((S.N,) * 3)[cor]
        ax = axs[1, col]; im = ax.imshow(sim, cmap="magma", origin="lower", vmin=np.percentile(sim, 2), vmax=1)
        ax.plot(seed[1], seed[0], "c+", ms=14, mew=2)
        ax.set_title(f"output cut to {k} PC{'s' if k > 1 else ''} ({ex16[:k].sum():.0%} of variance):\n"
                     f"cosine similarity to the seed (+)", fontsize=11)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Step 2b. More and fewer channels: the network's wider layers are coarser (top); cutting the output to fewer "
                 "PCs (bottom). Coronal, superior up", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5); fig.savefig(figdir / "voi20_step2b_channels.png"); plt.close(fig)
    # --- figure C: continuous, channel-free
    fig, axs = plt.subplots(2, 3, figsize=(21, 15.5), dpi=100)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    seeds = [np.unravel_index(np.argmax(ndi.gaussian_filter(pcs[..., k], 3)), pcs.shape[:2]) for k in range(3)]
    for col, (sd, name) in enumerate(zip(seeds, ("seed at max PC1", "seed at max PC2", "seed at max PC3"))):
        sidx = np.ravel_multi_index((sd[0], c, sd[1]), (S.N,) * 3)
        sim = (Xn @ Xn[sidx]).reshape((S.N,) * 3)[cor]
        ax = axs[0, col]; ax.imshow(sim, cmap="magma", origin="lower", vmin=np.percentile(sim, 2), vmax=1)
        ax.plot(sd[1], sd[0], "c+", ms=14, mew=2)
        ax.set_title(f"{name}: cosine similarity of every voxel's 16-vector to the seed's\n(continuous, 1 = same shape)", fontsize=11)
    g = np.zeros((S.N, S.N), np.float32); Fc = F[:, :, c - 3:c + 4, :]
    for f in Fc:
        for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            g += ndi.gaussian_filter(f, 1.0, order=o, truncate=2.0)[:, 3, :] ** 2
    g = np.sqrt(g)
    axs[1, 0].imshow(g, cmap="magma", origin="lower", vmin=0, vmax=np.percentile(g, 99))
    axs[1, 0].set_title("shape-space gradient |df/dx| (continuous boundary strength)", fontsize=11)
    axs[1, 1].imshow(v[cor], cmap="gray", origin="lower", vmin=lo_, vmax=hi_)
    for sd in seeds:
        axs[1, 1].plot(sd[1], sd[0], "c+", ms=14, mew=2)
    axs[1, 1].set_title("intensity with the three seeds", fontsize=11)
    axs[1, 2].imshow(rgb3(F[(slice(None),) + cor]), origin="lower"); axs[1, 2].set_title("shape-space RGB (step 1), for reference", fontsize=11)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Step 2c. A continuous, channel-free read-out: similarity to a seed, and the rate of change of shape", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5); fig.savefig(figdir / "voi20_step2c_continuous.png"); plt.close(fig)
    json.dump({"pca_explained_16": [round(float(x), 4) for x in ex16], "loadings_pc1_3": B16[:3].round(3).tolist()},
              open(S.DATA / "voi20_step2.json", "w"), indent=1)
    print("wrote", figdir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
