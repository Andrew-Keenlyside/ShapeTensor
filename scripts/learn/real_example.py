"""The pilot models on real HiP-CT: a 96-slice slab of the S-21-46 20 µm cortex VOI, untouched (8-bit, no resampling).

    python scripts/learn/real_example.py RUN_DIR [RUN_DIR ...] [--out FILE]

Per model (one row): on the middle axial slice, the heaviest and second line populations' axes (DEC, weighted by
mass), the mixture head's axis, the line-population count N and the shape fraction -- all read from the model output
by the engine -- with a zoom on the grey-white boundary. Top row: the image and the structure tensor's axis at the
model's window (gradient sigma 1, integration rho 5 voxels) for reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy import ndimage

from shapetensor import st
from shapetensor.learn.engine import PopulationEngine
from shapetensor.learn.generator import eig3
from shapetensor.viz import dec_rgb

SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; HALF_Z = 48; ZW = 192


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    arr = zarr.open_group(SRC, mode="r")["0"]; c = arr.shape[0] // 2
    img = np.asarray(arr[c - HALF_Z:c + HALF_Z]).astype(np.uint8)                              # (96, 768, 768)
    v = torch.as_tensor(img, device=DEV).float()
    thr = otsu(img); tissue = img > thr
    win = tuple(float(q) for q in np.quantile(img[tissue], [0.005, 0.995]))
    k = HALF_Z; kc = k // 4
    g = img[k].astype(np.float32); g = np.clip((g - win[0]) / (win[1] - win[0]), 0, 1)
    tis = tissue[k]; tis_c = ndimage.zoom(tis.astype(np.float32), 0.25, order=1) > 0.5
    # zoom: the window of largest smoothed intensity gradient inside tissue (the grey-white boundary)
    sm = ndimage.gaussian_filter(g, 4); gm = np.hypot(*np.gradient(sm)) * tis
    score = ndimage.uniform_filter(gm, ZW); score[:ZW // 2] = 0; score[-ZW // 2:] = 0; score[:, :ZW // 2] = 0; score[:, -ZW // 2:] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape); y0, x0 = cy - ZW // 2, cx - ZW // 2
    zs = (slice(y0, y0 + ZW), slice(x0, x0 + ZW)); zc = (slice(y0 // 4, (y0 + ZW) // 4), slice(x0 // 4, (x0 + ZW) // 4))
    d_st = st.structure_tensor(v, sigma=1.0, rho=5.0, truncate=3.0, excise=False, order="zyx", device=DEV)[:, k].cpu().numpy()
    ncol = 6
    fig, axs = plt.subplots(len(a.runs) + 1, ncol, figsize=(4 * ncol, 4.1 * (len(a.runs) + 1)), facecolor="white", squeeze=False)
    top = [("HiP-CT, S-21-46 cortex VOI, 20 µm (axial)", np.stack([g] * 3, -1)),
           ("structure tensor axis (σ 1, ρ 5) × tissue", dec_rgb(d_st) * tis[..., None]),
           ("zoom: image", np.stack([g[zs]] * 3, -1)), ("zoom: structure tensor axis", (dec_rgb(d_st) * tis[..., None])[zs])]
    for ax, (t, im) in zip(axs[0], top):
        ax.imshow(im.clip(0, 1), interpolation="nearest"); ax.set_title(t, fontsize=9)
    axs[0, 0].add_patch(plt.Rectangle((x0, y0), ZW, ZW, fill=False, ec="yellow", lw=1.5))
    axs[0, 0].plot([20, 20 + 250], [745, 745], color="yellow", lw=3); axs[0, 0].text(20, 730, "5 mm", color="yellow", fontsize=8)
    axs[0, 4].hist(img[tissue].ravel(), bins=128, color="#666"); axs[0, 4].axvline(win[0], color="r"); axs[0, 4].axvline(win[1], color="r")
    axs[0, 4].set_title("tissue intensities and the model's window", fontsize=9); axs[0, 5].axis("off")
    for r, run in enumerate(a.runs, start=1):
        eng = PopulationEngine(str(Path(run) / "model.pt"), device=DEV)
        with torch.no_grad():
            net = eng.network(v, win)
            pop = eng.populations(net, labels=False)
        D, M = pop["dirs_cells"].cpu().numpy(), pop["mass"].cpu().numpy()                    # (R, 3, c), (R, c)
        mn = float(np.quantile(M[0][M[0] > 0], 0.95)) if (M[0] > 0).any() else 1.0
        _, Vm = eig3(net["mixture"]); mix = Vm[2][:, kc].cpu().numpy()
        N = pop["N"][kc].cpu().numpy(); fr = pop["fraction"][kc].cpu().numpy()
        w0 = np.clip(M[0][kc] / mn, 0, 1)[..., None]; w1 = np.clip(M[1][kc] / mn, 0, 1)[..., None] if M.shape[0] > 1 else 0 * w0
        name = Path(run).name; ck = torch.load(Path(run) / "model.pt", map_location="cpu")
        panels = [(f"{name} (step {ck['step']}): heaviest line axis × mass", dec_rgb(D[0][:, kc]) * w0),
                  ("second line population axis × mass", dec_rgb(D[1][:, kc]) * w1 if M.shape[0] > 1 else np.zeros_like(w0) * [0, 0, 0]),
                  ("mixture-head axis × tissue", dec_rgb(mix) * tis_c[..., None]),
                  ("zoom: heaviest line axis × mass", (dec_rgb(D[0][:, kc]) * w0)[zc])]
        for ax, (t, im) in zip(axs[r], panels):
            ax.imshow(np.asarray(im).clip(0, 1), interpolation="nearest"); ax.set_title(t, fontsize=9)
        h = axs[r, 4].imshow(np.where(tis_c, N, -1), cmap=matplotlib.colors.ListedColormap(["#1b1b1b", "#555555", "#2a78d6", "#e34948", "#f0b429", "#ffffff"]),
                             vmin=-1.5, vmax=4.5, interpolation="nearest")
        axs[r, 4].set_title("line populations per cell (N)", fontsize=9); plt.colorbar(h, ax=axs[r, 4], fraction=0.046, ticks=[0, 1, 2, 3, 4])
        h = axs[r, 5].imshow(np.where(tis_c, fr, np.nan), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axs[r, 5].set_title("shape fraction", fontsize=9); plt.colorbar(h, ax=axs[r, 5], fraction=0.046)
        share = {n: float((N[tis_c] == n).mean()) for n in range(4)}
        print(f"{name}: N share in tissue {share}, fraction median {float(np.median(fr[tis_c])):.2f}", flush=True)
        del net, pop; torch.cuda.empty_cache()
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pilot models on real HiP-CT (no real-data training, synthetic pilots only): every map read from the model output by the engine", fontsize=12)
    out = Path(a.out) if a.out else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_pilot/real_hipct_example.png")
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=60); print("wrote", out)


if __name__ == "__main__":
    main()
