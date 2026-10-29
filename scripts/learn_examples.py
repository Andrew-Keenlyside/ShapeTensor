"""What a training sample of the population model looks like, with today's generator: a layout with its labels under
several nuisance draws (contrast, polarity, anisotropic PSF, rings, streaks, noise), the same layout at three voxel sizes,
and a rendition of anatomix's recipe (random label map, random intensities) for comparison.

    python scripts/learn_examples.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor.artefacts import rings, streaks
from shapetensor.filters import gaussian
from shapetensor.phantoms import Grid, TextureParams, build, direction
from shapetensor.validate import SETTINGS
from shapetensor.viz import dec_rgb

DEV = "cuda"


def nuisances(img: np.ndarray, seed: int):
    """Four draws of the acquisition / contrast nuisances on one layout: as rendered; inverted contrast + gamma + bias;
    a PSF twice as long along z + rings; streaks + heavy noise."""
    rng = np.random.default_rng(seed)
    v = torch.as_tensor(img, device=DEV).float()
    out = {"as rendered": v}
    inv = 255 - v; g = rng.uniform(0.6, 1.6); inv = 255 * (inv / 255).clamp(0, 1) ** g
    Z, Y, X = v.shape; zz = torch.linspace(-1, 1, Z, device=DEV)[:, None, None]; xx = torch.linspace(-1, 1, X, device=DEV)[None, None, :]
    bias = 1 + 0.25 * (zz * rng.uniform(-1, 1) + xx * rng.uniform(-1, 1))
    out["inverted, gamma, bias field"] = (inv * bias).clamp(0, 255)
    psf = gaussian(v, (2.0, 0.6, 0.6), truncate=3.0)
    out["anisotropic PSF (z) + rings"] = torch.as_tensor(rings(psf.round().clamp(0, 255).to(torch.uint8).cpu().numpy(), 30.0, seed=seed, device=DEV)[0], device=DEV).float()
    st = torch.as_tensor(streaks(img, 60.0, spread_deg=90.0, seed=seed, device=DEV)[0], device=DEV).float()
    out["streaks + heavy noise"] = (st + 25 * torch.randn_like(st)).clamp(0, 255)
    return out


def anatomix_style(shape=(96, 96, 96), n_labels=8, seed=0):
    """A rendition of the anatomix recipe: a random smooth label map (noise blobs), a random intensity per label, blur,
    a bias field, noise -- no fibre texture, no orientation."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    fields = torch.stack([gaussian(torch.randn(shape, device=DEV, generator=g), 6.0) for _ in range(n_labels)])
    labels = fields.argmax(0)
    outs = []
    for k in range(3):
        inten = torch.rand(n_labels, device=DEV, generator=g) * 200 + 30
        img = inten[labels] + 15 * torch.randn(shape, device=DEV, generator=g)
        img = gaussian(img, float(torch.rand(1, generator=g, device=DEV) * 1.5 + 0.3))
        zz = torch.linspace(-1, 1, shape[0], device=DEV)[:, None, None]; img = img * (1 + 0.2 * zz)
        outs.append(img.clamp(0, 255))
    return labels, outs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_examples")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    fig = plt.figure(figsize=(22, 17), facecolor="white"); gs = fig.add_gridspec(4, 7, hspace=0.3, wspace=0.08)
    for row, (name, kind, kw, grid) in enumerate((("two fascicles crossing (band crossing 90°)", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192), margin=(32, 32, 32))),
                                                   ("three fascicles turning 90° (bend)", "bend", {}, Grid(core=(192, 32, 192), margin=(32, 32, 32))))):
        img, tr, meta = build(kind, 3, grid, tex=tex, texture=p["texture"], device=DEV, **kw); ky = img.shape[1] // 2
        draws = nuisances(img, seed=row)
        for j, (dname, vol) in enumerate(draws.items()):
            ax = fig.add_subplot(gs[row, j]); ax.imshow(vol[:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax.set_title(f"{dname}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
        # labels: populations (id), tangent (DEC), fraction / N
        ax = fig.add_subplot(gs[row, 4])
        if kind == "band_crossing":
            lab = np.zeros(tr["band1"].shape + (3,), np.float32); lab[tr["band1"]] = (0.2, 0.5, 0.9); lab[tr["band2"]] = (0.9, 0.3, 0.3); lab[tr["band1"] & tr["band2"]] = (0.8, 0.5, 0.9)
        else:
            fid = tr["fid"]; lab = np.zeros(fid.shape + (3,), np.float32)
            for k, c in zip((1, 2, 3), ((0.2, 0.5, 0.9), (0.9, 0.3, 0.3), (0.3, 0.8, 0.4))):
                lab[fid == k] = c
        ax.imshow(lab); ax.set_title("label: population identity", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[row, 5]); d1 = np.nan_to_num(tr["dir1"]).transpose(2, 0, 1); ax.imshow(dec_rgb(d1) * (tr["npop"] > 0)[..., None]); ax.set_title("label: local tangent (DEC)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[row, 6]); ax.imshow(tr["npop"], cmap=matplotlib.colors.ListedColormap(["#222222", "#2a78d6", "#e34948"]), vmin=-0.5, vmax=2.5, interpolation="nearest"); ax.set_title("label: populations present (0 / 1 / 2)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    # the same layout at three voxel sizes
    img, tr, meta = build("band_crossing", 3, Grid(core=(192, 32, 192), margin=(32, 32, 32)), tex=tex, texture=p["texture"], device=DEV, cross_deg=60.0)
    v = torch.as_tensor(img, device=DEV).float(); ky = v.shape[1] // 2
    for j, (f, lab) in enumerate(((1, "20 µm (as rendered)"), (2, "40 µm (2× downsampled)"), (4, "80 µm (4× downsampled)"))):
        vol = v if f == 1 else Fn.avg_pool3d(v[None, None], f)[0, 0]
        ax = fig.add_subplot(gs[2, j]); ax.imshow(vol[:, ky // f].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax.set_title(f"scale: the same layout at {lab}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("scale", fontsize=10)
    ax = fig.add_subplot(gs[2, 3]); ax.imshow(v[96:160, ky, 96:160].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax.set_title("20 µm, the crossing at 1:1 (64 voxels)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2, 4]); vol = Fn.avg_pool3d(v[None, None], 2)[0, 0]; ax.imshow(vol[48:80, ky // 2, 48:80].cpu().numpy(), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_title("40 µm, the crossing at 1:1 (32 voxels)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2, 5]); vol = Fn.avg_pool3d(v[None, None], 4)[0, 0]; ax.imshow(vol[24:40, ky // 4, 24:40].cpu().numpy(), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_title("80 µm, the crossing at 1:1 (16 voxels)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.add_subplot(gs[2, 6]).axis("off")
    # anatomix-style
    labels, outs = anatomix_style()
    ax = fig.add_subplot(gs[3, 0]); ax.imshow(labels[48].cpu().numpy(), cmap="tab10", interpolation="nearest"); ax.set_title("anatomix recipe: a random label map", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel("anatomix-style", fontsize=10)
    for j, o in enumerate(outs):
        ax = fig.add_subplot(gs[3, 1 + j]); ax.imshow(o[48].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax.set_title(f"random intensities per label, draw {j + 1}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for j in range(4, 7):
        fig.add_subplot(gs[3, j]).axis("off")
    fig.suptitle("Training data for the population model (rows 1-2: one layout, four nuisance draws, its labels), the same layout at three voxel sizes (row 3), and anatomix's recipe (row 4)", fontsize=12)
    fig.savefig(out / "learn_examples.png", dpi=80); print("wrote", out / "learn_examples.png")


if __name__ == "__main__":
    main()
