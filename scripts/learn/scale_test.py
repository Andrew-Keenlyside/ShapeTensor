"""A basic resolution test: one physical structure at several voxel sizes, the same trained model on each.

    python scripts/learn/scale_test.py --model RUN/model.pt [--out FILE]

One smoothly bending line population is drawn once on a fine grid (192³, fibre radius 3 voxels) and resampled by
averaging to 96³ and 48³ (radius 1.5 and 0.75 voxels), plus a non-cubic crop of the fine grid. At each resolution the
model's heaviest population axis and mixture axis are compared with that resolution's own true axis (the label
tensors averaged over each 4³ cell), where a cell holds fibre. The output window is fixed in voxels (sigma 5, stride
4), so coarser inputs are described over larger physical neighbourhoods -- what the planned sigma_A input removes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor.learn.engine import PopulationEngine
from shapetensor.learn.generator import Layer, direction_field, eig3, layer_density, render_layer
from shapetensor.viz import dec_rgb

DEV = "cuda"


def truth_axis(occ, T):
    """Per 4³ cell: the occupancy-weighted label tensor's principal axis, and the cell's fibre occupancy."""
    w = Fn.avg_pool3d(occ[None, None], 4)[0, 0]
    Tc = Fn.avg_pool3d((T * occ[None])[None], 4)[0] / w.clamp_min(1e-6)[None]
    _, V = eig3(Tc); return V[2], w


def predict(eng, img):
    lo, hi = float(np.quantile(img.cpu().numpy(), 0.005)), float(np.quantile(img.cpu().numpy(), 0.995))
    net = eng.network(img, (lo, hi))
    pop = eng.populations(net, labels=False)
    _, Vm = eig3(net["mixture"])
    return pop["dirs_cells"][0], Vm[2], pop["N"]


def err(a, b, m):
    e = torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))
    return e, (float(e[m].median()) if bool(m.any()) else float("nan"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    eng = PopulationEngine(a.model, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(3); S = (192, 192, 192)
    fld = direction_field(S, 90.0, g, DEV)
    L = Layer("line", fld, 3.0, 240.0, 0.0, torch.ones(S, dtype=torch.bool, device=DEV), 60.0); L.density = layer_density(L, 0.25)
    occ, T = render_layer(L, S, g, DEV)
    fine = (100 + 70 * occ + 4 * torch.randn(S, device=DEV, generator=g))
    cases = []
    for f, name in ((1, "192³ (fibre radius 3 vox)"), (2, "96³ (radius 1.5)"), (4, "48³ (radius 0.75)")):
        img = Fn.avg_pool3d(fine[None, None], f)[0, 0] if f > 1 else fine
        o = Fn.avg_pool3d(occ[None, None], f)[0, 0] if f > 1 else occ
        Tf = Fn.avg_pool3d((T * occ[None])[None], f)[0] / o.clamp_min(1e-6)[None] if f > 1 else T
        cases.append((name, f, img, o, Tf))
    crop = (slice(8, 8 + 120), slice(20, 20 + 152), slice(4, 4 + 184))                               # a non-cubic block (120 x 152 x 184)
    cases.append(("non-cubic crop 120×152×184", 1, fine[crop], occ[crop], T[(slice(None),) + crop]))
    fig, axs = plt.subplots(len(cases), 6, figsize=(24, 4.2 * len(cases)), facecolor="white", squeeze=False)
    summary = []
    for r, (name, f, img, o, Tf) in enumerate(cases):
        ta, w = truth_axis(o, Tf)
        pa, ma, N = predict(eng, img)
        m = w > 0.15
        e_s, med_s = err(pa, ta, m); e_m, med_m = err(ma, ta, m)
        summary.append((name, med_s, med_m, int(m.sum())))
        cy = img.shape[1] // 2; c = cy // 4
        ax = axs[r]
        ax[0].imshow(img[:, cy].cpu().numpy(), cmap="gray"); ax[0].set_title(f"{name}: image (z-x)", fontsize=9)
        ax[1].imshow((dec_rgb(ta[:, :, c].cpu().numpy()) * w[:, c].clamp(0, 1).cpu().numpy()[..., None] * 2).clip(0, 1), interpolation="nearest"); ax[1].set_title("true axis per cell (DEC × fibre)", fontsize=9)
        ax[2].imshow((dec_rgb(pa[:, :, c].cpu().numpy()) * w[:, c].clamp(0, 1).cpu().numpy()[..., None] * 2).clip(0, 1), interpolation="nearest"); ax[2].set_title("predicted: heaviest population axis", fontsize=9)
        ax[3].imshow((dec_rgb(ma[:, :, c].cpu().numpy()) * w[:, c].clamp(0, 1).cpu().numpy()[..., None] * 2).clip(0, 1), interpolation="nearest"); ax[3].set_title("predicted: mixture-head axis", fontsize=9)
        em = np.where(m[:, c].cpu().numpy(), e_m[:, c].cpu().numpy(), np.nan)
        h = ax[4].imshow(em, cmap="RdYlGn_r", vmin=0, vmax=60, interpolation="nearest"); plt.colorbar(h, ax=ax[4], fraction=0.046)
        ax[4].set_title(f"mixture axis error (deg): median {med_m:.1f}\nslots: median {med_s:.1f} ({int(m.sum())} fibre cells)", fontsize=9)
        ax[5].hist(e_m[m].cpu().numpy(), bins=30, range=(0, 90), color="#7e3ff2", alpha=0.7, label="mixture")
        ax[5].hist(e_s[m].cpu().numpy(), bins=30, range=(0, 90), color="#2a78d6", alpha=0.5, label="slots"); ax[5].legend(fontsize=8)
        ax[5].set_title("axis error over the whole volume", fontsize=9)
        for a_ in ax[:5]:
            a_.set_xticks([]); a_.set_yticks([])
        print(f"{name:30s} slots {med_s:5.1f} deg   mixture {med_m:5.1f} deg   ({int(m.sum())} fibre cells, input {tuple(img.shape)})", flush=True)
    fig.suptitle(f"One structure at several voxel sizes, one model ({Path(a.model).parent.name}): the model runs at any size; its window is fixed in voxels", fontsize=12)
    out = Path(a.out) if a.out else Path(a.model).parent / "scale_test.png"
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=55); print("wrote", out)


if __name__ == "__main__":
    main()
