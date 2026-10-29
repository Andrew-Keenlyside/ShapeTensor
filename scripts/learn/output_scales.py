"""Different output resolutions from one input, without resampling it.

    python scripts/learn/output_scales.py --model RUN/model.pt [--out FILE]

From the model's output on one 96³ volume (one smoothly bending line population):

* **denser** -- the network's 4-voxel lattice placed at every phase: the input offset by 0-3 voxels per axis (cropped,
  never resampled), the outputs interleaved: stride 2 (8 offsets) and stride 1 (64 offsets), at the native window;
* **coarser** -- the description re-weighted in the shape space: the mixture head's tensor times the cell's mass, and the
  mass, Gaussian-smoothed over cells (sigma 1, 2, 4 cells) and read on a coarser grid (stride 4, 8, 16 voxels).
  Mass x tensor and mass add, so this is the description a coarser window would give.

Each output is compared with the truth computed by the identical operator from the labels.
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

from shapetensor.filters import gaussian
from shapetensor.learn.engine import PopulationEngine
from shapetensor.learn.generator import Layer, direction_field, eig3, layer_density, render_layer
from shapetensor.viz import dec_rgb

DEV = "cuda"


def cells(occ, T, sigma=5.0):
    """The target's operator on the labels: Gaussian window, 4³ average -> per cell: mass and mass x tensor."""
    m = Fn.avg_pool3d(gaussian(occ, sigma, truncate=3.0)[None, None], 4)[0, 0]
    mT = Fn.avg_pool3d(gaussian(T * occ[None], sigma, truncate=3.0)[None], 4)[0]
    return m, mT


def axis(mT, m):
    _, V = eig3(mT / m.clamp_min(1e-9)[None]); return V[2]


def coarsen(m, mT, sc, stride):
    """Shape-space weighting: Gaussian over cells (sigma ``sc`` cells), then every ``stride`` cells."""
    if sc > 0:
        m = gaussian(m, sc, truncate=3.0); mT = gaussian(mT, sc, truncate=3.0)
    s = slice(stride // 2, None, stride)
    return m[s, s, s], mT[:, s, s, s]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    eng = PopulationEngine(a.model, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(5); S = (96, 96, 96)
    fld = direction_field(S, 45.0, g, DEV)
    L = Layer("line", fld, 1.5, 120.0, 0.0, torch.ones(S, dtype=torch.bool, device=DEV), 60.0); L.density = layer_density(L, 0.25)
    occ, T = render_layer(L, S, g, DEV)
    img = 100 + 70 * occ + 4 * torch.randn(S, device=DEV, generator=g)
    win = (float(np.quantile(img.cpu().numpy(), 0.005)), float(np.quantile(img.cpu().numpy(), 0.995)))

    def model_cells(x):
        net = eng.network(x, win)
        mass = net["mass"].sum(0)                                                                # the cell's populations' mass
        return mass, net["mixture"] * mass[None]                                                  # mass x mixture tensor (additive)

    rows = []
    # native: stride 4
    pm, pmT = model_cells(img); tm, tmT = cells(occ, T)
    rows.append(("native: stride 4, window σ 5", pm, pmT, tm, tmT, 4))
    # denser: the lattice at every phase (input cropped, never resampled)
    for st in (2, 1):
        n = (96 - 4) // 4 * 4
        D = 4 // st; out_p = None
        for dz in range(0, 4, st):
            for dy in range(0, 4, st):
                for dx in range(0, 4, st):
                    x = img[dz:dz + n, dy:dy + n, dx:dx + n]
                    m_, mT_ = model_cells(x); tm_, tmT_ = cells(occ[dz:dz + n, dy:dy + n, dx:dx + n], T[:, dz:dz + n, dy:dy + n, dx:dx + n])
                    if out_p is None:
                        c = m_.shape[0]
                        out_p = [torch.zeros(c * D, c * D, c * D, device=DEV), torch.zeros(6, c * D, c * D, c * D, device=DEV),
                                 torch.zeros(c * D, c * D, c * D, device=DEV), torch.zeros(6, c * D, c * D, c * D, device=DEV)]
                    iz, iy, ix = dz // st, dy // st, dx // st
                    out_p[0][iz::D, iy::D, ix::D] = m_; out_p[1][:, iz::D, iy::D, ix::D] = mT_
                    out_p[2][iz::D, iy::D, ix::D] = tm_; out_p[3][:, iz::D, iy::D, ix::D] = tmT_
        rows.append((f"denser: stride {st} ({D ** 3} lattice phases), window σ 5", out_p[0], out_p[1], out_p[2], out_p[3], st))
    # coarser: shape-space weighting of the native cells
    for sc, stride in ((1, 2), (2, 4), (4, 4)):
        cm, cmT = coarsen(pm, pmT, sc, stride); ctm, ctmT = coarsen(tm, tmT, sc, stride)
        eff = math.sqrt(25 + 1.25 + (4 * sc) ** 2)
        rows.append((f"coarser: stride {4 * stride}, window σ ≈ {eff:.0f} (shape-space weighted)", cm, cmT, ctm, ctmT, 4 * stride))
    fig, axs = plt.subplots(len(rows), 5, figsize=(20, 3.9 * len(rows)), facecolor="white", squeeze=False)
    for r, (name, m, mT, tm_, tmT_, st) in enumerate(rows):
        pa, ta = axis(mT, m), axis(tmT_, tm_)
        e = torch.rad2deg(torch.acos((pa * ta).sum(0).abs().clamp(max=1)))
        sel = tm_ > 0.05
        med = float(e[sel].median()) if bool(sel.any()) else float("nan")
        c = m.shape[1] // 2
        wt = (tm_[:, c] / 0.3).clamp(0, 1).cpu().numpy()[..., None]
        ax = axs[r]
        ax[0].imshow(img[:, 48].cpu().numpy(), cmap="gray"); ax[0].set_title("input (96³, the same every row)", fontsize=9)
        ax[1].imshow((dec_rgb(ta[:, :, c].cpu().numpy()) * wt).clip(0, 1), interpolation="nearest"); ax[1].set_title(f"truth: {name}", fontsize=8)
        ax[2].imshow((dec_rgb(pa[:, :, c].cpu().numpy()) * wt).clip(0, 1), interpolation="nearest"); ax[2].set_title(f"model: {tuple(m.shape)} outputs", fontsize=9)
        em = np.where(sel[:, c].cpu().numpy(), e[:, c].cpu().numpy(), np.nan)
        h = ax[3].imshow(em, cmap="RdYlGn_r", vmin=0, vmax=30, interpolation="nearest"); plt.colorbar(h, ax=ax[3], fraction=0.046)
        ax[3].set_title(f"axis error (deg): median {med:.1f}", fontsize=9)
        ax[4].hist(e[sel].cpu().numpy(), bins=30, range=(0, 45), color="#7e3ff2"); ax[4].set_title(f"error over the volume ({int(sel.sum())} outputs)", fontsize=9)
        for a_ in ax[:4]:
            a_.set_xticks([]); a_.set_yticks([])
        print(f"{name:58s} output grid {tuple(m.shape)}  median axis error {med:5.1f} deg", flush=True)
    fig.suptitle(f"One 96³ input, several output resolutions ({Path(a.model).parent.name}): denser by the lattice's phases, coarser by shape-space weighting", fontsize=12)
    out = Path(a.out) if a.out else Path(a.model).parent / "output_scales.png"
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=55); print("wrote", out)


if __name__ == "__main__":
    main()
