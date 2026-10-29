"""The shape grammar, one family per row: clean in three planes, a HiP-CT acquisition, the label directions and the
per-voxel descriptors (thickness, hollowness, angularity).

    python scripts/learn/grammar.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.generator import Layer, Sample, _composite, direction_field, layer_density, nuisance, principal, render_layer
from shapetensor.viz import dec_rgb

DEV = "cuda"
S = (96, 96, 96)


def family(name, kind, fraction=0.12, coherence=40.0, seed=0, contrast=70.0, **kw):
    g = torch.Generator(device=DEV).manual_seed(seed); ones = torch.ones(S, dtype=torch.bool, device=DEV)
    fld = direction_field(S, coherence, g, DEV)
    L = Layer(kind, fld, kw.pop("radius", 1.5), kw.pop("length", 60.0), 0.0, ones, contrast, **kw)
    L.density = layer_density(L, fraction)
    L.occupancy, L.tensor = render_layer(L, S, g, DEV)
    layers = [L]; bg = torch.full(S, 100.0, device=DEV) + 3 * torch.randn(S, device=DEV, generator=g)
    img = _composite(bg, torch.full(S, 100.0, device=DEV), layers, DEV)
    return name, Sample(img, layers, bg, torch.zeros(S, dtype=torch.long, device=DEV), "flat", 1.0, "general", {}, torch.full(S, 100.0, device=DEV))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_stress")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [family("fibres (solid, Gaussian profile)", "line", radius=1.2),
            family("hollow fibres / vessels (ring)", "line", radius=4.0, section="ring", wall=0.35, fraction=0.15, seed=1),
            family("ribbons (ellipse 0.3)", "line", radius=4.0, section="ellipse", section_aspect=0.3, seed=2),
            family("square fibres (superellipse 6)", "line", radius=3.0, section="superellipse", section_exp=6.0, seed=3),
            family("membranes (sheets)", "sheet", radius=1.0, length=18.0, fraction=0.08, seed=4),
            family("bilayer membranes (hollow sheets)", "sheet", radius=0.8, length=18.0, bilayer=4.0, fraction=0.08, seed=5),
            family("blobs (cells)", "blob", radius=2.5, length=4.0, seed=6),
            family("smooth ellipsoids", "solid", radius=4.0, primitive="superquadric", axes=(1.0, 1.3, 1.8), exps=(1.0, 1.0), seed=7),
            family("box-like (superquadric 0.25)", "solid", radius=4.0, primitive="superquadric", axes=(1.0, 1.0, 1.4), exps=(0.25, 0.25), seed=8),
            family("needles", "solid", radius=1.5, primitive="superquadric", axes=(1.0, 1.0, 5.0), exps=(1.0, 1.0), seed=9),
            family("platelets (polyhedral)", "solid", radius=2.0, primitive="polyhedron", axes=(4.0, 3.0, 1.0), seed=10),
            family("crystals (polyhedra)", "solid", radius=4.0, primitive="polyhedron", axes=(1.0, 1.0, 1.5), seed=11),
            family("shards", "solid", radius=4.0, primitive="shard", axes=(1.2, 1.0, 2.0), seed=12),
            family("vesicles (shells)", "solid", radius=6.0, primitive="superquadric", axes=(1.0, 1.0, 1.2), exps=(1.0, 1.0), shell=1.5, fraction=0.2, seed=13)]
    fig, axs = plt.subplots(len(rows), 8, figsize=(26, 3.3 * len(rows)), facecolor="white", squeeze=False)
    c = S[0] // 2
    for r, (name, s) in enumerate(rows):
        ax = axs[r]; L = s.layers[0]
        for j, (sl, lab) in enumerate((((slice(None), c), "z-x"), ((c,), "y-x"), ((slice(None), slice(None), c), "z-y"))):
            ax[j].imshow(s.image[sl].cpu().numpy(), cmap="gray"); ax[j].set_title(f"{name}\nclean, {lab}" if j == 0 else f"clean, {lab}", fontsize=8)
        d = nuisance(s, seed=r, device=DEV, modality="hipct", strength=0.5)
        ax[3].imshow(d["image"][:, c].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
        ax[3].set_title(f"hipct s=0.5: presence {d['record']['visibility'][0]}, orient {d['record']['orientation_visibility'][0]}", fontsize=7)
        occ = L.occupancy[:, c].cpu().numpy()
        v, ev = principal(L.tensor[:, :, c], "min" if L.kind == "sheet" else "max"); ev = ev.cpu().numpy()
        aniso = (ev[0] - ev[1]) / np.maximum(ev[0], 1e-6) if L.kind != "sheet" else (ev[1] - ev[2]) / np.maximum(ev[0], 1e-6)
        ax[4].imshow((dec_rgb(v.cpu().numpy()) * (occ * np.maximum(aniso, 0.25))[..., None]).clip(0, 1))
        ax[4].set_title("label: axis (DEC) × occupancy\n(dim where the tensor is isotropic)", fontsize=8)
        for j, (k, vmax, cmap) in enumerate(((0, 8.0, "viridis"), (1, 1.0, "magma"), (2, 1.0, "cividis"))):
            m = L.desc[k][:, c].cpu().numpy() * (occ > 0.2)
            ax[5 + j].imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            vals = L.desc[k][L.occupancy > 0.5]
            ax[5 + j].set_title(f"{('thickness (vox)', 'hollowness', 'angularity')[k]}: median {float(vals.median()) if vals.numel() else 0:.2f}", fontsize=8)
        for a_ in ax:
            a_.set_xticks([]); a_.set_yticks([])
        print(name, flush=True)
    fig.suptitle("The shape grammar: every family drawn with exact labels -- the label's axis and the per-voxel descriptors (thickness, hollowness, angularity)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / "shape_grammar.png", dpi=55)
    print("wrote", out / "shape_grammar.png")


if __name__ == "__main__":
    main()
