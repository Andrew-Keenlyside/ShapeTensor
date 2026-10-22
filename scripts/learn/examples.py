"""Examples of the population model's synthesis: layered shapes over a background, one acquisition per modality, the
labels (occupancy per layer, the shape tensors' directions), the layers' visibility, crossing groups, the boundary case.

    python scripts/learn/examples.py [--out DIR] [--seeds 0 1 2 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.generator import boundary_sample, nuisance, principal, sample
from shapetensor.learn.targets import target
from shapetensor.viz import dec_rgb

DEV = "cuda"
COLS = np.array([[0.2, 0.55, 0.95], [0.92, 0.3, 0.3], [0.3, 0.8, 0.4], [0.95, 0.75, 0.2], [0.7, 0.4, 0.9], [0.1, 0.8, 0.8], [0.9, 0.5, 0.1]])
MODS = ("hipct", "ct", "mri", "histology")


def label_panels(ax_occ, ax_dir, s, d, ky):
    """Occupancy per layer (colour per shape layer; the interface violet; the isotropic remainder grey) and the label
    directions as DEC -- a line's direction, a sheet's normal -- weighted by occupancy, by how linear/planar the tensor
    is, and by the layer's visibility."""
    occ = d["occupancy"][:, :, ky].cpu().numpy()
    rgb = np.zeros(occ.shape[1:] + (3,), np.float32); ci = 0
    for k, L in enumerate(s.layers):
        c = np.array([0.6, 0.3, 0.9]) if L.role == "interface" else np.array([0.45, 0.45, 0.45]) if L.role == "isotropic" else COLS[ci % len(COLS)]
        ci += L.role == "shape"
        rgb += occ[k][..., None] * c
    ax_occ.imshow(rgb.clip(0, 1))
    if ax_dir is None:
        return
    dec = np.zeros(occ.shape[1:] + (3,), np.float32)
    for k, L in enumerate(s.layers):
        if L.role == "isotropic":
            continue
        v, ev = principal(d["tensors"][k][:, :, ky], "min" if L.kind == "sheet" else "max")
        ev = ev.cpu().numpy()
        w = (ev[0] - ev[1]) / np.maximum(ev[0], 1e-6) if L.kind != "sheet" else (ev[1] - ev[2]) / np.maximum(ev[0], 1e-6)
        vis = d["visibility"][k][:, ky].cpu().numpy() * d.get("orientation_visibility", d["visibility"])[k][:, ky].cpu().numpy()
        dec += dec_rgb(v.cpu().numpy()) * (occ[k] * w * vis)[..., None]
    ax_dir.imshow(dec.clip(0, 1))


def target_panels(ax_n, ax_dom, s, d, ky, step=4):
    """The lattice target: the number of populations per cell (slots with mass > 0.02) and the heaviest slot's axis
    (a line's direction, a sheet's normal) as DEC × its mass."""
    t = target(s, d); c = ky // step
    N = ((t["mass"] > 0.02) & (t["kind"] == 0)).sum(0)[:, c].cpu().numpy()
    ax_n.imshow(N, cmap=matplotlib.colors.ListedColormap(["#1b1b1b", "#2a78d6", "#e34948", "#f0b429", "#ffffff"]), vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax_n.set_title(f"target: line populations per cell (mass > 0.02)\nblack 0 · blue 1 · red 2 · yellow 3 · white 4", fontsize=8)
    k = t["kind"][0][:, c]; v_line, ev = principal(t["slots"][0][:, :, c], "max"); v_sheet, _ = principal(t["slots"][0][:, :, c], "min")
    planar = ((k == 1) | (k == 3))
    v = torch.where(planar[None], v_sheet, v_line)
    aniso = torch.where(planar, (ev[1] - ev[2]) / ev[0].clamp_min(1e-6), (ev[0] - ev[1]) / ev[0].clamp_min(1e-6))   # an isotropic slot has no axis: dark
    wgt = ((t["mass"][0][:, c] / 0.3).clamp(0, 1) * aniso).cpu().numpy()
    ax_dom.imshow((dec_rgb(v.cpu().numpy()) * wgt[..., None]).clip(0, 1), interpolation="nearest")
    ax_dom.set_title("target: heaviest population's axis (DEC) × mass × anisotropy", fontsize=8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_examples")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = len(a.seeds) + 2; fig, axs = plt.subplots(rows, 9, figsize=(33, 3.9 * rows), facecolor="white")
    records = {}
    for r, seed in enumerate(a.seeds):
        s = sample((a.size,) * 3, seed=seed, device=DEV); ky = a.size // 2
        draws = [nuisance(s, seed=seed * 10 + j, device=DEV, modality=m) for j, m in enumerate(MODS)]
        ax = axs[r]
        kinds = [L.kind + ("×" if L.group == 0 else "") + ("·occ" if L.occluding else "") + ("·hollow" if L.hollow else "") for L in s.layers if L.role == "shape"]
        ax[0].imshow(s.image[:, ky].cpu().numpy(), cmap="gray"); cr = s.record.get("crossing")
        ax[0].set_title(f"seed {seed}: clean, '{s.style}', scale {s.scale:.2f}\n{', '.join(kinds) or 'background only'}" + (f"\ncrossing group {cr['n']} at {cr['theta_deg']}°" if cr else ""), fontsize=8)
        for j, d in enumerate(draws):
            ax[1 + j].imshow(d["image"][:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            keys = [k for k in d["record"] if k not in ("modality", "gamma", "psf", "window", "visibility", "orientation_visibility", "axis")]
            ax[1 + j].set_title(f"{d['record']['modality']} (axis {'zyx'[d['record']['axis']]}): " + ", ".join(keys)[:58] + f"\nvisibility {d['record']['visibility'][:-1]}", fontsize=7)
        label_panels(ax[5], ax[6], s, draws[0], ky)
        target_panels(ax[7], ax[8], s, draws[0], ky)
        ax[5].set_title("labels: occupancy (colour per layer; violet interface;\ngrey isotropic remainder)", fontsize=8)
        ax[6].set_title("labels: line direction / sheet normal (DEC)\n× occupancy × linearity/planarity × presence × orientation visibility", fontsize=8)
        for a_ in ax:
            a_.set_xticks([]); a_.set_yticks([])
        records[seed] = {"record": s.record, "layers": [{"kind": L.kind, "role": L.role, "radius": round(L.radius, 2), "length": round(L.length, 1), "density": round(L.density, 3),
                                                          "contrast": round(L.contrast, 1), "occluding": L.occluding, "hollow": L.hollow, "group": L.group,
                                                          "coherence": round(L.coherence, 1), "bend_radius": round(L.bend_radius, 1)} for L in s.layers],
                         "draws": [d["record"] for d in draws]}
        print(f"seed {seed}: {s.style} scale {s.scale:.2f} {kinds} crossing {cr}", flush=True)
    # crossing groups at two angles, hipct draws, with their labels
    ax = axs[-2]
    for j, th in enumerate((30.0, 60.0)):
        s = sample((a.size,) * 3, k_layers=0, seed=50 + j, device=DEV, style="tissue", crossing=True, cross_deg=th, scale=1.0)
        d = nuisance(s, seed=500 + j, device=DEV, modality="hipct"); ky = a.size // 2
        ax[3 * j].imshow(s.image[:, ky].cpu().numpy(), cmap="gray"); ax[3 * j].set_title(f"crossing group at {th:.0f}° (clean), weights {s.record['crossing']['weights']}", fontsize=8)
        ax[3 * j + 1].imshow(d["image"][:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[3 * j + 1].set_title(f"hipct draw, visibility {d['record']['visibility'][:-1]}", fontsize=8)
        label_panels(ax[3 * j + 2], ax[6] if j == 1 else None, s, d, ky)
        ax[3 * j + 2].set_title("occupancy per population", fontsize=8)
    ax[6].set_title("60° group: directions (DEC)", fontsize=8)
    target_panels(ax[7], ax[8], s, d, ky)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    # the boundary case
    ax = axs[-1]; Zb = 96
    for j, (step, con) in enumerate(((60.0, 20.0), (120.0, 15.0))):
        b = boundary_sample((Zb,) * 3, step=step, contrast=con, seed=5 + j, device=DEV); d = nuisance(b, seed=100 + j, device=DEV, modality="hipct")
        ax[3 * j].imshow(b.image[:, Zb // 2].cpu().numpy(), cmap="gray"); ax[3 * j].set_title(f"boundary: step {step:.0f}, fibre contrast {con:.0f}, 30° to the normal", fontsize=8)
        ax[3 * j + 1].imshow(d["image"][:, Zb // 2].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[3 * j + 1].set_title(f"hipct draw, visibility {d['record']['visibility'][:-1]}", fontsize=8)
        label_panels(ax[3 * j + 2], ax[6] if j == 1 else None, b, d, Zb // 2)
        ax[3 * j + 2].set_title("occupancy: fibres (blue), interface (violet)", fontsize=8)
    ax[6].set_title("boundary (step 120): directions (DEC)", fontsize=8)
    target_panels(ax[7], ax[8], b, d, Zb // 2)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Synthesis for the population model: layered shapes over a background, one acquisition per modality (acquisition axis drawn), "
                 "labels from the rendered shapes (shape tensors), visibility per layer; crossing groups; the boundary case", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "synthesis_examples.png", dpi=70)
    json.dump(records, open(out / "synthesis_examples.json", "w"), indent=1, default=str); print("wrote", out / "synthesis_examples.png")


if __name__ == "__main__":
    main()
