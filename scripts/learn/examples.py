"""Examples of the population model's synthesis: layered random shapes with an orientation state over a background,
one nuisance draw per modality profile, the labels, and the full-volume target (every voxel has a state).

    python scripts/learn/examples.py [--out DIR] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.generator import boundary_sample, nuisance, sample, target, westin
from shapetensor.lift import hemisphere
from shapetensor.viz import dec_rgb

DEV = "cuda"
COLS = np.array([[0.2, 0.55, 0.95], [0.92, 0.3, 0.3], [0.3, 0.8, 0.4], [0.95, 0.75, 0.2], [0.7, 0.4, 0.9], [0.5, 0.5, 0.5]])
MODS = ("hipct", "ct", "mri", "histology")


def westin_rgb(odf, grid, ky):
    """Westin shares at a lattice slice as RGB: red linear (lines), green planar (sheets, interfaces), blue spherical (blobs, flat)."""
    w = westin(odf[:, :, ky, :], grid).permute(1, 2, 0).cpu().numpy()
    return w.clip(0, 1)


def profile(ax, odf_c, grid):
    g = grid.cpu().numpy(); phi = np.arange(0, 180, 2.0)
    for (e1, e2), col, lab in (((np.array([0, 0, 1.0]), np.array([1.0, 0, 0])), "#2a5db0", "x-z plane"), ((np.array([0, 1.0, 0]), np.array([1.0, 0, 0])), "#e34948", "y-z plane")):
        circ = np.outer(np.cos(np.radians(phi)), e1) + np.outer(np.sin(np.radians(phi)), e2)
        K = np.exp((np.abs(circ @ g.T) ** 2 - 1) / (2 * math.sin(math.radians(9)) ** 2)); K /= K.sum(1, keepdims=True)
        ax.plot(phi, K @ odf_c, color=col, lw=2, label=lab)
    ax.legend(fontsize=7); ax.set_xlim(0, 180)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_examples")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = hemisphere(160, DEV)
    rows = len(a.seeds) + 1; fig, axs = plt.subplots(rows, 9, figsize=(33, 3.8 * rows), facecolor="white")
    records = {}
    for r, seed in enumerate(a.seeds):
        s = sample((a.size,) * 3, seed=seed, device=DEV); ky = a.size // 2
        draws = [nuisance(s, seed=seed * 10 + j, device=DEV, modality=m) for j, m in enumerate(MODS)]
        tg = target(s, draws[0], grid)
        ax = axs[r]
        kinds = [L.kind for L in s.layers]
        ax[0].imshow(s.image[:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
        ax[0].set_title(f"seed {seed}: clean, background '{s.style}', scale {s.scale:.2f}\n{len(kinds)} layers: {', '.join(kinds[:-2])} + interface + isotropic", fontsize=8)
        for j, d in enumerate(draws):
            ax[1 + j].imshow(d["image"][:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            keys = [k for k in d["record"] if k not in ("quantised", "modality", "gamma", "psf")]
            ax[1 + j].set_title(f"{d['record']['modality']}: " + ", ".join(str(k) for k in keys)[:70], fontsize=7)
        occ = draws[0]["occupancy"][:, ky].cpu().numpy()
        rgb = np.zeros(occ.shape[1:] + (3,), np.float32)
        for k in range(occ.shape[0]):
            rgb += occ[k][..., None] * COLS[k % len(COLS)]
        ax[5].imshow(rgb.clip(0, 1)); ax[5].set_title("labels: occupancy per layer (grey = isotropic remainder,\nviolet = interface)", fontsize=8)
        dom = occ.argmax(0); fld = draws[0]["fields"][:, :, ky].cpu().numpy()
        dec = np.zeros(occ.shape[1:] + (3,), np.float32)
        for k, L in enumerate(s.layers):
            w = (dom == k) * occ[k] * (1.0 if L.kind == "line" else 0.45 if L.kind == "sheet" else 0.1)
            dec += dec_rgb(fld[k]) * w[..., None]
        ax[6].imshow(dec.clip(0, 1)); ax[6].set_title("labels: direction (line) / normal (sheet) as DEC, × occupancy", fontsize=8)
        ax[7].imshow(westin_rgb(tg["odf"], grid, ky // 4), interpolation="nearest")
        ax[7].set_title("target, every cell: the state of the shapes present (Westin)\nred linear · green planar · blue flat", fontsize=8)
        ax[8].imshow(tg["fraction"][:, ky // 4].cpu().numpy(), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax[8].set_title("target, every cell: shape fraction of the window\n(0 flat background → 1 all shape); classes alongside", fontsize=8)
        for a_ in ax[:9]:
            a_.set_xticks([]); a_.set_yticks([])
        records[seed] = {"style": s.style, "scale": round(s.scale, 3),
                         "layers": [{"kind": L.kind, "radius": round(L.radius, 2), "length": round(L.length, 1), "density": round(L.density, 3), "contrast": round(L.contrast, 1),
                                     "occluding": L.occluding, "coherence": round(L.coherence, 1), "kappa": round(L.kappa, 1)} for L in s.layers],
                         "draws": [d["record"] for d in draws]}
        print(f"seed {seed}: {s.style} scale {s.scale:.2f} " + json.dumps(records[seed]["layers"][:-2]), flush=True)
    # the boundary case: faint fibres through an intensity step
    ax = axs[-1]; Zb = 96
    for j, (step, con) in enumerate(((60.0, 20.0), (60.0, 40.0), (120.0, 15.0))):
        b = boundary_sample((Zb,) * 3, step=step, contrast=con, seed=5 + j, device=DEV); d = nuisance(b, seed=100 + j, device=DEV, modality="hipct")
        ax[3 * j].imshow(b.image[:, Zb // 2].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[3 * j].set_title(f"boundary case: step {step:.0f}, fibre contrast {con:.0f}, 30° to the normal (clean)", fontsize=8)
        ax[3 * j + 1].imshow(d["image"][:, Zb // 2].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[3 * j + 1].set_title("hipct draw: " + ", ".join(k for k in d["record"] if k not in ("quantised", "modality", "gamma", "psf"))[:60], fontsize=7)
        tb = target(b, d, grid); g = grid.cpu().numpy(); phi = np.arange(0, 180, 1.0)
        circ = np.outer(np.cos(np.radians(phi)), np.array([1.0, 0, 0])) + np.outer(np.sin(np.radians(phi)), np.array([0, 0, 1.0]))   # the z-x plane: holds the normal (0°) and the line (30°)
        K = np.exp((np.abs(circ @ g.T) ** 2 - 1) / (2 * math.sin(math.radians(6)) ** 2)); K /= K.sum(1, keepdims=True)
        zc = Zb // 8; cl = tb["classes"][:, zc].mean((1, 2)).cpu().numpy()
        for zz_, col, lab in ((zc, "#1f9d55", f"at the step (classes: line {cl[0]:.2f}, interface {cl[3]:.2f}, flat {cl[4]:.2f})"), (2, "#c0392b", "away from it")):
            o = tb["odf"][:, zz_].mean((1, 2)).cpu().numpy(); ax[3 * j + 2].plot(phi, K @ o, color=col, lw=2, label=lab)
        ax[3 * j + 2].axvline(30, color="#888", lw=0.8, ls="--"); ax[3 * j + 2].axvline(90, color="#888", lw=0.8, ls=":")
        ax[3 * j + 2].legend(fontsize=7, loc="upper right"); ax[3 * j + 2].set_xlim(0, 180); ax[3 * j + 2].set_xlabel("angle from the step's normal, z-x plane (°)", fontsize=8)
        ax[3 * j + 2].set_title("target state: the line's lobe (30°) with the step's ring (90°) at the step; the lobe alone away", fontsize=8)
    for j in range(3):
        for a_ in (ax[3 * j], ax[3 * j + 1]):
            a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Synthesis for the population model: layered shapes over a background (regions / tissue / cells), one draw per modality profile (HiP-CT, CT, MRI, histology), the labels, "
                 "and the full-volume target; last row: faint fibres through an intensity step", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "synthesis_examples.png", dpi=72)
    json.dump(records, open(out / "synthesis_examples.json", "w"), indent=1); print("wrote", out / "synthesis_examples.png")


if __name__ == "__main__":
    main()
