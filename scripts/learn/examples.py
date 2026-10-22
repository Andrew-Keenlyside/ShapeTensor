"""Examples of the population model's synthesis: layered random shapes with an orientation state, nuisance draws, labels.

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

from shapetensor.learn.generator import nuisance, sample, target
from shapetensor.lift import hemisphere
from shapetensor.viz import dec_rgb

DEV = "cuda"
COLS = np.array([[0.2, 0.55, 0.95], [0.92, 0.3, 0.3], [0.3, 0.8, 0.4], [0.95, 0.75, 0.2]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_examples")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = hemisphere(160, DEV)
    rows = len(a.seeds); fig, axs = plt.subplots(rows, 8, figsize=(30, 3.9 * rows), facecolor="white")
    records = {}
    for r, seed in enumerate(a.seeds):
        s = sample((a.size,) * 3, seed=seed, device=DEV); ky = a.size // 2
        draws = [nuisance(s, seed=seed * 10 + j, device=DEV) for j in range(3)]
        tg = target(s, draws[0], grid)
        ax = axs[r]
        ax[0].imshow(s.image[:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"seed {seed}: clean composite ({len(s.layers)} layers: {', '.join(L.kind for L in s.layers)})", fontsize=9)
        for j, d in enumerate(draws):
            ax[1 + j].imshow(d["image"][:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            keys = [k for k in d["record"] if k not in ("quantised",)]; ax[1 + j].set_title("nuisance draw: " + ", ".join(str(k) for k in keys)[:60], fontsize=8)
        occ = draws[0]["occupancy"][:, ky].cpu().numpy()
        rgb = np.zeros(occ.shape[1:] + (3,), np.float32)
        for k in range(occ.shape[0]):
            rgb += occ[k][..., None] * COLS[k % 4]
        ax[4].imshow(rgb.clip(0, 1)); ax[4].set_title("labels: occupancy per layer (colour = layer)", fontsize=9)
        # the dominant layer's direction (DEC) weighted by its occupancy and its linearity
        dom = occ.argmax(0); fld = draws[0]["fields"][:, :, ky].cpu().numpy()
        dec = np.zeros(occ.shape[1:] + (3,), np.float32)
        for k, L in enumerate(s.layers):
            w = (dom == k) * occ[k] * (1.0 if L.kind == "line" else 0.35 if L.kind == "sheet" else 0.1)
            dec += dec_rgb(fld[k]) * w[..., None]
        ax[5].imshow(dec.clip(0, 1)); ax[5].set_title("labels: direction (line) / normal (sheet) as DEC, × occupancy × linearity", fontsize=8)
        # the target on the sphere at the slice: the number of lobes/rings is implicit; show the ODF's anisotropy (max/mean)
        odf = tg["odf"]; aniso = (odf.max(0).values / odf.mean(0).clamp_min(1e-6))[:, ky // 4, :].cpu().numpy()
        h = ax[6].imshow(aniso, cmap="magma", vmin=1, vmax=6, interpolation="nearest"); ax[6].set_title("target: anisotropy of the mixture on the sphere (max/mean)", fontsize=8)
        # the target distribution at the centre as a polar-ish profile in the x-z plane and the y-z plane
        cz, cyy, cx = a.size // 8, ky // 4, a.size // 8
        g = grid.cpu().numpy(); o = odf[:, cz, cyy, cx].cpu().numpy()
        phi = np.arange(0, 180, 2.0)
        for plane, (e1, e2), col, lab in ((0, (np.array([0, 0, 1.0]), np.array([1.0, 0, 0])), "#2a5db0", "x-z plane"), (1, (np.array([0, 1.0, 0]), np.array([1.0, 0, 0])), "#e34948", "y-z plane")):
            circ = np.outer(np.cos(np.radians(phi)), e1) + np.outer(np.sin(np.radians(phi)), e2)
            K = np.exp((np.abs(circ @ g.T) ** 2 - 1) / (2 * math.sin(math.radians(9)) ** 2)); K /= K.sum(1, keepdims=True)
            ax[7].plot(phi, K @ o, color=col, lw=2, label=lab)
        ax[7].legend(fontsize=8); ax[7].set_title("target at the centre: the orientation state on two great circles", fontsize=8); ax[7].set_xlim(0, 180)
        for a_ in ax[:7]:
            a_.set_xticks([]); a_.set_yticks([])
        records[seed] = {"layers": [{"kind": L.kind, "radius": round(L.radius, 2), "length": round(L.length, 1), "density": round(L.density, 3), "contrast": round(L.contrast, 1),
                                     "occluding": L.occluding, "coherence": round(L.coherence, 1), "kappa": round(L.kappa, 1)} for L in s.layers],
                         "draws": [d["record"] for d in draws]}
        print(f"seed {seed}: {json.dumps(records[seed]['layers'])}", flush=True)
    fig.suptitle("Synthesis for the population model: layered random shapes (lines, sheets, blobs) over an anatomix-style background; three nuisance draws; labels; the target on the sphere", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "synthesis_examples.png", dpi=75)
    json.dump(records, open(out / "synthesis_examples.json", "w"), indent=1); print("wrote", out / "synthesis_examples.png")


if __name__ == "__main__":
    main()
