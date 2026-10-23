"""QC of the density outputs: target vs predicted orientation density on held-out samples, as polar glyphs.

    python scripts/learn/qc_density.py --run DIR [--n 2] [--out FILE]

Per held-out sample (the run's own generator and final strength), a zoom of the middle axial slice: the image; the
lattice SH head (sigma 5, one glyph per 4-voxel cell) against the target density of the target slots; and, with a
query decoder, the density at sigma 2, 5 and 10 voxels on a 2-voxel grid against the exact per-layer target. Each glyph
is the density's cut in the slice plane, its shape normalised per glyph, its size ∝ √(order-0 mass), coloured by the
DEC of its top peak; a second peak (>= 0.3 of the first) is drawn as a white tick.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import torch

from shapetensor.learn.generator import draw
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.query import layer_density
from shapetensor.learn.sh import density_sh, peaks, real_sh
from shapetensor.learn.targets import target
from shapetensor.viz import dec_rgb

DEV = "cuda"
PHI = torch.linspace(0, 2 * math.pi, 49)[:-1]


def glyphs(ax, coef, px, py, lmax, R, mref):
    """``coef (c, n)`` at plot positions ``px, py (n,)``."""
    u = torch.stack([torch.zeros_like(PHI), torch.sin(PHI), torch.cos(PHI)], 1).to(coef.device)
    f = (real_sh(u, lmax) @ coef).clamp_min(0).cpu().numpy()                                          # (A, n)
    size = np.sqrt(np.clip(coef[0].cpu().numpy() / mref, 0, 1))
    r = f / np.maximum(f.max(0, keepdims=True), 1e-9) * R * size[None]
    ph = PHI.numpy()[:, None]
    poly = np.stack([px[None] + r * np.cos(ph), py[None] + r * np.sin(ph)], -1).transpose(1, 0, 2)
    D, V = peaks(coef, lmax, rel=0.3)
    col = dec_rgb(D[0].cpu().numpy())
    keep = size > 0.08
    ax.add_collection(PolyCollection(poly[keep], facecolors=np.concatenate([col[keep], np.full((int(keep.sum()), 1), 0.4)], 1), edgecolors=col[keep], linewidths=0.8))
    two = (V[1] > 0).cpu().numpy() & keep
    d2 = D[1].cpu().numpy()
    seg = np.stack([np.stack([px[two] - 0.5 * R * d2[2, two], py[two] - 0.5 * R * d2[1, two]], -1),
                    np.stack([px[two] + 0.5 * R * d2[2, two], py[two] + 0.5 * R * d2[1, two]], -1)], 1)
    ax.add_collection(LineCollection(seg, colors="white", linewidths=1.0))
    return int(two.sum()), int(keep.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=2); ap.add_argument("--out")
    ap.add_argument("--zoom", type=int, default=40)
    a = ap.parse_args(); run = Path(a.run)
    ck = torch.load(run / "model.pt", map_location=DEV); args = ck["args"]; lmax = int(args.get("sh_lmax", 0))
    assert lmax, "no SH head in this run"
    net = PopulationNet.from_args(args).to(DEV); net.load_state_dict(ck["model"]); net.eval()
    size = args["size"]; gen = args.get("gen") or {}; sg = args.get("strength1", 1.0)
    has_q = bool(args.get("query"))
    cols = ["image", "lattice σ5: target", "lattice σ5: predicted"] + ([f"query σ{s}: {w}" for s in (2, 5, 10) for w in ("target", "predicted")] if has_q else [])
    fig, axs = plt.subplots(a.n, len(cols), figsize=(4.2 * len(cols), 4.4 * a.n), facecolor="white", squeeze=False)
    Z = a.zoom; z0 = size // 2 - Z // 2; z0 -= z0 % 4
    for i in range(a.n):
        s, d = draw((size,) * 3, seed=10_000_000 + i, device=DEV, strength=sg, **gen)
        t = target(s, d)
        x = (d["image"].float() / 255.0)[None, None]
        k = size // 2
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(x)
        img = d["image"][k].float().cpu().numpy()
        ax = axs[i, 0]; ax.imshow(img[z0:z0 + Z, z0:z0 + Z], cmap="gray", extent=(-0.5, Z - 0.5, Z - 0.5, -0.5)); ax.set_title(f"held-out {i}: image ({s.style}, {d['record']['modality']})", fontsize=9)
        kc = k // 4
        cy, cx = np.mgrid[z0 // 4:(z0 + Z) // 4, z0 // 4:(z0 + Z) // 4]
        px, py = (4 * cx + 1.5 - z0).ravel(), (4 * cy + 1.5 - z0).ravel()
        ct = density_sh(t["slots"][None].float(), t["mass"][None].float(), lmax)[0][:, kc, cy, cx].reshape(-1, cx.size)
        cp = P["sh"][0].float()[:, kc, cy, cx].reshape(-1, cx.size)
        mref = float(torch.quantile(ct[0][ct[0] > 0], 0.9)) if bool((ct[0] > 0).any()) else 1.0
        for j, (c, name) in enumerate(((ct, "target"), (cp, "predicted"))):
            ax = axs[i, 1 + j]; ax.imshow(img[z0:z0 + Z, z0:z0 + Z], cmap="gray", alpha=0.5, extent=(-0.5, Z - 0.5, Z - 0.5, -0.5))
            n2, nk = glyphs(ax, c, px, py, lmax, 1.9, mref); ax.set_title(f"lattice σ5, {name}: {n2}/{nk} with 2 peaks", fontsize=9)
        if has_q:
            gy, gx = np.mgrid[z0:z0 + Z:2, z0:z0 + Z:2] + 0.5
            pts = torch.tensor(np.stack([np.full(gy.size, k + 0.0), gy.ravel(), gx.ravel()], 1), device=DEV, dtype=torch.float32)
            for sj, sig in enumerate((2.0, 5.0, 10.0)):
                sv = torch.full((pts.shape[0],), sig, device=DEV)
                qt = layer_density(s, d, pts, sv, lmax).T
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    qp = net(x, pts[None], sv[None])["q_sh"][0].float().T
                mq = float(torch.quantile(qt[0][qt[0] > 0], 0.9)) if bool((qt[0] > 0).any()) else 1.0
                for j, (c, name) in enumerate(((qt, "target"), (qp, "predicted"))):
                    ax = axs[i, 3 + 2 * sj + j]; ax.imshow(img[z0:z0 + Z, z0:z0 + Z], cmap="gray", alpha=0.5, extent=(-0.5, Z - 0.5, Z - 0.5, -0.5))
                    n2, nk = glyphs(ax, c, gx.ravel() - z0, gy.ravel() - z0, lmax, 0.95, mq); ax.set_title(f"query σ{sig:g}, {name}: {n2}/{nk} two peaks", fontsize=9)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, Z - 0.5); ax.set_ylim(Z - 0.5, -0.5)
    fig.suptitle(f"Density QC: {run.name} (step {ck['step']}); glyph = in-plane cut, colour = top peak DEC, white tick = second peak", fontsize=11)
    out = Path(a.out) if a.out else run / "qc_density.png"
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=60); print("wrote", out)


if __name__ == "__main__":
    main()
