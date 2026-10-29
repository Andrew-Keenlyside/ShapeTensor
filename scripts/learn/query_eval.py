"""Generality of the query decoder: description scale, sub-voxel position, input resolution.

    python scripts/learn/query_eval.py --run DIR [--n 8] [--out FILE]

On held-out draws of the run's generator (each under a fixed cube rotation), against the exact target (the per-voxel
density windowed at sigma, `query.voxel_density` / `sample_density`):

1. **scale**: sigma = 1.5, 2, 3, 5, 8, 12 voxels at random interior points -- the top peak's axis error where the
   target has one clear peak, and how often a second target peak is found;
2. **position**: the same error binned by the query's distance from the nearest stride-4 cell centre (0 = on the
   lattice, 1.73 = a cell corner) -- an off-lattice query should be no worse;
3. **input resolution**: the input averaged 2x2x2 (half the resolution), the same physical points (coordinates / 2,
   shifted to the coarse voxel centres) and sigma / 2 voxels (the same physical scale), against the same target;
   reported for sigma >= 3 (at half resolution sigma / 2 >= 1.5).
Also the structure tensor's axis error at each sigma (gradient sigma 1, window sigma) as the classical reference.
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
from shapetensor.learn.generator import draw
from shapetensor.learn.losses import apply_cube, cube_transform
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.query import _sample, cube_points, sample_density, voxel_density
from shapetensor.learn.sh import peaks, peaks_gated

DEV = "cuda"
SIGMAS = (1.5, 2.0, 3.0, 5.0, 8.0, 12.0)


def st_axes(img: torch.Tensor, pts: torch.Tensor, sigma: float) -> torch.Tensor:
    """Structure tensor's least-gradient direction at points ``(Q, 3)``: gradient sigma 1, window ``sigma``; ``(3, Q)``."""
    x = gaussian(img.float(), 1.0, truncate=3.0); gz, gy, gx = torch.gradient(x)
    G = torch.stack([gaussian(c, sigma, truncate=3.0) for c in (gz * gz, gy * gy, gx * gx, gz * gy, gz * gx, gy * gx)])
    g = _sample(G, pts)                                                                                # (6, Q)
    M = torch.stack([torch.stack([g[0], g[3], g[4]], -1), torch.stack([g[3], g[1], g[5]], -1), torch.stack([g[4], g[5], g[2]], -1)], -2)
    _, V = torch.linalg.eigh(M)
    return V[..., :, 0].T


def ang(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out")
    ap.add_argument("--q", type=int, default=1500)
    a = ap.parse_args(); run = Path(a.run)
    ck = torch.load(run / "model.pt", map_location=DEV); args = ck["args"]; lmax = int(args["sh_lmax"])
    assert args.get("query"), "no query decoder in this run"
    net = PopulationNet.from_args(args).to(DEV); net.load_state_dict(ck["model"]); net.eval()
    size = args["size"]; gen = args.get("gen") or {}; sg = args.get("strength1", 1.0)
    rec = {s: {"e": [], "st": [], "two_t": 0, "two_hit": 0, "dist": [], "e_half": []} for s in SIGMAS}
    for i in range(a.n):
        size = 96                                                                                     # large enough that the half-resolution input (48³)
        s, d = draw((size,) * 3, seed=10_000_000 + i, device=DEV, strength=sg, **gen)                   # keeps the scored points away from padding
        F = voxel_density(s, d, lmax)
        perm, flips = cube_transform(torch.Generator().manual_seed(i))
        img = apply_cube(d["image"].float() / 255.0, perm, flips)
        Fr = torch.einsum("ij,j...->i...", __import__("shapetensor.learn.query", fromlist=["sh_rotation"]).sh_rotation(perm, flips, lmax, DEV),
                          apply_cube(F, perm, flips))
        g = torch.Generator(device=DEV).manual_seed(i)
        lo, hi = 30.0, size - 31.0
        pts = lo + (hi - lo) * torch.rand(a.q, 3, generator=g, device=DEV)
        half = Fn.avg_pool3d(img[None, None], 2)[0, 0]
        for sig in SIGMAS:
            sv = torch.full((a.q,), sig, device=DEV)
            ct = sample_density(Fr, pts, sv).T                                                         # (c, Q)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cp = net(img[None, None], pts[None], sv[None])["q_sh"][0].float().T
            td, tv, _ = peaks_gated(ct, lmax); pd, pv, _ = peaks_gated(cp, lmax)                       # calibrated (review 5)
            one = (tv[0] > 0) & (tv[1] == 0) & (ct[0] > 0.1 * math.sqrt(4 * math.pi))
            two = (tv[1] > 0) & (ct[0] > 0.1 * math.sqrt(4 * math.pi))
            R = rec[sig]
            if bool(one.any()):
                R["e"].append(ang(pd[0][:, one], td[0][:, one]))
                R["st"].append(ang(st_axes(img, pts[one], sig), td[0][:, one]))
                c = (pts[one] - 1.5) / 4; R["dist"].append((4 * (c - c.round())).norm(dim=1))
            R["two_t"] += int(two.sum()); R["two_hit"] += int(((pv[1] > 0) & two).sum())
            if sig >= 3:                                                                               # half resolution, same physical points and scale
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ch = net(half[None, None], ((pts - 0.5) / 2)[None], (sv / 2)[None])["q_sh"][0].float().T
                hd, _, _ = peaks_gated(ch, lmax)
                if bool(one.any()):
                    R["e_half"].append(ang(hd[0][:, one], td[0][:, one]))
        print(f"sample {i} done", flush=True)
    cat = lambda L: torch.cat(L).cpu().numpy() if L else np.zeros(0)
    rows = []
    for sig in SIGMAS:
        R = rec[sig]; e, st, eh = cat(R["e"]), cat(R["st"]), cat(R["e_half"])
        rows.append((sig, np.median(e) if e.size else np.nan, np.median(st) if st.size else np.nan, R["two_hit"] / max(R["two_t"], 1),
                     np.median(eh) if eh.size else np.nan, e.size))
        print(f"sigma {sig:5.1f}: top-peak error {rows[-1][1]:5.1f} deg (structure tensor {rows[-1][2]:5.1f}), second peak found "
              f"{rows[-1][3]:.2f}, half-resolution input {rows[-1][4]:5.1f}  (n={e.size})", flush=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    sg_ = [r[0] for r in rows]
    axs[0].plot(sg_, [r[1] for r in rows], "o-", label="query decoder"); axs[0].plot(sg_, [r[2] for r in rows], "s--", label="structure tensor (window σ)")
    axs[0].plot(sg_, [r[4] for r in rows], "^-", label="decoder, half-resolution input (σ/2 voxels)")
    axs[0].set_xscale("log"); axs[0].set_xlabel("description scale σ (voxels)"); axs[0].set_ylabel("top-peak axis error (deg, median)"); axs[0].legend(); axs[0].set_title("scale and input resolution")
    axs[1].plot(sg_, [r[3] for r in rows], "o-"); axs[1].set_xscale("log"); axs[1].set_ylim(0, 1); axs[1].set_xlabel("σ (voxels)")
    axs[1].set_title("second target peak found by the decoder")
    e5, d5 = cat(rec[5.0]["e"]), cat(rec[5.0]["dist"])
    if e5.size:
        bins = np.linspace(0, 1.8, 7); idx = np.digitize(d5, bins)
        axs[2].plot([(bins[j - 1] + bins[j]) / 2 for j in range(1, len(bins))], [np.median(e5[idx == j]) if (idx == j).any() else np.nan for j in range(1, len(bins))], "o-")
    axs[2].set_xlabel("distance from the nearest lattice cell centre (voxels)"); axs[2].set_ylabel("error at σ 5 (deg)"); axs[2].set_title("off-lattice queries")
    fig.suptitle(f"Query decoder generality: {run.name} (step {ck['step']})")
    out = Path(a.out) if a.out else run / "query_eval.png"
    plt.tight_layout(); fig.savefig(out, dpi=70); print("wrote", out)


if __name__ == "__main__":
    main()
