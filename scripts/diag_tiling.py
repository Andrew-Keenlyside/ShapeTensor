"""dev-vit tiling: block and edge effects of each blend, on the HiP-CT VOI (S-21-46 20 µm cortex, a 160 x 256 x 256
block at its centre) and on synthetic fibre-free texture.

Per configuration (blend, stride, shifts):
  * block index: the features' step ||F(x + e) - F(x)|| (unit features) against the position's phase in the 64-voxel tile
    lattice, per axis; the step at the core faces (phase 63) over the median step (1 = no block effect);
  * patch-grid index: the same profile at period 8, max over median;
  * translation consistency: features of the volume shifted by (13, 21, 7) voxels against the shifted features,
    median and p95 of ||F_shifted - shift(F)|| over the common interior (tiling that leaves no trace commutes with
    translation; seams and grid do not);
  * seconds; and a figure: the first three principal components of a mid slice as RGB per configuration, and the
    step profiles.

    python scripts/diag_tiling.py [OUT_DIR]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor.features import make_engine
from shapetensor.fibre_sets import iso_textures

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/tiling")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"; SHIFT = (13, 21, 7)
S2 = [(0, 0, 0), (4, 4, 4)]; S4 = [(0, 0, 0), (2, 2, 2), (4, 4, 4), (6, 6, 6)]
S8 = [(k, k, k) for k in range(8)]
import os
CONFIGS2 = [("core 64 (first version)", dict(blend="core", shifts=S2)),
            ("sphere, stride 64", dict(blend="sphere", stride=64, shifts=S2)),
            ("sphere, stride 64, grid filter", dict(blend="sphere", stride=64, shifts=S2, grid_filter=True)),
            ("hann, stride 64, grid filter", dict(blend="hann", stride=64, shifts=S2, grid_filter=True)),
            ("sphere, stride 48, grid filter", dict(blend="sphere", stride=48, shifts=S2, grid_filter=True)),
            ("sphere, stride 64, 8 diagonal shifts", dict(blend="sphere", stride=64, shifts=S8)),
            ("sphere, stride 64, 1 shift, grid filter", dict(blend="sphere", stride=64, shifts=[(0, 0, 0)], grid_filter=True))]
CONFIGS = [("core 64 (first version)", dict(blend="core", shifts=S2)),
           ("hann, stride 64", dict(blend="hann", stride=64, shifts=S2)),
           ("sphere, stride 64", dict(blend="sphere", stride=64, shifts=S2)),
           ("gauss, stride 64", dict(blend="gauss", stride=64, shifts=S2)),
           ("sphere, stride 48", dict(blend="sphere", stride=48, shifts=S2)),
           ("hann, stride 32", dict(blend="hann", stride=32, shifts=S2)),
           ("hann, stride 64, 4 shifts", dict(blend="hann", stride=64, shifts=S4)),
           ("sphere, stride 48, 4 shifts", dict(blend="sphere", stride=48, shifts=S4))]


def step_profiles(F, period, interior=32):
    """Mean step per phase (mod ``period``) along each axis, over the interior."""
    out = []
    for ax in range(3):
        d = (F.narrow(ax + 1, 1, F.shape[ax + 1] - 1) - F.narrow(ax + 1, 0, F.shape[ax + 1] - 1)).norm(dim=0)   # (.., n-1, ..)
        sl = [slice(interior, -interior)] * 3; sl[ax] = slice(None); d = d[tuple(sl)]
        m = d.movedim(ax, 0).reshape(d.shape[ax], -1).mean(1)                                      # per position along the axis
        pos = torch.arange(m.shape[0], device=m.device)
        prof = torch.zeros(period, device=m.device).index_add_(0, pos % period, m) / torch.bincount(pos % period, minlength=period).clamp_min(1)
        out.append(prof.cpu().numpy())
    return out


def pca_rgb(F):
    C = F.shape[0]; X = F.reshape(C, -1).T.astype(np.float64); X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 50000), replace=False)], full_matrices=False)
    Y = X @ Vt[:3].T; lo, hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
    return np.clip((Y - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(F.shape[1], F.shape[2], 3)


def main():
    global CONFIGS
    if os.environ.get("ROUND") == "2":
        CONFIGS = CONFIGS2
    tag = "_round2" if os.environ.get("ROUND") == "2" else ""
    OUT.mkdir(parents=True, exist_ok=True)
    a = zarr.open_group(SRC, mode="r")["0"]; c = [n // 2 for n in a.shape]
    real = torch.as_tensor(np.asarray(a[c[0] - 80:c[0] + 80, c[1] - 128:c[1] + 128, c[2] - 128:c[2] + 128]).astype(np.float32), device=DEV)
    tex = iso_textures("calibrated", DEV)[0]
    res, slices, profs = {}, {}, {}
    for name, kw in CONFIGS:
        eng = make_engine("anatomix_devvit", device=DEV, **kw)
        r = {}
        for what, v in (("real", real), ("texture", tex)):
            torch.cuda.synchronize(); t0 = time.time()
            F = eng(v).float(); torch.cuda.synchronize(); sec = time.time() - t0
            p64, p8 = step_profiles(F, 64), step_profiles(F, 8)
            block = [float(p[63] / np.median(p)) for p in p64]
            grid = [float(p.max() / np.median(p)) for p in p8]
            Fs = eng(v[SHIFT[0]:, SHIFT[1]:, SHIFT[2]:].contiguous()).float()
            ref = F[:, SHIFT[0]:, SHIFT[1]:, SHIFT[2]:]
            inner = (slice(None),) + tuple(slice(32, n - 32) for n in Fs.shape[1:])
            e = (Fs[inner] - ref[inner]).norm(dim=0).flatten()
            e = e[torch.randperm(e.numel(), device=e.device)[:2_000_000]]
            r[what] = {"seconds": round(sec, 1), "block index z/y/x": [round(b, 2) for b in block], "block index mean": round(float(np.mean(block)), 2),
                       "patch-grid index mean": round(float(np.mean(grid)), 2),
                       "translation error median / p95": [round(float(e.median()), 4), round(float(torch.quantile(e, 0.95)), 4)]}
            if what == "real":
                slices[name] = pca_rgb(F[:, F.shape[1] // 2].cpu().numpy()); profs[name] = p64
            del F, Fs, ref; torch.cuda.empty_cache()
        res[name] = r; print(name, json.dumps(r), flush=True)
        json.dump(res, open(OUT / f"diag_tiling{tag}.json", "w"), indent=1)
    n = len(CONFIGS)
    fig = plt.figure(figsize=(3.2 * min(n, 4), 3.4 * ((n + 3) // 4) + 3.0), facecolor="white")
    for i, (name, _) in enumerate(CONFIGS):
        ax = fig.add_subplot((n + 3) // 4 + 1, 4, i + 1); ax.imshow(slices[name], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        r = res[name]["real"]
        ax.set_title(f"{name}\nblock {r['block index mean']}, grid {r['patch-grid index mean']}, shift err {r['translation error median / p95'][0]}", fontsize=8)
    ax = fig.add_subplot((n + 3) // 4 + 1, 1, (n + 3) // 4 + 1)
    for name, _ in CONFIGS:
        p = np.mean([pp / np.median(pp) for pp in profs[name]], axis=0); ax.plot(np.arange(64), p, label=name)
    ax.set_xlabel("phase in the 64-voxel tile lattice (63 = core face)"); ax.set_ylabel("step / median step"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("dev-vit tiling on the HiP-CT VOI: features of a mid axial slice (PCs 1-3 as RGB) and the step profile", fontsize=10)
    plt.tight_layout(); fig.savefig(OUT / f"diag_tiling{tag}.png", dpi=110); print("figure", OUT / f"diag_tiling{tag}.png")


if __name__ == "__main__":
    main()
