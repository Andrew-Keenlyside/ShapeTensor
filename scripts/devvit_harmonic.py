"""From dev-vit to separated populations by a harmonic readout of its orientation code.

The signature D(theta) of dev-vit holds variance above the second order (l <= 2: 42%, l <= 4: 70%). A linear readout
M from the window-mean feature (32) to the in-plane harmonics h(theta) = [1, cos 2theta, sin 2theta, (cos 4theta,
sin 4theta)] is fitted on the 12 reference bundles (every lattice position a sample) and on fibre-free texture (h = 0).
At a crossing position the readout gives a degree-2 or degree-4 orientation function f(theta) = M W(x) . h(theta),
evaluated on 180 angles; its lobes are the populations. Degree 2 is the control (a planar tensor: one axis or none);
degree 4 can hold two lobes.

    python scripts/devvit_harmonic.py [--out DIR]
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

from shapetensor.features import make_engine
from shapetensor.phantoms import Grid, build
from shapetensor.validate import ANGLES, tex_for

import importlib.util, sys
spec = importlib.util.spec_from_file_location("DS", str(Path(__file__).with_name("devvit_separation.py")))
DS = importlib.util.module_from_spec(spec); spec.loader.exec_module(DS)
DEV, STEP, ORIGIN = DS.DEV, DS.STEP, DS.ORIGIN


def harmonics(theta_deg, lmax):
    th = torch.deg2rad(torch.as_tensor(np.asarray(theta_deg, np.float32), device=DEV))
    cols = [torch.ones_like(th)] + [f(m * th) for m in range(2, lmax + 1, 2) for f in (torch.cos, torch.sin)]
    return torch.stack(cols, 1)                                                                 # (n, 1 + lmax)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/devvit_harmonic")
    ap.add_argument("--ridge", type=float, default=1e-2)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tex, texture = tex_for("calibrated"); grid = Grid()
    vit = make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32)
    zs, ys, xs = grid.inner; core = (slice(None), slice(zs.start // STEP, zs.stop // STEP), slice(ys.start // STEP, ys.stop // STEP), slice(xs.start // STEP, xs.stop // STEP))
    # samples: every core lattice position of the 12 bundles (target h(theta)) and of texture (target 0)
    Xs, ths, is_tex = [], [], []
    for k, ang in enumerate(ANGLES):
        img = build("bundle", 100 + k, grid, tex, texture=texture, device=DEV, angle=float(ang))[0]
        W = DS.window_mean(vit(torch.as_tensor(img, device=DEV).float()))[core].reshape(32, -1).T
        Xs.append(W); ths.append(torch.full((W.shape[0],), float(ang), device=DEV)); is_tex.append(torch.zeros(W.shape[0], dtype=torch.bool, device=DEV))
    img = build("texture", 7, grid, tex, texture=texture, device=DEV)[0]
    W = DS.window_mean(vit(torch.as_tensor(img, device=DEV).float()))[core].reshape(32, -1).T
    Xs.append(W); ths.append(torch.zeros(W.shape[0], device=DEV)); is_tex.append(torch.ones(W.shape[0], dtype=torch.bool, device=DEV))
    X = torch.cat(Xs); th = torch.cat(ths); tx = torch.cat(is_tex)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6); Xn = (X - mu) / sd
    Xn1 = torch.cat([Xn, torch.ones(Xn.shape[0], 1, device=DEV)], 1)                            # with an intercept
    readouts = {}
    for lmax in (2, 4):
        H = harmonics(th.cpu().numpy(), lmax); H[tx] = 0                                        # texture: no orientation function
        M = torch.linalg.solve(Xn1.T @ Xn1 + a.ridge * Xn.shape[0] * torch.eye(Xn1.shape[1], device=DEV), Xn1.T @ H)   # ridge regression (33, 1 + lmax)
        fit = 1 - float(((Xn1 @ M - H) ** 2).sum() / ((H - H.mean(0)) ** 2).sum())
        readouts[lmax] = M; print(f"degree-{lmax} readout fitted: R^2 {fit:.3f}", flush=True)
    theta = np.arange(0, 180, 1.0); results = {}
    fig, axs = plt.subplots(4, 3, figsize=(15, 15), facecolor="white")
    for row, (name, kind, kw, g) in enumerate((("crossing 90", "crossing", {"cross_deg": 90.0}, Grid()), ("crossing 60", "crossing", {"cross_deg": 60.0}, Grid()),
                                               ("crossing 45", "crossing", {"cross_deg": 45.0}, Grid()), ("band crossing 90", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192))))):
        img, tr, meta = build(kind, 1, g, tex=tex, texture=texture, device=DEV, **kw); v = torch.as_tensor(img, device=DEV).float()
        W = DS.window_mean(vit(v)); cs = W.shape[1:]
        Wn = torch.cat([((W.reshape(32, -1).T - mu) / sd), torch.ones(W[0].numel(), 1, device=DEV)], 1)
        ax = axs[row]; Z, Y, X_ = v.shape; cz, cy, cx = (Z // 2 - ORIGIN) // STEP, (Y // 2 - ORIGIN) // STEP, (X_ // 2 - ORIGIN) // STEP
        c = cz * cs[1] * cs[2] + cy * cs[2] + cx
        for j, lmax in enumerate((2, 4)):
            coef = Wn @ readouts[lmax]                                                            # (V, 1 + lmax)
            f = (harmonics(theta, lmax) @ coef.T).clamp_min(0)                                    # (180, V): the orientation function, non-negative
            dirs, mass, pk = DS.lobes_1d(f, theta)
            dirs = dirs.reshape(2, 3, *cs); mass = mass.reshape(2, *cs)
            sc = DS.score(dirs, mass, tr, g, cs); results.setdefault(name, {})[f"degree {lmax}"] = sc
            print(f"{name:18s} degree-{lmax} readout: {json.dumps(sc)}", flush=True)
            ax[j].plot(theta, f[:, c].cpu().numpy(), color="#2a5db0", lw=2)
            for ang_ in meta["angles"]:
                ax[j].axvline(ang_ % 180, color="#e34948", ls="--")
            ax[j].set_xlim(0, 180); ax[j].set_title(f"{name}: degree-{lmax} orientation function at the centre (red: truth)", fontsize=10)
        N = (mass > 0).sum(0)[:, cy, :].cpu().numpy()
        ax[2].imshow(N, cmap=matplotlib.colors.ListedColormap(["#222222", "#2a78d6", "#e34948"]), vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax[2].set_title("N per cell, degree-4 readout", fontsize=10); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle("dev-vit -> populations by a harmonic readout of its orientation code (degree 2 = the structure-tensor order, degree 4 = above it)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "devvit_harmonic.png", dpi=90)
    json.dump(results, open(out / "devvit_harmonic.json", "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
