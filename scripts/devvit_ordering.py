"""Deconvolve-then-separate against separate-then-deconvolve, from dev-vit, on the cut-population case.

Variants (the readout is the degree-4 harmonic readout of ``devvit_harmonic.py``):
  R        readout -> lobes (no smoothing);
  R+M      readout -> carry-forward of f(., theta) along theta for every theta -> lobes   (separate, then deconvolve);
  S+R      the window-mean feature W smoothed along its dominant axis (the degree-2 readout's) -> readout -> lobes
           (deconvolve, then separate);
  T+R      sliceTCA-smoothed features -> readout -> lobes (the TCA form of deconvolve-then-separate).
Population images: the layer f(x, theta within 20 deg of each truth direction) on the mid-y slice; N per cell; the
scores in the overlap; the T-junction's false extension as the control.

    python scripts/devvit_ordering.py [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.completion import _sampler
from shapetensor.features import make_engine
from shapetensor.phantoms import Grid, build, direction
from shapetensor.populations import momentum
from shapetensor.validate import ANGLES, lattice_scores, tex_for

spec = importlib.util.spec_from_file_location("DS", str(Path(__file__).with_name("devvit_separation.py")))
DS = importlib.util.module_from_spec(spec); spec.loader.exec_module(DS)
DEV, STEP, ORIGIN = DS.DEV, DS.STEP, DS.ORIGIN
THETA = np.arange(0, 180, 2.0)


def harmonics(theta_deg, lmax):
    th = torch.deg2rad(torch.as_tensor(np.asarray(theta_deg, np.float32), device=DEV))
    return torch.stack([torch.ones_like(th)] + [f(m * th) for m in range(2, lmax + 1, 2) for f in (torch.cos, torch.sin)], 1)


def fit_readouts(eng, grid, tex, texture, ridge=1e-2):
    zs, ys, xs = grid.inner; core = (slice(None), slice(zs.start // STEP, zs.stop // STEP), slice(ys.start // STEP, ys.stop // STEP), slice(xs.start // STEP, xs.stop // STEP))
    Xs, ths, tx = [], [], []
    for k, ang in enumerate(ANGLES):
        img = build("bundle", 100 + k, grid, tex, texture=texture, device=DEV, angle=float(ang))[0]
        W = DS.window_mean(eng(torch.as_tensor(img, device=DEV).float()).float())[core].reshape(32, -1).T
        Xs.append(W); ths.append(torch.full((W.shape[0],), float(ang), device=DEV)); tx.append(torch.zeros(W.shape[0], dtype=torch.bool, device=DEV))
    img = build("texture", 7, grid, tex, texture=texture, device=DEV)[0]
    W = DS.window_mean(eng(torch.as_tensor(img, device=DEV).float()).float())[core].reshape(32, -1).T
    Xs.append(W); ths.append(torch.zeros(W.shape[0], device=DEV)); tx.append(torch.ones(W.shape[0], dtype=torch.bool, device=DEV))
    X = torch.cat(Xs); th = torch.cat(ths); tx = torch.cat(tx)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6); Xn1 = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, device=DEV)], 1)
    M = {}
    for lmax in (2, 4):
        H = harmonics(th.cpu().numpy(), lmax); H[tx] = 0
        M[lmax] = torch.linalg.solve(Xn1.T @ Xn1 + ridge * X.shape[0] * torch.eye(Xn1.shape[1], device=DEV), Xn1.T @ H)
    return SimpleNamespace(mu=mu, sd=sd, M=M)


def readout(R, W, lmax=4):
    """``W (32, cz, cy, cx)`` -> the orientation function ``f (n_theta, cz, cy, cx)`` (non-negative)."""
    cs = W.shape[1:]
    Wn = torch.cat([((W.reshape(32, -1).T - R.mu) / R.sd), torch.ones(W[0].numel(), 1, device=DEV)], 1)
    return (harmonics(THETA, lmax) @ (Wn @ R.M[lmax]).T).clamp_min(0).reshape(len(THETA), *cs)


def dominant_axis(R, W):
    """The degree-2 readout's single axis per cell, as a (3, cz, cy, cx) field (z, y, x)."""
    f2 = readout(R, W, lmax=2); th = THETA[f2.reshape(len(THETA), -1).argmax(0).cpu().numpy()]
    th = torch.deg2rad(torch.as_tensor(th, device=DEV, dtype=torch.float32)).reshape(W.shape[1:])
    return torch.stack([torch.sin(th), torch.zeros_like(th), torch.cos(th)])


def smooth_along(W, d, sigma=3.0, reach=6):
    """``W (K, cells)`` smoothed along the varying direction field ``d (3, cells)`` (a Gaussian over ``t`` in cells)."""
    sample = _sampler(tuple(W.shape[1:]), DEV)
    acc = torch.zeros_like(W); norm = torch.zeros(W.shape[1:], device=DEV)
    ones = torch.ones((1,) + tuple(W.shape[1:]), device=DEV)
    for t in range(-reach, reach + 1):
        w = math.exp(-0.5 * (t / sigma) ** 2)
        acc += w * (W if t == 0 else sample(W, d, float(t))); norm += w * (ones[0] if t == 0 else sample(ones, d, float(t))[0])
    return acc / norm.clamp_min(1e-6)


def lobes_to_pop(f, cs):
    dirs, mass, _ = DS.lobes_1d(f.reshape(len(THETA), -1), THETA)
    dirs = dirs.reshape(2, 3, *cs); mass = mass.reshape(2, *cs)
    return {"N": (mass > 0).sum(0), "mass": mass, "dirs_cells": dirs}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/devvit_ordering")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tex, texture = tex_for("calibrated"); grid = Grid()
    vit = make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32)
    tca = make_engine("devvit_slicetca", device=DEV, l2=False, out_dtype=torch.float32)
    R_raw = fit_readouts(vit, grid, tex, texture); R_tca = fit_readouts(tca, grid, tex, texture)
    model = SimpleNamespace(step=STEP, origin=ORIGIN)
    grid_theta = torch.as_tensor(np.stack([np.sin(np.radians(THETA)), np.zeros_like(THETA), np.cos(np.radians(THETA))], 1), dtype=torch.float32, device=DEV)
    cases = (("band crossing 90", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192))), ("crossing 60", "crossing", {"cross_deg": 60.0}, Grid()),
             ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}, Grid(core=(192, 32, 192))))
    results = {}
    fig, axs = plt.subplots(len(cases) * 4, 3, figsize=(12, 3.3 * len(cases) * 4), facecolor="white")
    rr = 0
    for name, kind, kw, g in cases:
        img, tr, meta = build(kind, 1, g, tex=tex, texture=texture, device=DEV, **kw); v = torch.as_tensor(img, device=DEV).float()
        W = DS.window_mean(vit(v).float()); cs = W.shape[1:]; cy = cs[1] // 2
        f = readout(R_raw, W)
        variants = {"R: readout -> lobes": f,
                    "R+M: readout -> carry-forward along each theta -> lobes": momentum(f, grid_theta, sigma_along=3.0, reach=6, angular_deg=0.0),
                    "S+R: W smoothed along its dominant axis -> readout -> lobes": readout(R_raw, smooth_along(W, dominant_axis(R_raw, W))),
                    "T+R: sliceTCA features -> readout -> lobes": readout(R_tca, DS.window_mean(tca(v).float()))}
        d1, d2 = direction(meta["angles"][0]), direction(meta["angles"][1])
        sel = [torch.as_tensor(np.abs(grid_theta.cpu().numpy() @ d) >= math.cos(math.radians(20)), device=DEV) for d in (d1, d2)]
        for vname, fv in variants.items():
            pop = lobes_to_pop(fv, cs); sc = lattice_scores(pop, tr, g, meta, model)
            results.setdefault(name, {})[vname] = sc; print(f"{name:18s} {vname:62s} {json.dumps(sc)}", flush=True)
            ax = axs[rr]; rr += 1
            for j, (dsel, lab) in enumerate(zip(sel, ("population 1 (band 1)", "population 2 (the crossed band)"))):
                layer = fv[dsel].sum(0)[:, cy, :].cpu().numpy()
                ax[j].imshow(layer, cmap="magma", vmin=0, vmax=max(float(np.percentile(fv.sum(0)[:, cy, :].cpu().numpy(), 99)), 1e-6)); ax[j].set_title(f"{name} | {vname.split(':')[0]}: layer of {lab}", fontsize=9)
            ax[2].imshow(pop["N"][:, cy, :].cpu().numpy(), cmap=matplotlib.colors.ListedColormap(["#222222", "#2a78d6", "#e34948"]), vmin=-0.5, vmax=2.5, interpolation="nearest"); ax[2].set_title("N per cell", fontsize=9)
            for a_ in ax:
                a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("dev-vit: the population layers under each ordering of separation and deconvolution (carry-forward / smoothing)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out / "devvit_ordering.png", dpi=80)
    json.dump(results, open(out / "devvit_ordering.json", "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
