"""From dev-vit features to separated intersecting populations: the signature deconvolution.

1. The signature dictionary: dev-vit's mean feature of a single population at 12 orientations (reference bundles in the
   x-z plane), window-averaged; its angular order (how much of D(theta) is a second-order code, l <= 2, and how much lies
   above) -- the ceiling of any per-position separation in that space.
2. On the crossings, the window-mean feature at every lattice position written as a non-negative mixture of the 12
   signatures plus a texture signature (NNLS); the coefficients over angle are the orientation distribution of the shape;
   its lobes (nmax 2, >= 30 deg apart, rel 0.3) are the populations. Scored on the lattice as the chain is.

    python scripts/devvit_separation.py [--out DIR]
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
import torch.nn.functional as Fn

from shapetensor.features import make_engine
from shapetensor.filters import gaussian
from shapetensor.phantoms import Grid, build
from shapetensor.validate import ANGLES, angle_deg, cell_truth, tex_for

DEV = "cuda"
STEP, ORIGIN, SIGMA_W = 4, 2, 10.0


def window_mean(F: torch.Tensor) -> torch.Tensor:
    """``F (K, Z, Y, X)`` -> the Gaussian window mean (sigma 10) on the lattice ``(K, z, y, x)``."""
    P = Fn.avg_pool3d(F[None], STEP)[0]
    return gaussian(P, SIGMA_W / STEP, truncate=3.0)


def nnls(A: torch.Tensor, B: torch.Tensor, iters: int = 400) -> torch.Tensor:
    """Non-negative ``X`` with ``A X ≈ B`` (FISTA), ``A (K, n)``, ``B (K, V)`` -> ``(n, V)``."""
    L = float(torch.linalg.matrix_norm(A, 2)) ** 2
    AtA, AtB = A.T @ A, A.T @ B
    x = torch.zeros(A.shape[1], B.shape[1], device=A.device); y = x.clone(); t = 1.0
    for _ in range(iters):
        xn = (y - (AtA @ y - AtB) / L).clamp_min(0)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
    return x


def lobes_1d(dist: torch.Tensor, angles: np.ndarray, nmax: int = 2, rel: float = 0.3, min_sep: float = 30.0):
    """Lobes of distributions over in-plane angles ``dist (n, V)``: local maxima (circular), strongest first, >= min_sep
    apart, each lobe's mass over the angles within 20 deg; directions as (z, y, x) unit vectors."""
    n, V = dist.shape; ang = torch.as_tensor(angles, device=dist.device, dtype=torch.float32)
    prev, nxt = torch.roll(dist, 1, 0), torch.roll(dist, -1, 0)
    ismax = (dist >= prev) & (dist >= nxt) & (dist > 0)
    s = torch.where(ismax, dist, torch.zeros_like(dist))
    dang = ((ang[:, None] - ang[None, :] + 90) % 180 - 90).abs()                                 # circular angular distance (axial)
    dirs = torch.zeros(nmax, 3, V, device=dist.device); mass = torch.zeros(nmax, V, device=dist.device); peak_ang = torch.zeros(nmax, V, device=dist.device)
    for r in range(nmax):
        val, idx = s.max(0); ok = val > 0
        w = dist * (dang[idx] <= 20).T.float() * ok[None]
        mass[r] = w.sum(0)
        th = torch.deg2rad(2 * ang)[:, None]                                                    # axial circular mean of the lobe's angles
        mean = 0.5 * torch.atan2((w * torch.sin(th)).sum(0), (w * torch.cos(th)).sum(0))
        peak_ang[r] = torch.rad2deg(mean) % 180
        dirs[r] = torch.stack([torch.sin(mean), torch.zeros_like(mean), torch.cos(mean)])
        s = torch.where((dang[idx] < min_sep).T & ok[None], torch.zeros_like(s), s)
    order = mass.argsort(0, descending=True); mass = mass.gather(0, order); dirs = dirs.gather(0, order[:, None].expand(-1, 3, -1)); peak_ang = peak_ang.gather(0, order)
    keep = (mass >= rel * mass[:1]) & (mass > 0); keep[0] = mass[0] > 0
    return dirs * keep[:, None], mass * keep, peak_ang


def score(dirs, mass, tr, grid, cs):
    """N and directions against the truth on the lattice, the core only (as ``validate.lattice_scores``)."""
    npop, d1, d2, _ = cell_truth(tr, cs, STEP, ORIGIN, DEV)
    zs, ys, xs = grid.inner
    inner = torch.zeros(cs, dtype=torch.bool, device=DEV); inner[zs.start // STEP:zs.stop // STEP, ys.start // STEP:ys.stop // STEP, xs.start // STEP:xs.stop // STEP] = True
    N = (mass > 0).sum(0); r = {}
    for cat, k in (("none", 0), ("one", 1), ("two", 2)):
        sel = inner & (npop == k)
        if int(sel.sum()) == 0:
            continue
        rr = {f"N={j}": round(float((N[sel] == j).float().mean()), 3) for j in range(3)}
        if k == 1 and int((sel & (N >= 1)).sum()):
            rr["err"] = round(float(angle_deg(dirs[0], d1)[sel & (N >= 1)].median()), 1)
        if k == 2:
            e11, e12, e21, e22 = angle_deg(dirs[0], d1), angle_deg(dirs[0], d2), angle_deg(dirs[1], d1), angle_deg(dirs[1], d2)
            sw = (e12 + e21) < (e11 + e22); ea, eb = torch.where(sw, e12, e11), torch.where(sw, e21, e22)
            s2 = sel & (N >= 2)
            rr["pair"] = round(float(((ea + eb) / 2)[s2].median()), 1) if int(s2.sum()) else None
            rr["both15"] = round(float(((ea <= 15) & (eb <= 15) & (N >= 2))[sel].float().mean()), 3)
        r[cat] = rr
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/devvit_separation")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tex, texture = tex_for("calibrated"); grid = Grid()
    vit = make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32)
    zs, ys, xs = grid.inner; core = (slice(zs.start // STEP, zs.stop // STEP), slice(ys.start // STEP, ys.stop // STEP), slice(xs.start // STEP, xs.stop // STEP))
    # 1. the signatures: dev-vit's window-mean feature of a bundle at each angle, averaged over the core
    sig = []
    for k, ang in enumerate(ANGLES):
        img = build("bundle", 100 + k, grid, tex, texture=texture, device=DEV, angle=float(ang))[0]
        W = window_mean(vit(torch.as_tensor(img, device=DEV).float()))
        sig.append(W[(slice(None),) + core].reshape(W.shape[0], -1).mean(1))
    texi = build("texture", 7, grid, tex, texture=texture, device=DEV)[0]
    Wt = window_mean(vit(torch.as_tensor(texi, device=DEV).float()))
    sig_tex = Wt[(slice(None),) + core].reshape(Wt.shape[0], -1).mean(1)
    D = torch.stack(sig, 1)                                                                     # (K, 12)
    th = torch.deg2rad(torch.as_tensor(ANGLES, device=DEV, dtype=torch.float32))
    Dc = D - D.mean(1, keepdim=True); tot = float((Dc ** 2).sum())
    order = {}
    for lmax in (2, 4, 6):                                                                      # axial harmonics cos/sin of 2, 4, 6 theta
        cols = [torch.ones_like(th)] + [f(m * th) for m in range(2, lmax + 1, 2) for f in (torch.cos, torch.sin)]
        X = torch.stack(cols, 1); coef = torch.linalg.lstsq(X, D.T).solution; resid = D.T - X @ coef
        order[f"l<={lmax}"] = round(1 - float((resid ** 2).sum()) / tot, 4)
    print("signature angular order (variance of D(theta) about its mean explained):", json.dumps(order), flush=True)
    A = torch.cat([D, sig_tex[:, None]], 1); A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-9)   # (K, 13)
    results = {"signature order": order, "cases": {}}
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), facecolor="white")
    for row, (name, kind, kw, g) in enumerate((("crossing 90", "crossing", {"cross_deg": 90.0}, Grid()), ("crossing 60", "crossing", {"cross_deg": 60.0}, Grid()),
                                               ("band crossing 90", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192))))):
        img, tr, meta = build(kind, 1, g, tex=tex, texture=texture, device=DEV, **kw); v = torch.as_tensor(img, device=DEV).float()
        W = window_mean(vit(v)); cs = W.shape[1:]
        B = W.reshape(W.shape[0], -1); B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-9)
        X = nnls(A, B); dist = X[:12]
        dist_s = 0.5 * dist + 0.25 * (torch.roll(dist, 1, 0) + torch.roll(dist, -1, 0))        # one neighbour of circular smoothing (15 deg steps)
        dirs, mass, pk = lobes_1d(dist_s, ANGLES)
        dirs = dirs.reshape(2, 3, *cs); mass = mass.reshape(2, *cs)
        sc = score(dirs, mass, tr, g, cs); results["cases"][name] = sc
        print(f"{name:18s} signature deconvolution: {json.dumps(sc)}", flush=True)
        Z, Y, X_ = v.shape; cz, cy, cx = (Z // 2 - ORIGIN) // STEP, (Y // 2 - ORIGIN) // STEP, (X_ // 2 - ORIGIN) // STEP
        c = cz * cs[1] * cs[2] + cy * cs[2] + cx
        ax = axs[row]
        ax[0].bar(ANGLES, dist[:, c].cpu().numpy(), width=12, color="#2a5db0"); ax[0].set_xlim(-8, 180)
        for ang_ in meta["angles"]:
            ax[0].axvline(ang_ % 180, color="#e34948", ls="--")
        ax[0].set_title(f"{name}: the shape-space distribution at the centre\n(NNLS onto 12 dev-vit signatures; red: truth)", fontsize=10)
        N = (mass > 0).sum(0)[:, cy, :].cpu().numpy()
        ax[1].imshow(N, cmap=matplotlib.colors.ListedColormap(["#222222", "#2a78d6", "#e34948"]), vmin=-0.5, vmax=2.5, interpolation="nearest"); ax[1].set_title("N per cell", fontsize=10)
        ax[2].imshow(pk.reshape(2, *cs)[0][:, cy, :].cpu().numpy(), cmap="hsv", vmin=0, vmax=180); ax[2].set_title("first lobe angle (hsv, 0-180)", fontsize=10)
        for a_ in ax[1:]:
            a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"dev-vit signature deconvolution; signature angular order {json.dumps(order)}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "devvit_signature_deconvolution.png", dpi=90)
    json.dump(results, open(out / "devvit_separation.json", "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
