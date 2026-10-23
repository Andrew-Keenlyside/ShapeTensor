"""Contrast routes: augmentation applied to a finished image every time it is used, so one geometry (one target) is
seen through many contrasts -- the network cannot key on a modality's or a physics' intensity signature.

Applied at batch time to buffered training images (``train.stack``), each use a fresh draw. Every operation keeps the
geometry, so the target (from the geometry) is unchanged:

* **transfer functions** -- a random piecewise-linear curve (2-8 knots at random positions), monotone (either
  direction) or not (folds: intensity order destroyed), guarded against collapsing the range; a **spatially varying**
  one (two curves blended by a smooth random field: contrast that changes across the block); gamma;
* **filters** -- unsharp masking (negative: blur), a Laplacian-of-Gaussian term (phase-contrast-like edges of either
  sign), a **relief** term (a directional derivative along a random 3-D direction: DIC-like shadow casting);
* **fields** -- a smooth multiplicative gain and additive offset;
* **noise** -- Gaussian at a random level, after the curve (so its spectrum and level vary independently of it);
* **missing data** -- **missing lines** (scan-line or detector dropouts: thin lines through the block along one axis,
  all or part of it, filled with a constant, the local mean or the neighbouring line) and **small missing blocks**
  (cuboids of 2-12 voxels: bubbles, missing tiles, dropouts; filled with a constant, noise or a blur). Small gaps
  keep their target: the populations continue through them and the network learns to carry them across from context.

``strength`` (0-1) scales every probability and magnitude (a curriculum: tied to the difficulty level). Scalars come
from a numpy generator (no device synchronisation); fields are drawn on the device.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from ..filters import gaussian

P = {"lut": 0.7, "nonmono": 0.5, "spatial": 0.25, "gamma": 0.3, "unsharp": 0.3, "log": 0.15, "relief": 0.15, "field": 0.4,
     "noise": 0.4, "lines": 0.25, "blocks": 0.25}


def random_lut(rng: np.random.Generator, monotone: bool, min_tv: float = 0.6):
    """Knots ``(xk, yk)`` of a random curve on [0, 1]: 2-8 inner knots; monotone (rising or falling) or not, its total
    variation at least ``min_tv`` (a curve that flattens the range would erase the structure)."""
    for _ in range(50):
        k = int(rng.integers(2, 9))
        xk = np.concatenate([[0.0], np.sort(rng.uniform(0, 1, k)), [1.0]])
        yk = rng.uniform(0, 1, k + 2)
        if monotone:
            yk = np.sort(yk) if rng.random() < 0.5 else np.sort(yk)[::-1]
            yk = (yk - yk.min()) / max(yk.max() - yk.min(), 1e-6)                                 # full range
        if np.abs(np.diff(yk)).sum() >= min_tv:
            return xk, yk
    return np.array([0.0, 1.0]), np.array([0.0, 1.0])


def apply_lut(x: torch.Tensor, xk, yk) -> torch.Tensor:
    xk_t = torch.tensor(xk, dtype=x.dtype, device=x.device); yk_t = torch.tensor(np.ascontiguousarray(yk), dtype=x.dtype, device=x.device)
    i = torch.searchsorted(xk_t, x.clamp(0, 1).contiguous(), right=True).clamp(1, xk_t.numel() - 1)
    x0, x1, y0, y1 = xk_t[i - 1], xk_t[i], yk_t[i - 1], yk_t[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0).clamp_min(1e-6)


def smooth_field(shape, sigma: float, device, g: torch.Generator | None = None) -> torch.Tensor:
    """A smooth random field of unit spread (drawn coarse, upsampled)."""
    f = max(1, int(sigma // 2))
    small = torch.randn([max(2, s // f) for s in shape], device=device, generator=g)
    small = gaussian(small, max(sigma / f, 0.5))
    small = (small - small.mean()) / small.std().clamp_min(1e-6)
    return Fn.interpolate(small[None, None], size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]


def missing_lines(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Thin lines (1-2 voxels) along a random axis, through the whole block or part of it, lost."""
    Z, Y, X = x.shape; ax = int(rng.integers(0, 3)); n = int(rng.integers(1, 13))
    others = [a for a in range(3) if a != ax]
    mode = ["zero", "one", "mean", "neighbour"][int(rng.integers(0, 4))]
    x = x.clone(); sh = x.shape
    for _ in range(n):
        p = [int(rng.integers(0, sh[a])) for a in others]; w = int(rng.integers(1, 3))
        lo = int(rng.integers(0, sh[ax] // 2)) if rng.random() < 0.5 else 0
        hi = int(rng.integers(lo + sh[ax] // 4, sh[ax] + 1)) if lo else sh[ax]
        idx = [slice(None)] * 3
        idx[ax] = slice(lo, hi); idx[others[0]] = slice(p[0], p[0] + w); idx[others[1]] = slice(p[1], p[1] + w)
        idx = tuple(idx)
        if mode == "zero":
            x[idx] = 0.0
        elif mode == "one":
            x[idx] = 1.0
        elif mode == "mean":
            x[idx] = x.mean()
        else:                                                                                      # the neighbouring line repeated
            nb = list(idx); o = others[0]; s0 = max(0, p[0] - w); nb[o] = slice(s0, s0 + w)
            src = x[tuple(nb)]
            if src.shape == x[idx].shape:
                x[idx] = src
    rec["missing_lines"] = (ax, n, mode)
    return x


def missing_blocks(x: torch.Tensor, rng: np.random.Generator, rec: dict, max_side: int = 12) -> torch.Tensor:
    """Small cuboids (2-``max_side`` voxels a side) lost: filled with a constant, noise or a blur of their content."""
    n = int(rng.integers(1, 9)); x = x.clone(); sh = x.shape
    mode = ["zero", "one", "mean", "noise", "blur"][int(rng.integers(0, 5))]
    for _ in range(n):
        s = [int(rng.integers(2, max_side + 1)) for _ in range(3)]
        o = [int(rng.integers(0, max(1, sh[a] - s[a]))) for a in range(3)]
        idx = tuple(slice(o[a], o[a] + s[a]) for a in range(3))
        if mode == "zero":
            x[idx] = 0.0
        elif mode == "one":
            x[idx] = 1.0
        elif mode == "mean":
            x[idx] = x.mean()
        elif mode == "noise":
            x[idx] = torch.rand(x[idx].shape, device=x.device)
        else:
            x[idx] = x[idx].mean()
    rec["missing_blocks"] = (n, mode)
    return x


def contrast_route(x: torch.Tensor, rng: np.random.Generator, strength: float = 1.0, g: torch.Generator | None = None,
                   rec: dict | None = None) -> torch.Tensor:
    """One random contrast route for an image ``x (Z, Y, X)`` in 0-1 (see the module docstring); returns 0-1."""
    rec = {} if rec is None else rec
    s = float(min(max(strength, 0.0), 1.0))
    on = lambda k: rng.random() < P[k] * s
    y = x.float()
    if on("lut"):                                                                                  # a transfer function, possibly varying in space
        mono = not (rng.random() < P["nonmono"])
        L1 = random_lut(rng, mono)
        if on("spatial"):
            L2 = random_lut(rng, mono or rng.random() < 0.5)
            w = torch.sigmoid(2.0 * smooth_field(y.shape, float(rng.uniform(10, 40)), y.device, g))
            y = w * apply_lut(y, *L1) + (1 - w) * apply_lut(y, *L2); rec["lut"] = ("spatial", mono)
        else:
            y = apply_lut(y, *L1); rec["lut"] = ("global", mono)
    if on("gamma"):
        gm = math.exp(rng.uniform(-0.7, 0.7)); y = y.clamp(0, 1) ** gm; rec["gamma"] = round(gm, 2)
    if on("unsharp"):                                                                              # sharpening (a > 0) or blur (a < 0)
        a = float(rng.uniform(-0.7, 2.5)); sg = float(math.exp(rng.uniform(math.log(0.7), math.log(3.0))))
        y = y + a * (y - gaussian(y, sg)); rec["unsharp"] = (round(a, 2), round(sg, 2))
    if on("log"):                                                                                  # edges of either sign
        sg = float(math.exp(rng.uniform(math.log(0.7), math.log(2.0)))); b = float(rng.uniform(-3, 3)) * sg * sg
        sm = gaussian(y, sg); lap = sum(torch.gradient(torch.gradient(sm, dim=a_)[0], dim=a_)[0] for a_ in range(3))
        y = y - b * lap; rec["log"] = round(b, 2)
    if on("relief"):                                                                               # a directional derivative (DIC-like)
        d = rng.normal(size=3); d /= np.linalg.norm(d); sg = float(rng.uniform(0.7, 2.0)); c = float(rng.uniform(-3, 3)) * sg
        sm = gaussian(y, sg); gr = torch.gradient(sm)
        y = y + c * sum(float(d[a_]) * gr[a_] for a_ in range(3)); rec["relief"] = round(c, 2)
    if on("field"):
        y = y * (1 + float(rng.uniform(0.05, 0.4)) * smooth_field(y.shape, float(rng.uniform(15, 60)), y.device, g)) \
            + float(rng.uniform(0, 0.15)) * smooth_field(y.shape, float(rng.uniform(15, 60)), y.device, g); rec["field"] = True
    lo, hi = torch.quantile(y.flatten()[:: max(1, y.numel() // 50000)], torch.tensor([0.002, 0.998], device=y.device))
    y = ((y - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)                                          # back to the full range
    if on("noise"):
        sd = float(math.exp(rng.uniform(math.log(0.004), math.log(0.06))))
        y = (y + sd * torch.randn(y.shape, device=y.device, generator=g)).clamp(0, 1); rec["noise"] = round(sd, 3)
    if on("lines"):
        y = missing_lines(y, rng, rec)
    if on("blocks"):
        y = missing_blocks(y, rng, rec)
    return y
