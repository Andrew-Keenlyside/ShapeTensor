"""A per-voxel orientation readout of the shape space, self-calibrated — the division of the shape space by
orientation (stage 2 of the pipeline), without labels.

The orientation-selective shape space (encoder block 2) encodes a voxel's fibre orientation linearly (results log
§18). The map from features to orientation is fitted on the data itself: the teacher is the excised structure
tensor's direction at its *confident, single-population* sites (high orientation gap), which needs no labels; the
target is the axial outer product d dᵀ (six numbers, sign-free); the model is a ridge regression from the locally
averaged, standardised features. Applied per voxel everywhere — including where the window tensor is unreliable —
it gives each voxel an orientation and a confidence (the anisotropy of the predicted tensor). Per window, the
voxels' orientations are a distribution on the sphere whose *modes* are the populations: a kernel density on a
hemisphere grid, peaks above a null-calibrated relative height, memberships by the nearest mode (stage 2), and a
membership-weighted tensor per mode (stage 3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .common import eigh


def outer6(d: torch.Tensor) -> torch.Tensor:
    """``d (..., 3)`` unit -> the six unique entries of d dᵀ: zz, yy, xx, zy, zx, yx."""
    z, y, x = d[..., 0], d[..., 1], d[..., 2]
    return torch.stack([z * z, y * y, x * x, z * y, z * x, y * x], -1)


def tensor_from6(t6: torch.Tensor) -> torch.Tensor:
    zz, yy, xx, zy, zx, yx = [t6[..., k] for k in range(6)]
    return torch.stack([torch.stack([zz, zy, zx], -1), torch.stack([zy, yy, yx], -1), torch.stack([zx, yx, xx], -1)], -2)


class OrientationReadout:
    """Ridge regression from shape features to the orientation tensor, fitted on confident single-population voxels."""

    def __init__(self, smooth: int = 3, ridge: float = 1.0):
        self.smooth, self.ridge = int(smooth), float(ridge)
        self.mean = self.std = self.W = None

    def _prep(self, F: torch.Tensor) -> torch.Tensor:
        f = F.float()
        if self.smooth > 1:
            f = Fn.avg_pool3d(f[None], self.smooth, stride=1, padding=self.smooth // 2)[0]
        return f

    def fit(self, blocks, n_sample: int = 200_000, seed: int = 0):
        """``blocks``: iterable of ``(F (C, Z, Y, X), teacher (3, Z, Y, X), weight (Z, Y, X))``; voxels are drawn
        with probability ∝ weight (zero excludes). Returns self."""
        g = torch.Generator().manual_seed(seed)
        Xs, Ys = [], []
        per = max(1, n_sample // max(1, len(blocks)))
        for F, d, w in blocks:
            f = self._prep(F)
            wv = w.reshape(-1).float().cpu()
            idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T)
            Ys.append(outer6(d.reshape(3, -1)[:, idx].T))
        X = torch.cat(Xs); Y = torch.cat(Ys)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = torch.cat([(X - self.mean) / self.std, torch.ones(len(X), 1, device=X.device)], 1)
        A = Xn.T @ Xn + self.ridge * torch.eye(Xn.shape[1], device=X.device)
        self.W = torch.linalg.solve(A, Xn.T @ Y)                                          # (C + 1, 6)
        pred = Xn @ self.W
        self.fit_r2 = float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        return self

    def predict(self, F: torch.Tensor):
        """Per-voxel orientation ``(3, Z, Y, X)`` (unit, axial) and confidence ``(Z, Y, X)`` in [0, 1]."""
        f = self._prep(F)
        C, Z, Y, X = f.shape
        Xn = ((f.reshape(C, -1).T - self.mean) / self.std)
        t6 = Xn @ self.W[:-1] + self.W[-1]
        T = tensor_from6(t6)
        lam, V = eigh(T)                                                                  # ascending
        d = V[..., :, 2]                                                                  # principal eigenvector
        conf = ((lam[:, 2] - lam[:, 1]) / lam[:, 2].abs().clamp_min(1e-6)).clamp(0, 1)
        return d.T.reshape(3, Z, Y, X), conf.reshape(Z, Y, X)


def hemisphere(n: int = 1500, device="cpu") -> torch.Tensor:
    """Roughly uniform unit vectors on a hemisphere (z >= 0), a Fibonacci set, (n, 3) in (z, y, x)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = i / n                                                                             # z in (0, 1]
    phi = i * math.pi * (3 - math.sqrt(5))
    r = torch.sqrt(1 - z * z)
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)


def window_modes(o: torch.Tensor, w: torch.Tensor, kappa_deg: float = 15.0, rel: float = 0.5, min_sep_deg: float = 25.0,
                 nmax: int = 3, grid: torch.Tensor | None = None):
    """Modes of axial orientations ``o (n, 3)`` with weights ``w (n,)``: a kernel density on the hemisphere grid
    (von Mises–Fisher-like, width ``kappa_deg``), peaks picked greedily above ``rel`` x the top peak and
    ``min_sep_deg`` apart. Returns modes ``(nmax, 3)`` (zero rows where absent), their relative heights and N."""
    u = hemisphere(1500, o.device) if grid is None else grid
    k = 1.0 / (2 * math.sin(math.radians(kappa_deg)) ** 2)
    c2 = (o @ u.T) ** 2                                                                   # (n, m)
    dens = (w[:, None] * torch.exp(k * (c2 - 1))).sum(0)                                 # (m,)
    modes = torch.zeros(nmax, 3, device=o.device); heights = torch.zeros(nmax, device=o.device)
    top = float(dens.max()); taken = torch.zeros(len(u), dtype=torch.bool, device=o.device)
    cos_sep = math.cos(math.radians(min_sep_deg)); n = 0
    for r in range(nmax):
        d = dens.masked_fill(taken, -1.0)
        j = int(d.argmax())
        if d[j] < rel * top or d[j] <= 0:
            break
        modes[r] = u[j]; heights[r] = d[j] / top; n += 1
        taken |= (u @ u[j]).abs() >= cos_sep
    return modes, heights, n


def assign(o: torch.Tensor, modes: torch.Tensor, n: int, soft_deg: float = 20.0) -> torch.Tensor:
    """Memberships ``(n_vox, nmax)`` of orientations ``o`` to the modes (soft, by angular closeness)."""
    m = torch.zeros(o.shape[0], modes.shape[0], device=o.device)
    if n == 0:
        return m
    k = 1.0 / (2 * math.sin(math.radians(soft_deg)) ** 2)
    c2 = (o @ modes[:n].T) ** 2
    m[:, :n] = torch.exp(k * (c2 - 1))
    m[:, :n] = m[:, :n] / m[:, :n].sum(1, keepdim=True).clamp_min(1e-12)
    return m
