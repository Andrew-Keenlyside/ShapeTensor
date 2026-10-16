"""Separating fibre populations in the shape space, label-free, per window (stage 2 of the pipeline).

A population's signature in an orientation-selective shape space is the pattern of channel energies its voxels
excite. Per site, the window's voxels give a cloud of channel-energy vectors; the principal directions of that
cloud (the eigenvectors of the window's energy covariance) are the signatures present, and a voxel's membership
of population *r* is its (non-negative) projection on signature *r*. One tensor per population is then the
membership-weighted structure tensor of the image gradients (stage 3). N is the number of signatures whose
eigenvalue exceeds a null threshold calibrated on single-population windows. No clustering by k-means, no
labels: the division of the shape space is its own principal decomposition.
"""
from __future__ import annotations

import torch

from .common import eigh


def window_indices(center, half: int, shape):
    z, y, x = center
    zs = torch.arange(max(0, z - half), min(shape[0], z + half + 1), device=center.device)
    ys = torch.arange(max(0, y - half), min(shape[1], y + half + 1), device=center.device)
    xs = torch.arange(max(0, x - half), min(shape[2], x + half + 1), device=center.device)
    return zs, ys, xs


def separate_site(F: torch.Tensor, G: torch.Tensor, center, rho: float, truncate: float = 3.0, nmax: int = 3,
                  eps: float = 1e-6) -> dict:
    """One site. ``F (C, Z, Y, X)`` features, ``G (3, Z, Y, X)`` image gradients (z, y, x), ``center`` (z, y, x)
    voxel index. Returns the signature eigenvalues (descending), the signatures ``U (C, nmax)``, the per-population
    directions ``dirs (nmax, 3)`` from the membership-weighted tensors, and the memberships' total weight."""
    half = int(truncate * rho + 0.5)
    zs, ys, xs = window_indices(center, half, F.shape[1:])
    f = F[:, zs][:, :, ys][:, :, :, xs].float()                                 # (C, nz, ny, nx)
    g = G[:, zs][:, :, ys][:, :, :, xs].float()
    dz, dy, dx = zs - center[0], ys - center[1], xs - center[2]
    w = torch.exp(-0.5 * (dz[:, None, None] ** 2 + dy[None, :, None] ** 2 + dx[None, None, :] ** 2).float() / rho ** 2)
    e = f.reshape(f.shape[0], -1) ** 2                                             # channel energies per voxel (C, n)
    e = e / (e.sum(0, keepdim=True) + eps)                                         # normalised: the pattern, not the amplitude
    wv = w.reshape(-1)
    mean = (e * wv).sum(1, keepdim=True) / wv.sum()
    ec = (e - mean) * wv.sqrt()
    cov = ec @ ec.T / wv.sum()                                                     # (C, C) weighted covariance
    lam, U = eigh(cov[None]); lam, U = lam[0].flip(0), U[0].flip(1)               # descending
    sig = U[:, :nmax]                                                              # (C, nmax) signatures
    m = ((e - mean).T @ sig).clamp_min(0)                                          # (n, nmax) memberships
    gv = g.reshape(3, -1)
    dirs = torch.zeros(nmax, 3, device=F.device); weight = torch.zeros(nmax, device=F.device)
    for r in range(nmax):
        ww = wv * m[:, r]
        S = (gv * ww) @ gv.T                                                       # membership-weighted structure tensor
        wsum = ww.sum()
        if wsum > 0:
            _, V = eigh(S[None] / wsum); dirs[r] = V[0][:, 0]; weight[r] = wsum
    return {"lam": lam[:nmax], "signatures": sig, "dirs": dirs, "weight": weight}


def separate_sites(F: torch.Tensor, G: torch.Tensor, centers: torch.Tensor, rho: float, truncate: float = 3.0, nmax: int = 3) -> dict:
    """``separate_site`` over ``centers (N, 3)``; stacked outputs."""
    out = {"lam": [], "dirs": [], "weight": []}
    for c in centers:
        r = separate_site(F, G, c, rho, truncate, nmax)
        for k in out:
            out[k].append(r[k])
    return {k: torch.stack(v) for k, v in out.items()}


def count_from_eigenvalues(lam: torch.Tensor, ratio_thr: float, nmax: int = 3) -> torch.Tensor:
    """N per site: 1 + the number of further signatures whose eigenvalue exceeds ``ratio_thr`` x the first."""
    r = lam[:, 1:nmax] / lam[:, :1].clamp_min(1e-12)
    return 1 + (r > ratio_thr).sum(1)
