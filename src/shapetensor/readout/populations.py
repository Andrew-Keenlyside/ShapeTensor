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
    proj = (e - mean).T @ sig                                                      # (n, nmax) signed projections
    gv = g.reshape(3, -1)
    dirs = torch.zeros(nmax, 3, device=F.device); weight = torch.zeros(nmax, device=F.device); gap = torch.zeros(nmax, device=F.device)
    for r in range(nmax):
        # an eigenvector's sign is arbitrary: each side of the signature is a candidate population; keep the side whose
        # voxels form the more coherent fibre (the larger orientation gap of their weighted tensor)
        best = None
        for sgn in (1.0, -1.0):
            ww = wv * (sgn * proj[:, r]).clamp_min(0)
            wsum = ww.sum()
            if wsum <= 0:
                continue
            S = (gv * ww) @ gv.T / wsum                                                # membership-weighted structure tensor
            lam_s, V = eigh(S[None]); lam_s, V = lam_s[0], V[0]                        # ascending
            gp = ((lam_s[1] - lam_s[0]) / lam_s[1].clamp_min(1e-12)).clamp(0, 1)      # the gap of the two smallest: how well the null direction is set
            if best is None or gp > best[0]:
                best = (gp, V[:, 0], wsum)
        if best is not None:
            gap[r], dirs[r], weight[r] = best
    return {"lam": lam[:nmax], "signatures": sig, "dirs": dirs, "weight": weight, "gap": gap}


def separate_sites(F: torch.Tensor, G: torch.Tensor, centers: torch.Tensor, rho: float, truncate: float = 3.0, nmax: int = 3) -> dict:
    """``separate_site`` over ``centers (N, 3)``; stacked outputs."""
    out = {"lam": [], "dirs": [], "weight": [], "gap": []}
    for c in centers:
        r = separate_site(F, G, c, rho, truncate, nmax)
        for k in out:
            out[k].append(r[k])
    return {k: torch.stack(v) for k, v in out.items()}


def count_from_eigenvalues(lam: torch.Tensor, ratio_thr: float, nmax: int = 3) -> torch.Tensor:
    """N per site: 1 + the number of further signatures whose eigenvalue exceeds ``ratio_thr`` x the first."""
    r = lam[:, 1:nmax] / lam[:, :1].clamp_min(1e-12)
    return 1 + (r > ratio_thr).sum(1)
