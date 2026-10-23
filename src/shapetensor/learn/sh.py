"""The populations' orientation density in real even spherical harmonics: a permutation-free, additive output.

Per cell the density ``f(u) = sum_k m_k w_k(u)`` -- each population's angular density ``w_k`` from its tensor (a line a
Watson lobe, a sheet a girdle, a blob flat; :func:`~shapetensor.learn.targets.render_odf`) weighted by its mass -- is
expanded in real even spherical harmonics up to ``lmax``. It needs no matching of populations to slots (it is a sum),
it resolves crossings from ``l = 4`` (the mean tensor, ``l <= 2``, cannot), the isotropic remainder adds to ``l = 0``
only, and it is additive in space: a Gaussian window at any scale applied to the coefficients is the expansion of
the window applied to the populations -- exact coarsening, and the natural object for any description scale.
"""
from __future__ import annotations

import math

import torch

from .targets import render_odf


def n_coef(lmax: int) -> int:
    return sum(2 * l + 1 for l in range(0, lmax + 1, 2))


def bands(lmax: int) -> list[slice]:
    """The coefficient slice of each even order ``l = 0, 2, ..., lmax``."""
    out, i = [], 0
    for l in range(0, lmax + 1, 2):
        out.append(slice(i, i + 2 * l + 1)); i += 2 * l + 1
    return out


def real_sh(dirs: torch.Tensor, lmax: int) -> torch.Tensor:
    """Real even spherical harmonics at unit ``dirs (N, 3)`` in (z, y, x): ``(N, n_coef)``, orthonormal on the sphere.
    ``m < 0: sqrt2 (-1)^m Im Y_l^|m|``, ``m = 0: Y_l^0``, ``m > 0: sqrt2 (-1)^m Re Y_l^m``; polar axis z, azimuth from x
    towards y; fully normalised Legendre recurrence."""
    d = dirs.float()
    cz = d[:, 0].clamp(-1, 1); sz = (1 - cz * cz).clamp_min(0).sqrt(); phi = torch.atan2(d[:, 1], d[:, 2]); N = d.shape[0]
    P = [[None] * (lmax + 1) for _ in range(lmax + 1)]
    P[0][0] = torch.full((N,), 1.0 / math.sqrt(4 * math.pi), device=d.device)
    for m in range(1, lmax + 1):
        P[m][m] = -math.sqrt((2 * m + 1) / (2 * m)) * sz * P[m - 1][m - 1]
    for m in range(0, lmax):
        P[m + 1][m] = math.sqrt(2 * m + 3) * cz * P[m][m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m)); b = a * math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[l][m] = a * cz * P[l - 1][m] - b * P[l - 2][m]
    cols, s2 = [], math.sqrt(2.0)
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            cols.append(P[l][0] if m == 0 else s2 * (-1) ** m * P[l][m] * torch.cos(m * phi) if m > 0 else s2 * (-1) ** m * P[l][-m] * torch.sin(-m * phi))
    return torch.stack(cols, 1)


def fibonacci_sphere(n: int, device="cpu") -> torch.Tensor:
    """``n`` near-uniform unit vectors on the whole sphere, ``(n, 3)`` in (z, y, x)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n; r = torch.sqrt(1 - z * z); phi = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)


class SHBasis:
    """A sphere grid, its basis and the least-squares projection; cached per (lmax, n, device)."""
    _cache: dict = {}

    def __new__(cls, lmax: int, n: int = 600, device="cpu"):
        key = (lmax, n, str(device))
        if key not in cls._cache:
            o = super().__new__(cls)
            o.lmax, o.grid = lmax, fibonacci_sphere(n, device)
            o.Y = real_sh(o.grid, lmax)                                                               # (n, c)
            o.proj = torch.linalg.pinv(o.Y.double()).float()                                          # (c, n)
            cls._cache[key] = o
        return cls._cache[key]


def density_sh(slots: torch.Tensor, mass: torch.Tensor, lmax: int, kappa: float = 20.0, n: int = 600) -> torch.Tensor:
    """The populations' density in SH: slots ``(B, K, 6, c...)``, masses ``(B, K, c...)`` -> ``(B, n_coef, c...)``."""
    b = SHBasis(lmax, n, slots.device)
    f = torch.stack([render_odf(slots[i].float(), mass[i].float(), b.grid, kappa) for i in range(slots.shape[0])])   # (B, n, c...)
    return torch.einsum("cn,bn...->bc...", b.proj, f)


def evaluate_sh(coef: torch.Tensor, lmax: int, dirs: torch.Tensor, n: int = 600) -> torch.Tensor:
    """The density at ``dirs (m, 3)`` from ``coef (c, ...)``: ``(m, ...)``."""
    Y = real_sh(dirs.to(coef.device), lmax)
    return torch.einsum("mc,c...->m...", Y, coef.float())


def peaks(coef: torch.Tensor, lmax: int, nmax: int = 3, rel: float = 0.5, min_sep_deg: float = 25.0, n: int = 400, chunk: int = 8192):
    """Local maxima of the density ``coef (c, N)`` (cells flattened) on a hemisphere grid of about ``n`` points:
    ``dirs (nmax, 3, N)`` and ``vals (nmax, N)`` (zero where absent). A peak is kept if it exceeds every grid point
    within ``min_sep_deg`` (axially) and is at least ``rel`` of the cell's highest value."""
    g = fibonacci_sphere(2 * n, coef.device); g = g[g[:, 0] >= 0]
    cos_sep = math.cos(math.radians(min_sep_deg))
    C = (g @ g.T).abs(); C.fill_diagonal_(-1.0)
    k = int((C >= cos_sep).sum(1).max())
    nbr = C.topk(k, dim=1).indices; ok = C.gather(1, nbr) >= cos_sep                              # (n, k) padded neighbour lists
    dirs_out, vals_out = [], []
    for s0 in range(0, coef.shape[1], chunk):
        f = evaluate_sh(coef[:, s0:s0 + chunk], lmax, g)                                              # (n, M)
        fn = torch.where(ok[:, :, None], f[nbr], torch.full_like(f[nbr], -1e30)).amax(1)              # (n, M)
        keep = (f > fn) & (f > 0) & (f >= rel * f.amax(0, keepdim=True))
        v, idx = torch.where(keep, f, torch.zeros_like(f)).topk(nmax, dim=0)                          # (nmax, M)
        dirs_out.append(g[idx].movedim(-1, 1) * (v > 0)[:, None]); vals_out.append(v)
    return torch.cat(dirs_out, -1), torch.cat(vals_out, -1)
