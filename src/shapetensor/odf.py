"""Spherical-harmonic machinery for the gradient-direction ODF: the real even SH basis (torch, fully normalised
recurrences), the Funk-Radon transform (q-ball), constrained spherical deconvolution with a single-population "ring"
response (Tournier 2007, applied to gradient directions), and peak finding.

Why it transfers from dMRI: one fibre population's gradients lie in the plane perpendicular to it, so the energy
distribution of gradient directions is a great-circle RING -- high perpendicular to the fibre, as the diffusion
signal is. The FRT turns each ring into a peak at the fibre; CSD deconvolves the measured ring to sharpen it. A step
edge contributes a DELTA (one direction, its normal), which the FRT turns into a ring and CSD into a smeared band, so
it does not produce a sharp false peak -- but its energy should be excised first when it is large (TCA names it).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def n_coef(lmax: int) -> int:
    return sum(2 * l + 1 for l in range(0, lmax + 1, 2))


def lm_index(lmax: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(0, lmax + 1, 2) for m in range(-l, l + 1)]


def real_sh(dirs: torch.Tensor, lmax: int) -> torch.Tensor:
    """Real even spherical harmonics at unit ``dirs (N, 3)`` (z, y, x order of the volume axes): ``(N, ncoef)``.

    Convention (matches scipy ``sph_harm_y`` with the Condon-Shortley phase, as the prototypes used):
    ``m < 0: sqrt2 (-1)^m Im Y_l^|m|``, ``m = 0: Y_l^0``, ``m > 0: sqrt2 (-1)^m Re Y_l^m``. The polar axis is the
    volume's first axis (z); azimuth from the third (x) towards the second (y). Fully normalised Legendre recurrence,
    so no factorials.
    """
    d = dirs.float()
    cz = d[:, 0].clamp(-1, 1)
    sz = (1 - cz * cz).clamp_min(0).sqrt()
    phi = torch.atan2(d[:, 1], d[:, 2])
    N = d.shape[0]
    # P[l][m] fully normalised (includes 1/sqrt(4 pi) and the CS phase)
    P = [[None] * (lmax + 1) for _ in range(lmax + 1)]
    P[0][0] = torch.full((N,), 1.0 / math.sqrt(4 * math.pi), device=d.device)
    for m in range(1, lmax + 1):
        P[m][m] = -math.sqrt((2 * m + 1) / (2 * m)) * sz * P[m - 1][m - 1]
    for m in range(0, lmax):
        P[m + 1][m] = math.sqrt(2 * m + 3) * cz * P[m][m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = a * math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[l][m] = a * cz * P[l - 1][m] - b * P[l - 2][m]
    cols = []
    s2 = math.sqrt(2.0)
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            if m == 0:
                cols.append(P[l][0])
            elif m > 0:
                cols.append(s2 * (-1) ** m * P[l][m] * torch.cos(m * phi))
            else:
                cols.append(s2 * (-1) ** m * P[l][-m] * torch.sin(-m * phi))
    return torch.stack(cols, 1)


def frt_weights(lmax: int) -> torch.Tensor:
    """Funk-Hecke: an order-l coefficient is multiplied by ``2 pi P_l(0)``."""
    from scipy.special import eval_legendre
    w = np.concatenate([np.full(2 * l + 1, 2 * np.pi * eval_legendre(l, 0.0)) for l in range(0, lmax + 1, 2)])
    return torch.as_tensor(w, dtype=torch.float32)


def hemisphere(n: int = 2000) -> np.ndarray:
    """Near-uniform unit vectors on the z >= 0 hemisphere (Fibonacci lattice), as (z, y, x)."""
    k = np.arange(n) + 0.5
    c = 1 - k / n
    r = np.sqrt(1 - c ** 2)
    ph = k * np.pi * (3 - np.sqrt(5))
    return np.stack([c, r * np.sin(ph), r * np.cos(ph)], 1)


def zonal_kernel(lmax: int, profile, n_theta: int = 2000) -> torch.Tensor:
    """Per-coefficient factors of the spherical convolution with the axially symmetric kernel ``profile(cos_theta)``:
    Funk-Hecke, ``f_lm -> 2 pi int_{-1}^{1} profile(t) P_l(t) dt * f_lm``."""
    from scipy.special import eval_legendre
    t = np.linspace(-1, 1, n_theta)
    pr = np.asarray(profile(t), dtype=np.float64)
    out = []
    for l in range(0, lmax + 1, 2):
        lam = 2 * np.pi * np.trapezoid(pr * eval_legendre(l, t), t)
        out.extend([lam] * (2 * l + 1))
    return torch.as_tensor(np.array(out), dtype=torch.float32)


def ring_profile(width: float):
    """A great-circle ring of angular half-width ``width`` (radians) about the equator: ``exp(-t^2 / (2 w^2))``
    in ``t = cos(theta)`` -- the gradient-direction distribution of one fibre population with dispersion ``width``."""
    w = math.sin(width)
    return lambda t: np.exp(-0.5 * (np.asarray(t) / w) ** 2)


def fit_ring_width(coef: torch.Tensor, fibres: torch.Tensor, lmax: int, grid: int = 4000) -> float:
    """Estimate the single-population response width from sites of KNOWN fibre direction: sample each site's ODF on
    the sphere, bin by angle to its fibre, average the profile, fit a Gaussian in ``cos(theta)`` by moments."""
    dirs = torch.as_tensor(hemisphere(grid), dtype=torch.float32, device=coef.device)
    Y = real_sh(dirs, lmax)                                            # (grid, ncoef)
    vals = coef @ Y.T                                                  # (N, grid)
    t = (fibres.float() @ dirs.T).abs()                                # |cos angle| (N, grid)
    v = vals.clamp_min(0)
    m2 = (v * t * t).sum() / v.sum()
    return float(math.asin(min(0.999, math.sqrt(max(float(m2), 1e-6)))))


def frt(coef: torch.Tensor, lmax: int) -> torch.Tensor:
    return coef * frt_weights(lmax).to(coef.device)


def csd(coef: torch.Tensor, lmax: int, kernel: torch.Tensor, lam: float = 1.0, tau: float = 0.1, iters: int = 50,
        grid: int = 300, chunk: int = 20000) -> torch.Tensor:
    """Constrained spherical deconvolution (Tournier 2007) of the signal ``coef (N, ncoef)`` by the zonal
    ``kernel (ncoef,)``: ``f = argmin |diag(k) f - s|^2 + lam |L f|^2``, ``L`` the SH rows at the directions where
    the current iterate is below ``tau`` x its mean (iterated ``iters`` times). Returns the fODF coefficients."""
    dirs = torch.as_tensor(hemisphere(grid), dtype=torch.float32, device=coef.device)
    Yg = real_sh(dirs, lmax)                                           # (grid, ncoef)
    k = kernel.to(coef.device)
    out = torch.empty_like(coef)
    kk = torch.diag(k * k)
    for a in range(0, coef.shape[0], chunk):
        s = coef[a:a + chunk]
        rhs = (k * s)                                                  # (n, ncoef)
        f = rhs / (k * k).clamp_min(1e-12)                             # unconstrained start
        f[:, 0] = s[:, 0] / k[0]
        for _ in range(iters):
            amp = f @ Yg.T                                             # (n, grid)
            mean = amp.mean(1, keepdim=True)
            neg = (amp < tau * mean).float()                           # (n, grid)
            LtL = torch.einsum("ng,gi,gj->nij", neg, Yg, Yg)          # (n, ncoef, ncoef)
            A = kk[None] + lam * LtL
            f = torch.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        out[a:a + chunk] = f
    return out


_NB_CACHE = {}


def _grid_neighbours(grid: int, nbr_deg: float, device):
    """The hemisphere grid ``D (grid, 3)`` and each point's neighbours within ``nbr_deg`` (axially), as a padded
    index table ``(grid, K)`` (padded with the point itself)."""
    key = (grid, nbr_deg, str(device))
    if key not in _NB_CACHE:
        D = torch.as_tensor(hemisphere(grid), dtype=torch.float32, device=device)
        adj = ((D @ D.T).abs() > math.cos(math.radians(nbr_deg)))
        K = int(adj.sum(1).max())
        idx = torch.arange(grid, device=device)[:, None].expand(grid, K).clone()
        for i in range(grid):
            nb = torch.nonzero(adj[i], as_tuple=False).squeeze(1)
            idx[i, :nb.numel()] = nb
        _NB_CACHE[key] = (D, idx)
    return _NB_CACHE[key]


def peaks(coef: torch.Tensor, lmax: int, nmax: int = 3, rel: float = 0.5, min_sep_deg: float = 25.0,
          nbr_deg: float = 7.0, grid: int = 2000, floor: float = 0.0, n_cand: int = 8, chunk: int = 4096):
    """Local maxima of the ODF given by ``coef (N, ncoef)`` on a hemisphere grid: returns ``dirs (N, nmax, 3)``
    (zero rows where absent), ``amps (N, nmax)`` and ``count (N,)``. A peak must exceed ``rel`` x the site's maximum
    and the absolute ``floor``, be a maximum over its ``nbr_deg`` neighbourhood, and lie ``min_sep_deg`` from kept
    peaks (axially). Fully vectorised on the device (sites in chunks of ``chunk``; the ``n_cand`` strongest
    candidates per site are considered)."""
    dev = coef.device
    D, nb = _grid_neighbours(grid, nbr_deg, dev)
    Y = real_sh(D, lmax)
    N = coef.shape[0]
    dirs = torch.zeros((N, nmax, 3), device=dev)
    amps = torch.zeros((N, nmax), device=dev)
    count = torch.zeros((N,), dtype=torch.int64, device=dev)
    cos_sep = math.cos(math.radians(min_sep_deg))
    ar = torch.arange(nmax, device=dev)
    for a in range(0, N, chunk):
        O = coef[a:a + chunk] @ Y.T                                    # (n, grid)
        O = O - O.min(1, keepdim=True).values
        nb_max = O[:, nb].max(2).values                                # (n, grid): max over each point's neighbourhood
        mx = O.max(1, keepdim=True).values
        cand = (O >= nb_max - 1e-7) & (O >= rel * mx) & (O > floor)
        score = torch.where(cand, O, torch.full_like(O, -1.0))
        top_v, top_i = score.topk(n_cand, dim=1)                       # (n, n_cand)
        n = O.shape[0]
        kept = torch.zeros((n, nmax, 3), device=dev)
        kamp = torch.zeros((n, nmax), device=dev)
        cnt = torch.zeros((n,), dtype=torch.int64, device=dev)
        rows = torch.arange(n, device=dev)
        for j in range(n_cand):
            d = D[top_i[:, j]]                                         # (n, 3)
            valid = (top_v[:, j] > 0) & (cnt < nmax)
            dots = (kept * d[:, None, :]).sum(-1).abs()                # (n, nmax)
            active = ar[None, :] < cnt[:, None]
            sep_ok = ((dots < cos_sep) | ~active).all(1)
            take = valid & sep_ok
            slot = cnt.clamp(max=nmax - 1)
            kept[rows[take], slot[take]] = d[take]
            kamp[rows[take], slot[take]] = top_v[take, j]
            cnt = cnt + take.long()
        dirs[a:a + chunk], amps[a:a + chunk], count[a:a + chunk] = kept, kamp, cnt
    return dirs, amps, count
