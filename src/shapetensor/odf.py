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


def peaks(coef: torch.Tensor, lmax: int, nmax: int = 3, rel: float = 0.5, min_sep_deg: float = 25.0,
          nbr_deg: float = 7.0, grid: int = 2000, floor: float = 0.0):
    """Local maxima of the ODF given by ``coef (N, ncoef)`` on a hemisphere grid: returns ``dirs (N, nmax, 3)``
    (zero rows where absent), ``amps (N, nmax)`` and ``count (N,)``. A peak must exceed ``rel`` x the site's maximum
    and the absolute ``floor``, be a maximum over its ``nbr_deg`` neighbourhood, and lie ``min_sep_deg`` from kept
    peaks (axially)."""
    dev = coef.device
    D = torch.as_tensor(hemisphere(grid), dtype=torch.float32, device=dev)
    Y = real_sh(D, lmax)
    O = coef @ Y.T                                                     # (N, grid)
    O = O - O.min(1, keepdim=True).values
    cosnb = math.cos(math.radians(nbr_deg))
    adj = ((D @ D.T).abs() > cosnb)                                    # (grid, grid) incl. self
    # a grid point is a local max if it is >= every neighbour
    # (dense max over neighbours; grid 2000 -> 4e6 entries, fine)
    nb_max = torch.full_like(O, -1.0)
    for a in range(0, grid, 200):
        m = adj[a:a + 200]                                             # (200, grid)
        vals = torch.where(m[None], O[:, None, :], torch.full_like(O[:, None, :], -1.0))
        nb_max[:, a:a + 200] = vals.max(2).values
    is_max = (O >= nb_max - 1e-7)
    mx = O.max(1, keepdim=True).values
    cand = is_max & (O >= rel * mx) & (O > floor)
    N = coef.shape[0]
    dirs = torch.zeros((N, nmax, 3), device=dev)
    amps = torch.zeros((N, nmax), device=dev)
    count = torch.zeros((N,), dtype=torch.int64, device=dev)
    cos_sep = math.cos(math.radians(min_sep_deg))
    order = torch.argsort(O * cand.float(), dim=1, descending=True)[:, :20]     # top candidates per site
    Oc = torch.gather(O, 1, order)
    Cc = torch.gather(cand, 1, order)
    for n in range(N):
        kept = []
        for j in range(order.shape[1]):
            if not bool(Cc[n, j]):
                break
            d = D[order[n, j]]
            if all(float((d @ D[k]).abs()) < cos_sep for k in kept):
                kept.append(int(order[n, j]))
                dirs[n, len(kept) - 1] = d
                amps[n, len(kept) - 1] = Oc[n, j]
                if len(kept) == nmax:
                    break
        count[n] = len(kept)
    return dirs, amps, count
