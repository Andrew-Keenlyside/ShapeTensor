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


_SHC: dict = {}


def _sh_consts(lmax: int, dev):
    key = (lmax, str(dev))
    if key not in _SHC:
        M = lmax + 1
        cd = torch.tensor([1.0] + [-math.sqrt((2 * m + 1) / (2 * m)) for m in range(1, M)], dtype=torch.float32)
        c1 = torch.tensor([math.sqrt(2 * m + 3) for m in range(M)], dtype=torch.float32)
        A = torch.zeros(M, M); Bc = torch.zeros(M, M)
        for l in range(2, M):
            for m in range(0, l - 1):
                a = math.sqrt((4 * l * l - 1) / (l * l - m * m)); b = a * math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
                A[l, m] = a; Bc[l, m] = b
        mm = torch.arange(M)
        rec = mm[None, :] <= (torch.arange(M)[:, None] - 2); sub = mm[None, :] == (torch.arange(M)[:, None] - 1); dia = mm[None, :] == torch.arange(M)[:, None]
        S = torch.tensor([math.sqrt(2.0) * (-1) ** m for m in range(M)], dtype=torch.float32)
        ls = list(range(0, lmax + 1, 2)); idx = []
        for li, l in enumerate(ls):                                                                   # columns: (table, l-row, m) -> flat index
            for m in range(-l, l + 1):
                idx.append((0 if m == 0 else (2 if m < 0 else 1), li, abs(m)))
        flat = torch.tensor([(t * len(ls) + li) * M + m for t, li, m in idx])
        _SHC[key] = [x.to(dev) for x in (cd, c1, A, Bc, rec, sub, dia, S, torch.arange(M, dtype=torch.float32), flat)] + [ls]
    return _SHC[key]


def real_sh(dirs: torch.Tensor, lmax: int) -> torch.Tensor:
    """Real even spherical harmonics at unit ``dirs (N, 3)`` in (z, y, x): ``(N, n_coef)``, orthonormal on the sphere.
    ``m < 0: sqrt2 (-1)^m Im Y_l^|m|``, ``m = 0: Y_l^0``, ``m > 0: sqrt2 (-1)^m Re Y_l^m``; polar axis z, azimuth from x
    towards y; fully normalised Legendre recurrence -- vectorised over m (each element's float32 arithmetic in the
    order of :func:`_real_sh_loop`, so bitwise equal; ~3.5x fewer kernel launches at lmax 16: review 5)."""
    d = dirs.float(); N = d.shape[0]; dev = d.device
    cd, c1, A, Bc, rec, sub, dia, S, mf, flat, ls = _sh_consts(lmax, dev)
    cz = d[:, 0].clamp(-1, 1); sz = (1 - cz * cz).clamp_min(0).sqrt(); phi = torch.atan2(d[:, 1], d[:, 2]); M = lmax + 1
    diag = [torch.full((N,), 1.0 / math.sqrt(4 * math.pi), device=dev)]                            # P[m][m]: a product chain
    for m in range(1, M):
        diag.append(cd[m] * sz * diag[-1])
    Pmm = torch.stack(diag, 1)                                                                      # (N, M)
    SUB = c1 * cz[:, None] * Pmm                                                                    # P[m+1][m] = c1_m cz P[m][m]
    zero = torch.zeros((), device=dev)
    rows = {}; p2 = torch.zeros(N, M, device=dev); p1 = torch.zeros(N, M, device=dev)
    for l in range(M):
        cur = torch.where(rec[l], A[l] * cz[:, None] * p1 - Bc[l] * p2, torch.where(sub[l], SUB, torch.where(dia[l], Pmm, zero)))
        rows[l] = cur; p2, p1 = p1, cur
    P = torch.stack([rows[l] for l in ls], 1)                                                       # (N, L, M)
    cosm = torch.cos(mf * phi[:, None])[:, None]; sinm = torch.sin(mf * phi[:, None])[:, None]
    T = torch.stack([P, S * P * cosm, S * P * sinm], 1)                                             # (N, 3, L, M)
    return T.reshape(N, -1).index_select(1, flat)


def _real_sh_loop(dirs: torch.Tensor, lmax: int) -> torch.Tensor:
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
    return torch.einsum("mc,c...->m...", Y, coef[:Y.shape[1]].float())                               # extra channels (descriptor densities) ignored


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


def peaks_gated(coef: torch.Tensor, lmax: int, sharp: float = 0.5, neg: float = 0.2, second: float = 0.42, **kw):
    """Calibrated peaks of ``coef (c, N)``: a cell's density counts only if its mean is positive, at most ``neg`` of
    the sphere is negative and its top peak stands ``sharp`` x the mean above the mean; a second peak counts if it
    stands at least ``second`` of the first's height above the mean ((f2 - mean) / (f1 - mean); 0.42 gives the exact
    target a 10% false-second rate on single-line cells). Returns ``dirs (nmax, 3, N)``, ``vals (nmax, N)`` (zeros
    where absent) and the gate ``(N,)``."""
    nmax = kw.pop("nmax", 3)
    D, V = peaks(coef, lmax, nmax=nmax, rel=0.0, **kw)
    mean = coef[0].float() / math.sqrt(4 * math.pi)                                                   # the density's mean over the sphere
    g = fibonacci_sphere(800, coef.device); f = evaluate_sh(coef, lmax, g)                             # (800, N)
    negshare = (f < 0).float().mean(0)
    gate = (mean > 0) & (negshare <= neg) & (V[0] - mean >= sharp * mean)
    rel2 = (V - mean[None]) / (V[0] - mean).clamp_min(1e-9)[None]                                      # (nmax, N)
    keep = gate[None] & (V > 0) & ((torch.arange(nmax, device=V.device)[:, None] == 0) | (rel2 >= second))
    return D * keep[:, None], V * keep, gate
