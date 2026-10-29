"""The lift: an array read as a field over positions × directions, from its local lagged autocorrelation.

A population of structures along ``d`` is nearly invariant along ``d``, so its contribution to the array's local
autocorrelation ``C(l)`` is a ridge along ``d``; independent populations add. Read on a shell of lags, the normalised
local autocorrelation as a function of direction therefore peaks at every population's direction — the **lag ODF**.
The structure tensor is its second moment at zero lag and holds one direction; the lift holds any number.

Everything is exact on the lattice: the lags are the integer vectors of a shell ``r_min ≤ |l| ≤ r_max`` (no
interpolation, so no direction-dependent blur), each lag's products are pooled onto a lattice of ``step`` voxels and
summed into the directions of a hemisphere grid with an angular kernel, then averaged over the window ``sigma_w`` on
the lattice. The cost is one shifted product per lag at full resolution and everything else on the lattice.

Per block:

1. :func:`lift` -- ``ρ (n_dirs, z, y, x)`` on the lattice ``origin + step * i``;
2. :func:`presence` -- the lift's peak above its median direction; the sensitivity weight rises from the presence
   floor measured on an isotropic surrogate of the block (:func:`iso_surrogate`: the block's own radially averaged
   power spectrum with random phases) to twice it;
3. :func:`deconvolve` -- the angular blur of one population (its **response**, :func:`self_response`: measured on
   the block's own most anisotropic single-lobe positions, or :func:`response_profile` on data with known
   directions) removed by non-negative least squares.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .filters import gaussian


def hemisphere(n: int = 160, device="cpu", dims: int = 3) -> torch.Tensor:
    """Roughly uniform unit vectors on a hemisphere (z >= 0), a Fibonacci set, ``(n, 3)`` in (z, y, x); with ``dims=2``
    a semicircle in the (y, x) plane (z = 0), ``n`` directions over 180°."""
    if dims == 2:
        phi = (torch.arange(n, dtype=torch.float64) + 0.5) * math.pi / n
        return torch.stack([torch.zeros(n, dtype=torch.float64), torch.sin(phi), torch.cos(phi)], 1).float().to(device)
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = torch.sqrt(1 - z * z)
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)


def shell_lags(r_min: float, r_max: float, dims: int = 3) -> np.ndarray:
    """The integer lag vectors ``(n, 3)`` in (z, y, x) with ``r_min <= |l| <= r_max`` on one half of the lattice (the
    first nonzero component positive): every lag and its negative are read together. With ``dims=2`` the lags lie in
    the (y, x) plane (z = 0)."""
    R = int(math.ceil(r_max))
    ax = np.arange(-R, R + 1)
    L = np.stack(np.meshgrid(*([ax] * dims), indexing="ij"), -1).reshape(-1, dims)
    n = np.linalg.norm(L, axis=1)
    L = L[(n >= r_min) & (n <= r_max)]
    first = np.argmax(L != 0, axis=1)
    keep = L[np.arange(len(L)), first] > 0
    L = L[keep].astype(np.int64)
    return np.concatenate([np.zeros((len(L), 1), np.int64), L], 1) if dims == 2 else L


def direction_weights(grid: torch.Tensor, lags: np.ndarray, kernel_deg: float, radial: str = "uniform") -> torch.Tensor:
    """``W (n_dirs, n_lags)``: each lag's unit direction spread over the grid by an axial Gaussian kernel of
    ``kernel_deg`` (a lag counts for the grid directions it lies near), normalised per grid direction so that every
    direction averages the lags around it. ``radial="uniform"`` weights each lag by ``1 / |l|²`` so that every radius
    of the shell counts alike (the lattice has ``∝ r²`` lags at radius ``r``; without the weight the longest lags, where
    a texture's correlation has decayed, would dominate); ``"count"`` leaves the lattice's weighting."""
    r = np.linalg.norm(lags, axis=1)
    u = torch.as_tensor(lags / r[:, None], dtype=torch.float32, device=grid.device)
    k = 1.0 / (2 * math.sin(math.radians(kernel_deg)) ** 2)
    W = torch.exp(k * ((grid @ u.T) ** 2 - 1))                                                   # (n_dirs, n_lags)
    if radial == "uniform":
        W = W * torch.as_tensor(1.0 / r ** 2, dtype=torch.float32, device=grid.device)[None]
    return W / W.sum(1, keepdim=True).clamp_min(1e-12)


def _axes(step, origin):
    """Per-axis ``(step, origin)`` tuples from scalars or tuples."""
    st = (step,) * 3 if isinstance(step, int) else tuple(int(s) for s in step)
    og = (origin,) * 3 if isinstance(origin, int) else tuple(int(o) for o in origin)
    return st, og


def _pool(x: torch.Tensor, step, origin) -> torch.Tensor:
    """``x (Z, Y, X)`` box-averaged onto cells of ``step`` voxels per axis (a scalar or a tuple) centred on
    ``origin + step * i`` (the cell of position ``i`` covers voxels ``origin - step // 2 + step * i ...``),
    replicate-padded at the faces; an axis with step 1 is left as it is."""
    st, og = _axes(step, origin)
    if all(s == 1 for s in st):
        return x
    pad = []
    for n, s, o in zip(reversed(x.shape), reversed(st), reversed(og)):
        lo = s // 2 - o
        hi = (-(n + lo)) % s
        pad += [max(lo, 0), hi]
    xp = Fn.pad(x[None, None], pad, mode="replicate")
    return Fn.avg_pool3d(xp, st)[0, 0]


def lift(v: torch.Tensor, grid: torch.Tensor, *, r_min: float = 2.5, r_max: float = 6.5, sigma_w: float = 10.0, sigma_m: float = 8.0,
         step: int = 4, origin: int = 2, kernel_deg: float = 8.0, lag_batch: int = 16, lags: np.ndarray | None = None,
         weights: torch.Tensor | None = None, sigma_f: float = 0.0, radial: str = "uniform") -> torch.Tensor:
    """``v (Z, Y, X)`` or ``(K, Z, Y, X)`` -> ``ρ (n_dirs, z, y, x)``: the normalised local autocorrelation over the
    shell of lags, read in the grid's directions, on the lattice. The products and the variance are box-pooled onto
    cells of ``step`` voxels and then averaged with a Gaussian of ``sigma_w`` voxels; the mean is removed at
    ``sigma_m`` and, with ``sigma_f`` > 0, the array first smoothed at ``sigma_f`` (a band-pass: voxel-scale structure --
    slice-wise acquisition artefacts, pixel noise -- does not enter the correlation). Channels add. Outside the block the products are zero (the faces are unreliable within
    ``r_max + 2 sigma_w`` voxels; see :meth:`Lift.interior`). ``weights (Z, Y, X)`` in [0, 1] leave voxels out of the
    correlation (a pair counts with ``w(x) w(x + l)``, and every direction is normalised by the weight of the pairs it
    kept) -- the structure tensor's excision carried over: a boundary's step cannot lift to a ring."""
    F = (v[None] if v.dim() == 3 else v).float()
    Fc = (gaussian(F, sigma_f) if sigma_f > 0 else F) - gaussian(F, sigma_m)                    # a band-pass: structure between sigma_f and sigma_m
    del F
    L = shell_lags(r_min, r_max) if lags is None else lags
    W = direction_weights(grid, L, kernel_deg, radial)                                         # (n_dirs, n_lags)
    Z, Y, X = Fc.shape[1:]
    cells = _pool(torch.zeros((Z, Y, X), device=Fc.device), step, origin).shape
    w = None if weights is None else weights.to(Fc.device).float()
    if w is not None:
        Fc = Fc * w[None]
    var = _pool((Fc * Fc).sum(0), step, origin)
    O = torch.zeros((grid.shape[0],) + tuple(cells), device=Fc.device)
    Q = torch.zeros_like(O) if w is not None else None
    st_, _ = _axes(step, origin)
    sw = tuple((sigma_w / s_) if n_ > 1 else 0.0 for s_, n_ in zip(st_, cells))                   # no window along a one-voxel axis (2-D)
    for b0 in range(0, len(L), lag_batch):
        Lb = L[b0:b0 + lag_batch]
        P = torch.empty((len(Lb),) + tuple(cells), device=Fc.device)
        Pw = torch.empty_like(P) if w is not None else None
        for j, l in enumerate(Lb):
            a = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(l, (Z, Y, X)))             # x
            b = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip(l, (Z, Y, X)))             # x + l
            pa = (Fc[(slice(None),) + a] * Fc[(slice(None),) + b]).sum(0)                           # f(x) f(x + l) at x ...
            prod = torch.zeros((Z, Y, X), device=Fc.device)
            prod[a] = pa; prod[b] += pa                                                             # ... and f(y) f(y - l) at y = x + l
            P[j] = _pool(prod, step, origin) / 2
            if w is not None:
                wa = w[a] * w[b]; prod = torch.zeros((Z, Y, X), device=Fc.device); prod[a] = wa; prod[b] += wa
                Pw[j] = _pool(prod, step, origin) / 2
        O += (W[:, b0:b0 + lag_batch] @ P.reshape(len(Lb), -1)).reshape(grid.shape[0], *cells)
        if w is not None:
            Q += (W[:, b0:b0 + lag_batch] @ Pw.reshape(len(Lb), -1)).reshape(grid.shape[0], *cells)
        del P, Pw
    if any(x_ > 0 for x_ in sw):
        O = gaussian(O, sw, truncate=3.0); var = gaussian(var, sw, truncate=3.0)
        if Q is not None:
            Q = gaussian(Q, sw, truncate=3.0)
    if Q is not None:                                                                          # each direction over the weight of its own pairs,
        wvar = _pool(w * w, step, origin); wvar = gaussian(wvar, sw, truncate=3.0) if any(x_ > 0 for x_ in sw) else wvar
        return (O / Q.clamp_min(1e-12)) / (var / wvar.clamp_min(1e-12)).clamp_min(1e-12)[None]   # the variance over the weight of its voxels
    return O / var.clamp_min(1e-12)[None]


def iso_surrogate(v: torch.Tensor, mask=None, seed: int = 0, isotropic: bool = True) -> torch.Tensor:
    """A random field with the block's power spectrum and random phases: the same scales and contrast, no local
    structure -- the null for :func:`presence`. With ``isotropic`` (the default) the spectrum is radially averaged, so
    the surrogate has no anisotropy at all; without, it keeps the block's full spectrum, including the anisotropy of
    the acquisition (an elongated point-spread function, ring and streak energy, slice-wise structure) *and* of any
    dominant texture orientation -- the surrogate for the **spectrum null** (:meth:`Orientations.null_profile`), to be
    measured on a region with many orientations. ``mask`` restricts the spectrum to the tissue (zero elsewhere)."""
    v = v.float()
    x = v - v.mean() if mask is None else torch.where(mask, v - v[mask].mean(), torch.zeros_like(v))
    P = torch.fft.fftn(x).abs() ** 2
    g = torch.Generator(device=v.device).manual_seed(seed)
    noise = torch.fft.fftn(torch.randn(v.shape, device=v.device, generator=g))
    if isotropic:
        k = torch.stack(torch.meshgrid(*(torch.fft.fftfreq(n, device=v.device) for n in v.shape), indexing="ij")).norm(dim=0)
        nb = 256; idx = (k / k.max() * (nb - 1)).round().long()
        prof = torch.zeros(nb, device=v.device).scatter_add_(0, idx.flatten(), P.flatten()) / torch.bincount(idx.flatten(), minlength=nb).clamp_min(1)
        s = torch.fft.ifftn(noise * prof[idx].sqrt()).real
    else:
        s = torch.fft.ifftn(noise * P.sqrt()).real
    ref = x[mask] if mask is not None else x
    return s / s.std().clamp_min(1e-12) * ref.std()


def presence(O: torch.Tensor, mode: str = "excess") -> torch.Tensor:
    """How much of the lift stands above its median direction, ``(z, y, x)``: near zero for isotropic texture.
    ``"excess"``: the mean positive excess over the directions (the mass in the lobes, whatever their number -- two
    populations sharing the mass measure the same as one); ``"peak"``: the peak's height above the median (halves when
    the mass splits into two lobes)."""
    med = O.median(0).values
    if mode == "peak":
        return O.max(0).values - med
    return (O - med[None]).clamp_min(0).mean(0) * 4.0                                              # x4: about the peak height for one lobe of the usual width


def response_profile(O: torch.Tensor, dirs: torch.Tensor, grid: torch.Tensor, mask=None, bin_deg: float = 2.5) -> torch.Tensor:
    """The mean of ``O - min`` against the angle to each position's population direction ``dirs (3, ...)`` over
    ``mask``, normalised to 1 at 0°, in ``bin_deg`` bins over [0, 90] (empty bins interpolated), monotone."""
    nd = O.shape[0]; Of = O.reshape(nd, -1); Of = Of - Of.min(0, keepdim=True).values
    D = dirs.reshape(3, -1)
    if mask is not None:
        m = mask.reshape(-1); Of, D = Of[:, m], D[:, m]
    ang = torch.rad2deg(torch.acos((grid @ D).abs().clamp(max=1.0)))                              # (nd, V)
    nb = int(round(90 / bin_deg)); idx = (ang / bin_deg).long().clamp(max=nb - 1)
    s = torch.zeros(nb, device=O.device).scatter_add_(0, idx.flatten(), Of.flatten())
    c = torch.zeros(nb, device=O.device).scatter_add_(0, idx.flatten(), torch.ones_like(Of.flatten()))
    R = (s / c.clamp_min(1)).cpu().numpy(); ok = c.cpu().numpy() > 0
    x = np.arange(nb)
    R = np.interp(x, x[ok], R[ok]) if ok.any() else np.ones(nb)
    R = np.minimum.accumulate(R / max(R[0], 1e-12))
    return torch.as_tensor(R, dtype=torch.float32, device=O.device)


def _profile_at(R: torch.Tensor, ang_deg: torch.Tensor, bin_deg: float) -> torch.Tensor:
    x = (ang_deg / bin_deg - 0.5).clamp(0, R.numel() - 1)
    i0 = x.floor().long().clamp(max=R.numel() - 2); f = x - i0
    return R[i0] * (1 - f) + R[i0 + 1] * f


def response_matrix(R: torch.Tensor, grid: torch.Tensor, bin_deg: float = 2.5, rings: bool = False) -> torch.Tensor:
    """The deconvolution's dictionary. Columns ``0 .. n_dirs - 1``: a population along grid direction ``j`` -- the response
    ``R`` (a profile over the angle to ``j``, linear in angle); with ``rings`` columns ``n_dirs .. 2 n_dirs - 1``: a sheet
    with normal ``j`` -- the great circle perpendicular to ``j`` blurred by the same response (``R`` over the angle to
    the circle, ``90° - angle to the normal``); the last column an isotropic term. A boundary that contains a
    population's direction is then explained by a lobe and a ring, not by two lobes."""
    ang = torch.rad2deg(torch.acos((grid @ grid.T).abs().clamp(max=1.0)))                        # (n_dirs, n_dirs)
    cols = [_profile_at(R, ang, bin_deg)]
    if rings:
        cols.append(_profile_at(R, 90.0 - ang, bin_deg))
    cols.append(torch.ones(grid.shape[0], 1, device=grid.device))
    return torch.cat(cols, 1)


def deconvolve(O: torch.Tensor, A: torch.Tensor, iters: int = 300, all_terms: bool = False) -> torch.Tensor:
    """Non-negative ``f`` with ``A f ≈ O - min`` per position (FISTA): the lobe coefficients ``(n_dirs, ...)``, or with
    ``all_terms`` every column's ``(A.shape[1], ...)``."""
    nd = O.shape[0]; shp = O.shape[1:]
    B = O.reshape(nd, -1); B = B - B.min(0, keepdim=True).values
    L = float(torch.linalg.matrix_norm(A, 2)) ** 2
    AtA, AtB = A.T @ A, A.T @ B
    x = torch.zeros(A.shape[1], B.shape[1], device=O.device); y = x.clone(); t = 1.0
    for _ in range(iters):
        xn = (y - (AtA @ y - AtB) / L).clamp_min(0)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
    return (x if all_terms else x[:nd]).reshape(-1, *shp)


def angular_smooth(f: torch.Tensor, grid: torch.Tensor, deg: float) -> torch.Tensor:
    if deg <= 0:
        return f
    k = 1.0 / (2 * math.sin(math.radians(deg)) ** 2)
    K = torch.exp(k * ((grid @ grid.T) ** 2 - 1)); K = K / K.sum(1, keepdim=True)
    return (K @ f.reshape(f.shape[0], -1)).reshape(f.shape)


def default_response(bin_deg: float = 2.5, half_width_deg: float = 33.0, device="cpu") -> torch.Tensor:
    """A Gaussian profile over the angle with the given half-height width (33°: the single-population response
    measured on the HiP-CT phantoms at the default shell), for blocks that offer no single-lobe positions to measure
    one on."""
    x = torch.arange(int(round(90 / bin_deg)), dtype=torch.float32, device=device) * bin_deg
    s = half_width_deg / math.sqrt(2 * math.log(2))
    return torch.exp(-0.5 * (x / s) ** 2)


def self_response(O: torch.Tensor, grid: torch.Tensor, *, top: float = 0.05, single: float = 0.2, bin_deg: float = 2.5, min_cells: int = 200):
    """A response from the block itself: the positions in the top ``top`` share of :func:`presence` whose second lobe
    is below ``single`` of the first, aligned on their own (refined) peaks. If fewer than ``min_cells`` qualify the
    selection is loosened step by step (``top`` to 0.2, ``single`` to 0.5); if still too few, :func:`default_response`.
    Returns the profile and the count (0 for the default)."""
    from .populations import peaks
    p = presence(O)
    base = O - O.min(0, keepdim=True).values
    d, m = peaks(base, grid, nmax=2, rel=0.0, min_sep_deg=30.0)
    for t, sg in ((top, single), (0.1, 0.3), (0.2, 0.4), (0.2, 0.5)):
        thr = torch.quantile(p.flatten()[:: max(1, p.numel() // 200_000)], 1 - t)
        sel = (p >= thr) & (m[1] <= sg * m[0])
        if int(sel.sum()) >= min_cells:
            return response_profile(O, d[0], grid, sel, bin_deg), int(sel.sum())
    return default_response(bin_deg, device=O.device), 0
