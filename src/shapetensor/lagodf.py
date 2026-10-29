"""Fibre populations as the ridges of the local lagged autocorrelation on the sphere (classical, calibration-light).

A population of fibres along ``d`` is (nearly) invariant along ``d``, so its contribution to the image's local
autocorrelation ``C(l)`` is a ridge, constant along ``d`` and decaying across it. Independent populations add
(``C = C_1 + C_2``: intermixed crossings; a window over interleaved bundles sums their parts). At a lag radius ``r``
the normalised local autocorrelation as a function of direction, ``ρ(r u)``, therefore peaks at every population's
direction: the **lag ODF** (the lag mode of the positions x lags tensor, per position). It is to the structure tensor
what Q-ball is to the diffusion tensor: the structure tensor is its second moment at an infinitesimal lag and cannot
hold two directions.

Per position (on a lattice of ``step`` voxels):

1. :func:`lag_odf` -- ``ρ(r u)`` over ``n_dirs`` hemisphere directions, averaged over ``radii``, the products
   smoothed by a Gaussian window ``sigma_w`` (the local estimate) after removing a local mean (``sigma_m``);
2. :func:`presence` -- the peak's height above the median direction: no fibres, no peak. The sensitivity weight rises
   from the presence noise floor -- by default measured per block on an isotropic surrogate with the block's own power
   spectrum (:func:`iso_surrogate`; its 99th percentile) -- to twice it;
3. :func:`deconvolve` -- the angular blur of one population (its **response**, :func:`response_profile`: from synthetic
   bundles or from the block's own most anisotropic positions, :func:`self_response`) removed by non-negative least
   squares, which sharpens the lobes so that crossings separate;
4. optional momentum along each direction (:func:`shapetensor.populations.momentum`: smooths each population along its
   own direction and bridges where it holds on both sides, never extends a free end);
5. :func:`peaks` -- up to ``nmax`` lobes at least ``min_sep_deg`` apart, kept when their share of the position's
   distribution is at least ``min_share`` and ``rel`` of the strongest, each direction refined off the grid (the
   principal axis of its lobe). A lobe's mass is its share times the position's weight (zero below ``floor``);
6. :func:`ring_fraction` -- sheets (tissue boundaries, vessel walls, sulci) correlate along a whole plane: their
   distribution is a ring, which peak-picking would report as two crossing populations. Where the share of the
   distribution's in-plane mass lying away from every kept peak exceeds ``max_ring`` and no peak stands
   ``min_prominence`` times above the ring, the position is a sheet, N = 0.
7. :func:`label_populations` -- identities instead of per-position ranks: each position's lobe (a fixel) is linked to
   the lobe of a neighbouring position that it and that lobe best align with (mutually), within ``max_angle_deg``,
   best-aligned first, never merging two populations that share a position; the sets are the populations 1, 2, 3, ...
   (largest first), so a bundle keeps one number along its bends and through the crossings it passes, and two
   crossing bundles keep theirs.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .filters import gaussian
from .separate import hemisphere


def shift_field(F: torch.Tensor, u: torch.Tensor, t: float) -> torch.Tensor:
    """``F (K, Z, Y, X)`` sampled at ``x + t u`` (trilinear, border-padded)."""
    K, Z, Y, X = F.shape
    base = torch.stack(torch.meshgrid(*(torch.arange(n, device=F.device, dtype=torch.float32) for n in (Z, Y, X)), indexing="ij"))
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=F.device).view(3, 1, 1, 1)
    p = (base + t * u.view(3, 1, 1, 1).to(F.device)) * scale - 1.0
    g = torch.stack([p[2], p[1], p[0]], -1)[None]
    return Fn.grid_sample(F[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0]


def along(f: torch.Tensor, u: torch.Tensor, sigma: float, truncate: float = 2.5) -> torch.Tensor:
    """Scalar field ``f (z, y, x)`` smoothed along direction ``u`` (lattice units) by a Gaussian of ``sigma`` lattice
    steps, unit taps, normalised where the line leaves the block."""
    n = int(math.ceil(truncate * sigma)); f4 = torch.stack([f, torch.ones_like(f)])
    acc = torch.zeros_like(f4)
    for t in range(-n, n + 1):
        w = math.exp(-0.5 * (t / sigma) ** 2)
        acc += w * (f4 if t == 0 else _shift_zero(f4, u, float(t)))
    return acc[0] / acc[1].clamp_min(1e-12)


def _shift_zero(F: torch.Tensor, u: torch.Tensor, t: float) -> torch.Tensor:
    K, Z, Y, X = F.shape
    base = torch.stack(torch.meshgrid(*(torch.arange(n, device=F.device, dtype=torch.float32) for n in (Z, Y, X)), indexing="ij"))
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=F.device).view(3, 1, 1, 1)
    p = (base + t * u.view(3, 1, 1, 1).to(F.device)) * scale - 1.0
    g = torch.stack([p[2], p[1], p[0]], -1)[None]
    return Fn.grid_sample(F[None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def lag_odf(F: torch.Tensor, grid: torch.Tensor, *, radii=(3.0, 6.0), sigma_w: float = 10.0, sigma_m: float = 8.0,
            step: int = 4, origin: int = 2, sigma_along: float | None = None) -> torch.Tensor:
    """``F (Z, Y, X)`` or ``(K, Z, Y, X)`` -> ``ρ (n_dirs, z, y, x)`` on the lattice ``origin + step * i``: the local
    normalised autocorrelation at lags ``± r u`` (``u`` the grid's directions, averaged over ``radii``), the products
    and the variance smoothed by a Gaussian of ``sigma_w`` voxels, the mean removed at ``sigma_m``. With
    ``sigma_along`` (voxels) each direction's products and variance are further smoothed along that direction only:
    a window elongated along the population it measures, which averages along the fibres without blurring across
    them into neighbouring tissue."""
    F = (F[None] if F.dim() == 3 else F).float()
    Fc = F - gaussian(F, sigma_m)
    sl = (slice(origin, None, step),) * 3
    var = gaussian((Fc * Fc).sum(0), sigma_w)[sl]
    O = torch.empty((grid.shape[0],) + tuple(var.shape), device=F.device)
    for d in range(grid.shape[0]):
        acc = torch.zeros(F.shape[1:], device=F.device)
        for r in radii:
            acc += (Fc * (shift_field(Fc, grid[d], r) + shift_field(Fc, grid[d], -r))).sum(0)
        a = gaussian(acc / (2 * len(radii)), sigma_w)[sl]
        if sigma_along:
            O[d] = along(a, grid[d], sigma_along / step) / along(var, grid[d], sigma_along / step).clamp_min(1e-12)
        else:
            O[d] = a / var.clamp_min(1e-12)
    return O


def iso_surrogate(v: torch.Tensor, mask=None, seed: int = 0) -> torch.Tensor:
    """A random field with the block's radially averaged power spectrum (random phases): the same scales and contrast,
    no anisotropy -- the null for :func:`presence`. ``mask`` restricts the spectrum to the tissue (zero elsewhere)."""
    v = v.float()
    x = v - v.mean() if mask is None else torch.where(mask, v - v[mask].mean(), torch.zeros_like(v))
    P = torch.fft.fftn(x).abs() ** 2
    k = torch.stack(torch.meshgrid(*(torch.fft.fftfreq(n, device=v.device) for n in v.shape), indexing="ij")).norm(dim=0)
    nb = 256; idx = (k / k.max() * (nb - 1)).round().long()
    prof = torch.zeros(nb, device=v.device).scatter_add_(0, idx.flatten(), P.flatten()) / torch.bincount(idx.flatten(), minlength=nb).clamp_min(1)
    g = torch.Generator(device=v.device).manual_seed(seed)
    s = torch.fft.ifftn(torch.fft.fftn(torch.randn(v.shape, device=v.device, generator=g)) * prof[idx].sqrt()).real
    ref = x[mask] if mask is not None else x
    return s / s.std().clamp_min(1e-12) * ref.std()


def presence(O: torch.Tensor) -> torch.Tensor:
    """The lag ODF's peak above its median direction, ``(z, y, x)``: near zero for isotropic texture."""
    return O.max(0).values - O.median(0).values


def response_profile(O: torch.Tensor, dirs: torch.Tensor, grid: torch.Tensor, mask=None, bin_deg: float = 2.5) -> torch.Tensor:
    """The mean of ``O - min`` against the angle to each position's population direction ``dirs (3, ...)`` over
    ``mask``, normalised to 1 at 0°, in ``bin_deg`` bins over [0, 90] (empty bins interpolated)."""
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
    R = np.interp(x, x[ok], R[ok])
    R = np.minimum.accumulate(R / R[0])                                                             # monotone non-increasing
    return torch.as_tensor(R, dtype=torch.float32, device=O.device)


def response_matrix(R: torch.Tensor, grid: torch.Tensor, bin_deg: float = 2.5) -> torch.Tensor:
    """``(n_dirs, n_dirs + 1)``: column ``j`` the response centred on grid direction ``j`` (linear in angle), the last
    column an isotropic term."""
    ang = torch.rad2deg(torch.acos((grid @ grid.T).abs().clamp(max=1.0)))
    x = (ang / bin_deg - 0.5).clamp(0, R.numel() - 1)
    i0 = x.floor().long().clamp(max=R.numel() - 2); f = x - i0
    K = R[i0] * (1 - f) + R[i0 + 1] * f
    return torch.cat([K, torch.ones(grid.shape[0], 1, device=grid.device)], 1)


def deconvolve(O: torch.Tensor, A: torch.Tensor, iters: int = 300) -> torch.Tensor:
    """Non-negative ``f`` with ``A f ≈ O - min`` per position (FISTA): ``(n_dirs, ...)`` (the isotropic term dropped)."""
    nd = O.shape[0]; shp = O.shape[1:]
    B = O.reshape(nd, -1); B = B - B.min(0, keepdim=True).values
    L = float(torch.linalg.matrix_norm(A, 2)) ** 2
    AtA, AtB = A.T @ A, A.T @ B
    x = torch.zeros(A.shape[1], B.shape[1], device=O.device); y = x.clone(); t = 1.0
    for _ in range(iters):
        xn = (y - (AtA @ y - AtB) / L).clamp_min(0)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
    return x[:nd].reshape(nd, *shp)


def angular_smooth(f: torch.Tensor, grid: torch.Tensor, deg: float) -> torch.Tensor:
    if deg <= 0:
        return f
    k = 1.0 / (2 * math.sin(math.radians(deg)) ** 2)
    K = torch.exp(k * ((grid @ grid.T) ** 2 - 1)); K = K / K.sum(1, keepdim=True)
    return (K @ f.reshape(f.shape[0], -1)).reshape(f.shape)


def peaks(f: torch.Tensor, grid: torch.Tensor, *, nmax: int = 2, rel: float = 0.3, min_mass: float = 0.0, min_sep_deg: float = 30.0,
          nbr_deg: float | None = None, lobe_deg: float = 20.0):
    """Lobes of distributions ``f (n_dirs, ...)``: grid local maxima (no neighbour within ``nbr_deg`` higher), taken
    strongest first at least ``min_sep_deg`` apart; each lobe's mass is ``f`` over the directions within ``lobe_deg``
    of its peak that are nearer it than any other kept peak, and its direction their weighted principal axis. Lobes
    with mass below ``rel`` of the strongest or ``min_mass`` are dropped. Returns ``dirs (nmax, 3, ...)`` and
    ``mass (nmax, ...)`` (largest first, zero where absent)."""
    nd = f.shape[0]; shp = f.shape[1:]; Ff = f.reshape(nd, -1); V = Ff.shape[1]; dev = f.device
    cosg = (grid @ grid.T).abs()
    if nbr_deg is None:
        nbr_deg = 1.3 * math.degrees(math.sqrt(2 * math.pi / nd))                                   # ~1.3 grid spacings
    nbr = (cosg >= math.cos(math.radians(nbr_deg))) & ~torch.eye(nd, dtype=torch.bool, device=dev)
    is_max = torch.ones_like(Ff, dtype=torch.bool)
    for k in range(nd):
        nk = nbr[k].nonzero().flatten()
        if len(nk):
            is_max[k] = Ff[k] >= Ff[nk].max(0).values
    s = torch.where(is_max & (Ff > 0), Ff, torch.zeros_like(Ff))
    cos_sep = math.cos(math.radians(min_sep_deg)); pk = torch.full((nmax, V), -1, dtype=torch.long, device=dev)
    for r in range(nmax):
        val, idx = s.max(0); ok = val > 0
        pk[r] = torch.where(ok, idx, torch.full_like(idx, -1))
        s = torch.where((cosg[idx] >= cos_sep).T & ok[None], torch.zeros_like(s), s)
    cp = torch.stack([torch.where(pk[r] >= 0, cosg[:, pk[r].clamp_min(0)], torch.full((nd, V), -1.0, device=dev)) for r in range(nmax)])
    owner = cp.argmax(0); cos_lobe = math.cos(math.radians(lobe_deg))
    dirs = torch.zeros(nmax, 3, V, device=dev); mass = torch.zeros(nmax, V, device=dev)
    for r in range(nmax):
        w = Ff * ((owner == r) & (cp[r] >= cos_lobe)).float()
        mass[r] = w.sum(0)
        M = torch.einsum("dv,di,dj->vij", w, grid, grid)
        dirs[r] = torch.linalg.eigh(M.double())[1][:, :, -1].float().T
    order = mass.argsort(0, descending=True)
    mass = mass.gather(0, order); dirs = dirs.gather(0, order[:, None, :].expand(-1, 3, -1))
    keep = (mass >= rel * mass[:1]) & (mass >= min_mass) & (mass > 0)
    return (dirs * keep[:, None]).reshape(nmax, 3, *shp), (mass * keep).reshape(nmax, *shp)


def ring_fraction(f: torch.Tensor, grid: torch.Tensor, dirs: torch.Tensor, mass: torch.Tensor, *, plane_deg: float = 12.0,
                  away_deg: float = 25.0, prominence: bool = False):
    """For distributions ``f (n_dirs, ...)`` and their kept peaks ``dirs (nmax, 3, ...)`` (``mass`` zero where absent):
    the plane of ``f`` (normal = least axis of its second moment), the mass within ``plane_deg`` of that plane, and the
    share of it farther than ``away_deg`` from every kept peak. About 0.6 for a ring (a sheet), near 0 for one or two
    lines (populations). With ``prominence`` also the largest value of ``f`` over the mean of the ring's directions
    away from the peaks (a population lying in a sheet, such as a fascicle along its boundary, stands above it)."""
    nd = f.shape[0]; shp = f.shape[1:]; Ff = f.reshape(nd, -1); V = Ff.shape[1]
    M = torch.einsum("dv,di,dj->vij", Ff, grid, grid)
    n = torch.linalg.eigh(M.double())[1][:, :, 0].float()                                         # (V, 3) plane normal
    inplane = ((grid @ n.T).abs() <= math.sin(math.radians(plane_deg))).float()                    # (nd, V)
    D = dirs.reshape(dirs.shape[0], 3, -1); has = mass.reshape(mass.shape[0], -1) > 0
    near = torch.zeros(nd, V, dtype=torch.bool, device=f.device); cos_a = math.cos(math.radians(away_deg))
    for r in range(D.shape[0]):
        near |= ((grid @ D[r]).abs() >= cos_a) & has[r][None]
    tot = (Ff * inplane).sum(0); away = inplane * (~near).float()
    ring = ((Ff * away).sum(0) / tot.clamp_min(1e-12)).reshape(shp)
    if not prominence:
        return ring
    level = (Ff * away).sum(0) / away.sum(0).clamp_min(1)
    return ring, (Ff.max(0).values / level.clamp_min(1e-12)).reshape(shp)


def self_response(O: torch.Tensor, grid: torch.Tensor, *, top: float = 0.05, single: float = 0.2, bin_deg: float = 2.5):
    """A response from the block itself: the positions in the top ``top`` share of :func:`presence` whose second lag
    ODF lobe is below ``single`` of the first, aligned on their own (refined) peaks."""
    p = presence(O); thr = torch.quantile(p.flatten()[:: max(1, p.numel() // 200_000)], 1 - top)
    base = O - O.min(0, keepdim=True).values
    d, m = peaks(base, grid, nmax=2, rel=0.0, min_sep_deg=30.0)
    sel = (p >= thr) & (m[1] <= single * m[0])
    return response_profile(O, d[0], grid, sel, bin_deg), int(sel.sum())


class LagODF:
    """The pathway's parameters (voxels unless stated) and a response; :meth:`populations` per block."""

    def __init__(self, *, n_dirs: int = 160, radii=(3.0, 6.0), sigma_w: float = 10.0, sigma_m: float = 8.0, step: int = 4, window_along=None,
                 min_weight: float = 0.2,
                 origin: int = 2, response=None, bin_deg: float = 2.5, iters: int = 300, angular_deg: float = 6.0,
                 nmax: int = 2, rel: float = 0.3, min_sep_deg: float = 30.0, floor: float = 0.06, full: float = 0.12,
                 min_share: float = 0.15, momentum: bool = False, sigma_along: float = 3.0, reach: int = 6, max_ring: float = 0.35,
                 null_q: float = 0.99, full_factor: float = 2.0, null_crop: int = 256,
                 min_prominence: float = 5.0, edge: float | None = None, device="cuda"):
        self.grid = hemisphere(n_dirs, device)
        self.radii, self.sigma_w, self.sigma_m, self.step, self.origin = tuple(radii), float(sigma_w), float(sigma_m), int(step), int(origin)
        self.window_along, self.min_weight = window_along, float(min_weight)
        self.response, self.bin_deg, self.iters, self.angular_deg = response, float(bin_deg), int(iters), float(angular_deg)
        self.nmax, self.rel, self.min_sep_deg, self.floor, self.full, self.min_share = int(nmax), float(rel), float(min_sep_deg), float(floor), float(full), float(min_share)
        self.momentum, self.sigma_along, self.reach, self.device = bool(momentum), float(sigma_along), int(reach), device
        self.max_ring, self.min_prominence = float(max_ring), float(min_prominence)
        self.edge = float(edge) if edge is not None else max(self.radii) + 2 * self.sigma_w             # voxels too near a face to trust
        self.null_q, self.full_factor, self.null_crop = float(null_q), float(full_factor), int(null_crop)
        self.last_floor = None

    def odf(self, v: torch.Tensor) -> torch.Tensor:
        return lag_odf(v.to(self.device), self.grid, radii=self.radii, sigma_w=self.sigma_w, sigma_m=self.sigma_m, step=self.step,
                       origin=self.origin, sigma_along=self.window_along)

    def fit_response(self, volumes, dirs_fn):
        """The response from volumes with known single-population directions: ``dirs_fn(i, O) -> (dirs (3, ...), mask)``."""
        num, cnt = 0, 0
        for i, v in enumerate(volumes):
            O = self.odf(v); d, m = dirs_fn(i, O)
            n = int(m.sum()); num = num + n * response_profile(O, d, self.grid, m, self.bin_deg); cnt += n
        self.response = num / max(cnt, 1)
        return self

    def interior(self, vshape, cshape) -> torch.Tensor:
        """1 at lattice positions at least ``edge`` voxels from every face of the block, else 0."""
        m = torch.ones(tuple(cshape), device=self.device)
        for ax, (n, c) in enumerate(zip(vshape, cshape)):
            x = self.origin + self.step * torch.arange(c, device=self.device, dtype=torch.float32)
            ok = ((x >= self.edge) & (x <= n - 1 - self.edge)).float()
            shp = [1, 1, 1]; shp[ax] = c; m = m * ok.view(shp)
        return m

    def null_floor(self, v: torch.Tensor, mask=None) -> float:
        """The ``null_q`` quantile of :func:`presence` on the block's isotropic surrogate (its interior, a central crop
        of at most ``null_crop`` voxels per axis)."""
        s = iso_surrogate(v.to(self.device), mask)
        sl = tuple(slice(max(0, (n - self.null_crop) // 2), max(0, (n - self.null_crop) // 2) + min(n, self.null_crop)) for n in s.shape)
        sc = s[sl].contiguous(); pn = presence(self.odf(sc))
        inner = self.interior(sc.shape, pn.shape) > 0
        vals = pn[inner] if inner.any() else pn.flatten()
        return float(torch.quantile(vals[:: max(1, vals.numel() // 500_000)], self.null_q))

    def weight(self, p: torch.Tensor, floor: float | None = None) -> torch.Tensor:
        """Sensitivity weight from presence: 0 at the floor, 1 from ``full_factor`` x the floor (``floor`` /
        ``full`` when the floor is fixed), linear between."""
        lo = self.floor if floor is None else floor
        hi = self.full if floor is None else self.full_factor * floor
        return ((p - lo) / max(hi - lo, 1e-6)).clamp(0, 1)

    def populations(self, v: torch.Tensor, *, O=None, response=None, use_momentum=None, floor="null", mask=None) -> dict:
        """``floor="null"`` (default): the presence floor from the block's isotropic surrogate (:meth:`null_floor`,
        tissue ``mask`` optional); a number: that floor (weight 1 from ``full_factor`` x it); None: the fixed
        ``floor`` / ``full``."""
        O = self.odf(v) if O is None else O
        R = response if response is not None else self.response
        if R is None:
            R, _ = self_response(O, self.grid, bin_deg=self.bin_deg)
        A = response_matrix(R, self.grid, self.bin_deg)
        f = angular_smooth(deconvolve(O, A, self.iters), self.grid, self.angular_deg)
        if isinstance(floor, str):
            floor = self.null_floor(v, mask)
        self.last_floor = floor
        p = presence(O); w = self.weight(p, floor) * self.interior(v.shape[-3:], O.shape[1:])
        g = f / f.sum(0, keepdim=True).clamp_min(1e-12) * w[None]                                 # sums to the weight
        if self.momentum if use_momentum is None else use_momentum:
            from .populations import momentum
            g = momentum(g, self.grid, sigma_along=self.sigma_along, reach=self.reach, angular_deg=0.0)
        wt = g.sum(0)                                                                              # the weight after bridging
        dirs, share = peaks(g / wt.clamp_min(1e-12)[None], self.grid, nmax=self.nmax, rel=self.rel, min_mass=self.min_share,
                            min_sep_deg=self.min_sep_deg)
        ring, prom = ring_fraction(g, self.grid, dirs, share, prominence=True)
        sheet = (ring > self.max_ring) & (prom < self.min_prominence) & (wt >= self.min_weight)
        mass = share * ((wt >= self.min_weight) & ~sheet).float()[None] * wt[None]                 # N = 0 below the weight floor and on sheets
        return {"N": (mass > 0).sum(0), "mass": mass, "share": share, "dirs_cells": dirs, "presence": p, "weight": wt, "fodf": g,
                "ring": ring, "prominence": prom, "sheet": sheet}


def _half_offsets(connectivity: int = 26):
    offs = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) > (0, 0, 0)]
    if connectivity == 6:
        offs = [o for o in offs if sum(map(abs, o)) == 1]
    elif connectivity == 18:
        offs = [o for o in offs if sum(map(abs, o)) <= 2]
    return offs


def _constrained_components(n: int, src, dst, order, pa, pb, tol: float = 0.0, tol_min: int = 0):
    """Kruskal over edges ``src[order], dst[order]`` with cannot-link pairs ``(pa[i], pb[i])`` (fixels of one position):
    two components merge only if at most ``max(tol_min, tol * size of the smaller)`` cannot-link pairs would fall
    inside the merged one. Returns each node's root."""
    global _KRUSKAL
    if _KRUSKAL is None:
        _KRUSKAL = _compile_kruskal()
    return _KRUSKAL(n, src, dst, order, pa, pb, float(tol), int(tol_min))


_KRUSKAL = None


def _compile_kruskal():
    from numba import njit

    @njit(cache=True)
    def run(n, src, dst, order, pa, pb, tol, tol_min):
        parent = np.arange(n); size = np.ones(n, np.int64)
        m = 2 * len(pa)
        ent_partner = np.empty(m, np.int64); ent_next = np.full(m, -1, np.int64)
        head = np.full(n, -1, np.int64); tail = np.full(n, -1, np.int64); length = np.zeros(n, np.int64)
        for i in range(len(pa)):
            for e, a, b in ((2 * i, pa[i], pb[i]), (2 * i + 1, pb[i], pa[i])):
                ent_partner[e] = b
                if head[a] == -1:
                    head[a] = e
                else:
                    ent_next[tail[a]] = e
                tail[a] = e; length[a] += 1

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                nx = parent[x]; parent[x] = r; x = nx
            return r
        for j in range(len(order)):
            ra = find(src[order[j]]); rb = find(dst[order[j]])
            if ra == rb:
                continue
            small, other = (ra, rb) if length[ra] <= length[rb] else (rb, ra)
            allow = max(tol_min, int(tol * min(size[ra], size[rb])))
            e = head[small]; bad = 0
            while e != -1:
                if find(ent_partner[e]) == other:
                    bad += 1
                    if bad > allow:
                        break
                e = ent_next[e]
            if bad > allow:
                continue
            big, little = (ra, rb) if size[ra] >= size[rb] else (rb, ra)
            parent[little] = big; size[big] += size[little]
            if head[little] != -1:
                if head[big] == -1:
                    head[big] = head[little]
                else:
                    ent_next[tail[big]] = head[little]
                tail[big] = tail[little]; length[big] += length[little]
        roots = np.empty(n, np.int64)
        for x in range(n):
            roots[x] = find(x)
        return roots
    return run


def label_populations(dirs: torch.Tensor, mass: torch.Tensor, *, max_angle_deg: float = 20.0, min_fixels: int = 8,
                      connectivity: int = 26, conflict_tol: float = 0.02, conflict_min: int = 2):
    """Population identities for fixels ``dirs (K, 3, Z, Y, X)`` with ``mass (K, Z, Y, X)`` (zero = absent). Candidate
    links: fixel ``a`` at ``p`` and ``b`` at a neighbour ``p + o`` when each is the other's best-aligned fixel there and
    they are within ``max_angle_deg``. Links are taken best-aligned first (Kruskal), and two populations merge only if
    the merged one would hold two fixels of one position at more than ``max(conflict_min, conflict_tol x the smaller's
    fixels)`` positions: a population follows its bends, but chains of in-between directions (unresolved crossing
    positions, noise) cannot join two populations that cross, while a few noisy fixels cannot keep one population in
    pieces. Where a population still holds two fixels of a position the weaker loses the label. Populations are
    numbered 1, 2, ... by size; fewer than ``min_fixels`` fixels get 0. Returns ``labels (K, Z, Y, X)`` (int64, 0 =
    none) and a table (per label: fixels, mass, mean direction (z, y, x))."""
    K = dirs.shape[0]; shp = tuple(mass.shape[1:]); V = int(np.prod(shp)); dev = mass.device
    cos_max = math.cos(math.radians(max_angle_deg))
    idx = torch.arange(V, device=dev).reshape(shp)
    src, dst, cost = [], [], []
    for o in _half_offsets(connectivity):
        a_sl = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(o, shp))
        b_sl = tuple(slice(sa.start + d, sa.stop + d) for sa, d in zip(a_sl, o))
        A = dirs[(slice(None), slice(None)) + a_sl]; B = dirs[(slice(None), slice(None)) + b_sl]
        ma = mass[(slice(None),) + a_sl] > 0; mb = mass[(slice(None),) + b_sl] > 0
        C = torch.einsum("ic...,jc...->ij...", A, B).abs()                                         # (K, K, ...)
        C = torch.where(ma[:, None] & mb[None, :], C, torch.full_like(C, -1.0))
        best_b = C.argmax(1); best_a = C.argmax(0)                                                  # (K, ...): for each a its b; for each b its a
        ia = idx[a_sl]; ib = idx[b_sl]
        for k1 in range(K):
            k2 = best_b[k1]
            c = C[k1].gather(0, k2[None])[0]
            mutual = best_a.gather(0, k2[None])[0] == k1
            ok = mutual & (c >= cos_max)
            if ok.any():
                src.append((k1 * V + ia[ok]).cpu()); dst.append((k2[ok] * V + ib[ok]).cpu()); cost.append((1 - c[ok]).cpu())
    nodes = K * V
    if src:
        s_ = torch.cat(src).numpy().astype(np.int64); d_ = torch.cat(dst).numpy().astype(np.int64); c_ = torch.cat(cost).numpy()
    else:
        s_ = d_ = np.zeros(0, np.int64); c_ = np.zeros(0, np.float32)
    present = (mass.reshape(-1) > 0).cpu().numpy()
    pres = present.reshape(K, V)
    pa, pb = [], []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            both = np.nonzero(pres[k1] & pres[k2])[0]
            pa.append(k1 * V + both); pb.append(k2 * V + both)
    pa = np.concatenate(pa).astype(np.int64) if pa else np.zeros(0, np.int64)
    pb = np.concatenate(pb).astype(np.int64) if pb else np.zeros(0, np.int64)
    comp = _constrained_components(nodes, s_, d_, np.argsort(c_, kind="stable").astype(np.int64), pa, pb, conflict_tol, conflict_min)
    if len(pa):                                                                                   # one fixel per position and population
        clash = comp[pa] == comp[pb]
        wm = mass.reshape(-1).cpu().numpy()
        weaker = np.where(wm[pa[clash]] >= wm[pb[clash]], pb[clash], pa[clash])
        present = present.copy(); present[weaker] = False
    comp = np.where(present, comp, -1)
    ids, counts = np.unique(comp[present], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remap = np.zeros(comp.max() + 2 if comp.size else 1, np.int64)
    n_lab = 0
    for j in order:
        if counts[j] < min_fixels:
            break
        n_lab += 1; remap[ids[j] + 1] = n_lab
    lab = remap[comp + 1]
    labels = torch.as_tensor(lab, device=dev).reshape(K, *shp)
    D = dirs.movedim(1, -1).reshape(-1, 3)
    w = mass.reshape(-1); L = labels.reshape(-1)
    table = []
    if n_lab:
        M = torch.zeros(n_lab + 1, 3, 3, device=dev)
        M.index_add_(0, L, w[:, None, None] * D[:, :, None] * D[:, None, :])
        cnt = torch.bincount(L, minlength=n_lab + 1); msum = torch.zeros(n_lab + 1, device=dev).index_add_(0, L, w)
        ev = torch.linalg.eigh(M[1:].double())[1][:, :, -1].float()
        for j in range(n_lab):
            table.append({"label": j + 1, "fixels": int(cnt[j + 1]), "mass": round(float(msum[j + 1]), 2),
                          "direction": [round(float(x), 3) for x in ev[j]]})
    return labels, table
