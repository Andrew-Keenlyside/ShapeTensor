"""Object scenes: the generator built around objects instead of layers that follow a field.

A scene is a set of **objects**, each with a skeleton (a curve, a tree of curves, a surface, a closed shell), attributes
that vary along it, an identity and a parent. One renderer turns them into the volume fractions, shape tensors,
descriptors (in µm) and object identities of every voxel. Everything is batched on the GPU; scalar parameters are
drawn on the CPU (numpy) so a scene never synchronises the device per random number.

What it adds over :mod:`~shapetensor.learn.generator` (the overnight review of 23 Oct):

1. **variation along an object**: radius, flattening (a round tube collapsing to a ribbon and a band, its perimeter
   kept), wall thickness, lumen (open, narrowing, closed), section shape (superellipse), contrast, beading and
   striation are smooth profiles along each curve, with discrete events (stenoses, collapses);
2. **topology**: trees grown by Murray's law (radius split r^3 = r1^3 + r2^3, the optimal branch angles), tapered
   ends, capillary-like networks (k-nearest-neighbour graphs), branch-point and endpoint maps;
3. **identity and hierarchy**: an id per object (a fibre, a vessel tree, a cell, a sheet), a segment id and a parent
   (a nucleus's cell, a fibre's bundle); per voxel the two heaviest objects and their fractions;
4. **arrangement**: fibres packed in bundles without overlap (a jittered hexagonal section), twist, fanning, dispersion;
   crossing types (angled, kissing, fanning); fibres deflected around cells and vessels;
5. **shape families**: helices, lamellar tubes (concentric walls), beaded and striated fibres, folded and fenestrated
   sheets (bilayers), foams (Voronoi walls), porous walls (a level set), cells with nuclei (nested), crystals (the old
   solids);
6. **scale**: a voxel size per scene (µm) with physical size priors per family -- what is resolvable and what is not
   follows from it (unresolved fibres are drawn as their bundle's partial volume); large canvases cropped for training;
7. **layout**: laminae -- a folded depth coordinate, cell density and fibre direction (radial / tangential) by depth;
8. **acquisition**: histology tears and folds, local-tomography truncation, a second contrast channel, on top of
   :func:`~shapetensor.learn.generator.nuisance` (run with CPU-drawn scalars and a capped high-pass scale);
9. **targets**: the same layer targets (one layer per population: a bundle, a tree, a sheet), plus identities,
   descriptors in µm, branch / endpoint maps and the orientation-recoverability map as a confidence target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from ..filters import gaussian
from .generator import (Layer, Sample, _composite, background, eye6, layer_density, line_tensor, nuisance, render_layer,
                        sheet_tensor, smooth_noise)

H = 0.5                                    # sampling step of the renderer (voxels): 8 points per voxel volume


# ------------------------------------------------------------------------------------------------------------------
# random numbers: scalars on the CPU, arrays on the device
# ------------------------------------------------------------------------------------------------------------------

class Rng:
    """Scalars from numpy (no device synchronisation); ``g`` a torch generator on the device for arrays."""

    def __init__(self, seed: int, device):
        self.np = np.random.default_rng(seed)
        self.g = torch.Generator(device=device).manual_seed(int(seed) + 104729)
        self.device = device

    def u(self, a, b):
        return float(self.np.uniform(a, b))

    def lu(self, a, b):
        return float(math.exp(self.np.uniform(math.log(a), math.log(b))))

    def i(self, a, b):                                                     # integer in [a, b]
        return int(self.np.integers(a, b + 1))

    def p(self, q):
        return bool(self.np.random() < q)

    def unit(self, n=None):
        v = self.np.normal(size=(n or 1, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return torch.tensor(v if n else v[0], dtype=torch.float32, device=self.device)

    def randn(self, *shape):
        return torch.randn(shape, generator=self.g, device=self.device)

    def rand(self, *shape):
        return torch.rand(shape, generator=self.g, device=self.device)


def smooth1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian smoothing along the last axis of ``x (N, S)`` or ``(N, S, C)`` (replicate ends), unit variance kept."""
    if sigma <= 0.3:
        return x
    r = int(3 * sigma + 0.5); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) / sigma) ** 2)
    k = k / k.norm()                                                       # white noise in, unit variance out
    sq = x.dim() == 2
    y = x[:, None] if sq else x.movedim(-1, 1).reshape(-1, 1, x.shape[1])
    y = Fn.conv1d(Fn.pad(y, (r, r), mode="replicate"), k.view(1, 1, -1))
    return y[:, 0] if sq else y.view(x.shape[0], x.shape[2], -1).movedim(1, -1)


# ------------------------------------------------------------------------------------------------------------------
# curves: paths, frames, attribute profiles
# ------------------------------------------------------------------------------------------------------------------

def smooth_paths(start: torch.Tensor, d0: torch.Tensor, length: torch.Tensor, S: int, curv: torch.Tensor, corr: float,
                 rng: Rng) -> torch.Tensor:
    """``N`` paths of ``S`` points from ``start (N, 3)`` along ``d0 (N, 3)``, each ``length`` voxels: the direction drifts
    by a smoothed random walk (``curv``: turning per voxel, rad; ``corr``: its correlation length, voxels)."""
    ds = (length / max(S - 1, 1))[:, None, None]
    w = smooth1d(rng.randn(start.shape[0], S, 3), corr / max(float(ds.mean()), 1e-3))
    d = Fn.normalize(d0[:, None] + torch.cumsum(w * curv[:, None, None] * ds, 1), dim=-1)
    return start[:, None] + torch.cumsum(d * ds, 1) - d[:, :1] * ds


def tangents(P: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(P)
    d[:, 1:-1] = P[:, 2:] - P[:, :-2]; d[:, 0] = P[:, 1] - P[:, 0]; d[:, -1] = P[:, -1] - P[:, -2]
    return Fn.normalize(d, dim=-1)


def rmf(P: torch.Tensor, T: torch.Tensor, stride: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation-minimising frames along curves (double reflection, Wang et al. 2008): ``e1, e2 (N, S, 3)``. Computed on
    every ``stride``-th sample (the frame turns slowly at the renderer's sampling) and interpolated, re-orthogonalised
    against the full tangents."""
    N, S, _ = P.shape
    if stride > 1 and S > 4 * stride:
        idx = torch.arange(0, S, stride, device=P.device)
        if int(idx[-1]) != S - 1:
            idx = torch.cat([idx, torch.tensor([S - 1], device=P.device)])
        e1c, _ = rmf(P[:, idx], tangents(P[:, idx]), 1)
        pos = torch.arange(S, device=P.device, dtype=torch.float32)
        j = torch.searchsorted(idx.float(), pos, right=True).clamp(1, idx.numel() - 1)
        w = ((pos - idx[j - 1].float()) / (idx[j] - idx[j - 1]).float()).view(1, S, 1)
        e1 = (1 - w) * e1c[:, j - 1] + w * e1c[:, j]
        e1 = Fn.normalize(e1 - (e1 * T).sum(-1, keepdim=True) * T, dim=-1)
        return e1, torch.linalg.cross(T, e1, dim=-1)
    ref = torch.where(T[:, :1, :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=P.device), torch.tensor([0, 1.0, 0], device=P.device)).expand(N, 1, 3)[:, 0]
    r = Fn.normalize(ref - (ref * T[:, 0]).sum(-1, keepdim=True) * T[:, 0], dim=-1)
    R = [r]
    for i in range(S - 1):
        v1 = P[:, i + 1] - P[:, i]; c1 = (v1 * v1).sum(-1, keepdim=True).clamp_min(1e-12)
        rL = r - (2 / c1) * (v1 * r).sum(-1, keepdim=True) * v1
        tL = T[:, i] - (2 / c1) * (v1 * T[:, i]).sum(-1, keepdim=True) * v1
        v2 = T[:, i + 1] - tL; c2 = (v2 * v2).sum(-1, keepdim=True)
        r = torch.where(c2 > 1e-12, rL - (2 / c2.clamp_min(1e-12)) * (v2 * rL).sum(-1, keepdim=True) * v2, rL)
        r = Fn.normalize(r - (r * T[:, i + 1]).sum(-1, keepdim=True) * T[:, i + 1], dim=-1)
        R.append(r)
    e1 = torch.stack(R, 1)
    return e1, torch.linalg.cross(T, e1, dim=-1)


def profile(N: int, S: int, base, rel_sd: float, corr_samples: float, rng: Rng) -> torch.Tensor:
    """A smooth positive profile ``(N, S)``: ``base * exp(rel_sd * smooth noise)``."""
    base = base if torch.is_tensor(base) else torch.full((N,), float(base), device=rng.device)
    return base[:, None] * torch.exp(rel_sd * smooth1d(rng.randn(N, S), corr_samples))


def events(N: int, S: int, rate: float, depth: tuple, width: tuple, rng: Rng) -> torch.Tensor:
    """A multiplicative profile ``(N, S)`` in (0, 1] with Gaussian dips (stenoses, collapses): ``rate`` dips per curve on
    average, depth and width (samples) drawn uniformly in the given ranges."""
    Kmax = max(1, int(3 * rate + 2))
    on = (rng.rand(N, Kmax) < rate / Kmax).float()
    s0 = rng.rand(N, Kmax) * S; w = width[0] + (width[1] - width[0]) * rng.rand(N, Kmax)
    dep = (depth[0] + (depth[1] - depth[0]) * rng.rand(N, Kmax)) * on
    s = torch.arange(S, device=rng.device, dtype=torch.float32).view(1, S, 1)
    return torch.prod(1 - dep[:, None] * torch.exp(-0.5 * ((s - s0[:, None]) / w[:, None]) ** 2), dim=-1)


@dataclass
class Curves:
    """A batch of ``N`` curves of ``S`` samples (voxel coordinates) with profiles: ``rad`` the section's nominal radius
    (voxels; the perimeter of the flattened section is kept), ``flat`` its flattening (minor / major, 1 round), ``wall``
    the wall thickness (voxels; 0 solid), ``sexp`` the superellipse exponent (2 elliptic), ``contrast`` and ``lumen_c``
    (grey levels: the wall's and the lumen contents'), ``lam`` concentric walls per curve, ``phi`` the section's rotation
    about the axis, ``density`` the volume fraction of the section that is material (an unresolved bundle < 1), ids."""
    P: torch.Tensor
    rad: torch.Tensor
    flat: torch.Tensor
    wall: torch.Tensor
    sexp: torch.Tensor
    contrast: torch.Tensor
    lumen_c: torch.Tensor
    obj: torch.Tensor
    seg: torch.Tensor
    lam: torch.Tensor | None = None
    phi: torch.Tensor | None = None
    density: torch.Tensor | None = None
    fibre_r: torch.Tensor | None = None    # descriptor: the constituents' radius (voxels) when the curve is an unresolved bundle

    @property
    def N(self):
        return self.P.shape[0]


def curves_like(P, rad, rng: Rng, obj, seg, contrast, **kw) -> Curves:
    N, S = P.shape[:2]; dev = P.device
    z = lambda v: torch.full((N, S), float(v), device=dev)
    return Curves(P=P, rad=rad, flat=kw.pop("flat", z(1.0)), wall=kw.pop("wall", z(0.0)), sexp=kw.pop("sexp", z(2.0)),
                  contrast=contrast if torch.is_tensor(contrast) and contrast.dim() == 2 else z(contrast),
                  lumen_c=kw.pop("lumen_c", z(0.0)), obj=obj, seg=seg, **kw)


# ------------------------------------------------------------------------------------------------------------------
# rendering: points with weights splatted (trilinear) onto the grid, several channels at once
# ------------------------------------------------------------------------------------------------------------------

class Canvas:
    """Accumulates one population (layer) at a time: volume fraction, shape tensor, descriptors, contrast; and, across
    the whole scene, the (voxel, object) weights for the identity maps."""

    NDESC = 5                              # outer radius, wall, lumen fraction, flattening, angularity (voxels / 0-1)

    def __init__(self, shape, device):
        self.shape = tuple(shape); self.device = device; self.n = int(np.prod(shape))
        self.id_keys, self.id_w = [], []
        self.lumen = torch.zeros(self.n, device=device); self.lumen_c = torch.zeros(self.n, device=device)
        self.reset()

    def reset(self):
        self.occ = torch.zeros(self.n, device=self.device)
        self.acc = torch.zeros(self.n, 6 + self.NDESC + 1, device=self.device)   # tensor, descriptors, contrast (all weighted)

    def splat(self, Q, W, V, obj=None):
        """Points ``Q (n, 3)`` with weights ``W (n,)`` (volumes, voxel units) carrying ``V (n, 12)`` (weighted later)."""
        Z, Y, X = self.shape
        p0 = Q.floor(); fr = Q - p0; i0 = p0.long()
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    iz, iy, ix = i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx
                    w = W * (fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dx else 1 - fr[:, 2])
                    ok = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X) & (w > 0)
                    lin = ((iz * Y + iy) * X + ix)[ok]; ww = w[ok]
                    self.occ.index_add_(0, lin, ww)
                    self.acc.index_add_(0, lin, ww[:, None] * V[ok])
        if obj is not None:                                                  # nearest voxel, for the identities
            n = Q.round().long()
            ok = (n[:, 0] >= 0) & (n[:, 0] < Z) & (n[:, 1] >= 0) & (n[:, 1] < Y) & (n[:, 2] >= 0) & (n[:, 2] < X)
            lin = (n[ok, 0] * Y + n[ok, 1]) * X + n[ok, 2]
            self.id_keys.append(lin * 65536 + obj[ok]); self.id_w.append(W[ok])

    def splat_lumen(self, Q, W, c):
        Z, Y, X = self.shape
        n = Q.round().long()
        ok = (n[:, 0] >= 0) & (n[:, 0] < Z) & (n[:, 1] >= 0) & (n[:, 1] < Y) & (n[:, 2] >= 0) & (n[:, 2] < X)
        lin = (n[ok, 0] * Y + n[ok, 1]) * X + n[ok, 2]
        self.lumen.index_add_(0, lin, W[ok]); self.lumen_c.index_add_(0, lin, W[ok] * c[ok])

    def add_dense(self, occ, tensor, desc, contrast, obj_map=None):
        """A population given on the grid (implicit surfaces): ``occ (Z, Y, X)``, ``tensor (6, ...)``, ``desc (5, ...)``."""
        o = occ.reshape(-1)
        self.occ += o
        self.acc[:, :6] += o[:, None] * tensor.reshape(6, -1).T
        self.acc[:, 6:6 + self.NDESC] += o[:, None] * desc.reshape(self.NDESC, -1).T
        self.acc[:, -1] += o * (contrast.reshape(-1) if torch.is_tensor(contrast) else float(contrast))
        if obj_map is not None:
            m = o > 1e-3; lin = torch.nonzero(m)[:, 0]
            self.id_keys.append(lin * 65536 + obj_map.reshape(-1)[m]); self.id_w.append(o[m])

    def finish(self):
        """This population's maps: occupancy (clamped), tensor (trace 1), descriptors, mean contrast map."""
        occ = self.occ.clamp_min(0); w = occ.clamp_min(1e-9)[:, None]
        T = self.acc[:, :6] / w; D = self.acc[:, 6:6 + self.NDESC] / w; C = self.acc[:, -1] / w[:, 0]
        empty = occ < 1e-6
        T = torch.where(empty[:, None], eye6(self.device).view(1, 6) / 3, T)
        tr = T[:, :3].sum(1, keepdim=True).clamp_min(1e-9); T = T / tr
        sh = self.shape
        out = {"occupancy": occ.clamp(0, 1).view(sh), "tensor": T.T.reshape(6, *sh), "desc": (D * (~empty)[:, None]).T.reshape(self.NDESC, *sh),
               "contrast": C.view(sh)}
        self.reset()
        return out

    def identities(self, K: int = 2):
        """Per voxel the ``K`` heaviest objects and their share of the voxel: ``ids (K, Z, Y, X)`` (-1 none), ``frac``."""
        ids = torch.full((K, self.n), -1, dtype=torch.long, device=self.device); frac = torch.zeros(K, self.n, device=self.device)
        if self.id_keys:
            keys = torch.cat(self.id_keys); w = torch.cat(self.id_w)
            uk, inv = torch.unique(keys, return_inverse=True)
            ws = torch.zeros(uk.shape[0], device=self.device).index_add_(0, inv, w)
            vox, obj = uk // 65536, uk % 65536
            # sort by voxel, then by weight descending
            o1 = torch.argsort(-ws); vox1, obj1, ws1 = vox[o1], obj[o1], ws[o1]
            o2 = torch.argsort(vox1, stable=True); vox2, obj2, ws2 = vox1[o2], obj1[o2], ws1[o2]
            _, counts = torch.unique_consecutive(vox2, return_counts=True)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(vox2.shape[0], device=self.device) - torch.repeat_interleave(starts, counts)
            tot = torch.zeros(self.n, device=self.device).index_add_(0, vox2, ws2).clamp_min(1e-9)
            for k in range(K):
                m = rank == k
                ids[k, vox2[m]] = obj2[m]; frac[k, vox2[m]] = ws2[m] / tot[vox2[m]]
        return ids.view(K, *self.shape), frac.view(K, *self.shape)


def _template2d(n_side: int, device) -> torch.Tensor:
    """A centred grid on [-1, 1]^2 with ``n_side`` points per side: ``(m, 2)`` and its spacing."""
    v = (torch.arange(n_side, device=device, dtype=torch.float32) + 0.5) / n_side * 2 - 1
    u, w = torch.meshgrid(v, v, indexing="ij")
    return torch.stack([u.reshape(-1), w.reshape(-1)], 1), 2.0 / n_side


def render_curves(cv: Canvas, c: Curves, max_points: float = 2.5e7):
    """Sweep each curve's cross-section along it and splat the material (and the lumen contents)."""
    P = c.P; N, S, _ = P.shape; dev = P.device
    T = tangents(P); e1, e2 = rmf(P, T)
    ds = torch.zeros(N, S, device=dev)
    seglen = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    ds[:, 1:] += 0.5 * seglen; ds[:, :-1] += 0.5 * seglen
    f = c.flat.clamp(0.02, 1.0); a = 2 * c.rad / (1 + f); b = f * a              # perimeter-preserving flattening
    t = c.wall
    lam = c.lam if c.lam is not None else torch.ones(N, device=dev, dtype=torch.long)
    phi = c.phi if c.phi is not None else torch.zeros(N, S, device=dev)
    dens = c.density if c.density is not None else torch.ones(N, device=dev)
    open_ = (t > 0) & (b - t > 0.35)                                            # a lumen narrower than ~0.7 voxel has closed
    lum_frac = torch.where(open_, ((a - t) * (b - t)).clamp_min(0) / (a * b).clamp_min(1e-9), torch.zeros_like(a))
    ang = ((c.sexp - 2) / (c.sexp + 2)).clamp(0, 1)
    fr = c.fibre_r if c.fibre_r is not None else None
    amax = a.amax(1)
    edges = [0, 1.5, 4, 10, 25, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = torch.nonzero((amax > lo) & (amax <= hi))[:, 0]
        if sel.numel() == 0:
            continue
        am = float(amax[sel].max())
        n_side = max(4, int(math.ceil(2 * am / H)))
        U, du = _template2d(n_side, dev)                                        # (m, 2)
        m = U.shape[0]
        per = max(1, int(max_points // m))
        pairs = torch.stack(torch.meshgrid(sel, torch.arange(S, device=dev), indexing="ij"), -1).reshape(-1, 2)
        keep_pair = c.rad[pairs[:, 0], pairs[:, 1]] > 1e-3
        pairs = pairs[keep_pair]
        for s0 in range(0, pairs.shape[0], per):
            pr = pairs[s0:s0 + per]; ci, si = pr[:, 0], pr[:, 1]
            A_, B_, T_, n_ = a[ci, si][:, None], b[ci, si][:, None], t[ci, si][:, None], c.sexp[ci, si][:, None]
            x = U[None, :, 0] * A_; y = U[None, :, 1] * B_
            rho_n = U[None, :, 0].abs() ** n_ + U[None, :, 1].abs() ** n_
            inside = rho_n <= 1
            op = open_[ci, si][:, None]
            inner = (x.abs() / (A_ - T_).clamp_min(1e-3)) ** n_ + (y.abs() / (B_ - T_).clamp_min(1e-3)) ** n_ < 1
            L_ = lam[ci][:, None]
            depth = (1 - rho_n.clamp_min(0) ** (1 / n_)) * 0.5 * (A_ + B_)       # distance in from the outer surface (approx.)
            band = torch.floor(depth / T_.clamp_min(1e-3))
            multi = (L_ > 1) & (T_ > 0)
            wall_pt = torch.where(multi, inside & (band.remainder(2) == 0) & (band < 2 * L_ - 1), inside & ~(op & inner))
            lumen_pt = inside & op & inner & ~multi
            w_area = (A_ * B_ * du * du * ds[ci, si][:, None])                    # volume of each template point
            ph = phi[ci, si][:, None]
            xr = x * torch.cos(ph) - y * torch.sin(ph); yr = x * torch.sin(ph) + y * torch.cos(ph)
            Q = P[ci, si][:, None] + xr[..., None] * e1[ci, si][:, None] + yr[..., None] * e2[ci, si][:, None]   # (p, m, 3)
            tt = line_tensor(T[ci, si])                                         # (p, 6)
            size = 0.5 * (a[ci, si] + b[ci, si]) if fr is None else fr[ci]          # the constituents' radius for an unresolved bundle
            desc = torch.stack([size, t[ci, si], lum_frac[ci, si], (b / a.clamp_min(1e-9))[ci, si], ang[ci, si]], 1)   # (p, 5)
            V = torch.cat([tt, desc, c.contrast[ci, si][:, None]], 1)            # (p, 12)
            idx = torch.nonzero(wall_pt)
            if idx.numel():
                pi, mi = idx[:, 0], idx[:, 1]
                cv.splat(Q[pi, mi], (w_area[:, 0] * dens[ci])[pi], V[pi], obj=c.obj[ci][pi])
            lidx = torch.nonzero(lumen_pt & (c.lumen_c[ci, si][:, None] != 0))
            if lidx.numel():
                pi, mi = lidx[:, 0], lidx[:, 1]
                cv.splat_lumen(Q[pi, mi], w_area[pi, 0], c.lumen_c[ci, si][pi])


# ------------------------------------------------------------------------------------------------------------------
# points-based families (sheets, cells) and implicit ones (foams, porous walls)
# ------------------------------------------------------------------------------------------------------------------

def noise2d(n_u: int, n_v: int, sigma: float, rng: Rng) -> torch.Tensor:
    """Smooth 2-D noise of unit standard deviation (``sigma`` in samples)."""
    x = rng.randn(1, n_u, n_v)
    x = gaussian(x, (0.0, sigma, sigma), truncate=3.0) if sigma > 0.3 else x
    return (x[0] - x.mean()) / x.std().clamp_min(1e-6)


def _fib_sphere(n: int, device) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n; r = torch.sqrt((1 - z * z).clamp_min(0)); ph = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([z, r * torch.sin(ph), r * torch.cos(ph)], 1)


def _rot(rng: Rng, n: int) -> torch.Tensor:
    """``n`` random rotation matrices ``(n, 3, 3)``."""
    q = torch.tensor(rng.np.normal(size=(n, 4)), dtype=torch.float32, device=rng.device); q = Fn.normalize(q, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _sym6(M: torch.Tensor) -> torch.Tensor:
    return torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)


@dataclass
class Group:
    """One population (one layer of the targets): a bundle, a tree, a sheet, a foam, the cells' membranes..."""
    name: str
    kind: str                              # line | sheet | blob | solid (the targets' kind; descriptors carry the rest)
    radius: float                          # a typical size (voxels), for the visibility statistic
    contrast: float
    curves: list = field(default_factory=list)
    points: list = field(default_factory=list)       # (Q, W, V (n, 12), obj)
    dense: list = field(default_factory=list)        # (occ, tensor, desc, contrast, obj_map)
    layer: Layer | None = None                        # an old-style solid layer


class Scene:
    """Draws the objects of one scene in physical units on a ``shape`` canvas (see the module docstring)."""

    FAMILIES = {"bundle": 0.30, "tree": 0.14, "network": 0.06, "helix": 0.06, "sheet": 0.12, "foam": 0.05, "porous": 0.04,
                "cells": 0.13, "crystals": 0.05, "radial": 0.05}

    def __init__(self, shape, seed: int, device="cuda", voxel_um: float | None = None, content: str = "general"):
        self.shape = tuple(shape); self.device = device; self.r = Rng(seed, device); self.seed = seed
        self.voxel_um = float(voxel_um) if voxel_um else self.r.lu(1.0, 60.0)
        self.content = content
        self.groups: list[Group] = []; self.next_obj = 1
        self.branch, self.ends = [], []                    # (n, 3) positions
        self.obstacles = []                                # (centres (n, 3), radii (n,)) that fibres yield to
        self.parents = {}                                  # object id -> parent id (nuclei in cells, fibres in bundles)
        self.depth = None                                  # the laminar depth coordinate (voxels), if a layout was drawn
        self.record = {"voxel_um": round(self.voxel_um, 2), "families": []}

    # ---- helpers
    def ids(self, n: int) -> torch.Tensor:
        out = torch.arange(self.next_obj, self.next_obj + n, device=self.device); self.next_obj += n
        return out

    def size(self, lo_um, hi_um, lo_vox=0.45, hi_vox=None) -> float | None:
        """A size drawn log-uniformly in the physical range intersected with what the canvas can render (voxels)."""
        hi_vox = hi_vox or min(self.shape) / 4
        lo, hi = max(lo_um / self.voxel_um, lo_vox), min(hi_um / self.voxel_um, hi_vox)
        return self.r.lu(lo, hi) if hi > lo * 1.05 else None

    def centre_point(self, margin=0.25) -> torch.Tensor:
        return torch.tensor([self.r.u(margin * n, (1 - margin) * n) for n in self.shape], device=self.device)

    def contrast(self, lo=15, hi=140) -> float:
        return self.r.u(lo, hi) * (1 if self.r.p(0.5) else -1)

    # ---- layout: laminae along a folded depth coordinate
    def layout(self):
        r = self.r; Z, Y, X = self.shape
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        g = torch.stack(torch.meshgrid(*[torch.arange(n, device=self.device, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        c = torch.tensor([Z / 2, Y / 2, X / 2], device=self.device)
        rel = g - c; u = rel @ e1; v = rel @ e2
        L = int(2 * max(self.shape)); hgt = noise2d(L, L, r.lu(10, 40), r) * r.u(0, 12)
        gu = (u / (L / 2)).clamp(-1, 1); gv = (v / (L / 2)).clamp(-1, 1)
        hs = Fn.grid_sample(hgt[None, None], torch.stack([gv, gu], -1).view(1, 1, -1, 2), align_corners=True).view(self.shape)
        self.depth = rel @ n0 - hs
        K = r.i(2, 5); span = 0.5 * max(self.shape)
        self.bands = sorted([r.u(-span, span) for _ in range(K)])
        self.band_density = [r.u(0.05, 1.0) for _ in range(K + 1)]
        gz, gy, gx = torch.gradient(self.depth)
        self.normal_field = Fn.normalize(torch.stack([gz, gy, gx]), dim=0)
        self.record["layout"] = {"bands": len(self.bands)}

    def band_of(self, P: torch.Tensor) -> torch.Tensor:
        d = self.sample_grid(self.depth[None], P)[0]
        return torch.bucketize(d, torch.tensor(self.bands, device=self.device))

    def sample_grid(self, F: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Trilinear samples of ``F (C, Z, Y, X)`` at voxel points ``P (..., 3)``: ``(C, ...)``."""
        sz = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
        gn = (2 * P / (sz - 1) - 1).flip(-1).reshape(1, 1, 1, -1, 3)
        out = Fn.grid_sample(F[None].float(), gn, align_corners=True, padding_mode="border")[0, :, 0, 0]
        return out.view(F.shape[0], *P.shape[:-1])

    # ---- curves: bundles, trees, networks, helices
    def _fibre_extras(self, N, S, rad, rng):
        """Beading, striation, hollowness (myelin-like rings), lamellae for a set of fibres."""
        kw = {}
        if rng.p(0.2):                                                      # beading: periodic swellings
            lam = rng.lu(4, 20) * float(rad.mean()); s = torch.arange(S, device=self.device, dtype=torch.float32) * H
            ph = rng.rand(N, 1) * 6.28
            rad = rad * (1 + rng.u(0.2, 0.8) * torch.relu(torch.sin(2 * math.pi * s[None] / lam + ph)) ** 2)
        con = None
        if rng.p(0.15):                                                     # striation: periodic contrast along the axis
            lam = rng.lu(2, 12); s = torch.arange(S, device=self.device, dtype=torch.float32) * H
            con = 1 + rng.u(0.2, 0.6) * torch.sin(2 * math.pi * s[None] / lam + rng.rand(N, 1) * 6.28)
        if rng.p(0.18) and float(rad.mean()) >= 1.2:                        # hollow fibres
            kw["wall"] = rad * rng.u(0.2, 0.45); kw["lumen_c"] = torch.full_like(rad, rng.u(-40, 40))
            if rng.p(0.3):
                kw["lam"] = torch.full((N,), rng.i(2, 4), device=self.device, dtype=torch.long); kw["wall"] = rad * rng.u(0.08, 0.15)
        if rng.p(0.15):                                                     # angular (square-ish or diamond-ish) sections
            kw["sexp"] = torch.full_like(rad, rng.lu(0.8, 6.0))
        if rng.p(0.15):                                                     # ribbons
            kw["flat"] = torch.full_like(rad, rng.lu(0.25, 0.8)); kw["phi"] = smooth1d(rng.randn(N, S), 20) * 0.3
        return rad, con, kw

    def bundle(self, cross_with: Curves | None = None, mode: str = "angled"):
        r = self.r; S0 = int(2.2 * max(self.shape) / H)
        Rb = self.size(20, 3000, 1.5, min(self.shape) / 5)
        rf_um = r.lu(0.25, 12.0); rf = rf_um / self.voxel_um
        if Rb is None:
            return None
        Lc = 2.0 * max(self.shape)
        if cross_with is not None:                                          # through a point of an existing bundle
            k = r.i(S0 // 4, 3 * S0 // 4); P0 = cross_with.P[0, min(k, cross_with.P.shape[1] - 1)]
            t0 = tangents(cross_with.P[:1])[0, min(k, cross_with.P.shape[1] - 1)]
            p = Fn.normalize(torch.linalg.cross(t0, r.unit()), dim=0)
            if mode == "kissing":
                ang = math.radians(r.u(5, 20)); d = math.cos(ang) * t0 + math.sin(ang) * p
                nrm = Fn.normalize(torch.linalg.cross(d, torch.linalg.cross(t0, d)), dim=0)
                P0 = P0 + nrm * (float(cross_with.rad[0].mean()) + Rb)
            else:
                ang = math.radians(r.u(20, 90)); d = math.cos(ang) * t0 + math.sin(ang) * p
            c = P0
        else:
            c = self.centre_point(); d = r.unit()
        start = (c - d * Lc / 2)[None]; S = S0
        Pc = smooth_paths(start, d[None], torch.tensor([Lc], device=self.device), S, torch.tensor([r.lu(0.001, 0.02)], device=self.device), r.lu(15, 80), r)
        s = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
        fan = r.u(0.3, 1.5) if (mode == "fanning" or r.p(0.15)) else 0.0
        Rs = Rb * (1 + fan * torch.relu(s - 0.5) * 2)                       # fanning: the bundle widens past its middle
        contrast = self.contrast(); bid = int(self.ids(1)[0])
        if rf >= 0.45:                                                      # resolved fibres, packed on a jittered hexagonal section
            sp = max(2 * rf * r.u(1.05, 1.6), 1.0) / Rb
            n_side = int(2 / sp) + 2
            ii, jj = torch.meshgrid(torch.arange(-n_side, n_side + 1, device=self.device), torch.arange(-n_side, n_side + 1, device=self.device), indexing="ij")
            o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp
            o = o + 0.15 * sp * (r.rand(*o.shape) - 0.5) * 2               # jitter, kept below the gap: no overlaps
            o = o[o.norm(dim=1) <= 1 - rf / Rb]
            if o.shape[0] > 400:
                o = o[torch.randperm(o.shape[0], generator=r.g, device=self.device)[:400]]
            if o.shape[0] == 0:
                o = torch.zeros(1, 2, device=self.device)
            N = o.shape[0]
            T = tangents(Pc); e1, e2 = rmf(Pc, T)
            tau = r.u(-0.03, 0.03); psi = tau * s * Lc
            disp = 0.25 * sp * smooth1d(r.randn(N, S, 2), 30)                # dispersion within half a spacing: no overlaps
            oo = o[:, None] + disp
            ox = oo[..., 0] * torch.cos(psi) - oo[..., 1] * torch.sin(psi); oy = oo[..., 0] * torch.sin(psi) + oo[..., 1] * torch.cos(psi)
            P = Pc + (Rs[None, :, None] * (ox[..., None] * e1 + oy[..., None] * e2))
            rad = (rf * torch.exp(0.15 * r.randn(N)))[:, None].expand(N, S).clone()
            rad, con, kw = self._fibre_extras(N, S, rad, r)
            cmap = contrast * (con if con is not None else 1.0) * torch.ones(N, S, device=self.device)
            obj = self.ids(N)
            for o_ in obj.tolist():
                self.parents[o_] = bid
            cv = curves_like(P, rad, r, obj=obj, seg=torch.full((N,), bid, device=self.device), contrast=cmap, **kw)
            g = Group("bundle", "line", float(rad.median()), contrast, curves=[cv])
            self.record["families"].append({"bundle": {"fibres": N, "fibre_um": round(rf_um, 2), "radius_vox": round(Rb, 1), "fan": round(fan, 2), "mode": mode if cross_with is not None else "single"}})
        else:                                                               # unresolved fibres: the bundle's partial volume
            phi_ = r.u(0.3, 0.8)
            cv = curves_like(Pc, Rs[None].clone(), r, obj=torch.tensor([bid], device=self.device), seg=torch.tensor([bid], device=self.device),
                             contrast=contrast, density=torch.tensor([phi_], device=self.device), fibre_r=torch.tensor([rf], device=self.device))
            g = Group("bundle", "line", rf, contrast, curves=[cv])
            self.record["families"].append({"bundle": {"unresolved": True, "fibre_um": round(rf_um, 2), "packing": round(phi_, 2), "radius_vox": round(Rb, 1), "mode": mode if cross_with is not None else "single"}})
        g.centreline = curves_like(Pc, Rs[None].clone(), r, obj=torch.tensor([bid], device=self.device), seg=torch.tensor([bid], device=self.device), contrast=contrast)
        return g

    def tree(self):
        r = self.r
        r0 = self.size(15, 800, 1.5, min(self.shape) / 7)
        rmin = max(3.0 / self.voxel_um, 0.45)
        if r0 is None or r0 < rmin * 1.5:
            return None
        tid = int(self.ids(1)[0]); contrast = self.contrast(); g = Group("tree", "line", r0, contrast)
        g.murray = []
        wf = r.u(0.08, 0.3); f0 = 1.0 if r.p(0.6) else r.u(0.5, 1.0)       # arterial-like wall; veins flatter
        lumen_c = r.u(-60, 80) if r.p(0.7) else 0.0
        c = self.centre_point(0.1); d = r.unit()
        start = (c - d * r0 * 4)[None]; dirs = d[None]; rads = torch.tensor([r0], device=self.device)
        nseg = 0; segs = []
        for depth in range(10):
            A = start.shape[0]
            if A == 0 or nseg > 300:
                break
            L = torch.tensor([r.lu(5, 12) * float(x) for x in rads.tolist()], device=self.device)
            S = max(8, int(math.ceil(float(L.max()) / H)))
            P = smooth_paths(start, dirs, L, S, torch.full((A,), r.lu(0.005, 0.04), device=self.device), float(L.mean()), r)
            # children (Murray: r^3 = r1^3 + r2^3; optimal angles) where they stay above the smallest radius
            q = torch.tensor([r.u(0.15, 0.5) for _ in range(A)], device=self.device)
            r1 = rads * (1 - q) ** (1 / 3); r2 = rads * q ** (1 / 3)
            grow = r1 >= rmin
            if nseg + 2 * int(grow.sum()) > 300:
                grow = grow & (torch.rand(A, generator=r.g, device=self.device) < 0.5)
            prof = rads[:, None].expand(A, S).clone()
            sfrac = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
            tip = torch.where(grow, r1, rads * 0.5)                          # taper the last quarter to the larger child (or a tip)
            ramp = ((sfrac - 0.75) / 0.25).clamp(0, 1)
            prof = prof * (1 - ramp) + tip[:, None] * ramp
            prof = prof * events(A, S, 0.3, (0.2, 0.6), (3, 12), r)           # stenoses
            flat = f0 * events(A, S, 0.35, (0.5, 0.95), (4, 15), r)          # collapses: the section flattens, the lumen closes
            seg_ids = torch.arange(nseg, nseg + A, device=self.device); nseg += A
            cvs = curves_like(P, prof, r, obj=torch.full((A,), tid, device=self.device), seg=seg_ids, contrast=contrast,
                              flat=flat, wall=prof * wf, lumen_c=torch.full((A, S), lumen_c, device=self.device))
            segs.append(cvs)
            ends = P[:, -1]; tend = tangents(P)[:, -1]
            if bool((~grow).any()):
                self.ends.append(ends[~grow])
            if not bool(grow.any()):
                break
            self.branch.append(ends[grow])
            r0g, r1g, r2g = rads[grow], r1[grow], r2[grow]
            g.murray.append(torch.stack([r0g, r1g, r2g], 1))                    # each branching's parent and child radii
            cos1 = ((r0g ** 4 + r1g ** 4 - r2g ** 4) / (2 * r0g ** 2 * r1g ** 2)).clamp(-1, 1)
            cos2 = ((r0g ** 4 + r2g ** 4 - r1g ** 4) / (2 * r0g ** 2 * r2g ** 2)).clamp(-1, 1)
            te = tend[grow]; pe = Fn.normalize(torch.linalg.cross(te, r.unit(int(grow.sum()))), dim=1)
            d1 = Fn.normalize(cos1[:, None] * te + torch.sqrt(1 - cos1 ** 2)[:, None] * pe, dim=1)
            d2 = Fn.normalize(cos2[:, None] * te - torch.sqrt(1 - cos2 ** 2)[:, None] * pe, dim=1)
            start = torch.cat([ends[grow], ends[grow]]); dirs = torch.cat([d1, d2]); rads = torch.cat([r1g, r2g])
        g.curves = segs
        self.obstacles.append((torch.cat([cv.P[:, ::4].reshape(-1, 3) for cv in segs]), torch.cat([cv.rad[:, ::4].reshape(-1) for cv in segs])))
        self.record["families"].append({"tree": {"root_vox": round(r0, 1), "segments": nseg, "wall_frac": round(wf, 2)}})
        return g

    def network(self):
        r = self.r
        rc = self.size(2.5, 6.0, 0.45, 3.0)
        if rc is None:
            return None
        n = r.i(30, 160); m = torch.tensor([0.1 * s for s in self.shape], device=self.device)
        pts = m + r.rand(n, 3) * (torch.tensor(self.shape, device=self.device) - 2 * m)
        D = torch.cdist(pts, pts); D.fill_diagonal_(1e9)
        k = 3; nb = D.topk(k, largest=False).indices
        e = torch.stack([torch.arange(n, device=self.device).repeat_interleave(k), nb.reshape(-1)], 1)
        e = torch.unique(torch.sort(e, 1).values, dim=0)
        a, b = pts[e[:, 0]], pts[e[:, 1]]; L = (b - a).norm(dim=1)
        S = max(8, int(math.ceil(float(L.max()) / H)))
        s = torch.linspace(0, 1, S, device=self.device)[None, :, None]
        wob = smooth1d(r.randn(e.shape[0], S, 3), S / 4) * (0.08 * L[:, None, None]) * torch.sin(math.pi * s)
        P = a[:, None] + (b - a)[:, None] * s + wob
        rad = (rc * torch.exp(0.15 * r.randn(e.shape[0])))[:, None].expand(-1, S).clone()
        nid = int(self.ids(1)[0]); contrast = self.contrast()
        kw = {"wall": rad * 0.35, "lumen_c": torch.full_like(rad, r.u(-40, 60))} if rc >= 2.0 else {}
        cv = curves_like(P, rad, r, obj=torch.full((e.shape[0],), nid, device=self.device), seg=torch.arange(e.shape[0], device=self.device), contrast=contrast, **kw)
        deg = torch.bincount(e.reshape(-1), minlength=n)
        if bool((deg >= 3).any()):
            self.branch.append(pts[deg >= 3])
        self.record["families"].append({"network": {"nodes": n, "edges": int(e.shape[0]), "radius_vox": round(rc, 2)}})
        return Group("network", "line", rc, contrast, curves=[cv])

    def helix(self):
        r = self.r
        rf = self.size(0.5, 30, 0.45, 4.0)
        if rf is None:
            return None
        rho = rf * r.u(1.5, 5.0); pitch = rf * r.u(4, 25); n_str = r.i(1, 3)
        c = self.centre_point(); d = r.unit(); Lc = 2.0 * max(self.shape); S = int(Lc / H)
        Pc = smooth_paths((c - d * Lc / 2)[None], d[None], torch.tensor([Lc], device=self.device), S, torch.tensor([r.lu(0.001, 0.01)], device=self.device), 40.0, r)
        T = tangents(Pc); e1, e2 = rmf(Pc, T)
        sl = torch.arange(S, device=self.device, dtype=torch.float32) * (Lc / (S - 1))
        ph = torch.arange(n_str, device=self.device, dtype=torch.float32)[:, None] * (2 * math.pi / n_str)
        ang = 2 * math.pi * sl[None] / pitch + ph
        P = Pc + rho * (torch.cos(ang)[..., None] * e1 + torch.sin(ang)[..., None] * e2)
        rad = torch.full((n_str, S), rf, device=self.device); contrast = self.contrast()
        cv = curves_like(P, rad, r, obj=self.ids(n_str), seg=torch.zeros(n_str, dtype=torch.long, device=self.device), contrast=contrast)
        self.record["families"].append({"helix": {"strands": n_str, "radius_vox": round(rf, 2), "pitch_vox": round(pitch, 1)}})
        return Group("helix", "line", rf, contrast, curves=[cv])

    def radial(self):
        """Fibres along (radial) or across (tangential) the laminar depth coordinate, confined to one band."""
        if self.depth is None:
            return None
        r = self.r; rf = self.size(0.5, 20, 0.45, 3.0)
        if rf is None:
            return None
        k = r.i(0, len(self.bands)); lo = self.bands[k - 1] if k > 0 else -1e9; hi = self.bands[k] if k < len(self.bands) else 1e9
        radial = r.p(0.6); n = r.i(20, 150)
        F = self.normal_field if radial else Fn.normalize(torch.linalg.cross(self.normal_field, r.unit().view(3, 1, 1, 1).expand_as(self.normal_field), dim=0), dim=0)
        P = (r.rand(n, 3) * torch.tensor(self.shape, device=self.device))
        S = int(1.5 * max(self.shape) / H); pts = [P]
        for _ in range(S - 1):
            d = Fn.normalize(self.sample_grid(F, P).T, dim=1)
            if len(pts) > 1:
                prev = P - pts[-2]; d = torch.where(((d * prev).sum(1, keepdim=True)) < 0, -d, d)
            P = P + H * d; pts.append(P)
        Pc = torch.stack(pts, 1)
        dep = self.sample_grid(self.depth[None], Pc)[0]
        rad = torch.full((n, S), rf, device=self.device) * ((dep > lo) & (dep < hi)).float()
        contrast = self.contrast()
        cv = curves_like(Pc, rad, r, obj=self.ids(n), seg=torch.zeros(n, dtype=torch.long, device=self.device), contrast=contrast)
        self.record["families"].append({"radial": {"radial": radial, "fibres": n, "band": k}})
        return Group("radial" if radial else "tangential", "line", rf, contrast, curves=[cv])

    # ---- surfaces and implicit walls
    def sheet(self):
        r = self.r
        ht = self.size(0.25, 150, 0.5, min(self.shape) / 10)
        if ht is None:
            return None
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        c = self.centre_point(); Lh = 0.9 * max(self.shape); nu = int(2 * Lh / H)
        sf = r.lu(8, 40); hgt = noise2d(nu, nu, sf / H, r) * (r.u(0, 1.0) * sf)
        gu, gv = torch.gradient(hgt, spacing=H)
        uu = (torch.arange(nu, device=self.device, dtype=torch.float32) - nu / 2) * H
        U, V = torch.meshgrid(uu, uu, indexing="ij")
        keep = torch.ones_like(hgt, dtype=torch.bool)
        if r.p(0.25):                                                       # fenestrated
            holes = noise2d(nu, nu, r.lu(2, 10) / H, r); keep = holes < float(torch.quantile(holes.flatten()[::97], 1 - r.u(0.05, 0.35)))
        nrm = Fn.normalize(n0[None, None] - gu[..., None] * e1 - gv[..., None] * e2, dim=-1)
        area = torch.sqrt(1 + gu ** 2 + gv ** 2)
        base = c + U[..., None] * e1 + V[..., None] * e2 + hgt[..., None] * n0
        bil = r.u(1.5, 4.0) * ht if r.p(0.25) else 0.0
        n_st = r.i(2, 4) if r.p(0.2) else 1; spacing = r.u(3, 8) * ht
        contrast = self.contrast(); g = Group("sheet", "sheet", ht, contrast)
        nt = max(2, int(math.ceil(2 * ht / H))); tt = (torch.arange(nt, device=self.device, dtype=torch.float32) + 0.5) / nt * 2 * ht - ht
        leaf = [0.0] if bil == 0 else [-0.5 * bil - ht, 0.5 * bil + ht]
        for k in range(n_st):
            oid = self.ids(1)
            off = (k - (n_st - 1) / 2) * spacing
            for lf in leaf:
                Q = base[keep][:, None] + (off + lf + tt)[None, :, None] * nrm[keep][:, None]
                W = (area[keep] * H * H * (2 * ht / nt))[:, None].expand(-1, nt)
                tens = sheet_tensor(nrm[keep])
                hol = bil / (bil + 2 * ht) if bil else 0.0
                desc = torch.tensor([ht, 0.0, hol, 0.0, 0.0], device=self.device).expand(tens.shape[0], 5)
                V = torch.cat([tens, desc, torch.full((tens.shape[0], 1), contrast, device=self.device)], 1)
                g.points.append((Q.reshape(-1, 3), W.reshape(-1), V[:, None].expand(-1, nt, -1).reshape(-1, 12), oid.expand(Q.shape[0] * nt)))
        self.record["families"].append({"sheet": {"half_thickness_vox": round(ht, 2), "bilayer": round(bil, 1), "stack": n_st, "fenestrated": bool((~keep).any())}})
        return g

    def foam(self, porous: bool = False):
        r = self.r
        cell = self.size(20, 1500, 6.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.5, r.u(0.06, 0.2) * cell); contrast = self.contrast(); oid = int(self.ids(1)[0])
        Z, Y, X = self.shape; dev = self.device
        occ = torch.zeros(self.shape, device=dev); nrm = torch.zeros((3,) + self.shape, device=dev)
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        if not porous:
            n = max(4, int(Z * Y * X / cell ** 3)); seeds = r.rand(n, 3) * torch.tensor(self.shape, device=dev)
            for z0 in range(0, Z, 16):
                q = g_[z0:z0 + 16].reshape(-1, 3); D = torch.cdist(q, seeds)
                d2, i2 = D.topk(2, dim=1, largest=False)
                s1, s2 = seeds[i2[:, 0]], seeds[i2[:, 1]]; sep = (s2 - s1).norm(dim=1).clamp_min(1e-6)
                dist = (d2[:, 1] ** 2 - d2[:, 0] ** 2) / (2 * sep)
                occ[z0:z0 + 16] = (0.5 + hw - dist).clamp(0, 1).view(-1, Y, X)
                nrm[:, z0:z0 + 16] = ((s2 - s1) / sep[:, None]).T.reshape(3, -1, Y, X)
        else:
            phi_ = smooth_noise(self.shape, cell / 3, r.g, dev)
            gz, gy, gx = torch.gradient(phi_); gr = torch.stack([gz, gy, gx]); gm = gr.norm(dim=0).clamp_min(1e-6)
            occ = (0.5 + hw - phi_.abs() / gm).clamp(0, 1); nrm = gr / gm
        if r.p(0.5):                                                        # confined to a region
            reg = gaussian(r.rand(*self.shape), r.lu(10, 30)); occ = occ * (reg > float(torch.quantile(reg.flatten()[::101], r.u(0.2, 0.6)))).float()
        tens = sheet_tensor(nrm.movedim(0, -1)).movedim(-1, 0)
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = hw; desc[2] = max(0.0, 1 - 3 * hw / cell)
        g = Group("porous" if porous else "foam", "sheet", hw, contrast, dense=[(occ, tens, desc, contrast, torch.full(self.shape, oid, device=dev, dtype=torch.long))])
        self.record["families"].append({"porous" if porous else "foam": {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2)}})
        return g

    def cells(self):
        r = self.r; dev = self.device
        rc = self.size(3, 40, 1.5, 14.0)
        if rc is None:
            return None
        elong = r.lu(1.0, 2.5); dmin = 2.1 * rc * max(1.0, elong) * 1.12          # bounding spheres (with the size jitter) never overlap
        n_c = int(min(4000, 2.5 * np.prod(self.shape) / dmin ** 3))
        C = r.rand(n_c, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        D = torch.cdist(C, C); earlier = torch.ones_like(D, dtype=torch.bool).tril(-1)
        C = C[~((D < dmin) & earlier).any(1)]                                # random placement, a candidate dropped if it hits an earlier one
        keep = r.rand(C.shape[0]) < r.u(0.3, 1.0)
        if self.depth is not None:                                          # laminar density
            bd = torch.tensor(self.band_density, device=dev)[self.band_of(C)]
            keep = keep & (r.rand(C.shape[0]) < bd)
        C = C[keep]; C = C[torch.randperm(C.shape[0], generator=r.g, device=dev)[:700]]; n = C.shape[0]   # shuffled before the cap
        if n == 0:
            return None
        Rm = _rot(r, n) if r.p(0.5) else _rot(r, 1).expand(n, 3, 3)
        ax = torch.stack([torch.ones(n, device=dev), torch.ones(n, device=dev), torch.full((n,), elong, device=dev)], 1) * rc * torch.exp(0.12 * r.randn(n, 1)).clamp(max=1.12)
        A = Rm @ torch.diag_embed(ax)                                       # (n, 3, 3): unit sphere -> cell surface
        tm = max(0.5, r.u(0.05, 0.15) * rc); nuc = r.u(0.3, 0.6)
        cmem, cnuc = self.contrast(), self.contrast()
        cid = self.ids(n); nid = self.ids(n)
        for a_, b_ in zip(cid.tolist(), nid.tolist()):
            self.parents[b_] = a_
        nd = max(32, int(4 * math.pi * float(ax.max()) ** 2 / H ** 2)); Dd = _fib_sphere(nd, dev)            # (nd, 3)
        nt = max(1, int(math.ceil(tm / H)))
        Ainv_T = torch.linalg.inv(A).transpose(1, 2)
        gm = Fn.normalize(torch.einsum("nij,dj->ndi", Ainv_T, Dd), dim=-1)                                   # outward normals
        dA = torch.linalg.det(A).abs()[:, None] * torch.einsum("nij,dj->ndi", Ainv_T, Dd).norm(dim=-1) * (4 * math.pi / nd)
        gmem = Group("membranes", "sheet", tm / 2, cmem); gnuc = Group("nuclei", "blob", nuc * rc, cnuc)
        for k in range(nt):
            rho = 1 - (k + 0.5) * tm / nt / rc
            Q = C[:, None] + rho * torch.einsum("nij,dj->ndi", A, Dd)
            W = dA * (tm / nt)
            V = torch.cat([sheet_tensor(gm), torch.stack([torch.full_like(W, tm / 2), torch.zeros_like(W), torch.full_like(W, (1 - tm / rc) ** 2), torch.zeros_like(W), torch.zeros_like(W)], -1),
                           torch.full_like(W, cmem)[..., None]], -1)
            gmem.points.append((Q.reshape(-1, 3), W.reshape(-1), V.reshape(-1, 12), cid[:, None].expand(-1, nd).reshape(-1)))
        # nuclei: filled ellipsoids, their tensor the normalised second moment
        An = A * nuc; ns = max(4, int(math.ceil(2 * float(ax.max()) * nuc / H)))
        U3 = (torch.arange(ns, device=dev, dtype=torch.float32) + 0.5) / ns * 2 - 1
        G3 = torch.stack(torch.meshgrid(U3, U3, U3, indexing="ij"), -1).reshape(-1, 3); G3 = G3[G3.norm(dim=1) <= 1]
        off = (r.rand(n, 3) - 0.5) * 0.4 * rc * (1 - nuc)
        Q = (C + off)[:, None] + torch.einsum("nij,mj->nmi", An, G3)
        W = (torch.linalg.det(An).abs() * (2 / ns) ** 3)[:, None].expand(-1, G3.shape[0])
        M2 = An @ An.transpose(1, 2); M2 = M2 / M2.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]
        Vn = torch.cat([_sym6(M2), torch.stack([ax.mean(1) * nuc, torch.zeros(n, device=dev), torch.zeros(n, device=dev), ax.min(1).values / ax.max(1).values, torch.zeros(n, device=dev)], -1),
                        torch.full((n, 1), cnuc, device=dev)], -1)
        gnuc.points.append((Q.reshape(-1, 3), W.reshape(-1), Vn[:, None].expand(-1, G3.shape[0], -1).reshape(-1, 12), nid[:, None].expand(-1, G3.shape[0]).reshape(-1)))
        self.obstacles.append((C, ax.max(1).values))
        self.record["families"].append({"cells": {"n": n, "radius_vox": round(rc, 2), "elongation": round(elong, 2), "nucleus": round(nuc, 2)}})
        return [gmem, gnuc]

    def crystals(self):
        r = self.r; rad = self.size(2, 200, 1.0, 6.0)
        if rad is None:
            return None
        from .generator import direction_field
        prim = ["polyhedron", "shard", "superquadric"][r.i(0, 2)]
        L = Layer("solid", direction_field(self.shape, r.lu(10, 60), self.r.g, self.device), rad, 2 * rad, 0.0,
                  torch.ones(self.shape, dtype=torch.bool, device=self.device), self.contrast(), primitive=prim,
                  axes=(r.lu(0.6, 2.0), r.lu(0.6, 2.0), r.lu(0.6, 3.0)), exps=(r.lu(0.2, 2.0), r.lu(0.2, 2.0)))
        L.density = layer_density(L, r.lu(0.02, 0.15))
        self.record["families"].append({"crystals": {"primitive": prim, "radius_vox": round(rad, 2)}})
        return Group("crystals", "solid", rad, float(L.contrast), layer=L)

    # ---- fibres yield to cells and vessels
    def deflect(self, cv: Curves, iters: int = 4, margin: float = 0.5):
        if not self.obstacles:
            return cv
        C = torch.cat([o[0] for o in self.obstacles]); R = torch.cat([o[1] for o in self.obstacles])
        P = cv.P.clone(); N, S, _ = P.shape
        rr = cv.rad.reshape(-1)
        for _ in range(iters):
            flat = P.reshape(-1, 3); push = torch.zeros_like(flat)
            for i0 in range(0, flat.shape[0], 200000):
                q = flat[i0:i0 + 200000]; d = torch.cdist(q, C)
                ov = (R[None] + rr[i0:i0 + 200000, None] + margin - d).clamp_min(0)
                if bool((ov > 0).any()):
                    dirv = (q[:, None] - C[None]) / d[..., None].clamp_min(1e-6)
                    push[i0:i0 + 200000] = (ov[..., None] * dirv).sum(1)
            P = P + push.view(N, S, 3)
            sm = smooth1d(P.reshape(N, S, 3), 1.5)                          # smooth along the curve (unit-variance kernel: re-normalise the mean)
            P = sm / smooth1d(torch.ones(N, S, 3, device=P.device), 1.5)
        cv.P = P
        return cv

    # ---- the scene
    def build(self, families: dict | None = None, n_groups: tuple = (1, 5), layout_p: float = 0.3):
        r = self.r
        if r.p(layout_p):
            self.layout()
        fam = dict(self.FAMILIES); fam.update(families or {})
        names = [k for k, v in fam.items() if v > 0 and (k != "radial" or self.depth is not None)]
        p = np.array([fam[k] for k in names], dtype=float); p /= p.sum()
        n = r.i(*n_groups)
        chosen = list(r.np.choice(names, size=n, p=p))
        order = {"cells": 0, "tree": 1, "network": 2}                          # obstacles first, fibres after
        chosen.sort(key=lambda k: order.get(k, 5))
        bundles = []
        for name in chosen:
            if name == "bundle":
                prev = bundles[-1].centreline if bundles and r.p(0.5) else None
                mode = ["angled", "kissing", "fanning"][r.i(0, 2)] if prev is not None else "single"
                gs = self.bundle(prev, mode)
                if gs is not None:
                    gs.curves = [self.deflect(c) for c in gs.curves]; bundles.append(gs)
            elif name == "foam":
                gs = self.foam(False)
            elif name == "porous":
                gs = self.foam(True)
            else:
                gs = getattr(self, name)()
            if gs is None:
                continue
            for g_ in (gs if isinstance(gs, list) else [gs]):
                self.groups.append(g_)
        if not self.groups:                                                    # at least one population
            g_ = self.bundle()
            if g_ is not None:
                self.groups.append(g_)
        return self

    def render(self) -> dict:
        """All populations on the canvas: per group occupancy / tensor / desc (5, voxels and 0-1) / contrast map; the
        lumen contents; identities; branch and endpoint maps."""
        cv = Canvas(self.shape, self.device); out = []
        for g_ in self.groups:
            if g_.layer is not None:
                occ, tens = render_layer(g_.layer, self.shape, self.r.g, self.device)
                d3 = g_.layer.desc
                desc = torch.stack([d3[0], torch.zeros_like(d3[0]), d3[1], torch.ones_like(d3[0]) * (occ > 0), d3[2]])
                cv.add_dense(occ, tens, desc, g_.contrast, torch.full(self.shape, int(self.ids(1)[0]), device=self.device, dtype=torch.long))
            for c in g_.curves:
                render_curves(cv, c)
            for Q, W, V, obj in g_.points:
                cv.splat(Q, W, V, obj=obj)
            for occ, tens, desc, con, om in g_.dense:
                cv.add_dense(occ, tens, desc, con, om)
            res = cv.finish(); res["group"] = g_; out.append(res)
        ids, frac = cv.identities(2)
        lum = cv.lumen.view(self.shape).clamp(0, 1); lum_c = cv.lumen_c.view(self.shape) / cv.lumen.view(self.shape).clamp_min(1e-9)
        maps = torch.zeros((2,) + self.shape, device=self.device)
        for k, pts in enumerate((self.branch, self.ends)):
            if pts:
                Pp = torch.cat(pts); idx = Pp.round().long()
                ok = ((idx >= 0) & (idx < torch.tensor(self.shape, device=self.device))).all(1)
                if bool(ok.any()):
                    maps[k].index_put_((idx[ok, 0], idx[ok, 1], idx[ok, 2]), torch.ones(int(ok.sum()), device=self.device), accumulate=True)
                    maps[k] = gaussian(maps[k], 1.0) * (2 * math.pi) ** 1.5
        return {"groups": out, "ids": ids, "id_frac": frac, "lumen": lum, "lumen_c": lum_c, "points": maps}


# ------------------------------------------------------------------------------------------------------------------
# a training pair: a crop of a rendered scene, composited, acquired
# ------------------------------------------------------------------------------------------------------------------

_CACHE: dict = {}


def render_scene(canvas, scene_seed: int, device, voxel_um=None, families=None, n_groups=(1, 5), layout_p=0.3, content="general"):
    key = (tuple(canvas), scene_seed, str(device), voxel_um, str(families), tuple(n_groups), layout_p, content)
    if key not in _CACHE:
        _CACHE.clear()                                                      # one scene at a time (crops of it are drawn in sequence)
        sc = Scene(canvas, scene_seed, device, voxel_um, content).build(families, n_groups, layout_p)
        _CACHE[key] = (sc, sc.render())
    return _CACHE[key]


def _extra_acquisition(img, layers, rng: Rng, modality: str, rec: dict):
    """Artefacts the acquisition model lacks: histology tears (tissue missing along a crack, labels removed with it) and
    folds (a band of a section doubled), local-tomography truncation (low frequencies lost) for HiP-CT / CT."""
    shp = img.shape
    if modality in ("histology", "mixed") and rng.p(0.3):
        k = rng.i(1, 3); a = rng.i(0, shp[0] - 1); n = noise2d(shp[1], shp[2], rng.lu(6, 20), rng)
        crack = (n.abs() < rng.u(0.02, 0.08)).float()
        sl = slice(max(0, a - k), min(shp[0], a + k))
        img[sl] = img[sl] * (1 - crack) + float(img.min()) * crack
        for L in layers:
            L.occupancy[sl] = L.occupancy[sl] * (1 - crack)
        rec["tear"] = (a, k)
    if modality in ("histology", "mixed") and rng.p(0.2):
        a = rng.i(1, shp[0] - 2); b0 = rng.i(0, shp[1] - 8); w = rng.i(3, 8)
        if b0 + 2 * w <= shp[1]:                                            # the neighbouring band folded over it
            img[a, b0:b0 + w] = img[a, b0:b0 + w] + 0.6 * img[a, b0 + w:b0 + 2 * w]
        rec["fold"] = (a, b0, w)
    if modality in ("hipct", "ct", "mixed") and rng.p(0.3):
        s = rng.lu(10, 40); alpha = rng.u(0.2, 0.8)
        img = img - alpha * (gaussian(img, s) - img.mean())
        rec["local_tomography"] = (round(s, 1), round(alpha, 2))
    return img


def draw_scene(shape=(64, 64, 64), seed: int = 0, device="cuda", modality: str | None = None, strength: float = 1.0,
               canvas: int = 128, crops_per_scene: int = 8, voxel_um: float | None = None, families: dict | None = None,
               n_groups=(1, 5), layout_p: float = 0.3, content: str = "general", style: str | None = None, second_contrast: bool = False):
    """A training pair from an object scene, in the format of :func:`~shapetensor.learn.generator.draw`: ``(Sample,
    drawn)``. Scenes are rendered on a ``canvas``³ grid and ``crops_per_scene`` consecutive seeds share one (cut at
    different places), so a scene's cost is spread over its crops. ``drawn`` also carries ``ids``/``id_frac`` (the two
    heaviest objects per voxel), ``desc_um`` (per layer: size and wall in µm, lumen fraction, flattening, angularity),
    ``points`` (branch points, endpoints), ``recoverability`` (the orientation-visibility map: a confidence target),
    ``parents`` and the voxel size."""
    sid = seed // max(1, crops_per_scene)
    sc, ren = render_scene((canvas,) * 3, 7_000_000 + sid, device, voxel_um, families, n_groups, layout_p, content)
    r = Rng(seed, device)
    names = ["hipct", "ct", "mri", "histology", "mixed"]; modality = modality or names[r.i(0, 4)]
    # the crop
    o = [r.i(0, canvas - s) for s in shape]; sl = tuple(slice(a, a + s) for a, s in zip(o, shape))
    style = style or ["flat", "regions", "tissue", "cells"][r.i(0, 3)]
    bg, lab, R, extra = background(tuple(shape), style, r.g, device, 1.0, content)
    layers = []
    for res in ren["groups"]:
        g_ = res["group"]; occ = res["occupancy"][sl].clone()
        if float(occ.max()) <= 1e-3:
            continue
        L = Layer(g_.kind, torch.zeros(3, 1, 1, 1, device=device), max(g_.radius, 0.5), 2 * max(g_.radius, 0.5), 0.0,
                  torch.ones(tuple(shape), dtype=torch.bool, device=device), g_.contrast, occupancy=occ, tensor=res["tensor"][(slice(None),) + sl].clone())
        d5 = res["desc"][(slice(None),) + sl]
        L.desc = torch.stack([d5[0], d5[2], d5[4]])                          # the layer descriptors: thickness, hollowness, angularity
        L.desc5 = d5.clone(); L.cmap = res["contrast"][sl]; L.name = g_.name
        layers.append(L)
    shapes_ = layers + extra
    img = _composite(bg, R, shapes_, device)                                 # normalises the occupancies, adds the interface and isotropic states
    for L in layers:                                                         # contrast varying along objects (striation, profiles)
        img = img + (L.cmap - L.contrast) * L.occupancy
    lum = ren["lumen"][sl]; img = img + lum * ren["lumen_c"][sl]
    rec = {"scene": sid, "crop": o, "style": style, **sc.record}
    img = _extra_acquisition(img, layers, r, modality, rec)
    s = Sample(img, shapes_, bg, lab, style, 1.0, content, rec, R)
    d = nuisance(s, seed=seed, device=device, strength=strength, modality=modality, warp=False, fast=True, hp_max=12.0)
    d["ids"] = ren["ids"][(slice(None),) + sl]; d["id_frac"] = ren["id_frac"][(slice(None),) + sl]
    d["points"] = ren["points"][(slice(None),) + sl]
    d["desc_um"] = [torch.stack([L.desc5[0] * sc.voxel_um, L.desc5[1] * sc.voxel_um, L.desc5[2], L.desc5[3], L.desc5[4]]) for L in layers]
    d["recoverability"] = d["orientation_visibility"]; d["parents"] = sc.parents; d["voxel_um"] = sc.voxel_um
    d["record"].update({k: v for k, v in rec.items() if k not in d["record"]})
    if second_contrast:                                                      # a second contrast of the same scene (multi-contrast MRI)
        img2 = bg * 0.5 + 60
        for L in layers:
            img2 = img2 + r.u(-120, 120) * L.occupancy
        d["image2"] = img2
    return s, d
