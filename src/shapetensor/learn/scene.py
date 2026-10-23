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
from .physics import BY_MODALITY as PHYS_BY_MODALITY, MODES as PHYS_MODES, _to_grey, draw_material, draw_sequence, form_image


def _grey(x: torch.Tensor) -> torch.Tensor:
    """Any image to the common grey range (its 1-99 % to 30-220) before routes are combined."""
    return _to_grey(x)

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


def curves_subset(c: Curves, keep: torch.Tensor) -> Curves | None:
    """The curves of ``c`` selected by a boolean or index tensor over its ``N`` curves (None if empty)."""
    idx = torch.nonzero(keep)[:, 0] if keep.dtype == torch.bool else keep
    if idx.numel() == 0:
        return None
    pick = lambda v: None if v is None else v[idx]
    return Curves(P=c.P[idx], rad=c.rad[idx], flat=c.flat[idx], wall=c.wall[idx], sexp=c.sexp[idx], contrast=c.contrast[idx],
                  lumen_c=c.lumen_c[idx], obj=c.obj[idx], seg=c.seg[idx], lam=pick(c.lam), phi=pick(c.phi), density=pick(c.density),
                  fibre_r=pick(c.fibre_r))


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

    FAMILIES = {"bundle": 0.24, "tree": 0.11, "network": 0.05, "helix": 0.04, "sheet": 0.09, "foam": 0.04, "porous": 0.03,
                "cells": 0.09, "crystals": 0.04, "radial": 0.04, "plywood": 0.04, "fascicles": 0.05, "trabecular": 0.03,
                "tubules": 0.04, "epithelium": 0.03, "dendrite": 0.02, "gland": 0.02}

    def __init__(self, shape, seed: int, device="cuda", voxel_um: float | None = None, content: str = "general"):
        self.shape = tuple(shape); self.device = device; self.r = Rng(seed, device); self.seed = seed
        self.voxel_um = float(voxel_um) if voxel_um else self.r.lu(1.0, 60.0)
        self.content = content
        self.groups: list[Group] = []; self.next_obj = 1
        self.branch, self.ends = [], []                    # (n, 3) positions
        self.obstacles = []                                # (centres (n, 3), radii (n,)) that fibres yield to
        self.parents = {}                                  # object id -> parent id (nuclei in cells, fibres in bundles)
        self.obj_r = {}                                    # object id -> its smallest radius (voxels): which level of the hierarchy resolves
        self.depth = None                                  # the laminar depth coordinate (voxels), if a layout was drawn
        self.record = {"voxel_um": round(self.voxel_um, 2), "families": []}

    # ---- helpers
    def ids(self, n: int, radius: float | torch.Tensor | None = None) -> torch.Tensor:
        """``n`` new object ids; ``radius`` (voxels, scalar or per object) recorded for the identity target."""
        out = torch.arange(self.next_obj, self.next_obj + n, device=self.device); self.next_obj += n
        if radius is not None:
            rr = radius.reshape(-1).tolist() if torch.is_tensor(radius) else [float(radius)] * n
            for o_, r_ in zip(range(int(out[0]), int(out[0]) + n), rr if len(rr) == n else rr * n):
                self.obj_r[o_] = float(r_)
        return out

    def identity_lut(self, min_r: float = 1.5) -> torch.Tensor:
        """A lookup ``id -> identity target``: an object smaller than ``min_r`` voxels is identified by its parent (up the
        chain: an unresolved fibre by its bundle), so the identity asks only what the resolution can separate."""
        lut = torch.arange(self.next_obj + 1, device=self.device)
        for o_ in range(1, self.next_obj):
            t = o_
            while self.obj_r.get(t, 1e9) < min_r and t in self.parents:
                t = self.parents[t]
            lut[o_] = t
        return lut

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

    def bundle(self, cross_with: Curves | None = None, mode: str = "angled", axis=None, centre=None, radius=None, curv=None):
        """A bundle of fibres (resolved: packed on a jittered hexagonal section; unresolved: the bundle's partial volume).
        ``mode``: ``single``; ``angled`` / ``kissing`` / ``fanning`` (through a point of ``cross_with``); ``split`` (the
        tract divides past a point into two halves diverging by an angle -- returned as two groups, one population each)."""
        r = self.r; S0 = int(2.2 * max(self.shape) / H)
        Rb = radius or self.size(20, 3000, 1.5, min(self.shape) / 5)
        rf_um = r.lu(0.25, 12.0); rf = rf_um / self.voxel_um
        if Rb is None:
            return None
        Lc = 2.0 * max(self.shape); cross = None
        if cross_with is not None:                                          # through a point of an existing bundle, inside the canvas
            Pw = cross_with.P[0]; sz = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
            inner = ((Pw > 0.2 * sz) & (Pw < 0.8 * sz)).all(1)
            cand = torch.nonzero(inner)[:, 0] if bool(inner.any()) else torch.arange(Pw.shape[0] // 4, 3 * Pw.shape[0] // 4, device=self.device)
            k = int(cand[r.i(0, cand.numel() - 1)]); P0 = Pw[k]
            t0 = tangents(cross_with.P[:1])[0, k]
            p = Fn.normalize(torch.linalg.cross(t0, r.unit()), dim=0)
            if mode == "kissing":
                ang = math.radians(r.u(5, 20)); d = math.cos(ang) * t0 + math.sin(ang) * p
                nrm = Fn.normalize(torch.linalg.cross(d, torch.linalg.cross(t0, d)), dim=0)
                P0 = P0 + nrm * (float(cross_with.rad[0].mean()) + Rb)
            else:
                ang = math.radians(r.u(20, 90)); d = math.cos(ang) * t0 + math.sin(ang) * p
            c = P0; cross = (P0, t0, d)
        else:
            c = centre if centre is not None else self.centre_point(); d = axis if axis is not None else r.unit()
        start = (c - d * Lc / 2)[None]; S = S0
        Pc = smooth_paths(start, d[None], torch.tensor([Lc], device=self.device), S, torch.tensor([curv if curv is not None else r.lu(0.001, 0.02)], device=self.device), r.lu(15, 80), r)
        s = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
        fan = r.u(0.3, 1.5) if (mode == "fanning" or r.p(0.15)) else 0.0
        Rs = Rb * (1 + fan * torch.relu(s - 0.5) * 2)                       # fanning: the bundle widens past its middle
        contrast = self.contrast(); bid = int(self.ids(1, Rb)[0])
        side = None
        if rf >= 0.45:                                                      # resolved fibres, packed on a jittered hexagonal section
            sp = max(2 * rf * 1.15 * r.u(1.05, 1.5), 1.0) / Rb                  # spacing clears the largest fibre (radius jitter capped at 1.15)
            n_side = int(2 / sp) + 2
            ii, jj = torch.meshgrid(torch.arange(-n_side, n_side + 1, device=self.device), torch.arange(-n_side, n_side + 1, device=self.device), indexing="ij")
            o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp
            o = o + 0.1 * sp * (r.rand(*o.shape) - 0.5) * 2                # jitter, kept below the gap: no overlaps
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
            if mode == "split":                                             # the two halves diverge past s0 (smooth onset)
                v = r.unit()[:2]; v = v / v.norm().clamp_min(1e-6); s0 = r.u(0.3, 0.6); th = math.radians(r.u(15, 60))
                side = torch.where((o @ v) >= 0, 1.0, -1.0)
                ramp = smooth1d(torch.relu(s - s0)[None] * Lc, 6.0)[0] / smooth1d(torch.ones(1, S, device=self.device), 6.0)[0]
                lat = side[:, None] * math.tan(th / 2) * ramp[None]            # (N, S) voxels
                P = P + lat[..., None] * (v[0] * e1 + v[1] * e2)
                self.record.setdefault("split", []).append(round(math.degrees(th), 1))
            rad = (rf * torch.exp(0.12 * r.randn(N)).clamp(0.8, 1.15))[:, None].expand(N, S).clone()
            rad, con, kw = self._fibre_extras(N, S, rad, r)
            cmap = contrast * (con if con is not None else 1.0) * torch.ones(N, S, device=self.device)
            obj = self.ids(N, rad[:, 0])
            for o_ in obj.tolist():
                self.parents[o_] = bid
            cv = curves_like(P, rad, r, obj=obj, seg=torch.full((N,), bid, device=self.device), contrast=cmap, **kw)
            g = Group("bundle", "line", float(rad.median()), contrast, curves=[cv])
            g.offsets, g.rf = o, float(rad.max())
            self.record["families"].append({"bundle": {"fibres": N, "fibre_um": round(rf_um, 2), "radius_vox": round(Rb, 1), "fan": round(fan, 2), "mode": mode if (cross_with is not None or mode == "split") else "single"}})
        else:                                                               # unresolved fibres: the bundle's partial volume
            phi_ = r.u(0.3, 0.8)
            cv = curves_like(Pc, Rs[None].clone(), r, obj=torch.tensor([bid], device=self.device), seg=torch.tensor([bid], device=self.device),
                             contrast=contrast, density=torch.tensor([phi_], device=self.device), fibre_r=torch.tensor([rf], device=self.device))
            g = Group("bundle", "line", rf, contrast, curves=[cv]); g.offsets = None
            self.record["families"].append({"bundle": {"unresolved": True, "fibre_um": round(rf_um, 2), "packing": round(phi_, 2), "radius_vox": round(Rb, 1), "mode": mode if cross_with is not None else "single"}})
        g.centreline = curves_like(Pc, Rs[None].clone(), r, obj=torch.tensor([bid], device=self.device), seg=torch.tensor([bid], device=self.device), contrast=contrast)
        g.cross, g.Rb = cross, float(Rs.max())
        if side is not None:                                                # one population per half
            out = []
            for sg in (1.0, -1.0):
                c_ = curves_subset(g.curves[0], side == sg)
                if c_ is not None:
                    h = Group("bundle", "line", g.radius, contrast, curves=[c_]); h.centreline, h.cross, h.Rb, h.offsets = g.centreline, None, g.Rb, None
                    out.append(h)
            return out
        return g

    def interleave(self, ga: Group, gb: Group):
        """Two resolved bundles crossing at an angle interdigitate: in the crossing region the fibres of each are
        regrouped into fascicles (slabs of thickness ``w`` across both directions) that alternate -- A's slab k moves to
        the k-th even place, B's to the k-th odd place, a gap of a fibre diameter between them -- so no fibre crosses
        another; the bundles thicken where they cross. A fibre's displacement fades in over a taper along it."""
        P0, tA, dB = gb.cross
        n = torch.linalg.cross(tA, dB); 
        if float(n.norm()) < 1e-3:
            return
        n = n / n.norm(); r = self.r
        rfm = max(getattr(ga, "rf", 0.5), getattr(gb, "rf", 0.5))
        w = r.u(3, 10) * 2 * rfm; gap = 2.2 * rfm
        mA = Fn.normalize(torch.linalg.cross(n, tA), dim=0); mB = Fn.normalize(torch.linalg.cross(n, dB), dim=0)
        for g, parity, m_other, R_other in ((ga, 0, mB, gb.Rb), (gb, 1, mA, ga.Rb)):
            for c in g.curves:
                rel = c.P - P0
                h = rel @ n
                k = torch.floor(h / w)
                shift = k * (w + 2 * gap) + parity * (w + gap)
                dist = (rel @ m_other).abs()                                 # in-plane distance from the other bundle's axis
                taper = max(2.0 * w, 0.5 * R_other)
                wt = ((R_other + rfm + taper - dist) / taper).clamp(0, 1)
                wt = wt * wt * (3 - 2 * wt)                                  # smoothstep
                wt = smooth1d(wt, 4.0) / smooth1d(torch.ones_like(wt), 4.0)
                c.P = c.P + (wt * shift)[..., None] * n
        self.record.setdefault("interleaved", []).append({"slab_vox": round(w, 1)})

    def tree(self, style: str = "vessel"):
        """A branching tree grown generation by generation. ``vessel``: Murray's law (r^3 = r1^3 + r2^3) and its optimal
        angles, a wall and lumen, stenoses and collapses; ``dendrite``: Rall's 3/2 law, thin and tortuous, solid, a
        soma; ``gland``: a narrow epithelial tube (thick wall, open lumen) ending in acini (hollow spheres). Returned as
        two populations: at every branching the smaller child changes side, so a branch point always separates the
        departing branch from the parent and its continuation (the same object id on both sides)."""
        r = self.r
        if style == "dendrite":
            r0 = self.size(0.5, 6, 0.45, 3.0); rmin = max(0.15 / self.voxel_um, 0.3); ex = 1.5; tort = (0.03, 0.15); Lf = (8, 25)
        elif style == "gland":
            r0 = self.size(15, 150, 1.5, min(self.shape) / 10); rmin = max(8.0 / self.voxel_um, 0.8); ex = 3.0; tort = (0.005, 0.03); Lf = (3, 6)
        else:
            r0 = self.size(15, 800, 1.5, min(self.shape) / 7); rmin = max(3.0 / self.voxel_um, 0.45); ex = 3.0; tort = (0.005, 0.04); Lf = (5, 12)
        if r0 is None or r0 < rmin * 1.5:
            return None
        tid = int(self.ids(1, r0)[0]); contrast = self.contrast(); g = Group("tree", "line", r0, contrast)
        g.murray = []
        if style == "dendrite":
            wf, f0, lumen_c = 0.0, 1.0, 0.0
        elif style == "gland":
            wf, f0, lumen_c = r.u(0.3, 0.55), 1.0, r.u(-60, 60)
        else:
            wf = r.u(0.08, 0.3); f0 = 1.0 if r.p(0.6) else r.u(0.5, 1.0)       # arterial-like wall; veins flatter
            lumen_c = r.u(-60, 80) if r.p(0.7) else 0.0
        c = self.centre_point(0.1); d = r.unit()
        start = (c - d * r0 * 4)[None]; dirs = d[None]; rads = torch.tensor([r0], device=self.device)
        side = torch.zeros(1, device=self.device, dtype=torch.long)
        nseg = 0; segs, sides, ends_all = [], [], []
        for depth in range(12):
            A = start.shape[0]
            if A == 0 or nseg > 300:
                break
            L = torch.tensor([r.lu(*Lf) * float(x) for x in rads.tolist()], device=self.device)
            S = max(8, int(math.ceil(float(L.max()) / H)))
            P = smooth_paths(start, dirs, L, S, torch.full((A,), r.lu(*tort), device=self.device), float(L.mean()) / (3 if style == "dendrite" else 1), r)
            # children where they stay above the smallest radius (r^ex = r1^ex + r2^ex)
            q = torch.tensor([r.u(0.15, 0.5) for _ in range(A)], device=self.device)
            r1 = rads * (1 - q) ** (1 / ex); r2 = rads * q ** (1 / ex)
            grow = r1 >= rmin
            if nseg + 2 * int(grow.sum()) > 300:
                grow = grow & (torch.rand(A, generator=r.g, device=self.device) < 0.5)
            prof = rads[:, None].expand(A, S).clone()
            sfrac = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
            tip = torch.where(grow, r1, rads * (0.5 if style != "gland" else 1.0))   # taper the last quarter to the larger child (or a tip)
            ramp = ((sfrac - 0.75) / 0.25).clamp(0, 1)
            prof = prof * (1 - ramp) + tip[:, None] * ramp
            if style == "vessel":
                prof = prof * events(A, S, 0.3, (0.2, 0.6), (3, 12), r)       # stenoses
                flat = f0 * events(A, S, 0.35, (0.5, 0.95), (4, 15), r)      # collapses: the section flattens, the lumen closes
            else:
                flat = torch.ones(A, S, device=self.device)
            seg_ids = torch.arange(nseg, nseg + A, device=self.device); nseg += A
            kw = {"wall": prof * wf, "lumen_c": torch.full((A, S), lumen_c, device=self.device)} if wf > 0 else {}
            cvs = curves_like(P, prof, r, obj=torch.full((A,), tid, device=self.device), seg=seg_ids, contrast=contrast, flat=flat, **kw)
            segs.append(cvs); sides.append(side)
            ends = P[:, -1]; tend = tangents(P)[:, -1]
            if bool((~grow).any()):
                self.ends.append(ends[~grow]); ends_all.append((ends[~grow], rads[~grow]))
            if not bool(grow.any()):
                break
            self.branch.append(ends[grow])
            r0g, r1g, r2g = rads[grow], r1[grow], r2[grow]
            g.murray.append(torch.stack([r0g, r1g, r2g], 1))                    # each branching's parent and child radii
            if ex == 3.0:                                                       # Murray's optimal angles
                cos1 = ((r0g ** 4 + r1g ** 4 - r2g ** 4) / (2 * r0g ** 2 * r1g ** 2)).clamp(-1, 1)
                cos2 = ((r0g ** 4 + r2g ** 4 - r1g ** 4) / (2 * r0g ** 2 * r2g ** 2)).clamp(-1, 1)
            else:                                                               # dendrites: wide, irregular angles
                cos1 = torch.cos(torch.deg2rad(10 + 30 * torch.rand(int(grow.sum()), generator=r.g, device=self.device)))
                cos2 = torch.cos(torch.deg2rad(30 + 50 * torch.rand(int(grow.sum()), generator=r.g, device=self.device)))
            te = tend[grow]; pe = Fn.normalize(torch.linalg.cross(te, r.unit(int(grow.sum()))), dim=1)
            d1 = Fn.normalize(cos1[:, None] * te + torch.sqrt(1 - cos1 ** 2)[:, None] * pe, dim=1)
            d2 = Fn.normalize(cos2[:, None] * te - torch.sqrt(1 - cos2 ** 2)[:, None] * pe, dim=1)
            start = torch.cat([ends[grow], ends[grow]]); dirs = torch.cat([d1, d2]); rads = torch.cat([r1g, r2g])
            side = torch.cat([side[grow], 1 - side[grow]])                       # the larger child keeps its parent's side
        g.curves = segs
        self.obstacles.append((torch.cat([cv.P[:, ::4].reshape(-1, 3) for cv in segs]), torch.cat([cv.rad[:, ::4].reshape(-1) for cv in segs])))
        self.record["families"].append({"tree": {"style": style, "root_vox": round(r0, 1), "segments": nseg, "wall_frac": round(wf, 2)}})
        out = []
        for sd in (0, 1):                                                       # two populations by branch side
            cs = [curves_subset(cv, sv == sd) for cv, sv in zip(segs, sides)]
            cs = [c_ for c_ in cs if c_ is not None]
            if cs:
                h = Group("tree" if style == "vessel" else style, "line", r0, contrast, curves=cs); h.murray = g.murray
                out.append(h)
        if style == "dendrite":                                                 # the soma: a filled ellipsoid at the root
            rs = r0 * r.u(3, 6); out.append(self._solids(start=None, C=c[None] - d[None] * r0 * 4, A=torch.eye(3, device=self.device)[None] * rs,
                                                         name="soma", oid=torch.tensor([tid], device=self.device)))
        if style == "gland" and ends_all:                                       # acini: hollow spheres at the blind ends
            Ce = torch.cat([e for e, _ in ends_all]); Re = torch.cat([rr_ for _, rr_ in ends_all]) * r.u(1.8, 3.0)
            out.append(self._shells(Ce, Re, wall=float(wf) * float(Re.mean()) * 0.8, name="acini", parent=tid))
        return out

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
        nid = int(self.ids(1, rc)[0]); contrast = self.contrast()
        kw = {"wall": rad * 0.35, "lumen_c": torch.full_like(rad, r.u(-40, 60))} if rc >= 2.0 else {}
        cv = curves_like(P, rad, r, obj=torch.full((e.shape[0],), nid, device=self.device), seg=torch.arange(e.shape[0], device=self.device), contrast=contrast, **kw)
        deg = torch.bincount(e.reshape(-1), minlength=n)
        if bool((deg >= 3).any()):
            self.branch.append(pts[deg >= 3])
        self.record["families"].append({"network": {"nodes": n, "edges": int(e.shape[0]), "radius_vox": round(rc, 2)}})
        # one population per dominant axis of the edge: edges meeting at a node at large angles are apart
        dom = ((b - a) / L[:, None].clamp_min(1e-6)).abs().argmax(1)
        out = []
        for j in range(3):
            c_ = curves_subset(cv, dom == j)
            if c_ is not None:
                out.append(Group("network", "line", rc, contrast, curves=[c_]))
        return out

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
        cv = curves_like(P, rad, r, obj=self.ids(n_str, rf), seg=torch.zeros(n_str, dtype=torch.long, device=self.device), contrast=contrast)
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
        cv = curves_like(Pc, rad, r, obj=self.ids(n, rf), seg=torch.zeros(n, dtype=torch.long, device=self.device), contrast=contrast)
        self.record["families"].append({"radial": {"radial": radial, "fibres": n, "band": k}})
        return Group("radial" if radial else "tangential", "line", rf, contrast, curves=[cv])

    # ---- surfaces and implicit walls
    def sheet(self, normal=None, centre=None, half_thickness=None, height=None):
        """A folded sheet (fenestrated, bilayer, stacked by chance); ``height = (sf, amp, kx, ky, ph, e1, e2)`` puts it on
        the analytic fold of an epithelium (its basement membrane) instead of a random one."""
        r = self.r
        ht = half_thickness or self.size(0.25, 150, 0.5, min(self.shape) / 10)
        if ht is None:
            return None
        n0 = normal if normal is not None else r.unit()
        a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        c = centre if centre is not None else self.centre_point(); Lh = 0.9 * max(self.shape); nu = int(2 * Lh / H)
        uu = (torch.arange(nu, device=self.device, dtype=torch.float32) - nu / 2) * H
        U, V = torch.meshgrid(uu, uu, indexing="ij")
        if height is not None:
            sf, amp, kx, ky, ph, e1, e2 = height
            hgt = amp * torch.sin(2 * math.pi * kx * U + ph) * torch.cos(2 * math.pi * ky * V)
        else:
            sf = r.lu(8, 40); hgt = noise2d(nu, nu, sf / H, r) * (r.u(0, 1.0) * sf)
        gu, gv = torch.gradient(hgt, spacing=H)
        keep = torch.ones_like(hgt, dtype=torch.bool)
        if height is None and r.p(0.25):                                    # fenestrated
            holes = noise2d(nu, nu, r.lu(2, 10) / H, r); keep = holes < float(torch.quantile(holes.flatten()[::97], 1 - r.u(0.05, 0.35)))
        nrm = Fn.normalize(n0[None, None] - gu[..., None] * e1 - gv[..., None] * e2, dim=-1)
        area = torch.sqrt(1 + gu ** 2 + gv ** 2)
        base = c + U[..., None] * e1 + V[..., None] * e2 + hgt[..., None] * n0
        bil = r.u(1.5, 4.0) * ht if (height is None and r.p(0.25)) else 0.0
        n_st = r.i(2, 4) if (height is None and r.p(0.2)) else 1; spacing = r.u(3, 8) * ht
        contrast = self.contrast(); g = Group("sheet", "sheet", ht, contrast)
        nt = max(2, int(math.ceil(2 * ht / H))); tt = (torch.arange(nt, device=self.device, dtype=torch.float32) + 0.5) / nt * 2 * ht - ht
        leaf = [0.0] if bil == 0 else [-0.5 * bil - ht, 0.5 * bil + ht]
        for k in range(n_st):
            oid = self.ids(1, 1e3)                                          # an extended surface: separable at any thickness
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
        hw = max(0.5, r.u(0.06, 0.2) * cell); contrast = self.contrast(); oid = int(self.ids(1, 1e3)[0])
        Z, Y, X = self.shape; dev = self.device
        occ = torch.zeros(self.shape, device=dev); nrm = torch.zeros((3,) + self.shape, device=dev)
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        if not porous:
            n = max(4, int(Z * Y * X / cell ** 3)); seeds = r.rand(n, 3) * torch.tensor(self.shape, device=dev)
            bs = 32; m = 2.5 * cell + 2                                                  # blocks against the seeds near them only (memory ~ block x local seeds)
            for z0 in range(0, Z, bs):
                for y0 in range(0, Y, bs):
                    for x0 in range(0, X, bs):
                        q = g_[z0:z0 + bs, y0:y0 + bs, x0:x0 + bs]; bsh = q.shape[:3]; q = q.reshape(-1, 3)
                        lo = torch.tensor([z0, y0, x0], device=dev, dtype=torch.float32) - m; hi = lo + bs + 2 * m
                        near = ((seeds >= lo) & (seeds <= hi)).all(1)
                        sd = seeds[near] if int(near.sum()) >= 2 else seeds
                        d2, i2 = torch.cdist(q, sd).topk(2, dim=1, largest=False)
                        s1, s2 = sd[i2[:, 0]], sd[i2[:, 1]]; sep = (s2 - s1).norm(dim=1).clamp_min(1e-6)
                        dist = (d2[:, 1] ** 2 - d2[:, 0] ** 2) / (2 * sep)
                        occ[z0:z0 + bs, y0:y0 + bs, x0:x0 + bs] = (0.5 + hw - dist).clamp(0, 1).view(bsh)
                        nrm[:, z0:z0 + bs, y0:y0 + bs, x0:x0 + bs] = ((s2 - s1) / sep[:, None]).T.reshape(3, *bsh)
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
        out = self._cell_groups(C, Rm, ax, rc)
        self.record["families"].append({"cells": {"n": n, "radius_vox": round(rc, 2), "elongation": round(elong, 2)}})
        return out

    def _cell_groups(self, C, Rm, ax, rc):
        """Cells at centres ``C (n, 3)``, orientations ``Rm (n, 3, 3)``, semi-axes ``ax (n, 3)``: membranes (thin shells,
        sheet tensors) and nuclei (filled ellipsoids, their tensor the normalised second moment), nucleus -> cell."""
        r = self.r; dev = self.device; n = C.shape[0]
        A = Rm @ torch.diag_embed(ax)                                       # (n, 3, 3): unit sphere -> cell surface
        tm = max(0.5, r.u(0.05, 0.15) * rc); nuc = r.u(0.3, 0.6)
        cmem, cnuc = self.contrast(), self.contrast()
        cid = self.ids(n, ax.min(1).values); nid = self.ids(n, ax.min(1).values * nuc)
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
        off = (r.rand(n, 3) - 0.5) * 0.4 * rc * (1 - nuc)
        gn = self._solids(C=C + off, A=A * nuc, name="nuclei", oid=nid, contrast=cnuc)
        gnuc.points = gn.points
        self.obstacles.append((C, ax.max(1).values))
        return [gmem, gnuc]

    def _solids(self, C, A, name, oid, contrast=None, start=None, shape: str = "ellipsoid", exps=(2.0, 2.0), normals=None):
        """Filled solids ``x = C + A g`` for ``g`` in a unit shape (``ellipsoid``; ``superquadric`` with exponents
        ``exps``; ``polyhedron``: inside every plane ``n_i . g <= 1`` of ``normals``), their tensor the normalised second
        moment of the solid -- a blob's shape, no category. ``C (n, 3)``, ``A (n, 3, 3)``, ``oid (n,)``."""
        dev = self.device; n = C.shape[0]
        contrast = self.contrast() if contrast is None else contrast
        amax = float(torch.linalg.matrix_norm(A, ord=2).max())
        ns = max(4, int(math.ceil(2 * amax / H)))
        U3 = (torch.arange(ns, device=dev, dtype=torch.float32) + 0.5) / ns * 2 - 1
        G3 = torch.stack(torch.meshgrid(U3, U3, U3, indexing="ij"), -1).reshape(-1, 3)
        if shape == "superquadric":
            e1, e2 = exps
            inside = ((G3[:, 1].abs() ** (2 / e2) + G3[:, 2].abs() ** (2 / e2)) ** (e2 / e1) + G3[:, 0].abs() ** (2 / e1)) <= 1
        elif shape == "polyhedron":
            inside = ((G3 @ normals.T) <= 1).all(1)
        else:
            inside = G3.norm(dim=1) <= 1
        G3 = G3[inside]
        if G3.shape[0] == 0:
            G3 = torch.zeros(1, 3, device=dev)
        Q = C[:, None] + torch.einsum("nij,mj->nmi", A, G3)
        W = (torch.linalg.det(A).abs() * (2 / ns) ** 3)[:, None].expand(-1, G3.shape[0])
        Mg = (G3[:, :, None] * G3[:, None, :]).mean(0)                                                     # the unit shape's second moment
        M2 = A @ Mg @ A.transpose(1, 2); M2 = M2 / M2.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-9)[:, None, None]
        sv = torch.linalg.svdvals(A)                                                                        # (n, 3) descending
        Vn = torch.cat([_sym6(M2), torch.stack([sv.mean(1), torch.zeros(n, device=dev), torch.zeros(n, device=dev), sv[:, 2] / sv[:, 0].clamp_min(1e-9),
                                                torch.full((n,), float(min(1.0, max(0.0, (exps[0] - 2) / (exps[0] + 2)))) if shape != "polyhedron" else 0.6, device=dev)], -1),
                        torch.full((n, 1), contrast, device=dev)], -1)
        g = Group(name, "blob", float(sv.mean()), contrast)
        g.points.append((Q.reshape(-1, 3), W.reshape(-1), Vn[:, None].expand(-1, G3.shape[0], -1).reshape(-1, 12), oid[:, None].expand(-1, G3.shape[0]).reshape(-1)))
        return g

    def _shells(self, C, R, wall: float, name: str, parent: int | None = None, contrast=None):
        """Hollow spheres (centres ``C (n, 3)``, outer radii ``R (n,)``, wall thickness in voxels): sheet tensors along
        the wall, descriptors (half wall, lumen fraction)."""
        dev = self.device; n = C.shape[0]; contrast = self.contrast() if contrast is None else contrast
        wall = max(0.5, float(wall)); oid = self.ids(n, R)
        if parent is not None:
            for o_ in oid.tolist():
                self.parents[o_] = parent
        nd = max(32, int(4 * math.pi * float(R.max()) ** 2 / H ** 2)); Dd = _fib_sphere(nd, dev)
        nt = max(1, int(math.ceil(wall / H))); g = Group(name, "sheet", wall / 2, contrast)
        tens = sheet_tensor(Dd)
        for k in range(nt):
            rho = R - (k + 0.5) * wall / nt                                                             # (n,)
            Q = C[:, None] + rho[:, None, None] * Dd[None]
            W = (4 * math.pi * rho ** 2 / nd * (wall / nt))[:, None].expand(-1, nd)
            lum = ((R - wall).clamp_min(0) / R) ** 3
            desc = torch.stack([torch.full((n, nd), wall / 2, device=dev), torch.zeros(n, nd, device=dev), lum[:, None].expand(-1, nd),
                                torch.zeros(n, nd, device=dev), torch.zeros(n, nd, device=dev)], -1)
            V = torch.cat([tens[None].expand(n, -1, -1), desc, torch.full((n, nd, 1), contrast, device=dev)], -1)
            g.points.append((Q.reshape(-1, 3), W.reshape(-1), V.reshape(-1, 12), oid[:, None].expand(-1, nd).reshape(-1)))
        self.obstacles.append((C, R))
        return g

    def crystals(self):
        """Compact solids (polyhedra, shards, superquadrics), placed without overlap, each its own object: a blob
        population whose tensors are the solids' second moments (rendered as points: no device synchronisation)."""
        r = self.r; dev = self.device; rad = self.size(2, 200, 1.0, 8.0)
        if rad is None:
            return None
        prim = ["polyhedron", "shard", "superquadric"][r.i(0, 2)]
        ax = torch.tensor([r.lu(0.6, 2.0), r.lu(0.6, 2.0), r.lu(0.6, 3.0)], device=dev)
        if prim == "shard":
            ax[r.i(0, 2)] = r.u(0.15, 0.35)
        ax = ax / ax.prod() ** (1 / 3) * rad
        dmin = 2.05 * float(ax.max()) * 1.15
        n_c = int(min(3000, r.u(0.3, 1.2) * np.prod(self.shape) / dmin ** 3))
        C = r.rand(max(n_c, 1), 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        D = torch.cdist(C, C); earlier = torch.ones_like(D, dtype=torch.bool).tril(-1)
        C = C[~((D < dmin) & earlier).any(1)][:600]; n = C.shape[0]
        sc = torch.exp(0.15 * r.randn(n, 1)).clamp(0.75, 1.15)
        A = _rot(r, n) @ torch.diag_embed(ax[None] * sc)
        nrm = Fn.normalize(torch.tensor(r.np.normal(size=(r.i(5, 12), 3)), dtype=torch.float32, device=dev), dim=1) if prim == "polyhedron" else None
        exps = (r.lu(0.3, 2.0), r.lu(0.3, 2.0))
        g = self._solids(C, A, "crystals", self.ids(n, float(ax.min()) * sc.reshape(-1)), shape={"shard": "superquadric"}.get(prim, prim), exps=exps, normals=nrm)
        self.obstacles.append((C, sc.reshape(-1) * float(ax.max())))
        self.record["families"].append({"crystals": {"primitive": prim, "n": n, "radius_vox": round(rad, 2)}})
        return g

    # ---- the added families: plywood lamellae, fascicle hierarchies, trabeculae, tubules, epithelia
    def plywood(self):
        """Stacked lamellae of parallel fibres whose direction turns from one lamella to the next (orthogonal plywood,
        90 degrees; twisted / Bouligand, a constant small angle): lamellar bone, cornea, cuticle. Unresolved fibrils:
        each lamella a slab of line tensors; alternate lamellae are separate populations."""
        r = self.r; dev = self.device
        tl = self.size(2, 40, 1.5, min(self.shape) / 8)
        if tl is None:
            return None
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        rel = g_ - torch.tensor(self.shape, device=dev, dtype=torch.float32) / 2
        L = int(2 * max(self.shape)); hgt = noise2d(L, L, r.lu(15, 50), r) * r.u(0, 6)
        gu = (rel @ e1 / (L / 2)).clamp(-1, 1); gv = (rel @ e2 / (L / 2)).clamp(-1, 1)
        hs = Fn.grid_sample(hgt[None, None], torch.stack([gv, gu], -1).view(1, 1, -1, 2), align_corners=True).view(self.shape)
        depth = rel @ n0 - hs
        k = torch.floor(depth / tl); frac = depth / tl - k
        dth = math.radians(90.0 if r.p(0.5) else r.u(10, 45)); th0 = r.u(0, math.pi)
        theta = th0 + k * dth
        d = torch.cos(theta)[..., None] * e1 + torch.sin(theta)[..., None] * e2                     # (Z, Y, X, 3)
        gap = r.u(0.05, 0.2); edge = 0.5 / tl
        occ = (((frac - gap / 2) / edge).clamp(0, 1) * ((1 - gap / 2 - frac) / edge).clamp(0, 1)) * r.u(0.4, 0.85)
        tens = line_tensor(d).movedim(-1, 0)
        rf_um = r.lu(0.05, 1.0); rf = rf_um / self.voxel_um
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = rf
        contrast = self.contrast(); oid = self.ids(1, tl)
        out = []
        for par in (0, 1):
            m = (k.remainder(2) == par).float()
            out.append(Group("lamellae", "line", rf, contrast, dense=[(occ * m, tens, desc, contrast, torch.full(self.shape, int(oid[0]), device=dev, dtype=torch.long))]))
        self.record["families"].append({"plywood": {"lamella_vox": round(tl, 1), "turn_deg": round(math.degrees(dth), 1)}})
        return out

    def fascicles(self):
        """A hierarchy: fascicles (each a packed bundle of fibres, crimped) inside sheaths, packed side by side along a
        common axis -- tendon, muscle, nerve. Fibres -> fascicle -> the whole; the sheaths a hollow-tube population."""
        r = self.r; dev = self.device
        Rt = self.size(100, 6000, 4.0, min(self.shape) / 3)
        if Rt is None:
            return None
        nf = r.i(3, 12); Rf = Rt / math.sqrt(nf) * r.u(0.6, 0.85)
        d = r.unit(); c = self.centre_point(0.3)
        a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(d[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(d, a), dim=0); e2 = torch.linalg.cross(d, e1)
        # fascicle centres: a jittered hexagonal packing in the section
        sp = 2.15 * Rf; nn_ = int(Rt / sp) + 2
        ii, jj = torch.meshgrid(torch.arange(-nn_, nn_ + 1, device=dev), torch.arange(-nn_, nn_ + 1, device=dev), indexing="ij")
        o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp
        o = o[o.norm(dim=1) <= max(Rt - Rf, 0.0) + 1e-3][:nf]
        crimp_a, crimp_l = r.u(0, 0.15) * Rf, r.lu(4, 15) * Rf
        groups, sheath_curves = [], []
        wall = max(0.5, r.u(0.04, 0.12) * Rf); cs = self.contrast()
        tid = int(self.ids(1, Rt)[0])
        fib_curves = []
        for j in range(o.shape[0]):
            cen = c + o[j, 0] * e1 + o[j, 1] * e2
            g = self.bundle(None, "single", axis=d, centre=cen, radius=Rf - wall)
            if g is None or isinstance(g, list):
                continue
            # crimp: a sinusoidal wave of the whole fascicle in a random section direction
            ph = r.u(0, 6.28); wv = Fn.normalize(r.unit() - (r.unit() @ d) * d, dim=0)
            for cv in g.curves:
                sl = (cv.P - c) @ d
                cv.P = cv.P + (crimp_a * torch.sin(2 * math.pi * sl / crimp_l + ph))[..., None] * wv
                for o_ in cv.obj.tolist():
                    self.parents.setdefault(o_, tid)
            fib_curves += g.curves
            Pc = g.centreline.P + (crimp_a * torch.sin(2 * math.pi * ((g.centreline.P - c) @ d) / crimp_l + ph))[..., None] * wv
            sid = self.ids(1, Rf); self.parents[int(sid[0])] = tid
            sheath_curves.append(curves_like(Pc, torch.full_like(Pc[..., 0], Rf), r, obj=sid, seg=sid, contrast=cs, wall=torch.full_like(Pc[..., 0], wall)))
        if not fib_curves:
            return None
        out = [Group("fascicle_fibres", "line", float(np.median([float(cv.rad.median()) for cv in fib_curves])), self.contrast(), curves=fib_curves)]
        if sheath_curves:
            out.append(Group("sheaths", "line", Rf, cs, curves=sheath_curves))
        self.record["families"].append({"fascicles": {"n": len(sheath_curves), "fascicle_vox": round(Rf, 1), "crimp": round(crimp_a / Rf, 2)}})
        return out

    def trabecular(self):
        """Trabecular bone-like walls from two smooth level sets: plates (near the zero set of one: sheets) and rods (near
        the intersection of two: lines along the cross product of their gradients), in complementary regions."""
        r = self.r; dev = self.device
        cell = self.size(200, 2000, 8.0, min(self.shape) / 2)
        if cell is None:
            return None
        p1 = smooth_noise(self.shape, cell / 3, r.g, dev); p2 = smooth_noise(self.shape, cell / 3, r.g, dev)
        g1 = torch.stack(torch.gradient(p1)); g2 = torch.stack(torch.gradient(p2))
        n1 = g1.norm(dim=0).clamp_min(1e-6); n2 = g2.norm(dim=0).clamp_min(1e-6)
        d1 = p1.abs() / n1; d2 = p2.abs() / n2
        hp = max(0.6, r.u(0.04, 0.12) * cell); hr = max(0.7, r.u(0.05, 0.15) * cell)
        reg = gaussian(r.rand(*self.shape), cell / 2); reg = (reg - reg.mean()) / reg.std().clamp_min(1e-6)
        plate_share = r.u(0.2, 0.8); thr = float(torch.quantile(reg.flatten()[::97], plate_share))
        plate_reg = torch.sigmoid((thr - reg) * 6)
        occ_p = (0.5 + hp - d1).clamp(0, 1) * plate_reg
        occ_r = (0.5 + hr - torch.sqrt(d1 ** 2 + d2 ** 2)).clamp(0, 1) * (1 - plate_reg)
        tp = sheet_tensor((g1 / n1).movedim(0, -1)).movedim(-1, 0)
        tr = line_tensor(Fn.normalize(torch.linalg.cross(g1, g2, dim=0), dim=0).movedim(0, -1)).movedim(-1, 0)
        dp = torch.zeros((5,) + self.shape, device=dev); dp[0] = hp
        dr = torch.zeros((5,) + self.shape, device=dev); dr[0] = hr
        oid = self.ids(1, hp); cst = self.contrast(); om = torch.full(self.shape, int(oid[0]), device=dev, dtype=torch.long)
        self.record["families"].append({"trabecular": {"cell_vox": round(cell, 1), "plate_share": round(plate_share, 2)}})
        return [Group("plates", "sheet", hp, cst, dense=[(occ_p, tp, dp, cst, om)]), Group("rods", "line", hr, cst, dense=[(occ_r, tr, dr, cst, om)])]

    def tubules(self):
        """Convoluted tubules (kidney-like): many tortuous epithelial tubes with a thick wall and an open lumen."""
        r = self.r; dev = self.device
        rt = self.size(8, 60, 1.2, 8.0)
        if rt is None:
            return None
        n = int(min(120, r.u(0.2, 0.6) * np.prod(self.shape) / (math.pi * rt ** 2 * 2 * max(self.shape)) + 3))
        L = torch.full((n,), 1.5 * max(self.shape), device=dev); S = int(1.5 * max(self.shape) / H)
        start = r.rand(n, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        P = smooth_paths(start - 0.5 * L[:, None] * r.unit(n), r.unit(n), L, S, torch.full((n,), r.lu(0.05, 0.25) / rt ** 0.5, device=dev), r.lu(3, 8) * rt, r)
        rad = torch.full((n, S), rt, device=dev) * torch.exp(0.1 * r.randn(n, 1))
        cv = curves_like(P, rad, r, obj=self.ids(n, rad[:, 0]), seg=torch.zeros(n, dtype=torch.long, device=dev), contrast=self.contrast(),
                         wall=rad * r.u(0.3, 0.6), lumen_c=torch.full_like(rad, r.u(-50, 50)))
        cv = self.deflect(cv, extra=[(P[:, ::6].reshape(-1, 3), rad[:, ::6].reshape(-1))], self_ids=True)
        self.record["families"].append({"tubules": {"n": n, "radius_vox": round(rt, 2)}})
        return Group("tubules", "line", rt, float(cv.contrast[0, 0]), curves=[cv])

    def epithelium(self):
        """An epithelium: columnar cells packed on a folded surface (a basement membrane), elongated along its normal,
        with nuclei; the membrane a sheet population."""
        r = self.r; dev = self.device
        rc = self.size(3, 15, 1.2, 6.0)
        if rc is None:
            return None
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        c = self.centre_point(0.35); Lh = 0.75 * max(self.shape); sp = 2.1 * rc
        nu = int(2 * Lh / sp)
        ii, jj = torch.meshgrid(torch.arange(nu, device=dev), torch.arange(nu, device=dev), indexing="ij")
        uv = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp - Lh
        sf = r.lu(10, 40); amp = r.u(0, 0.3) * sf                                                   # folds up to ~60 deg steep
        kx, ky, ph = r.u(0.5, 1.5) / sf, r.u(0.5, 1.5) / sf, r.u(0, 6.28)
        hgt = amp * torch.sin(2 * math.pi * kx * uv[:, 0] + ph) * torch.cos(2 * math.pi * ky * uv[:, 1])
        dhu = amp * 2 * math.pi * kx * torch.cos(2 * math.pi * kx * uv[:, 0] + ph) * torch.cos(2 * math.pi * ky * uv[:, 1])
        dhv = -amp * 2 * math.pi * ky * torch.sin(2 * math.pi * kx * uv[:, 0] + ph) * torch.sin(2 * math.pi * ky * uv[:, 1])
        nrm = Fn.normalize(n0[None] - dhu[:, None] * e1 - dhv[:, None] * e2, dim=1)
        hc = rc * r.u(1.5, 4.0)                                                                     # half height of the columnar cells
        base = c + uv[:, :1] * e1 + uv[:, 1:] * e2 + hgt[:, None] * n0
        C = base + nrm * hc
        ok = ((C > -hc) & (C < torch.tensor(self.shape, device=dev) + hc)).all(1)
        C, nrm = C[ok][:700], nrm[ok][:700]; n = C.shape[0]
        if n == 0:
            return None
        t1 = Fn.normalize(torch.linalg.cross(nrm, e1[None].expand_as(nrm)), dim=1); t2 = torch.linalg.cross(nrm, t1)
        Rm = torch.stack([t1, t2, nrm], -1)                                                          # columns: in-plane, in-plane, normal
        ax = torch.stack([torch.full((n,), rc * 0.95, device=dev), torch.full((n,), rc * 0.95, device=dev), torch.full((n,), hc, device=dev)], 1)
        out = self._cell_groups(C, Rm, ax, rc)
        bm = self.sheet(normal=n0, centre=c, half_thickness=max(0.5, 0.15 * rc), height=(sf, amp, kx, ky, ph, e1, e2))
        if bm is not None:
            bm.name = "basement"; out.append(bm)
        self.record["families"].append({"epithelium": {"cells": n, "radius_vox": round(rc, 2), "height_vox": round(2 * hc, 1)}})
        return out

    # ---- fibres yield to cells and vessels
    def deflect(self, cv: Curves, iters: int = 4, margin: float = 0.5, extra=None, self_ids: bool = False):
        """Curves yield to obstacles (spheres ``(centres, radii)``): the scene's (cells, vessels) or ``extra`` ones only
        (a bundle kissing another; tubules among themselves, with ``self_ids`` -- ``extra`` sampled from these curves,
        every 6th point -- so no curve yields to itself)."""
        obs = extra if extra is not None else self.obstacles
        if not obs:
            return cv
        C = torch.cat([o[0] for o in obs]); R = torch.cat([o[1] for o in obs])
        P = cv.P.clone(); N, S, _ = P.shape
        rr = cv.rad.reshape(-1)
        owner_q = torch.arange(N, device=P.device).repeat_interleave(S)
        owner_c = torch.arange(N, device=P.device).repeat_interleave((S + 5) // 6) if self_ids else None
        for _ in range(iters):
            flat = P.reshape(-1, 3); push = torch.zeros_like(flat)
            if self_ids:
                C = P[:, ::6].reshape(-1, 3)
            rows = max(2048, int(1.5e8 / max(C.shape[0], 1)))                # ~0.6 GB per distance block, whatever the obstacle count
            for i0 in range(0, flat.shape[0], rows):
                q = flat[i0:i0 + rows]; d = torch.cdist(q, C)
                ov = (R[None] + rr[i0:i0 + rows, None] + margin - d).clamp_min(0)
                if self_ids:
                    ov = ov * (owner_q[i0:i0 + rows, None] != owner_c[None]).float()
                w = ov / d.clamp_min(1e-6)                                   # push = sum_c ov_c (q - C_c) / d_c
                push[i0:i0 + rows] = q * w.sum(1, keepdim=True) - w @ C
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
        chosen = [str(c_) for c_ in r.np.choice(names, size=n, p=p)]
        order = {"cells": 0, "crystals": 0, "epithelium": 0, "tree": 1, "gland": 1, "network": 2}   # obstacles first, fibres after
        chosen.sort(key=lambda k: order.get(k, 5))
        bundles = []
        for name in chosen:
            if name == "bundle":
                prev = bundles[-1] if bundles and r.p(0.6) else None
                mode = ["angled", "angled", "kissing", "fanning"][r.i(0, 3)] if prev is not None else ("split" if r.p(0.15) else "single")
                gs = self.bundle(prev.centreline if prev is not None else None, mode)
                if gs is not None:
                    for g_ in (gs if isinstance(gs, list) else [gs]):
                        if mode == "angled" and prev is not None and prev.offsets is not None and g_.offsets is not None and r.p(0.7):
                            self.interleave(prev, g_)                              # fascicles interdigitate where the bundles cross
                        if mode == "kissing" and prev is not None:                 # the newer bundle flattens against the other
                            cl = prev.centreline
                            g_.curves = [self.deflect(c, extra=[(cl.P[0, ::4], cl.rad[0, ::4])]) for c in g_.curves]
                        g_.curves = [self.deflect(c) for c in g_.curves]
                        bundles.append(g_)
            elif name == "foam":
                gs = self.foam(False)
            elif name == "porous":
                gs = self.foam(True)
            elif name in ("dendrite", "gland"):
                gs = self.tree(name)
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
        return {"groups": out, "ids": ids, "id_frac": frac, "lumen": lum, "lumen_c": lum_c, "points": maps,
                "point_lists": (torch.cat(self.branch) if self.branch else torch.zeros(0, 3, device=self.device),
                                torch.cat(self.ends) if self.ends else torch.zeros(0, 3, device=self.device))}


# ------------------------------------------------------------------------------------------------------------------
# a training pair: a crop of a rendered scene, composited, acquired
# ------------------------------------------------------------------------------------------------------------------

_CACHE: dict = {}
CACHE_SCENES = 3


def render_scene(canvas, scene_seed: int, device, voxel_um=None, families=None, n_groups=(1, 5), layout_p=0.3, content="general", style=None):
    """The scene (cached: crops of one scene are drawn in sequence) and its background on the whole canvas -- a
    texture, regions and possibly a cell stain, so crops at any zoom of one scene share them."""
    key = (tuple(canvas), scene_seed, str(device), voxel_um, str(families), tuple(n_groups), layout_p, content, style)
    if key in _CACHE:
        _CACHE[key] = _CACHE.pop(key)                                       # most recently used last
    else:
        while len(_CACHE) >= CACHE_SCENES:                                  # a few scenes (training draws and paired renders interleave)
            _CACHE.pop(next(iter(_CACHE)))
        sc = Scene(canvas, scene_seed, device, voxel_um, content).build(families, n_groups, layout_p)
        st = style or ["flat", "regions", "tissue", "cells"][sc.r.i(0, 3)]
        bgp = background(tuple(canvas), st, sc.r.g, device, 1.0, content, nprng=sc.r.np)
        _CACHE[key] = (sc, sc.render(), bgp, st)
    return _CACHE[key]


def _pool_fields(occ, fields: list, zoom: int):
    """Average ``occ (Z, Y, X)`` over ``zoom``³ blocks, and occupancy-weighted averages of ``fields`` (each ``(C, Z, Y, X)``):
    the same population rendered at a voxel ``zoom`` times larger (partial volume is linear)."""
    if zoom == 1:
        return occ, fields
    po = Fn.avg_pool3d(occ[None, None], zoom)[0, 0]
    out = [Fn.avg_pool3d((f * occ[None])[None], zoom)[0] / po.clamp_min(1e-9)[None] for f in fields]
    return po, out


def _crop(x, sl, zoom: int, mode: str = "mean"):
    """A crop of ``x (..., Z, Y, X)`` at slices ``sl`` reduced by ``zoom`` (``mean``: block average; ``pick``: the block's
    first voxel -- labels)."""
    x = x[(Ellipsis,) + tuple(sl)]
    if zoom == 1:
        return x
    if mode == "pick":
        return x[(Ellipsis,) + (slice(None, None, zoom),) * 3]
    lead = x.shape[:-3]
    return Fn.avg_pool3d(x.reshape(1, -1, *x.shape[-3:]).float(), zoom)[0].reshape(*lead, *[n // zoom for n in x.shape[-3:]])


def _point_map(pts: torch.Tensor, origin, zoom: int, shape, device) -> torch.Tensor:
    """Branch points / endpoints within a crop as unit-peak Gaussians (sigma 1 voxel of the crop)."""
    m = torch.zeros(shape, device=device)
    if pts.shape[0]:
        idx = ((pts - torch.tensor(origin, device=device, dtype=torch.float32)) / zoom).round().long()
        ok = ((idx >= 0) & (idx < torch.tensor(shape, device=device))).all(1)
        if bool(ok.any()):
            m.index_put_((idx[ok, 0], idx[ok, 1], idx[ok, 2]), torch.ones(int(ok.sum()), device=device), accumulate=True)
            m = gaussian(m, 1.0) * (2 * math.pi) ** 1.5
    return m


def _slice_profile(img, rng: Rng, rec: dict):
    """Native anisotropic acquisition: slices of thickness ``t`` (a Gaussian or box profile) every ``sp`` voxels along a
    random axis (a gap when ``sp > t``), at a random phase, interpolated back to the grid."""
    ax = rng.i(0, 2); t = rng.lu(1.5, 6.0); sp = t * rng.u(0.7, 1.6); n = img.shape[ax]
    x = img.movedim(ax, -1)
    if rng.p(0.5):                                                          # Gaussian profile (FWHM t)
        x = gaussian(x, t / 2.355, truncate=3.0, axes=(-1,))
        prof = "gauss"
    else:                                                                   # box profile
        k = max(1, int(round(t))); x = Fn.avg_pool1d(Fn.pad(x.reshape(-1, 1, n), (k // 2, (k - 1) // 2), mode="replicate"), k, stride=1).reshape(x.shape)
        prof = "box"
    pos = torch.arange(rng.u(0, sp), n - 1, sp, device=img.device)
    if pos.numel() < 2:
        return img
    i0 = pos.floor().long().clamp(0, n - 2); w = pos - i0
    samp = x[..., i0] * (1 - w) + x[..., i0 + 1] * w                         # (..., m) the acquired slices
    grid = torch.arange(n, device=img.device, dtype=torch.float32)
    j = torch.searchsorted(pos, grid).clamp(1, pos.numel() - 1)
    if rng.p(0.3):                                                          # nearest-slice resampling
        near = torch.where((grid - pos[j - 1]).abs() < (pos[j] - grid).abs(), j - 1, j)
        y = samp[..., near]
    else:
        ww = ((grid - pos[j - 1]) / (pos[j] - pos[j - 1])).clamp(0, 1)
        y = samp[..., j - 1] * (1 - ww) + samp[..., j] * ww
    rec["slice_profile"] = {"axis": ax, "thick": round(t, 2), "spacing": round(sp, 2), "profile": prof}
    return y.movedim(-1, ax).contiguous()


def _extra_acquisition(img, layers, rng: Rng, modality: str, rec: dict):
    """Artefacts the acquisition model lacks: native slice profiles (MRI, histology), histology tears (tissue missing
    along a crack, labels removed with it) and folds (a band of a section doubled), local-tomography truncation (low
    frequencies lost) for HiP-CT / CT."""
    shp = img.shape
    if modality in ("mri", "histology", "mixed") and rng.p(0.35):
        img = _slice_profile(img, rng, rec)
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


def _log_desc(d5: torch.Tensor) -> torch.Tensor:
    """The scene's 5 descriptors (size, wall in voxels; lumen fraction, flattening, angularity) as learned:
    ``log2(1 + size)``, the wall relative to the size, then the three fractions."""
    return torch.stack([torch.log2(1 + d5[0].clamp_min(0)), (d5[1] / d5[0].clamp_min(1e-3)).clamp(0, 1), d5[2], d5[3], d5[4]])


def draw_scene(shape=(64, 64, 64), seed: int = 0, device="cuda", modality: str | None = None, strength: float = 1.0,
               canvas: int = 128, crops_per_scene: int = 8, voxel_um: float | None = None, families: dict | None = None,
               n_groups=(1, 5), layout_p: float = 0.3, content: str = "general", style: str | None = None, second_contrast: bool = False,
               zoom: int = 1, physics: str | None = None, p_physics: float = 0.75, p_warp: float = 0.5, acq_seed: int | None = None,
               crop: tuple | None = None, identity_min_r: float = 1.5, p_blend: float = 0.35, p_aniso: float = 0.25, p_scramble: float = 0.3):
    """A training pair from an object scene, in the format of :func:`~shapetensor.learn.generator.draw`: ``(Sample,
    drawn)``. Scenes are rendered on a ``canvas``³ grid and ``crops_per_scene`` consecutive seeds share one (cut at
    different places), so a scene's cost is spread over its crops.

    * ``zoom``: the crop covers ``zoom`` times the extent, every field block-averaged (the same scene at a voxel
      ``zoom`` times larger -- multi-scale crops; paired renders with :func:`draw_pair`);
    * ``physics``: the image formation (:mod:`~shapetensor.learn.physics`; None: drawn for the modality with
      probability ``p_physics``, else additive);
    * ``p_warp``: an elastic warp that moves the labels, identities, points and descriptors with the image;
    * ``acq_seed``: the acquisition's seed (modality, physics, materials, nuisance) apart from the crop's;
    * contrast routes at image formation: ``p_blend`` -- a blend with a second physics (either polarity), ``p_aniso`` -- an
      orientation-dependent term (``sin^2`` or ``cos^2`` of each population's angle to a random direction, its own
      amplitude and sign) under any physics, ``p_scramble`` -- each population and background region its own offset.
      With :mod:`~shapetensor.learn.augment` at batch time, one geometry reaches the network through many contrasts.

    ``drawn`` also carries ``ids``/``id_frac`` (the two heaviest objects per voxel), ``identity`` (the object at the level
    the resolution separates: smaller than ``identity_min_r`` voxels -> its parent), ``descriptors5`` (per layer:
    log2(1 + size), relative wall, lumen fraction, flattening, angularity), ``desc_um``, ``points`` (branch points,
    endpoints), ``recoverability``, ``parents``, the voxel size and the physics."""
    sid = seed // max(1, crops_per_scene)
    sc, ren, bgp, st = render_scene((canvas,) * 3, 7_000_000 + sid, device, voxel_um, families, n_groups, layout_p, content, style)
    r = Rng(seed, device); ra = Rng(acq_seed if acq_seed is not None else seed + 31337, device)
    names = ["hipct", "ct", "mri", "histology", "mixed"]; modality = modality or names[ra.i(0, 4)]
    ext = [s_ * zoom for s_ in shape]
    assert all(e <= canvas for e in ext), "the zoomed crop must fit the canvas"
    o = list(crop) if crop is not None else [r.i(0, canvas - e) for e in ext]
    sl = [slice(a, a + e) for a, e in zip(o, ext)]
    shape = tuple(shape)
    # the background, cut from the scene's
    bg0, lab0, R0, extra0 = bgp
    bg = _crop(bg0, sl, zoom); R = _crop(R0, sl, zoom); lab = _crop(lab0, sl, zoom, "pick")
    extra = []
    for E in extra0:
        oc, (tc,) = _pool_fields(E.occupancy[tuple(sl)], [E.tensor[(slice(None),) + tuple(sl)]], zoom)
        L = Layer(E.kind, torch.zeros(3, 1, 1, 1, device=device), E.radius / zoom, E.length / zoom, 0.0, torch.ones(shape, dtype=torch.bool, device=device),
                  E.contrast, occupancy=oc.clone(), tensor=tc.clone())
        L.desc = torch.tensor([E.radius / zoom, 0.0, 0.0], device=device).view(3, 1, 1, 1) * (oc > 0)[None]
        extra.append(L)
    layers = []
    for res in ren["groups"]:
        g_ = res["group"]
        occ, (tens, d5, cm) = _pool_fields(res["occupancy"][tuple(sl)], [res["tensor"][(slice(None),) + tuple(sl)], res["desc"][(slice(None),) + tuple(sl)],
                                                                          res["contrast"][tuple(sl)][None]], zoom)
        if float(occ.max()) <= 1e-3:
            continue
        if zoom > 1:
            d5 = d5.clone(); d5[0] = d5[0] / zoom; d5[1] = d5[1] / zoom                    # sizes in the crop's voxels
        L = Layer(g_.kind, torch.zeros(3, 1, 1, 1, device=device), max(g_.radius / zoom, 0.5), 2 * max(g_.radius / zoom, 0.5), 0.0,
                  torch.ones(shape, dtype=torch.bool, device=device), g_.contrast, occupancy=occ.clone(), tensor=tens.clone())
        L.desc = torch.stack([d5[0], d5[2], d5[4]])                          # the layer descriptors: thickness, hollowness, angularity
        L.desc5 = d5.clone(); L.cmap = cm[0]; L.name = g_.name
        layers.append(L)
    shapes_ = layers + extra
    img = _composite(bg, R, shapes_, device)                                 # normalises the occupancies, adds the interface and isotropic states
    lum = _crop(ren["lumen"], sl, zoom); lum_c = _crop(ren["lumen"] * ren["lumen_c"], sl, zoom) / lum.clamp_min(1e-9)
    rec = {"scene": sid, "crop": o, "zoom": zoom, "style": st, **sc.record}
    mode = physics or (ra.np.choice(PHYS_BY_MODALITY[modality]) if ra.p(p_physics) else "additive")
    n_sh = len(layers) + len(extra)
    filled = sum((L.occupancy for L in shapes_[:n_sh]), torch.zeros(shape, device=device))
    occs = torch.stack([L.occupancy for L in shapes_[:n_sh]] + [torch.minimum(lum, (1 - filled).clamp_min(0))])
    tens = torch.stack([L.tensor for L in shapes_[:n_sh]] + [eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3])
    texture = (bg - bg.mean()) / bg.std().clamp_min(1e-6)
    def physics_image(m):
        if m == "additive":
            out = img.clone()
            for L in layers:                                                 # contrast varying along objects (striation, profiles)
                out = out + (L.cmap - L.contrast) * L.occupancy
            return out + lum * lum_c
        return form_image(m, occs, tens, texture, [draw_material(ra.np, m) for _ in range(occs.shape[0])],
                          draw_material(ra.np, m, background=True), draw_sequence(ra.np, m))
    img_main = physics_image(mode)
    routes = [mode]
    if ra.p(p_blend):                                                        # physics as a continuum: a blend with another (either polarity)
        m2 = str(ra.np.choice([m_ for m_ in PHYS_MODES if m_ != mode])); w = ra.u(0.3, 0.9)
        a = _grey(img_main); b = _grey(physics_image(m2)); b = 255 - b if ra.p(0.5) else b
        img_main = w * a + (1 - w) * b; routes.append(f"{m2}:{1 - w:.2f}")
    if ra.p(p_aniso):                                                        # orientation-dependent contrast under any physics
        gd = ra.unit(); gq = torch.stack([tens[k, 0] * gd[0] ** 2 + tens[k, 1] * gd[1] ** 2 + tens[k, 2] * gd[2] ** 2
                                          + 2 * (tens[k, 3] * gd[0] * gd[1] + tens[k, 4] * gd[0] * gd[2] + tens[k, 5] * gd[1] * gd[2]) for k in range(n_sh)]) if n_sh else None
        if gq is not None:
            amp = torch.tensor([ra.u(10, 80) * (1 if ra.p(0.5) else -1) for _ in range(n_sh)], device=device).view(-1, 1, 1, 1)
            sin2 = (1 - gq).clamp(0, 1) if ra.p(0.5) else gq.clamp(0, 1)
            img_main = _grey(img_main) + (amp * occs[:n_sh] * sin2).sum(0); routes.append("aniso")
    if ra.p(p_scramble):                                                     # each population and region its own offset, on top of the physics
        g0 = _grey(img_main)
        off = sum((ra.u(5, 60) * (1 if ra.p(0.5) else -1)) * occs[k] for k in range(n_sh)) if n_sh else 0.0
        nl = int(lab.max()) + 1; reg = torch.tensor([ra.u(-40, 40) for _ in range(nl)], device=device)[lab]
        img_main = g0 + off + reg; routes.append("scramble")
    img = img_main
    rec["physics"] = mode; rec["routes"] = routes
    img = _extra_acquisition(img, layers, ra, modality, rec)
    s = Sample(img, shapes_, bg, lab, st, 1.0, content, rec, R)
    d = nuisance(s, seed=acq_seed if acq_seed is not None else seed, device=device, strength=strength, modality=modality, warp=ra.p(p_warp), fast=True, hp_max=12.0,
                 vis_opts={"per_layer_hp": True, "local_corr": True})
    ids = _crop(ren["ids"], sl, zoom, "pick"); frac = _crop(ren["id_frac"], sl, zoom, "pick")
    pts = torch.stack([_point_map(pl, o, zoom, shape, device) for pl in ren["point_lists"]])
    D5 = torch.zeros(len(s.layers), 5, *shape, device=device)
    for k, L in enumerate(s.layers):
        if hasattr(L, "desc5"):
            D5[k] = _log_desc(L.desc5)
    grid = d.get("warp_grid")
    if grid is not None:                                                     # the warp moves the identities, points and descriptors too
        from .generator import _resample
        ids = _resample(ids.float(), grid, mode="nearest").round().long(); frac = _resample(frac, grid)
        pts = _resample(pts, grid)
        D5 = _resample(D5.reshape(-1, *shape), grid).reshape(D5.shape)
        D5[:, 0] = torch.log2(1 + (2 ** D5[:, 0] - 1) * d["stretch"][None])     # sizes scale with the local stretch
    lut = sc.identity_lut(identity_min_r)
    d["identity"] = torch.where(ids[0] >= 0, lut[ids[0].clamp(0, lut.numel() - 1)], ids[0])
    d["ids"], d["id_frac"], d["points"], d["descriptors5"] = ids, frac, pts, D5
    vu = sc.voxel_um * zoom
    d["desc_um"] = [torch.stack([L.desc5[0] * vu, L.desc5[1] * vu, L.desc5[2], L.desc5[3], L.desc5[4]]) for L in layers]
    d["recoverability"] = d["orientation_visibility"]; d["parents"] = sc.parents; d["voxel_um"] = vu
    d["record"].update({k: v for k, v in rec.items() if k not in d["record"]})
    if second_contrast:                                                      # a second contrast of the same scene (multi-contrast MRI)
        img2 = bg * 0.5 + 60
        for L in layers:
            img2 = img2 + ra.u(-120, 120) * L.occupancy
        d["image2"] = img2
    return s, d


def draw_pair(shape=(64, 64, 64), seed: int = 0, device="cuda", kind: str = "zoom", **kw):
    """Two renders of one scene for consistency: ``zoom`` -- the crop and the region twice its extent around it at
    half the resolution (the fine crop's lattice cells 2³ -> one coarse cell; ``offset``: the coarse cell of the fine
    crop's first cell pair); ``modality`` -- the same crop under another acquisition (modality, physics, materials,
    noise). Returns ``(s1, d1), (s2, d2), info``."""
    canvas = kw.get("canvas", 128); r = Rng(seed + 99991, device)
    if kind == "zoom":
        ext = [2 * n for n in shape]; assert all(e <= canvas for e in ext)
        oc = [8 * r.i(0, (canvas - e) // 8) for e in ext]                    # the coarse crop's origin (multiple of 8: lattice-aligned)
        of = [a + 8 * r.i(0, n // 8) for a, n in zip(oc, shape)]             # the fine crop inside it
        a = draw_scene(shape, seed, device, crop=of, zoom=1, **kw)
        b = draw_scene(shape, seed, device, crop=oc, zoom=2, acq_seed=seed + 4242, **kw)
        return a, b, {"kind": "zoom", "offset_cells": [(f - c) // 8 for f, c in zip(of, oc)], "scale": 2}
    o = [r.i(0, canvas - n) for n in shape]
    a = draw_scene(shape, seed, device, crop=o, **kw)
    b = draw_scene(shape, seed, device, crop=o, acq_seed=seed + 4242, **kw)
    return a, b, {"kind": "modality", "offset_cells": [0, 0, 0], "scale": 1}
