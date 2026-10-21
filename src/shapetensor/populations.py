"""Flexible-N populations in the shape space, with population-specific momentum. Classical statistics and linear
filtering on anatomix-dev-vit features (the design requirement: an adaptive number of populations per voxel, smoothed
along each population's own direction).

On the 80 µm lattice of :mod:`shapetensor.regions` (dev-vit decoder 1 and output, free of tile seams and patch grid):

1. **Orientation distributions as the calibration's targets.** Every calibration sample carries a distribution over
   ``n_dirs`` hemisphere directions (a kernel of ``kappa_deg`` about each of its directions, its total the sample's
   fibre mass): cells where the excised structure tensor is confident -- one peak at its direction; synthetic
   crossings -- two peaks, at their known directions, half the mass each; synthetic fibre-free texture -- no mass.
2. **Each cell's distribution = the mean of its neighbours' targets** (``k`` nearest in a Fisher discriminant space of
   the calibration classes: direction bins, crossings, texture), zero if the cell lies outside every region (its k-th
   neighbour beyond the radius the calibration's own samples reach from other regions). Non-parametric regression: no
   network, no fixed number of clusters.
3. **Momentum** (:func:`momentum`): each direction's share of the distribution field is smoothed along that direction
   (a Gaussian of ``sigma_along`` cells along ``u``, after a small angular smoothing so bends are followed), never above
   its *support* -- the cell's own value or, where the direction holds on both sides within ``reach`` cells, the
   weaker side's (a crossing or a partial-volume dip is bridged; a free end is not extended).
4. **Populations = the distribution's lobes** (:func:`shapetensor.classical.lobes`): up to ``nmax`` peaks above ``rel``
   of the largest and above ``floor`` of one population's mass, at least ``min_sep_deg`` apart -- N per cell from the
   data; each lobe's second moment is the population's tensor, its mass the weight.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .classical import lobes
from .merge import multi_field, single_field
from .pipeline import teacher
from .regions import _cell_dyads, _knn, lattice_features
from .separate import hemisphere, sample_idx
from .tensors import direction_and_gap


def kernel_odf(dirs: torch.Tensor, weights: torch.Tensor, grid: torch.Tensor, kappa_deg: float) -> torch.Tensor:
    """``dirs (n, m, 3)`` and ``weights (n, m)`` -> distributions ``(n, n_dirs)``: each direction a normalised kernel on
    the grid, scaled by its weight (so a distribution sums to its sample's fibre mass)."""
    k = 1.0 / (2 * math.sin(math.radians(kappa_deg)) ** 2)
    K = torch.exp(k * ((torch.einsum("nmc,dc->nmd", dirs, grid)) ** 2 - 1))                        # (n, m, n_dirs)
    K = K / K.sum(-1, keepdim=True).clamp_min(1e-12)
    return (K * weights[..., None]).sum(1)


def _line_sampler(cs, device):
    Z, Y, X = cs
    base = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32) for n in cs), indexing="ij"))
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=device).view(3, 1, 1, 1)

    def at(f: torch.Tensor, u: torch.Tensor, t: float) -> torch.Tensor:
        """Scalar field ``f (Z, Y, X)`` at ``x + t u`` for a fixed direction ``u (3,)`` (zeros outside)."""
        p = (base + t * u.view(3, 1, 1, 1)) * scale - 1.0
        g = torch.stack([p[2], p[1], p[0]], -1)[None]
        return Fn.grid_sample(f[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    return at


def momentum(O: torch.Tensor, grid: torch.Tensor, *, sigma_along: float = 3.0, reach: int = 6, angular_deg: float = 10.0) -> torch.Tensor:
    """Population-specific smoothing of distributions ``O (n_dirs, Z, Y, X)`` (cells). For each direction ``u``, after a
    small angular smoothing over directions (~``angular_deg``):

    * ``sm``: the field smoothed along ``u`` (Gaussian ``sigma_along`` cells, truncated at ``reach``) -- momentum within
      the population;
    * ``bridge``: the weaker of the two sides' largest value within ``reach`` cells along ``u`` (before the smoothing,
      so its tail cannot fake a side) -- the level the population holds on *both* sides (zero at a free end);
    * the result ``min(max(sm, bridge), max(O, bridge))``: a dip between two holding sides (a crossing, a partial-volume
      gap) is lifted to the bridge; a spike is smoothed; beyond a free end the smoothing's tail is cut by the cell's own
      (zero) value, so nothing is extended."""
    nd = O.shape[0]; cs = tuple(O.shape[1:])
    if angular_deg > 0:
        k = 1.0 / (2 * math.sin(math.radians(angular_deg)) ** 2)
        Ka = torch.exp(k * ((grid @ grid.T) ** 2 - 1)); Ka = Ka / Ka.sum(1, keepdim=True)
        Oa = (Ka @ O.reshape(nd, -1)).reshape(O.shape)
    else:
        Oa = O
    at = _line_sampler(cs, O.device)
    ts = list(range(-reach, reach + 1))
    w = torch.tensor([math.exp(-0.5 * (t / sigma_along) ** 2) for t in ts], device=O.device); w = w / w.sum()
    out = torch.empty_like(O)
    for d in range(nd):
        u = grid[d]; f = Oa[d]
        sm = torch.zeros(cs, device=O.device)
        for wi, t in zip(w, ts):
            sm += wi * (f if t == 0 else at(f, u, float(t)))
        fw = torch.zeros(cs, device=O.device); bw = torch.zeros(cs, device=O.device)
        for t in range(1, reach + 1):
            fw = torch.maximum(fw, at(f, u, float(t))); bw = torch.maximum(bw, at(f, u, -float(t)))
        bridge = torch.minimum(fw, bw)
        out[d] = torch.minimum(torch.maximum(sm, bridge), torch.maximum(O[d], bridge))
    return out


class PopulationModel:
    def __init__(self, k: int = 15, n_pcs: int = 50, reject_q: float = 0.99, gap_q: float = 0.7, per_block: int = 4000, seed: int = 0,
                 sigma: float = 1.0, rho: float = 10.0, truncate: float = 3.0, margin: int = 32, n_dirs: int = 90, kappa_deg: float = 12.0,
                 n_bins: int = 32, n_lda: int = 24, group_cells: int = 32):
        self.k, self.n_pcs, self.reject_q, self.gap_q, self.per_block, self.seed = int(k), int(n_pcs), float(reject_q), float(gap_q), int(per_block), int(seed)
        self.sigma, self.rho, self.truncate, self.margin = sigma, rho, truncate, int(margin)
        self.n_dirs, self.kappa_deg, self.n_bins, self.n_lda, self.group_cells = int(n_dirs), float(kappa_deg), int(n_bins), int(n_lda), int(group_cells)

    def fit(self, eng, volumes, tissues=None, crossings=(), iso=()):
        """``volumes``/``tissues``: blocks whose confident cells are single-population samples; ``crossings``:
        ``(volume, d1, d2)`` synthetic crossings with their two directions (z, y, x); ``iso``: fibre-free texture."""
        dev = volumes[0].device; g = torch.Generator(device="cpu").manual_seed(self.seed)
        tissues = tissues or [None] * len(volumes)
        m = self.margin // 4; gc_ = self.group_cells
        self.grid = hemisphere(self.n_dirs, dev); bins = hemisphere(self.n_bins, dev)

        def groups(cs, base):
            zz, yy, xx = (torch.arange(n, device=dev) // gc_ for n in cs)
            ny, nx = -(-cs[1] // gc_), -(-cs[2] // gc_)
            return (base + (zz[:, None, None] * ny + yy[None, :, None]) * nx + xx[None, None, :]).flatten()

        def inner(cs):
            t = torch.zeros(cs, dtype=torch.bool, device=dev); t[m:cs[0] - m, m:cs[1] - m, m:cs[2] - m] = True
            return t

        X, DIR, WT, CLS, GRP = [], [], [], [], []; base = 0
        cells = []
        for v, t in zip(volumes, tissues):                                                        # single-population samples
            L = lattice_features(eng, v); cs = L.shape[1:]
            d, gap = teacher(v, self.sigma, self.rho, self.truncate)
            w = torch.ones_like(gap) if t is None else t.float()
            dc, _ = _cell_dyads(d, w * gap, cs)
            pad = [0, (-gap.shape[2]) % 4, 0, (-gap.shape[1]) % 4, 0, (-gap.shape[0]) % 4]
            gcell = Fn.avg_pool3d(Fn.pad((gap * w)[None, None], pad), 4)[0, 0][:cs[0], :cs[1], :cs[2]]
            ok = inner(cs)
            if t is not None:
                ok &= Fn.avg_pool3d(Fn.pad(t.float()[None, None], pad), 4)[0, 0][:cs[0], :cs[1], :cs[2]] > 0.5
            cells.append((L, dc, gcell, ok))
        allg = torch.cat([gcell[ok] for _, _, gcell, ok in cells])
        thr = float(torch.quantile(allg[torch.randperm(allg.numel(), generator=g)[:2_000_000].to(dev)], self.gap_q))
        for L, dc, gcell, ok in cells:
            idx = sample_idx((ok & (gcell >= thr)).flatten().float(), self.per_block, g)
            grp = groups(L.shape[1:], base); base = int(grp.max()) + 1
            X.append(L.reshape(L.shape[0], -1)[:, idx].T)
            dd = dc.reshape(3, -1)[:, idx].T
            DIR.append(torch.stack([dd, torch.zeros_like(dd)], 1)); WT.append(torch.tensor([1.0, 0.0], device=dev).expand(len(idx), 2))
            CLS.append((dd @ bins.T).abs().argmax(1)); GRP.append(grp[idx])
        for j, (v, d1, d2) in enumerate(crossings):                                              # synthetic crossings: two directions
            L = lattice_features(eng, v); cs = L.shape[1:]
            idx = sample_idx(inner(cs).flatten().float(), self.per_block, g)
            grp = groups(cs, base); base = int(grp.max()) + 1
            X.append(L.reshape(L.shape[0], -1)[:, idx].T)
            D2 = torch.stack([torch.as_tensor(d1, device=dev, dtype=torch.float32), torch.as_tensor(d2, device=dev, dtype=torch.float32)])
            DIR.append(D2[None].expand(len(idx), 2, 3)); WT.append(torch.tensor([0.5, 0.5], device=dev).expand(len(idx), 2))
            CLS.append(torch.full((len(idx),), self.n_bins + j, device=dev)); GRP.append(grp[idx])
        for t in iso:                                                                              # texture: no mass
            L = lattice_features(eng, t); cs = L.shape[1:]
            idx = sample_idx(inner(cs).flatten().float(), self.per_block, g)
            grp = groups(cs, base); base = int(grp.max()) + 1
            X.append(L.reshape(L.shape[0], -1)[:, idx].T)
            DIR.append(torch.zeros(len(idx), 2, 3, device=dev)); WT.append(torch.zeros(len(idx), 2, device=dev))
            CLS.append(torch.full((len(idx),), self.n_bins + len(crossings), device=dev)); GRP.append(grp[idx])
        X = torch.cat(X); self.targets = kernel_odf(torch.cat(DIR), torch.cat(WT), self.grid, self.kappa_deg)
        cls = torch.cat(CLS); self.group = torch.cat(GRP)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xs = (X - self.mean) / self.std
        _, S, V = torch.pca_lowrank(Xs, q=min(self.n_pcs, Xs.shape[1]), center=False, niter=4)
        self.V = V[:, :self.n_pcs]; Y = Xs @ self.V
        # Fisher discriminant over the calibration classes (direction bins, each crossing, texture)
        mu = Y.mean(0); Sw = torch.zeros(Y.shape[1], Y.shape[1], device=dev); Sb = torch.zeros_like(Sw)
        for c in cls.unique():
            Yc = Y[cls == c]; mc = Yc.mean(0)
            Sw += (Yc - mc).T @ (Yc - mc); Sb += len(Yc) * torch.outer(mc - mu, mc - mu)
        Sw = Sw / len(Y); Sw = Sw + 1e-3 * torch.trace(Sw) / Sw.shape[0] * torch.eye(Sw.shape[0], device=dev); Sb = Sb / len(Y)
        w_, Vw = torch.linalg.eigh(Sw.double()); Wi = (Vw @ torch.diag(w_.clamp_min(1e-12) ** -0.5) @ Vw.T).float()
        _, Ve = torch.linalg.eigh((Wi @ Sb @ Wi).double())
        self.Lm = Wi @ Ve[:, -self.n_lda:].float()
        self.Y = Y @ self.Lm
        _, dist = _knn(self.Y, self.Y, self.k, q_group=self.group, y_group=self.group)
        self.radius = float(torch.quantile(dist[:, -1], self.reject_q))
        self.report = {"single samples": int((cls < self.n_bins).sum()), "crossing samples": int(((cls >= self.n_bins) & (cls < self.n_bins + len(crossings))).sum()),
                       "texture samples": int((cls == self.n_bins + len(crossings)).sum()), "gap threshold": round(thr, 4), "radius": round(self.radius, 3)}
        return self

    def odf(self, L: torch.Tensor):
        """Lattice features ``(C, z, y, x)`` -> distributions ``(n_dirs, z, y, x)`` (the neighbours' mean target, zero
        outside every region) and the distance ratio ``(z, y, x)``."""
        cs = L.shape[1:]
        Q = (((L.reshape(L.shape[0], -1).T - self.mean) / self.std) @ self.V) @ self.Lm
        idx, dist = _knn(Q, self.Y, self.k)
        O = self.targets[idx].mean(1)                                                              # (M, n_dirs)
        ratio = dist[:, -1] / self.radius
        O = O * (ratio <= 1.0).float()[:, None]
        return O.T.reshape((self.n_dirs,) + tuple(cs)).contiguous(), ratio.reshape(cs)

    def populations(self, eng, v: torch.Tensor, *, use_momentum: bool = True, nmax: int = 3, rel: float = 0.25, floor: float = 0.15,
                    min_sep_deg: float = 25.0, sigma_along: float = 3.0, reach: int = 6, angular_deg: float = 10.0) -> dict:
        """The block's populations: per cell the distribution (after momentum), N, the lobes' directions and masses; and at
        every voxel the merged field and the multi-vector field (``vecs`` / ``weights``, largest first)."""
        L = lattice_features(eng, v)
        O, ratio = self.odf(L)
        if use_momentum:
            O = momentum(O, self.grid, sigma_along=sigma_along, reach=reach, angular_deg=angular_deg)
        S, mass = lobes(O, self.grid, nmax=nmax, rel=rel, min_sep_deg=min_sep_deg)
        mass = torch.where(mass >= floor, mass, torch.zeros_like(mass))
        N = (mass > 0).sum(0)
        dirs_c = torch.stack([direction_and_gap(S[r])[0] for r in range(nmax)])
        size = tuple(int(n) for n in v.shape)
        up = lambda x: Fn.interpolate(x[None] if x.dim() == 4 else x[None, None], scale_factor=4, mode="trilinear", align_corners=False)[0][..., :size[0], :size[1], :size[2]]
        Sf = torch.stack([up(S[r]) for r in range(nmax)])
        mf = torch.stack([up(mass[r][None])[0] for r in range(nmax)]).clamp_min(0)
        dirs = torch.stack([direction_and_gap(Sf[r])[0] for r in range(nmax)])
        A = mf / mf.max(0).values.clamp_min(1e-12)[None]
        vec, _, weight, T = single_field(Sf, A, floor=0.1, return_tensor=True)
        vecs, weights = multi_field(dirs, mf)
        return {"odf": O, "N": N, "mass": mass, "dirs_cells": dirs_c, "ratio": ratio, "vec": vec, "vecs": vecs, "weights": weights, "weight": mf.sum(0)}
