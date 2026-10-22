"""The chain, and the field at the input's resolution.

:class:`Orientations` runs lift → presence (null from the block's isotropic surrogate) → deconvolution (response from
the block itself) → carry-forward along each direction → lobes → sheets → identities, on a lattice of ``step`` voxels,
then reads the lifted field at every input voxel (:meth:`Orientations.field`). One scale parameter: ``scale`` is the
window (in voxels) that every other length is tied to.

Outputs (vectors in (z, y, x) component order unless stated; nothing is masked -- the weights say where a population
is present):

* ``vecs (nmax, 3, Z, Y, X)``, ``weights (nmax, Z, Y, X)``, ``labels (nmax, Z, Y, X)``: the co-located vectors, largest
  first, each with its weight (zero where absent: below the presence floor, or a sheet) and its population identity;
* ``vec (3, Z, Y, X)``, ``weight (Z, Y, X)``: the strongest direction everywhere, with the presence weight -- the
  single field, never cut;
* ``N (Z, Y, X)``, ``sheet``; and on the lattice ``fodf``, ``dirs_cells``, ``mass``, ``presence``, ``labels_cells``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from .lift import (angular_smooth, deconvolve, hemisphere, iso_surrogate, lift, presence, response_matrix, self_response, shell_lags)
from .populations import label_populations, lobe_ring_ratio, momentum, peaks, ring_fraction


class Orientations:
    """The pipeline's parameters, all tied to ``scale`` (the window σ in voxels) unless given; see the module notes."""

    def __init__(self, *, scale: float = 10.0, n_dirs: int = 160, step: int | None = None, lags=(0.25, 0.65), kernel_deg: float = 8.0,
                 mean_scale: float = 0.3, fine_scale: float = 0.0, excise: float = 3.0, radial: str = "count",
                 nmax: int = 2, min_sep_deg: float = 30.0, rel: float = 0.3, min_share: float = 0.15,
                 momentum: bool = True, reach: int | None = None, sigma_along: float | None = None, momentum_angular_deg: float = 0.0,
                 valley: float = 0.0, angular_deg: float = 6.0, bin_deg: float = 2.5, iters: int = 300,
                 max_ring: float = 0.35, min_prominence: float = 5.0, ring_rel: float = 1.0, ring_on: str = "fodf", rings: bool = False, max_sheet_share: float = 0.5, presence_mode: str = "excess",
                 null_q: float = 0.99, full_factor: float = 2.0, min_weight: float = 0.2, null_crop: int = 256, null: str = "isotropic",
                 label_angle_deg: float = 20.0, min_fixels: int = 8, device: str = "cuda"):
        self.scale = float(scale); self.device = device
        self.grid = hemisphere(n_dirs, device)
        self.step = int(step) if step is not None else max(1, int(round(self.scale / 2.5)))
        self.origin = self.step // 2
        self.r_min, self.r_max = lags[0] * self.scale, lags[1] * self.scale
        self.lags = shell_lags(self.r_min, self.r_max)
        self.sigma_m = float(mean_scale) * self.scale; self.sigma_f = float(fine_scale) * self.scale; self.kernel_deg = float(kernel_deg); self.excise = float(excise)
        self.radial = str(radial)
        self.nmax, self.min_sep_deg, self.rel, self.min_share = int(nmax), float(min_sep_deg), float(rel), float(min_share)
        self.momentum = bool(momentum); self.valley = float(valley); self.momentum_angular_deg = float(momentum_angular_deg)
        self.reach = int(reach) if reach is not None else max(1, int(round(2.4 * self.scale / self.step)))
        self.sigma_along = float(sigma_along) if sigma_along is not None else max(0.5, 1.2 * self.scale / self.step)
        self.angular_deg, self.bin_deg, self.iters = float(angular_deg), float(bin_deg), int(iters)
        self.max_ring, self.min_prominence, self.ring_rel, self.presence_mode = float(max_ring), float(min_prominence), float(ring_rel), str(presence_mode)
        self.ring_on = str(ring_on); self.rings = bool(rings); self.max_sheet_share = float(max_sheet_share)
        self.null_q, self.full_factor, self.min_weight, self.null_crop = float(null_q), float(full_factor), float(min_weight), int(null_crop)
        if null not in ("isotropic", "spectrum"):
            raise ValueError("null must be 'isotropic' or 'spectrum'")
        self.null = null; self.bias = None
        self.label_angle_deg, self.min_fixels = float(label_angle_deg), int(min_fixels)
        self.edge = self.r_max + 2 * self.scale                                                     # voxels too near a face to trust
        self.response = None; self.last_floor = None

    # -- the lattice stages ---------------------------------------------------------------------------------------

    def lift(self, v: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """The lift of ``v``; with ``excise`` > 0 and no ``weights`` given, voxels whose coarse gradient exceeds ``excise``
        x the median (a boundary, a vessel wall) are left out of the correlation (:func:`shapetensor.st.excision_weights`)."""
        v = v.to(self.device)
        if weights is None and self.excise > 0:
            from .st import excision_weights
            weights = excision_weights(v if v.dim() == 3 else v[0], factor=self.excise)
        return lift(v, self.grid, r_min=self.r_min, r_max=self.r_max, sigma_w=self.scale, sigma_m=self.sigma_m, sigma_f=self.sigma_f,
                    step=self.step, origin=self.origin, kernel_deg=self.kernel_deg, lags=self.lags, weights=weights, radial=self.radial)

    def interior(self, vshape, cshape) -> torch.Tensor:
        """1 at lattice positions at least ``edge`` voxels from every face of the block, else 0."""
        m = torch.ones(tuple(cshape), device=self.device)
        for ax, (n, c) in enumerate(zip(vshape, cshape)):
            x = self.origin + self.step * torch.arange(c, device=self.device, dtype=torch.float32)
            ok = ((x >= self.edge) & (x <= n - 1 - self.edge)).float()
            shp = [1, 1, 1]; shp[ax] = c; m = m * ok.view(shp)
        return m

    def _null_lift(self, v: torch.Tensor, mask, seed: int, isotropic: bool):
        v = v.to(self.device)
        sl = tuple(slice(max(0, (n - self.null_crop) // 2), max(0, (n - self.null_crop) // 2) + min(n, self.null_crop)) for n in v.shape[-3:])
        vc = v[(...,) + sl].contiguous(); mc = None if mask is None else mask.to(self.device)[sl].contiguous()
        s = iso_surrogate(vc if vc.dim() == 3 else vc[0], mc, seed, isotropic=isotropic)
        O = self.lift(s)
        return O, self.interior(s.shape, O.shape[1:]) > 0

    def null_profile(self, v: torch.Tensor, mask=None, seed: int = 0) -> torch.Tensor:
        """The **spectrum null**'s directional bias ``(n_dirs,)``: the mean lift, over the interior positions of the
        surrogate that keeps the block's full (anisotropic) spectrum, minus its median direction. Subtracted from
        every lift (``bias``), it removes what the acquisition alone puts into the field -- a point-spread function
        longer along the rotation axis, rings, slice-wise structure -- which an isotropic surrogate cannot see. It also
        removes a texture orientation that dominates the whole region, so measure it on a region with many
        orientations (a whole volume's calibration region), not on one bundle."""
        O, inner = self._null_lift(v, mask, seed, isotropic=False)
        b = O.reshape(O.shape[0], -1)[:, inner.flatten()].mean(1) if bool(inner.any()) else O.reshape(O.shape[0], -1).mean(1)
        return b - b.median()

    def null_floor(self, v: torch.Tensor, mask=None, seed: int = 0, bias: torch.Tensor | None = None) -> float:
        """The ``null_q`` quantile of :func:`presence` on the block's surrogate (isotropic, or with the full spectrum
        and the ``bias`` removed when the null is ``"spectrum"``), over a central crop of at most ``null_crop`` voxels per
        axis, its interior."""
        bias = self.bias if bias is None else bias
        O, inner = self._null_lift(v, mask, seed, isotropic=(self.null == "isotropic" or bias is None))
        if bias is not None and self.null == "spectrum":
            O = O - bias.view(-1, 1, 1, 1)
        pn = presence(O, self.presence_mode)
        vals = pn[inner] if bool(inner.any()) else pn.flatten()
        return float(torch.quantile(vals[:: max(1, vals.numel() // 500_000)], self.null_q))

    def weight(self, p: torch.Tensor, floor: float) -> torch.Tensor:
        """Sensitivity weight from presence: 0 at the floor, 1 from ``full_factor`` x the floor, linear between."""
        return ((p - floor) / max((self.full_factor - 1) * floor, 1e-6)).clamp(0, 1)

    def populations(self, v: torch.Tensor, *, O=None, response=None, floor="null", mask=None, labels: bool = True, bias=None) -> dict:
        """The lattice stages on ``v (Z, Y, X)`` (or ``(K, Z, Y, X)``). ``floor``: ``"null"`` measures the presence floor
        on the block's surrogate; a number uses that floor (a volume-level value for chunked runs). ``response``: a
        profile (see :mod:`shapetensor.lift`); None estimates it from the block. ``bias``: the spectrum null's directional
        profile to subtract (``self.bias`` if set; with ``null="spectrum"`` and neither given it is measured on the block)."""
        v = v.to(self.device)
        O = self.lift(v) if O is None else O
        bias = self.bias if bias is None else bias
        if bias is None and self.null == "spectrum":
            bias = self.null_profile(v, mask)
        if bias is not None:
            O = O - bias.view(-1, 1, 1, 1)
        R = response if response is not None else self.response
        if R is None:
            R, _ = self_response(O, self.grid, bin_deg=self.bin_deg)
        A = response_matrix(R, self.grid, self.bin_deg, rings=self.rings)
        nd = self.grid.shape[0]
        x = deconvolve(O, A, self.iters, all_terms=True)
        f = angular_smooth(x[:nd], self.grid, self.angular_deg)
        sheet_share = x[nd:2 * nd].sum(0) / x[:2 * nd].sum(0).clamp_min(1e-12) if self.rings else torch.zeros_like(x[0])
        del x
        if isinstance(floor, str):
            floor = self.null_floor(v, mask, bias=bias)
        self.last_floor = float(floor)
        p = presence(O, self.presence_mode); w = self.weight(p, floor) * self.interior(v.shape[-3:], O.shape[1:])
        g = f / f.sum(0, keepdim=True).clamp_min(1e-12) * w[None]                                 # sums to the weight
        if self.momentum:
            g = momentum(g, self.grid, sigma_along=self.sigma_along, reach=self.reach, angular_deg=self.momentum_angular_deg)
        wt = g.sum(0)                                                                              # the weight after bridging
        dirs, share = peaks(g / wt.clamp_min(1e-12)[None], self.grid, nmax=self.nmax, rel=self.rel, min_mass=self.min_share,
                            min_sep_deg=self.min_sep_deg, valley=self.valley)
        if self.ring_rel < 1.0:                                                                    # a further lobe on a sheet's ring is not a population
            rr = lobe_ring_ratio(g / wt.clamp_min(1e-12)[None] if self.ring_on == "fodf" else O, self.grid, dirs, share)
            share = share * (rr <= self.ring_rel).float()
        ring, prom = ring_fraction(g, self.grid, dirs, share)
        sheet = ((ring > self.max_ring) & (prom < self.min_prominence) & (wt >= self.min_weight)) | (sheet_share > self.max_sheet_share)
        mass = share * ((wt >= self.min_weight) & ~sheet).float()[None] * wt[None]                 # N = 0 below the weight floor and on sheets
        out = {"N": (mass > 0).sum(0), "mass": mass, "share": share, "dirs_cells": dirs, "presence": p, "weight": wt, "fodf": g,
               "ring": ring, "prominence": prom, "sheet": sheet, "sheet_share": sheet_share, "floor": float(floor), "response": R, "lift": O,
               "bias": bias}
        if labels:
            out["labels_cells"], out["table"] = label_populations(dirs, mass, max_angle_deg=self.label_angle_deg, min_fixels=self.min_fixels)
        return out

    # -- the field at every voxel ----------------------------------------------------------------------------------

    def _coords(self, shape, cs, z0, z1):
        """The lattice coordinates of voxels ``z0:z1`` of a block of ``shape`` whose lattice has ``cs`` cells per axis, as
        a grid_sample grid ``(1, zs, Y, X, 3)`` in (x, y, z) order, normalised to the cell lattice."""
        Z, Y, X = shape
        ax = []
        for n, c, lo, hi in ((Z, cs[0], z0, z1), (Y, cs[1], 0, Y), (X, cs[2], 0, X)):
            j = torch.arange(lo, hi, device=self.device, dtype=torch.float32)
            ax.append(((j - self.origin) / self.step) * (2.0 / max(c - 1, 1)) - 1.0)
        gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
        return torch.stack([gx, gy, gz], -1)[None]

    def field(self, pop: dict, shape, *, labels: bool = True, tensor: bool = False, slab_bytes: int = 1 << 30) -> dict:
        """The lifted field read at every voxel of a block of ``shape (Z, Y, X)``: ``vecs``, ``weights``, ``labels``,
        ``vec``, ``weight``, ``N``, ``sheet`` (and ``tensor (6, Z, Y, X)`` = the field's second moment, for blending)."""
        Z, Y, X = shape; nd = self.grid.shape[0]; dev = self.device
        g = pop["fodf"]; wt = pop["weight"]; sheet = pop["sheet"].float()
        cs = tuple(g.shape[1:])
        lab_c = pop.get("labels_cells"); dirs_c = pop["dirs_cells"]
        K = self.nmax
        vecs = torch.zeros((K, 3, Z, Y, X), device=dev); weights = torch.zeros((K, Z, Y, X), device=dev)
        labs = torch.zeros((K, Z, Y, X), dtype=torch.int64, device=dev) if labels and lab_c is not None else None
        wvox = torch.zeros((Z, Y, X), device=dev); svox = torch.zeros((Z, Y, X), device=dev)
        T = torch.zeros((6, Z, Y, X), device=dev) if tensor else None
        zs = max(1, int(slab_bytes // (nd * Y * X * 4)))
        gz, gy, gx = self.grid[:, 0], self.grid[:, 1], self.grid[:, 2]
        cos_lab = math.cos(math.radians(self.label_angle_deg))
        for z0 in range(0, Z, zs):
            z1 = min(Z, z0 + zs); grd = self._coords(shape, cs, z0, z1)
            gs = Fn.grid_sample(g[None], grd, mode="bilinear", padding_mode="border", align_corners=True)[0]      # (nd, zs, Y, X)
            w_s = Fn.grid_sample(wt[None, None], grd, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
            s_s = Fn.grid_sample(sheet[None, None], grd, mode="nearest", padding_mode="border", align_corners=True)[0, 0]
            tot = gs.sum(0).clamp_min(1e-12)
            d, share = peaks(gs / tot[None], self.grid, nmax=K, rel=self.rel, min_mass=self.min_share, min_sep_deg=self.min_sep_deg, valley=self.valley)
            not_sheet = (s_s < 0.5).float(); keep = (w_s >= self.min_weight).float() * not_sheet
            wk = torch.cat([not_sheet[None], keep[None].expand(K - 1, -1, -1, -1)]) if K > 1 else not_sheet[None]      # the first vector everywhere (weighted), the extras thresholded
            vecs[:, :, z0:z1] = d; weights[:, z0:z1] = share * w_s[None] * wk
            wvox[z0:z1] = w_s; svox[z0:z1] = s_s
            if T is not None:
                T[:, z0:z1] = torch.stack([c @ gs.reshape(nd, -1) for c in (gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz)]).reshape(6, z1 - z0, Y, X)
            if labs is not None:                                                                   # the label of the nearest cell's best-aligned fixel
                cz = ((torch.arange(z0, z1, device=dev) - self.origin) / self.step).round().long().clamp(0, cs[0] - 1)
                cy = ((torch.arange(Y, device=dev) - self.origin) / self.step).round().long().clamp(0, cs[1] - 1)
                cx = ((torch.arange(X, device=dev) - self.origin) / self.step).round().long().clamp(0, cs[2] - 1)
                Lc = lab_c[:, cz][:, :, cy][:, :, :, cx]                                             # (K, zs, Y, X)
                Dc = dirs_c[:, :, cz][:, :, :, cy][:, :, :, :, cx]                                   # (K, 3, zs, Y, X)
                for r in range(K):
                    cosr = torch.einsum("c...,kc...->k...", d[r], Dc).abs()                          # (K, zs, Y, X)
                    cosr = torch.where(Lc > 0, cosr, torch.full_like(cosr, -1.0))
                    best = cosr.argmax(0); cbest = cosr.gather(0, best[None])[0]
                    lab_r = Lc.gather(0, best[None])[0]
                    labs[r, z0:z1] = torch.where((cbest >= cos_lab) & (weights[r, z0:z1] > 0), lab_r, torch.zeros_like(lab_r))
            del gs
        out = {"vecs": vecs, "weights": weights, "vec": vecs[0], "weight": wvox * (1 - svox), "N": (weights >= self.min_weight * weights.sum(0).clamp_min(1e-12) * 0 + (wvox[None] >= self.min_weight) * (weights > 0)).sum(0) if False else ((weights > 0) & (wvox[None] >= self.min_weight)).sum(0), "sheet": svox > 0.5}
        if labs is not None:
            out["labels"] = labs
        if T is not None:
            out["tensor"] = T
        return out

    def run(self, v: torch.Tensor, *, mask=None, floor="null", response=None, labels: bool = True, native: bool = True, tensor: bool = False, bias=None) -> dict:
        v = v.to(self.device).float()
        pop = self.populations(v, floor=floor, response=response, mask=mask, labels=labels, bias=bias)
        if native:
            pop["N_cells"], pop["sheet_cells"] = pop["N"], pop["sheet"]                          # the lattice's; "N" / "sheet" become the voxels'
            pop.update(self.field(pop, v.shape[-3:], labels=labels, tensor=tensor))
        return pop


def orientations(image, *, spacing=None, device: str = "cuda", **kwargs) -> dict:
    """The whole chain on one array ``(Z, Y, X)`` (numpy, cupy or torch; grey levels or any scalar field; ``(K, Z, Y, X)``
    channels add). ``spacing (dz, dy, dx)``: anisotropic voxels are resampled to the finest axis for the analysis and the
    field is resampled back. Keyword arguments are :class:`Orientations` parameters (``scale`` above all)."""
    from .st import _to_torch, from_isotropic, spacing_ratio, to_isotropic
    v = _to_torch(image, device, torch.float32)
    sp = spacing_ratio(spacing)
    vi = to_isotropic(v, sp) if sp is not None else v
    out = Orientations(device=device, **kwargs).run(vi)
    if sp is not None:
        shape = v.shape[-3:]
        for k in ("vecs", "weights", "vec", "weight"):
            out[k] = from_isotropic(out[k], shape)
        for k in ("N", "sheet", "labels"):
            if k in out:
                out[k] = from_isotropic(out[k].float(), shape, mode="nearest").to(out[k].dtype)
        out["vecs"] = out["vecs"] / out["vecs"].norm(dim=1, keepdim=True).clamp_min(1e-12); out["vec"] = out["vecs"][0]
    return out
