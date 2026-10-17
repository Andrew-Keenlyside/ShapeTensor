"""Archived: up to ``nmax`` directions per site from the CSD of the gradient-direction ODF (the first form of
"several vectors per voxel", replaced by the population layers' multi-vector field)."""
from __future__ import annotations

import math

import torch

from ..filters import gaussian
from ..st import _like, _to_torch, from_isotropic, spacing_ratio, to_isotropic
from .tensor import gradients, odf_dense


def fixels_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, lmax: int = 8, reduce: int | None = None,
              csd_width_deg: float = 20.0, peak_rel: float = 0.5, min_sep_deg: float = 25.0, nmax: int = 3,
              interface_gate: float | None = 3.0, sigma_coarse: float = 3.0, upsample: bool = True, spacing=None, device: str = "cuda"):
    """Returns ``dirs (nmax, 3, ...)`` in (x, y, z) component order (zero where absent), ``fracs (nmax, ...)`` and
    ``npop (...)``, on the ``reduce``-voxel site grid (default ``rho / 4``) or nearest-upsampled to the volume grid.
    Sites whose coarse gradient magnitude exceeds ``interface_gate`` × the median report no population."""
    from .odf import csd, peaks, ring_profile, zonal_kernel
    v = _to_torch(volume, device, torch.float32)
    sp = spacing_ratio(spacing)
    if sp is not None:
        shape_in = tuple(v.shape); v = to_isotropic(v, sp)
    Z, Y, X = v.shape
    b = reduce or max(1, int(rho // 4))
    core = ((0, Z // b * b), (0, Y // b * b), (0, X // b * b))
    G = gradients(v[None], sigma, truncate)
    O, sites = odf_dense(G, rho, truncate, b, core, lmax=lmax); del G
    kern = zonal_kernel(lmax, ring_profile(math.radians(csd_width_deg)))
    f = csd(O.reshape(-1, O.shape[-1]), lmax, kern)
    d, a, cnt = peaks(f, lmax, nmax=nmax, rel=peak_rel, min_sep_deg=min_sep_deg)
    frac = a / a.sum(1, keepdim=True).clamp_min(1e-30)
    shp = sites.shape
    dirs = d.reshape(shp + (nmax, 3)).permute(3, 4, 0, 1, 2).flip(1).contiguous()
    fracs = frac.reshape(shp + (nmax,)).permute(3, 0, 1, 2).contiguous()
    npop = cnt.reshape(shp)
    if interface_gate is not None:
        gm = sum(gaussian(v, sigma_coarse, order=o, truncate=truncate) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
        (z0, z1), (y0, y1), (x0, x1) = core
        gm_s = torch.nn.functional.avg_pool3d(gm[None, None, z0:z1, y0:y1, x0:x1], b)[0, 0] if b > 1 else gm[z0:z1, y0:y1, x0:x1]
        at = gm_s > interface_gate * gm_s.median()
        dirs[:, :, at] = 0.0; fracs[:, at] = 0.0; npop[at] = 0
    if upsample and b > 1:
        import torch.nn.functional as Fn
        up = lambda t: Fn.interpolate(t.reshape(-1, 1, *shp).float(), scale_factor=(b, b, b), mode="nearest")[:, 0]
        dirs = up(dirs).reshape(nmax, 3, *(s * b for s in shp)); fracs = up(fracs).reshape(nmax, *(s * b for s in shp))
        npop = up(npop[None])[0].to(torch.int64).reshape(*(s * b for s in shp))
        pad = [(0, n - s * b) for n, s in zip((Z, Y, X), shp)]
        if any(p[1] for p in pad):
            pz, py, px = pad
            dirs = Fn.pad(dirs, (0, px[1], 0, py[1], 0, pz[1])); fracs = Fn.pad(fracs, (0, px[1], 0, py[1], 0, pz[1])); npop = Fn.pad(npop, (0, px[1], 0, py[1], 0, pz[1]))
        if sp is not None:
            dirs, fracs, npop = from_isotropic(dirs, shape_in, "nearest"), from_isotropic(fracs, shape_in, "nearest"), from_isotropic(npop, shape_in, "nearest")
    return _like(dirs, volume), _like(fracs, volume), _like(npop, volume)
