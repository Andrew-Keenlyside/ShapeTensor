"""The channel-resolved tensor ``K = G_rho * (g g^T)`` and the gradient-direction ODF, on a site grid.

``g`` is the vector of all channel gradients at a voxel, indexed ``c * 3 + i`` (channel, then axis in Z, Y, X order
as BRIDGE stores vectors), so ``K`` is ``3C x 3C`` per site; summed over channels it is the familiar 3 x 3 structure
tensor, and with one channel it IS the structure tensor (tests/test_tensor.py checks that against scipy exactly).

Products are formed at full resolution (where the texture lives), then block-averaged by ``reduce`` voxels in z and x
(the sites), summed along y with the full-resolution Gaussian at the requested ``y_rows`` only (exact: a weighted sum
per row), and smoothed in z and x with a Gaussian of ``rho / reduce`` on the site grid. The effective window is
``G_rho`` convolved with a ``reduce``-voxel box, i.e. ``rho_eff^2 = rho^2 + reduce^2 / 12``; callers keep
``reduce <= rho / 4`` so the widening is under 1%. Sites are the centres of the reduce-blocks inside ``core_zx``.

The phantom suite scores one site row (the mid-y plane), as the earlier prototypes did; :func:`k_dense` is the whole-
block form (reduce in all three axes) for fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .filters import gaussian, gaussian_at_rows, kernel1d


def pair_index(nc3: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle (a <= b) pairs of the 3C gradient maps, in the packed order used throughout."""
    a, b = np.triu_indices(nc3)
    return a, b


def gradients(F: torch.Tensor, sigma: float, truncate: float, out_dtype=torch.float16) -> torch.Tensor:
    """``F (C, Z, Y, X)`` -> ``G (3C, Z, Y, X)``, row ``c * 3 + i`` = d f_c / d x_i, axes i in (Z, Y, X) order."""
    C = F.shape[0]
    G = torch.empty((3 * C,) + tuple(F.shape[1:]), dtype=out_dtype, device=F.device)
    for c in range(C):
        f = F[c].float()
        for i in range(3):
            o = [0, 0, 0]
            o[i] = 1
            G[c * 3 + i] = gaussian(f, sigma, order=o, truncate=truncate).to(out_dtype)
    return G


@dataclass
class Sites:
    """Where the K rows are: full-resolution voxel coordinates (z, y, x) of each site, and the grid they form."""
    z: np.ndarray            # (nz,) site centres along z, in voxel units of the block
    y: np.ndarray            # (ny,)
    x: np.ndarray            # (nx,)
    reduce: int
    rho: float

    @property
    def shape(self):
        return (len(self.z), len(self.y), len(self.x))

    def grid(self):
        """``(nz, ny, nx, 3)`` float32 coordinates."""
        Z, Y, X = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        return np.stack([Z, Y, X], -1).astype(np.float32)


def _reduce_zx(p: torch.Tensor, b: int) -> torch.Tensor:
    """Block-mean ``(Z, Ys, X)`` -> ``(Z//b, Ys, X//b)``."""
    if b == 1:
        return p
    Z, Ys, X = p.shape
    return p.reshape(Z // b, b, Ys, X // b, b).mean((1, 4))


def k_sites(G: torch.Tensor, rho: float, truncate: float, reduce: int, y_rows, core_zx, weights: torch.Tensor | None = None,
            pairs: tuple | None = None, dtype=torch.float32) -> tuple[torch.Tensor, Sites]:
    """``K`` at the sites of the rows ``y_rows``: ``(ny, nz, nx, P)`` packed upper triangle, P = 3C(3C+1)/2.

    ``core_zx = ((z0, z1), (x0, x1))`` is the region whose sites are kept (block coordinates); it must be aligned to
    ``reduce``. ``weights (Z, Y, X)`` in [0, 1], if given, excises gradients (fix A): ``K = G*(w g g^T) / G*(w)``.
    """
    b = int(reduce)
    n3, Z, Y, X = G.shape
    (z0, z1), (x0, x1) = core_zx
    if Z % b or X % b or z0 % b or z1 % b or x0 % b or x1 % b:
        raise ValueError(f"block ({Z}, {X}) and core {core_zx} must be multiples of reduce {b}")
    y_rows = [int(j) for j in y_rows]
    r = int(truncate * rho + 0.5)
    ys0, ys1 = max(0, min(y_rows) - r), min(Y, max(y_rows) + r + 1)
    rows_in_slab = [j - ys0 for j in y_rows]
    a_idx, b_idx = pair_index(n3) if pairs is None else pairs
    P = len(a_idx)
    buf = torch.empty((P, Z // b, len(y_rows), X // b), dtype=dtype, device=G.device)
    w = None if weights is None else weights[:, ys0:ys1, :].to(dtype)
    for k in range(P):
        p = G[a_idx[k], :, ys0:ys1, :].to(dtype) * G[b_idx[k], :, ys0:ys1, :].to(dtype)
        if w is not None:
            p = p * w
        buf[k] = gaussian_at_rows(_reduce_zx(p, b), rho, rows_in_slab, axis=1, truncate=truncate)
    # z, x smoothing on the site grid, batched over (P, rows)
    buf = gaussian(buf, rho / b, order=(0, 0), truncate=truncate, axes=(1, 3))
    if w is not None:
        wn = gaussian_at_rows(_reduce_zx(w, b), rho, rows_in_slab, axis=1, truncate=truncate)
        wn = gaussian(wn, rho / b, order=(0, 0), truncate=truncate, axes=(0, 2)).clamp_min(1e-12)
        buf = buf / wn[None]
    K = buf[:, z0 // b:z1 // b, :, x0 // b:x1 // b].permute(2, 1, 3, 0).contiguous()      # (ny, nz, nx, P)
    sites = Sites(z=(np.arange(z0, z1, b) + (b - 1) / 2.0), y=np.array(y_rows, float),
                  x=(np.arange(x0, x1, b) + (b - 1) / 2.0), reduce=b, rho=float(rho))
    return K, sites


def k_dense(G: torch.Tensor, rho: float, truncate: float, reduce: int, core, weights: torch.Tensor | None = None,
            dtype=torch.float32) -> tuple[torch.Tensor, Sites]:
    """Whole-block ``K`` on a ``reduce``-grid in all three axes: ``(nz, ny, nx, P)`` over ``core = ((z0,z1),(y0,y1),(x0,x1))``."""
    b = int(reduce)
    n3, Z, Y, X = G.shape
    (z0, z1), (y0, y1), (x0, x1) = core
    for n in (Z, Y, X, z0, z1, y0, y1, x0, x1):
        if n % b:
            raise ValueError("block and core must be multiples of reduce")
    a_idx, b_idx = pair_index(n3)
    P = len(a_idx)
    buf = torch.empty((P, Z // b, Y // b, X // b), dtype=dtype, device=G.device)
    w = None if weights is None else weights.to(dtype)
    for k in range(P):
        p = G[a_idx[k]].to(dtype) * G[b_idx[k]].to(dtype)
        if w is not None:
            p = p * w
        buf[k] = p.reshape(Z // b, b, Y // b, b, X // b, b).mean((1, 3, 5)) if b > 1 else p
    buf = gaussian(buf, rho / b, truncate=truncate, axes=(1, 2, 3))
    if w is not None:
        wn = w.reshape(Z // b, b, Y // b, b, X // b, b).mean((1, 3, 5)) if b > 1 else w
        buf = buf / gaussian(wn, rho / b, truncate=truncate).clamp_min(1e-12)[None]
    K = buf[:, z0 // b:z1 // b, y0 // b:y1 // b, x0 // b:x1 // b].permute(1, 2, 3, 0).contiguous()
    sites = Sites(z=(np.arange(z0, z1, b) + (b - 1) / 2.0), y=(np.arange(y0, y1, b) + (b - 1) / 2.0),
                  x=(np.arange(x0, x1, b) + (b - 1) / 2.0), reduce=b, rho=float(rho))
    return K, sites


def odf_sites(G: torch.Tensor, rho: float, truncate: float, reduce: int, y_rows, core_zx, lmax: int = 8,
              channels=None, z_chunk: int = 32) -> tuple[torch.Tensor, Sites]:
    """The gradient-direction energy distribution in even real spherical harmonics, at the same sites as
    :func:`k_sites`: ``(ny, nz, nx, ncoef)``. Per voxel and channel the unit gradient direction weighted by ``|g|^2``,
    summed over channels (``channels``: indices, default all), windowed exactly as ``K``. Its l <= 2 part is the
    structure tensor; the higher orders are what the FRT / CSD read-outs use (:mod:`shapetensor.odf`)."""
    from .odf import n_coef, real_sh
    b = int(reduce)
    n3, Z, Y, X = G.shape
    C = n3 // 3
    chans = list(range(C)) if channels is None else list(channels)
    (z0, z1), (x0, x1) = core_zx
    y_rows = [int(j) for j in y_rows]
    r = int(truncate * rho + 0.5)
    ys0, ys1 = max(0, min(y_rows) - r), min(Y, max(y_rows) + r + 1)
    rows_in_slab = [j - ys0 for j in y_rows]
    nc = n_coef(lmax)
    acc = torch.zeros((nc, Z // b, ys1 - ys0, X // b), dtype=torch.float32, device=G.device)
    for c in chans:
        for za in range(0, Z, z_chunk * b):
            zb = min(Z, za + z_chunk * b)
            g = G[c * 3:c * 3 + 3, za:zb, ys0:ys1, :].float()                       # (3, zc, ys, X)
            mag2 = (g * g).sum(0)
            u = g / mag2.clamp_min(1e-20).sqrt()
            Ysh = real_sh(u.reshape(3, -1).T, lmax).T.reshape((nc,) + mag2.shape)   # (nc, zc, ys, X)
            Ysh *= mag2
            if b > 1:
                Ysh = Ysh.reshape(nc, (zb - za) // b, b, ys1 - ys0, X // b, b).mean((2, 5))
            acc[:, za // b:zb // b] += Ysh
            del g, mag2, u, Ysh
    buf = gaussian_at_rows(acc, rho, rows_in_slab, axis=2, truncate=truncate)
    buf = gaussian(buf, rho / b, order=(0, 0), truncate=truncate, axes=(1, 3))
    O = buf[:, z0 // b:z1 // b, :, x0 // b:x1 // b].permute(2, 1, 3, 0).contiguous()
    sites = Sites(z=(np.arange(z0, z1, b) + (b - 1) / 2.0), y=np.array(y_rows, float),
                  x=(np.arange(x0, x1, b) + (b - 1) / 2.0), reduce=b, rho=float(rho))
    return O, sites


def odf_dense(G: torch.Tensor, rho: float, truncate: float, reduce: int, core, lmax: int = 8, channels=None,
              z_rows: int = 32) -> tuple[torch.Tensor, Sites]:
    """Whole-block gradient-direction ODF (as :func:`odf_sites`) on a ``reduce``-grid in all three axes:
    ``(nz, ny, nx, ncoef)`` over ``core = ((z0,z1),(y0,y1),(x0,x1))``."""
    from .odf import n_coef, real_sh
    b = int(reduce)
    n3, Z, Y, X = G.shape
    C = n3 // 3
    chans = list(range(C)) if channels is None else list(channels)
    (z0, z1), (y0, y1), (x0, x1) = core
    nc = n_coef(lmax)
    acc = torch.zeros((nc, Z // b, Y // b, X // b), dtype=torch.float32, device=G.device)
    step = max(b, (z_rows // b) * b)
    for c in chans:
        for za in range(0, Z, step):
            zb = min(Z, za + step)
            g = G[c * 3:c * 3 + 3, za:zb].float()
            mag2 = (g * g).sum(0)
            u = g / mag2.clamp_min(1e-20).sqrt()
            Ysh = real_sh(u.reshape(3, -1).T, lmax).T.reshape((nc,) + mag2.shape)
            Ysh *= mag2
            if b > 1:
                Ysh = Ysh.reshape(nc, (zb - za) // b, b, Y // b, b, X // b, b).mean((2, 4, 6))
            acc[:, za // b:zb // b] += Ysh
            del g, mag2, u, Ysh
    acc = gaussian(acc, rho / b, truncate=truncate, axes=(1, 2, 3))
    O = acc[:, z0 // b:z1 // b, y0 // b:y1 // b, x0 // b:x1 // b].permute(1, 2, 3, 0).contiguous()
    sites = Sites(z=(np.arange(z0, z1, b) + (b - 1) / 2.0), y=(np.arange(y0, y1, b) + (b - 1) / 2.0),
                  x=(np.arange(x0, x1, b) + (b - 1) / 2.0), reduce=b, rho=float(rho))
    return O, sites


def excision_weights(vol: torch.Tensor, sigma_coarse: float = 3.0, factor: float = 3.0, truncate: float = 3.0,
                     mask: torch.Tensor | None = None, gate: torch.Tensor | None = None) -> torch.Tensor:
    """Fix A's gradient excision as a weight map in [0, 1]: 0 where the coarse (``sigma_coarse``) gradient magnitude
    exceeds ``factor`` x its median (over ``mask``, default everywhere), else 1. With ``gate (Z, Y, X)`` in [0, 1]
    (a boundary-strength map upsampled to the block), the excision applies only in proportion to the gate:
    ``w = 1 - gate * (1 - w0)``, which is "excision only near detected interfaces" (arm H2)."""
    v = vol.float()
    g2 = sum(gaussian(v, sigma_coarse, order=o, truncate=truncate) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    gm = g2.sqrt()
    ref = gm[mask] if mask is not None else gm
    med = ref.flatten().median()
    w0 = (gm <= factor * med).float()
    if gate is None:
        return w0
    return 1.0 - gate.float().clamp(0, 1) * (1.0 - w0)


def upsample_to(block_shape, coarse: torch.Tensor, sites: Sites) -> torch.Tensor:
    """Nearest-neighbour upsample of a per-site scalar map ``(nz, ny, nx)`` back to the block grid (for gates)."""
    import torch.nn.functional as F
    b = sites.reduce
    Z, Y, X = block_shape
    up = F.interpolate(coarse[None, None].float(), scale_factor=(b, b, b), mode="nearest")[0, 0]
    out = torch.zeros(block_shape, dtype=torch.float32, device=coarse.device)
    z0, y0, x0 = int(sites.z[0] - (b - 1) / 2), int(sites.y[0] - (b - 1) / 2), int(sites.x[0] - (b - 1) / 2)
    zs, ys, xs = up.shape
    out[z0:z0 + zs, y0:y0 + ys, x0:x0 + xs] = up[:Z - z0, :Y - y0, :X - x0]
    return out
