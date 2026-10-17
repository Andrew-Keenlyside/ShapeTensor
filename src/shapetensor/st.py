"""The structure tensor, in the layout of the ``structure-tensor`` package (Skielex/structure-tensor), so these
tensors substitute wherever that package's are consumed. Everything runs on the GPU in torch; inputs may be numpy,
cupy or torch arrays and the result comes back as the caller's kind (cupy through DLPack, without a copy).

Conventions, identical to ``structure_tensor.structure_tensor_3d`` / ``eig_special_3d``:

* a volume is ``(Z, Y, X)``; derivatives are scipy's Gaussian derivatives with ``mode='nearest'`` and ``truncate``;
* ``S`` is ``(6, Z, Y, X)`` holding ``[Sxx, Syy, Szz, Sxy, Sxz, Syz]`` with ``x`` the LAST axis;
* ``eig_special_3d(S)`` returns ``val (3, ...)`` (descending by default) and ``vec (3, ...)``, the unit eigenvector of
  the SMALLEST eigenvalue with components in ``(x, y, z)`` order, or all three as ``(3, 3, ...)`` with ``full=True``.

:func:`excised_structure_tensor_3d` is the boundary-robust tensor: gradients whose coarse magnitude exceeds three
times the median (a tissue interface, a vessel wall) are left out of the integration, so the direction is not
dragged into the plane of the boundary. It is the package's baseline and the teacher of the pipeline's stage 2.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .filters import gaussian


# ---------------------------------------------------------------------------------------------------------------
# array kinds
# ---------------------------------------------------------------------------------------------------------------

def _is_cupy(a) -> bool:
    return type(a).__module__.startswith("cupy")


def _to_torch(a, device: str, dtype=None) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        t = a.to(device)
    elif _is_cupy(a):
        t = torch.from_dlpack(a)
        if str(t.device) != str(torch.device(device)):
            t = t.to(device)
    else:
        t = torch.as_tensor(np.asarray(a), device=device)
    return t if dtype is None else t.to(dtype)


def _like(t: torch.Tensor, like):
    """``t`` as the kind of ``like``: numpy, cupy (zero copy) or torch."""
    if isinstance(like, torch.Tensor):
        return t
    if _is_cupy(like):
        import cupy
        return cupy.from_dlpack(t.contiguous())
    return t.cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# anisotropic voxels
# ---------------------------------------------------------------------------------------------------------------

def spacing_ratio(spacing):
    """``spacing (dz, dy, dx)`` -> the voxel sizes relative to the finest axis, or None when isotropic. ``sigma`` and
    ``rho`` are in voxels of the finest axis; along a coarser axis the filters shrink by this ratio and the
    derivative is divided by it, so gradients and directions are in physical (finest-voxel) units."""
    if spacing is None:
        return None
    sp = tuple(float(s) for s in spacing)
    m = min(sp)
    sp = tuple(s / m for s in sp)
    return None if all(abs(s - 1.0) < 1e-9 for s in sp) else sp


def _per_axis(value: float, spacing):
    return value if spacing is None else tuple(value / s for s in spacing)


def to_isotropic(v: torch.Tensor, spacing, mode: str = "trilinear") -> torch.Tensor:
    """Resample ``v (..., Z, Y, X)`` with voxel ratios ``spacing`` onto the finest-axis grid."""
    import torch.nn.functional as Fn
    sp = spacing_ratio(spacing)
    if sp is None:
        return v
    size = tuple(int(round(n * s)) for n, s in zip(v.shape[-3:], sp))
    out = Fn.interpolate(v.reshape(-1, 1, *v.shape[-3:]).float(), size=size, mode=mode, **({} if mode == "nearest" else {"align_corners": False}))
    return out.reshape(*v.shape[:-3], *size).to(v.dtype)


def from_isotropic(t: torch.Tensor, shape, mode: str = "trilinear") -> torch.Tensor:
    """Resample ``t (..., Z', Y', X')`` back onto the original ``shape (Z, Y, X)``."""
    import torch.nn.functional as Fn
    out = Fn.interpolate(t.reshape(-1, 1, *t.shape[-3:]).float(), size=tuple(shape), mode=mode, **({} if mode == "nearest" else {"align_corners": False}))
    return out.reshape(*t.shape[:-3], *shape).to(t.dtype)


# ---------------------------------------------------------------------------------------------------------------
# the tensor
# ---------------------------------------------------------------------------------------------------------------

def _derivatives(v: torch.Tensor, sigma: float, truncate: float, spacing=None):
    sig = _per_axis(sigma, spacing)
    sz, sy, sx = (1.0, 1.0, 1.0) if spacing is None else spacing
    Vx = gaussian(v, sig, order=(0, 0, 1), truncate=truncate) / sx
    Vy = gaussian(v, sig, order=(0, 1, 0), truncate=truncate) / sy
    Vz = gaussian(v, sig, order=(1, 0, 0), truncate=truncate) / sz
    return Vx, Vy, Vz


def _integrate(prods, rho: float, truncate: float, weights: torch.Tensor | None, spacing=None):
    """``S[k] = G_rho * (w * prod_k) / G_rho * w`` (a plain mean when ``weights`` is None)."""
    r = _per_axis(rho, spacing)
    S = torch.empty((6,) + tuple(prods[0].shape), dtype=prods[0].dtype, device=prods[0].device)
    wn = None if weights is None else gaussian(weights, r, truncate=truncate).clamp_min(1e-12)
    for k, p in enumerate(prods):
        S[k] = gaussian(p if weights is None else p * weights, r, truncate=truncate)
        if wn is not None:
            S[k] /= wn
    return S


def excision_weights(vol: torch.Tensor, sigma_coarse: float = 3.0, factor: float = 3.0, truncate: float = 3.0,
                     mask: torch.Tensor | None = None, spacing=None) -> torch.Tensor:
    """The excision as a weight map in {0, 1}: 0 where the coarse (``sigma_coarse``) gradient magnitude exceeds
    ``factor`` × its median (over ``mask``, default everywhere), else 1."""
    v = vol.float()
    sp = (1.0, 1.0, 1.0) if spacing is None else tuple(spacing)
    sig = sigma_coarse if spacing is None else tuple(sigma_coarse / s for s in sp)
    gm = sum((gaussian(v, sig, order=o, truncate=truncate) / sp[i]) ** 2 for i, o in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))).sqrt()
    ref = gm[mask] if (mask is not None and bool(mask.any())) else gm
    med = ref.flatten().median()
    if not torch.isfinite(med) or med <= 0:
        return torch.ones_like(gm)
    return (gm <= factor * med).float()


def structure_tensor_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, weights=None, spacing=None,
                        device: str = "cuda", dtype=torch.float32):
    """``S (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz]`` of ``volume (Z, Y, X)``: derivatives at ``sigma``, integration
    at ``rho``. ``weights (Z, Y, X)`` in [0, 1] excise gradients with a normalised integration. ``spacing (dz, dy, dx)``:
    the voxel size per axis for anisotropic data (``sigma`` / ``rho`` in finest-voxel units; the tensor in physical
    coordinates)."""
    v = _to_torch(volume, device, dtype)
    sp = spacing_ratio(spacing)
    Vx, Vy, Vz = _derivatives(v, sigma, truncate, sp)
    w = None if weights is None else _to_torch(weights, device, dtype)
    S = _integrate((Vx * Vx, Vy * Vy, Vz * Vz, Vx * Vy, Vx * Vz, Vy * Vz), rho, truncate, w, sp)
    return _like(S, volume)


def excised_structure_tensor_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, factor: float = 3.0,
                                sigma_coarse: float = 3.0, mask=None, spacing=None, device: str = "cuda", dtype=torch.float32):
    """The boundary-robust tensor: gradients whose coarse (``sigma_coarse``) magnitude exceeds ``factor`` × the
    reference median are left out of the integration, which is renormalised by the surviving weight. The reference
    median is taken over ``mask (Z, Y, X)`` (a tissue mask; the whole volume if None). Returns ``S`` as
    :func:`structure_tensor_3d`."""
    v = _to_torch(volume, device, dtype)
    sp = spacing_ratio(spacing)
    m = None if mask is None else _to_torch(mask, device, dtype) > 0.5
    w = excision_weights(v, sigma_coarse=sigma_coarse, factor=factor, truncate=truncate, mask=m, spacing=sp)
    Vx, Vy, Vz = _derivatives(v, sigma, truncate, sp)
    S = _integrate((Vx * Vx, Vy * Vy, Vz * Vz, Vx * Vy, Vx * Vz, Vy * Vz), rho, truncate, w, sp)
    return _like(S, volume)


def eig_special_3d(S, full: bool = False, eigenvalue_order: Literal["desc", "asc"] = "desc", *, device: str = "cuda",
                   chunk: int = 1 << 20):       # cusolver's batched solver rejects batches of ~2M and above
    """Eigensolution of the symmetric 3 × 3 tensors in ``S (6, ...)``, as the ``structure-tensor`` package returns it:
    ``val (3, ...)`` (descending unless ``eigenvalue_order='asc'``) and ``vec (3, ...)`` = the unit eigenvector of the
    smallest eigenvalue in ``(x, y, z)`` component order, or with ``full=True`` ``(3, 3, ...)`` with ``vec[i]`` the
    eigenvector of ``val[i]``."""
    St = _to_torch(S, device)
    shape = tuple(St.shape[1:])
    Sf = St.reshape(6, -1)
    N = Sf.shape[1]
    val = torch.empty((3, N), dtype=St.dtype, device=St.device)
    vec = torch.empty(((3, 3, N) if full else (3, N)), dtype=St.dtype, device=St.device)
    for a in range(0, N, chunk):
        s = Sf[:, a:a + chunk]
        M = torch.stack([torch.stack([s[0], s[3], s[4]], -1), torch.stack([s[3], s[1], s[5]], -1),
                         torch.stack([s[4], s[5], s[2]], -1)], -2)
        w, V = torch.linalg.eigh(M)
        if eigenvalue_order == "desc":
            w, V = w.flip(-1), V.flip(-1)
        val[:, a:a + chunk] = w.T
        if full:
            vec[:, :, a:a + chunk] = V.permute(2, 1, 0)
        else:
            vec[:, a:a + chunk] = V[:, :, 0 if eigenvalue_order == "asc" else 2].T
    return _like(val.reshape((3,) + shape), S), _like(vec.reshape(((3, 3) if full else (3,)) + shape), S)


def structure_tensor(image, *, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, excise: bool = True, mask=None,
                     order: str = "xyz", spacing=None, device: str = "cuda"):
    """The structure tensor's fibre direction ``(3, Z, Y, X)`` of ``image`` in one call (excised by default), as the
    caller's array kind, components in ``order`` (``"xyz"`` as the ``structure-tensor`` package, or ``"zyx"``). A list
    of co-registered images (several stains) adds their tensors."""
    if isinstance(image, (list, tuple)):
        S = sum(_to_torch(_tensor(im, sigma, rho, truncate, excise, mask, spacing, device), device) for im in image)
        _, vec = eig_special_3d(S, device=device)
        return _like(vec.flip(0) if order == "zyx" else vec, image[0])
    _, vec = eig_special_3d(_tensor(image, sigma, rho, truncate, excise, mask, spacing, device), device=device)
    return _like(_to_torch(vec, device).flip(0), image) if order == "zyx" else vec


def _tensor(image, sigma, rho, truncate, excise, mask, spacing, device):
    if excise:
        return excised_structure_tensor_3d(image, sigma, rho, truncate, mask=mask, spacing=spacing, device=device)
    return structure_tensor_3d(image, sigma, rho, truncate, spacing=spacing, device=device)


# ---------------------------------------------------------------------------------------------------------------
# a whole volume
# ---------------------------------------------------------------------------------------------------------------

def run_volume(src: str, dst: str, *, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, chunk=(256, 256, 256),
               excise: bool = True, excise_factor: float = 3.0, level: str | None = None, arrays=("S", "val", "vec"), dtype=np.float32,
               tissue_threshold: float | None = None, rank: int = 0, world: int = 1, spacing=None, device: str = "cuda", log=print):
    """The structure tensor over a whole zarr volume, chunk by chunk with halos, on one GPU; several processes split
    the chunk list by ``rank`` / ``world``. ``dst`` receives the ``arrays`` asked for among ``S (6, Z, Y, X)``,
    ``val (3, ...)``, ``vec (3, ...)`` (the direction in (x, y, z)) and ``gap`` (the orientation gap, ``(Z, Y, X)``),
    as ``dtype``; every chunk's output equals a one-pass computation. ``tissue_threshold``: chunks whose core holds
    no voxel at or above it are skipped and stay zero."""
    import zarr
    a = zarr.open(src, mode="r")
    if level is not None:
        a = a[level]
    Z, Y, X = a.shape
    g = zarr.open_group(dst, mode="a")
    cz, cy, cx = chunk
    halo = int(truncate * (sigma + rho) + 0.5) + (int(3.0 * truncate + 0.5) if excise else 0)
    shapes = {"S": (6, Z, Y, X), "val": (3, Z, Y, X), "vec": (3, Z, Y, X), "gap": (Z, Y, X)}
    for name in arrays:
        if name not in g:
            g.create_array(name, shape=shapes[name], dtype=dtype, chunks=(shapes[name][:-3] + (cz, cy, cx)))
    g.attrs.update({"sigma": sigma, "rho": rho, "truncate": truncate, "excised": excise, "excise_factor": excise_factor, "halo": halo,
                    "layout": "S = [xx, yy, zz, xy, xz, yz], x = last axis; vec in (x, y, z)", "tissue_threshold": tissue_threshold})
    tiles = [(z0, y0, x0) for z0 in range(0, Z, cz) for y0 in range(0, Y, cy) for x0 in range(0, X, cx)]
    skipped = 0
    for i, (z0, y0, x0) in enumerate(tiles):
        if i % world != rank:
            continue
        z1, y1, x1 = min(Z, z0 + cz), min(Y, y0 + cy), min(X, x0 + cx)
        lo = (max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)); hi = (min(Z, z1 + halo), min(Y, y1 + halo), min(X, x1 + halo))
        raw = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        core = (slice(z0 - lo[0], z1 - lo[0]), slice(y0 - lo[1], y1 - lo[1]), slice(x0 - lo[2], x1 - lo[2])); out_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        if tissue_threshold is not None and not (raw[core] >= tissue_threshold).any():
            skipped += 1; continue
        blk = torch.as_tensor(raw.astype(np.float32), device=device); del raw
        S = (excised_structure_tensor_3d(blk, sigma, rho, truncate, factor=excise_factor, spacing=spacing, device=device) if excise
             else structure_tensor_3d(blk, sigma, rho, truncate, spacing=spacing, device=device))
        Sc = S[(slice(None),) + core].contiguous(); del S
        val, vec = eig_special_3d(Sc, device=device)
        out = {"S": Sc, "val": val, "vec": vec, "gap": ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)}
        for name in arrays:
            g[name][((slice(None),) if name != "gap" else ()) + out_sl] = out[name].to(torch.from_numpy(np.zeros(1, dtype)).dtype).cpu().numpy()
        del Sc, val, vec, blk, out; torch.cuda.empty_cache()
        log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}) done")
    if skipped:
        log(f"[{rank}/{world}] {skipped} chunks without tissue skipped")
    return dst
