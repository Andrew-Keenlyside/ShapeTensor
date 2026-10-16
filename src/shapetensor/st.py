"""Volume API in the style of the ``structure-tensor`` package (Skielex/structure-tensor), so these tensors can be
substituted wherever that package's are consumed. Everything runs on the GPU in torch; inputs may be numpy, cupy or
torch arrays and the result comes back as the caller's kind (cupy through DLPack, without a copy).

Conventions, identical to ``structure_tensor.structure_tensor_3d`` / ``eig_special_3d``:

* a volume is ``(Z, Y, X)``; derivatives are scipy's Gaussian derivatives with ``mode='nearest'`` and ``truncate``
  (default 4.0 as in that package);
* ``S`` is ``(6, Z, Y, X)`` holding ``[Sxx, Syy, Szz, Sxy, Sxz, Syz]`` with ``x`` the LAST axis;
* ``eig_special_3d(S)`` returns ``val (3, ...)`` (descending by default) and ``vec (3, ...)``, the unit eigenvector of
  the SMALLEST eigenvalue with components in ``(x, y, z)`` order, or all three as ``(3, 3, ...)`` with ``full=True``.

Beyond the plain tensor:

* :func:`excised_structure_tensor_3d`: the same ``S`` with boundary gradients excised before integration (fix A);
* :func:`shape_structure_tensor_3d`: ``S`` summed over the gradients of a feature engine's channels;
* :func:`fixels_3d`: up to ``nmax`` directions with fractions per site from constrained spherical deconvolution of the
  gradient-direction ODF, on a coarse site grid or upsampled to the volume grid;
* :func:`run_volume`: a chunked driver over a zarr volume with halos, one process per GPU (``rank`` / ``world``).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

from .filters import gaussian
from .tensor import excision_weights, gradients, odf_dense


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
# the structure tensor and its variants
# ---------------------------------------------------------------------------------------------------------------

def _derivatives(v: torch.Tensor, sigma: float, truncate: float):
    Vx = gaussian(v, sigma, order=(0, 0, 1), truncate=truncate)
    Vy = gaussian(v, sigma, order=(0, 1, 0), truncate=truncate)
    Vz = gaussian(v, sigma, order=(1, 0, 0), truncate=truncate)
    return Vx, Vy, Vz


def _integrate(prods, rho: float, truncate: float, weights: torch.Tensor | None):
    """``S[k] = G_rho * (w * prod_k) / G_rho * w`` (plain mean when ``weights`` is None)."""
    S = torch.empty((6,) + tuple(prods[0].shape), dtype=prods[0].dtype, device=prods[0].device)
    wn = None if weights is None else gaussian(weights, rho, truncate=truncate).clamp_min(1e-12)
    for k, p in enumerate(prods):
        S[k] = gaussian(p if weights is None else p * weights, rho, truncate=truncate)
        if wn is not None:
            S[k] /= wn
    return S


def structure_tensor_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, weights=None, device: str = "cuda",
                        dtype=torch.float32):
    """``S (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz]`` of ``volume (Z, Y, X)``: derivatives at ``sigma``,
    integration at ``rho``. ``weights (Z, Y, X)`` in [0, 1], if given, excises gradients with a normalised
    integration (see :func:`excised_structure_tensor_3d`)."""
    v = _to_torch(volume, device, dtype)
    Vx, Vy, Vz = _derivatives(v, sigma, truncate)
    w = None if weights is None else _to_torch(weights, device, dtype)
    S = _integrate((Vx * Vx, Vy * Vy, Vz * Vz, Vx * Vy, Vx * Vz, Vy * Vz), rho, truncate, w)
    return _like(S, volume)


def excised_structure_tensor_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, factor: float = 3.0,
                                sigma_coarse: float = 3.0, gate=None, mask=None, device: str = "cuda",
                                dtype=torch.float32):
    """Fix A: gradients whose coarse (``sigma_coarse``) magnitude exceeds ``factor`` x the reference median are left
    out of the integration, which is renormalised by the surviving weight. The reference median is taken over
    ``mask (Z, Y, X)`` (a white-matter mask; the whole volume if None -- too low a threshold where much of the volume
    is smooth grey matter or background). ``gate (Z, Y, X)`` in [0, 1] restricts the excision to where it is > 0
    (e.g. near detected interfaces). Returns ``S`` as :func:`structure_tensor_3d`."""
    v = _to_torch(volume, device, dtype)
    g = None if gate is None else _to_torch(gate, device, dtype)
    m = None if mask is None else _to_torch(mask, device, dtype) > 0.5
    w = excision_weights(v, sigma_coarse=sigma_coarse, factor=factor, truncate=truncate, mask=m, gate=g)
    Vx, Vy, Vz = _derivatives(v, sigma, truncate)
    S = _integrate((Vx * Vx, Vy * Vy, Vz * Vz, Vx * Vy, Vx * Vz, Vy * Vz), rho, truncate, w)
    return _like(S, volume)


def shape_structure_tensor_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, engine: str = "anatomix48",
                              n_pcs: int = 16, engine_kwargs: dict | None = None, device: str = "cuda",
                              dtype=torch.float32):
    """``S`` summed over the gradients of a feature engine's channels (``sum_c grad f_c grad f_c^T``, integrated at
    ``rho``), in the layout of :func:`structure_tensor_3d`. The engine needs its own context margin on the volume
    (64 voxels for the anatomix engines)."""
    from .features import make_engine
    from .pca import PCA
    v = _to_torch(volume, device, torch.float32)
    eng = make_engine(engine, **(engine_kwargs or {}))
    F = eng(v)
    if F.shape[0] > n_pcs:
        F = PCA.fit(F, n_pcs)(F)
    prods = [torch.zeros(tuple(v.shape), dtype=dtype, device=device) for _ in range(6)]
    for c in range(F.shape[0]):
        Vx, Vy, Vz = _derivatives(F[c].float(), sigma, truncate)
        for k, (a, b) in enumerate(((Vx, Vx), (Vy, Vy), (Vz, Vz), (Vx, Vy), (Vx, Vz), (Vy, Vz))):
            prods[k] += (a * b).to(dtype)
    del F
    S = _integrate(prods, rho, truncate, None)
    return _like(S, volume)


def eig_special_3d(S, full: bool = False, eigenvalue_order: Literal["desc", "asc"] = "desc", *, device: str = "cuda",
                   chunk: int = 1 << 20):       # cusolver's batched Jacobi solver rejects batches of ~2M and above
    """Eigensolution of the symmetric 3 x 3 tensors in ``S (6, ...)``, as the ``structure-tensor`` package returns it:
    ``val (3, ...)`` (descending unless ``eigenvalue_order='asc'``) and ``vec (3, ...)`` = the unit eigenvector of the
    smallest eigenvalue in ``(x, y, z)`` component order, or with ``full=True`` ``(3, 3, ...)`` with ``vec[i]`` the
    eigenvector of ``val[i]``. Batched ``torch.linalg.eigh`` on the device in chunks of ``chunk`` tensors."""
    St = _to_torch(S, device)
    shape = tuple(St.shape[1:])
    Sf = St.reshape(6, -1)
    N = Sf.shape[1]
    val = torch.empty((3, N), dtype=St.dtype, device=St.device)
    vec = torch.empty(((3, 3, N) if full else (3, N)), dtype=St.dtype, device=St.device)
    for a in range(0, N, chunk):
        s = Sf[:, a:a + chunk]
        M = torch.stack([torch.stack([s[0], s[3], s[4]], -1), torch.stack([s[3], s[1], s[5]], -1),
                         torch.stack([s[4], s[5], s[2]], -1)], -2)                           # (n, 3, 3), ascending eigh
        w, V = torch.linalg.eigh(M)
        if eigenvalue_order == "desc":
            w, V = w.flip(-1), V.flip(-1)
        val[:, a:a + chunk] = w.T
        if full:
            vec[:, :, a:a + chunk] = V.permute(2, 1, 0)                                      # [i, component, n]
        else:
            k = 0 if eigenvalue_order == "asc" else 2
            vec[:, a:a + chunk] = V[:, :, k].T
    val = val.reshape((3,) + shape)
    vec = vec.reshape(((3, 3) if full else (3,)) + shape)
    return _like(val, S), _like(vec, S)


# ---------------------------------------------------------------------------------------------------------------
# the one-call API
# ---------------------------------------------------------------------------------------------------------------

def shapetensor(image, *, sigma: float = 1.0, rho: float = 16.0, truncate: float = 4.0, engine: str = "intensity",
                excise: bool = True, excise_mask=None, fixels: bool = False, order: str = "xyz", n_pcs: int = 16,
                engine_kwargs: dict | None = None, device: str = "cuda", **fixel_kwargs):
    """The fibre direction field of ``image (Z, Y, X)`` in one call, as the caller's array kind.

    Returns ``vec (3, Z, Y, X)``: the unit fibre direction per voxel (axial), components in ``order`` (``"xyz"`` as
    the structure-tensor package, or ``"zyx"``). With ``fixels=True`` returns ``(3, 3, Z, Y, X)``: up to three
    directions per voxel from the CSD of the gradient-direction ODF (zero where absent), strongest first.

    ``engine``: ``"intensity"`` (the image's own gradients; the default), or a feature engine (``"jet"``,
    ``"anatomix16"``, ``"anatomix48"``) whose channel gradients are summed instead. ``excise``: drop boundary
    gradients before integrating (the boundary-robust tensor; intensity engine only), with the excision threshold
    referred to the median over ``excise_mask (Z, Y, X)`` (a white-matter mask) when given. ``sigma`` / ``rho`` in
    voxels.
    """
    if fixels:
        dirs, _, _ = fixels_3d(image, sigma, rho, truncate, device=device, **fixel_kwargs)
        return dirs if order == "xyz" else _like(_to_torch(dirs, device).flip(1), image)
    if engine == "intensity":
        if excise:
            S = excised_structure_tensor_3d(image, sigma, rho, truncate, mask=excise_mask, device=device)
        else:
            S = structure_tensor_3d(image, sigma, rho, truncate, device=device)
    else:
        S = shape_structure_tensor_3d(image, sigma, rho, truncate, engine=engine, n_pcs=n_pcs,
                                      engine_kwargs=engine_kwargs, device=device)
    _, vec = eig_special_3d(S, device=device)
    if order == "zyx":
        vec = _like(_to_torch(vec, device).flip(0), image)
    return vec


# ---------------------------------------------------------------------------------------------------------------
# several directions per site
# ---------------------------------------------------------------------------------------------------------------

def fixels_3d(volume, sigma: float, rho: float, truncate: float = 4.0, *, lmax: int = 8, reduce: int | None = None,
              csd_width_deg: float = 20.0, peak_rel: float = 0.3, min_sep_deg: float = 25.0, nmax: int = 3,
              upsample: bool = True, device: str = "cuda"):
    """Up to ``nmax`` axial directions with fractions per site from the CSD of the gradient-direction ODF.

    Returns ``dirs (nmax, 3, ...)`` in ``(x, y, z)`` component order (zero where absent), ``fracs (nmax, ...)`` and
    ``npop (...)``, on the ``reduce``-voxel site grid (default ``rho / 4``) or nearest-upsampled to the volume grid.
    """
    from .odf import csd, peaks, ring_profile, zonal_kernel
    v = _to_torch(volume, device, torch.float32)
    Z, Y, X = v.shape
    b = reduce or max(1, int(rho // 4))
    core = ((0, Z // b * b), (0, Y // b * b), (0, X // b * b))
    G = gradients(v[None], sigma, truncate)
    O, sites = odf_dense(G, rho, truncate, b, core, lmax=lmax)
    del G
    kern = zonal_kernel(lmax, ring_profile(math.radians(csd_width_deg)))
    f = csd(O.reshape(-1, O.shape[-1]), lmax, kern)
    d, a, cnt = peaks(f, lmax, nmax=nmax, rel=peak_rel, min_sep_deg=min_sep_deg)
    frac = a / a.sum(1, keepdim=True).clamp_min(1e-30)
    shp = sites.shape
    dirs = d.reshape(shp + (nmax, 3)).permute(3, 4, 0, 1, 2).flip(1).contiguous()       # (nmax, (x,y,z), nz, ny, nx)
    fracs = frac.reshape(shp + (nmax,)).permute(3, 0, 1, 2).contiguous()
    npop = cnt.reshape(shp)
    if upsample and b > 1:
        import torch.nn.functional as Fn
        up = lambda t: Fn.interpolate(t.reshape(-1, 1, *shp).float(), scale_factor=(b, b, b), mode="nearest")[:, 0]
        dirs = up(dirs).reshape(nmax, 3, *(s * b for s in shp))
        fracs = up(fracs).reshape(nmax, *(s * b for s in shp))
        npop = up(npop[None])[0].to(torch.int64).reshape(*(s * b for s in shp))
        pad = [(0, n - s * b) for n, s in zip((Z, Y, X), shp)]
        if any(p[1] for p in pad):
            pz, py, px = pad
            dirs = torch.nn.functional.pad(dirs, (0, px[1], 0, py[1], 0, pz[1]))
            fracs = torch.nn.functional.pad(fracs, (0, px[1], 0, py[1], 0, pz[1]))
            npop = torch.nn.functional.pad(npop, (0, px[1], 0, py[1], 0, pz[1]))
    return _like(dirs, volume), _like(fracs, volume), _like(npop, volume)


# ---------------------------------------------------------------------------------------------------------------
# a chunked driver for whole volumes
# ---------------------------------------------------------------------------------------------------------------

def run_volume(src: str, dst: str, *, sigma: float = 1.0, rho: float = 16.0, truncate: float = 4.0,
               chunk=(256, 256, 256), outputs=("S", "eig", "excised", "fixels"), excise_factor: float = 3.0,
               fixel_kwargs: dict | None = None, level: str | None = None, rank: int = 0, world: int = 1,
               device: str = "cuda", log=print):
    """Compute the requested outputs for a whole zarr volume, chunk by chunk with halos, on one GPU; several
    processes split the chunk list by ``rank`` / ``world``. ``src``: a zarr array, or a group with ``level`` (an
    OME-Zarr level such as ``"0"``). ``dst``: a zarr group that receives ``S`` and ``S_excised (6, Z, Y, X)``,
    ``val (3, ...)``, ``vec (3, ...)``, ``vec_excised``, ``dirs (3, 3, ...)``, ``fracs (3, ...)``, ``npop`` as asked.
    Halos make every chunk's output identical to a one-pass computation (the halo covers the filters' reach).
    """
    import zarr
    a = zarr.open(src, mode="r")
    if level is not None:
        a = a[level]
    Z, Y, X = a.shape
    g = zarr.open_group(dst, mode="a")
    cz, cy, cx = chunk
    halo = int(truncate * (sigma + rho) + 0.5) + (int(3.0 * truncate + 0.5) if "excised" in outputs else 0)
    specs = {}
    if "S" in outputs:
        specs["S"] = ((6, Z, Y, X), np.float32)
    if "excised" in outputs:
        specs["S_excised"] = ((6, Z, Y, X), np.float32)
    if "eig" in outputs:
        specs["val"] = ((3, Z, Y, X), np.float32)
        specs["vec"] = ((3, Z, Y, X), np.float32)
        if "excised" in outputs:
            specs["vec_excised"] = ((3, Z, Y, X), np.float32)
    if "fixels" in outputs:
        specs["dirs"] = ((3, 3, Z, Y, X), np.float16)
        specs["fracs"] = ((3, Z, Y, X), np.float16)
        specs["npop"] = ((Z, Y, X), np.uint8)
    for name, (shape, dt) in specs.items():
        if name not in g:
            g.create_array(name, shape=shape, dtype=dt, chunks=(shape[:-3] + (cz, cy, cx)))
    g.attrs.update({"sigma": sigma, "rho": rho, "truncate": truncate, "excise_factor": excise_factor, "halo": halo,
                    "layout": "S = [xx, yy, zz, xy, xz, yz], x = last axis; vec in (x, y, z)"})
    tiles = [(z0, y0, x0) for z0 in range(0, Z, cz) for y0 in range(0, Y, cy) for x0 in range(0, X, cx)]
    for i, (z0, y0, x0) in enumerate(tiles):
        if i % world != rank:
            continue
        z1, y1, x1 = min(Z, z0 + cz), min(Y, y0 + cy), min(X, x0 + cx)
        lo = (max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo))
        hi = (min(Z, z1 + halo), min(Y, y1 + halo), min(X, x1 + halo))
        blk = torch.as_tensor(np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32), device=device)
        core = (slice(z0 - lo[0], z1 - lo[0]), slice(y0 - lo[1], y1 - lo[1]), slice(x0 - lo[2], x1 - lo[2]))
        out_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        if "S" in outputs or "eig" in outputs:
            S = structure_tensor_3d(blk, sigma, rho, truncate, device=device)
            if "S" in outputs:
                g["S"][(slice(None),) + out_sl] = S[(slice(None),) + core].cpu().numpy()
            if "eig" in outputs:
                val, vec = eig_special_3d(S[(slice(None),) + core].contiguous(), device=device)
                g["val"][(slice(None),) + out_sl] = val.cpu().numpy()
                g["vec"][(slice(None),) + out_sl] = vec.cpu().numpy()
            del S
        if "excised" in outputs:
            Sx = excised_structure_tensor_3d(blk, sigma, rho, truncate, factor=excise_factor, device=device)
            g["S_excised"][(slice(None),) + out_sl] = Sx[(slice(None),) + core].cpu().numpy()
            if "eig" in outputs:
                _, vecx = eig_special_3d(Sx[(slice(None),) + core].contiguous(), device=device)
                g["vec_excised"][(slice(None),) + out_sl] = vecx.cpu().numpy()
            del Sx
        if "fixels" in outputs:
            dirs, fracs, npop = fixels_3d(blk, sigma, rho, truncate, device=device, **(fixel_kwargs or {}))
            g["dirs"][(slice(None), slice(None)) + out_sl] = dirs[(slice(None), slice(None)) + core].half().cpu().numpy()
            g["fracs"][(slice(None),) + out_sl] = fracs[(slice(None),) + core].half().cpu().numpy()
            g["npop"][out_sl] = npop[core].to(torch.uint8).cpu().numpy()
            del dirs, fracs, npop
        del blk
        torch.cuda.empty_cache()
        log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}) done")
    return dst
