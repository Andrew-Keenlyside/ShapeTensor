"""The orientation field of a block: the package's product, computed densely on a coarse site grid and written as a
zarr group a downstream consumer can sample. Its design follows the validation (docs/TCA_FIELD_RESULTS.md):

* the single direction is the **intensity structure tensor with gradient excision** (fix A: gradients whose coarse
  magnitude exceeds ``excise_factor`` x the block median are dropped and the integration renormalised), which removes
  the interface shell at no cost in clear tissue; the plain tensor is kept beside it;
* the boundary field is the coarse gradient magnitude ``gm`` (the best detector) with Westin's ``cl`` (sheet
  dominance: the gate for "the orientation is at risk") and the normal ``n_cl``;
* the N-population field is the **CSD of the gradient-direction ODF** (peaks of the deconvolved fODF, a ring
  response of width ``csd_width_deg``), up to three directions with fractions from the peak amplitudes;
* optionally a shape-engine TCA direction (``shape_engine``) for comparison; it is not the recommended source.

Arrays on the site grid ``(nz, ny, nx)``, ``reduce`` voxels per site; vectors ``(3, nz, ny, nx)`` in (z, y, x) order,
axial, canonical sign (largest |component| positive):
``dir_st``, ``dir`` (excised), ``gap_st``, ``gap`` (quality: (λ2−λ3)/λ2), ``cl``, ``n_cl``, ``gm``, ``excised_share``,
``dirs (3, 3, …)``, ``fracs (3, …)``, ``npop``, and with a shape engine ``dir_shape``, ``gap_shape``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..features import make_engine
from ..filters import gaussian
from .odf import csd, peaks, ring_profile, zonal_kernel
from ..pca import PCA
from . import common, single
from .tensor import excision_weights, gradients, k_dense, odf_dense


def compute_field(vol: torch.Tensor, rho: float = 16.0, reduce: int | None = None, sigma: float = 1.0,
                  truncate: float = 3.0, margin: int | None = None, excise: bool = True, excise_factor: float = 3.0,
                  multi: bool = True, lmax: int = 8, csd_width_deg: float = 20.0, peak_rel: float = 0.3,
                  min_sep_deg: float = 25.0, shape_engine: str | None = None, n_pcs: int = 16,
                  engine_kwargs: dict | None = None, device: str = "cuda") -> dict:
    """``vol (Z, Y, X)`` grey levels with ``margin`` voxels of context on every face (default: the rho window plus
    the engine's own margin) -> the field on the core's site grid."""
    dev = device
    v = torch.as_tensor(vol, device=dev).float()
    Z, Y, X = v.shape
    b = reduce or max(1, int(rho // 4))
    eng = make_engine(shape_engine, **(engine_kwargs or {})) if shape_engine else None
    m = margin if margin is not None else int(truncate * rho + 0.5) + (getattr(eng, "margin", 0) if eng else 0)
    m = int(np.ceil(m / b) * b)
    core = ((m, (Z - m) // b * b), (m, (Y - m) // b * b), (m, (X - m) // b * b))
    out = {}
    Gi = gradients(v[None], sigma, truncate)
    # the plain tensor
    Ki, sites = k_dense(Gi, rho, truncate, b, core)
    e = single.eigen(common.unpack(Ki.reshape(-1, 6), 3))
    out.update({"dir_st": e["dir"], "gap_st": e["gap"], "cl": e["cl"],
                "n_cl": common.canonical_sign(common.eigh_desc(common.channel_sum(common.unpack(Ki.reshape(-1, 6), 3)))[1][..., :, 0])})
    del Ki
    # the boundary detector: coarse gradient magnitude over the window
    gm = sum(gaussian(v, 3.0, order=o, truncate=truncate) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    Kg, _ = k_dense(gm[None].half(), rho, truncate, b, core)
    out["gm"] = Kg.reshape(-1, Kg.shape[-1])[:, 0].clamp_min(0).sqrt()     # one map -> one packed entry (its square)
    del Kg
    # the excised tensor (fix A)
    if excise:
        w = excision_weights(v, sigma_coarse=3.0, factor=excise_factor, truncate=truncate)
        Kx, _ = k_dense(Gi, rho, truncate, b, core, weights=w.half())
        ex = single.eigen(common.unpack(Kx.reshape(-1, 6), 3))
        out.update({"dir": ex["dir"], "gap": ex["gap"], "excised_share": torch.full_like(ex["gap"], float(1.0 - w.mean()))})
        del Kx, w
    else:
        out.update({"dir": out["dir_st"], "gap": out["gap_st"]})
    del gm
    # N populations: CSD of the intensity gradient-direction ODF
    if multi:
        O, _ = odf_dense(Gi, rho, truncate, b, core, lmax=lmax)
        kern = zonal_kernel(lmax, ring_profile(math.radians(csd_width_deg)))
        f = csd(O.reshape(-1, O.shape[-1]), lmax, kern)
        d, a, cnt = peaks(f, lmax, nmax=3, rel=peak_rel, min_sep_deg=min_sep_deg)
        out.update({"dirs": common.canonical_sign(d), "fracs": a / a.sum(1, keepdim=True).clamp_min(1e-30), "npop": cnt})
        del O, f
    del Gi
    # optional: a shape-engine TCA direction
    if eng is not None:
        F = eng(v)
        if F.shape[0] > n_pcs:
            F = PCA.fit(F, n_pcs)(F)
        G = gradients(F, sigma, truncate)
        del F
        Ks, _ = k_dense(G, rho, truncate, b, core)
        del G
        Ksh = common.unpack(Ks.reshape(-1, Ks.shape[-1]), common.nc3_from_packed(Ks.shape[-1]))
        del Ks
        tc = single.tca(Ksh, rank=3)
        out.update({"dir_shape": tc["dir"], "gap_shape": tc["gap"]})
        del Ksh
    shape = sites.shape
    res = {"sites": sites, "shape": shape, "reduce": b, "rho": rho, "core": core, "excise": excise,
           "csd_width_deg": csd_width_deg, "peak_rel": peak_rel, "engine": eng.describe() if eng else None}
    for k, t in out.items():
        a = t.detach().cpu().numpy()
        if a.ndim == 1:
            res[k] = a.reshape(shape)
        elif a.ndim == 2 and a.shape[1] == 3:
            res[k] = a.reshape(shape + (3,)).transpose(3, 0, 1, 2)
        elif a.ndim == 2:
            res[k] = a.reshape(shape + (a.shape[1],)).transpose(3, 0, 1, 2)
        elif a.ndim == 3:                                        # (N, k, 3) -> (k, 3, nz, ny, nx)
            res[k] = a.reshape(shape + a.shape[1:]).transpose(3, 4, 0, 1, 2)
    return res


def write_store(path: str | Path, field: dict, voxel_mm: float, block_origin_vox=(0, 0, 0)) -> Path:
    """Write the field as a zarr v3 group with one array per entry and the grid in the attributes."""
    import zarr
    p = Path(path)
    g = zarr.open_group(str(p), mode="w")
    sites = field["sites"]
    attrs = {"voxel_mm": float(voxel_mm), "reduce": int(field["reduce"]), "rho_vox": float(field["rho"]),
             "excise": bool(field["excise"]), "csd_width_deg": field["csd_width_deg"], "peak_rel": field["peak_rel"],
             "engine": field.get("engine"),
             "site_origin_vox": [float(sites.z[0] + block_origin_vox[0]), float(sites.y[0] + block_origin_vox[1]),
                                 float(sites.x[0] + block_origin_vox[2])],
             "axes": "zyx", "vectors": "axial unit, (z, y, x), canonical sign (largest |component| positive)"}
    g.attrs.update(json.loads(json.dumps(attrs)))
    for k, a in field.items():
        if isinstance(a, np.ndarray):
            g.create_array(k, data=a.astype(np.float16 if a.dtype == np.float32 else a.dtype))
    return p
