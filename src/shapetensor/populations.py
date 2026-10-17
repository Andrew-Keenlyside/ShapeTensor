"""The pipeline's stages 2–4 on one block in one call, self-calibrated on that block: shape features → the
per-voxel orientation distribution and the signature dictionary (teacher: the excised structure tensor at its
confident sites, sampled evenly over orientation; no labels) → the block's populations (modes confirmed by their
masks) → one activation layer per population → one direction field per layer → the population masks, their union,
the strongest-population map, and the activation-weighted combined vector field (the single-field alternative to
the structure tensor).

    out = population_field(volume, rho=10.0)                     # dict of arrays, (z, y, x) component order
    vec = out["vec_combined"]; masks = out["masks"]; dirs = out["dirs"]
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from . import st
from .features import make_engine
from .readout.layers import (SignatureDictionary, activation_null, combined_field, layer_tensor_direction, population_masks, unmix)
from .readout.orientation import OrientationReadoutDist, hemisphere, modes_from_density


def _stratified_weights(d, gap, tissue, q=0.7, n_bins=24):
    u = hemisphere(n_bins, d.device)
    conf = tissue & (gap > _q(gap[tissue], q))
    idx = conf.nonzero()
    dv = d[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    b = ((dv @ u.T) ** 2).argmax(1)
    counts = torch.bincount(b, minlength=len(u)).float()
    w = torch.zeros_like(gap); w[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0 / counts[b].clamp_min(1)
    return w


def _q(x, q, n=2_000_000):
    x = x.reshape(-1)
    if x.numel() > n:
        x = x[torch.randint(0, x.numel(), (n,), device=x.device)]
    return torch.quantile(x.float(), q)


def population_field(volume, *, engine: str = "anatomix_enc2", rho: float = 10.0, sigma: float = 1.0, truncate: float = 4.0,
                     rho_teacher: float = 8.0, tissue=None, nmax: int = 3, smooth: int = 3, margin: int = 24, mask_threshold: float | None = None,
                     min_mask_share: float = 0.05, engine_kwargs: dict | None = None, device: str = "cuda") -> dict:
    """``volume (Z, Y, X)``; ``tissue`` an optional bool mask of the tissue to read (default: all). Returns a dict:
    ``modes (N, 3)``, ``activation (N, Z, Y, X)``, ``dirs (N, 3, Z, Y, X)``, ``gap (N, Z, Y, X)``, ``masks (N, Z, Y, X)``,
    ``union_mask``, ``strongest``, ``vec_combined (3, Z, Y, X)``, ``weight_combined``, ``residual``, ``tau``, ``n_candidates``."""
    v = st._to_torch(volume, device, torch.float32)
    tis = torch.ones_like(v, dtype=torch.bool) if tissue is None else st._to_torch(tissue, device) > 0.5
    F = make_engine(engine, **({"nyquist": "none"} | (engine_kwargs or {})))(v)
    # the teacher and the calibration sets, on this block
    S = st.excised_structure_tensor_3d(v, sigma, rho_teacher, truncate, device=device)
    val, vec = st.eig_special_3d(S, device=device); d = vec.flip(0); gap = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S
    inner = tuple(slice(margin, n - margin) for n in v.shape)
    w = torch.zeros_like(gap); w[inner] = _stratified_weights(d[(slice(None),) + inner], gap[inner], tis[inner])
    wb = torch.zeros_like(gap); wb[inner] = ((gap[inner] < _q(gap[inner][tis[inner]], 0.3)) & tis[inner]).float()
    dic = SignatureDictionary(smooth=smooth).fit([(F, d, w)], background=[(F, wb)])
    ro = OrientationReadoutDist(smooth=smooth).fit([(F, d, w)], n_sample=240_000)
    tau = activation_null(dic, [(F, d, w)])
    # the populations: modes of the block's summed per-voxel distribution, confirmed by their masks
    P = ro.predict_dist(F)
    dens = (P * tis.float()[None]).reshape(P.shape[0], -1).sum(1); del P
    modes, h, n = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=nmax, mass_floor=0.1)
    keep = list(range(n))
    while keep:
        A, R = unmix(F, dic, modes[keep]); As = Fn.avg_pool3d(A[None], 3, stride=1, padding=1)[0]
        masks, union, strongest = population_masks(As, tau)
        share = torch.stack([(mk & tis).float().sum() / tis.float().sum().clamp_min(1) for mk in masks])
        worst = int(share.argmin())
        if float(share[worst]) >= min_mask_share or len(keep) == 1:
            break
        keep.pop(worst)
    modes = modes[keep]
    dirs, gaps = [], []
    for k in range(len(keep)):
        dl, gl = layer_tensor_direction(A[k], sigma, rho, truncate); dirs.append(dl); gaps.append(gl)
    dirs = torch.stack(dirs); gaps = torch.stack(gaps)
    vec_c, wgt_c = combined_field(dirs, A, mask_threshold if mask_threshold is None else mask_threshold, smooth=3)
    return {"modes": modes, "activation": A, "dirs": dirs, "gap": gaps, "masks": masks, "union_mask": union, "strongest": strongest,
            "vec_combined": vec_c, "weight_combined": wgt_c, "residual": R, "tau": tau, "n_candidates": int(n)}
