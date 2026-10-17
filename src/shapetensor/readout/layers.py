"""Per-population layers of the shape space (stage 2 → stage 3, in the pipeline's own terms).

A fibre population excites a characteristic pattern of channels in an orientation-selective shape space — its
*signature*. The feature vector at a voxel is then a non-negative mixture of the signatures of the populations
present there: ``f(v) ≈ Σ_r a_r(v) S_r + residual``. Unmixing gives, per population, a full-volume **activation**
``a_r(v)`` — that population's own image, read from the shape space — and its feature layer ``a_r(v) S_r``; the
gradients of the layer are ``∇a_r ⊗ S_r``, so the population's structure tensor is the structure tensor of its
activation volume. Nothing is masked: a crossing voxel belongs to both layers in proportion to how strongly each
population's channels respond there.

The dictionary of signatures is built without labels, from the same self-calibration as the orientation readout:
the mean standardised feature vector of calibration voxels whose teacher orientation (the excised tensor at its
confident sites) lies near each direction of a hemisphere grid.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from .common import eigh
from .orientation import _sample_idx, hemisphere


class SignatureDictionary:
    """Mean standardised feature pattern per orientation (a hemisphere grid), from ``(F, teacher dirs, weights)`` blocks."""

    def __init__(self, smooth: int = 3, n_dirs: int = 160, kernel_deg: float = 12.0, transform: str = "none"):
        """``transform``: a compression of the rectified features before the dictionary -- "none", "sqrt" (√ of the
        non-negative part), "log1p" -- or "unit" (each voxel's standardised vector divided by its norm after
        standardisation: the pattern only, not the contrast amplitude)."""
        self.smooth, self.n_dirs, self.kernel_deg, self.transform = int(smooth), int(n_dirs), float(kernel_deg), transform
        self.mean = self.std = self.grid = self.D = None

    def _prep(self, F):
        f = F.float()
        if self.transform == "sqrt":
            f = f.clamp_min(0).sqrt()
        elif self.transform == "log1p":
            f = torch.log1p(f.clamp_min(0))
        if self.smooth > 1:
            f = Fn.avg_pool3d(f[None], self.smooth, stride=1, padding=self.smooth // 2)[0]
        return f

    def standardise(self, X):
        """Standardised rows ``(M, C)``; with transform "unit" also divided by each row's norm."""
        Xn = (X - self.mean) / self.std
        if self.transform == "unit":
            Xn = Xn / Xn.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return Xn

    def fit(self, blocks, n_sample: int = 300_000, seed: int = 0, background=None):
        """``blocks``: ``(F, teacher dirs, weights)`` for the signatures; ``background``: an optional list of
        ``(F, weights)`` whose voxels hold no fibre (e.g. the lowest teacher gap) -> a nuisance signature that absorbs
        what is not a population (background, interfaces) in the unmixing."""
        g = torch.Generator().manual_seed(seed)
        Xs, Ds = [], []
        per = max(1, n_sample // max(1, len(blocks)))
        for F, d, w in blocks:
            f = self._prep(F)
            wv = w.reshape(-1).float().cpu()
            idx = _sample_idx(wv, per, g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T); Ds.append(d.reshape(3, -1)[:, idx].T)
        X = torch.cat(Xs); Dd = torch.cat(Ds)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = self.standardise(X)
        self.grid = hemisphere(self.n_dirs, X.device)
        k = 1.0 / (2 * math.sin(math.radians(self.kernel_deg)) ** 2)
        W = torch.exp(k * ((Dd @ self.grid.T) ** 2 - 1))                                   # (N, n_dirs) kernel weights
        self.D = (W.T @ Xn) / W.sum(0)[:, None].clamp_min(1e-6)                            # (n_dirs, C) signatures
        self.background = None
        if background:
            Bs = []
            for F, w in background:
                f = self._prep(F); wv = w.reshape(-1).float().cpu()
                idx = _sample_idx(wv, per, g).to(F.device)
                Bs.append(f.reshape(f.shape[0], -1)[:, idx].T)
            self.background = self.standardise(torch.cat(Bs)).mean(0)
        return self

    def signature(self, direction: torch.Tensor) -> torch.Tensor:
        """The signature of an arbitrary direction (kernel-interpolated over the grid)."""
        k = 1.0 / (2 * math.sin(math.radians(self.kernel_deg)) ** 2)
        w = torch.exp(k * ((self.grid @ direction) ** 2 - 1))
        return (w[:, None] * self.D).sum(0) / w.sum().clamp_min(1e-6)


def nnls_batched(A: torch.Tensor, Y: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """Non-negative least squares ``min ||A x - y||²`` for many ``y``: ``A (C, N)``, ``Y (M, C)`` -> ``X (M, N)`` ≥ 0,
    by projected gradient with the exact Lipschitz step (small N, so this converges in tens of iterations)."""
    AtA = A.T @ A; AtY = Y @ A                                                              # (N, N), (M, N)
    L = float(torch.linalg.eigvalsh(AtA).max().clamp_min(1e-9))
    X = (AtY @ torch.linalg.pinv(AtA)).clamp_min(0)                                        # a warm start from the unconstrained (normal-equation) solution
    for _ in range(iters):
        X = (X - (X @ AtA - AtY) / L).clamp_min(0)
    return X


def unmix(F: torch.Tensor, dictionary: SignatureDictionary, modes: torch.Tensor, z_chunk: int = 16, iters: int = 60):
    """Per-voxel non-negative unmixing of the (standardised, smoothed) features onto the signatures of ``modes
    (N, 3)``: activations ``A (N, Z, Y, X)`` ≥ 0 and the residual energy share ``R (Z, Y, X)`` in [0, 1]."""
    f = dictionary._prep(F)
    C, Z, Y, X = f.shape
    S = torch.stack([dictionary.signature(m) for m in modes], 1)                            # (C, N)
    N = S.shape[1]
    if dictionary.background is not None:
        S = torch.cat([S, dictionary.background[:, None]], 1)                                 # + the nuisance signature
    A_out = torch.empty((N, Z, Y, X), device=F.device); R_out = torch.empty((Z, Y, X), device=F.device)
    for z in range(0, Z, z_chunk):
        fz = f[:, z:z + z_chunk]
        Xn = dictionary.standardise(fz.reshape(C, -1).T)                                       # (M, C)
        a = nnls_batched(S, Xn, iters)                                                       # (M, N)
        rec = a @ S.T
        res = ((Xn - rec) ** 2).sum(1) / (Xn ** 2).sum(1).clamp_min(1e-9)
        A_out[:, z:z + z_chunk] = a[:, :N].T.reshape(N, *fz.shape[1:]); R_out[z:z + z_chunk] = res.clamp(0, 1).reshape(fz.shape[1:])
    return A_out, R_out


def compress_activation(A: torch.Tensor, how: str = "none") -> torch.Tensor:
    """A pointwise compression of activations: "none", "sqrt" (√a) or "log" (log1p(10 a)). Monotone, so masks
    and argmax are unchanged; it flattens the peaks for weighting and display but cannot remove any spatial
    frequency -- the fibre-period oscillation of the activation stays (results log §25)."""
    A = A.clamp_min(0)
    if how == "none":
        return A
    if how == "sqrt":
        return A.sqrt()
    if how == "log":
        return torch.log1p(10.0 * A)
    raise ValueError(f"unknown activation compression {how!r}")


def layer_tensor_direction(a: torch.Tensor, sigma: float, rho: float, truncate: float = 4.0, return_tensor: bool = False):
    """The structure tensor of one activation volume ``a (Z, Y, X)`` -> its direction field ``(3, Z, Y, X)`` in
    (z, y, x) order and the orientation gap: the population's direction from its own layer's gradients. With
    ``return_tensor`` also the tensor itself ``(6, Z, Y, X)`` [xx, yy, zz, xy, xz, yz]."""
    from .. import st
    S = st.structure_tensor_3d(a, sigma, rho, truncate)
    val, vec = st.eig_special_3d(S)
    d, gap = vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
    return (d, gap, S) if return_tensor else (d, gap)


def combined_tensor_field(S: torch.Tensor, A: torch.Tensor, mask_threshold: float | None = None, smooth: int = 3):
    """The merge of the population tensors: ``S (N, 6, Z, Y, X)`` the layers' structure tensors, ``A (N, Z, Y, X)``
    the activations -> the single direction ``(3, Z, Y, X)`` (z, y, x) of ``Σ_r a_r S_r / tr(S_r)``, its gap and its
    weight (the total activation). An incoherent layer has a near-isotropic tensor and adds no direction, so it
    is discounted without a rule; this is the shape tensor's single field. ``mask_threshold`` as in
    :func:`combined_field`."""
    from .. import st
    Aw = A.clamp_min(0)
    tr = S[:, :3].sum(1).clamp_min(1e-12)
    T = (S * (Aw / tr)[:, None]).sum(0)
    val, vec = st.eig_special_3d(T)
    d, gap = vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
    weight = Aw.sum(0)
    if mask_threshold is not None:
        keep = (Fn.avg_pool3d(Aw[None], smooth, stride=1, padding=smooth // 2)[0] > mask_threshold).any(0)
        d = d * keep; weight = weight * keep; gap = gap * keep
    return d, gap, weight


# ---------------------------------------------------------------------------------------------------------------
# N populations, their masks, and the export of a population field
# ---------------------------------------------------------------------------------------------------------------

def activation_null(dictionary: SignatureDictionary, blocks, offsets_deg=(45.0, 60.0, 75.0), q: float = 0.8, per: int = 60_000, seed: int = 0, smooth: int = 3) -> float:
    """The activation a wrong population's signature reaches on single-population voxels: for each calibration block
    (one population, teacher direction d), unmix its fibre voxels onto {own signature, a signature rotated by each
    offset in the fibre plane} + background and take the ``q`` quantile of the rotated signatures' activations.
    A population's mask is where its (``smooth``³-averaged) activation exceeds this. Offsets near 90° are left out:
    in the double-angle code two signatures 90° apart are anti-parallel and their unmixing is degenerate."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for F, d, w in blocks:
        f = dictionary._prep(F); wv = w.reshape(-1).float().cpu()
        idx = _sample_idx(wv, per, g).to(F.device)
        Xn = dictionary.standardise(f.reshape(f.shape[0], -1)[:, idx].T)
        dd = d.reshape(3, -1)[:, idx].T
        _, V = torch.linalg.eigh(dd.T @ dd); d0 = V[:, -1]                                 # the axial mean: the principal direction (signs are arbitrary)
        # a perpendicular in-plane axis to rotate around: any unit vector orthogonal to d0
        ref = torch.tensor([0.0, 1.0, 0.0], device=d0.device) if abs(float(d0[1])) < 0.9 else torch.tensor([1.0, 0.0, 0.0], device=d0.device)
        e2 = torch.linalg.cross(d0, ref); e2 = e2 / e2.norm().clamp_min(1e-9)
        sigs = [dictionary.signature(d0)]
        for off in offsets_deg:
            r = math.radians(off); sigs.append(dictionary.signature(math.cos(r) * d0 + math.sin(r) * e2))
        S = torch.stack(sigs, 1)
        if dictionary.background is not None:
            S = torch.cat([S, dictionary.background[:, None]], 1)
        a = nnls_batched(S, Xn)
        vals.append(a[:, 1:1 + len(offsets_deg)].reshape(-1))
    v = torch.cat(vals)
    # the activations will be averaged over smooth^3 voxels before thresholding; the null of an average of n draws
    # of a one-sided noise is well below the per-voxel quantile -- approximate by averaging random groups
    if smooth > 1:
        n = smooth ** 3; m = (len(v) // n) * n
        v = v[torch.randperm(len(v), device=v.device)[:m]].reshape(-1, n).mean(1)
    return float(torch.quantile(v, q))


def population_masks(A: torch.Tensor, tau: float, min_voxels: int = 0):
    """Masks from the activations ``A (N, Z, Y, X)``: ``masks (N, Z, Y, X)`` = activation above the null ``tau``,
    ``union (Z, Y, X)`` = any population, ``strongest (Z, Y, X)`` int8 = 1 + the index of the largest activation
    inside the union (0 outside)."""
    masks = A > tau
    if min_voxels:
        for r in range(masks.shape[0]):
            if int(masks[r].sum()) < min_voxels:
                masks[r] = False
    union = masks.any(0)
    strongest = (A.argmax(0) + 1).to(torch.int8) * union.to(torch.int8)
    return masks, union, strongest


def detect_populations(P: torch.Tensor, grid: torch.Tensor, F: torch.Tensor, dictionary: SignatureDictionary, tau: float,
                       region=None, rel: float = 0.3, min_sep_deg: float = 25.0, nmax: int = 4, mass_floor: float = 0.1,
                       min_mask_share: float = 0.05):
    """N populations of a block without specifying N: the modes of the summed per-voxel distribution ``P (n_dirs, Z, Y,
    X)`` over ``region`` (a bool volume; default all), then a confirmation by the layers -- each candidate is unmixed
    with the others and kept only if its mask (activation above ``tau``) covers at least ``min_mask_share`` of the
    region. Returns the kept modes ``(N, 3)``, the activations ``(N, Z, Y, X)``, the residual and the masks."""
    from .orientation import modes_from_density
    reg = torch.ones(P.shape[1:], dtype=torch.bool, device=P.device) if region is None else region
    dens = (P * reg.float()[None]).reshape(P.shape[0], -1).sum(1)
    modes, h, n = modes_from_density(dens, grid, rel=rel, min_sep_deg=min_sep_deg, nmax=nmax, mass_floor=mass_floor)
    keep = list(range(n))
    while keep:
        A, R = unmix(F, dictionary, modes[keep])
        A = Fn.avg_pool3d(A[None], 3, stride=1, padding=1)[0]                                  # the masks are taken on the 3^3-averaged activation
        masks, union, strongest = population_masks(A, tau)
        share = torch.stack([(m & reg).float().sum() / reg.float().sum().clamp_min(1) for m in masks])
        worst = int(share.argmin())
        if float(share[worst]) >= min_mask_share or len(keep) == 1:
            return modes[keep], A, R, masks, union, strongest, {"candidates": int(n), "heights": h[:n].tolist(), "mask_shares": share.tolist()}
        keep.pop(worst)
    return modes[:0], None, None, None, None, None, {"candidates": int(n), "heights": h[:n].tolist(), "mask_shares": []}


def export_population_field(path, modes: torch.Tensor, A: torch.Tensor, dirs: torch.Tensor, gaps: torch.Tensor, masks: torch.Tensor,
                            union: torch.Tensor, strongest: torch.Tensor, voxel_mm: float, attrs: dict | None = None):
    """Write the population field as a zarr group: ``modes (N, 3)`` (z, y, x), ``activation (N, Z, Y, X)`` float16,
    ``dirs (N, 3, Z, Y, X)`` float16 (each layer's own tensor direction, (z, y, x)), ``gap (N, Z, Y, X)``,
    ``masks (N, Z, Y, X)`` bool (where that population's vectors go), ``union_mask (Z, Y, X)`` bool (any population:
    a node-sampling mask), ``strongest (Z, Y, X)`` int8 (1 + the index of the strongest population, 0 outside); ``vec_combined (3, Z, Y, X)``
    and ``weight_combined`` — the activation-weighted average vector per voxel (:func:`combined_field`, unmasked);
    ``vec_multi (N, 3, Z, Y, X)``, ``weight_multi`` and ``n_vectors`` — :func:`multi_vector_field`."""
    import numpy as np, zarr
    g = zarr.open_group(str(path), mode="w")
    g.create_array("modes", data=modes.cpu().numpy().astype(np.float32))
    g.create_array("activation", data=A.cpu().numpy().astype(np.float16))
    g.create_array("dirs", data=dirs.cpu().numpy().astype(np.float16))
    g.create_array("gap", data=gaps.cpu().numpy().astype(np.float16))
    g.create_array("masks", data=masks.cpu().numpy()); g.create_array("union_mask", data=union.cpu().numpy()); g.create_array("strongest", data=strongest.cpu().numpy())
    vec, wgt = combined_field(dirs, A)                                                      # the single-field alternative to the structure tensor, unmasked
    g.create_array("vec_combined", data=vec.cpu().numpy().astype(np.float16)); g.create_array("weight_combined", data=wgt.cpu().numpy().astype(np.float16))
    vm, wm, cnt = multi_vector_field(dirs, A, masks)                                        # several vectors per voxel: both where the masks intersect, else the strongest
    g.create_array("vec_multi", data=vm.cpu().numpy().astype(np.float16)); g.create_array("weight_multi", data=wm.cpu().numpy().astype(np.float16)); g.create_array("n_vectors", data=cnt.cpu().numpy())
    g.attrs.update({"voxel_mm": voxel_mm, "n_populations": int(modes.shape[0]), "component_order": "zyx", **(attrs or {})})
    return path


def multi_vector_field(dirs: torch.Tensor, A: torch.Tensor, masks: torch.Tensor):
    """Several vectors per voxel from the population layers, with no threshold argument: where two or more
    population masks cover a voxel, every covering population's vector; where one or none does, the strongest
    activated population's vector alone. ``dirs (N, 3, Z, Y, X)``, ``A (N, Z, Y, X)``, ``masks (N, Z, Y, X)`` bool ->
    ``vec (N, 3, Z, Y, X)`` ordered by activation (zero where absent), ``weight (N, Z, Y, X)`` (the activation) and
    ``count (Z, Y, X)`` int8, the number of vectors at the voxel (always >= 1)."""
    Aw = A.clamp_min(0)
    order = Aw.argsort(0, descending=True)                                                    # (N, Z, Y, X)
    As = Aw.gather(0, order); Ms = masks.gather(0, order)
    Ds = dirs.gather(0, order[:, None].expand(-1, 3, -1, -1, -1))
    keep = Ms & (masks.sum(0) >= 2)[None]; keep[0] = True
    return Ds * keep[:, None], As * keep, keep.sum(0).to(torch.int8)


def combined_field(dirs: torch.Tensor, A: torch.Tensor, mask_threshold: float | None = None, smooth: int = 3, gaps: torch.Tensor | None = None):
    """One vector per voxel from the population layers: the activation-weighted axial average of the populations'
    directions (the principal eigenvector of ``Σ_r a_r d_r d_rᵀ``), with the total activation as its weight — the
    alternative to the single-vector structure tensor. ``dirs (N, 3, Z, Y, X)`` (z, y, x), ``A (N, Z, Y, X)``.
    With ``mask_threshold`` a voxel keeps its vector only where some population's ``smooth``³-averaged activation
    exceeds it (zero vector elsewhere); None keeps every voxel. Returns ``vec (3, Z, Y, X)``, ``weight (Z, Y, X)``."""
    from .orientation import outer6, tensor_from6
    N, _, Z, Y, X = dirs.shape
    Aw = A.clamp_min(0) * (gaps.clamp(0, 1) if gaps is not None else 1.0)                     # ``gaps``: weight each layer also by its own coherence
    T6 = (outer6(dirs.permute(0, 2, 3, 4, 1)) * Aw[..., None]).sum(0)                        # (Z, Y, X, 6)
    lam, V = eigh(tensor_from6(T6.reshape(-1, 6)))
    vec = V[..., :, 2].T.reshape(3, Z, Y, X)
    weight = Aw.sum(0)
    if mask_threshold is not None:
        As = Fn.avg_pool3d(Aw[None], smooth, stride=1, padding=smooth // 2)[0] if smooth > 1 else Aw
        keep = (As > mask_threshold).any(0)
        vec = vec * keep[None]; weight = weight * keep
    return vec, weight
