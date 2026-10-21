"""The classical pathway: anatomix features, then only linear algebra and statistics.

1. **Features** -- any engine (anatomix-dev-vit with blended tiles and the grid filter, or the UNet), averaged by the
   readers' box.
2. **Signatures** -- the calibration's :class:`~shapetensor.separate.SignatureDictionary`: each grid direction's mean
   standardised feature pattern (kernel-smoothed over directions) and a background pattern, by plain averaging over the
   voxels where the excised structure tensor is confident. Nothing is trained.
3. **Separation in shape space = sparse deconvolution** (:func:`odf`): the standardised features are integrated over
   ``rho`` (as the structure tensor integrates the gradients), and the local mean is written as a non-negative
   combination of every direction's signature and the background -- generalised least squares: features and signatures
   whitened by the covariance of the dictionary's own residuals on the calibration sites (the features' variation that
   is not orientation), then non-negative least squares (projected gradient with momentum, every voxel at once). The
   coefficients over the directions are the orientation distribution *of the shape* (constrained spherical
   deconvolution, with the features in place of the diffusion signal); if features mix linearly the local mean is the
   volume-fraction mixture of the signatures, and a crossing is two lobes.
4. **Presence** -- a goodness of fit: the share of the feature energy (beyond the background-only fit) the oriented
   signatures explain, integrated over ``rho`` (:func:`explained`).
5. **Populations = the distribution's lobes** (:func:`lobes`): its local maxima on the sphere above ``rel`` of the
   largest, at least ``min_sep_deg`` apart, at most ``nmax``.
6. **A tensor per population** -- each lobe's second moment ``sum_u a_u u u^T`` over its directions: the principal axis
   is the population's direction, the lobe's mass its weight.
7. **Merge** -- unchanged (:mod:`shapetensor.merge`).

The deconvolution runs on a lattice every ``stride`` voxels (the integration over ``rho`` makes the distribution smooth
at that scale) and the lobes' tensors are interpolated back to every voxel.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from . import st
from .filters import gaussian
from .merge import multi_field, single_field
from .tensors import direction_and_gap


def _design(dictionary) -> torch.Tensor:
    """``(C, n_dirs + 1)``: every grid direction's signature and the background, as columns."""
    cols = [dictionary.D.T]
    if dictionary.background is not None:
        cols.append(dictionary.background[:, None])
    return torch.cat(cols, 1).float()


def nnls_fista(A: torch.Tensor, Y: torch.Tensor, iters: int = 200, l1: float = 0.0) -> torch.Tensor:
    """``min_x ||A x - y||² / 2 + l1 sum x`` subject to ``x >= 0`` for every row ``y`` of ``Y (M, C)``; ``A (C, N)``.
    Accelerated projected gradient (FISTA) with the step ``1 / ||A^T A||``, all rows at once. Returns ``(M, N)``."""
    G = A.T @ A; B = Y @ A                                                                            # (N, N), (M, N)
    L = float(torch.linalg.eigvalsh(G.double()).max())
    eta = 1.0 / max(L, 1e-12)
    X = torch.zeros_like(B); Z = X.clone(); t = 1.0
    for _ in range(iters):
        Xn = (Z - eta * (Z @ G - B + l1)).clamp_min(0)
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        Z = Xn + ((t - 1) / tn) * (Xn - X)
        X, t = Xn, tn
    return X


def whitener(dictionary, shrink: float = 0.05) -> torch.Tensor:
    """``Σ^(-1/2)`` of the dictionary's residual covariance, shrunk towards its mean variance by ``shrink``."""
    S = dictionary.residual_cov.double()
    S = (1 - shrink) * S + shrink * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    w, V = torch.linalg.eigh(S)
    return (V @ torch.diag(w.clamp_min(1e-10) ** -0.5) @ V.T).float()


def odf(cal, Fp: torch.Tensor, *, stride: int = 2, rho: float = 10.0, truncate: float = 3.0, iters: int = 300, l1: float = 0.0,
        chunk: int = 200_000, whiten: bool = True, integrate_first: bool = True):
    """The shape-space orientation distribution on a lattice every ``stride`` voxels, ``(n_dirs, z, y, x)``, and the
    explained share ``(z, y, x)``. ``Fp``: the readers' averaged features ``(C, Z, Y, X)``. ``integrate_first``: the
    standardised features are integrated over ``rho`` and their local mean deconvolved (else each voxel is deconvolved
    and the distributions integrated); ``whiten``: generalised least squares with the dictionary's residual covariance."""
    D = cal.dictionary
    A = _design(D)                                                                                    # (C, N + 1)
    C = Fp.shape[0]
    if integrate_first:
        Xf = (Fp.float() - D.mean.view(C, 1, 1, 1)) / D.std.view(C, 1, 1, 1)
        Xf = torch.stack([gaussian(Xf[k], rho, truncate=truncate) for k in range(C)])
        sub = Xf[:, ::stride, ::stride, ::stride]; del Xf
        _, z, y, x = sub.shape
        Xn = sub.reshape(C, -1).T.contiguous()
    else:
        sub = Fp[:, ::stride, ::stride, ::stride].float()
        _, z, y, x = sub.shape
        Xn = D.standardise(sub.reshape(C, -1).T)                                                      # (M, C)
    if whiten and getattr(D, "residual_cov", None) is not None:
        Wh = whitener(D)
        A = Wh @ A; Xn = Xn @ Wh.T
    nd = D.D.shape[0]
    a = torch.empty((Xn.shape[0], A.shape[1]), device=Xn.device)
    for i in range(0, Xn.shape[0], chunk):
        a[i:i + chunk] = nnls_fista(A, Xn[i:i + chunk], iters=iters, l1=l1)
    fit = a @ A.T
    if D.background is not None:                                                                      # background-only fit: its least-squares coefficient, >= 0
        b = A[:, -1]
        beta = (Xn @ b / (b @ b).clamp_min(1e-12)).clamp_min(0)
        r0 = ((Xn - beta[:, None] * b[None]) ** 2).sum(1)
    else:
        r0 = (Xn ** 2).sum(1)
    r1 = ((Xn - fit) ** 2).sum(1)
    expl = (1 - r1 / r0.clamp_min(1e-12)).clamp(0, 1).reshape(z, y, x)
    O = a[:, :nd].T.reshape(nd, z, y, x).contiguous(); del a, fit
    if not integrate_first:
        sig = rho / stride
        O = torch.stack([gaussian(O[k], sig, truncate=truncate) for k in range(nd)])
        expl = gaussian(expl, sig, truncate=truncate)
    return O, expl


def lobes(O: torch.Tensor, grid: torch.Tensor, *, nmax: int = 3, rel: float = 0.25, min_sep_deg: float = 25.0, nbr_deg: float = 16.0):
    """The lobes of distributions ``O (n_dirs, ...)`` over ``grid (n_dirs, 3)``: up to ``nmax`` local maxima (no grid
    neighbour within ``nbr_deg`` higher), above ``rel`` of the voxel's largest, at least ``min_sep_deg`` apart; each lobe's
    directions are those nearer its peak than any other kept peak and within ``min_sep_deg``. Returns per lobe (largest
    first) its tensor products ``(nmax, 6, ...)`` (``tr(M) I - M``, gradient-product order, ``M`` the lobe's normalised
    second moment) and mass ``(nmax, ...)`` (zero where absent)."""
    nd = O.shape[0]; shp = O.shape[1:]
    Of = O.reshape(nd, -1)                                                                            # (nd, V)
    cosg = (grid @ grid.T).abs()
    nbr = (cosg >= math.cos(math.radians(nbr_deg))) & ~torch.eye(nd, dtype=torch.bool, device=O.device)
    # local maxima: no neighbour higher
    is_max = torch.ones_like(Of, dtype=torch.bool)
    for k in range(nd):
        nk = nbr[k].nonzero().flatten()
        if len(nk):
            is_max[k] = Of[k] >= Of[nk].max(0).values
    top = Of.max(0).values
    score = torch.where(is_max & (Of >= rel * top[None]), Of, torch.zeros_like(Of))
    cos_sep = math.cos(math.radians(min_sep_deg))
    peaks = torch.full((nmax, Of.shape[1]), -1, dtype=torch.long, device=O.device)
    s = score.clone()
    for r in range(nmax):
        val, idx = s.max(0)
        ok = val > 0
        peaks[r] = torch.where(ok, idx, torch.full_like(idx, -1))
        close = cosg[idx] >= cos_sep                                                                  # (V, nd): too close to this peak
        s = torch.where(close.T & ok[None], torch.zeros_like(s), s)
    ux, uy, uz = grid[:, 2], grid[:, 1], grid[:, 0]
    C6 = torch.stack([ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz], 1)                       # (nd, 6)
    pk = peaks.clamp_min(0)
    cpk = torch.stack([cosg[:, pk[r]] for r in range(nmax)])                                         # (nmax, nd, V): |u . peak_r|
    cpk = torch.where((peaks >= 0)[:, None, :], cpk, torch.full_like(cpk, -1.0))
    owner = cpk.argmax(0)                                                                             # nearest kept peak per direction
    S = torch.zeros((nmax, 6, Of.shape[1]), device=O.device); mass = torch.zeros((nmax, Of.shape[1]), device=O.device)
    for r in range(nmax):
        inl = (owner == r) & (cpk[r] >= cos_sep) & (peaks[r] >= 0)[None]
        w = Of * inl.float()                                                                          # (nd, V)
        m = w.sum(0); mass[r] = m
        M = (C6.T @ w) / m.clamp_min(1e-12)[None]                                                     # (6, V)
        tr = M[0] + M[1] + M[2]
        M = -M; M[0] += tr; M[1] += tr; M[2] += tr
        S[r] = M
    return S.reshape((nmax, 6) + shp), mass.reshape((nmax,) + shp)


def _up(t: torch.Tensor, shape) -> torch.Tensor:
    """``t (..., z, y, x)`` on the lattice -> ``(..., Z, Y, X)`` (trilinear, the lattice's voxels at full-grid 0, s, 2s ...)."""
    lead = t.shape[:-3]
    f = t.reshape((1, -1) + tuple(t.shape[-3:]))
    s = [shape[i] / t.shape[-3 + i] for i in range(3)]
    out = Fn.interpolate(f, size=tuple(int(n) for n in shape), mode="trilinear", align_corners=False)
    return out.reshape(tuple(lead) + tuple(shape))


def classical_field(cal, volume, tissue=None, *, features=None, stride: int = 2, nmax: int = 3, rel: float = 0.25, min_sep_deg: float = 25.0,
                    iters: int = 200, l1: float = 0.0, merge_floor: float = 0.1, device: str = "cuda") -> dict:
    """The classical pathway on one block (the readers calibrated already: only the signature dictionary is used).
    Returns the keys the other pipelines return where they apply: ``vec`` the merged field, ``vecs`` / ``weights`` the
    lobes' directions and masses (largest first), ``activation`` the lobes' masses per rank, ``explained`` the
    presence statistic, ``n_populations`` the most lobes at any voxel."""
    v = st._to_torch(volume, device, torch.float32)
    F = cal.features(v) if features is None else features
    Fp = cal.dictionary.prep(F)
    O, expl = odf(cal, Fp, stride=stride, rho=cal.rho, truncate=cal.truncate, iters=iters, l1=l1)
    S, mass = lobes(O, cal.dictionary.grid, nmax=nmax, rel=rel, min_sep_deg=min_sep_deg); del O
    shape = tuple(v.shape)
    S = _up(S, shape); mass = _up(mass, shape).clamp_min(0); expl = _up(expl, shape)
    dirs, gaps = [], []
    for r in range(nmax):
        d, g = direction_and_gap(S[r]); dirs.append(d); gaps.append(g)
    dirs = torch.stack(dirs)
    A = mass / mass.max(0).values.clamp_min(1e-12)[None]
    vec, vec_gap, _, T = single_field(S, A, floor=merge_floor, return_tensor=True)
    vecs, weights = multi_field(dirs, mass)
    return {"vec": vec, "vec_gap": vec_gap, "tensor": T, "vecs": vecs, "weights": weights, "dirs": dirs, "gap": torch.stack(gaps),
            "activation": mass, "explained": expl, "n_populations": int((mass.flatten(1).max(1).values > 0).sum())}
