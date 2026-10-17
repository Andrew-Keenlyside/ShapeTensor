"""Multi-direction read-outs: CP with two pieces per population, and the ways of choosing how many populations.

* :func:`cp_als`: batched CP-ALS on ``Y (N, P, C, I)`` (the prototypes' solver).
* :func:`pair_geometric`: the perfect matching of pieces whose pairs are most perpendicular; a pair's fibre is the
  normal of the plane its two directions span.
* N selection:
  - :func:`n_by_bic` (baseline to retire): CP at ranks 2, 4, 6 and the BIC, with fraction floor and lobe merge;
  - :func:`n_by_peaks`: N and coarse directions from the peaks of an ODF (FRT or CSD; :mod:`shapetensor.odf`);
  - :func:`refine_fixed_n`: CP at rank 2N initialised from given directions (the CSD scaffold's refinement);
  - :func:`overcomplete_group`: CP at a generous rank, pieces grouped into planes greedily (rank-free in effect);
  - :func:`stability`: a residual bootstrap of the CP fit; a population is kept if its direction is reproduced.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from . import common


def cp_als(Y: torch.Tensor, R: int, init, starts: int = 3, iters: int = 150, seed: int = 0):
    """Batched CP-ALS of ``Y (N, P, C, I)``, best of ``starts`` per item; the first start from ``init = (B0 (N, C, >=k),
    D0 (N, I, >=k))`` (columns beyond those given are random), the rest random. Returns ``A (N, P, R)``, ``B (N, C, R)``,
    ``D (N, I, R)`` (B, D unit columns) and the relative squared error ``(N,)``."""
    N, P, C, I = Y.shape
    dev = Y.device
    g = torch.Generator(device=dev).manual_seed(seed)
    yy = (Y ** 2).sum((1, 2, 3))
    best = torch.full((N,), float("inf"), device=dev)
    bA, bB, bD = (torch.zeros(N, n, R, device=dev) for n in (P, C, I))
    eye = torch.eye(R, device=dev)

    def solve(Gm, M):
        reg = 1e-6 * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
        return torch.linalg.solve(Gm + reg, M.transpose(1, 2)).transpose(1, 2)

    for s in range(starts):
        B = torch.randn(N, C, R, device=dev, generator=g)
        D = torch.randn(N, I, R, device=dev, generator=g)
        if s == 0 and init is not None:
            B0, D0 = init
            if B0 is not None:
                k = min(R, B0.shape[2])
                B[:, :, :k] = B0[:, :, :k]
            if D0 is not None:
                k = min(R, D0.shape[2])
                D[:, :, :k] = D0[:, :, :k]
        for _ in range(iters):
            A = solve((B.mT @ B) * (D.mT @ D), torch.einsum("npci,ncr,nir->npr", Y, B, D))
            B = solve((A.mT @ A) * (D.mT @ D), torch.einsum("npci,npr,nir->ncr", Y, A, D))
            D = solve((A.mT @ A) * (B.mT @ B), torch.einsum("npci,npr,ncr->nir", Y, A, B))
            nb = B.norm(dim=1, keepdim=True).clamp_min(1e-12)
            nd = D.norm(dim=1, keepdim=True).clamp_min(1e-12)
            B = B / nb
            D = D / nd
            A = A * nb * nd
        err = ((Y - torch.einsum("npr,ncr,nir->npci", A, B, D)) ** 2).sum((1, 2, 3)) / yy.clamp_min(1e-30)
        take = err < best
        best = torch.where(take, err, best)
        bA[take], bB[take], bD[take] = A[take], B[take], D[take]
    return bA, bB, bD, best


def matchings(n: int):
    """All perfect matchings of ``range(n)`` (n even) as tuples of pairs."""
    if n == 0:
        return [()]
    out = []
    for j in range(1, n):
        rest = [k for k in range(1, n) if k != j]
        for m in matchings(len(rest)):
            out.append(((0, j),) + tuple((rest[a], rest[b]) for a, b in m))
    return out


def block_gram(A, B, D):
    """``(N, R, R)``: the inner products of the rank-1 pieces (energy of a set of pieces = sum of its block)."""
    return (A.mT @ A) * (B.mT @ B) * (D.mT @ D)


def pair_geometric(D: torch.Tensor, W: torch.Tensor):
    """Pieces ``D (N, 3, R)`` (R even) paired by the matching whose least-perpendicular pair is most perpendicular.
    Returns ``fibres (N, R/2, 3)`` (unit, canonical sign), ``energy (N, R/2)`` (pair's block sum of ``W``),
    ``min_sine (N,)`` (the chosen matching's worst pair) and the matching index per site."""
    N, _, R = D.shape
    ms = matchings(R)
    Dt = D.mT                                                           # (N, R, 3)
    cross = torch.cross(Dt[:, :, None, :], Dt[:, None, :, :], dim=-1)   # (N, R, R, 3)
    sines = torch.linalg.norm(cross, dim=-1)
    scores = torch.stack([torch.stack([sines[:, r, s] for r, s in m], 1).min(1).values for m in ms], 1)  # (N, M)
    best = scores.argmax(1)
    fib = torch.zeros((N, R // 2, 3), device=D.device)
    en = torch.zeros((N, R // 2), device=D.device)
    for mi, m in enumerate(ms):
        sel = best == mi
        if not bool(sel.any()):
            continue
        for q, (r, s) in enumerate(m):
            c = cross[sel, r, s]
            fib[sel, q] = c / torch.linalg.norm(c, dim=-1, keepdim=True).clamp_min(1e-12)
            en[sel, q] = W[sel, r, r] + W[sel, s, s] + 2 * W[sel, r, s]
    return common.canonical_sign(fib), en, scores.max(1).values, best


def _prepare(K: torch.Tensor):
    K = K.float()
    Y = common.cholesky_factor(K)
    Kn = common.normalise_trace(K.double())
    T = common.channel_sum(Kn).float()
    Cg = common.channel_gram(Kn).float()
    init = (common.eigh(Cg)[1].flip(-1), common.eigh(T)[1].flip(-1))
    return Y, init


def postfilter(fib: torch.Tensor, frac: torch.Tensor, f_min: float = 0.10, sep_deg: float = 20.0, nmax: int = 3):
    """dMRI-style rules on candidate populations: drop fractions under ``f_min``, merge within ``sep_deg`` (the
    stronger kept), renormalise. ``fib (N, k, 3)``, ``frac (N, k)`` -> padded ``(N, nmax, 3)``, ``(N, nmax)``, ``count``."""
    N, k, _ = fib.shape
    dev = fib.device
    out_f = torch.zeros((N, nmax, 3), device=dev)
    out_w = torch.zeros((N, nmax), device=dev)
    cnt = torch.zeros((N,), dtype=torch.int64, device=dev)
    cos_sep = math.cos(math.radians(sep_deg))
    order = frac.argsort(1, descending=True)
    fib_np, frac_np, order_np = fib.cpu().numpy(), frac.cpu().numpy(), order.cpu().numpy()
    of, ow, oc = out_f.cpu().numpy(), out_w.cpu().numpy(), cnt.cpu().numpy()
    for n in range(N):
        kept = []
        for q in order_np[n]:
            if frac_np[n, q] < f_min or len(kept) == nmax:
                continue
            if all(abs(float(fib_np[n, q] @ fib_np[n, j])) < cos_sep for j in kept):
                kept.append(q)
        if kept:
            w = frac_np[n, kept]
            of[n, :len(kept)] = fib_np[n, kept]
            ow[n, :len(kept)] = w / w.sum()
            oc[n] = len(kept)
    return torch.as_tensor(of, device=dev), torch.as_tensor(ow, device=dev), torch.as_tensor(oc, device=dev)


def n_by_bic(K: torch.Tensor, nmax: int = 3, starts: int = 4, iters: int = 200, f_min: float = 0.10,
             sep_deg: float = 20.0, seed: int = 0) -> dict:
    """The earlier prototype's rule set: CP at ranks 2N for N = 1..nmax, N by BIC on the CP residual, then the
    fraction floor and lobe merge. Kept as the baseline the review says to retire."""
    Y, init = _prepare(K)
    N, P, C, I = Y.shape
    n = P * C * I
    fits = []
    for Np in range(1, nmax + 1):
        A, B, D, err = cp_als(Y, 2 * Np, init, starts=starts, iters=iters, seed=seed)
        W = block_gram(A, B, D)
        fib, en, msine, _ = pair_geometric(D, W)
        frac = en / en.sum(1, keepdim=True).clamp_min(1e-30)
        k = 2 * Np * (P + C + I - 2)
        bic = n * torch.log(err.clamp_min(1e-12)) + k * math.log(n)
        fits.append((fib, frac, bic, msine))
    bics = torch.stack([f[2] for f in fits], 1)
    nb = bics.argmin(1)
    fib = torch.zeros((N, nmax, 3), device=K.device)
    frac = torch.zeros((N, nmax), device=K.device)
    for i, (f, w, _, _) in enumerate(fits):
        sel = nb == i
        fib[sel, :i + 1] = f[sel]
        frac[sel, :i + 1] = w[sel]
    out_f, out_w, cnt = postfilter(fib, frac, f_min, sep_deg, nmax)
    return {"fibres": out_f, "fracs": out_w, "count": cnt, "n_bic": nb + 1}


def n_by_peaks(odf_coef: torch.Tensor, lmax: int, mode: str = "csd", kernel: torch.Tensor | None = None,
               rel: float = 0.5, min_sep_deg: float = 25.0, nmax: int = 3, floor: float = 0.0, **csd_kw) -> dict:
    """N and coarse directions from the peaks of the FRT (``mode='frt'``) or CSD (``'csd'``, needs ``kernel``) of the
    gradient-direction ODF ``odf_coef (N, ncoef)``."""
    from .. import odf as O
    if mode == "frt":
        f = O.frt(odf_coef, lmax)
    elif mode == "csd":
        if kernel is None:
            raise ValueError("csd needs the zonal kernel of the single-population response")
        f = O.csd(odf_coef, lmax, kernel, **csd_kw)
    else:
        raise ValueError(mode)
    dirs, amps, cnt = O.peaks(f, lmax, nmax=nmax, rel=rel, min_sep_deg=min_sep_deg, floor=floor)
    frac = amps / amps.sum(1, keepdim=True).clamp_min(1e-30)
    return {"fibres": common.canonical_sign(dirs), "fracs": frac, "count": cnt, "fodf": f}


def _plane_basis(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit directions spanning the plane perpendicular to unit ``u (N, 3)``."""
    e = torch.zeros_like(u)
    e[:, u.abs().argmin(1)] = 1.0
    e = torch.zeros_like(u)
    idx = u.abs().argmin(1)
    e[torch.arange(u.shape[0]), idx] = 1.0
    a = torch.cross(u, e, dim=-1)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = torch.cross(u, a, dim=-1)
    return a, b


def refine_fixed_n(K: torch.Tensor, fibres0: torch.Tensor, count: torch.Tensor, iters: int = 150, seed: int = 0,
                   f_min: float = 0.10, sep_deg: float = 20.0, nmax: int = 3) -> dict:
    """CP at rank 2N per site, N given by ``count (N,)`` and the start by ``fibres0 (N, nmax, 3)`` (two directions in
    each fibre's perpendicular plane). Sites are grouped by N and solved in batches; pieces are paired as initialised
    (the geometric check is reported as ``min_sine``). Returns refined ``fibres``, ``fracs``, ``count``."""
    N = K.shape[0]
    dev = K.device
    out_f = torch.zeros((N, nmax, 3), device=dev)
    out_w = torch.zeros((N, nmax), device=dev)
    out_s = torch.zeros((N,), device=dev)
    out_e = torch.zeros((N,), device=dev)
    Y, init = _prepare(K)
    for n in range(1, nmax + 1):
        sel = torch.nonzero(count == n, as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            continue
        D0 = torch.zeros((sel.numel(), 3, 2 * n), device=dev)
        for q in range(n):
            a, b = _plane_basis(fibres0[sel, q])
            D0[:, :, 2 * q], D0[:, :, 2 * q + 1] = a, b
        A, B, D, err = cp_als(Y[sel], 2 * n, (init[0][sel], D0), starts=1, iters=iters, seed=seed)
        W = block_gram(A, B, D)
        fib = torch.zeros((sel.numel(), n, 3), device=dev)
        en = torch.zeros((sel.numel(), n), device=dev)
        ms = torch.ones((sel.numel(),), device=dev)
        for q in range(n):
            r, s = 2 * q, 2 * q + 1
            c = torch.cross(D[:, :, r], D[:, :, s], dim=-1)
            sn = c.norm(dim=-1)
            fib[:, q] = c / sn[:, None].clamp_min(1e-12)
            en[:, q] = W[:, r, r] + W[:, s, s] + 2 * W[:, r, s]
            ms = torch.minimum(ms, sn)
        frac = en / en.sum(1, keepdim=True).clamp_min(1e-30)
        f, w, c = postfilter(common.canonical_sign(fib), frac, f_min, sep_deg, nmax)
        out_f[sel], out_w[sel], out_s[sel], out_e[sel] = f, w, ms, err
    cnt = (out_w > 0).sum(1)
    return {"fibres": out_f, "fracs": out_w, "count": cnt, "min_sine": out_s, "fit_err": out_e}


def overcomplete_group(K: torch.Tensor, rank: int = 6, nmax: int = 3, f_min: float = 0.10, sep_deg: float = 20.0,
                       min_sine: float = 0.3, starts: int = 3, iters: int = 200, seed: int = 0) -> dict:
    """CP at a generous ``rank``; every pair of pieces proposes the normal of its plane as a fibre, weighted by the
    pair's energy and perpendicularity; pairs are accepted greedily (most energy first) if their pieces are unused,
    their sine exceeds ``min_sine`` and their fibre is ``sep_deg`` from accepted fibres. No k is chosen: N emerges
    from the pieces present (bounded by ``nmax``)."""
    Y, init = _prepare(K)
    A, B, D, err = cp_als(Y, rank, init, starts=starts, iters=iters, seed=seed)
    W = block_gram(A, B, D)
    N = K.shape[0]
    Dt = D.mT
    cross = torch.cross(Dt[:, :, None, :], Dt[:, None, :, :], dim=-1)
    sines = torch.linalg.norm(cross, dim=-1)
    pairs = list(itertools.combinations(range(rank), 2))
    en = torch.stack([W[:, r, r] + W[:, s, s] + 2 * W[:, r, s] for r, s in pairs], 1)       # (N, npairs)
    sn = torch.stack([sines[:, r, s] for r, s in pairs], 1)
    score = en * sn
    order = score.argsort(1, descending=True).cpu().numpy()
    en_np, sn_np = en.cpu().numpy(), sn.cpu().numpy()
    nrm = torch.zeros((N, len(pairs), 3), device=K.device)
    for j, (r, s) in enumerate(pairs):
        c = cross[:, r, s]
        nrm[:, j] = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nrm_np = nrm.cpu().numpy()
    cos_sep = math.cos(math.radians(sep_deg))
    out_f = np.zeros((N, nmax, 3), np.float32)
    out_w = np.zeros((N, nmax), np.float32)
    for n in range(N):
        used, kept = set(), []
        for j in order[n]:
            r, s = pairs[j]
            if r in used or s in used or sn_np[n, j] < min_sine:
                continue
            if all(abs(float(nrm_np[n, j] @ nrm_np[n, k])) < cos_sep for k in kept):
                kept.append(j)
                used |= {r, s}
            if len(kept) == nmax:
                break
        if kept:
            w = en_np[n, kept]
            out_f[n, :len(kept)] = nrm_np[n, kept]
            out_w[n, :len(kept)] = w / w.sum()
    f, w, c = postfilter(torch.as_tensor(out_f, device=K.device), torch.as_tensor(out_w, device=K.device), f_min, sep_deg, nmax)
    return {"fibres": common.canonical_sign(f), "fracs": w, "count": c, "fit_err": err}


def stability(K: torch.Tensor, fibres: torch.Tensor, count: torch.Tensor, n_boot: int = 4, tol_deg: float = 10.0,
              seed: int = 0, iters: int = 100) -> dict:
    """Residual bootstrap of the fixed-N CP fit: the fitted array plus resampled residual entries is refitted from
    the same start ``n_boot`` times; a population is kept if every refit reproduces its direction within ``tol_deg``.
    This is a parametric stability test (the resamples are of the CP residual, not of the window's voxels);
    the grid-phase resample is the data-level alternative when the engine exposes phases."""
    N = K.shape[0]
    dev = K.device
    Y, init = _prepare(K)
    g = torch.Generator(device=dev).manual_seed(seed)
    keep = torch.ones((N, fibres.shape[1]), dtype=torch.bool, device=dev)
    spread = torch.zeros((N, fibres.shape[1]), device=dev)
    nmax = fibres.shape[1]
    for n in range(1, nmax + 1):
        sel = torch.nonzero(count == n, as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            continue
        D0 = torch.zeros((sel.numel(), 3, 2 * n), device=dev)
        for q in range(n):
            a, b = _plane_basis(fibres[sel, q])
            D0[:, :, 2 * q], D0[:, :, 2 * q + 1] = a, b
        Ys = Y[sel]
        A, B, D, err = cp_als(Ys, 2 * n, (init[0][sel], D0), starts=1, iters=iters, seed=seed)
        fit = torch.einsum("npr,ncr,nir->npci", A, B, D)
        res = Ys - fit
        flat = res.reshape(sel.numel(), -1)
        for bi in range(n_boot):
            perm = torch.randperm(flat.shape[1], generator=g, device=dev)
            Yb = fit + flat[:, perm].reshape(res.shape)
            Ab, Bb, Db, _ = cp_als(Yb, 2 * n, (init[0][sel], D0), starts=1, iters=iters, seed=seed + 1 + bi)
            for q in range(n):
                c = torch.cross(Db[:, :, 2 * q], Db[:, :, 2 * q + 1], dim=-1)
                c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                ang = common.angle_deg(c, fibres[sel, q])
                spread[sel, q] = torch.maximum(spread[sel, q], ang)
    keep = spread <= tol_deg
    keep[torch.arange(nmax, device=dev)[None, :] >= count[:, None]] = False
    return {"keep": keep, "spread": spread}
