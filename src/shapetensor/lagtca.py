"""Lag-TCA: the fibre populations of a window as the sources of its lagged feature covariances (no calibration).

For a window of the block, the tensor ``T[p, c, l] = F_c(p + l)``: the features (dev-vit's output, grid-filtered,
optionally reduced to principal components over the block, centred in the window) at positions ``p`` (every
``pstride`` voxels) and offsets ``l`` around them (the centre and a hemisphere of directions at a few radii).

A plain CP decomposition of ``T`` does not separate populations (a field is not low rank over positions x lags). The
tensor that is low rank is the channels x channels x lags covariance ``C_l = E_p[F(p) F(p + l)^T]``: if the features
are a mixture ``F = A s`` of sources whose autocorrelations differ, every ``C_l = A diag(g(l)) A^T``, a symmetric CP
(INDSCAL) whose factors are the mixing ``A`` and the sources' lag profiles ``g``. :func:`sobi_window` solves it as SOBI
(Belouchrani et al. 1997): whitening by ``C_0`` and the orthogonal joint diagonalisation of the whitened lagged
covariances (:func:`joint_diagonalize`). Per source:

* the **lag profile** ``g_r(l)`` -- a fibre population's features stay nearly constant along its fibres, so its profile
  stays near ``g_r(0)`` along the population's direction: the direction is the least axis of a quadratic fitted to the
  squared differences ``(g_r(0) - g_r(l))² / |l|²`` (:func:`invariance_direction`), the anisotropy (how much more
  invariant that axis is than the next) the confidence. The profile is also the population's own anisotropic kernel;
* the **strength** -- its share of the window's feature variance;
* the **activation** -- its power at each position (where in the window it is: interleaved populations occupy parts
  of a window, intermixed ones all of it).

N per window, adaptive: sources anisotropic enough to be fibres, merged when within ``min_sep_deg`` of each other
(a population's features span several sources; :func:`merge_sources`). :func:`lagtca_field` runs overlapping windows
and merges each position's candidates (:func:`consolidate`): up to ``nmax`` directions per position, with weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from .separate import hemisphere


def lag_offsets(radii=(2.0, 4.0, 6.0), n_dirs: int = 30, device="cpu") -> torch.Tensor:
    """``(L, 3)`` offsets (z, y, x) in voxels: the centre and ``n_dirs`` hemisphere directions at each radius."""
    u = hemisphere(n_dirs, device)
    return torch.cat([torch.zeros(1, 3, device=device)] + [r * u for r in radii])


def window_tensor(F: torch.Tensor, lo, size, offsets: torch.Tensor, pstride: int = 2) -> torch.Tensor:
    """``F (K, Z, Y, X)`` -> ``T (P, K, L)`` over positions ``lo + pstride * i`` within ``size`` (trilinear at offsets)."""
    K, Z, Y, X = F.shape; dev = F.device
    ax = [torch.arange(l, l + s, pstride, device=dev, dtype=torch.float32) for l, s in zip(lo, size)]
    p = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)                          # (P, 3)
    q = p[:, None, :] + offsets[None]                                                                # (P, L, 3)
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=dev)
    g = (q * scale - 1.0)[..., [2, 1, 0]]                                                            # grid_sample wants (x, y, z)
    s = Fn.grid_sample(F[None].float(), g.reshape(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border", align_corners=True)
    return s.reshape(K, p.shape[0], offsets.shape[0]).permute(1, 0, 2).contiguous(), p


def invariance_direction(g: torch.Tensor, offsets: torch.Tensor):
    """The axis along which a lag profile ``g (L,)`` (offset 0 first) stays most nearly its centre value: the squared
    differences ``(g(0) - g(l))² / |l|²`` fitted by ``u^T M u`` over the offsets' directions, the direction ``M``'s least
    eigenvector and the anisotropy ``(λ2 - λ1) / λ2`` (how much more invariant the best axis is than the next: 1 for a
    population constant along one axis only, whatever its variation across). Robust where a profile oscillates."""
    g = g * torch.sign(g[0] + 1e-12)
    l = offsets[1:]; r2 = (l ** 2).sum(1).clamp_min(1e-12); u = l / r2.sqrt()[:, None]
    E = (g[0] - g[1:]) ** 2 / r2
    X = torch.stack([u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2, 2 * u[:, 0] * u[:, 1], 2 * u[:, 0] * u[:, 2], 2 * u[:, 1] * u[:, 2]], 1)
    m = torch.linalg.lstsq(X.double(), E.double()[:, None]).solution[:, 0]
    M = torch.stack([torch.stack([m[0], m[3], m[4]]), torch.stack([m[3], m[1], m[5]]), torch.stack([m[4], m[5], m[2]])])
    w, V = torch.linalg.eigh(M)
    aniso = float((w[1] - w[0]) / w[1].clamp_min(1e-12)) if float(w[1]) > 0 else 0.0
    return V[:, 0].float(), max(0.0, min(1.0, aniso))


def joint_diagonalize(Cs: torch.Tensor, sweeps: int = 50, tol: float = 1e-8) -> torch.Tensor:
    """Orthogonal joint diagonalisation of symmetric matrices ``Cs (L, n, n)`` by Jacobi rotations (Cardoso &
    Souloumiac 1996): ``V (n, n)`` with ``V^T C_l V`` as diagonal as possible for every ``l``."""
    dev = Cs.device; A = Cs.detach().to("cpu", torch.float64).clone(); n = A.shape[1]; V = torch.eye(n, dtype=torch.float64)   # tiny matrices: CPU
    for _ in range(sweeps):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                g1 = A[:, p, p] - A[:, q, q]; g2 = A[:, p, q] + A[:, q, p]
                ton = float((g1 * g1).sum() - (g2 * g2).sum()); toff = float(2 * (g1 * g2).sum())
                theta = 0.5 * math.atan2(toff, ton + math.sqrt(ton * ton + toff * toff))
                c, s_ = math.cos(theta), math.sin(theta)
                if abs(s_) > tol:
                    changed = True
                    G = torch.eye(n, dtype=torch.float64); G[p, p] = c; G[q, q] = c; G[p, q] = -s_; G[q, p] = s_
                    A = G.T @ A @ G; V = V @ G
        if not changed:
            break
    return V.float().to(dev)


def sobi_window(T: torch.Tensor, offsets: torch.Tensor, keep: int = 8, min_aniso: float = 0.35, min_var: float = 0.02, sweeps: int = 50,
                min_sep_deg: float | None = 30.0):
    """Populations of one window as the sources of its lagged covariances (SOBI: the symmetric CP of the channels x
    channels x lags covariance tensor). ``T (P, K, L)`` -> components (direction, anisotropy, strength = the source's
    share of the features' variance (its mixing column's energy), activation = source power per position), the
    components with anisotropic autocorrelation only. A population's features span several sources: with
    ``min_sep_deg`` the sources are merged into populations (:func:`merge_sources`)."""
    Tc = T - T.mean(0, keepdim=True)
    P_, K, L = Tc.shape
    X0 = Tc[:, :, 0]
    C0 = (X0.T @ X0) / P_
    w, E = torch.linalg.eigh(C0.double())
    k = min(keep, K); w, E = w[-k:].float(), E[:, -k:].float()
    W = (E / w.clamp_min(1e-12).sqrt()[None]).T                                                       # (k, K) whitening
    Cs = torch.stack([0.5 * (W @ ((X0.T @ Tc[:, :, l]) / P_) @ W.T + (W @ ((X0.T @ Tc[:, :, l]) / P_) @ W.T).T) for l in range(1, L)])
    V = joint_diagonalize(Cs, sweeps)
    U = V.T @ W                                                                                         # (k, K) unmixing
    S = torch.einsum("rk,pkl->prl", U, Tc)                                                             # sources at every position and offset
    g = (S[:, :, :1] * S).mean(0)                                                                       # (k, L): each source's autocorrelation profile
    Amix = torch.linalg.pinv(U)                                                                         # (K, k) mixing
    energy = (Amix ** 2).sum(0) * g[:, 0]
    var = energy / energy.sum().clamp_min(1e-12)
    comps = []
    for r in range(k):
        d, an = invariance_direction(g[r], offsets)
        if an >= min_aniso and float(var[r]) >= min_var:
            act = S[:, r, 0] ** 2
            comps.append({"direction": d, "anisotropy": an, "strength": float(var[r]), "activation": act / act.mean().clamp_min(1e-12)})
    if min_sep_deg is not None:
        comps = merge_sources(comps, min_sep_deg)
    return len(comps), comps


def merge_sources(comps, min_sep_deg: float = 30.0):
    """Greedy, the strongest (share x anisotropy) first: each population absorbs the sources within ``min_sep_deg`` of
    it -- direction the principal axis of their weighted dyads, strength the sum of their shares, anisotropy and
    activation (mean 1 over the window) their weighted means."""
    cos_sep = math.cos(math.radians(min_sep_deg)); left = sorted(comps, key=lambda c: -c["strength"] * c["anisotropy"]); out = []
    while left:
        seed = left[0]["direction"]
        grp = [c for c in left if abs(float(c["direction"] @ seed)) >= cos_sep]
        left = [c for c in left if abs(float(c["direction"] @ seed)) < cos_sep]
        w = torch.tensor([c["strength"] * c["anisotropy"] for c in grp])
        M = sum(float(wi) * torch.outer(c["direction"], c["direction"]) for wi, c in zip(w, grp))
        d = torch.linalg.eigh(M.double())[1][:, -1].float()
        out.append({"direction": d, "strength": sum(c["strength"] for c in grp),
                    "anisotropy": float((w * torch.tensor([c["anisotropy"] for c in grp])).sum() / w.sum()),
                    "activation": sum(float(wi) * c["activation"] for wi, c in zip(w, grp)) / float(w.sum()), "sources": len(grp)})
    return out


def consolidate(cands_d: torch.Tensor, cands_w: torch.Tensor, nmax: int = 2, min_sep_deg: float = 30.0, rel: float = 0.2):
    """Per voxel candidates ``d (V, M, 3)``, ``w (V, M)`` (zero weight = none) -> up to ``nmax`` populations: greedy, the
    strongest first, each absorbing the candidates within ``min_sep_deg`` (their weighted mean dyad's principal axis),
    kept when above ``rel`` of the strongest. Returns ``dirs (nmax, V, 3)`` and ``weights (nmax, V)``."""
    V, M, _ = cands_d.shape; dev = cands_d.device
    cos_sep = math.cos(math.radians(min_sep_deg))
    w = cands_w.clone(); dirs = torch.zeros(nmax, V, 3, device=dev); wts = torch.zeros(nmax, V, device=dev)
    for k in range(nmax):
        j = w.argmax(1); top = w.gather(1, j[:, None])[:, 0]
        seed = cands_d[torch.arange(V, device=dev), j]                                               # (V, 3)
        near = ((cands_d * seed[:, None, :]).sum(-1).abs() >= cos_sep) & (w > 0)
        ww = w * near.float()
        Mx = torch.einsum("vm,vmi,vmj->vij", ww, cands_d, cands_d)
        val, vec = torch.linalg.eigh(Mx.double())
        dirs[k] = vec[:, :, -1].float(); wts[k] = ww.sum(1)
        w = w * (~near).float()
    keep = wts >= rel * wts[0:1].clamp_min(1e-12)
    return dirs, wts * keep.float()


def lagtca_field(F: torch.Tensor, *, window: int = 32, stride: int = 16, pstride: int = 4, sample: int = 2, origin: int = 2, offsets=None,
                 keep: int = 8, min_aniso: float = 0.35, min_var: float = 0.02, nmax: int = 2, min_sep_deg: float = 30.0,
                 rel: float = 0.2, floor: float = 0.05, sweeps: int = 50):
    """Populations of a block from its features ``F (K, Z, Y, X)`` on the lattice ``origin + pstride * i``: each window
    (``window`` voxels every ``stride``) solved by :func:`sobi_window`, its populations' local weights (share x
    activation, at most 1, x anisotropy; the window sampled every ``sample`` voxels, the activation averaged to the
    lattice) spread over the window's positions with a ramp to its faces, averaged over the
    windows covering each position and merged by :func:`consolidate`. Returns ``dirs (nmax, 3, z, y, x)``, ``weights
    (nmax, z, y, x)`` (zero below ``floor`` or ``rel`` of the strongest) and a report."""
    K, Z, Y, X = F.shape; dev = F.device
    offsets = offsets if offsets is not None else lag_offsets(device=dev)
    shape = (Z, Y, X)
    lat = [len(range(origin, n, pstride)) for n in shape]
    starts = []
    for n, m in zip(shape, lat):
        k = max(1, math.ceil((m * pstride - window) / stride) + 1)
        starts.append(sorted({origin + min(i * stride, max(0, (m - window // pstride) * pstride)) for i in range(k)}))
    nwin = window // pstride
    ramp1 = (1.0 - (torch.arange(nwin, device=dev, dtype=torch.float32) + 0.5 - nwin / 2).abs() / (nwin / 2)).clamp_min(0.05)
    ramp = ramp1[:, None, None] * ramp1[None, :, None] * ramp1[None, None, :]
    wsum = torch.zeros(lat, device=dev)
    cand_d, cand_w, counts = [], [], []
    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                li = [(a - origin) // pstride for a in (z0, y0, x0)]
                n = [min(nwin, m - l) for m, l in zip(lat, li)]
                q = pstride // sample; h = (pstride - sample) / 2                                     # q samples per lattice step, centred on it
                T, _ = window_tensor(F, (z0 - h, y0 - h, x0 - h), [k * pstride for k in n], offsets, sample)
                _, comps = sobi_window(T, offsets, keep, min_aniso, min_var, sweeps, min_sep_deg=min_sep_deg)
                counts.append(len(comps))
                r = ramp[:n[0], :n[1], :n[2]]
                sl = (slice(li[0], li[0] + n[0]), slice(li[1], li[1] + n[1]), slice(li[2], li[2] + n[2]))
                wsum[sl] += r
                for c in comps:
                    act = Fn.avg_pool3d(c["activation"].reshape(1, 1, *[k * q for k in n]), q)[0, 0]
                    loc = (c["strength"] * act).clamp(max=1.0) * c["anisotropy"]
                    wmap = torch.zeros(lat, device=dev); wmap[sl] = loc * r
                    cand_w.append(wmap.flatten()); cand_d.append(c["direction"])
    V = lat[0] * lat[1] * lat[2]
    rep = {"windows": len(counts), "populations per window": {k: counts.count(k) for k in sorted(set(counts))}, "candidates": len(cand_w)}
    if not cand_w:
        return torch.zeros(nmax, 3, *lat, device=dev), torch.zeros(nmax, *lat, device=dev), rep
    W = torch.stack(cand_w, 1) / wsum.flatten().clamp_min(1e-12)[:, None]                             # (V, M)
    Dd = torch.stack(cand_d).to(dev)[None].expand(V, -1, -1)
    dirs, wts = consolidate(Dd, W, nmax, min_sep_deg, rel)
    wts = wts * (wts >= floor).float()
    return dirs.permute(0, 2, 1).reshape(nmax, 3, *lat), wts.reshape(nmax, *lat), rep


def reduce_features(F: torch.Tensor, k: int = 16) -> torch.Tensor:
    """``F (C, Z, Y, X)`` -> its first ``k`` principal components over the block, ``(k, Z, Y, X)``."""
    C = F.shape[0]; X = F.reshape(C, -1).float().T
    mu = X.mean(0); sub = X[torch.randperm(X.shape[0], device=X.device)[:200_000]] - mu
    _, _, V = torch.pca_lowrank(sub, q=k, center=False, niter=4)
    return ((X - mu) @ V[:, :k]).T.reshape((k,) + tuple(F.shape[1:])).contiguous()
