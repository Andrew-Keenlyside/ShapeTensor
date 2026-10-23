"""The lobe-mixture output: populations as explicit direction lobes, compared with the exact density in harmonics.

Each lattice cell holds ``J`` lobes, each a **Watson** density ``w(u) = exp(kappa ((a.u)^2 - 1)) / Z(kappa)`` about an
axis ``a`` -- a bipolar lobe for ``kappa > 0`` (lines), a girdle about ``a`` for ``kappa < 0`` (sheets), flat at 0 --
with a mass ``m`` and per-lobe descriptors ``q`` (log size, relative wall, lumen fraction, flattening, angularity). The
cell's density is ``f(u) = 4 pi sum_j m_j w_j(u)`` (integral ``4 pi sum m``, as the exact target's), and the j-th
descriptor density ``f_q(u) = 4 pi sum_j m_j q_j w_j(u)``.

Why lobes (review notes of 23 Oct): a free SH head under uncertainty shrinks its high orders and its peaks wander (no
committed second population); slots need a matching and copy their axes. A lobe is a committed axis whose
concentration carries the spread; the sum needs no matching; the density is non-negative by construction; two lobes
at different axes ARE a crossing, and the loss in harmonics compares them with the exact density of the orientations
present (any number of populations, any kind, no categories).

Its spherical-harmonic expansion is closed-form (Funk--Hecke): ``c_lm = 4 pi m lambda_l(kappa) Y_lm(a)`` with
``lambda_l(kappa)`` the mean of the Legendre polynomial ``P_l(a.u)`` under the lobe (a quadrature in ``t = a.u``, so
differentiable in ``kappa``). ``Y_lm(a)`` is evaluated as a polynomial in the axis's components (exact for even
``l``: harmonic polynomials restricted to the sphere) -- smooth everywhere, no pole in the gradient.

Losses: the density's per-order squared error over a running energy per order (no down-weighting of line-rich
batches); a **direction cross-entropy** -- the target density's distribution over directions against the predicted
one's, both truncated to the same order, which penalises a missing lobe (a second population) far more than the
squared error does; descriptor densities; a discriminative identity loss on object ids; branch / endpoint maps;
consistency between paired renders (resolution, rotation).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .sh import fibonacci_sphere, n_coef, real_sh

KAPPA = (-10.0, 20.0)                 # the concentration range: exactly what one lobe of the target can be -- every voxel's
                                      # density is a line lobe (+20) or a sheet girdle (-10) and windowing only broadens them;
                                      # sharper bounds let girdles intersect into fake peaks (the trap seen in direct fits)
NDESC = 5
MASS_OFFSET = -0.87                   # a lobe's mass ~0.03 at the start: a cell's typical population mass; far smaller (-2.5:
                                      # 6e-4) stalled training -- Adam moves a raw output ~lr per step, so the masses took
                                      # hundreds of steps to matter and orientation never entered the loss

# ---------------------------------------------------------------------------------------------------------------
# Watson lobes in harmonics
# ---------------------------------------------------------------------------------------------------------------

_GL: dict = {}


def _gauss_legendre(n: int, device):
    key = (n, str(device))
    if key not in _GL:
        t, w = np.polynomial.legendre.leggauss(n)
        _GL[key] = (torch.tensor(t, dtype=torch.float32, device=device), torch.tensor(w, dtype=torch.float32, device=device))
    return _GL[key]


def legendre_even(t: torch.Tensor, lmax: int) -> torch.Tensor:
    """``P_l(t)`` for even ``l = 0..lmax``: ``(lmax/2 + 1, *t.shape)``."""
    P = [torch.ones_like(t), t]
    for l in range(1, lmax):
        P.append(((2 * l + 1) * t * P[l] - l * P[l - 1]) / (l + 1))
    return torch.stack(P[0:lmax + 1:2])


def watson_lambda(kappa: torch.Tensor, lmax: int, n: int = 96) -> torch.Tensor:
    """Funk--Hecke coefficients ``lambda_l(kappa)`` of the Watson density, even ``l <= lmax``: ``(L, *kappa.shape)``
    (``lambda_0 = 1``). The mean of ``P_l(t)`` under ``exp(kappa t^2)`` on ``[-1, 1]``, by Gauss--Legendre quadrature
    (exponent shifted by ``max(kappa, 0)``: no overflow)."""
    t, w = _gauss_legendre(n, kappa.device)
    k = kappa.float()[..., None]                                                                  # (..., 1)
    e = torch.exp(k * (t * t) - k.clamp_min(0)) * w                                                # (..., n)
    Pl = legendre_even(t, lmax)                                                                    # (L, n)
    num = torch.einsum("...n,ln->l...", e, Pl)
    return num / e.sum(-1)[None].clamp_min(1e-30)


def watson_logz(kappa: torch.Tensor, n: int = 96) -> torch.Tensor:
    """``log`` of the Watson normaliser over the sphere, ``log int exp(kappa ((a.u)^2 - 1)) du``."""
    t, w = _gauss_legendre(n, kappa.device)
    k = kappa.float()[..., None]
    return torch.log(2 * math.pi * (torch.exp(k * (t * t - 1)) * w).sum(-1))


_POLY: dict = {}


def _monomials(u: torch.Tensor, deg: int) -> torch.Tensor:
    """All monomials ``z^a y^b x^c`` with ``a + b + c = deg`` of unit vectors ``u (N, 3)``: ``(N, (deg+1)(deg+2)/2)``."""
    z, y, x = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    cols = []
    for a in range(deg, -1, -1):
        for b in range(deg - a, -1, -1):
            c = deg - a - b
            cols.append(z ** a * y ** b * x ** c)
    return torch.cat(cols, 1)


def sh_poly(u: torch.Tensor, lmax: int) -> torch.Tensor:
    """Real even spherical harmonics (the basis of :func:`~shapetensor.learn.sh.real_sh`) at unit ``u (N, 3)`` as
    polynomials in its components -- each order ``l`` a fixed linear map of the degree-``l`` monomials (fitted once,
    exact) -- so the gradient is smooth everywhere (``real_sh`` has a pole). ``(N, n_coef)``."""
    key = (lmax, str(u.device))
    if key not in _POLY:
        g = fibonacci_sphere(3000, u.device).double()
        Y = real_sh(g.float(), lmax).double()
        Ms, i = [], 0
        for l in range(0, lmax + 1, 2):
            mono = _monomials(g, l)
            sol = torch.linalg.lstsq(mono, Y[:, i:i + 2 * l + 1]).solution                         # (n_mono, 2l + 1)
            Ms.append(sol.float()); i += 2 * l + 1
        _POLY[key] = Ms
    Ms = _POLY[key]
    return torch.cat([_monomials(u.float(), 2 * j) @ M for j, M in enumerate(Ms)], 1)


def lobes_to_sh(axis: torch.Tensor, weight: torch.Tensor, kappa: torch.Tensor, lmax: int) -> torch.Tensor:
    """Lobes ``axis (B, J, 3, ...)`` (any length; normalised here), weights ``(B, J, ...)`` (a mass, or a mass times a
    descriptor), concentrations ``(B, J, ...)`` -> the SH coefficients of ``4 pi sum_j weight_j w_j``:
    ``(B, n_coef, ...)``."""
    B, J = weight.shape[:2]; sp = weight.shape[2:]
    a = Fn.normalize(axis.float(), dim=2).movedim(2, -1).reshape(-1, 3)
    Y = sh_poly(a, lmax).view(B, J, *sp, -1)                                                       # (B, J, ..., n_coef)
    lam = watson_lambda(kappa, lmax)                                                               # (L, B, J, ...)
    reps = torch.tensor([2 * l + 1 for l in range(0, lmax + 1, 2)], device=weight.device)
    lam_c = torch.repeat_interleave(lam, reps, dim=0).movedim(0, -1)                               # (B, J, ..., n_coef)
    c = (4 * math.pi) * (weight.float()[..., None] * lam_c * Y).sum(1)                             # (B, ..., n_coef)
    return c.movedim(-1, 1)


def decode(raw: torch.Tensor, J: int, nd: int = NDESC) -> dict:
    """The lobe head's raw channels ``(B, J (6 + nd), ...)`` -> ``axis (B, J, 3, ...)`` (unit), ``mass``, ``kappa``,
    ``desc (B, J, nd, ...)`` (log size >= 0; the four fractions in 0-1)."""
    B = raw.shape[0]; sp = raw.shape[2:]
    r = raw.float().view(B, J, 6 + nd, *sp)
    axis = Fn.normalize(r[:, :, 0:3], dim=2, eps=1e-6)
    mass = Fn.softplus(r[:, :, 3] + MASS_OFFSET) ** 2 * 0.25                                       # √mass softplus: small masses keep a gradient
    kappa = KAPPA[0] + (KAPPA[1] - KAPPA[0]) * torch.sigmoid(r[:, :, 4] - 0.13)                    # ~4 at init: broad lobes whose axes still get a gradient
    q = torch.cat([Fn.softplus(r[:, :, 6:7]), torch.sigmoid(r[:, :, 7:6 + nd])], 2) if nd else r[:, :, 6:6]
    return {"axis": axis, "mass": mass, "kappa": kappa, "desc": q, "spare": r[:, :, 5]}


def render(lob: dict, lmax: int, desc_lmax: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The lobes' density SH ``(B, n_coef(lmax), ...)`` and, with ``desc_lmax``, the descriptor densities
    ``(B, nd n_coef(desc_lmax), ...)`` (descriptor-major)."""
    f = lobes_to_sh(lob["axis"], lob["mass"], lob["kappa"], lmax)
    if not desc_lmax:
        return f, None
    nd = lob["desc"].shape[2]
    fq = [lobes_to_sh(lob["axis"], lob["mass"] * lob["desc"][:, :, j], lob["kappa"], desc_lmax) for j in range(nd)]
    return f, torch.cat(fq, 1)


# ---------------------------------------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------------------------------------

_EMA: dict = {}


def sh_mse_running(pred: torch.Tensor, tgt: torch.Tensor, key: str = "sh", floor_mass: float = 0.05, decay: float = 0.99,
                   update: bool = True) -> torch.Tensor:
    """Per even order, the squared error over a **running** mean of the target's energy in that order (per cell), the
    orders averaged; ``(B, c, ...)`` with orders on dim 1. The floor (the energy of mass ``floor_mass`` in a cell, in
    order 0) keeps empty batches finite; the running mean keeps a line-rich batch from being down-weighted."""
    from .losses import bands_for
    from .sh import bands
    lmax = 2 * (len(bands_for(pred.shape[1])) - 1)
    total = 0.0; bl = bands(lmax)
    fl = (floor_mass * math.sqrt(4 * math.pi)) ** 2
    for i, b in enumerate(bl):
        k = (key, i)
        e = float((tgt[:, b].float() ** 2).sum(1).mean()) if (update or k not in _EMA) else _EMA[k]
        if update:
            _EMA[k] = e if k not in _EMA else decay * _EMA[k] + (1 - decay) * e
        norm = _EMA.get(k, e) + 0.1 * fl
        total = total + ((pred[:, b].float() - tgt[:, b].float()) ** 2).sum(1).mean() / norm
    return total / len(bl)


def direction_ce(pred: torch.Tensor, tgt: torch.Tensor, lmax: int, n: int = 256, floor: float = 0.02, m_min: float = 0.02) -> torch.Tensor:
    """The direction cross-entropy ``KL(p_t || p_hat)`` per cell between the target's and the prediction's densities as
    distributions over directions (both from their SH to ``lmax``, clipped at ``floor`` x the mean, normalised),
    weighted by the target's mass (cells below ``m_min`` left out): a lobe the prediction lacks costs ``log`` of the
    floor, not the square of a small coefficient. ``(B, c, ...)``."""
    key = ("ce", n, lmax, str(pred.device))
    if key not in _POLY:
        g = fibonacci_sphere(2 * n, pred.device); g = g[g[:, 0] >= 0]
        _POLY[key] = sh_poly(g, lmax)                                                              # (n, c), cached
    Y = _POLY[key]
    nc = Y.shape[1]
    P = torch.einsum("nc,bc...->bn...", Y, pred[:, :nc].float()); T = torch.einsum("nc,bc...->bn...", Y, tgt[:, :nc].float())
    mt = tgt[:, 0].float() / math.sqrt(4 * math.pi)                                                # the target's mass (B, ...)
    mp = pred[:, 0].float() / math.sqrt(4 * math.pi)
    fl_t = floor * mt.clamp_min(1e-6)[:, None]; fl_p = floor * mp.clamp_min(1e-6)[:, None]
    pt = T.clamp_min(0) + fl_t; pt = pt / pt.sum(1, keepdim=True)
    pp = P.clamp_min(0) + fl_p; pp = pp / pp.sum(1, keepdim=True)
    kl = (pt * (pt.log() - pp.log())).sum(1)                                                       # (B, ...)
    w = (mt / m_min).clamp(0, 1) * (mt > 0.25 * m_min)
    return (kl * w).sum() / w.sum().clamp_min(1.0)


def desc_mse(pred: torch.Tensor, tgt: torch.Tensor, nd: int = NDESC, floor_mass: float = 0.05) -> torch.Tensor:
    """Descriptor densities ``(B, nd n, ...)``: each descriptor's squared error over its target energy (plus the
    energy of mass ``floor_mass`` with descriptor 1 in every cell), averaged."""
    n = pred.shape[1] // nd; out = 0.0
    fl = tgt[:, 0].numel() * (floor_mass * math.sqrt(4 * math.pi)) ** 2
    for j in range(nd):
        p_, t_ = pred[:, j * n:(j + 1) * n].float(), tgt[:, j * n:(j + 1) * n].float()
        out = out + ((p_ - t_) ** 2).sum() / ((t_ ** 2).sum() + fl)
    return out / nd


def identity_loss(emb: torch.Tensor, ids: torch.Tensor, delta_v: float = 0.5, delta_d: float = 1.5, min_px: int = 6,
                  max_obj: int = 64) -> torch.Tensor:
    """The discriminative loss (De Brabandere et al. 2017) on an embedding field ``emb (B, E, ...)`` against object ids
    ``(B, ...)`` (-1: none): voxels of an object pulled within ``delta_v`` of its mean, means pushed ``2 delta_d``
    apart, means kept near the origin. Per sample, at most ``max_obj`` objects (the largest)."""
    B, E = emb.shape[:2]; tot = 0.0; n = 0
    for b in range(B):
        e = emb[b].float().reshape(E, -1).T; t = ids[b].reshape(-1)
        ok = t >= 0
        if int(ok.sum()) < min_px:
            continue
        e, t = e[ok], t[ok]
        u, inv, cnt = torch.unique(t, return_inverse=True, return_counts=True)
        keep = cnt >= min_px
        if int(keep.sum()) == 0:
            continue
        order = torch.argsort(cnt, descending=True)[:max_obj]
        sel = torch.zeros_like(keep); sel[order] = True; keep = keep & sel
        m = keep[inv]; e, inv = e[m], inv[m]
        remap = torch.cumsum(keep.long(), 0) - 1; inv = remap[inv]; K = int(keep.sum())
        mu = torch.zeros(K, E, device=e.device).index_add_(0, inv, e) / torch.bincount(inv, minlength=K).clamp_min(1)[:, None].float()
        var = (torch.relu((e - mu[inv]).norm(dim=1) - delta_v) ** 2)
        lv = (torch.zeros(K, device=e.device).index_add_(0, inv, var) / torch.bincount(inv, minlength=K).clamp_min(1).float()).mean()
        if K > 1:
            D = ((mu[:, None] - mu[None]) ** 2).sum(-1).add(1e-6).sqrt(); off = ~torch.eye(K, dtype=torch.bool, device=e.device)   # smooth at 0
            ld = (torch.relu(2 * delta_d - D[off]) ** 2).mean()
        else:
            ld = e.sum() * 0.0
        lr = mu.norm(dim=1).mean()
        tot = tot + lv + ld + 1e-3 * lr; n += 1
    return tot / max(n, 1) if n else emb.sum() * 0.0


def points_loss(pred: torch.Tensor, tgt: torch.Tensor, pos_w: float = 20.0) -> torch.Tensor:
    """Branch-point / endpoint density maps at the lattice: squared error weighted towards the points."""
    t = tgt.float(); return (((pred.float() - t) ** 2) * (1 + pos_w * (t > 0.01).float())).mean()


def pool_sh(field: torch.Tensor, sigma_cells: float, factor: int) -> torch.Tensor:
    """A lattice SH field ``(B, c, z, y, x)`` blurred by a Gaussian of ``sigma_cells`` and averaged ``factor``³: the
    same density described at a coarser lattice (exact up to the window's shape: the density is linear)."""
    from ..filters import gaussian
    x = gaussian(field.float(), sigma_cells, truncate=3.0) if sigma_cells > 0 else field.float()
    return Fn.avg_pool3d(x, factor) if factor > 1 else x


def lobe_peaks(lob: dict, rel: float = 0.25, merge_deg: float = 15.0, k_min: float = 2.0):
    """The populations a lobe set reads as, no harmonics: lobes with ``kappa > k_min`` (directional) and mass at least
    ``rel`` of the cell's largest, lobes within ``merge_deg`` merged (masses added). Returns ``dirs (J, 3, ...)`` and
    ``mass (J, ...)`` sorted by mass, per sample: ``(B, J, 3, ...)``, ``(B, J, ...)``."""
    a, m, k = lob["axis"].float(), lob["mass"].float(), lob["kappa"].float()
    m = m * (k.abs() > k_min)
    J = m.shape[1]; cosm = math.cos(math.radians(merge_deg))
    o = m.argsort(1, descending=True); m = torch.gather(m, 1, o); a = torch.gather(a, 1, o[:, :, None].expand_as(a))
    for i in range(J):
        for j in range(i + 1, J):
            same = ((a[:, i] * a[:, j]).sum(1).abs() >= cosm) & (m[:, i] > 0) & (m[:, j] > 0)
            m_i = torch.where(same, m[:, i] + m[:, j], m[:, i]); m_j = torch.where(same, torch.zeros_like(m[:, j]), m[:, j])
            m = torch.stack([m_i if q == i else (m_j if q == j else m[:, q]) for q in range(J)], 1)
    m = m * (m >= rel * m.amax(1, keepdim=True))
    o = m.argsort(1, descending=True)
    return torch.gather(a, 1, o[:, :, None].expand_as(a)), torch.gather(m, 1, o)
