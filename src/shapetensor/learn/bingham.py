"""Bingham orientation densities: an entity's ODF in the entity-field model (notes/plan_entity_field_model.md).

Convention. Directions ``u`` are unit vectors in (z, y, x), antipodally symmetric. A component is a frame ``R (..., 3, 3)``
whose columns are ``(a, b, m)`` and concentrations ``kappa (..., 2) = (kappa1, kappa2)``, ``kappa1 >= kappa2 >= 0``::

    p(u) = exp(-kappa1 (a.u)^2 - kappa2 (b.u)^2) / c(kappa),     c(kappa) = (1/4 pi) int exp(...) dOmega

so ``p`` has **unit mean over the sphere** (a density against the uniform measure ``dOmega / 4 pi``) -- the convention
of :func:`~shapetensor.learn.targets.render_odf` and :func:`~shapetensor.learn.lobes.lobes_to_sh`. ``m`` is the mode
axis, ``a`` the axis the density is most concentrated against. ``kappa = 0`` is isotropic; ``kappa1 = kappa2`` large a
peaked line (a Watson lobe of concentration ``kappa`` about ``m``); ``kappa1`` large, ``kappa2 = 0`` a girdle about the
great circle normal to ``a`` (a sheet with normal ``a``: a Watson density of concentration ``-kappa1`` about ``a``);
``kappa1 >> kappa2 > 0`` a fan.

Integrals. With ``t = m.u``, ``s = 1 - t^2`` and the azimuth integrated in closed form (modified Bessel functions), every
quantity is a 1-D Gauss--Legendre integral in ``t`` -- exact to float64 at any concentration (nodes cluster at the
poles, where a sharp density lives)::

    c(kappa)     = 1/2 int_-1^1 e^(-s kappa2) I0e(s delta) dt,                      delta = (kappa1 - kappa2) / 2
    E[(a.u)^2]   = 1/2 E_w[s (1 - I1e/I0e)],   E[(b.u)^2] = 1/2 E_w[s (1 + I1e/I0e)],   E[(m.u)^2] = E_w[t^2]

(``w(t)`` the integrand of ``c``). :func:`fit` inverts the second moments by Newton's method on these integrals;
:func:`to_sh` takes the SH coefficients of the axis-aligned density from a table over ``(kappa1, kappa2)`` (built once by
the same integrals with Bessel functions of every order, cached on disk) and rotates them by the frame (per band, by
evaluating the basis at rotated points -- exact for the even harmonics).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .sh import fibonacci_sphere, n_coef, real_sh

KAPPA_MAX = 600.0                     # the table's range (a lobe of half-width ~2.3 deg at 300); beyond it concentrations clamp
_NT = 96                              # Gauss--Legendre nodes on t in [0, 1] (the integrands are even in t; 96 within 2.2e-11
                                      # of 384 for every kappa <= 600, 64 fails beyond 150: review 5)
_GL: dict = {}
_TAB: dict = {}
_ROT: dict = {}
CACHE = Path(os.environ.get("SHAPETENSOR_CACHE", str(Path.home() / ".cache" / "shapetensor")))


def _nodes(device, n: int = _NT):
    key = (n, str(device))
    if key not in _GL:
        t, w = np.polynomial.legendre.leggauss(n)
        t = 0.5 * (t + 1.0); w = 0.5 * w                                                            # [0, 1]
        _GL[key] = (torch.tensor(t, dtype=torch.float64, device=device), torch.tensor(w, dtype=torch.float64, device=device))
    return _GL[key]


def _stats(kappa: torch.Tensor):
    """``(c, E[(a.u)^2], E[(b.u)^2])`` for ``kappa (..., 2)`` (float64): the normaliser (mean over the sphere) and the
    second moments along ``a`` and ``b``."""
    k = kappa.double().clamp_min(0); t, w = _nodes(k.device)
    k1, k2 = k[..., 0:1], k[..., 1:2]                                                               # (..., 1)
    s = 1.0 - t * t                                                                                 # (n,)
    d = 0.5 * (k1 - k2).abs()
    z = s * d                                                                                       # (..., n)
    i0, i1 = torch.special.i0e(z), torch.special.i1e(z)
    wt = torch.exp(-s * torch.minimum(k1, k2)) * i0 * w                                             # e^(-s k_min) I0e(s delta): the larger kappa's axis carries -I1
    c = wt.sum(-1)                                                                                  # (1/2) int_-1^1 = int_0^1 (even)
    es = (wt * s).sum(-1) / c; r = (wt * s * (i1 / i0.clamp_min(1e-300))).sum(-1) / c
    lo, hi = 0.5 * (es - r), 0.5 * (es + r)                                                         # the axis with the larger kappa has the smaller moment
    first_big = k[..., 0] >= k[..., 1]
    ea = torch.where(first_big, lo, hi); eb = torch.where(first_big, hi, lo)
    return c, ea, eb


def log_normaliser(kappa: torch.Tensor) -> torch.Tensor:
    """``log c(kappa)``, ``c = (1/4 pi) int exp(-kappa1 (a.u)^2 - kappa2 (b.u)^2) dOmega`` (in (0, 1]); ``(...)``."""
    return torch.log(_stats(kappa)[0]).to(kappa.dtype if kappa.is_floating_point() else torch.float32)


def moments(R: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The second moment ``E[u u^T] (..., 3, 3)`` (trace 1) of the components ``R (..., 3, 3)``, ``kappa (..., 2)``."""
    _, ea, eb = _stats(kappa)
    lam = torch.stack([ea, eb, 1.0 - ea - eb], -1).to(R.dtype)                                      # along a, b, m
    return (R * lam[..., None, :]) @ R.transpose(-1, -2)


def fit(T: torch.Tensor, iters: int = 40, tol: float = 1e-11):
    """Moment matching: the Bingham component whose second moment is ``T (..., 3, 3)`` (symmetric PSD, any trace --
    normalised here). Returns ``R (..., 3, 3)`` (a proper rotation; columns ``(a, b, m)``: eigenvectors of the smallest,
    middle and largest eigenvalue) and ``kappa (..., 2)`` with ``kappa1 >= kappa2 >= 0`` (Newton's method on the moment
    integrals in ``asinh(kappa)``, damped; an isotropic ``T`` gives ``kappa ~ 0``; moments outside the achievable region
    are clamped to it; concentrations clamp at :data:`KAPPA_MAX`)."""
    Td = T.double(); Td = 0.5 * (Td + Td.transpose(-1, -2))
    Td = Td / Td.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-30)
    ev, V = torch.linalg.eigh(Td)                                                                   # ascending
    ev = ev.clamp_min(1e-9); ev = ev / ev.sum(-1, keepdim=True)
    det = torch.linalg.det(V); V = torch.cat([V[..., :2], V[..., 2:] * det[..., None, None].sign()], -1)
    la, lb = ev[..., 0], ev[..., 1]
    target = torch.stack([la, lb], -1)
    k0 = torch.stack([(0.5 / la - 1.5).clamp_min(0), (0.5 / lb - 1.5).clamp_min(0)], -1).clamp_max(KAPPA_MAX)
    y = torch.asinh(k0); ymax = math.asinh(KAPPA_MAX)
    h = 1e-6
    for _ in range(iters):
        k = torch.sinh(y)
        _, ea, eb = _stats(k); F = torch.stack([ea, eb], -1) - target
        if float(F.abs().max()) < tol:
            break
        J = []
        for j in range(2):                                                                          # forward differences in asinh(kappa)
            yj = y.clone(); yj[..., j] += h; _, ea_, eb_ = _stats(torch.sinh(yj))
            J.append((torch.stack([ea_, eb_], -1) - F - target) / h)
        J = torch.stack(J, -1)                                                                      # (..., 2 (moments), 2 (kappas))
        detJ = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        good = detJ.abs() > 1e-14
        inv = torch.stack([torch.stack([J[..., 1, 1], -J[..., 0, 1]], -1), torch.stack([-J[..., 1, 0], J[..., 0, 0]], -1)], -2) / torch.where(good, detJ, torch.ones_like(detJ))[..., None, None]
        dy = -(inv @ F[..., None])[..., 0]
        dy = torch.where(good[..., None], dy, torch.zeros_like(dy)).clamp(-1.0, 1.0)                 # damped: at most a factor e per step
        y = (y + dy).clamp(0.0, ymax)
    k = torch.sinh(y)
    k = torch.stack([torch.maximum(k[..., 0], k[..., 1]), torch.minimum(k[..., 0], k[..., 1])], -1)  # la <= lb keeps the order; guard ties
    return V.to(T.dtype if T.is_floating_point() else torch.float32), k.to(T.dtype if T.is_floating_point() else torch.float32)


def pdf(u: torch.Tensor, R: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The density at directions ``u (..., 3)`` of components ``R (..., 3, 3)``, ``kappa (..., 2)`` (broadcast), unit mean
    over the sphere: ``(...)``."""
    a, b = R[..., :, 0], R[..., :, 1]
    q = kappa[..., 0] * (u * a).sum(-1) ** 2 + kappa[..., 1] * (u * b).sum(-1) ** 2
    return torch.exp(-q - log_normaliser(kappa).to(q.dtype))


def mixture_pdf(u: torch.Tensor, R: torch.Tensor, kappa: torch.Tensor, mass: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``sum_e mass_e p_e(u)``: directions ``u (N, 3)``, components ``R (..., E, 3, 3)``, ``kappa (..., E, 2)``, ``mass (..., E)``,
    ``mask (..., E)`` (bool, optional) -> ``(..., N)``."""
    p = pdf(u.view(*([1] * (R.dim() - 2)), -1, 3), R[..., None, :, :], kappa[..., None, :])         # (..., E, N)
    w = mass if mask is None else mass * mask.to(mass.dtype)
    return (w[..., None] * p).sum(-2)


# ---------------------------------------------------------------------------------------------------------------
# spherical harmonics
# ---------------------------------------------------------------------------------------------------------------

LMAX_TABLE = 16
_NK = 112                             # table points per concentration axis, uniform in asinh(kappa) on [0, asinh(KAPPA_MAX)]


def _build_table(lmax: int = LMAX_TABLE, nk: int = _NK, nt: int = 512) -> np.ndarray:
    """SH coefficients (unit-mean density, :func:`~shapetensor.learn.sh.real_sh`'s basis) of the axis-aligned component
    -- mode ``m`` on z, ``b`` on y, ``a`` on x -- on a grid of ``(kappa1, kappa2)`` (``kappa1`` on x, ``kappa2`` on y,
    either order): ``(nk, nk, n_coef(lmax))``. The azimuth integral of ``cos(2 j phi)`` is ``2 pi (-1)^j e^(-s k_y...)``
    in closed form (modified Bessel ``I_j``); the polar one by Gauss--Legendre."""
    from scipy.special import ive
    t, w = np.polynomial.legendre.leggauss(nt); t = 0.5 * (t + 1); w = 0.5 * w                          # [0, 1], even integrands
    s = 1 - t * t
    dirs = torch.tensor(np.stack([t, np.zeros_like(t), np.sqrt(np.clip(s, 0, None))], 1), dtype=torch.float64)   # phi = 0 (azimuth from x)
    Theta = real_sh(dirs.float(), lmax).double().numpy()                                              # (nt, nc): the polar parts (cos terms at phi 0)
    ms = [m for l in range(0, lmax + 1, 2) for m in range(-l, l + 1)]
    ys = np.linspace(0, math.asinh(KAPPA_MAX), nk); ks = np.sinh(ys)
    out = np.zeros((nk, nk, len(ms)))
    KX, KY = np.meshgrid(ks, ks, indexing="ij")                                                     # kappa on x (= a), on y (= b)
    # exponent -s (kx cos^2 phi + ky sin^2 phi) = -s ((kx + ky)/2 + (kx - ky)/2 cos 2 phi)
    dlt = 0.5 * (KX - KY); z = s * np.abs(dlt)[..., None]; base = np.exp(-s * np.minimum(KX, KY)[..., None])   # (nk, nk, nt)
    sgn = np.where(dlt >= 0, -1.0, 1.0)[..., None]                                                  # e^(-z cos 2phi): order j picks (-1)^j (+1 when kx < ky)
    mean = (base * ive(0, z) * w).sum(-1)                                                           # (1/2) int_-1^1 ... = int_0^1
    for jj in range(0, lmax // 2 + 1):
        phi_int = 2 * math.pi * sgn ** jj * base * ive(jj, z)                                       # int_0^2pi e^(-s(...)) cos(2 jj phi) dphi
        for c_, m in enumerate(ms):
            if m == 2 * jj:
                out[..., c_] = 2 * (Theta[:, c_] * phi_int * w).sum(-1) / mean                      # int_-1^1 = 2 int_0^1
    return out


def _table(device, lmax: int):
    key = (str(device), lmax)
    if key not in _TAB:
        f = CACHE / f"bingham_sh_l{LMAX_TABLE}_k{_NK}_v1.npy"
        if f.exists():
            tab = np.load(f)
        else:
            tab = _build_table()
            try:
                CACHE.mkdir(parents=True, exist_ok=True); np.save(f, tab)
            except OSError:
                pass
        _TAB[key] = torch.tensor(tab[..., :n_coef(lmax)], dtype=torch.float32, device=device)
    return _TAB[key]


def _canonical_sh(kappa: torch.Tensor, lmax: int) -> torch.Tensor:
    """Bilinear lookup of the axis-aligned coefficients for ``kappa (E, 2)`` (``kappa1`` on x, ``kappa2`` on y): ``(E, nc)``."""
    tab = _table(kappa.device, lmax); nk = tab.shape[0]; ymax = math.asinh(KAPPA_MAX)
    y = torch.asinh(kappa.float().clamp(0, KAPPA_MAX)) / ymax * (nk - 1)
    i0 = y.floor().long().clamp(0, nk - 2); f = (y - i0).clamp(0, 1)
    ix, iy, fx, fy = i0[:, 0], i0[:, 1], f[:, 0:1], f[:, 1:2]
    return ((1 - fx) * (1 - fy) * tab[ix, iy] + fx * (1 - fy) * tab[ix + 1, iy] + (1 - fx) * fy * tab[ix, iy + 1] + fx * fy * tab[ix + 1, iy + 1])


def _rot_basis(lmax: int, device, n: int = 400):
    key = (lmax, str(device))
    if key not in _ROT:
        g = fibonacci_sphere(n, device).double(); Y = real_sh(g.float(), lmax).double()
        pinvs, i = [], 0
        for l in range(0, lmax + 1, 2):
            pinvs.append(torch.linalg.pinv(Y[:, i:i + 2 * l + 1]).float()); i += 2 * l + 1         # (2l+1, n)
        _ROT[key] = (g.float(), pinvs)
    return _ROT[key]


def sh_rotate(coef: torch.Tensor, Q: torch.Tensor, lmax: int) -> torch.Tensor:
    """The coefficients ``(E, nc)`` of ``f(Q^T u)`` given those of ``f`` and orthogonal ``Q (E, 3, 3)`` (even orders:
    exact for any orthogonal ``Q``) -- per band ``D_l^T = pinv(Y_l(G)) Y_l(G Q^T)``, ``c_l <- D_l c_l``."""
    g, pinvs = _rot_basis(lmax, coef.device)
    E = Q.shape[0]; pts = (g[None] @ Q.transpose(-1, -2).float()).reshape(-1, 3)                    # (E n, 3): Q g
    Yr = real_sh(pts, lmax).view(E, g.shape[0], -1)                                                 # (E, n, nc)
    out, i = [], 0
    for b, P in enumerate(pinvs):
        nb = P.shape[0]
        DT = P[None] @ Yr[:, :, i:i + nb]                                                           # (E, nb, nb) = D^T
        out.append((DT.transpose(1, 2) @ coef[:, i:i + nb, None])[..., 0]); i += nb
    return torch.cat(out, 1)


def to_sh(R: torch.Tensor, kappa: torch.Tensor, lmax: int = 8, chunk: int = 65536) -> torch.Tensor:
    """SH coefficients (:func:`~shapetensor.learn.sh.real_sh` basis, ``c_lm = int p Y_lm dOmega`` of the unit-mean
    density: ``c_00 = sqrt(4 pi)``) of components ``R (..., 3, 3)``, ``kappa (..., 2)``: ``(..., n_coef(lmax))``.
    ``lmax <=`` :data:`LMAX_TABLE`; concentrations beyond :data:`KAPPA_MAX` clamp."""
    assert lmax <= LMAX_TABLE and lmax % 2 == 0
    lead = R.shape[:-2]; Rf = R.reshape(-1, 3, 3).float(); kf = kappa.reshape(-1, 2).float(); out = []
    for s0 in range(0, Rf.shape[0], chunk):
        Rc, kc = Rf[s0:s0 + chunk], kf[s0:s0 + chunk]
        c0 = _canonical_sh(kc, lmax)                                                                # axis-aligned: m on z, b on y, a on x
        Q = torch.stack([Rc[:, :, 2], Rc[:, :, 1], Rc[:, :, 0]], -1)                                 # canonical -> actual (orthogonal; even SH ignore its sign)
        out.append(sh_rotate(c0, Q, lmax))
    return torch.cat(out, 0).view(*lead, -1)


# ---------------------------------------------------------------------------------------------------------------
# the model's vote head, losses, sampling
# ---------------------------------------------------------------------------------------------------------------

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Zhou et al.'s continuous rotation representation: ``x (..., 6)`` -> a proper rotation ``(..., 3, 3)`` (columns: the
    Gram--Schmidt of the two 3-vectors and their cross product)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = Fn.normalize(a1, dim=-1, eps=1e-8)
    b2 = Fn.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2, dim=-1)], -1)


def frame_loss(R_pred: torch.Tensor, R_true: torch.Tensor, kappa_true: torch.Tensor, k0: float = 2.0) -> torch.Tensor:
    """An axis-invariant frame loss in [0, 1] per component ``(...)``: ``w_m (1 - (m.m')^2) + w_a (1 - (a.a')^2)``,
    normalised by the weights. The mode axis counts where it is defined (``w_m = tanh(kappa2 / k0)``: a line or a fan;
    a girdle's mode is any direction in its plane), the ``a`` axis where the density is anisotropic about the mode
    (``w_a = tanh((kappa1 - kappa2) / k0)``: a fan or a girdle; a peaked line's ``a`` is arbitrary). Signs never matter."""
    m, mt = R_pred[..., :, 2], R_true[..., :, 2]; a, at = R_pred[..., :, 0], R_true[..., :, 0]
    wm = torch.tanh(kappa_true[..., 1].clamp_min(0) / k0); wa = torch.tanh((kappa_true[..., 0] - kappa_true[..., 1]).clamp_min(0) / k0)
    lm = 1 - (m * mt).sum(-1) ** 2; la = 1 - (a * at).sum(-1) ** 2
    return (wm * lm + wa * la) / (wm + wa).clamp_min(1e-6)


def sample(R: torch.Tensor, kappa: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """``n`` samples ``(n, 3)`` of one component ``R (3, 3)``, ``kappa (2,)`` -- Kent, Ganeiber & Mardia (2013): an angular
    central Gaussian envelope and rejection (acceptance > ~50 % at any concentration)."""
    dev = R.device; lam = torch.tensor([float(kappa[0]), float(kappa[1]), 0.0], dtype=torch.float64, device=dev)
    q = 3; lo, hi = 1.0, float(q)
    for _ in range(80):                                                                              # b: sum 1 / (b + 2 lam_i) = 1
        b = 0.5 * (lo + hi); f = float((1.0 / (b + 2 * lam)).sum()) - 1.0
        lo, hi = (b, hi) if f > 0 else (lo, b)
    om = 1 + 2 * lam / b; Mstar = math.exp(-(q - b) / 2) * (q / b) ** (q / 2)
    out, have = [], 0
    while have < n:
        m_ = max(2 * (n - have), 64)
        y = torch.randn(m_, 3, dtype=torch.float64, device=dev, generator=generator) / om.sqrt()
        x = y / y.norm(dim=1, keepdim=True)
        qx = (x * x * lam).sum(1); qo = (x * x * om).sum(1)
        acc = torch.rand(m_, dtype=torch.float64, device=dev, generator=generator) < torch.exp(-qx) * qo ** (q / 2) / Mstar
        out.append(x[acc]); have += int(acc.sum())
    x = torch.cat(out)[:n].float()                                                                   # canonical: axis 0 = a, 1 = b, 2 = m
    return x @ R.float().T
