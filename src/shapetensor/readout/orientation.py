"""A per-voxel orientation readout of the shape space, self-calibrated — the division of the shape space by
orientation (stage 2 of the pipeline), without labels.

The orientation-selective shape space (encoder block 2) encodes a voxel's fibre orientation linearly (results log
§18). The map from features to orientation is fitted on the data itself: the teacher is the excised structure
tensor's direction at its *confident, single-population* sites (high orientation gap), which needs no labels; the
target is the axial outer product d dᵀ (six numbers, sign-free); the model is a ridge regression from the locally
averaged, standardised features. Applied per voxel everywhere — including where the window tensor is unreliable —
it gives each voxel an orientation and a confidence (the anisotropy of the predicted tensor). Per window, the
voxels' orientations are a distribution on the sphere whose *modes* are the populations: a kernel density on a
hemisphere grid, peaks above a null-calibrated relative height, memberships by the nearest mode (stage 2), and a
membership-weighted tensor per mode (stage 3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .common import eigh


def outer6(d: torch.Tensor) -> torch.Tensor:
    """``d (..., 3)`` unit -> the six unique entries of d dᵀ: zz, yy, xx, zy, zx, yx."""
    z, y, x = d[..., 0], d[..., 1], d[..., 2]
    return torch.stack([z * z, y * y, x * x, z * y, z * x, y * x], -1)


def tensor_from6(t6: torch.Tensor) -> torch.Tensor:
    zz, yy, xx, zy, zx, yx = [t6[..., k] for k in range(6)]
    return torch.stack([torch.stack([zz, zy, zx], -1), torch.stack([zy, yy, yx], -1), torch.stack([zx, yx, xx], -1)], -2)


class OrientationReadout:
    """Ridge regression from shape features to the orientation tensor, fitted on confident single-population voxels."""

    def __init__(self, smooth: int = 3, ridge: float = 1.0):
        self.smooth, self.ridge = int(smooth), float(ridge)
        self.mean = self.std = self.W = None

    def _prep(self, F: torch.Tensor) -> torch.Tensor:
        f = F.float()
        if self.smooth > 1:
            f = Fn.avg_pool3d(f[None], self.smooth, stride=1, padding=self.smooth // 2)[0]
        return f

    def fit(self, blocks, n_sample: int = 200_000, seed: int = 0):
        """``blocks``: iterable of ``(F (C, Z, Y, X), teacher (3, Z, Y, X), weight (Z, Y, X))``; voxels are drawn
        with probability ∝ weight (zero excludes). Returns self."""
        g = torch.Generator().manual_seed(seed)
        Xs, Ys = [], []
        per = max(1, n_sample // max(1, len(blocks)))
        for F, d, w in blocks:
            f = self._prep(F)
            wv = w.reshape(-1).float().cpu()
            idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T)
            Ys.append(outer6(d.reshape(3, -1)[:, idx].T))
        X = torch.cat(Xs); Y = torch.cat(Ys)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = torch.cat([(X - self.mean) / self.std, torch.ones(len(X), 1, device=X.device)], 1)
        A = Xn.T @ Xn + self.ridge * torch.eye(Xn.shape[1], device=X.device)
        self.W = torch.linalg.solve(A, Xn.T @ Y)                                          # (C + 1, 6)
        pred = Xn @ self.W
        self.fit_r2 = float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        return self

    def predict(self, F: torch.Tensor, z_chunk: int = 16):
        """Per-voxel orientation ``(3, Z, Y, X)`` (unit, axial) and confidence ``(Z, Y, X)`` in [0, 1]; z-chunked."""
        f = self._prep(F)
        C, Z, Y, X = f.shape
        d_out = torch.empty((3, Z, Y, X), device=F.device); c_out = torch.empty((Z, Y, X), device=F.device)
        for z in range(0, Z, z_chunk):
            fz = f[:, z:z + z_chunk]
            Xn = ((fz.reshape(C, -1).T - self.mean) / self.std)
            t6 = Xn @ self.W[:-1] + self.W[-1]
            lam, V = eigh(tensor_from6(t6))                                               # ascending
            d = V[..., :, 2]                                                              # principal eigenvector
            conf = ((lam[:, 2] - lam[:, 1]) / lam[:, 2].abs().clamp_min(1e-6)).clamp(0, 1)
            d_out[:, z:z + z_chunk] = d.T.reshape(3, *fz.shape[1:]); c_out[z:z + z_chunk] = conf.reshape(fz.shape[1:])
        return d_out, c_out


def modes_dense(o: torch.Tensor, conf: torch.Tensor, rho: float, truncate: float, reduce: int, core, lmax: int = 8,
                rel: float = 0.5, min_sep_deg: float = 25.0, nmax: int = 3):
    """The window modes on a whole site grid at once: the confidence-weighted density of the per-voxel orientations
    in even spherical harmonics (order ``lmax``, angular resolution ~25 deg at 8), its peaks per site. Returns
    ``modes (nz, ny, nx, nmax, 3)`` (z, y, x; zero where absent), ``fractions (nz, ny, nx, nmax)``, ``n (nz, ny, nx)``
    and the sites. The same numbers as :func:`window_modes` up to the kernel (SH truncation vs the 15 deg kernel)."""
    from ..tensor import odf_dense
    from ..odf import peaks
    G = (o * conf.clamp_min(0).sqrt()[None]).to(torch.float16)                      # odf_dense weights a direction by |g|^2 = conf
    O, sites = odf_dense(G, rho, truncate, reduce, core, lmax=lmax)
    d, a, cnt = peaks(O.reshape(-1, O.shape[-1]), lmax, nmax=nmax, rel=rel, min_sep_deg=min_sep_deg)
    shp = O.shape[:3]
    return d.reshape(shp + (nmax, 3)), (a / a.sum(1, keepdim=True).clamp_min(1e-30)).reshape(shp + (nmax,)), cnt.reshape(shp), sites


class OrientationReadoutMLP(OrientationReadout):
    """The same readout with a small MLP (features -> hidden -> the six tensor entries), fitted by Adam on the same
    self-calibrated targets; captures the cos²-type tuning the linear map cannot."""

    def __init__(self, smooth: int = 3, hidden: int = 64, steps: int = 1500, lr: float = 2e-3, weight_decay: float = 1e-4, seed: int = 0):
        super().__init__(smooth=smooth, ridge=0.0)
        self.hidden, self.steps, self.lr, self.wd, self.seed = int(hidden), int(steps), float(lr), float(weight_decay), int(seed)
        self.net = None

    def fit(self, blocks, n_sample: int = 200_000, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        Xs, Ys = [], []
        per = max(1, n_sample // max(1, len(blocks)))
        for F, d, w in blocks:
            f = self._prep(F)
            wv = w.reshape(-1).float().cpu()
            idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T); Ys.append(outer6(d.reshape(3, -1)[:, idx].T))
        X = torch.cat(Xs); Y = torch.cat(Ys)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - self.mean) / self.std
        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.GELU(), torch.nn.Linear(self.hidden, self.hidden),
                                       torch.nn.GELU(), torch.nn.Linear(self.hidden, 6)).to(X.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Xn); bs = min(8192, n)
        for step in range(self.steps):
            i = torch.randint(0, n, (bs,), device=X.device)
            loss = ((self.net(Xn[i]) - Y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = torch.cat([self.net(Xn[a:a + 65536]) for a in range(0, n, 65536)])
            self.fit_r2 = float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        return self

    def predict(self, F: torch.Tensor, z_chunk: int = 16):
        f = self._prep(F)
        C, Z, Y, X = f.shape
        d_out = torch.empty((3, Z, Y, X), device=F.device); c_out = torch.empty((Z, Y, X), device=F.device)
        with torch.no_grad():
            for z in range(0, Z, z_chunk):
                fz = f[:, z:z + z_chunk]
                t6 = self.net((fz.reshape(C, -1).T - self.mean) / self.std)
                lam, V = eigh(tensor_from6(t6))
                d = V[..., :, 2]; conf = ((lam[:, 2] - lam[:, 1]) / lam[:, 2].abs().clamp_min(1e-6)).clamp(0, 1)
                d_out[:, z:z + z_chunk] = d.T.reshape(3, *fz.shape[1:]); c_out[z:z + z_chunk] = conf.reshape(fz.shape[1:])
        return d_out, c_out


class OrientationReadoutDist(OrientationReadoutMLP):
    """Features -> a distribution over ``n_dirs`` hemisphere directions (softmax), trained with the same self-calibrated
    teacher as a soft target (a kernel of ``target_deg`` around the teacher direction). Unlike the tensor regression,
    a voxel that contains two populations keeps two bumps rather than their average, and the window's population
    density is the sum of its voxels' distributions."""

    def __init__(self, smooth: int = 3, hidden: int = 64, steps: int = 2000, lr: float = 2e-3, weight_decay: float = 1e-4, seed: int = 0,
                 n_dirs: int = 160, target_deg: float = 12.0):
        super().__init__(smooth=smooth, hidden=hidden, steps=steps, lr=lr, weight_decay=weight_decay, seed=seed)
        self.n_dirs, self.target_deg = int(n_dirs), float(target_deg)
        self.grid = None

    def fit(self, blocks, n_sample: int = 200_000, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        Xs, Ds = [], []
        per = max(1, n_sample // max(1, len(blocks)))
        for F, d, w in blocks:
            f = self._prep(F)
            wv = w.reshape(-1).float().cpu()
            idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T); Ds.append(d.reshape(3, -1)[:, idx].T)
        X = torch.cat(Xs); D = torch.cat(Ds)
        self.grid = hemisphere(self.n_dirs, X.device)
        k = 1.0 / (2 * math.sin(math.radians(self.target_deg)) ** 2)
        Y = torch.exp(k * ((D @ self.grid.T) ** 2 - 1)); Y = Y / Y.sum(1, keepdim=True)         # soft targets (N, n_dirs)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - self.mean) / self.std
        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.GELU(), torch.nn.Linear(self.hidden, self.hidden),
                                       torch.nn.GELU(), torch.nn.Linear(self.hidden, self.n_dirs)).to(X.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Xn); bs = min(8192, n)
        for step in range(self.steps):
            i = torch.randint(0, n, (bs,), device=X.device)
            logp = torch.log_softmax(self.net(Xn[i]), 1)
            loss = -(Y[i] * logp).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P = torch.cat([torch.softmax(self.net(Xn[a:a + 65536]), 1) for a in range(0, n, 65536)])
            self.fit_r2 = float((P * Y).sum(1).mean() / (Y * Y).sum(1).mean())              # overlap with the target, 1 = perfect
        return self

    def predict_dist(self, F: torch.Tensor, z_chunk: int = 16):
        """Per-voxel distributions ``(n_dirs, Z, Y, X)``."""
        f = self._prep(F); C, Z, Y, X = f.shape
        out = torch.empty((self.n_dirs, Z, Y, X), device=F.device)
        with torch.no_grad():
            for z in range(0, Z, z_chunk):
                fz = f[:, z:z + z_chunk]
                P = torch.softmax(self.net((fz.reshape(C, -1).T - self.mean) / self.std), 1)
                out[:, z:z + z_chunk] = P.T.reshape(self.n_dirs, *fz.shape[1:])
        return out

    def predict(self, F: torch.Tensor, z_chunk: int = 16):
        """The distribution's peak direction per voxel and its confidence (the peak's share of the mass)."""
        P = self.predict_dist(F, z_chunk)
        j = P.argmax(0); conf = P.max(0).values.clamp(0, 1)
        return self.grid[j].permute(3, 0, 1, 2).contiguous(), conf


def smooth_field(o: torch.Tensor, conf: torch.Tensor, k: int = 5):
    """Average the per-voxel orientation tensors ``conf * d dᵀ`` over a ``k``³ neighbourhood (a fascicle is several
    voxels wide; a voxel's readout is noisy) and re-read the direction and confidence from the averaged tensor."""
    T6 = outer6(o.permute(1, 2, 3, 0)) * conf[..., None]                              # (Z, Y, X, 6)
    T6 = Fn.avg_pool3d(T6.permute(3, 0, 1, 2)[None], k, stride=1, padding=k // 2)[0].permute(1, 2, 3, 0)
    lam, V = eigh(tensor_from6(T6.reshape(-1, 6)))
    d = V[..., :, 2]; c = ((lam[:, 2] - lam[:, 1]) / lam[:, 2].abs().clamp_min(1e-9)).clamp(0, 1)
    return d.T.reshape(3, *o.shape[1:]), c.reshape(o.shape[1:])


def hemisphere(n: int = 1500, device="cpu") -> torch.Tensor:
    """Roughly uniform unit vectors on a hemisphere (z >= 0), a Fibonacci set, (n, 3) in (z, y, x)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = i / n                                                                             # z in (0, 1]
    phi = i * math.pi * (3 - math.sqrt(5))
    r = torch.sqrt(1 - z * z)
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)


def window_modes(o: torch.Tensor, w: torch.Tensor, kappa_deg: float = 15.0, rel: float = 0.5, min_sep_deg: float = 25.0,
                 nmax: int = 3, grid: torch.Tensor | None = None, mass_floor: float = 0.0):
    """Modes of axial orientations ``o (n, 3)`` with weights ``w (n,)``: a kernel density on the hemisphere grid
    (von Mises–Fisher-like, width ``kappa_deg``), peaks picked greedily above ``rel`` x the top peak and
    ``min_sep_deg`` apart. Returns modes ``(nmax, 3)`` (zero rows where absent), their relative heights and N."""
    u = hemisphere(1500, o.device) if grid is None else grid
    k = 1.0 / (2 * math.sin(math.radians(kappa_deg)) ** 2)
    c2 = (o @ u.T) ** 2                                                                   # (n, m)
    dens = (w[:, None] * torch.exp(k * (c2 - 1))).sum(0)                                 # (m,)
    return modes_from_density(dens, u, rel=rel, min_sep_deg=min_sep_deg, nmax=nmax, mass_floor=mass_floor)


def modes_from_density(dens: torch.Tensor, u: torch.Tensor, rel: float = 0.5, min_sep_deg: float = 25.0, nmax: int = 3,
                       mass_floor: float = 0.0):
    """Peaks of a density ``dens (m,)`` over grid directions ``u (m, 3)``: greedy, above ``rel`` x the top, at least
    ``min_sep_deg`` apart; with ``mass_floor`` a candidate must own that share of the density (the grid points nearer
    to it than to any kept mode)."""
    modes = torch.zeros(nmax, 3, device=dens.device); heights = torch.zeros(nmax, device=dens.device)
    top = float(dens.max()); taken = torch.zeros(len(u), dtype=torch.bool, device=dens.device)
    cos_sep = math.cos(math.radians(min_sep_deg)); n = 0
    total = float(dens.clamp_min(0).sum())
    for r in range(nmax):
        d = dens.masked_fill(taken, -1.0)
        j = int(d.argmax())
        if d[j] < rel * top or d[j] <= 0:
            break
        if mass_floor > 0 and r > 0:
            near = ((u @ u[j]) ** 2 >= (u @ modes[:r].T).pow(2).max(1).values)
            if float(dens.clamp_min(0)[near].sum()) / max(total, 1e-12) < mass_floor:
                taken[j] = True; continue
        modes[r] = u[j]; heights[r] = d[j] / top; n += 1
        taken |= (u @ u[j]).abs() >= cos_sep
    return modes, heights, n


def assign(o: torch.Tensor, modes: torch.Tensor, n: int, soft_deg: float = 20.0) -> torch.Tensor:
    """Memberships ``(n_vox, nmax)`` of orientations ``o`` to the modes (soft, by angular closeness)."""
    m = torch.zeros(o.shape[0], modes.shape[0], device=o.device)
    if n == 0:
        return m
    k = 1.0 / (2 * math.sin(math.radians(soft_deg)) ** 2)
    c2 = (o @ modes[:n].T) ** 2
    m[:, :n] = torch.exp(k * (c2 - 1))
    m[:, :n] = m[:, :n] / m[:, :n].sum(1, keepdim=True).clamp_min(1e-12)
    return m
