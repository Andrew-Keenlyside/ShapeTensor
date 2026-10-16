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

from .orientation import hemisphere


class SignatureDictionary:
    """Mean standardised feature pattern per orientation (a hemisphere grid), from ``(F, teacher dirs, weights)`` blocks."""

    def __init__(self, smooth: int = 3, n_dirs: int = 160, kernel_deg: float = 12.0):
        self.smooth, self.n_dirs, self.kernel_deg = int(smooth), int(n_dirs), float(kernel_deg)
        self.mean = self.std = self.grid = self.D = None

    def _prep(self, F):
        f = F.float()
        if self.smooth > 1:
            f = Fn.avg_pool3d(f[None], self.smooth, stride=1, padding=self.smooth // 2)[0]
        return f

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
            idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
            Xs.append(f.reshape(f.shape[0], -1)[:, idx].T); Ds.append(d.reshape(3, -1)[:, idx].T)
        X = torch.cat(Xs); Dd = torch.cat(Ds)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - self.mean) / self.std
        self.grid = hemisphere(self.n_dirs, X.device)
        k = 1.0 / (2 * math.sin(math.radians(self.kernel_deg)) ** 2)
        W = torch.exp(k * ((Dd @ self.grid.T) ** 2 - 1))                                   # (N, n_dirs) kernel weights
        self.D = (W.T @ Xn) / W.sum(0)[:, None].clamp_min(1e-6)                            # (n_dirs, C) signatures
        self.background = None
        if background:
            Bs = []
            for F, w in background:
                f = self._prep(F); wv = w.reshape(-1).float().cpu()
                idx = torch.multinomial(wv / wv.sum(), min(per, int((wv > 0).sum())), replacement=False, generator=g).to(F.device)
                Bs.append(f.reshape(f.shape[0], -1)[:, idx].T)
            self.background = ((torch.cat(Bs) - self.mean) / self.std).mean(0)
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
    X = torch.linalg.lstsq(A, Y.T).solution.T.clamp_min(0)                                 # a warm start from the unconstrained solution
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
        Xn = (fz.reshape(C, -1).T - dictionary.mean) / dictionary.std                       # (M, C)
        a = nnls_batched(S, Xn, iters)                                                       # (M, N)
        rec = a @ S.T
        res = ((Xn - rec) ** 2).sum(1) / (Xn ** 2).sum(1).clamp_min(1e-9)
        A_out[:, z:z + z_chunk] = a[:, :N].T.reshape(N, *fz.shape[1:]); R_out[z:z + z_chunk] = res.clamp(0, 1).reshape(fz.shape[1:])
    return A_out, R_out


def layer_tensor_direction(a: torch.Tensor, sigma: float, rho: float, truncate: float = 4.0):
    """The structure tensor of one activation volume ``a (Z, Y, X)`` -> its direction field ``(3, Z, Y, X)`` in
    (z, y, x) order and the orientation gap: the population's direction from its own layer's gradients."""
    from .. import st
    S = st.structure_tensor_3d(a, sigma, rho, truncate)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
