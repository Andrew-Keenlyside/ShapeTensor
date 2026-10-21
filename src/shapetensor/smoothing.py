"""Feature smoothing by sliceTCA: the dev pipeline's replacement for the 3³ box average.

After Pellegrino, Stein & Cayco-Gajic (Nature Neuroscience 2024): a tensor is approximated by a sum of *slice*
components, each a vector along one mode times a full matrix (the slice) over the other two. Kept at low slice rank,
the reconstruction holds the structure the data share along each mode and drops what is idiosyncratic to single
entries.

Here the tensor is built per ``window``³ region of the feature field: ``positions × channels × lags`` -- every
position's neighbourhood, the lags a ``patch``³ grid of offsets ``spacing`` voxels apart. The three component kinds:

* position-slicing: a spatial map × a channel-by-lag pattern (a local structure with its own shape signature),
* channel-slicing: a channel signature × a position-by-lag pattern (one signature whose spatial layout is free),
* lag-slicing: a lag pattern × a position-by-channel pattern (one neighbourhood shape, e.g. elongated along a fibre).

Each kind's term is a low-rank matrix in its mode's unfolding, so the fit is backfitting: each kind in turn is the best
rank-``r`` approximation (truncated SVD) of its unfolding of what the other kinds leave. A voxel's smoothed feature is
the average of its reconstructions over every patch that covers it.

Settings from the phantom tests on the UNet's encoder block 2 (results log, 2026-10-18): slice rank 2 for each kind and
a lag step of 3 voxels (a 13-voxel neighbourhood) were the best of {2, 8} × {1, 3}: bend error 23° -> 4°, readout fit
0.41 -> 0.87, but band crossings lose one band at most settings. Cost ~25 s per 3.5 M voxels (unbatched).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as Fn


def slice_tca(T: torch.Tensor, ranks=(4, 4, 4), iters: int = 8) -> tuple[torch.Tensor, dict]:
    """``T (P, C, L)`` -> its slice-rank ``ranks`` approximation and ``{"explained": share of the energy kept}``."""
    terms = [torch.zeros_like(T) for _ in range(3)]
    for _ in range(iters):
        for k, r in enumerate(ranks):
            if r <= 0:
                continue
            R = T - sum(terms[j] for j in range(3) if j != k)
            M = R.movedim(k, 0).reshape(R.shape[k], -1)                                    # mode-k unfolding
            if M.shape[0] <= 512:                                                          # small side: exact, via its Gram
                w, V = torch.linalg.eigh(M @ M.T)
                U = V[:, -r:]
                Mk = U @ (U.T @ M)
            else:
                U, S, Vh = torch.svd_lowrank(M, q=min(r + 6, min(M.shape)), niter=3)
                Mk = (U[:, :r] * S[:r]) @ Vh[:, :r].T
            shp = (R.shape[k],) + tuple(s for j, s in enumerate(R.shape) if j != k)
            terms[k] = Mk.reshape(shp).movedim(0, k)
    That = sum(terms)
    return That, {"explained": float(1 - ((T - That) ** 2).sum() / (T ** 2).sum().clamp_min(1e-12))}


def _lags(patch: int, spacing: int, device) -> torch.Tensor:
    h = patch // 2
    offs = [o * spacing for o in range(-h, h + 1)]
    return torch.tensor(list(itertools.product(offs, offs, offs)), device=device)          # (L, 3)


def denoise_features(F: torch.Tensor, ranks=(4, 4, 4), spacing: int = 1, patch: int = 5, window: int = 32,
                     pos_stride: int = 2, iters: int = 8, centre: bool = True) -> tuple[torch.Tensor, dict]:
    """``F (C, Z, Y, X)`` -> the sliceTCA-denoised features (same shape, float32) and a report. Every ``window``³
    region (tiling the volume) is decomposed on its own; patches reach ``(patch // 2) * spacing`` voxels beyond it.
    ``centre``: each window's channel means are removed before and restored after (the means are averages over the
    whole window, already quiet)."""
    C, Z, Y, X = F.shape
    dev = F.device
    pad = (patch // 2) * spacing
    Fp = Fn.pad(F.float()[None], (pad,) * 6, mode="replicate")[0]                          # (C, Z+2p, Y+2p, X+2p)
    Zp, Yp, Xp = Fp.shape[1:]
    lags = _lags(patch, spacing, dev)                                                       # (L, 3)
    acc = torch.zeros((C, Zp * Yp * Xp), device=dev); cnt = torch.zeros(Zp * Yp * Xp, device=dev)
    flat = Fp.reshape(C, -1)
    explained = []
    for z0 in range(0, Z, window):
        for y0 in range(0, Y, window):
            for x0 in range(0, X, window):
                zz = torch.arange(z0, min(z0 + window, Z), pos_stride, device=dev) + pad
                yy = torch.arange(y0, min(y0 + window, Y), pos_stride, device=dev) + pad
                xx = torch.arange(x0, min(x0 + window, X), pos_stride, device=dev) + pad
                pos = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), -1).reshape(-1, 3)   # (P, 3)
                vox = pos[:, None, :] + lags[None]                                          # (P, L, 3)
                idx = (vox[..., 0] * Yp + vox[..., 1]) * Xp + vox[..., 2]                  # (P, L)
                T = flat[:, idx].permute(1, 0, 2).contiguous()                              # (P, C, L)
                mu = T.mean((0, 2), keepdim=True) if centre else 0.0
                That, info = slice_tca(T - mu, ranks, iters)
                explained.append(info["explained"])
                That = That + mu
                acc.index_add_(1, idx.reshape(-1), That.permute(1, 0, 2).reshape(C, -1))
                cnt.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=dev))
    out = (acc / cnt.clamp_min(1)).reshape(C, Zp, Yp, Xp)
    covered = (cnt > 0).reshape(Zp, Yp, Xp)[pad:pad + Z, pad:pad + Y, pad:pad + X]
    out = out[:, pad:pad + Z, pad:pad + Y, pad:pad + X]
    out = torch.where(covered[None], out, F.float())                                        # any voxel no patch reached: as it was
    ex = torch.tensor(explained)
    return out.contiguous(), {"windows": len(explained), "explained_median": round(float(ex.median()), 3),
                              "uncovered": int((~covered).sum()), "ranks": list(ranks), "spacing": spacing, "patch": patch,
                              "window": window}


class SmoothedEngine:
    """A feature engine followed by sliceTCA smoothing: ``make_engine("devvit_slicetca")``. Same contract as any engine;
    the margin grows by the patches' reach."""

    def __init__(self, base, ranks=(2, 2, 2), spacing: int = 3, patch: int = 5, window: int = 32, pos_stride: int = 2, iters: int = 8):
        self.base = base
        self.kw = {"ranks": tuple(int(r) for r in ranks), "spacing": int(spacing), "patch": int(patch), "window": int(window),
                   "pos_stride": int(pos_stride), "iters": int(iters)}
        self.name = f"{base.name}+slicetca"; self.n_channels = base.n_channels
        self.margin = int(getattr(base, "margin", 0)) + (int(patch) // 2) * int(spacing)
        self.last_report: dict | None = None

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        F = self.base(vol)
        out, self.last_report = denoise_features(F, **self.kw)
        return out.to(F.dtype)

    def describe(self) -> dict:
        d = dict(self.base.describe()) if hasattr(self.base, "describe") else {"engine": self.base.name}
        return {**d, "engine": self.name, "slicetca": self.kw, "margin": self.margin}
