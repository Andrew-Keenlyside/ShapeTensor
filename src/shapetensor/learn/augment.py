"""Contrast routes: augmentation applied to a finished image every time it is used, so one geometry (one target) is
seen through many contrasts -- the network cannot key on a modality's or a physics' intensity signature.

Applied at batch time to buffered training images (``train.stack``), each use a fresh draw. Every operation keeps the
geometry, so the target (from the geometry) is unchanged:

* **transfer functions** -- a random piecewise-linear curve (2-8 knots at random positions), monotone (either
  direction) or not (folds: intensity order destroyed), guarded against collapsing the range; a **spatially varying**
  one (two curves blended by a smooth random field: contrast that changes across the block); gamma;
* **filters** -- unsharp masking (negative: blur), a Laplacian-of-Gaussian term (phase-contrast-like edges of either
  sign), a **relief** term (a directional derivative along a random 3-D direction: DIC-like shadow casting);
* **fields** -- a smooth multiplicative gain and additive offset;
* **noise** -- Gaussian at a random level, after the curve (so its spectrum and level vary independently of it);
* **acquisition** -- each axis its own resolution and slice thickness; MRI motion (k-space segments of a translated
  object) and spikes; a noise level varying in space; resampling artefacts (aliased downsampling, sub-voxel shifts,
  ideal low-pass ringing); a low bit depth; lossy compression (JPEG-like block DCT, JPEG2000-like wavelet); and one
  of MONAI's maintained transforms (histogram shift, smooth-field contrast, Gibbs, k-space spikes, low resolution,
  Rician noise);
* **sections and optics** -- per-plane gain and offset flicker with occasional one-voxel plane shifts (serial sections),
  light-sheet shadows / curtaining (lines sharp across one axis, constant along another), an axially elongated PSF
  (confocal, light-sheet), photon shot noise, saturation (clipped at one end);
* **missing data** -- **missing lines** (scan-line or detector dropouts: thin lines through the block along one axis,
  all or part of it, filled with a constant, the local mean or the neighbouring line), **small missing blocks**
  (cuboids of 2-12 voxels: bubbles, missing tiles, dropouts; filled with a constant, noise or a blur), **missing
  slices** (1-4 whole planes: blank, the mean, the neighbour repeated) and **thin slabs** (2-4 planes, the same fills
  or interpolated). Small gaps keep their target: the populations continue through them and the network learns to
  carry them across from context. Large gaps are drawn at generation time, where the visibility decides the target:
  cut-out chunks (tissue removed, the medium in its place, the targets emptied) in ``scene.draw_scene`` and thick slabs
  of lost sections in ``generator.nuisance``.

``strength`` (0-1) scales every probability and magnitude (a curriculum: tied to the difficulty level). Scalars come
from a numpy generator (no device synchronisation); fields are drawn on the device.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from ..filters import gaussian

P = {"lut": 0.7, "nonmono": 0.35, "spatial": 0.25, "gamma": 0.3, "unsharp": 0.3, "log": 0.15, "relief": 0.15, "field": 0.4,
     "noise": 0.4, "lines": 0.25, "blocks": 0.25, "resolution": 0.25, "motion": 0.1, "spikes": 0.08, "varnoise": 0.2,
     "resample": 0.15, "quantise": 0.1, "compress": 0.12, "library": 0.15, "ghosting": 0.1, "saturate": 0.08, "sections": 0.12,
     "lsstripes": 0.08, "axial_psf": 0.12, "shot": 0.12, "slices": 0.12, "thin_slabs": 0.08}


def random_lut(rng: np.random.Generator, monotone: bool, min_tv: float = 0.6, max_slope: float = 5.0, max_folds: int = 2):
    """Knots ``(xk, yk)`` of a random curve on [0, 1]: 2-8 inner knots; monotone (rising or falling) or not (at most
    ``max_folds`` changes of direction), its total variation at least ``min_tv`` (a curve that flattens the range would
    erase the structure) and its slope at most ``max_slope`` (a steep segment turns the image's noise into speckle --
    seen on real HiP-CT with unbounded curves)."""
    for _ in range(200):
        k = int(rng.integers(2, 9))
        xk = np.concatenate([[0.0], np.sort(rng.uniform(0, 1, k)), [1.0]])
        yk = rng.uniform(0, 1, k + 2)
        if monotone:
            yk = np.sort(yk) if rng.random() < 0.5 else np.sort(yk)[::-1]
            yk = (yk - yk.min()) / max(yk.max() - yk.min(), 1e-6)                                 # full range
        dy = np.diff(yk); slope = np.abs(dy) / np.maximum(np.diff(xk), 1e-6)
        folds = int((np.sign(dy[1:]) * np.sign(dy[:-1]) < 0).sum())
        if np.abs(dy).sum() >= min_tv and slope.max() <= max_slope and folds <= max_folds:
            return xk, yk
    return np.array([0.0, 1.0]), np.array([0.0, 1.0])


def apply_lut(x: torch.Tensor, xk, yk) -> torch.Tensor:
    xk_t = torch.tensor(xk, dtype=x.dtype, device=x.device); yk_t = torch.tensor(np.ascontiguousarray(yk), dtype=x.dtype, device=x.device)
    i = torch.searchsorted(xk_t, x.clamp(0, 1).contiguous(), right=True).clamp(1, xk_t.numel() - 1)
    x0, x1, y0, y1 = xk_t[i - 1], xk_t[i], yk_t[i - 1], yk_t[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0).clamp_min(1e-6)


def smooth_field(shape, sigma: float, device, g: torch.Generator | None = None) -> torch.Tensor:
    """A smooth random field of unit spread (drawn coarse, upsampled)."""
    f = max(1, int(sigma // 2))
    small = torch.randn([max(2, s // f) for s in shape], device=device, generator=g)
    small = gaussian(small, max(sigma / f, 0.5))
    small = (small - small.mean()) / small.std().clamp_min(1e-6)
    return Fn.interpolate(small[None, None], size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]


def missing_lines(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Thin lines (1-2 voxels) along a random axis, through the whole block or part of it, lost."""
    Z, Y, X = x.shape; ax = int(rng.integers(0, 3)); n = int(rng.integers(1, 13))
    others = [a for a in range(3) if a != ax]
    mode = ["zero", "one", "mean", "neighbour"][int(rng.integers(0, 4))]
    x = x.clone(); sh = x.shape
    for _ in range(n):
        p = [int(rng.integers(0, sh[a])) for a in others]; w = int(rng.integers(1, 3))
        lo = int(rng.integers(0, sh[ax] // 2)) if rng.random() < 0.5 else 0
        hi = int(rng.integers(lo + sh[ax] // 4, sh[ax] + 1)) if lo else sh[ax]
        idx = [slice(None)] * 3
        idx[ax] = slice(lo, hi); idx[others[0]] = slice(p[0], p[0] + w); idx[others[1]] = slice(p[1], p[1] + w)
        idx = tuple(idx)
        if mode == "zero":
            x[idx] = 0.0
        elif mode == "one":
            x[idx] = 1.0
        elif mode == "mean":
            x[idx] = x.mean()
        else:                                                                                      # the neighbouring line repeated
            nb = list(idx); o = others[0]; s0 = max(0, p[0] - w); nb[o] = slice(s0, s0 + w)
            src = x[tuple(nb)]
            if src.shape == x[idx].shape:
                x[idx] = src
    rec["missing_lines"] = (ax, n, mode)
    return x


def missing_blocks(x: torch.Tensor, rng: np.random.Generator, rec: dict, max_side: int = 12) -> torch.Tensor:
    """Small cuboids (2-``max_side`` voxels a side) lost: filled with a constant, noise or a blur of their content."""
    n = int(rng.integers(1, 9)); x = x.clone(); sh = x.shape
    mode = ["zero", "one", "mean", "noise", "blur"][int(rng.integers(0, 5))]
    for _ in range(n):
        s = [int(rng.integers(2, max_side + 1)) for _ in range(3)]
        o = [int(rng.integers(0, max(1, sh[a] - s[a]))) for a in range(3)]
        idx = tuple(slice(o[a], o[a] + s[a]) for a in range(3))
        if mode == "zero":
            x[idx] = 0.0
        elif mode == "one":
            x[idx] = 1.0
        elif mode == "mean":
            x[idx] = x.mean()
        elif mode == "noise":
            x[idx] = torch.rand(x[idx].shape, device=x.device)
        else:
            x[idx] = x[idx].mean()
    rec["missing_blocks"] = (n, mode)
    return x


def _gen(x: torch.Tensor, rng: np.random.Generator) -> torch.Generator:
    """A device generator seeded from ``rng`` (a route's fields reproducible from its numpy draw)."""
    return torch.Generator(device=x.device).manual_seed(int(rng.integers(0, 2 ** 31 - 1)))


def _plane_fill(x: torch.Tensor, ax: int, lo: int, hi: int, mode: str) -> None:
    """Planes ``lo:hi`` along ``ax`` of ``x`` (in place) lost: ``blank`` (0 or 1, either end), ``mean``, ``neighbour``
    (the plane before repeated), ``interp`` (linear between the planes either side)."""
    y = x.movedim(ax, 0); n = y.shape[0]
    if mode == "blank0":
        y[lo:hi] = 0.0
    elif mode == "blank1":
        y[lo:hi] = 1.0
    elif mode == "mean":
        y[lo:hi] = y.mean()
    elif mode == "neighbour":
        y[lo:hi] = y[lo - 1 if lo > 0 else hi].clone()
    else:
        a, b = y[lo - 1 if lo > 0 else hi].clone(), y[hi if hi < n else lo - 1].clone()
        t = (torch.arange(1, hi - lo + 1, device=x.device, dtype=x.dtype) / (hi - lo + 1)).view(-1, 1, 1)
        y[lo:hi] = (1 - t) * a + t * b


def missing_slices(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """1-4 whole planes along a random axis lost: blank (either end), the mean or the neighbour repeated."""
    ax = int(rng.integers(0, 3)); n = int(rng.integers(1, 5)); x = x.clone(); m = x.shape[ax]
    mode = ["blank0", "blank1", "mean", "neighbour"][int(rng.integers(0, 4))]
    idx = sorted(int(v) for v in rng.choice(np.arange(1, m - 1), size=min(n, m - 2), replace=False))
    for i in idx:
        _plane_fill(x, ax, i, i + 1, mode)
    rec["missing_slices"] = (ax, idx, mode)
    return x


def thin_slab(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """One slab of 2-4 consecutive planes along a random axis lost (the fills of :func:`missing_slices`, or
    interpolated across)."""
    ax = int(rng.integers(0, 3)); w = int(rng.integers(2, 5)); x = x.clone(); m = x.shape[ax]
    lo = int(rng.integers(1, max(2, m - w - 1))); mode = ["blank0", "blank1", "mean", "neighbour", "interp"][int(rng.integers(0, 5))]
    _plane_fill(x, ax, lo, min(m, lo + w), mode)
    rec["thin_slab"] = (ax, lo, w, mode)
    return x


def sections(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Serial sections along a random axis: each plane its own gain (sd 2-10 %) and offset (sd 1-4 %), and a few planes
    shifted by one voxel in-plane (edges replicated)."""
    ax = int(rng.integers(0, 3)); g = _gen(x, rng); y = x.movedim(ax, 0).clone(); n = y.shape[0]
    gs, os_ = float(rng.uniform(0.02, 0.1)), float(rng.uniform(0.01, 0.04))
    y = y * (1 + gs * torch.randn(n, device=x.device, generator=g)).view(-1, 1, 1) + (os_ * torch.randn(n, device=x.device, generator=g)).view(-1, 1, 1)
    k = int(rng.integers(0, 5))
    for i in rng.choice(n, size=min(k, n), replace=False):
        d = int(rng.integers(1, 3)); s_ = 1 if rng.random() < 0.5 else -1
        p = Fn.pad(y[int(i)][None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        y[int(i)] = p[1:-1, 1 + s_:p.shape[1] - 1 + s_] if d == 2 else p[1 + s_:p.shape[0] - 1 + s_, 1:-1]
    rec["sections"] = (ax, round(gs, 3), round(os_, 3), k)
    return y.movedim(0, ax).clamp(0, 1)


def ls_stripes(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Light-sheet shadows / curtaining, mild: a multiplicative field ``1 - a S`` with ``S`` lines (1-4 voxels wide) sharp
    across one axis and constant along another, slowly modulated; ``|a| <= 0.25``, darker or brighter."""
    axs = rng.permutation(3); a_ax, b_ax = int(axs[0]), int(axs[1]); g = _gen(x, rng)
    S = torch.zeros(x.shape[a_ax], device=x.device)
    for _ in range(int(rng.integers(2, 13))):
        c0 = int(rng.integers(0, x.shape[a_ax])); S[c0:c0 + int(rng.integers(1, 5))] += float(rng.uniform(0.3, 1.0))
    S = S.clamp(max=1.0).view([-1 if d == a_ax else 1 for d in range(3)])
    env = (1 + 0.4 * smooth_field(x.shape, float(rng.uniform(15, 50)), x.device, g)).clamp(0.2, 1.8)
    a = float(rng.uniform(0.05, 0.25)) * (1 if rng.random() < 0.6 else -1)
    rec["lsstripes"] = (a_ax, b_ax, round(a, 2))
    return (x * (1 - a * S * env)).clamp(0, 1)


def axial_psf(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """An axially elongated PSF (confocal / light-sheet): sigma 1-3 voxels along one axis, 0-0.7 across."""
    ax = int(rng.integers(0, 3)); sg = [float(rng.uniform(0, 0.7)) for _ in range(3)]; sg[ax] = float(rng.uniform(1, 3))
    rec["axial_psf"] = (ax, [round(v, 2) for v in sg])
    return gaussian(x, tuple(sg), truncate=3.0)


def shot_noise(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Photon shot noise: Poisson at 20-500 photons for the full range."""
    Pn = float(math.exp(rng.uniform(math.log(20), math.log(500)))); rec["shot"] = round(Pn)
    return (torch.poisson(x.clamp(0, 1) * Pn, generator=_gen(x, rng)) / Pn).clamp(0, 1)


def saturate(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Saturation: 3-15 % of the voxels clipped at one end, the rest stretched to the full range."""
    fr = float(rng.uniform(0.03, 0.15)); top = rng.random() < 0.6
    q = float(torch.quantile(x.flatten()[:: max(1, x.numel() // 50000)].float(), 1 - fr if top else fr))
    rec["saturate"] = (round(fr, 3), "top" if top else "bottom")
    return (x.clamp(max=q) / max(q, 1e-6)) if top else ((x.clamp(min=q) - q) / max(1 - q, 1e-6))


def per_axis_resolution(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Each axis its own resolution (SynthSeg's random resolution): spacing f_a in 1-4 voxels, a slice thickness (the
    blur's FWHM) of 0.6-1.6 x the spacing -- thickness and spacing apart -- sampled at a random phase, interpolated
    back (nearest, linear or by Fourier zero-padding)."""
    f = [float(math.exp(rng.uniform(0, math.log(4)))) if rng.random() < 0.6 else 1.0 for _ in range(3)]
    if max(f) < 1.2:
        f[int(rng.integers(0, 3))] = float(rng.uniform(1.5, 4))
    th = [fa * float(rng.uniform(0.6, 1.6)) for fa in f]
    y = gaussian(x, tuple(t / 2.355 if fa > 1.01 else 0.0 for t, fa in zip(th, f)), truncate=3.0)
    small = [max(4, int(round(n / fa))) for n, fa in zip(x.shape, f)]
    lo = Fn.interpolate(y[None, None], size=small, mode="trilinear", align_corners=False)
    mode = ["nearest", "trilinear", "fourier"][int(rng.integers(0, 3))]
    if mode == "fourier":
        out = fourier_resize(lo[0, 0], x.shape)
    else:
        out = Fn.interpolate(lo, size=x.shape, mode=mode, **({} if mode == "nearest" else {"align_corners": False}))[0, 0]
    rec["resolution"] = ([round(v, 2) for v in f], [round(v, 2) for v in th], mode)
    return out


def fourier_resize(x: torch.Tensor, shape) -> torch.Tensor:
    """Sinc interpolation by zero-padding (or cropping) the centred spectrum: ringing at edges, as a scanner's own."""
    K = torch.fft.fftshift(torch.fft.fftn(x.float()))
    out = torch.zeros(tuple(shape), dtype=K.dtype, device=x.device)
    # the zero frequency sits at n // 2 after fftshift: align it (n_in // 2 -> n_out // 2), for odd and even sizes alike
    src = [slice(max(0, a // 2 - b // 2), max(0, a // 2 - b // 2) + min(a, b)) for a, b in zip(x.shape, shape)]
    dst = [slice(max(0, b // 2 - a // 2), max(0, b // 2 - a // 2) + min(a, b)) for a, b in zip(x.shape, shape)]
    out[tuple(dst)] = K[tuple(src)]
    y = torch.fft.ifftn(torch.fft.ifftshift(out)).real * (float(np.prod(shape)) / float(np.prod(x.shape)))
    return y


def kspace_motion(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """MRI motion: k-space lines along a phase-encode axis acquired in 2-6 segments, each with the object translated
    (an exact Fourier phase ramp) -- ghosts and blur along that axis (TorchIO's RandomMotion, by translation)."""
    pe = int(rng.integers(0, 3)); n = x.shape[pe]; nseg = int(rng.integers(2, 7))
    K = torch.fft.fftn(x.float()); out = torch.zeros_like(K)
    freqs = [torch.fft.fftfreq(m, device=x.device) for m in x.shape]
    cuts = np.sort(rng.choice(np.arange(1, n), size=nseg - 1, replace=False)) if rng.random() < 0.6 else None
    if cuts is not None:                                                                           # contiguous blocks of lines
        bounds = np.concatenate([[0], cuts, [n]]); seg_of = np.zeros(n, dtype=np.int64)
        for j in range(nseg):
            seg_of[bounds[j]:bounds[j + 1]] = j
    else:                                                                                          # interleaved lines
        seg_of = rng.integers(0, nseg, n)
    perm = np.argsort(np.argsort(np.fft.fftfreq(n)))                                              # line index in acquisition (centre-out) order
    seg_of = seg_of[perm]
    seg_t = torch.tensor(seg_of, device=x.device)
    for j in range(nseg):
        t = rng.normal(0, rng.uniform(0.5, 3.0), 3) if j else np.zeros(3)
        ph = sum(freqs[a].view([-1 if b == a else 1 for b in range(3)]) * float(t[a]) for a in range(3))
        sel = (seg_t == j).view([-1 if b == pe else 1 for b in range(3)])
        out = out + K * torch.exp(-2j * math.pi * ph) * sel
    rec["motion"] = (pe, nseg)
    return torch.fft.ifftn(out).abs()


def kspace_ghosting(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """MRI ghosting: every n-th k-space line along a phase-encode axis scaled (TorchIO's RandomGhosting) -- n - 1
    faint copies of the object shifted by multiples of N / n along that axis (the centre line kept)."""
    pe = int(rng.integers(0, 3)); n = int(rng.integers(2, 7)); a = float(rng.uniform(0.3, 1.0))
    K = torch.fft.fftn(x.float()); m = x.shape[pe]
    w = torch.ones(m, device=x.device); w[::n] = 1 - a; w[0] = 1.0
    K = K * w.view([-1 if b == pe else 1 for b in range(3)])
    rec["ghosting"] = (pe, n, round(a, 2))
    return torch.fft.ifftn(K).abs()


def kspace_spikes(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """k-space spikes (herringbone stripes; TorchIO's RandomSpike, MONAI's RandKSpaceSpikeNoise)."""
    K = torch.fft.fftn(x.float()); m = float(K.abs().max()); n = int(rng.integers(1, 4))
    for _ in range(n):
        idx = tuple(int(rng.integers(0, s)) for s in x.shape)
        K[idx] = K[idx] + float(rng.uniform(0.05, 0.4)) * m * complex(math.cos(rng.uniform(0, 6.28)), math.sin(rng.uniform(0, 6.28)))
    rec["spikes"] = n
    return torch.fft.ifftn(K).abs()


def resample_artefacts(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """What resampling does to data: an aliased (unfiltered) downsampling, a sub-voxel Fourier shift, an ideal low-pass
    (ringing) -- each then brought back to the grid."""
    kind = ["alias", "shift", "lowpass"][int(rng.integers(0, 3))]
    if kind == "alias":
        f = int(rng.integers(2, 4)); o = [int(rng.integers(0, f)) for _ in range(3)]
        small = x[o[0]::f, o[1]::f, o[2]::f]
        y = Fn.interpolate(small[None, None], size=x.shape, mode="nearest" if rng.random() < 0.5 else "trilinear")[0, 0]
    elif kind == "shift":
        t = rng.uniform(-0.5, 0.5, 3); K = torch.fft.fftn(x.float())
        ph = sum(torch.fft.fftfreq(m, device=x.device).view([-1 if b == a else 1 for b in range(3)]) * float(t[a]) for a, m in enumerate(x.shape))
        y = torch.fft.ifftn(K * torch.exp(-2j * math.pi * ph)).real
    else:
        keep = float(rng.uniform(0.35, 0.8))
        small = [max(4, int(n * keep)) for n in x.shape]
        y = fourier_resize(fourier_resize(x, small), x.shape)
    rec["resample"] = kind
    return y


def quantise(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """A low bit depth (3-6 bits): posterised intensities."""
    b = int(rng.integers(3, 7)); q = 2 ** b - 1
    rec["quantise"] = b
    return torch.round(x.clamp(0, 1) * q) / q


_DCT: dict = {}


def compress(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """Lossy compression: JPEG-like (8x8 block DCT per slice along a random axis, coefficients quantised by a table
    growing with frequency) or JPEG2000-like (a three-level Haar wavelet with its detail coefficients quantised)."""
    if rng.random() < 0.6:
        key = str(x.device)
        if key not in _DCT:
            n = torch.arange(8, dtype=torch.float32, device=x.device)
            C = torch.cos(math.pi * (2 * n[None] + 1) * n[:, None] / 16) * math.sqrt(2 / 8); C[0] /= math.sqrt(2)
            _DCT[key] = C
        C = _DCT[key]; ax = int(rng.integers(0, 3)); y = x.movedim(ax, 0)
        A, B, D = y.shape; pb, pd = (-B) % 8, (-D) % 8
        yp = Fn.pad(y[None], (0, pd, 0, pb), mode="replicate")[0]
        blocks = yp.reshape(A, (B + pb) // 8, 8, (D + pd) // 8, 8).permute(0, 1, 3, 2, 4)
        coef = C @ blocks @ C.T
        qual = float(rng.uniform(0.01, 0.07)); fq = torch.arange(8, device=x.device, dtype=torch.float32)
        Q = qual * (1 + fq[:, None] + fq[None, :])
        coef = torch.round(coef / Q) * Q
        rebuilt = (C.T @ coef @ C).permute(0, 1, 3, 2, 4).reshape(A, B + pb, D + pd)[:, :B, :D]
        rec["compress"] = ("jpeg", ax, round(qual, 3))
        return rebuilt.movedim(0, ax)
    # Haar wavelet, three levels (dims padded to a multiple of 8)
    pads = [(-n) % 8 for n in x.shape]
    y = Fn.pad(x[None, None], (0, pads[2], 0, pads[1], 0, pads[0]), mode="replicate")[0, 0]
    q = float(rng.uniform(0.01, 0.06)); levels = []; cur = y
    for _ in range(3):
        a = cur
        for d in range(3):                                                                         # along each axis: averages and differences
            e, o = a.narrow(d, 0, a.shape[d]).unfold(d, 2, 2).unbind(-1)
            a = torch.cat([(e + o) / 2, (e - o) / 2], d)
        lo = a[: a.shape[0] // 2, : a.shape[1] // 2, : a.shape[2] // 2].clone()
        det = a.clone(); det[: a.shape[0] // 2, : a.shape[1] // 2, : a.shape[2] // 2] = 0
        det = torch.round(det / q) * q
        levels.append((det, a.shape)); cur = lo
    rec_ = cur
    for det, shp in reversed(levels):
        a = det.clone(); a[: shp[0] // 2, : shp[1] // 2, : shp[2] // 2] = rec_
        for d in reversed(range(3)):
            h = a.shape[d] // 2
            s_, df = a.narrow(d, 0, h), a.narrow(d, h, h)
            a = torch.stack([s_ + df, s_ - df], -1).movedim(-1, d + 1).reshape(a.shape)
        rec_ = a
    rec["compress"] = ("wavelet", round(q, 3))
    return rec_[: x.shape[0], : x.shape[1], : x.shape[2]]


def library_route(x: torch.Tensor, rng: np.random.Generator, rec: dict) -> torch.Tensor:
    """One transform from the maintained libraries: MONAI (on the tensor's device) -- histogram shift, smooth-field
    contrast, Gibbs ringing, k-space spikes, simulated low resolution, Rician noise -- or TorchIO (CPU) -- motion,
    ghosting, spikes, bias field, anisotropy. Each costs 1-120 ms per 96^3 volume (tools_check.py)."""
    try:
        import monai.transforms as mt
    except Exception:
        return x
    choices = [lambda: mt.RandHistogramShift(num_control_points=(4, 12), prob=1.0),
               lambda: mt.RandSmoothFieldAdjustContrast(spatial_size=tuple(x.shape), rand_size=(4, 4, 4), gamma=(0.4, 2.5), prob=1.0),
               lambda: mt.RandGibbsNoise(prob=1.0, alpha=(0.3, 0.8)),
               lambda: mt.RandKSpaceSpikeNoise(prob=1.0, intensity_range=(10, 13)),
               lambda: mt.RandSimulateLowResolution(prob=1.0, zoom_range=(0.3, 0.8)),
               lambda: mt.RandRicianNoise(prob=1.0, std=0.05, relative=True)]
    try:                                                                                           # TorchIO's (CPU) k-space and field transforms
        import torchio as tio
        tchoices = [lambda: tio.RandomMotion(degrees=5, translation=4, num_transforms=int(rng.integers(1, 4))),
                    lambda: tio.RandomGhosting(num_ghosts=(2, 8), intensity=(0.3, 1.0)), lambda: tio.RandomSpike(num_spikes=(1, 3), intensity=(0.3, 1.0)),
                    lambda: tio.RandomBiasField(coefficients=0.5), lambda: tio.RandomAnisotropy(downsampling=(1.5, 5))]
    except Exception:
        tchoices = []
    k = int(rng.integers(0, len(choices) + len(tchoices)))
    if k >= len(choices):                                                                           # TorchIO draws from torch's global generator:
        t = tchoices[k - len(choices)]()                                                            # seed it from ours for reproducibility
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
            y = t(tio.ScalarImage(tensor=x[None].float().cpu())).data[0].float()
        rec["library"] = "torchio." + type(t).__name__
        return y.to(x.device)
    t = choices[k]()
    t.set_random_state(seed=int(rng.integers(0, 2 ** 31 - 1)))
    y = t(x[None].float())
    y = y.as_tensor() if hasattr(y, "as_tensor") else y
    rec["library"] = "monai." + type(t).__name__
    return torch.as_tensor(y, device=x.device)[0].float()


def contrast_route(x: torch.Tensor, rng: np.random.Generator, strength: float = 1.0, g: torch.Generator | None = None,
                   rec: dict | None = None) -> torch.Tensor:
    """One random contrast route for an image ``x (Z, Y, X)`` in 0-1 (see the module docstring); returns 0-1."""
    rec = {} if rec is None else rec
    s = float(min(max(strength, 0.0), 1.0))
    on = lambda k: rng.random() < P[k] * s
    y = x.float()
    if on("lut"):                                                                                  # a transfer function, possibly varying in space
        mono = not (rng.random() < P["nonmono"])
        L1 = random_lut(rng, mono)
        if on("spatial"):
            L2 = random_lut(rng, mono or rng.random() < 0.5)
            w = torch.sigmoid(2.0 * smooth_field(y.shape, float(rng.uniform(10, 40)), y.device, g))
            y = w * apply_lut(y, *L1) + (1 - w) * apply_lut(y, *L2); rec["lut"] = ("spatial", mono)
        else:
            y = apply_lut(y, *L1); rec["lut"] = ("global", mono)
    if on("gamma"):
        gm = math.exp(rng.uniform(-0.7, 0.7)); y = y.clamp(0, 1) ** gm; rec["gamma"] = round(gm, 2)
    if on("unsharp"):                                                                              # sharpening (a > 0) or blur (a < 0)
        a = float(rng.uniform(-0.7, 2.5)); sg = float(math.exp(rng.uniform(math.log(0.7), math.log(3.0))))
        y = y + a * (y - gaussian(y, sg)); rec["unsharp"] = (round(a, 2), round(sg, 2))
    if on("log"):                                                                                  # edges of either sign
        sg = float(math.exp(rng.uniform(math.log(0.7), math.log(2.0)))); b = float(rng.uniform(-3, 3)) * sg * sg
        sm = gaussian(y, sg); lap = sum(torch.gradient(torch.gradient(sm, dim=a_)[0], dim=a_)[0] for a_ in range(3))
        y = y - b * lap; rec["log"] = round(b, 2)
    if on("relief"):                                                                               # a directional derivative (DIC-like)
        d = rng.normal(size=3); d /= np.linalg.norm(d); sg = float(rng.uniform(0.7, 2.0)); c = float(rng.uniform(-3, 3)) * sg
        sm = gaussian(y, sg); gr = torch.gradient(sm)
        y = y + c * sum(float(d[a_]) * gr[a_] for a_ in range(3)); rec["relief"] = round(c, 2)
    if on("field"):
        y = y * (1 + float(rng.uniform(0.05, 0.4)) * smooth_field(y.shape, float(rng.uniform(15, 60)), y.device, g)) \
            + float(rng.uniform(0, 0.15)) * smooth_field(y.shape, float(rng.uniform(15, 60)), y.device, g); rec["field"] = True
    if on("sections"):
        y = sections(y, rng, rec)
    if on("lsstripes"):
        y = ls_stripes(y.clamp(0, 1), rng, rec)
    lo, hi = torch.quantile(y.flatten()[:: max(1, y.numel() // 50000)], torch.tensor([0.002, 0.998], device=y.device))
    y = ((y - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)                                          # back to the full range
    if on("resolution"):
        y = per_axis_resolution(y, rng, rec)
    if on("axial_psf"):
        y = axial_psf(y, rng, rec)
    if on("resample"):
        y = resample_artefacts(y, rng, rec)
    if on("motion"):
        y = kspace_motion(y, rng, rec)
    if on("spikes"):
        y = kspace_spikes(y, rng, rec)
    if on("ghosting"):
        y = kspace_ghosting(y, rng, rec)
    if on("library"):
        y = library_route(y, rng, rec)
    lo, hi = torch.quantile(y.flatten()[:: max(1, y.numel() // 50000)], torch.tensor([0.002, 0.998], device=y.device))
    y = ((y - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    if on("noise"):
        sd = float(math.exp(rng.uniform(math.log(0.004), math.log(0.06))))
        y = (y + sd * torch.randn(y.shape, device=y.device, generator=g)).clamp(0, 1); rec["noise"] = round(sd, 3)
    if on("shot"):
        y = shot_noise(y, rng, rec)
    if on("saturate"):
        y = saturate(y, rng, rec)
    if on("varnoise"):                                                                             # a noise level that varies across the block
        sd = float(math.exp(rng.uniform(math.log(0.005), math.log(0.05))))
        fld = torch.exp(float(rng.uniform(0.3, 1.0)) * smooth_field(y.shape, float(rng.uniform(15, 50)), y.device, g))
        y = (y + sd * fld * torch.randn(y.shape, device=y.device, generator=g)).clamp(0, 1); rec["varnoise"] = round(sd, 3)
    if on("compress"):
        y = compress(y, rng, rec).clamp(0, 1)
    if on("quantise"):
        y = quantise(y, rng, rec)
    if on("lines"):
        y = missing_lines(y, rng, rec)
    if on("blocks"):
        y = missing_blocks(y, rng, rec)
    if on("slices"):
        y = missing_slices(y, rng, rec)
    if on("thin_slabs"):
        y = thin_slab(y, rng, rec)
    return y
