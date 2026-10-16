"""Label-free image statistics, measured the same way on a real block and on a phantom, so the phantoms can be
matched to the data they stand in for (and the match reported).

* ``noise_sd``: the standard deviation of the fine residual ``I - G_2 * I`` in the 10% smoothest 32^3 blocks of
  tissue, divided by the filter's noise gain (measured on white noise), i.e. the noise the texture sits on;
* ``dog_rms``: the RMS of difference-of-Gaussian bands (1-2, 2-4, 4-8, 8-16 voxels) over tissue, noise part removed
  in quadrature: the texture amplitude per scale;
* ``modes``: the main intensity modes of the histogram (kernel-smoothed), i.e. the tissue levels and their steps;
* ``grad_energy``: mean squared gradient at sigma 1 over tissue.
"""
from __future__ import annotations

import numpy as np
import torch

from ..filters import gaussian

BANDS = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))


def _noise_gain(sigma: float, device, n: int = 96) -> float:
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randn((n, n, n), generator=g).to(device)
    r = w - gaussian(w, sigma, truncate=3.0)
    return float(r[16:-16, 16:-16, 16:-16].std())


def _dog_gain(s1: float, s2: float, device, n: int = 96) -> float:
    g = torch.Generator(device="cpu").manual_seed(1)
    w = torch.randn((n, n, n), generator=g).to(device)
    r = gaussian(w, s1, truncate=3.0) - gaussian(w, s2, truncate=3.0)
    return float(r[24:-24, 24:-24, 24:-24].std())


def profile(vol: np.ndarray, tissue_mask: np.ndarray | None = None, device="cuda", block: int = 32) -> dict:
    """The statistics above for ``vol (Z, Y, X)`` grey levels, over ``tissue_mask`` (default: all)."""
    v = torch.as_tensor(np.asarray(vol, np.float32), device=device)
    m = torch.ones_like(v, dtype=torch.bool) if tissue_mask is None else torch.as_tensor(np.asarray(tissue_mask, bool), device=device)
    out = {}
    # noise: residual SD in the smoothest blocks
    res = v - gaussian(v, 2.0, truncate=3.0)
    Z, Y, X = v.shape
    bz, by, bx = Z // block, Y // block, X // block
    r = res[:bz * block, :by * block, :bx * block].reshape(bz, block, by, block, bx, block)
    mk = m[:bz * block, :by * block, :bx * block].reshape(bz, block, by, block, bx, block).float().mean((1, 3, 5))
    sd = r.std((1, 3, 5))
    ok = mk > 0.95
    sds = sd[ok] if bool(ok.any()) else sd.flatten()
    q = torch.quantile(sds, 0.10)
    gain = _noise_gain(2.0, device)
    out["noise_sd"] = float(q / gain)
    out["noise_sd_median_block"] = float(sds.median() / gain)
    # texture per scale: the band's RMS over tissue, its noise part (the band RMS in the 10% smoothest tissue
    # blocks, so correlated noise is measured as it is, not assumed white), texture = the rest in quadrature
    bands = {}
    for s1, s2 in BANDS:
        d = gaussian(v, s1, truncate=3.0) - gaussian(v, s2, truncate=3.0)
        rms = float(d[m].pow(2).mean().sqrt())
        db = d[:bz * block, :by * block, :bx * block].reshape(bz, block, by, block, bx, block)
        brms = db.pow(2).mean((1, 3, 5)).sqrt()
        bsel = brms[ok] if bool(ok.any()) else brms.flatten()
        noise_b = float(torch.quantile(bsel, 0.10))
        bands[f"{s1:g}-{s2:g}"] = {"rms": rms, "noise": noise_b, "rms_texture": float(np.sqrt(max(rms ** 2 - noise_b ** 2, 0.0))),
                                   "noise_white_equiv": out["noise_sd"] * _dog_gain(s1, s2, device)}
    out["dog_rms"] = bands
    g2 = sum(gaussian(v, 1.0, order=o, truncate=3.0) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    out["grad_energy_sigma1"] = float(g2[m].mean())
    # intensity modes
    vals = v[m].flatten()
    if vals.numel() > 4_000_000:
        vals = vals[torch.randperm(vals.numel(), device=device)[:4_000_000]]
    hist = torch.histc(vals, bins=256, min=0, max=255).cpu().numpy()
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); k /= k.sum()
    sm = np.convolve(hist, k, mode="same")
    peaks = [i for i in range(2, 254) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0.02 * sm.max()]
    out["modes"] = [{"level": int(i), "weight": float(sm[i] / sm.sum())} for i in peaks]
    out["mean"] = float(vals.mean()); out["median"] = float(vals.median())
    out["p05"], out["p95"] = float(torch.quantile(vals, 0.05)), float(torch.quantile(vals, 0.95))
    return out


def real_crop_profile(path: str, level: int = 0, threshold: float | None = None, device="cuda") -> dict:
    """The profile of an OME-Zarr crop (e.g. the 20 um S-21-46 block), overall and split at ``threshold`` into the
    brighter (WM-like) and darker tissue, each measured separately."""
    import zarr
    a = zarr.open(path, mode="r")[str(level)]
    vol = np.asarray(a[:]).astype(np.float32)
    out = {"shape": list(vol.shape), "all": profile(vol, None, device)}
    if threshold is not None:
        from scipy import ndimage as ndi
        sm = ndi.gaussian_filter(vol, 2.0)
        bright = sm >= threshold
        out["bright"] = profile(vol, bright, device)
        out["dark"] = profile(vol, ~bright, device)
        out["bright_fraction"] = float(bright.mean())
    return out
