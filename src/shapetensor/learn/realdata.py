"""Real volumes as model input: background of either polarity, and normalisation by the tissue, not the crop.

A real crop can hold mounting medium or air beyond the specimen -- dark (air, ethanol, a dark mount in X-ray) or bright
(glass and transmitted light in histology, a bright fill value). Either way it is a large, nearly flat mode at one end
of the intensity range. :func:`background_mask` finds it symmetrically (the end mode of the histogram, flat locally),
and :func:`normalise` maps the *tissue's* robust range to 0-1, so a crop that is half background is not squeezed into
a narrow band (the medium is clipped or kept, either convention, the generator draws both).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn


def _local_std(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    p = k // 2
    m = Fn.avg_pool3d(Fn.pad(x[None, None], (p,) * 6, mode="replicate"), k, stride=1)
    m2 = Fn.avg_pool3d(Fn.pad((x * x)[None, None], (p,) * 6, mode="replicate"), k, stride=1)
    return (m2 - m * m).clamp_min(0).sqrt()[0, 0]


def background_mask(x: torch.Tensor, band: float = 0.08, flat: float = 0.5, end: float = 0.15, levels: bool = False):
    """Voxels of a crop ``x (Z, Y, X)`` (any range) that belong to a background at either end of the range: within
    ``band`` (of the robust range) of a histogram mode in the lowest or highest ``end`` share of the range, and locally
    flatter than ``flat`` x the crop's median local spread. Bool ``(Z, Y, X)``; empty when there is no end mode."""
    x = x.float()
    flat_s = x.flatten()[:: max(1, x.numel() // 200000)]
    lo, hi = torch.quantile(flat_s, torch.tensor([0.001, 0.999], device=x.device))
    y = ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    h = torch.histc(y, 64, 0, 1); ne = max(1, int(round(64 * end)))
    ls = _local_std(y); thr = flat * float(ls.flatten()[:: max(1, ls.numel() // 200000)].median().clamp_min(1e-4))
    out = torch.zeros_like(y, dtype=torch.bool); found = []
    for idx in (int(h[:ne].argmax()), 64 - ne + int(h[-ne:].argmax())):
        centre = (idx + 0.5) / 64
        cand = ((y - centre).abs() < band) & (ls < thr)
        if float(cand.float().mean()) > 0.02:                                                     # a mode worth calling a background
            out |= cand; found.append(float(lo + centre * (hi - lo)))
    if bool(out.any()):                                                                            # keep it spatially coherent: grow, then shrink
        o = Fn.max_pool3d(out.float()[None, None], 3, 1, 1)
        out = (-Fn.max_pool3d(-o, 3, 1, 1))[0, 0] > 0.5
    if levels:
        return out, found, float(band * (hi - lo))
    return out


def normalise(x: torch.Tensor, q: tuple = (0.005, 0.995), clip_medium: bool = True):
    """``x`` mapped to 0-1 by the robust range of its tissue (the voxels :func:`background_mask` leaves), the medium
    clipped to 0 or 1 (``clip_medium``) or kept within range. Returns the image and the tissue share."""
    bgm, lv, bw = background_mask(x, levels=True)
    keep = ~Fn.max_pool3d(bgm.float()[None, None], 5, 1, 2)[0, 0].bool()                           # off the background and its rim
    for v in lv:                                                                                   # and off its intensity level (stray voxels)
        keep &= (x.float() - v).abs() > 1.5 * bw
    tissue = x[keep] if float(keep.float().mean()) > 0.05 else x.flatten()
    s = tissue.flatten()[:: max(1, tissue.numel() // 200000)].float()
    lo, hi = torch.quantile(s, torch.tensor(q, device=x.device))
    if not clip_medium:
        lo = torch.minimum(lo, x.min()); hi = torch.maximum(hi, x.max())
    return ((x.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), 1.0 - float(bgm.float().mean())


def tissue_share(x: np.ndarray | torch.Tensor) -> float:
    """The share of a crop that is not background (either polarity)."""
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    return 1.0 - float(background_mask(t.float()).float().mean())
