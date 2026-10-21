"""Reconstruction artefacts added to a block, for resilience tests: streaks and rings (synchrotron CT).

Conventions as :mod:`shapetensor.phantoms`: arrays ``(Z, Y, X)``, vectors ``(z, y, x)``, the tomographic rotation axis
along z, so both artefacts live in z slices.

* :func:`streaks` -- straight lines across a z slice, as beam-hardening and photon-starvation streaks are: a Gaussian
  cross-section ``width`` voxels wide in the slice and ``thickness`` voxels along z, bright or dark at random, on average
  ``per_slice`` lines per slice, crossing the whole slice. Their in-slice direction is ``angle_deg`` from the x axis
  (0: along x, inside the phantoms' fibre planes; 90: along y, across them) give or take ``spread_deg`` (small: the
  near-parallel streaks a dense object outside the block casts; 90: any direction).
* :func:`rings` -- concentric circles about the rotation axis at ``centre_yx`` (voxels, may lie outside the block), the
  same in every slice (a detector-column defect): locally a curved sheet whose normal points at the axis.

Both return the block with the artefact added (uint8 in, uint8 out, as the phantoms) and a description of it;
``amp`` is in grey levels (the phantoms' fibre darkening is ``TextureParams.amp x texture``: 50 at the calibrated
setting).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _as_float(img, device):
    return torch.as_tensor(np.asarray(img), device=device).float()


def _finish(vol: torch.Tensor, like) -> np.ndarray:
    out = vol.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
    return out if isinstance(like, np.ndarray) else torch.as_tensor(out)


def streak_direction(angle_deg: float) -> np.ndarray:
    """The (z, y, x) unit vector of a streak at ``angle_deg`` in its z slice (0 = along x, 90 = along y)."""
    a = math.radians(angle_deg)
    return np.array([0.0, math.sin(a), math.cos(a)], np.float32)


def streaks(img, amp: float, *, per_slice: float = 0.5, angle_deg: float = 0.0, spread_deg: float = 5.0, width: float = 0.8,
            thickness: float = 1.0, seed: int = 0, device: str = "cuda"):
    """``img (Z, Y, X)`` with streaks added: ``(image, info)``; ``info["direction"]`` the streaks' mean direction (z, y, x)."""
    rng = np.random.default_rng([seed, 5157])
    v = _as_float(img, device); Z, Y, X = v.shape
    yy, xx = torch.meshgrid(torch.arange(Y, device=device, dtype=torch.float32) - (Y - 1) / 2,
                            torch.arange(X, device=device, dtype=torch.float32) - (X - 1) / 2, indexing="ij")
    half = 0.5 * math.hypot(Y, X)
    n = rng.poisson(per_slice * Z)
    reach = int(math.ceil(3 * thickness))
    zs = torch.arange(Z, device=device, dtype=torch.float32)
    art = torch.zeros_like(v)
    for _ in range(n):
        z0 = rng.uniform(-0.5, Z - 0.5)
        phi = math.radians(angle_deg + rng.uniform(-spread_deg, spread_deg))
        off = rng.uniform(-half, half)
        a = amp * rng.uniform(0.5, 1.5) * (1 if rng.random() < 0.5 else -1)
        d = (yy * math.cos(phi) - xx * math.sin(phi) - off).abs()                       # distance to the line (direction (sin, cos) in (y, x))
        prof = torch.exp(-0.5 * (d / width) ** 2)
        lo, hi = max(0, int(math.floor(z0)) - reach), min(Z, int(math.ceil(z0)) + reach + 1)
        wz = torch.exp(-0.5 * ((zs[lo:hi] - z0) / thickness) ** 2)
        art[lo:hi] += a * wz[:, None, None] * prof[None]
    info = {"kind": "streaks", "amp": amp, "per_slice": per_slice, "angle_deg": angle_deg, "spread_deg": spread_deg, "width": width,
            "thickness": thickness, "n": int(n), "direction": streak_direction(angle_deg).tolist(),
            "rms": float(art.pow(2).mean().sqrt())}
    return _finish(v + art, img), info


def rings(img, amp: float, *, n: int = 6, centre_yx=None, r_range=None, width: float = 1.0, seed: int = 0, device: str = "cuda"):
    """``img (Z, Y, X)`` with ``n`` rings about the axis at ``centre_yx`` (default: mid-y, half a block beyond x = 0),
    radii uniform over ``r_range`` (default: the radii that cross the block): ``(image, info)``."""
    rng = np.random.default_rng([seed, 7464])
    v = _as_float(img, device); Z, Y, X = v.shape
    cy, cx = centre_yx if centre_yx is not None else ((Y - 1) / 2, -0.5 * X)
    yy, xx = torch.meshgrid(torch.arange(Y, device=device, dtype=torch.float32), torch.arange(X, device=device, dtype=torch.float32), indexing="ij")
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    lo, hi = r_range if r_range is not None else (float(r.min()), float(r.max()))
    art2 = torch.zeros((Y, X), device=device)
    radii = rng.uniform(lo, hi, size=n)
    for R in radii:
        a = amp * rng.uniform(0.5, 1.5) * (1 if rng.random() < 0.5 else -1)
        art2 += a * torch.exp(-0.5 * ((r - float(R)) / width) ** 2)
    info = {"kind": "rings", "amp": amp, "n": n, "centre_yx": [float(cy), float(cx)], "r_range": [lo, hi], "width": width,
            "rms": float(art2.pow(2).mean().sqrt())}
    return _finish(v + art2[None], img), info


ARTEFACTS = {"streaks": streaks, "rings": rings}


def add(kind: str, img, amp: float, **kw):
    """:func:`streaks` or :func:`rings` by name."""
    return ARTEFACTS[kind](img, amp, **kw)
