"""Small plotting helpers shared by the demonstration scripts: direction-encoded colour, sticks, streamline
segments, and the site-grid field builders (eigen direction and gap from a packed tensor, fixels from an ODF)."""
from __future__ import annotations

import math

import numpy as np
import torch

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]


def dec_rgb(d, w=None):
    """DEC colour from directions ``(3, ...)`` in (z, y, x) order: R = |x|, G = |y|, B = |z|, times ``w`` in [0, 1]."""
    rgb = np.stack([np.abs(d[2]), np.abs(d[1]), np.abs(d[0])], -1)
    if w is not None:
        rgb = rgb * np.clip(w, 0, 1)[..., None]
    return np.clip(rgb, 0, 1)


def sticks(ax, cz, cx, d, L, colors, lw=1.2):
    """Axial sticks at (cz, cx) mm for (z, y, x) directions ``d (n, 3)``; length by the in-plane part."""
    from matplotlib.collections import LineCollection
    inpl = np.hypot(d[:, 0], d[:, 2])
    ux, uz = d[:, 2] / np.maximum(inpl, 1e-9), d[:, 0] / np.maximum(inpl, 1e-9)
    h = L * inpl / 2
    segs = np.stack([np.stack([cx - ux * h, cz - uz * h], -1), np.stack([cx + ux * h, cz + uz * h], -1)], 1)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 1.0, capstyle="round"))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round"))


def eig_field(S6: torch.Tensor):
    """``(6, nz, ny, nx)`` packed tensor on a site grid -> directions ``(1, 3, nz, ny, nx)`` (z, y, x), gap, cl."""
    from .readout import common, single
    Kp = S6.reshape(6, -1).T
    e = single.eigen(common.unpack(Kp, 3))
    shp = S6.shape[1:]
    return e["dir"].T.reshape((1, 3) + shp).contiguous(), e["gap"].reshape(shp), e["cl"].reshape(shp)


def fixel_field(O: torch.Tensor, width_deg: float, rel: float, gate: torch.Tensor | None = None, nmax: int = 3, lmax: int = 8):
    """ODF coefficients ``(nz, ny, nx, ncoef)`` -> fixels ``(nmax, 3, nz, ny, nx)`` (z, y, x), fractions ``(nmax, ...)``
    and the count; zero where absent or where ``gate`` is True."""
    from .odf import csd, peaks, ring_profile, zonal_kernel
    shp = O.shape[:3]
    kern = zonal_kernel(lmax, ring_profile(math.radians(width_deg)))
    f = csd(O.reshape(-1, O.shape[-1]), lmax, kern)
    d, a, cnt = peaks(f, lmax, nmax=nmax, rel=rel)
    frac = a / a.sum(1, keepdim=True).clamp_min(1e-30)
    dirs = d.reshape(shp + (nmax, 3)).permute(3, 4, 0, 1, 2).contiguous()
    fracs = frac.reshape(shp + (nmax,)).permute(3, 0, 1, 2).contiguous()
    npop = cnt.reshape(shp).clone()
    if gate is not None:
        dirs[:, :, gate] = 0.0; fracs[:, gate] = 0.0; npop[gate] = 0
    return dirs, fracs, npop


def streamline_segments(ax, P, zsel, scale, lw=0.6, alpha=0.8, max_segments=60000):
    """Streamline polylines ``P (N, 2, T, 3)`` (NaN past the end) projected on (x, y), coloured by segment direction."""
    from matplotlib.collections import LineCollection
    segs, cols, n = [], [], 0
    for half in (0, 1):
        for i in range(P.shape[0]):
            q = P[i, half]; q = q[np.isfinite(q[:, 0])]
            if len(q) < 2 or not zsel(q):
                continue
            seg = np.stack([q[:-1], q[1:]], 1)
            d = seg[:, 1] - seg[:, 0]; d /= np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
            segs.append(seg[:, :, [2, 1]] * scale)
            cols.append(np.clip(np.stack([np.abs(d[:, 2]), np.abs(d[:, 1]), np.abs(d[:, 0])], 1), 0, 1))
            n += len(seg)
            if n > max_segments:
                break
    if segs:
        ax.add_collection(LineCollection(np.concatenate(segs), colors=np.concatenate(cols), linewidths=lw, alpha=alpha))


def dec_activation_overlay(d, act, img, mode: str = "saturation", gamma: float = 0.7, floor: float = 0.15, p: float = 99.0):
    """``mode``: "brightness" (DEC colour x activation on black), "saturation" or "opacity" (over the grey image)."""
    """An RGB image of a layer's orientation with its activation as *saturation* (default) or *opacity*: ``d (3, H, W)``
    directions in (z, y, x), ``act (H, W)`` the activation, ``img (H, W)`` the intensity. The orientation is never
    masked out -- where the activation is low the colour fades towards the grey image but keeps its hue."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    rgb = dec_rgb(d)
    w = np.clip(act / max(np.percentile(act, p), 1e-6), 0, 1) ** gamma
    if mode != "brightness":
        w = floor + (1 - floor) * w
    g = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    if mode == "brightness":
        return np.clip(rgb * w[..., None], 0, 1)                                   # the DEC map itself, brightness = activation (as an FA-weighted DEC)
    if mode == "saturation":
        hsv = rgb_to_hsv(rgb)
        hsv[..., 1] = hsv[..., 1] * w                                             # activation -> saturation
        hsv[..., 2] = 0.35 + 0.65 * g                                             # intensity -> value
        return hsv_to_rgb(hsv)
    out = rgb * w[..., None] + np.stack([g] * 3, -1) * (1 - w)[..., None]         # activation -> opacity over the grey image
    return np.clip(out, 0, 1)
