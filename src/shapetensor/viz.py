"""Plotting helpers: direction-encoded colour (DEC), a DEC overlay weighted by a layer's activation, sticks and
streamline segments."""
from __future__ import annotations

import numpy as np


def dec_rgb(d, w=None):
    """DEC colour from directions ``(3, ...)`` in (z, y, x) order: R = |x|, G = |y|, B = |z|, times ``w`` in [0, 1]."""
    rgb = np.stack([np.abs(d[2]), np.abs(d[1]), np.abs(d[0])], -1)
    if w is not None:
        rgb = rgb * np.clip(w, 0, 1)[..., None]
    return np.clip(rgb, 0, 1)


def dec_activation_overlay(d, act, img, mode: str = "opacity", gamma: float = 0.7, floor: float = 0.0, p: float = 99.0):
    """A layer's direction as DEC hue with its activation as ``mode`` = "opacity" or "saturation" over the grey image,
    or "brightness" (DEC × activation on black). ``d (3, H, W)`` in (z, y, x), ``act (H, W)``, ``img (H, W)``. The
    orientation is never masked out: where the activation is low the colour fades towards the image."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    rgb = dec_rgb(d)
    w = np.clip(act / max(np.percentile(act, p), 1e-6), 0, 1) ** gamma
    if mode != "brightness":
        w = floor + (1 - floor) * w
    g = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    if mode == "brightness":
        return np.clip(rgb * w[..., None], 0, 1)
    if mode == "saturation":
        hsv = rgb_to_hsv(rgb); hsv[..., 1] = hsv[..., 1] * w; hsv[..., 2] = 0.35 + 0.65 * g
        return hsv_to_rgb(hsv)
    return np.clip(rgb * w[..., None] + np.stack([g] * 3, -1) * (1 - w)[..., None], 0, 1)


def sticks(ax, cz, cx, d, L, colors, lw=1.2):
    """Axial sticks at (cz, cx) for (z, y, x) directions ``d (n, 3)``; length by the in-plane part."""
    from matplotlib.collections import LineCollection
    inpl = np.hypot(d[:, 0], d[:, 2])
    ux, uz = d[:, 2] / np.maximum(inpl, 1e-9), d[:, 0] / np.maximum(inpl, 1e-9)
    h = L * inpl / 2
    segs = np.stack([np.stack([cx - ux * h, cz - uz * h], -1), np.stack([cx + ux * h, cz + uz * h], -1)], 1)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 1.0, capstyle="round"))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round"))


def streamline_segments(ax, P, zsel, scale, lw=0.6, alpha=0.8, max_segments=60000):
    """Streamline polylines ``P (N, 2, T, 3)`` (NaN past the end) projected on (x, z), coloured by segment direction."""
    from matplotlib.collections import LineCollection
    segs, cols, n = [], [], 0
    for half in (0, 1):
        for i in range(P.shape[0]):
            q = P[i, half]; q = q[np.isfinite(q[:, 0])]
            if len(q) < 2 or not zsel(q):
                continue
            seg = np.stack([q[:-1], q[1:]], 1)
            d = seg[:, 1] - seg[:, 0]; d /= np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
            segs.append(seg[:, :, [2, 0]] * scale)
            cols.append(np.clip(np.stack([np.abs(d[:, 2]), np.abs(d[:, 1]), np.abs(d[:, 0])], 1), 0, 1))
            n += len(seg)
            if n > max_segments:
                break
    if segs:
        ax.add_collection(LineCollection(np.concatenate(segs), colors=np.concatenate(cols), linewidths=lw, alpha=alpha))
