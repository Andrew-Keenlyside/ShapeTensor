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


# ---------------------------------------------------------------------------------------------------------------
# the standard figure row
# ---------------------------------------------------------------------------------------------------------------

POP = ("#2f6db5", "#4e9a3f", "#c2571f")                          # population colours (blue, green, orange)
CMAPS = ("Blues", "Greens", "Oranges")


def shape_space(F, dictionary, A, tau, region, n: int = 12000, seed: int = 0):
    """A two-dimensional view of the shape space with the populations in it: the standardised (averaged) features
    of ``n`` voxels of the bool ``region`` on their two principal axes. Returns ``P (n, 2)``, ``rgb (n, 3)`` (the
    population whose activation is largest where it exceeds ``tau``, grey elsewhere), ``coloured (n,)`` and the
    flat indices ``pick`` of the voxels drawn."""
    import torch
    from matplotlib.colors import to_rgb
    flat = region.reshape(-1).nonzero(as_tuple=True)[0]
    g = torch.Generator(device=flat.device).manual_seed(seed)
    pick = flat[torch.randperm(len(flat), generator=g, device=flat.device)[:n]]
    f = dictionary.prep(F); X = dictionary.standardise(f.reshape(f.shape[0], -1)[:, pick].T); del f
    _, _, V = torch.pca_lowrank(X, q=4, center=True); P = ((X - X.mean(0)) @ V[:, :2]).cpu().numpy()
    Ap = A.reshape(A.shape[0], -1)[:, pick]; lab = Ap.argmax(0).cpu().numpy(); above = (Ap.max(0).values > tau).cpu().numpy()
    rgb = np.full((len(pick), 3), 0.82)
    for k in range(min(A.shape[0], len(POP))):
        rgb[(lab == k) & above] = to_rgb(POP[k])
    return P, rgb, above, pick


def scatter_space(ax, P, rgb, coloured, title=None):
    """The shape-space scatter, grey points under the coloured ones, square, no ticks."""
    order = np.argsort(coloured, kind="stable")
    ax.scatter(P[order, 0], P[order, 1], c=rgb[order], s=2.0, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=10)


def pipeline_row(axs, img, d_st, space, acts, vec, vec1, *, aref=None, label="image", extra=None):
    """One row of the standard figure: image | excised structure tensor | shape space | one activation layer per
    population | shape tensor, single field | multi-vector field, first vector. Every direction field is drawn as
    plain DEC at every voxel, the shape tensor's exactly as the structure tensor's; the activation appears only as
    its own layers. ``axs``: ``5 + n`` axes, ``n`` activation panels (blank beyond the populations found). Slices
    ``(H, W)``; directions ``(3, H, W)`` in (z, y, x); ``space`` = ``(P, rgb, coloured)``; ``aref`` the activation
    display maximum (the 99th percentile over the layers if None); ``extra``: ``{"st" | "single" | "first": text}``
    appended to those titles."""
    extra = extra or {}
    n = len(axs) - 5
    axs[0].imshow(img, cmap="gray", origin="lower"); axs[0].set_title(label, fontsize=10)
    axs[1].imshow(dec_rgb(d_st), origin="lower", interpolation="nearest"); axs[1].set_title(("excised structure tensor " + extra.get("st", "")).strip(), fontsize=10)
    scatter_space(axs[2], *space[:3], title=f"shape space: {len(acts)} population{'s' if len(acts) != 1 else ''} found")
    aref = aref if aref is not None else max(float(np.percentile(np.stack(acts), 99)), 1e-6)
    for k in range(n):
        ax = axs[3 + k]
        if k < len(acts):
            ax.imshow(acts[k], cmap=CMAPS[k % len(CMAPS)], vmin=0, vmax=aref, origin="lower", interpolation="nearest"); ax.set_title(f"activation layer: population {k + 1}", fontsize=10, color=POP[k % len(POP)])
        else:
            ax.imshow(np.full(img.shape, 1.0), cmap="gray", vmin=0, vmax=1, origin="lower"); ax.axis("off")
    axs[3 + n].imshow(dec_rgb(vec), origin="lower", interpolation="nearest"); axs[3 + n].set_title(("shape tensor, single " + extra.get("single", "")).strip(), fontsize=10)
    axs[4 + n].imshow(dec_rgb(vec1), origin="lower", interpolation="nearest"); axs[4 + n].set_title(("multi-vector field, first vector " + extra.get("first", "")).strip(), fontsize=10)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])


def identity_palette(n: int, seed: int = 3) -> np.ndarray:
    """``(n + 1, 3)`` colours for population labels 0..n: 0 black, the first 20 from tab20 (reordered so neighbours in
    the list differ), the rest random but bright."""
    import matplotlib
    base = np.asarray(matplotlib.colormaps["tab20"].colors)[[0, 6, 4, 2, 8, 10, 12, 14, 16, 18, 1, 7, 5, 3, 9, 11, 13, 15, 17, 19]]
    rng = np.random.default_rng(seed)
    extra = rng.uniform(0.25, 1.0, size=(max(0, n - len(base)), 3))
    return np.concatenate([np.zeros((1, 3)), base, extra])[: n + 1]


def identity_rgb(labels: np.ndarray, cell: int = 4, palette: np.ndarray | None = None, shape=None) -> np.ndarray:
    """Population identities of a slice's cells, ``labels (K, H, W)`` (0 = none; slot 0 the strongest), as an RGB
    image of ``cell`` pixels per cell: one population fills its cell, two split it along the diagonal (the first slot
    above it), three in thirds by angle about the centre."""
    K, H, W = labels.shape
    pal = identity_palette(int(labels.max())) if palette is None else palette
    yy, xx = np.mgrid[0:cell, 0:cell] + 0.5
    ang = (np.arctan2(yy - cell / 2, xx - cell / 2) + np.pi) / (2 * np.pi)                     # 0..1 about the centre
    present = labels > 0
    n = present.sum(0)
    upper = (yy < xx)[None, None]
    out = np.zeros((H * cell, W * cell, 3))
    for k in range(K):
        r = present[:k].sum(0)[..., None, None]                                                  # rank among the cell's present slots
        nn = n[..., None, None]
        own = ((nn == 1) | ((nn == 2) & (((r == 0) & upper) | ((r == 1) & ~upper)))
               | ((nn >= 3) & (ang[None, None] >= r / 3.0) & (ang[None, None] < (r + 1) / 3.0)))
        own = own & present[k][..., None, None]
        col = pal[labels[k]]                                                                     # (H, W, 3)
        tile = np.where(own[..., None], col[:, :, None, None, :], 0.0)
        out += tile.transpose(0, 2, 1, 3, 4).reshape(H * cell, W * cell, 3)
    if shape is not None:
        out = out[: shape[0], : shape[1]]
    return np.clip(out, 0, 1)
