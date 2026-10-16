"""Separable Gaussian filtering on the GPU with scipy-identical kernels.

Every kernel is scipy's ``_gaussian_kernel1d(sigma, order, radius)`` with ``radius = int(truncate * sigma + 0.5)``,
applied as scipy applies it (``correlate1d`` with the reversed kernel, i.e. a convolution) under ``mode='nearest'``
(replicate padding). So ``gaussian(x, s, order=(0, 0, 1), truncate=t)`` equals
``scipy.ndimage.gaussian_filter(x, s, order=[0, 0, 1], mode='nearest', truncate=t)`` to float32 rounding, which is what
BRIDGE's vertex stage computes (``bridge.stages._vertex_fields._structure_tensor``). Checked in tests/test_filters.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def kernel1d(sigma: float, order: int = 0, truncate: float = 3.0) -> np.ndarray:
    """scipy's Gaussian (derivative) kernel: ``q(x) * phi(x) / sum(phi)``, q the Hermite-type polynomial."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    phi = np.exp(-0.5 / sigma ** 2 * x ** 2)
    phi /= phi.sum()
    if order == 0:
        q = np.ones_like(x)
    elif order == 1:
        q = -x / sigma ** 2
    elif order == 2:
        q = (x ** 2 / sigma ** 2 - 1.0) / sigma ** 2
    else:
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    return q * phi


def filter_axis(x: torch.Tensor, kernel, axis: int, chunk_rows: int = 1 << 22) -> torch.Tensor:
    """Convolve ``x`` with a 1-D kernel along ``axis``, replicate padding (scipy ``mode='nearest'``).

    The axis is moved last and the rest flattened into rows, filtered in chunks of ``chunk_rows`` rows so the padded
    copy stays bounded. Any dtype torch's conv1d takes; the kernel is cast to ``x.dtype``.
    """
    k = torch.as_tensor(np.ascontiguousarray(np.asarray(kernel, dtype=np.float64)[::-1]), dtype=x.dtype, device=x.device)
    r = (k.numel() - 1) // 2
    xm = x.movedim(axis, -1)
    shape = xm.shape
    xf = xm.reshape(-1, 1, shape[-1])
    out = torch.empty_like(xf)
    w = k.view(1, 1, -1)
    for a in range(0, xf.shape[0], chunk_rows):
        part = xf[a:a + chunk_rows]
        if r:
            part = F.pad(part, (r, r), mode="replicate")
        out[a:a + chunk_rows] = F.conv1d(part, w)
    return out.reshape(shape).movedim(-1, axis)


def gaussian(x: torch.Tensor, sigma, order=(0, 0, 0), truncate: float = 3.0, axes=(-3, -2, -1)) -> torch.Tensor:
    """Separable Gaussian (derivative) filter over ``axes``; ``sigma`` a scalar or one value per axis."""
    sig = (sigma,) * len(axes) if np.isscalar(sigma) else tuple(sigma)
    for ax, s, o in zip(axes, sig, order):
        if s <= 0 and o == 0:
            continue
        x = filter_axis(x, kernel1d(float(s), int(o), truncate), ax)
    return x


def nyquist121(x: torch.Tensor, axes=(-3, -2, -1)) -> torch.Tensor:
    """The [1 2 1] / 4 filter along each axis: zero response at Nyquist, a 0.7-voxel blur otherwise. A 4-D field
    ``(C, Z, Y, X)`` is filtered channel by channel in place, so no field-sized copy is made."""
    k = np.array([0.25, 0.5, 0.25])
    if x.ndim == 4 and axes == (-3, -2, -1):
        for c in range(x.shape[0]):
            y = x[c]
            for ax in (0, 1, 2):
                y = filter_axis(y, k, ax)
            x[c] = y
        return x
    for ax in axes:
        x = filter_axis(x, k, ax)
    return x


def gaussian_at_rows(x: torch.Tensor, sigma: float, rows, axis: int, truncate: float = 3.0) -> torch.Tensor:
    """The Gaussian along ``axis`` evaluated only at the given index ``rows`` (a weighted sum per row, replicate
    padding), returned with ``axis`` reduced to ``len(rows)``. Exact and cheap where few output rows are needed."""
    k = torch.as_tensor(kernel1d(sigma, 0, truncate), dtype=x.dtype, device=x.device)
    r = (k.numel() - 1) // 2
    n = x.shape[axis]
    outs = []
    for j in rows:
        idx = torch.arange(j - r, j + r + 1, device=x.device).clamp_(0, n - 1)
        sl = x.index_select(axis, idx)
        shape = [1] * x.ndim
        shape[axis] = -1
        outs.append((sl * k.view(shape)).sum(axis, keepdim=True))
    return torch.cat(outs, axis)
