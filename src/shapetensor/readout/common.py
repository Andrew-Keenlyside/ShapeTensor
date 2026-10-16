"""Shared algebra on ``K``: packing, the channel-summed tensor, eigen read-outs, the Cholesky factor that turns a
site's ``K`` into the small 3-way array CP works on, Westin's shape measures and BRIDGE's sign rule."""
from __future__ import annotations

import numpy as np
import torch

from ..tensor import pair_index


def unpack(Kp: torch.Tensor, nc3: int) -> torch.Tensor:
    """Packed ``(N, P)`` -> symmetric ``(N, 3C, 3C)``."""
    a, b = pair_index(nc3)
    K = torch.zeros((Kp.shape[0], nc3, nc3), dtype=Kp.dtype, device=Kp.device)
    K[:, a, b] = Kp
    K[:, b, a] = Kp
    return K


def nc3_from_packed(P: int) -> int:
    n = int(round((-1 + np.sqrt(1 + 8 * P)) / 2))
    if n * (n + 1) // 2 != P:
        raise ValueError(f"{P} is not a packed-triangle length")
    return n


def normalise_trace(K: torch.Tensor) -> torch.Tensor:
    tr = K.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)
    return K / tr[:, None, None]


def channel_sum(K: torch.Tensor) -> torch.Tensor:
    """``(N, 3C, 3C)`` -> the 3 x 3 structure tensor summed over channels."""
    N, n3, _ = K.shape
    C = n3 // 3
    return K.reshape(N, C, 3, C, 3).diagonal(dim1=1, dim2=3).sum(-1)


def channel_gram(K: torch.Tensor) -> torch.Tensor:
    """``(N, C, C)``: the channel Gram summed over axes."""
    N, n3, _ = K.shape
    C = n3 // 3
    return K.reshape(N, C, 3, C, 3).diagonal(dim1=2, dim2=4).sum(-1)


def eigh_desc(M: torch.Tensor):
    """Eigenvalues descending ``(N, 3)`` and eigenvectors as columns ``(N, 3, 3)`` (column k for value k)."""
    w, V = torch.linalg.eigh(M)
    return w.flip(-1), V.flip(-1)


def null_dir(M: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the smallest eigenvalue of symmetric ``(N, 3, 3)``."""
    return torch.linalg.eigh(M)[1][..., :, 0]


def canonical_sign(v: torch.Tensor) -> torch.Tensor:
    """BRIDGE's rule: the largest-|component| made positive; ties go to the earlier axis (z, y, x)."""
    a0, a1, a2 = v[..., 0].abs(), v[..., 1].abs(), v[..., 2].abs()
    lead = torch.where((a0 >= a1) & (a0 >= a2), v[..., 0], torch.where(a1 >= a2, v[..., 1], v[..., 2]))
    return torch.where((lead < 0)[..., None], -v, v)


def westin(vals_desc: torch.Tensor) -> dict:
    """Westin's linear / planar / spherical measures and the orientation gap ``(l2 - l3) / l2`` from descending
    eigenvalues of a GRADIENT tensor (fibre = smallest eigenvector). ``cl`` is the step detector: a sheet of
    gradients is linear; a fibre's gradients are planar."""
    l1, l2, l3 = vals_desc[..., 0], vals_desc[..., 1], vals_desc[..., 2]
    tr = (l1 + l2 + l3).clamp_min(1e-30)
    return {"cl": (l1 - l2) / tr, "cp": 2 * (l2 - l3) / tr, "cs": 3 * l3 / tr,
            "gap": (l2 - l3) / l2.clamp_min(1e-30), "l1": l1, "l2": l2, "l3": l3}


def cholesky_factor(K: torch.Tensor) -> torch.Tensor:
    """``Y (N, 3C, C, 3)`` with ``Y^T Y = K`` (trace-normalised): the per-site 3-way array CP is fitted to.
    ``K`` symmetric PSD ``(N, 3C, 3C)``; a jittered retry where the factorisation fails numerically."""
    Kd = normalise_trace(K.double())
    n3 = K.shape[-1]
    eye = torch.eye(n3, device=K.device, dtype=Kd.dtype)
    L, info = torch.linalg.cholesky_ex(Kd + 1e-9 * eye)
    bad = info > 0
    if bool(bad.any()):
        L[bad] = torch.linalg.cholesky(Kd[bad] + 1e-6 * eye)
    return L.mT.float().reshape(K.shape[0], n3, n3 // 3, 3)


def angle_deg(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Axial angle between unit vectors ``(..., 3)``, in degrees."""
    c = (u * v).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(torch.arccos(c))
