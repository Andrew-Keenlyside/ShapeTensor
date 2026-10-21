"""Stage 3 — one tensor per population.

Each population's activation layer (stage 2) says where that population's fibres are. Its tensor is the image's
structure tensor with every gradient weighted by the activation (and by the boundary excision), integrated over the
same window ρ as the baseline: the gradients of the population's own fibres, and only those. The eigenvector of the
smallest eigenvalue is the population's direction; the orientation *gap* (λ2 − λ3) / λ2 says how unique it is
(0: no direction, 1: one clear line direction). The structure tensor of the activation image itself
(:func:`layer_tensor`) is the alternative; it is noisier (the activation is patchy at the fascicle scale).

The dev pipeline reads each population's direction from the shape features themselves (:func:`feature_products`,
:func:`shape_tensor`): the same weighting and integration, with the features' gradients in place of the image's.
"""
from __future__ import annotations

import torch

from . import st


def population_tensor(v: torch.Tensor, a: torch.Tensor, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, excision=None, floor: float = 0.0, spacing=None, products=None):
    """The population's tensor: the image's structure tensor with the gradients weighted by the population's
    activation ``a (Z, Y, X)`` plus ``floor`` (and by the excision weights, if given), normalised by the integrated
    weight — ``S_r = G_ρ ∗ ((a_r + floor) w ∇I ∇Iᵀ) / G_ρ ∗ ((a_r + floor) w)``. Where the population is present its
    fibres' gradients make the tensor; where its activation is below the floor (the activation null: no
    information about the population) the tensor falls back to the excised structure tensor. Returns
    ``S (6, Z, Y, X)``, the direction (z, y, x) and the gap. ``products``: the image's :func:`shapetensor.st.gradient_products`,
    if already computed (one set serves every population of a block)."""
    w = a.clamp_min(0) + floor
    if excision is not None:
        w = w * excision
    dev = v.device if isinstance(v, torch.Tensor) else "cuda"
    S = st.structure_tensor_3d(v, sigma, rho, truncate, weights=w, spacing=spacing, products=products, device=dev)
    d, gap = direction_and_gap(S)
    return S, d, gap


def feature_products(F: torch.Tensor, sigma: float = 1.0, truncate: float = 4.0, spacing=None) -> torch.Tensor:
    """The shape features' gradient products, summed over channels: ``(6, Z, Y, X)`` in :func:`shapetensor.st.gradient_products`
    order. Each channel is scaled to unit mean gradient energy over the block first, so every channel counts alike
    (a channel that is near-silent does not vanish, a loud one does not dominate). Integrated with an activation as the
    weight (:func:`shape_tensor`), it is a population's tensor read *directly from the shape* -- the axis along which
    that population's features change least -- instead of from the image intensity."""
    dev = F.device
    acc = None
    for c in range(F.shape[0]):
        P = st.gradient_products(F[c].float(), sigma, truncate, spacing=spacing, device=dev)
        e = (P[0] + P[1] + P[2]).mean().clamp_min(1e-12)
        acc = P / e if acc is None else acc + P / e
        del P
    return acc / F.shape[0]


def readout_products(P: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The orientation readout's per-voxel distributions ``P (n_dirs, Z, Y, X)`` over ``grid (n_dirs, 3)`` (z, y, x) as
    products for :func:`shape_tensor`, in :func:`shapetensor.st.gradient_products` order: ``tr(M) I - M`` with
    ``M = sum_u P_u u u^T``, so that the least eigenvector is the distribution's principal axis, as the least gradient
    axis is for an image. A population's direction read from the shape space without differentiating the features."""
    ux, uy, uz = grid[:, 2], grid[:, 1], grid[:, 0]
    M = torch.stack([torch.einsum("n,n...->...", c.to(P.dtype), P) for c in (ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz)])
    tr = M[0] + M[1] + M[2]
    S = -M
    S[0] += tr; S[1] += tr; S[2] += tr
    return S


def shape_tensor(products: torch.Tensor, a: torch.Tensor, rho: float = 10.0, truncate: float = 4.0, excision=None, floor: float = 0.0, spacing=None):
    """A population's tensor from the shape features: the channel-summed feature gradient products (:func:`feature_products`)
    weighted by the activation ``a`` plus ``floor`` (and by ``excision``), integrated over ``rho`` and normalised by the
    integrated weight. Returns ``S (6, Z, Y, X)``, the direction (z, y, x) and the gap, as :func:`population_tensor`."""
    w = a.clamp_min(0) + floor
    if excision is not None:
        w = w * excision
    S = st.structure_tensor_3d(a, 0.0, rho, truncate, weights=w, spacing=spacing, products=products, device=a.device)
    d, gap = direction_and_gap(S)
    return S, d, gap


def layer_tensor(a: torch.Tensor, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, spacing=None):
    """The structure tensor of one activation layer ``a (Z, Y, X)``: ``S (6, Z, Y, X)`` [xx, yy, zz, xy, xz, yz], the
    direction ``(3, Z, Y, X)`` in (z, y, x) order and the gap."""
    S = st.structure_tensor_3d(a, sigma, rho, truncate, spacing=spacing, device=a.device if isinstance(a, torch.Tensor) else "cuda")
    d, gap = direction_and_gap(S)
    return S, d, gap


def direction_and_gap(S: torch.Tensor):
    """The fibre direction (smallest eigenvalue's vector, in (z, y, x)) and the orientation gap of ``S (6, ...)``."""
    val, vec = st.eig_special_3d(S, device=S.device if isinstance(S, torch.Tensor) else "cuda")
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
