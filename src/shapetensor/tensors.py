"""Stage 3 — one tensor per population.

Each population's activation layer (stage 2) says where that population's fibres are. Its tensor is the image's
structure tensor with every gradient weighted by the activation (and by the boundary excision), integrated over the
same window ρ as the baseline: the gradients of the population's own fibres, and only those. The eigenvector of the
smallest eigenvalue is the population's direction; the orientation *gap* (λ2 − λ3) / λ2 says how unique it is
(0: no direction, 1: one clear line direction). The structure tensor of the activation image itself
(:func:`layer_tensor`) is the alternative; it is noisier (the activation is patchy at the fascicle scale).
"""
from __future__ import annotations

import torch

from . import st


def population_tensor(v: torch.Tensor, a: torch.Tensor, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, excision=None, floor: float = 0.0, spacing=None):
    """The population's tensor: the image's structure tensor with the gradients weighted by the population's
    activation ``a (Z, Y, X)`` plus ``floor`` (and by the excision weights, if given), normalised by the integrated
    weight — ``S_r = G_ρ ∗ ((a_r + floor) w ∇I ∇Iᵀ) / G_ρ ∗ ((a_r + floor) w)``. Where the population is present its
    fibres' gradients make the tensor; where its activation is below the floor (the activation null: no
    information about the population) the tensor falls back to the excised structure tensor. Returns
    ``S (6, Z, Y, X)``, the direction (z, y, x) and the gap."""
    w = a.clamp_min(0) + floor
    if excision is not None:
        w = w * excision
    S = st.structure_tensor_3d(v, sigma, rho, truncate, weights=w, spacing=spacing)
    d, gap = direction_and_gap(S)
    return S, d, gap


def layer_tensor(a: torch.Tensor, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, spacing=None):
    """The structure tensor of one activation layer ``a (Z, Y, X)``: ``S (6, Z, Y, X)`` [xx, yy, zz, xy, xz, yz], the
    direction ``(3, Z, Y, X)`` in (z, y, x) order and the gap."""
    S = st.structure_tensor_3d(a, sigma, rho, truncate, spacing=spacing)
    d, gap = direction_and_gap(S)
    return S, d, gap


def direction_and_gap(S: torch.Tensor):
    """The fibre direction (smallest eigenvalue's vector, in (z, y, x)) and the orientation gap of ``S (6, ...)``."""
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
