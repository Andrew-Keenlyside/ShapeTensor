"""Population axes refined from the image: a structure tensor per population, seeded by the model.

The population model separates a window's populations but draws their directions as broad lobes (its density targets
are spread to conc 0.94, ~13 deg; the lobes it draws ~23 deg; its density does not change with the query window), so
its axes are good to ~4-12 deg. A structure tensor is precise (~1 deg on a resolved fibre) but gives one direction per
window, blending crossings. Here each image gradient in the window is shared among the model's line populations by how
perpendicular it is to each axis (a fibre's gradients are perpendicular to it), and each axis is refined as the
smallest eigenvector of its own share of the gradients' dyads -- k-subspaces seeded by the model, a few iterations:
the first with a wide gate (``tau0_deg``, to capture a population from a seed ~10 deg off), then a narrow one
(``tau_deg``): at a synthetic 60-deg crossing a 15-deg gate let the other bundle's near-perpendicular gradients pull the
axes together (3-4 deg each, 53-56 deg apart), 8 deg left 1.5-2.2 (sigma 3).
"""
from __future__ import annotations

import math

import torch

from ..filters import gaussian


def refine_axes(image: torch.Tensor, points: torch.Tensor, sigma: float, axes: torch.Tensor, live: torch.Tensor, grad_sigma: float = 1.0,
                tau_deg: float = 8.0, tau0_deg: float = 15.0, iters: int = 3, chunk: int = 512) -> torch.Tensor:
    """Refined ``axes (Q, K, 3)`` of the line populations ``live (Q, K)`` at ``points (Q, 3)`` (voxels) with a Gaussian
    window ``sigma`` (voxels): gradients at ``grad_sigma``; a gradient's responsibility for population k ~ exp(-(g.a_k /
    |g|)^2 / 2 sin^2 tau) (1 when perpendicular to a_k), normalised over the live populations -- tau ``tau0_deg`` on the
    first iteration, ``tau_deg`` after; dead ones keep their input axis."""
    dev = image.device
    xs = gaussian(image.float()[None, None], grad_sigma)[0, 0]
    G = torch.stack(torch.gradient(xs), -1)                                                          # (Z, Y, X, 3)
    R = int(math.ceil(2.5 * sigma)); g1 = torch.arange(-R, R + 1, device=dev)
    off = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
    wo = torch.exp(-(off.float() ** 2).sum(-1) / (2 * sigma ** 2))                                   # (O,)
    s2s = [2 * math.sin(math.radians(tau0_deg if i == 0 else tau_deg)) ** 2 for i in range(iters)]; size = torch.tensor(image.shape, device=dev)
    out = axes.clone()
    for a0 in range(0, points.shape[0], chunk):
        p = points[a0:a0 + chunk].round().long(); c = p[:, None] + off[None]                         # (q, O, 3)
        inb = ((c >= 0) & (c < size)).all(-1); c = torch.minimum(c.clamp_min(0), size - 1)
        g = G[c[..., 0], c[..., 1], c[..., 2]] * inb[..., None]                                    # (q, O, 3)
        u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        A = torch.nn.functional.normalize(axes[a0:a0 + chunk].float(), dim=-1); L = live[a0:a0 + chunk]
        for s2 in s2s:
            r = torch.exp(-(u @ A.transpose(1, 2)) ** 2 / s2) * L[:, None, :]                       # (q, O, K)
            r = r / r.sum(-1, keepdim=True).clamp_min(1e-12)
            S = torch.einsum("o,qok,qoi,qoj->qkij", wo, r, g, g)                                     # (q, K, 3, 3): each population's ST
            V = torch.linalg.eigh(S + 1e-12 * torch.eye(3, device=dev))[1][..., 0]                  # smallest eigenvector: the axis
            A = torch.where(L[..., None], V * torch.sign((V * A).sum(-1, keepdim=True) + 1e-12), A)
        out[a0:a0 + chunk] = A
    return out
