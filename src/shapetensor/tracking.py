"""Standard deterministic streamline tractography on a (multi-)vector site grid, batched in torch.

The field is ``dirs (K, 3, nz, ny, nx)``: up to ``K`` unit axial directions per site in (z, y, x) component order,
zero rows where absent (a single-vector field is ``K = 1``). From every seed a streamline is integrated in both
directions: at each step the fixel of the nearest site that is most aligned with the current heading is taken
(sign-flipped to the heading), the position advanced by ``step`` sites along it, and the track stops when it
leaves ``mask``, when no fixel lies within ``max_angle_deg`` of the heading (a curvature threshold), when the site
has no fixel, or at ``max_steps``. Every field is tracked by the same code and from the same seeds, so the
comparison is between fields, not trackers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

REASONS = ("running", "mask", "angle", "absent", "max_steps")


def track(dirs: torch.Tensor, mask: torch.Tensor, seeds: torch.Tensor, *, step: float = 0.5, max_angle_deg: float = 30.0,
          max_steps: int = 1000, weights: torch.Tensor | None = None, min_weight: float = 0.0, device: str | None = None) -> dict:
    """``dirs (K, 3, nz, ny, nx)``, ``mask (nz, ny, nx)`` bool, ``seeds (N, 3)`` site coordinates (z, y, x). With
    ``weights (K, nz, ny, nx)`` (the activations of a no-mask multi-vector field) a vector is followed only where
    its weight is at least ``min_weight`` (the activation null ``tau`` is the natural choice).

    Returns ``points (N, 2, max_steps + 1, 3)`` (NaN after termination; index 1 is the backward half), ``n_steps
    (N, 2)``, ``reason (N, 2)`` (index into :data:`REASONS`), ``used_secondary (N, 2)`` (steps taken on a fixel other
    than the site's first), and ``length`` in sites.
    """
    dev = device or dirs.device
    D = dirs.to(dev).float(); M = mask.to(dev).bool(); P0 = seeds.to(dev).float()
    K, _, nz, ny, nx = D.shape
    N = P0.shape[0]
    cos_min = math.cos(math.radians(max_angle_deg))
    shape = torch.tensor([nz, ny, nx], device=dev)
    Df = D.permute(2, 3, 4, 0, 1).reshape(-1, K, 3)                       # (S, K, 3) per site
    valid = (Df.norm(dim=-1) > 0.5)                                        # (S, K)
    if weights is not None:
        valid &= weights.to(dev).float().permute(1, 2, 3, 0).reshape(-1, K) >= min_weight
    Mf = M.reshape(-1)

    def at(p):
        idx = torch.round(p).long().clamp(min=torch.zeros(3, device=dev, dtype=torch.long), max=shape - 1)
        inside = ((p >= -0.5) & (p <= (shape - 0.5))).all(1)
        flat = (idx[:, 0] * ny + idx[:, 1]) * nx + idx[:, 2]
        return flat, inside

    points = torch.full((N, 2, max_steps + 1, 3), float("nan"), device=dev)
    n_steps = torch.zeros((N, 2), dtype=torch.long, device=dev)
    reason = torch.zeros((N, 2), dtype=torch.long, device=dev)
    used2 = torch.zeros((N, 2), dtype=torch.long, device=dev)
    # the first heading: the seed site's first fixel (seeds without one terminate at once)
    flat0, inside0 = at(P0)
    h0 = Df[flat0, 0]                                                       # (N, 3)
    ok0 = inside0 & Mf[flat0] & valid[flat0, 0]
    for half, sgn in ((0, 1.0), (1, -1.0)):
        p = P0.clone(); h = sgn * h0
        active = ok0.clone()
        points[:, half, 0] = P0
        reason[~active, half] = REASONS.index("absent")
        for t in range(1, max_steps + 1):
            if not active.any():
                break
            ia = active.nonzero(as_tuple=True)[0]
            flat, inside = at(p[ia])
            inmask = inside & Mf[flat]
            d_all = Df[flat]                                                # (n, K, 3)
            cos = (d_all * h[ia, None]).sum(-1)                             # (n, K)
            cos = torch.where(valid[flat], cos.abs(), torch.full_like(cos, -1.0))
            best, k = cos.max(1)
            has = (best > -0.5)
            turn_ok = best >= cos_min
            stop_mask = ~inmask
            stop_absent = inmask & ~has
            stop_angle = inmask & has & ~turn_ok
            go = inmask & has & turn_ok
            # record the reasons for those that stop here
            r = torch.zeros_like(flat)
            r[stop_mask] = REASONS.index("mask"); r[stop_absent] = REASONS.index("absent"); r[stop_angle] = REASONS.index("angle")
            stopped = ia[~go]
            reason[stopped, half] = r[~go]
            active[stopped] = False
            ig = ia[go]
            dsel = d_all[go][torch.arange(int(go.sum()), device=dev), k[go]]
            sign = torch.sign((dsel * h[ig]).sum(-1, keepdim=True)); sign[sign == 0] = 1.0
            dsel = dsel * sign
            p[ig] = p[ig] + step * dsel
            h[ig] = dsel
            points[ig, half, t] = p[ig]
            n_steps[ig, half] = t
            used2[ig, half] += (k[go] != 0).long()
        still = active.nonzero(as_tuple=True)[0]
        reason[still, half] = REASONS.index("max_steps")
    return {"points": points, "n_steps": n_steps, "reason": reason, "used_secondary": used2, "length": n_steps.sum(1).float() * step}


def density(points: torch.Tensor, shape, n_steps: torch.Tensor | None = None) -> torch.Tensor:
    """Visits per site: ``(nz, ny, nx)`` counts of streamline points (one count per streamline per site)."""
    nz, ny, nx = shape
    P = points.reshape(-1, points.shape[-2], 3) if points.dim() == 4 else points
    out = torch.zeros(nz * ny * nx, device=points.device)
    N = P.shape[0]
    idx = torch.round(P).long()
    ok = torch.isfinite(P).all(-1) & (idx >= 0).all(-1) & (idx[..., 0] < nz) & (idx[..., 1] < ny) & (idx[..., 2] < nx)
    flat = (idx[..., 0] * ny + idx[..., 1]) * nx + idx[..., 2]
    # one count per streamline per site: unique (stream, site) pairs
    sid = torch.arange(N, device=points.device)[:, None].expand_as(flat)
    pairs = torch.stack([sid[ok], flat[ok]], 1)
    pairs = torch.unique(pairs, dim=0)
    out.index_add_(0, pairs[:, 1], torch.ones(pairs.shape[0], device=points.device))
    return out.reshape(nz, ny, nx)


def endpoint_divergence(a: dict, b: dict, step: float) -> torch.Tensor:
    """For streamlines tracked from the same seeds in two fields: the distance (sites) between the two tracks at the
    arclength where the shorter of the two halves ends, averaged over the two halves. Equal tracks give 0."""
    out = torch.zeros(a["n_steps"].shape[0], device=a["points"].device)
    for half in (0, 1):
        n = torch.minimum(a["n_steps"][:, half], b["n_steps"][:, half])
        i = torch.arange(n.shape[0], device=out.device)
        pa, pb = a["points"][i, half, n], b["points"][i, half, n]
        out += (pa - pb).norm(dim=-1).nan_to_num(0.0)
    return out / 2


def dice(x: torch.Tensor, y: torch.Tensor) -> float:
    a, b = x > 0, y > 0
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))
