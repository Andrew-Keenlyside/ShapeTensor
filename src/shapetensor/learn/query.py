"""Queries at any point and any description scale sigma_A: the exact target, its cube-group transform, the decoder.

The populations' orientation density at a point ``x`` and description scale ``sigma`` is additive over the shape
layers (the isotropic remainder excluded, as in :func:`~shapetensor.learn.targets.target`):

    f(u; x, sigma) = sum_L m_L(x, sigma) w(u | S_L(x, sigma)),   m_L = G_sigma * (o_L v_L),   m_L S_L = G_sigma * (o_L v_L T_L)

with ``o_L`` a layer's occupancy, ``v_L`` its visibility, ``T_L`` its label tensor (blended towards isotropic where its
orientation is not visible) and ``w`` the angular density of a tensor (a line a Watson lobe, a sheet a girdle, a blob
flat). It needs no slots, no merging rule and no cap on the number of populations; it is defined at any ``x`` (sub-voxel
included) and any ``sigma``, and coarsens exactly (it is linear in the windowed fields). It is stored in even SH.

Per training sample a fixed set of queries is drawn (points uniform in the interior, ``sigma`` from a few levels drawn
log-uniformly) and only their coefficients are kept; the cube-group augmentation moves the points and rotates the
coefficients exactly (precomputed rotation matrices in the SH basis).

The decoder (:class:`QueryDecoder`) is a local implicit function (LIIF, Chen et al. 2021): for a query it blends,
over the eight surrounding stride-4 cells, an MLP of the cell's features, the query's offset from it, the
trilinearly sampled stride-2 and stride-1 features at the query, and ``log sigma``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..filters import gaussian
from .sh import SHBasis, n_coef, real_sh
from .targets import render_odf


def _sample(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Trilinear samples of ``field (C, Z, Y, X)`` at voxel coordinates ``points (Q, 3)`` (z, y, x): ``(C, Q)``."""
    size = torch.tensor(field.shape[-3:], device=points.device, dtype=points.dtype)
    gn = 2 * points / (size - 1).clamp_min(1) - 1                                                    # align_corners: voxel centres at integers
    grid = gn.flip(-1).view(1, 1, 1, -1, 3)                                                           # grid_sample wants (x, y, z)
    return Fn.grid_sample(field[None].float(), grid, mode="bilinear", align_corners=True)[0, :, 0, 0]


def layer_density(s, drawn: dict, points: torch.Tensor, sigmas: torch.Tensor, lmax: int, kappa: float = 20.0, n: int = 600) -> torch.Tensor:
    """The populations' density in SH at ``points (Q, 3)`` (voxel coordinates) and per-query ``sigmas (Q,)`` (a few
    distinct levels): ``(Q, n_coef)``. ``s`` the sample, ``drawn`` its acquisition (occupancy, tensors, visibility)."""
    occ, T, vis = drawn["occupancy"], drawn["tensors"], drawn["visibility"]
    dev = occ.device
    shapes = [k for k, L in enumerate(s.layers) if L.role != "isotropic"]
    vis = vis.view(-1, 1, 1, 1) if vis.dim() == 1 else vis
    w = (occ[shapes] * vis[shapes]).float()                                                           # (S, Z, Y, X)
    T = T[shapes].float()
    ov = drawn.get("orientation_visibility")
    if ov is not None:
        ov = (ov.view(-1, 1, 1, 1) if ov.dim() == 1 else ov)[shapes]
        T = T * ov[:, None] + (1 - ov[:, None]) * torch.tensor([1, 1, 1, 0, 0, 0.0], device=dev).view(1, 6, 1, 1, 1) / 3
    b = SHBasis(lmax, n, dev)
    out = torch.zeros(points.shape[0], n_coef(lmax), device=dev)
    for lev in torch.unique(sigmas).tolist():
        q = sigmas == lev
        m = gaussian(w, lev, truncate=3.0)                                                            # (S, Z, Y, X)
        mS = gaussian(w[:, None] * T, lev, truncate=3.0)                                              # (S, 6, Z, Y, X)
        S_ = len(shapes)
        f = _sample(torch.cat([m, mS.reshape(S_ * 6, *m.shape[1:])]), points[q])                        # (S + 6S, q)
        mq = f[:S_].clamp_min(0); Sq = f[S_:].view(S_, 6, -1) / mq[:, None].clamp_min(1e-9)
        dens = render_odf(Sq, mq, b.grid, kappa)                                                       # (n, q)
        out[q] = (b.proj @ dens).T
    return out


def voxel_density(s, drawn: dict, lmax: int, kappa: float = 20.0, n: int = 600, chunk: int = 32768, floor: float = 1e-4,
                  desc_lmax: int = 0) -> torch.Tensor:
    """The exact per-voxel orientation density in SH, ``(n_coef, Z, Y, X)``: at every voxel, the sum over the shape
    layers of its visible occupancy times the angular density of its own label tensor (each lobe normalised over the
    sphere). A Gaussian window of any sigma applied to it is the window-averaged density of the orientations present
    -- exact for bends, branches and crossings alike (the per-layer form, :func:`layer_density`, describes a layer by
    its window-mean tensor instead). With ``desc_lmax``, also each descriptor's density (thickness, hollowness,
    angularity: ``sum_L o_L v_L q_L w``) to that order, appended: ``(n_coef(lmax) + 3 n_coef(desc_lmax), ...)``; at a
    density peak ``u``, ``q(u) = f_q(u) / f(u)`` is the descriptor of the population pointing there -- no slots."""
    occ, T, vis = drawn["occupancy"], drawn["tensors"], drawn["visibility"]
    dev = occ.device
    shapes = [k for k, L in enumerate(s.layers) if L.role != "isotropic"]
    vis = vis.view(-1, 1, 1, 1) if vis.dim() == 1 else vis
    D = drawn.get("descriptors") if desc_lmax else None
    nd = n_coef(desc_lmax) if desc_lmax else 0
    ov = drawn.get("orientation_visibility")
    if ov is not None:
        ov = ov.view(-1, 1, 1, 1) if ov.dim() == 1 else ov
    b = SHBasis(lmax, n, dev)
    gz, gy, gx = b.grid.unbind(1)
    quad = torch.stack([gz * gz, gy * gy, gx * gx, 2 * gz * gy, 2 * gz * gx, 2 * gy * gx], 1)              # (n, 6): u^T S u = quad @ S6
    nc = b.proj.shape[0]
    F = torch.zeros(nc + 3 * nd, *occ.shape[1:], device=dev)
    Ff = F.view(F.shape[0], -1)
    iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=dev) / 3
    for k in shapes:
        w = (occ[k] * vis[k if vis.shape[0] > 1 else 0]).reshape(-1).float()
        idx = (w > floor).nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        Tk = T[k].reshape(6, -1).float()
        ovk = ov[k].reshape(-1).float() if ov is not None and ov.shape[1:] != (1, 1, 1) else (ov[k].reshape(1).float() if ov is not None else None)
        for a in range(0, idx.numel(), chunk):
            ii = idx[a:a + chunk]
            Sv = Tk[:, ii]
            if ovk is not None:
                o_ = ovk[ii] if ovk.numel() > 1 else ovk
                Sv = Sv * o_ + (1 - o_) * iso[:, None]
            Sv = Sv / (Sv[0] + Sv[1] + Sv[2]).clamp_min(1e-9)                                          # trace 1
            e = torch.exp(kappa * (quad @ Sv - 1))                                                     # (n, m)
            e = e / e.mean(0, keepdim=True).clamp_min(1e-30)
            c = b.proj @ e                                                                             # (nc, m)
            Ff[:nc, ii] += c * w[ii]
            if nd and D is not None:
                q = D[k].reshape(3, -1)[:, ii].float()                                                 # (3, m)
                for j in range(3):
                    Ff[nc + j * nd: nc + (j + 1) * nd, ii] += c[:nd] * (w[ii] * q[j])
    return F


def sample_density(F: torch.Tensor, points: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """The per-voxel density ``F (c, Z, Y, X)`` windowed at each query's sigma and sampled at its point: ``(Q, c)``."""
    out = torch.zeros(points.shape[0], F.shape[0], device=F.device)
    for lev in torch.unique(sigmas).tolist():
        q = sigmas == lev
        out[q] = _sample(gaussian(F, lev, truncate=3.0), points[q]).T
    return out


def draw_queries(shape, n_points: int, n_levels: int, g: torch.Generator, device, margin: float = 12.0,
                 sigma_range=(1.5, 12.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """Query points uniform in the interior (``margin`` voxels from every face) and their sigmas: ``n_levels`` levels
    drawn log-uniformly in ``sigma_range``, each query given one."""
    lo = torch.tensor([margin] * 3, device=device); hi = torch.tensor([s - 1 - margin for s in shape], device=device, dtype=torch.float32)
    p = lo + (hi - lo) * torch.rand(n_points, 3, generator=g, device=device)
    a, b = math.log(sigma_range[0]), math.log(sigma_range[1])
    levels = torch.exp(a + (b - a) * torch.rand(n_levels, generator=g, device=device))
    return p, levels[torch.randint(0, n_levels, (n_points,), generator=g, device=device)]


_ROT: dict = {}


def sh_rotation(perm, flips, lmax: int, device) -> torch.Tensor:
    """The SH-basis matrix of a cube-group element (new axis j = old axis ``perm[j]``, reversed where ``flips[j]``):
    ``c' = R c``, for densities transformed as ``f'(v) = f(g^-1 v)``."""
    key = (tuple(perm), tuple(flips), lmax, str(device))
    if key not in _ROT:
        b = SHBasis(lmax, 600, device)
        v = b.grid; u = torch.zeros_like(v)
        for j in range(3):
            u[:, perm[j]] = (-1.0 if flips[j] else 1.0) * v[:, j]                                      # g^-1 v
        _ROT[key] = b.proj @ real_sh(u, lmax)
    return _ROT[key]


def cube_points(points: torch.Tensor, perm, flips, shape) -> torch.Tensor:
    """Voxel coordinates ``(..., 3)`` under the same cube-group element as :func:`~shapetensor.learn.losses.apply_cube`."""
    cols = []
    for j in range(3):
        c = points[..., perm[j]]
        cols.append((shape[perm[j]] - 1) - c if flips[j] else c)
    return torch.stack(cols, -1)


class QueryDecoder(nn.Module):
    """Populations' density (SH) at continuous points and scales from the network's three feature levels."""

    def __init__(self, c0: int, c1: int, c2: int, n_out: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c2 + c1 + c0 + 3 + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, feats, points: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """``feats`` (h0 stride 1, h1 stride 2, h stride 4), each ``(B, C, ...)``; ``points (B, Q, 3)`` voxel coordinates of
        the input; ``sigma (B, Q)`` voxels; ``context (B, Q)``: the visible half-width around each query in voxels (its
        distance to the nearest face of the visible region; default: of the input volume). Returns ``(B, Q, n_out)``."""
        h0, h1, h = feats
        B, Q = points.shape[:2]
        f0 = torch.stack([_sample(h0[b], points[b]) for b in range(B)])                               # (B, c0, Q): voxel centres at integers
        f1 = torch.stack([_sample(h1[b], (points[b] - 0.5) / 2) for b in range(B)])                   # stride 2: centres at 2i + 0.5
        u = (points - 1.5) / 4                                                                          # stride 4: centres at 4i + 1.5
        size = torch.tensor(h.shape[-3:], device=points.device)
        base = u.floor()
        hf = h.flatten(2)                                                                               # (B, c2, n)
        ls = torch.log(sigma.clamp_min(1e-3) / 5.0)[..., None]
        if context is None:                                                                             # the visible context around each query
            size_v = torch.tensor(h0.shape[-3:], device=points.device, dtype=points.dtype)
            context = torch.minimum(points, size_v - 1 - points).amin(-1)                               # (B, Q): distance to the volume's faces
        lc = torch.log(context.clamp(1.0, 64.0) / 32.0)[..., None]
        out = 0
        for o in range(8):
            off = torch.tensor([(o >> 2) & 1, (o >> 1) & 1, o & 1], device=points.device, dtype=points.dtype)
            idx = (base + off).clamp(min=torch.zeros(3, device=points.device), max=(size - 1).to(points.dtype))
            rel = u - idx                                                                               # (B, Q, 3) in cells
            wgt = (1 - rel.abs()).clamp_min(0).prod(-1)                                                 # trilinear weight of this corner
            lin = ((idx[..., 0] * size[1] + idx[..., 1]) * size[2] + idx[..., 2]).long()                # (B, Q)
            fc = torch.gather(hf, 2, lin[:, None].expand(-1, hf.shape[1], -1))                          # (B, c2, Q)
            z = torch.cat([fc, f1, f0], 1).transpose(1, 2)                                              # (B, Q, c)
            y = self.mlp(torch.cat([z.to(rel.dtype), rel, ls, lc], -1))
            out = out + wgt[..., None] * y
        return out
