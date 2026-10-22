"""The population model's target: the populations in each lattice cell, pooled per layer.

Each layer (population) is pooled on its own over the window: its visible volume ``w_k = occupancy_k * visibility_k``
and its shape tensor weighted by it, Gaussian-smoothed (``sigma_w`` voxels) and averaged to the lattice (``step``).
So a bend broadens one population's tensor (lower linearity) and never adds a population; populations appear only as
distinct layers. In each cell the layers are merged where they are the same population to the eye (same kind, axes
within ``merge_deg``), sorted by mass, and kept in ``K`` slots:

* ``slots (K, 6, z, y, x)`` -- each slot's shape tensor, trace 1 (zz, yy, xx, zy, zx, yx); zero where the slot is empty;
* ``mass (K, z, y, x)`` -- its visible volume fraction in the window (no normalisation by the cell's total: a faint
  population keeps its own small mass, a strong one its large mass);
* ``kind (K, z, y, x)`` -- line / sheet / blob / interface (0-3), -1 empty; ``layer (K, z, y, x)`` -- the dominant
  layer's index (the population's identity across cells), -1 empty;
* ``fraction`` -- the cell's visible shape volume (the sum of the masses, interfaces included);
* ``classes (5, z, y, x)`` -- soft shares of line / sheet / blob / interface / isotropic, summing to one (an invisible
  layer's volume counts as isotropic);
* ``size`` -- the shapes' mass-weighted radius (voxels), with ``size_mask`` where it is defined.

:func:`render_odf` turns slots into an orientation density on the sphere (for display, the chain's peak reader, and an
auxiliary loss): each slot a Bingham density ``exp(kappa uᵀ S u)`` weighted by its mass.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from ..filters import gaussian
from .generator import Sample, full3

KINDS = ("line", "sheet", "blob", "interface")
CLASSES = ("line", "sheet", "blob", "interface", "isotropic")
WF = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)                    # Frobenius weights of the 6 components (off-diagonals twice)


def _pool(x: torch.Tensor, sigma_w: float, step: int) -> torch.Tensor:
    """``(C, Z, Y, X)`` -> Gaussian window then the lattice average, ``(C, z, y, x)``."""
    return Fn.avg_pool3d(gaussian(x, sigma_w, truncate=3.0)[None], step)[0]


def _axis(T: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """The axis of tensors ``T (6, n)``: the largest eigenvector for lines and blobs, the smallest (the normal) for
    sheets and interfaces. ``(3, n)``."""
    _, V = torch.linalg.eigh(full3(T.T))
    big, small = V[..., -1], V[..., 0]
    planar = (kind == 1) | (kind == 3)
    return torch.where(planar[:, None], small, big).T


def target(s: Sample, drawn: dict, *, step: int = 4, sigma_w: float = 5.0, K: int = 4, merge_deg: float = 20.0, min_mass: float = 1e-3) -> dict:
    """The lattice target of a drawn sample (``drawn`` from :func:`~shapetensor.learn.generator.nuisance`, or the clean
    labels as ``{"occupancy", "tensors", "visibility"}``). See the module docstring."""
    occ, T, vis = drawn["occupancy"], drawn["tensors"], drawn["visibility"]
    dev = occ.device; L = len(s.layers)
    shapes = [k for k, Lr in enumerate(s.layers) if Lr.role != "isotropic"]
    kind_of = torch.tensor([KINDS.index("interface" if s.layers[k].role == "interface" else s.layers[k].kind) for k in shapes], device=dev)
    w = occ[shapes] * vis[shapes].view(-1, 1, 1, 1)                                             # visible volume per layer
    m = _pool(w, sigma_w, step)                                                                 # (S, z, y, x)
    M = torch.stack([_pool(w[i][None] * T[k], sigma_w, step) for i, k in enumerate(shapes)])   # (S, 6, z, y, x)
    cs = m.shape[1:]; n = m[0].numel(); S_ = len(shapes)
    tens = M.reshape(S_, 6, n) / m.reshape(S_, 1, n).clamp_min(1e-9)
    mass = m.reshape(S_, n)
    # merge the same population, keep K slots, in descending mass per cell
    order = mass.argsort(0, descending=True)
    slots = torch.zeros(K, 6, n, device=dev); smass = torch.zeros(K, n, device=dev)
    skind = torch.full((K, n), -1, dtype=torch.long, device=dev); slayer = torch.full((K, n), -1, dtype=torch.long, device=dev)
    saxis = torch.zeros(K, 3, n, device=dev); dropped = torch.zeros(n, device=dev)
    cosmin = math.cos(math.radians(merge_deg)); ar = torch.arange(n, device=dev)
    for j in range(S_):
        li = order[j]                                                                           # (n,) layer index (into shapes) at rank j
        t = tens[li, :, ar].T; mj = mass[li, ar]; kj = kind_of[li]                              # (6, n), (n,), (n,)
        ax = _axis(t, kj)
        live = mj > min_mass
        same = (skind == kj[None]) & ((saxis * ax[None]).sum(1).abs() >= cosmin) & (smass > 0)
        has = same.any(0) & live
        best = torch.where(same, smass, torch.full_like(smass, -1.0)).argmax(0)                   # merge into the heaviest matching slot
        free = (smass <= 0).float().argmax(0); nfree = (smass <= 0).any(0)
        new = live & ~has & nfree
        tgt = torch.where(has, best, free)
        sel = has | new
        oh = Fn.one_hot(tgt, K).T.bool() & sel[None]                                            # (K, n)
        mt = smass + mj[None]
        slots = torch.where(oh[:, None], (slots * smass[:, None] + t[None] * mj[None, None]) / mt[:, None].clamp_min(1e-9), slots)
        smass = torch.where(oh, mt, smass)
        skind = torch.where(oh & new[None], kj[None].expand(K, n), skind)
        slayer = torch.where(oh & new[None], torch.tensor(shapes, device=dev)[li][None].expand(K, n), slayer)
        saxis = torch.where(oh[:, None], _axis(slots.permute(1, 0, 2).reshape(6, -1), skind.clamp_min(0).reshape(-1)).reshape(3, K, n).permute(1, 0, 2), saxis)
        dropped = dropped + torch.where(live & ~sel, mj, torch.zeros_like(mj))
    o2 = smass.argsort(0, descending=True)                                                     # merging can reorder masses
    g = lambda x: torch.gather(x, 0, o2) if x.dim() == 2 else torch.gather(x, 0, o2[:, None].expand_as(x))
    slots, smass, skind, slayer = g(slots), g(smass), g(skind), g(slayer)
    # scalars: fraction, class shares, size
    fraction = mass.sum(0).clamp(0, 1)
    cls = torch.zeros(len(CLASSES), n, device=dev)
    for i, k in enumerate(shapes):
        cls[int(kind_of[i])] += mass[i]
    cls = cls * (fraction / mass.sum(0).clamp_min(1e-9))[None]
    cls[-1] = (1 - cls[:-1].sum(0)).clamp_min(0)
    radius = torch.tensor([s.layers[k].radius for k in shapes], device=dev)
    shape_like = torch.tensor([s.layers[k].role == "shape" for k in shapes], device=dev)
    ms = mass * shape_like[:, None]
    size = (ms * radius[:, None]).sum(0) / ms.sum(0).clamp_min(1e-9)
    rs = lambda x: x.reshape(x.shape[:-1] + cs)
    return {"slots": rs(slots), "mass": rs(smass), "kind": rs(skind), "layer": rs(slayer), "fraction": rs(fraction), "classes": rs(cls),
            "size": rs(size), "size_mask": rs(ms.sum(0) > 0.01), "dropped": rs(dropped), "step": step, "sigma_w": sigma_w}


def clean(s: Sample) -> dict:
    """The clean sample's labels in the form :func:`target` takes (visibility 1)."""
    occ = torch.stack([L.occupancy for L in s.layers]); T = torch.stack([L.tensor for L in s.layers])
    return {"occupancy": occ, "tensors": T, "visibility": torch.ones(occ.shape[0], device=occ.device)}


def render_odf(slots: torch.Tensor, mass: torch.Tensor, grid: torch.Tensor, kappa: float = 20.0) -> torch.Tensor:
    """The orientation density of slots ``(K, 6, ...)`` with masses ``(K, ...)`` on directions ``grid (n, 3)``: each
    slot ``exp(kappa (uᵀ S u - 1))`` normalised to unit mean over the grid, weighted by its mass; ``(n, ...)``. A line
    slot is a lobe (Watson, ``kappa``), a sheet a girdle (``kappa / 2``), a blob flat; a bend-broadened tensor a broad
    lobe."""
    zz, yy, xx, zy, zx, yx = slots.unbind(1)
    gz, gy, gx = grid[:, 0], grid[:, 1], grid[:, 2]
    sh = lambda v: v.view((-1,) + (1,) * (slots.dim() - 2))                                    # (n, 1, ...) against the slots' spatial dims
    q = (sh(gz * gz)[:, None] * zz[None] + sh(gy * gy)[:, None] * yy[None] + sh(gx * gx)[:, None] * xx[None]
         + 2 * (sh(gz * gy)[:, None] * zy[None] + sh(gz * gx)[:, None] * zx[None] + sh(gy * gx)[:, None] * yx[None]))   # (n, K, ...)
    f = torch.exp(kappa * (q - 1))
    f = f / f.mean(0, keepdim=True).clamp_min(1e-12)
    return (f * mass[None]).sum(1)
