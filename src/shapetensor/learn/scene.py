"""Object scenes: the generator built around objects instead of layers that follow a field.

A scene is a set of **objects**, each with a skeleton (a curve, a tree of curves, a surface, a closed shell), attributes
that vary along it, an identity and a parent. One renderer turns them into the volume fractions, shape tensors,
descriptors (in µm) and object identities of every voxel. Everything is batched on the GPU; scalar parameters are
drawn on the CPU (numpy) so a scene never synchronises the device per random number.

What it adds over :mod:`~shapetensor.learn.generator` (the overnight review of 23 Oct):

1. **variation along an object**: radius, flattening (a round tube collapsing to a ribbon and a band, its perimeter
   kept), wall thickness, lumen (open, narrowing, closed), section shape (superellipse), contrast, beading and
   striation are smooth profiles along each curve, with discrete events (stenoses, collapses);
2. **topology**: trees grown by Murray's law (radius split r^3 = r1^3 + r2^3, the optimal branch angles), tapered
   ends, capillary-like networks (k-nearest-neighbour graphs), branch-point and endpoint maps;
3. **identity and hierarchy**: an id per object (a fibre, a vessel tree, a cell, a sheet), a segment id and a parent
   (a nucleus's cell, a fibre's bundle); per voxel the two heaviest objects and their fractions;
4. **arrangement**: fibres packed in bundles without overlap (a jittered hexagonal section), twist, fanning, dispersion;
   crossing types (angled, kissing, fanning); fibres deflected around cells and vessels;
5. **shape families**: helices, lamellar tubes (concentric walls), beaded and striated fibres, folded and fenestrated
   sheets (bilayers), foams (Voronoi walls), porous walls (a level set), cells with nuclei (nested), crystals (the old
   solids);
6. **scale**: a voxel size per scene (µm) with physical size priors per family -- what is resolvable and what is not
   follows from it (unresolved fibres are drawn as their bundle's partial volume, mostly with the streaky texture of
   their packing: strands of varying density that wander -- the direction shows inside the bundle, 27 Oct); bundle
   paths gently curved or meandering (:data:`P_TORTUOUS`); large canvases cropped for training;
7. **layout**: laminae -- a folded depth coordinate, cell density and fibre direction (radial / tangential) by depth;
8. **acquisition**: histology tears and folds, local-tomography truncation, a second contrast channel, on top of
   :func:`~shapetensor.learn.generator.nuisance` (run with CPU-drawn scalars and a capped high-pass scale);
9. **targets**: the same layer targets (one layer per population: a bundle, a tree, a sheet), plus identities,
   descriptors in µm, branch / endpoint maps and the orientation-recoverability map as a confidence target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from ..filters import gaussian
from .generator import (Layer, Sample, _composite, background, companion_field, direction_field, draw_spread, eye6, layer_density,
                        line_tensor, nuisance, render_layer, sheet_tensor, smooth_noise, spread_e, spread_tensor)
from .physics import BY_MODALITY as PHYS_BY_MODALITY, MODES as PHYS_MODES, _to_grey, draw_material, draw_sequence, form_image


def _grey(x: torch.Tensor) -> torch.Tensor:
    """Any image to the common grey range (its 1-99 % to 30-220) before routes are combined."""
    return _to_grey(x)

H = 0.5                                    # sampling step of the renderer (voxels): 8 points per voxel volume
SHAPE_LIB = None                           # the real-shape library (scripts/learn/fetch_shapes.py); None: $SHAPETENSOR_SHAPES or the default
_LIB: dict = {}


def _library():
    import os
    from pathlib import Path
    return Path(SHAPE_LIB or os.environ.get("SHAPETENSOR_SHAPES", "/home/andrew/bridge_wmcov/shapetensor/shape_library"))


def neuron_files() -> list:
    if "neurons" not in _LIB:
        d = _library() / "neurons"
        _LIB["neurons"] = sorted(d.glob("*.npz")) if d.exists() else []
    return _LIB["neurons"]


def cortical_surfaces(hemi: str):
    """The BigBrain layer surfaces of a hemisphere, pial (0) to white (6): vertices (um), outward normals, KD-trees."""
    key = ("surf", hemi)
    if key not in _LIB:
        from scipy.spatial import cKDTree
        d = _library() / "surfaces"; out = []
        for k in range(7):
            f = d / f"layer{k}_{hemi}.npz"
            if not f.exists():
                _LIB[key] = None; return None
            z = np.load(f); out.append({"V": z["vertices"].astype(np.float64), "N": z["normals"].astype(np.float64), "tree": cKDTree(z["vertices"])})
        _LIB[key] = out
    return _LIB[key]


# ------------------------------------------------------------------------------------------------------------------
# random numbers: scalars on the CPU, arrays on the device
# ------------------------------------------------------------------------------------------------------------------

class Rng:
    """Scalars from numpy (no device synchronisation); ``g`` a torch generator on the device for arrays."""

    def __init__(self, seed: int, device):
        self.np = np.random.default_rng(seed)
        self.g = torch.Generator(device=device).manual_seed(int(seed) + 104729)
        self.device = device

    def u(self, a, b):
        return float(self.np.uniform(a, b))

    def lu(self, a, b):
        return float(math.exp(self.np.uniform(math.log(a), math.log(b))))

    def i(self, a, b):                                                     # integer in [a, b]
        return int(self.np.integers(a, b + 1))

    def p(self, q):
        return bool(self.np.random() < q)

    def unit(self, n=None):
        v = self.np.normal(size=(n or 1, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return torch.tensor(v if n else v[0], dtype=torch.float32, device=self.device)

    def randn(self, *shape):
        return torch.randn(shape, generator=self.g, device=self.device)

    def rand(self, *shape):
        return torch.rand(shape, generator=self.g, device=self.device)


def smooth1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian smoothing along the last axis of ``x (N, S)`` or ``(N, S, C)`` (replicate ends), unit variance kept."""
    if sigma <= 0.3:
        return x
    r = int(3 * sigma + 0.5); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) / sigma) ** 2)
    k = k / k.norm()                                                       # white noise in, unit variance out
    sq = x.dim() == 2
    y = x[:, None] if sq else x.movedim(-1, 1).reshape(-1, 1, x.shape[1])
    y = Fn.conv1d(Fn.pad(y, (r, r), mode="replicate"), k.view(1, 1, -1))
    return y[:, 0] if sq else y.view(x.shape[0], x.shape[2], -1).movedim(1, -1)


def smooth_noise1d(rng: "Rng", N: int, S: int, sigma: float, C: int | None = None) -> torch.Tensor:
    """White noise ``(N, S)`` or ``(N, S, C)`` smoothed along ``S`` with unit variance *everywhere*: drawn ``S + 2r`` long
    and convolved without padding. (28 Oct review: ``smooth1d``'s replicated ends gave the first and last samples 3-12x
    the variance, so every bundle grown from its centre kinked there -- at the crossing point -- fibres folded back and
    trees kinked at every branch start.)"""
    if sigma <= 0.3:
        return rng.randn(N, S) if C is None else rng.randn(N, S, C)
    r = int(3 * sigma + 0.5); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=rng.device, dtype=torch.float32) / sigma) ** 2)
    k = k / k.norm()                                                       # white noise in, unit variance out
    y = Fn.conv1d(rng.randn(N * (C or 1), S + 2 * r)[:, None], k.view(1, 1, -1))[:, 0]
    return y if C is None else y.view(N, C or 1, S).movedim(1, -1)


# ------------------------------------------------------------------------------------------------------------------
# curves: paths, frames, attribute profiles
# ------------------------------------------------------------------------------------------------------------------

def smooth_paths(start: torch.Tensor, d0: torch.Tensor, length: torch.Tensor, S: int, curv: torch.Tensor, corr: float,
                 rng: Rng, exact_start: bool = False) -> torch.Tensor:
    """``N`` paths of ``S`` points from ``start (N, 3)`` along ``d0 (N, 3)``, each ``length`` voxels: the direction drifts
    by a smoothed random walk (``curv``: turning per voxel, rad; ``corr``: its correlation length, voxels).
    ``exact_start``: the direction at the start is ``d0`` itself (no first turn)."""
    ds = (length / max(S - 1, 1))[:, None, None]
    w = smooth_noise1d(rng, start.shape[0], S, corr / max(float(ds.mean()), 1e-3), 3)
    inc = w * curv[:, None, None] * ds
    d = Fn.normalize(d0[:, None] + torch.cumsum(inc, 1) - (inc[:, :1] if exact_start else 0), dim=-1)
    return start[:, None] + torch.cumsum(d * ds, 1) - d[:, :1] * ds


def centred_paths(centre: torch.Tensor, d0: torch.Tensor, length: torch.Tensor, S: int, curv: torch.Tensor, corr: float,
                  rng: Rng) -> torch.Tensor:
    """As :func:`smooth_paths`, but each path grown both ways from ``centre (N, 3)`` along +-``d0``: it passes through the
    centre with tangent ``d0`` there however much it turns (a path grown from one end can curl off the canvas and miss the
    point it was meant to cross)."""
    h = S // 2 + 1
    fw = smooth_paths(centre, d0, length / 2, h, curv, corr, rng, exact_start=True)
    bw = smooth_paths(centre, -d0, length / 2, S - h + 1, curv, corr, rng, exact_start=True)
    return torch.cat([bw.flip(1)[:, :-1], fw], 1)


def tangents(P: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(P)
    d[:, 1:-1] = P[:, 2:] - P[:, :-2]; d[:, 0] = P[:, 1] - P[:, 0]; d[:, -1] = P[:, -1] - P[:, -2]
    return Fn.normalize(d, dim=-1)


def rmf(P: torch.Tensor, T: torch.Tensor, stride: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation-minimising frames along curves (double reflection, Wang et al. 2008): ``e1, e2 (N, S, 3)``. Computed on
    every ``stride``-th sample (the frame turns slowly at the renderer's sampling) and interpolated, re-orthogonalised
    against the full tangents."""
    N, S, _ = P.shape
    if stride > 1 and S > 4 * stride:
        idx = torch.arange(0, S, stride, device=P.device)
        if int(idx[-1]) != S - 1:
            idx = torch.cat([idx, torch.tensor([S - 1], device=P.device)])
        e1c, _ = rmf(P[:, idx], tangents(P[:, idx]), 1)
        pos = torch.arange(S, device=P.device, dtype=torch.float32)
        j = torch.searchsorted(idx.float(), pos, right=True).clamp(1, idx.numel() - 1)
        w = ((pos - idx[j - 1].float()) / (idx[j] - idx[j - 1]).float()).view(1, S, 1)
        e1 = (1 - w) * e1c[:, j - 1] + w * e1c[:, j]
        e1 = Fn.normalize(e1 - (e1 * T).sum(-1, keepdim=True) * T, dim=-1)
        return e1, torch.linalg.cross(T, e1, dim=-1)
    ref = torch.where(T[:, :1, :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=P.device), torch.tensor([0, 1.0, 0], device=P.device)).expand(N, 1, 3)[:, 0]
    r = Fn.normalize(ref - (ref * T[:, 0]).sum(-1, keepdim=True) * T[:, 0], dim=-1)
    R = [r]
    for i in range(S - 1):
        v1 = P[:, i + 1] - P[:, i]; c1 = (v1 * v1).sum(-1, keepdim=True).clamp_min(1e-12)
        rL = r - (2 / c1) * (v1 * r).sum(-1, keepdim=True) * v1
        tL = T[:, i] - (2 / c1) * (v1 * T[:, i]).sum(-1, keepdim=True) * v1
        v2 = T[:, i + 1] - tL; c2 = (v2 * v2).sum(-1, keepdim=True)
        r = torch.where(c2 > 1e-12, rL - (2 / c2.clamp_min(1e-12)) * (v2 * rL).sum(-1, keepdim=True) * v2, rL)
        r = Fn.normalize(r - (r * T[:, i + 1]).sum(-1, keepdim=True) * T[:, i + 1], dim=-1)
        R.append(r)
    e1 = torch.stack(R, 1)
    return e1, torch.linalg.cross(T, e1, dim=-1)


def profile(N: int, S: int, base, rel_sd: float, corr_samples: float, rng: Rng) -> torch.Tensor:
    """A smooth positive profile ``(N, S)``: ``base * exp(rel_sd * smooth noise)``."""
    base = base if torch.is_tensor(base) else torch.full((N,), float(base), device=rng.device)
    return base[:, None] * torch.exp(rel_sd * smooth_noise1d(rng, N, S, corr_samples))


def events(N: int, S: int, rate: float, depth: tuple, width: tuple, rng: Rng) -> torch.Tensor:
    """A multiplicative profile ``(N, S)`` in (0, 1] with Gaussian dips (stenoses, collapses): ``rate`` dips per curve on
    average, depth and width (samples) drawn uniformly in the given ranges."""
    Kmax = max(1, int(3 * rate + 2))
    on = (rng.rand(N, Kmax) < rate / Kmax).float()
    s0 = rng.rand(N, Kmax) * S; w = width[0] + (width[1] - width[0]) * rng.rand(N, Kmax)
    dep = (depth[0] + (depth[1] - depth[0]) * rng.rand(N, Kmax)) * on
    s = torch.arange(S, device=rng.device, dtype=torch.float32).view(1, S, 1)
    return torch.prod(1 - dep[:, None] * torch.exp(-0.5 * ((s - s0[:, None]) / w[:, None]) ** 2), dim=-1)


@dataclass
class Curves:
    """A batch of ``N`` curves of ``S`` samples (voxel coordinates) with profiles: ``rad`` the section's nominal radius
    (voxels; the perimeter of the flattened section is kept), ``flat`` its flattening (minor / major, 1 round), ``wall``
    the wall thickness (voxels; 0 solid), ``sexp`` the superellipse exponent (2 elliptic), ``contrast`` and ``lumen_c``
    (grey levels: the wall's and the lumen contents'), ``lam`` concentric walls per curve, ``phi`` the section's rotation
    about the axis, ``density`` the volume fraction of the section that is material (an unresolved bundle < 1), ids."""
    P: torch.Tensor
    rad: torch.Tensor
    flat: torch.Tensor
    wall: torch.Tensor
    sexp: torch.Tensor
    contrast: torch.Tensor
    lumen_c: torch.Tensor
    obj: torch.Tensor
    seg: torch.Tensor
    lam: torch.Tensor | None = None
    phi: torch.Tensor | None = None
    density: torch.Tensor | None = None
    fibre_r: torch.Tensor | None = None    # descriptor: the constituents' radius (voxels) when the curve is an unresolved bundle
    fibril: torch.Tensor | None = None     # (N, 3) longitudinal fibrils inside the section (29 Oct): density depth, period (voxels), phase
    spread: torch.Tensor | None = None     # (N,) each curve's own spread angle (degrees): its material ODF (1 - e) t t^T + e (I - t t^T) / 2,
                                           # e = sin^2 spread (29 Oct); hollow walls are sheets of material instead

    @property
    def N(self):
        return self.P.shape[0]


def curves_like(P, rad, rng: Rng, obj, seg, contrast, **kw) -> Curves:
    N, S = P.shape[:2]; dev = P.device
    z = lambda v: torch.full((N, S), float(v), device=dev)
    if kw.get("spread") is None:                                                   # each curve its own spread (29 Oct)
        kw["spread"] = draw_spread(N, rng.g if rng is not None else None, dev)          # (no generator: the global one)
    return Curves(P=P, rad=rad, flat=kw.pop("flat", z(1.0)), wall=kw.pop("wall", z(0.0)), sexp=kw.pop("sexp", z(2.0)),
                  contrast=contrast if torch.is_tensor(contrast) and contrast.dim() == 2 else z(contrast),
                  lumen_c=kw.pop("lumen_c", z(0.0)), obj=obj, seg=seg, **kw)


def curves_subset(c: Curves, keep: torch.Tensor) -> Curves | None:
    """The curves of ``c`` selected by a boolean or index tensor over its ``N`` curves (None if empty)."""
    idx = torch.nonzero(keep)[:, 0] if keep.dtype == torch.bool else keep
    if idx.numel() == 0:
        return None
    pick = lambda v: None if v is None else v[idx]
    return Curves(P=c.P[idx], rad=c.rad[idx], flat=c.flat[idx], wall=c.wall[idx], sexp=c.sexp[idx], contrast=c.contrast[idx],
                  lumen_c=c.lumen_c[idx], obj=c.obj[idx], seg=c.seg[idx], lam=pick(c.lam), phi=pick(c.phi), density=pick(c.density),
                  fibre_r=pick(c.fibre_r), spread=pick(c.spread), fibril=pick(c.fibril))


# ------------------------------------------------------------------------------------------------------------------
# rendering: points with weights splatted (trilinear) onto the grid, several channels at once
# ------------------------------------------------------------------------------------------------------------------

#: the specimen's deformation (:class:`Deform`; 29 Oct: widened -- real crops look like oriented texture under strong,
#: confounding distortions): the affine's per-axis scale +-``scale``, each shear pair drawn with ``p_shear`` up to
#: +-``shear``, a bulk shift up to +-``shift`` voxels per axis; the smooth warp's amplitude up to ``amp`` voxels over
#: ``sigma`` (log-uniform), a fine warp (``p_fine``: up to ``fine_amp`` over ``fine_sigma``), a twist about a random axis
#: (``p_twist``: up to ``twist_deg`` over the canvas) and a bend (``p_bend``: curvature up to ``bend`` / voxel). Before
#: 29 Oct: scale 0.2, shear 0.15 (p 0.25 per pair), no shift, amp 6 over 8-30, no fine warp, twist or bend.
DEFORM = {"scale": 0.25, "shear": 0.3, "p_shear": 0.5, "shift": 6.0, "amp": 10.0, "sigma": (8.0, 40.0), "p_fine": 0.4,
          "fine_amp": 2.0, "fine_sigma": (3.0, 8.0), "p_twist": 0.25, "twist_deg": 45.0, "p_bend": 0.25, "bend": 0.006}
#: the deformation before 29 Oct (the HiP-CT profile keeps it: ``draw_scene(profile="hipct")``, p_warp 0.5, p_affine 0.3)
DEFORM_LEGACY = {"scale": 0.2, "shear": 0.15, "p_shear": 0.25, "shift": 0.0, "amp": 6.0, "sigma": (8.0, 30.0), "p_fine": 0.0,
                 "fine_amp": 0.0, "fine_sigma": (3.0, 8.0), "p_twist": 0.0, "twist_deg": 0.0, "p_bend": 0.0, "bend": 0.0}


class Deform:
    """A specimen's deformation on the canvas (27 Oct; widened 29 Oct, :data:`DEFORM`): material at ``x`` moves to
    ``x + (A - I)(x - c) + t + w(x)`` -- an affine about the canvas centre (scales, shears: section compression,
    shrinkage, calibration, sheared mounting), a bulk shift ``t`` and a smooth nonlinear warp ``w`` (a multi-scale
    random field, a twist, a bend), kept invertible (det >= 0.2). It is applied where the scene is
    rendered, so the geometry itself deforms: swept and splatted material moves forward exactly (its direction by the
    local Jacobian F, its volume by det F, its sizes by det F^(1/3)); dense fields are pulled back through the inverse
    map. The crop-level warp it replaces resampled the crop alone and mirrored or smeared its edge into fake kinks and
    crossings. ``valid``: where the pulled-back source lies on the canvas (crops avoid the rest: no material was
    drawn beyond it)."""

    def __init__(self, shape, rng, warp: bool, affine: bool, device, cfg: dict | None = None):
        self.shape = tuple(shape); dev = device; Z, Y, X = shape
        self.c = torch.tensor([(Z - 1) / 2, (Y - 1) / 2, (X - 1) / 2], device=dev)
        D_ = cfg or DEFORM; A = np.eye(3); t = np.zeros(3); kinds = []
        if affine:
            A = A + np.diag([rng.u(-D_["scale"], D_["scale"]) for _ in range(3)])
            for (i, j) in ((0, 1), (0, 2), (1, 2)):
                if rng.p(D_["p_shear"]):
                    A[i, j] = rng.u(-D_["shear"], D_["shear"])
            t = np.array([rng.u(-D_["shift"], D_["shift"]) for _ in range(3)]) if D_["shift"] > 0 else t   # a bulk shift
        self.A = torch.tensor(A, dtype=torch.float32, device=dev); self.t = torch.tensor(t, dtype=torch.float32, device=dev)
        sig = rng.lu(*D_["sigma"]); amp = rng.u(0, D_["amp"]) if warp else 0.0
        n3 = torch.stack([smooth_noise(self.shape, sig, rng.g, dev) for _ in range(3)]) * amp if warp else torch.zeros((3,) + self.shape, device=dev)
        if warp:
            kinds.append("smooth")
            if D_["p_fine"] > 0 and rng.p(D_["p_fine"]):                                                                # a fine warp on top (crumpling, local tears healed)
                n3 = n3 + rng.u(0, D_["fine_amp"]) * torch.stack([smooth_noise(self.shape, rng.lu(*D_["fine_sigma"]), rng.g, dev) for _ in range(3)])
                kinds.append("fine")
            P_ = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape), indexing="ij"), 0) - self.c.view(3, 1, 1, 1)
            if D_["p_twist"] > 0 and rng.p(D_["p_twist"]):                                                               # a twist about a random axis
                a = rng.unit(); s_ = (P_ * a.view(3, 1, 1, 1)).sum(0)
                k = math.radians(rng.u(-D_["twist_deg"], D_["twist_deg"])) / max(self.shape)
                pp = P_ - s_[None] * a.view(3, 1, 1, 1); th = k * s_
                rot = torch.cos(th)[None] * pp + torch.sin(th)[None] * torch.linalg.cross(a.view(3, 1, 1, 1).expand_as(pp), pp, dim=0)
                n3 = n3 + (rot - pp); kinds.append("twist")
            if D_["p_bend"] > 0 and rng.p(D_["p_bend"]):                                                                # a bend: a parabola along a random axis
                a = rng.unit(); b = rng.unit(); b = b - (b * a).sum() * a; b = b / b.norm().clamp_min(1e-6)
                s_ = (P_ * a.view(3, 1, 1, 1)).sum(0)
                n3 = n3 + 0.5 * rng.u(-D_["bend"], D_["bend"]) * (s_ ** 2)[None] * b.view(3, 1, 1, 1); kinds.append("bend")
        scale = 1.0
        for _ in range(10):                                                                       # invertible: det F >= 0.2
            G = torch.stack([torch.stack(torch.gradient(n3[i] * scale, dim=(0, 1, 2)), 0) for i in range(3)])   # (3, 3, Z, Y, X)
            det = float(torch.linalg.det(self.A[..., None, None, None].permute(2, 3, 4, 0, 1) + G.permute(2, 3, 4, 0, 1)).min())
            if det > 0.2:
                break
            scale *= 0.7
        else:                                                                                     # (never invertible: no warp)
            scale = 0.0; G = torch.zeros((3, 3) + self.shape, device=dev); det = float(torch.linalg.det(self.A))
        self.W = n3 * scale; self.G = G                                                           # the warp and its gradient (canvas)
        wmax = float(self.W.abs().max()) if warp else 0.0
        self.record = {"affine": [round(float(v), 3) for v in A.flatten()] if affine else None, "shift": [round(float(v), 2) for v in t] if affine else None,
                       "warp": (round(wmax, 2), round(sig, 1)), "warp_kinds": kinds, "det_min": round(det, 2)}
        self.prepare()

    def prepare(self):
        """The inverse map on the canvas grid (each voxel's source) and where it lies on the canvas."""
        dev = self.A.device; self.Ai = torch.linalg.inv(self.A)
        Xg = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape), indexing="ij"), -1)
        self.src = self.inverse(Xg.reshape(-1, 3)).view(*self.shape, 3)                           # each voxel's source (for dense fields)
        lim = torch.tensor(self.shape, device=dev, dtype=torch.float32) - 1
        self.valid = ((self.src >= -0.5) & (self.src <= lim + 0.5)).all(-1)
        if self._res is not None:                                                                 # the inverse not found: not valid
            self.valid &= self._res.view(self.shape) < 0.1                                             # (28 Oct review: 0.5 let a 0.36-voxel miss pass)
        self.F_src = self.forward(self.src.reshape(-1, 3))[1]                                     # F at each voxel's source (once)
        self.st_src = torch.linalg.det(self.F_src).clamp_min(1e-3).pow(1 / 3).view(self.shape)

    def _field(self, Q):
        """The warp and its gradient at ``Q (n, 3)`` (trilinear; border beyond the canvas): ``(n, 3)``, ``(n, 3, 3)``."""
        F = torch.cat([self.W, self.G.reshape(9, *self.shape)])[None]                          # (1, 12, Z, Y, X)
        out = []
        for i in range(0, Q.shape[0], 1 << 21):
            q = Q[i:i + (1 << 21)]
            sz = torch.tensor(self.shape, device=q.device, dtype=torch.float32)
            g = (2 * q / (sz - 1).clamp_min(1) - 1).flip(-1).view(1, 1, 1, -1, 3)
            out.append(Fn.grid_sample(F, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0].T)
        o = torch.cat(out)
        return o[:, :3], o[:, 3:].reshape(-1, 3, 3)

    def forward(self, Q):
        """Material at ``Q (n, 3)`` moved: ``(Q', F)``, F the deformation gradient there."""
        lin = Q + (Q - self.c) @ (self.A - torch.eye(3, device=Q.device)).T + self.t
        if float(self.record["warp"][0]) <= 0:                                                    # an affine alone: no field to sample
            return lin, self.A[None].expand(Q.shape[0], 3, 3)
        w, gw = self._field(Q)
        return lin + w, self.A[None] + gw

    def _warp_at(self, Q):
        """The warp alone at ``Q (n, 3)`` (3 channels: the inverse's fixed point needs no gradient)."""
        sz = torch.tensor(self.shape, device=Q.device, dtype=torch.float32); out = []
        for i in range(0, Q.shape[0], 1 << 21):
            g = (2 * Q[i:i + (1 << 21)] / (sz - 1).clamp_min(1) - 1).flip(-1).view(1, 1, 1, -1, 3)
            out.append(Fn.grid_sample(self.W[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0].T)
        return torch.cat(out)

    def inverse(self, Y, iters: int = 10):
        """Where the material now at ``Y (n, 3)`` came from (fixed point of y = c + A (x - c) + w(x); exact for an affine
        alone)."""
        x = self.c + (Y - self.c - self.t) @ self.Ai.T
        if float(self.record["warp"][0]) > 0:
            for _ in range(iters):
                x = self.c + (Y - self.c - self.t - self._warp_at(x)) @ self.Ai.T
            for _ in range(2):                                                                    # Newton (the fixed point contracts
                fx, Fx = self.forward(x)                                                          # slowly where |A^-1 grad w| nears 1)
                x = x - torch.linalg.solve(Fx, (fx - Y)[..., None])[..., 0]
            self._res = (self.forward(x)[0] - Y).norm(dim=-1)
        else:
            self._res = None
        return x

    def push_tensor(self, T6, F):
        """Occupancy tensors ``T6 (n, 6)`` (zz yy xx zy zx yx) carried by ``F (n, 3, 3)``: F T F^T, trace 1."""
        T = torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], -1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], -1),
                         torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], -1)], -2)
        S = F @ T @ F.transpose(-1, -2); S = S / S.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None].clamp_min(1e-9)
        return torch.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]], -1)

    def pull(self, x: torch.Tensor, mode: str = "bilinear", padding: str = "zeros") -> torch.Tensor:
        """A canvas field ``x ((C,) Z, Y, X)`` deformed: each voxel takes its source's value (beyond the canvas 0 --
        no material was drawn there -- or, ``padding="border"``, the nearest face's: background, regions, identities)."""
        sq = x.dim() == 3; v = (x[None, None] if sq else x[None]).float()
        sz = torch.tensor(self.shape, device=x.device, dtype=torch.float32)
        g = (2 * self.src / (sz - 1).clamp_min(1) - 1).flip(-1)[None]
        out = Fn.grid_sample(v, g, mode=mode, padding_mode=padding, align_corners=True)[0]
        return (out[0] if sq else out).to(x.dtype)

    def pull_dense(self, occ, tensor, desc):
        """A dense population (occupancy, tensor (6, ...), descriptors (5, ...)) deformed: values from the source, the
        tensor carried by F there, sizes scaled by det F^(1/3)."""
        od = self.pull(torch.cat([occ[None], tensor, desc]))                                      # one resample for all the channels
        o, t, d = od[0], od[1:7], od[7:].clone()
        t = self.push_tensor(t.reshape(6, -1).T, self.F_src).T.reshape(6, *self.shape)
        d[0] = d[0] * self.st_src; d[1] = d[1] * self.st_src
        return o, t, d


class Canvas:
    """Accumulates one population (layer) at a time: volume fraction, shape tensor, descriptors, contrast; and, across
    the whole scene, the (voxel, object) weights for the identity maps."""

    NDESC = 5                              # outer radius, wall, lumen fraction, flattening, angularity (voxels / 0-1)

    KEYS_MAX = 30_000_000                  # (voxel, object) keys held before they are summed into unique ones (memory)
    OBJ = 1 << 22                          # a key is (voxel * NG + group) * OBJ + object: up to 4 M objects (granule fields pass 65 536)
    NG = 64                                # populations (groups) per scene
    NV = 11                                # per key also the weighted shape tensor (6) and descriptors (5): the entities' targets
    NVD = 5
    ID_MIN = 0.05                          # the identity target: an object owns a voxel's identity only with >= 5 % of it

    def __init__(self, shape, device, deform: "Deform | None" = None):
        self.shape = tuple(shape); self.device = device; self.n = int(np.prod(shape)); self.deform = deform
        self.id_keys, self.id_w, self.id_v = [], [], []; self._nkeys = 0; self.group = 0; self._clampf = {}
        self.lumen = torch.zeros(self.n, device=device); self.lumen_c = torch.zeros(self.n, device=device)
        self.reset()

    def reset(self):
        self.occ = torch.zeros(self.n, device=self.device)
        self.acc = torch.zeros(self.n, 6 + self.NDESC + 1, device=self.device)   # tensor, descriptors, contrast (all weighted)

    def splat(self, Q, W, V, obj=None, moved: bool = False):
        """Points ``Q (n, 3)`` with weights ``W (n,)`` (volumes, voxel units) carrying ``V (n, 12)`` (weighted later).
        With a :class:`Deform` the points move with the specimen (direction, volume and sizes with them); ``moved``:
        already deformed (swept curves deform their centrelines, :func:`render_curves`)."""
        if self.deform is not None and not moved:
            Q, Fq = self.deform.forward(Q); dF = torch.linalg.det(Fq).clamp_min(1e-3)
            W = W * dF; V = V.clone(); V[:, :6] = self.deform.push_tensor(V[:, :6], Fq); st = dF.pow(1 / 3)
            V[:, 6] = V[:, 6] * st; V[:, 7] = V[:, 7] * st
        Z, Y, X = self.shape
        p0 = Q.floor(); fr = Q - p0; i0 = p0.long(); kc, wc, vc = [], [], []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    iz, iy, ix = i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx
                    w = W * (fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dx else 1 - fr[:, 2])
                    ok = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X) & (w > 0)
                    lin = ((iz * Y + iy) * X + ix)[ok]; ww = w[ok]
                    self.occ.index_add_(0, lin, ww)
                    self.acc.index_add_(0, lin, ww[:, None] * V[ok])
                    if obj is not None:                                      # the same trilinear weights per (voxel, population, object):
                        kc.append((lin * self.NG + self.group) * self.OBJ + obj[ok]); wc.append(ww); vc.append(ww[:, None] * V[ok, :self.NV])
        if obj is not None and kc:                                           # entities sum exactly to the population; summed per call first
            uk, inv = torch.unique(torch.cat(kc), return_inverse=True)
            self._add_keys(uk, torch.zeros(uk.shape[0], device=self.device).index_add_(0, inv, torch.cat(wc)),
                           torch.zeros(uk.shape[0], self.NV, device=self.device).index_add_(0, inv, torch.cat(vc)))

    def _add_keys(self, keys, w, v):
        self.id_keys.append(keys); self.id_w.append(w); self.id_v.append(v.float()); self._nkeys += keys.numel()
        if self._nkeys > self.KEYS_MAX:                                        # sum repeated keys: one per (voxel, population, object)
            self._compact()

    def _compact(self):
        if not self.id_keys:
            return
        uk, inv = torch.unique(torch.cat(self.id_keys), return_inverse=True)
        ws = torch.zeros(uk.shape[0], device=self.device).index_add_(0, inv, torch.cat(self.id_w))
        vs = torch.zeros(uk.shape[0], self.NV, device=self.device).index_add_(0, inv, torch.cat(self.id_v))
        self.id_keys, self.id_w, self.id_v, self._nkeys = [uk], [ws], [vs], uk.numel()

    def entries(self):
        """Every (voxel, population, object) with its volume ``w`` (voxel units) and the w-weighted sums of its shape
        tensor (6) and descriptors (5): ``(vox int64, group int64, obj int64, w (n,), v (n, 11))`` -- the entities' targets
        (an entity's tensor is the exact sum of its objects'); scaled where the population's occupancy was clamped."""
        self._compact()
        if not self.id_keys:
            z = torch.zeros(0, dtype=torch.long, device=self.device)
            return z, z, z, torch.zeros(0, device=self.device), torch.zeros(0, self.NV, device=self.device)
        k = self.id_keys[0]; vg = k // self.OBJ
        vox, grp, obj, w, v = vg // self.NG, vg % self.NG, k % self.OBJ, self.id_w[0], self.id_v[0]
        if self._clampf:                                                     # as the population's occupancy: at most one voxel per voxel
            w = w.clone(); v = v.clone()
            for g_, (idx, val) in self._clampf.items():
                fac = torch.ones(self.n, device=self.device); fac[idx] = val
                m = grp == g_
                w[m] = w[m] * fac[vox[m]]; v[m] = v[m] * fac[vox[m]][:, None]
        return vox, grp, obj, w, v

    def splat_lumen(self, Q, W, c, moved: bool = False):
        if self.deform is not None and not moved:
            Q, Fq = self.deform.forward(Q); W = W * torch.linalg.det(Fq).clamp_min(1e-3)
        Z, Y, X = self.shape
        n = Q.round().long()
        ok = (n[:, 0] >= 0) & (n[:, 0] < Z) & (n[:, 1] >= 0) & (n[:, 1] < Y) & (n[:, 2] >= 0) & (n[:, 2] < X)
        lin = (n[ok, 0] * Y + n[ok, 1]) * X + n[ok, 2]
        self.lumen.index_add_(0, lin, W[ok]); self.lumen_c.index_add_(0, lin, W[ok] * c[ok])

    def add_dense(self, occ, tensor, desc, contrast, obj_map=None):
        """A population given on the grid (implicit surfaces): ``occ (Z, Y, X)``, ``tensor (6, ...)``, ``desc (5, ...)``."""
        occ, tensor, desc = occ.float(), tensor.float(), desc.float()                             # (stored in half precision)
        if obj_map is not None and not torch.is_tensor(obj_map):                                   # one object for the whole population
            obj_map = torch.full(self.shape, int(obj_map), device=self.device, dtype=torch.long)
        if self.deform is not None:
            occ, tensor, desc = self.deform.pull_dense(occ, tensor, desc)
            contrast = self.deform.pull(contrast.float()) if torch.is_tensor(contrast) and contrast.dim() == 3 else contrast
            obj_map = self.deform.pull(obj_map.float(), mode="nearest", padding="border").long() if obj_map is not None else None
        o = occ.reshape(-1)
        self.occ += o
        self.acc[:, :6] += o[:, None] * tensor.reshape(6, -1).T
        self.acc[:, 6:6 + self.NDESC] += o[:, None] * desc.reshape(self.NDESC, -1).T
        self.acc[:, -1] += o * (contrast.reshape(-1) if torch.is_tensor(contrast) else float(contrast))
        if obj_map is not None:
            m = o > 1e-3; lin = torch.nonzero(m)[:, 0]
            tv = torch.cat([tensor.reshape(6, -1)[:, m].T, desc.reshape(self.NVD, -1)[:, m].T], 1)
            self._add_keys((lin * self.NG + self.group) * self.OBJ + obj_map.reshape(-1)[m], o[m], o[m, None] * tv)

    def finish(self):
        """This population's maps: occupancy (clamped), tensor (trace 1), descriptors, mean contrast map. Where the
        occupancy passed one (splats overlapping), the factor that clamps it is kept for this population's entries."""
        occ = self.occ.clamp_min(0); w = occ.clamp_min(1e-9)[:, None]
        hi = torch.nonzero(occ > 1.0)[:, 0]
        if hi.numel():
            self._clampf[self.group] = (hi, 1.0 / occ[hi])
        T = self.acc[:, :6] / w; D = self.acc[:, 6:6 + self.NDESC] / w; C = self.acc[:, -1] / w[:, 0]
        empty = occ < 1e-6
        T = torch.where(empty[:, None], eye6(self.device).view(1, 6) / 3, T)
        tr = T[:, :3].sum(1, keepdim=True).clamp_min(1e-9); T = T / tr
        sh = self.shape
        out = {"occupancy": occ.clamp(0, 1).view(sh), "tensor": T.T.reshape(6, *sh), "desc": (D * (~empty)[:, None]).T.reshape(self.NDESC, *sh),
               "contrast": C.view(sh)}
        self.reset()
        return out

    def identities(self, K: int = 2):
        """Per voxel the ``K`` heaviest objects and their share of the voxel: ``ids (K, Z, Y, X)`` (-1 none), ``frac``."""
        ids = torch.full((K, self.n), -1, dtype=torch.long, device=self.device); frac = torch.zeros(K, self.n, device=self.device)
        if self.id_keys:
            k0 = torch.cat(self.id_keys); w = torch.cat(self.id_w)
            keys = (k0 // (self.OBJ * self.NG)) * self.OBJ + k0 % self.OBJ                 # the population folded out: (voxel, object)
            uk, inv = torch.unique(keys, return_inverse=True)
            ws = torch.zeros(uk.shape[0], device=self.device).index_add_(0, inv, w)
            big = ws >= self.ID_MIN                                                         # trilinear rims: an identity needs some of the voxel
            uk, ws = uk[big], ws[big]
            vox, obj = uk // self.OBJ, uk % self.OBJ
            # sort by voxel, then by weight descending
            o1 = torch.argsort(-ws); vox1, obj1, ws1 = vox[o1], obj[o1], ws[o1]
            o2 = torch.argsort(vox1, stable=True); vox2, obj2, ws2 = vox1[o2], obj1[o2], ws1[o2]
            _, counts = torch.unique_consecutive(vox2, return_counts=True)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(vox2.shape[0], device=self.device) - torch.repeat_interleave(starts, counts)
            tot = torch.zeros(self.n, device=self.device).index_add_(0, vox2, ws2).clamp_min(1e-9)
            for k in range(K):
                m = rank == k
                ids[k, vox2[m]] = obj2[m]; frac[k, vox2[m]] = ws2[m] / tot[vox2[m]]
        return ids.view(K, *self.shape), frac.view(K, *self.shape)


def _template2d(n_side: int, device) -> torch.Tensor:
    """A centred grid on [-1, 1]^2 with ``n_side`` points per side: ``(m, 2)`` and its spacing."""
    v = (torch.arange(n_side, device=device, dtype=torch.float32) + 0.5) / n_side * 2 - 1
    u, w = torch.meshgrid(v, v, indexing="ij")
    return torch.stack([u.reshape(-1), w.reshape(-1)], 1), 2.0 / n_side


SWEEP_STEP = 0.7          # voxels: the largest step of a section's outer edge between swept samples (28 Oct: fat or
                          # wandering curves stepped 2-3 voxels at their edges -- stripes across the fibres, rays of
                          # single sections -- 29 % of the curves' volume at the training's settings)


def _sweep_refine(c: Curves, step: float | None = None, kmax: int = 16) -> torch.Tensor:
    """Samples per step each curve needs for its section's outer edge to move <= ``step`` voxels between samples (the
    centreline's step plus the section's swing as the tangent turns): ``(N,)`` powers of two, <= ``kmax``."""
    P = c.P; step = SWEEP_STEP if step is None else step
    if P.shape[1] < 3:
        return torch.ones(P.shape[0], device=P.device)
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1); T = tangents(P); dT = (T[:, 1:] - T[:, :-1]).norm(dim=-1)
    a = 2 * c.rad / (1 + c.flat.clamp(0.02, 1.0)); a = torch.maximum(a[:, 1:], a[:, :-1])
    k = torch.ceil((seg + a * dT).amax(1) / step).clamp(1, kmax)
    return torch.exp2(torch.ceil(torch.log2(k)))


def _upsample_curves(c: Curves, k: int) -> Curves:
    """``c`` with ``k`` samples per step: positions on a Catmull-Rom spline through the samples (the tangent turns
    smoothly between them), profiles linearly."""
    N, S, _ = c.P.shape; t = torch.arange(k, device=c.P.device, dtype=torch.float32) / k
    P = c.P; Pm = torch.cat([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], 1)
    p0, p1, p2, p3 = (Pm[:, i:i + S - 1, None] for i in range(4)); tt = t.view(1, 1, k, 1)
    Q = 0.5 * (2 * p1 + (p2 - p0) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * tt ** 3)
    Pn = torch.cat([Q.reshape(N, (S - 1) * k, 3), P[:, -1:]], 1); w = t.view(1, 1, k)

    def lin(v):
        if v is None or v.dim() != 2 or v.shape[1] != S:
            return v
        return torch.cat([(v[:, :-1, None] * (1 - w) + v[:, 1:, None] * w).reshape(N, -1), v[:, -1:]], 1)
    return Curves(**{f: Pn if f == "P" else lin(getattr(c, f)) for f in c.__dataclass_fields__})


THIN_SPLAT_R = 0.5                 # curves thinner than this (voxels, solid, no lumen) are splatted along their centreline, one point
                                   # per sample carrying the section's area (the trilinear splat spreads it): a dense matrix of
                                   # sub-voxel fibres in seconds (29 Oct; the swept section cost 16-36 points per sample)


def _render_thin(cv: Canvas, c: Curves, moved: bool):
    """Thin solid curves: each centreline sample splatted with the volume pi a b ds, its own material ellipse and the
    descriptors the swept path gives a solid fibre."""
    P = c.P; N, S, _ = P.shape; dev = P.device
    T = tangents(P)
    ds = torch.zeros(N, S, device=dev); seglen = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    ds[:, 1:] += 0.5 * seglen; ds[:, :-1] += 0.5 * seglen
    f = c.flat.clamp(0.02, 1.0); a = 2 * c.rad / (1 + f); b = f * a
    dens = (c.density if c.density is not None else torch.ones(N, device=dev))[:, None]
    W = (math.pi * a * b * ds * dens).reshape(-1)
    ci = torch.arange(N, device=dev)[:, None].expand(N, S).reshape(-1); Tf = T.reshape(-1, 3)
    tt = spread_tensor(Tf, spread_e(c.spread[ci])) if c.spread is not None else line_tensor(Tf)
    size = (0.5 * (a + b)).reshape(-1) if c.fibre_r is None else c.fibre_r[ci]
    z = torch.zeros_like(size)
    desc = torch.stack([size, z, z, (b / a.clamp_min(1e-9)).reshape(-1), ((c.sexp - 2) / (c.sexp + 2)).clamp(0, 1).reshape(-1)], 1)
    V = torch.cat([tt, desc, c.contrast.reshape(-1)[:, None]], 1)
    keep = W > 0
    cv.splat(P.reshape(-1, 3)[keep], W[keep], V[keep], obj=c.obj[ci][keep], moved=moved)


def render_curves(cv: Canvas, c: Curves, max_points: float = 8e6, refine: bool = True, _deformed: bool = False):
    """Sweep each curve's cross-section along it and splat the material (and the lumen contents). ``refine``: curves
    whose section's edge would step more than :data:`SWEEP_STEP` between samples are swept on a finer sampling."""
    P = c.P; N, S, _ = P.shape; dev = P.device
    moved = getattr(cv, "deform", None) is not None
    if moved and not _deformed:                                                 # the centreline deformed with the specimen; its
        T0 = tangents(P).reshape(-1, 3)                                         # section's area by det F / |F t| (the length takes
        P, Fp = cv.deform.forward(P.reshape(-1, 3)); P = P.view(N, S, 3)       # |F t|: volume by det F, as point splats and dense
        dF = torch.linalg.det(Fp).clamp_min(1e-3)                               # pulls; the section's own shear neglected; a frozen
        lam = torch.where(T0.norm(dim=-1) > 0.5, (Fp @ T0[..., None])[..., 0].norm(dim=-1), dF.pow(1 / 3))   # end's zero tangent:
        st = (dF / lam.clamp_min(0.05)).sqrt().clamp(0.3, 3.0).view(N, S)      # isotropic)
        c = Curves(**{**{f: getattr(c, f) for f in c.__dataclass_fields__}, "P": P, "rad": c.rad * st,
                      "wall": c.wall * st if c.wall is not None else None})
    thin = (c.rad.amax(1) < THIN_SPLAT_R) & (c.wall.amax(1) <= 0) & (c.lumen_c.abs().amax(1) <= 0)
    if bool(thin.any()):                                                        # sub-voxel fibres: their centreline splatted (29 Oct)
        _render_thin(cv, curves_subset(c, thin), moved)
        if bool(thin.all()):
            return
        c = curves_subset(c, ~thin); P = c.P; N, S, _ = P.shape
    if refine:
        k = _sweep_refine(c)
        if bool((k > 1).any()):
            for kb in torch.unique(k).tolist():
                sub = curves_subset(c, k == kb)
                render_curves(cv, sub if kb == 1 else _upsample_curves(sub, int(kb)), max_points, refine=False, _deformed=True)
            return
    T = tangents(P); e1, e2 = rmf(P, T)
    ds = torch.zeros(N, S, device=dev)
    seglen = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    ds[:, 1:] += 0.5 * seglen; ds[:, :-1] += 0.5 * seglen
    f = c.flat.clamp(0.02, 1.0); a = 2 * c.rad / (1 + f); b = f * a              # perimeter-preserving flattening
    t = c.wall
    lam = c.lam if c.lam is not None else torch.ones(N, device=dev, dtype=torch.long)
    phi = c.phi if c.phi is not None else torch.zeros(N, S, device=dev)
    dens = c.density if c.density is not None else torch.ones(N, device=dev)
    open_ = (t > 0) & (b - t > 0.35)                                            # a lumen narrower than ~0.7 voxel has closed
    lum_frac = torch.where(open_, ((a - t) * (b - t)).clamp_min(0) / (a * b).clamp_min(1e-9), torch.zeros_like(a))
    ang = ((c.sexp - 2) / (c.sexp + 2)).clamp(0, 1)
    fr = c.fibre_r if c.fibre_r is not None else None
    amax = a.amax(1)
    edges = [0, 1.5, 4, 10, 25, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = torch.nonzero((amax > lo) & (amax <= hi))[:, 0]
        if sel.numel() == 0:
            continue
        am = float(amax[sel].max())
        n_side = max(4, int(math.ceil(2 * am / H)))
        U, du = _template2d(n_side, dev)                                        # (m, 2)
        m = U.shape[0]
        per = max(1, int(max_points // m))
        pairs = torch.stack(torch.meshgrid(sel, torch.arange(S, device=dev), indexing="ij"), -1).reshape(-1, 2)
        keep_pair = c.rad[pairs[:, 0], pairs[:, 1]] > 1e-3
        pairs = pairs[keep_pair]
        for s0 in range(0, pairs.shape[0], per):
            pr = pairs[s0:s0 + per]; ci, si = pr[:, 0], pr[:, 1]
            A_, B_, T_, n_ = a[ci, si][:, None], b[ci, si][:, None], t[ci, si][:, None], c.sexp[ci, si][:, None]
            x = U[None, :, 0] * A_; y = U[None, :, 1] * B_
            rho_n = U[None, :, 0].abs() ** n_ + U[None, :, 1].abs() ** n_
            inside = rho_n <= 1
            op = open_[ci, si][:, None]
            inner = (x.abs() / (A_ - T_).clamp_min(1e-3)) ** n_ + (y.abs() / (B_ - T_).clamp_min(1e-3)) ** n_ < 1
            L_ = lam[ci][:, None]
            depth = (1 - rho_n.clamp_min(0) ** (1 / n_)) * 0.5 * (A_ + B_)       # distance in from the outer surface (approx.)
            band = torch.floor(depth / T_.clamp_min(1e-3))
            multi = (L_ > 1) & (T_ > 0)
            wall_pt = torch.where(multi, inside & (band.remainder(2) == 0) & (band < 2 * L_ - 1), inside & ~(op & inner))
            lumen_pt = inside & op & inner & ~multi
            w_area = (A_ * B_ * du * du * ds[ci, si][:, None])                    # volume of each template point
            ph = phi[ci, si][:, None]
            xr = x * torch.cos(ph) - y * torch.sin(ph); yr = x * torch.sin(ph) + y * torch.cos(ph)
            Q = P[ci, si][:, None] + xr[..., None] * e1[ci, si][:, None] + yr[..., None] * e2[ci, si][:, None]   # (p, m, 3)
            tt = (spread_tensor(T[ci, si], spread_e(c.spread[ci])) if c.spread is not None   # (p, 6): the fibre's material ellipse
                  else line_tensor(T[ci, si]))
            size = 0.5 * (a[ci, si] + b[ci, si]) if fr is None else fr[ci]          # the constituents' radius for an unresolved bundle
            desc = torch.stack([size, t[ci, si], lum_frac[ci, si], (b / a.clamp_min(1e-9))[ci, si], ang[ci, si]], 1)   # (p, 5)
            V = torch.cat([tt, desc, c.contrast[ci, si][:, None]], 1)            # (p, 12)
            idx = torch.nonzero(wall_pt)
            if idx.numel():
                pi, mi = idx[:, 0], idx[:, 1]
                Vp = V[pi]
                hol = (op | multi)[:, 0][pi]                                    # a hollow tube's wall (or lamellae): a sheet of material,
                if bool(hol.any()):                                             # its normal the local radial direction (29 Oct)
                    rv = xr[pi, mi][:, None] * e1[ci, si][pi] + yr[pi, mi][:, None] * e2[ci, si][pi]
                    Vp = Vp.clone(); Vp[hol, :6] = sheet_tensor(Fn.normalize(rv[hol], dim=-1))
                Wp = (w_area[:, 0] * dens[ci])[pi]
                if c.fibril is not None:                                        # fibrils: the material's density varies across the section,
                    fa, fpd, fph = c.fibril[ci][pi].unbind(-1)                  # constant along the fibre (streaks along its axis)
                    xp, yp = xr[pi, mi], yr[pi, mi]; ca, sa = torch.cos(fph), torch.sin(fph)
                    pat = 0.5 + 0.25 * (torch.sin(2 * math.pi * (xp * ca + yp * sa) / fpd + fph) + torch.sin(2 * math.pi * (yp * ca - xp * sa) / (1.3 * fpd) + 2 * fph))
                    Wp = Wp * (1 - fa * pat)
                cv.splat(Q[pi, mi], Wp, Vp, obj=c.obj[ci][pi], moved=moved)
            lidx = torch.nonzero(lumen_pt & (c.lumen_c[ci, si][:, None] != 0))
            if lidx.numel():
                pi, mi = lidx[:, 0], lidx[:, 1]
                cv.splat_lumen(Q[pi, mi], w_area[pi, 0], c.lumen_c[ci, si][pi], moved=moved)


# ------------------------------------------------------------------------------------------------------------------
# points-based families (sheets, cells) and implicit ones (foams, porous walls)
# ------------------------------------------------------------------------------------------------------------------

def noise2d(n_u: int, n_v: int, sigma: float, rng: Rng) -> torch.Tensor:
    """Smooth 2-D noise of unit standard deviation (``sigma`` in samples)."""
    x = rng.randn(1, n_u, n_v)
    x = gaussian(x, (0.0, sigma, sigma), truncate=3.0) if sigma > 0.3 else x
    return (x[0] - x.mean()) / x.std().clamp_min(1e-6)


def _solids_budget(amax: float, shape: str = "ellipsoid", points: float = 2.0e6) -> int:
    """How many solids of largest semi-axis ``amax`` (voxels) the renderer's point budget holds (a filled solid is
    sampled on a (4 amax)^3 grid; a cell only on its membrane and nucleus)."""
    ns = max(4, int(math.ceil(2 * amax / H)))
    per = 0.52 * ns ** 3 if shape != "cell" else 4 * math.pi * (amax / H) ** 2 * 3 + 0.52 * (0.6 * ns) ** 3
    return int(max(1, min(400, points / max(per, 1.0))))


def _fib_sphere(n: int, device) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n; r = torch.sqrt((1 - z * z).clamp_min(0)); ph = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([z, r * torch.sin(ph), r * torch.cos(ph)], 1)


def _rot(rng: Rng, n: int) -> torch.Tensor:
    """``n`` random rotation matrices ``(n, 3, 3)``."""
    q = torch.tensor(rng.np.normal(size=(n, 4)), dtype=torch.float32, device=rng.device); q = Fn.normalize(q, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _sym6(M: torch.Tensor) -> torch.Tensor:
    return torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)


# ------------------------------------------------------------------------------------------------------------------
# scale: the voxel size of a scene, and which families belong at it
# ------------------------------------------------------------------------------------------------------------------

VOXEL_CORE = (1.0, 60.0)          # um: the tissue scales most of the training is for (HiP-CT, histology, light microscopy)
VOXEL_WIDE = (0.002, 2000.0)      # um: nanometres (cryo-ET, electron microscopy) to millimetres (clinical CT, MRI)
P_WIDE = 0.5                      # the share of scenes drawn on the wide prior (log-uniform over it)
#: the voxel sizes (um) at which each family is what the tissue holds; outside the range its weight falls off by decades
FAMILY_SCALE = {"bundle": (0.005, 200), "tree": (0.5, 3000), "network": (0.3, 10), "helix": (0.002, 100), "sheet": (0.002, 2000),
                "foam": (0.05, 2000), "porous": (0.01, 2000), "cells": (0.1, 20), "crystals": (0.002, 2000), "radial": (0.3, 60),
                "plywood": (0.01, 20), "fascicles": (0.5, 500), "trabecular": (2, 1000), "tubules": (0.3, 30), "epithelium": (0.1, 15),
                "dendrite": (0.05, 10), "gland": (0.5, 60), "neuron": (0.2, 10), "neuropil": (0.005, 30), "organelles": (0.002, 0.3),
                "compartments": (0.002, 0.2), "myelin": (0.003, 1.0), "nuclei": (0.005, 3), "myofibrils": (0.005, 1.0), "alveoli": (1, 500),
                "jigsaw": (0.1, 5), "honeycomb": (0.05, 30), "xylem": (0.5, 60), "lattice": (0.05, 50), "organs": (100, 5000), "granules": (0.01, 60), "tissue": (0.05, 200),
                "texture": (1e-4, 1e5)}           # oriented texture: at every scale
SCALE_FALLOFF = 0.6               # decades: the width of the fall-off outside a family's range (a Gaussian in log10 voxel size)
SCALE_FLOOR = 0.02                # never zero: a family out of its range still turns up now and then (generalisation)
P_UNRESOLVED_TEXTURE = 0.85       # unresolved bundles drawn with strand texture (the rest a smooth partial-volume tube)
MATRIX_P = 0.0                    # scenes with a matrix: the space around the families filled with objects (neuropil, cells) that
                                  # carry their own ODFs, instead of a painted background and free textures (29 Oct; off until
                                  # adopted -- figures and tests set it)
MATRIX_TYPES = ("neuropil", "cells")
MATRIX_FILL = (0.3, 0.7)          # the matrix neuropil's share of the volume
MATRIX_MAX_FIBRES = 250_000       # ... at most this many fibres per canvas
MATRIX_R_MIN = 0.3                # the matrix's and ground's fibres no thinner (voxels): micro-fascicles at coarse voxels
GROUND_FILL = 1.3                 # the ground's material per volume (over-filled: the composite scales each voxel back to one)
MATRIX_TWO_P = 0.3                # two matrix kinds in one scene
P_TORTUOUS = 0.25                 # bundles whose path meanders on a short correlation length (5-15 voxels; curvature 0.03-0.08 / voxel)
TEXTURE_MAX_STRANDS = 60_000      # the texture family's strands per population (cost: ~S x this points splatted)
TEXTURE_CLUSTERS = (32, 128)      # a texture field's objects: its strands grouped into compact patches (k-means on the seeds;
                                  # about one per 100 strands, within these bounds); None: the field one object (before 29 Oct)
STRAND_OBJECTS = True             # an unresolved bundle's strands (and its smooth partial volume) each an object under the bundle
                                  # (29 Oct); False: all of them the bundle's one object
P_VISIBLE_STRANDS = 0.8           # 29 Oct: unresolved bundles' and packed tissue's strands carry most of their material (streaks
                                  # along the fibres at strength 0), resolved fibres each their own contrast; the rest faint
P_TISSUE = 0.5                    # scenes of packed tissue (Scene.tissue: fibres wall to wall), where the voxel size suits
P_GREY_TISSUE = 0.25              # of those, grey-matter-like (sparse turning fibres among dense cells and neuropil)
P_NARROW = 0.5                    # crossing angles drawn at 8-40 deg (else 5-90): narrow crossings oversampled
P_RELATED_CONTRAST = 0.4          # crossing populations of one material: contrasts within +-10 % of each other
P_FAN = 0.25                      # bundles fanning past their middle (was 0.15)
ISO_SCENE_P = 0.15                # scenes of isotropic content only (granules, crystals): there, the right answer is no direction (was 0.08)


def scale_weight(family: str, voxel_um: float) -> float:
    """How much ``family`` belongs in a scene at ``voxel_um``: 1 inside its :data:`FAMILY_SCALE` range, falling off by
    decades outside it, never below :data:`SCALE_FLOOR`."""
    lo, hi = FAMILY_SCALE.get(family, (1e-3, 1e4)); lv = math.log10(max(voxel_um, 1e-6))
    d = max(0.0, math.log10(lo) - lv, lv - math.log10(hi))
    return max(SCALE_FLOOR, math.exp(-0.5 * (d / SCALE_FALLOFF) ** 2))


def _stretch_metric(axis: torch.Tensor, s: float) -> torch.Tensor:
    """A volume-preserving linear map that stretches distances along ``axis`` by ``s`` (cells measured in it come out
    ``s`` times shorter along the axis: ``s`` < 1 elongates them, ``s`` > 1 flattens them)."""
    a = Fn.normalize(axis, dim=0)
    M = torch.eye(3, device=axis.device) + (s - 1.0) * a[:, None] * a[None, :]
    return M / s ** (1.0 / 3.0)


@dataclass
class Group:
    """One population (one layer of the targets): a bundle, a tree, a sheet, a foam, the cells' membranes..."""
    name: str
    kind: str                              # line | sheet | blob | solid (the targets' kind; descriptors carry the rest)
    radius: float                          # a typical size (voxels), for the visibility statistic
    contrast: float
    curves: list = field(default_factory=list)
    points: list = field(default_factory=list)       # (Q, W, V (n, 12), obj)
    dense: list = field(default_factory=list)        # (occ, tensor, desc, contrast, obj_map)
    layer: Layer | None = None                        # an old-style solid layer


class Scene:
    """Draws the objects of one scene in physical units on a ``shape`` canvas (see the module docstring)."""

    FAMILIES = {"bundle": 0.24, "tree": 0.11, "network": 0.05, "helix": 0.04, "sheet": 0.12, "foam": 0.04, "porous": 0.03,
                "cells": 0.09, "crystals": 0.04, "radial": 0.04, "plywood": 0.04, "fascicles": 0.05, "trabecular": 0.03,
                "tubules": 0.04, "epithelium": 0.03, "dendrite": 0.02, "gland": 0.02, "neuron": 0.03, "neuropil": 0.03,
                "organelles": 0.05, "compartments": 0.05, "myelin": 0.04, "nuclei": 0.04, "myofibrils": 0.03, "alveoli": 0.03,
                "jigsaw": 0.02, "honeycomb": 0.03, "xylem": 0.02, "lattice": 0.03, "organs": 0.05, "granules": 0.04,
                "compose": 1.5,                    # compositional populations (learn/compose.py): about half of all populations
                "texture": 1.0}                    # oriented texture fields (29 Oct): ~25 % of the populations

    def __init__(self, shape, seed: int, device="cuda", voxel_um: float | None = None, content: str = "general"):
        self.shape = tuple(shape); self.device = device; self.r = Rng(seed, device); self.seed = seed
        if voxel_um:
            self.voxel_um = float(voxel_um)
        else:                                                                                   # the core scales, or anywhere from nm to mm
            self.voxel_um = self.r.lu(*VOXEL_WIDE) if self.r.p(P_WIDE) else self.r.lu(*VOXEL_CORE)
        self.content = content
        self.groups: list[Group] = []; self.next_obj = 1
        self.branch, self.ends = [], []                    # (n, 3) positions
        self.obstacles = []                                # (centres (n, 3), radii (n,)) that fibres yield to
        self.parents = {}                                  # object id -> parent id (nuclei in cells, fibres in bundles)
        self.obj_r = {}                                    # object id -> its smallest radius (voxels): which level of the hierarchy resolves
        self.obj_spread = {}                               # object id -> its fibres' spread angle (degrees; the last curve drawn for it)
        self.depth = None                                  # the laminar depth coordinate (voxels), if a layout was drawn
        self.outside = None                                # (Z, Y, X) bool: beyond the specimen (a real cortical layout's pial side)
        self.record = {"voxel_um": round(self.voxel_um, 4), "families": []}

    # ---- helpers
    def ids(self, n: int, radius: float | torch.Tensor | None = None) -> torch.Tensor:
        """``n`` new object ids; ``radius`` (voxels, scalar or per object) recorded for the identity target."""
        out = torch.arange(self.next_obj, self.next_obj + n, device=self.device); self.next_obj += n
        if radius is not None:
            rr = radius.reshape(-1).tolist() if torch.is_tensor(radius) else [float(radius)] * n
            for o_, r_ in zip(range(int(out[0]), int(out[0]) + n), rr if len(rr) == n else rr * n):
                self.obj_r[o_] = float(r_)
        return out

    def identity_lut(self, min_r: float = 1.5) -> torch.Tensor:
        """A lookup ``id -> identity target``: an object smaller than ``min_r`` voxels is identified by its parent (up the
        chain: an unresolved fibre by its bundle), so the identity asks only what the resolution can separate."""
        lut = torch.arange(self.next_obj + 1, device=self.device)
        for o_ in range(1, self.next_obj):
            t = o_
            while self.obj_r.get(t, 1e9) < min_r and t in self.parents:
                t = self.parents[t]
            lut[o_] = t
        return lut

    def size(self, lo_um, hi_um, lo_vox=0.45, hi_vox=None) -> float | None:
        """A size drawn log-uniformly in the physical range intersected with what the canvas can render (voxels)."""
        hi_vox = hi_vox or min(self.shape) / 4
        lo, hi = max(lo_um / self.voxel_um, lo_vox), min(hi_um / self.voxel_um, hi_vox)
        return self.r.lu(lo, hi) if hi > lo * 1.05 else None

    def centre_point(self, margin=0.25) -> torch.Tensor:
        return torch.tensor([self.r.u(margin * n, (1 - margin) * n) for n in self.shape], device=self.device)

    def contrast(self, lo=15, hi=140) -> float:
        return self.r.u(lo, hi) * (1 if self.r.p(0.5) else -1)

    # ---- layout: laminae along a folded depth coordinate
    def layout_real(self) -> bool:
        """A real cortical layout from the BigBrain layer surfaces (Wagstyl et al. 2020): a window of the canvas's extent at
        the scene's voxel size, centred near a random pial point (pial side to white matter), in a random orientation.
        The depth coordinate is the continuous layer index (0 pial ... 6 white; < 0 outside the brain, > 6 white matter)
        from signed distances to the seven surfaces (their nearest vertices' tangent planes, on a coarse grid,
        interpolated); bands are the layers, cell densities follow a laminar profile, the outside is beyond the specimen."""
        r = self.r; hemi = "left" if r.p(0.5) else "right"
        S = cortical_surfaces(hemi)
        if S is None:
            return False
        for _ in range(20):                                                                   # not the medial wall (pial = white there)
            i0 = r.i(0, S[0]["V"].shape[0] - 1); v0, n0 = S[0]["V"][i0], S[0]["N"][i0]
            thick = float(S[6]["tree"].query(v0[None])[0][0])
            if thick > 800:
                break
        else:
            return False
        centre = v0 - n0 * r.u(-0.2, 1.1) * thick
        Rm = _rot(r, 1)[0].cpu().numpy().astype(np.float64)
        G = 33; ax = np.linspace(-0.5, 0.5, G)
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) * (np.array(self.shape) * self.voxel_um)
        Xw = centre + g @ Rm.T                                                              # physical um of the coarse grid
        sdist = []
        for k in range(7):
            d, j = S[k]["tree"].query(Xw, k=6)
            w = 1.0 / np.maximum(d, 1e-3)
            sk = (((S[k]["V"][j] - Xw[:, None]) * S[k]["N"][j]).sum(-1) * w).sum(1) / w.sum(1)   # > 0: deeper than surface k
            sdist.append(sk)
        sd = np.stack(sdist, 1); tau = thick / 6
        depth = np.where(sd[:, 0] < 0, sd[:, 0] / tau, 6 + np.maximum(sd[:, 6], 0) / tau)
        for k in range(6):                                                                  # between surfaces k and k+1
            inside = (sd[:, k] >= 0) & (sd[:, k + 1] < 0)
            depth = np.where(inside, k + sd[:, k] / np.maximum(sd[:, k] - sd[:, k + 1], 1e-6), depth)
        D = torch.tensor(depth.reshape(G, G, G), dtype=torch.float32, device=self.device)
        self.depth = Fn.interpolate(D[None, None], size=self.shape, mode="trilinear", align_corners=True)[0, 0]
        self.bands = [float(k) for k in range(7)]
        prof = [0.0, 0.15, 0.9, 0.6, 0.9, 0.6, 0.7, 0.2]                                     # outside, I-VI, white matter
        self.band_density = [max(0.0, p_ * r.u(0.7, 1.3)) for p_ in prof]
        gz, gy, gx = torch.gradient(self.depth)
        self.normal_field = Fn.normalize(torch.stack([gz, gy, gx]), dim=0)
        self.outside = self.depth < -0.05
        self.record["layout"] = {"bigbrain": hemi, "vertex": i0, "thickness_um": round(thick), "outside": round(float(self.outside.float().mean()), 2)}
        return True

    def layout(self):
        if self.r.p(0.5) and self.layout_real():
            return
        r = self.r; Z, Y, X = self.shape
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        g = torch.stack(torch.meshgrid(*[torch.arange(n, device=self.device, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        c = torch.tensor([Z / 2, Y / 2, X / 2], device=self.device)
        rel = g - c; u = rel @ e1; v = rel @ e2
        L = int(2 * max(self.shape)); hgt = noise2d(L, L, r.lu(10, 40), r) * r.u(0, 12)
        gu = (u / (L / 2)).clamp(-1, 1); gv = (v / (L / 2)).clamp(-1, 1)
        hs = Fn.grid_sample(hgt[None, None], torch.stack([gv, gu], -1).view(1, 1, -1, 2), align_corners=True).view(self.shape)
        self.depth = rel @ n0 - hs
        K = r.i(2, 5); span = 0.5 * max(self.shape)
        self.bands = sorted([r.u(-span, span) for _ in range(K)])
        self.band_density = [r.u(0.05, 1.0) for _ in range(K + 1)]
        gz, gy, gx = torch.gradient(self.depth)
        self.normal_field = Fn.normalize(torch.stack([gz, gy, gx]), dim=0)
        self.record["layout"] = {"bands": len(self.bands)}

    def band_of(self, P: torch.Tensor) -> torch.Tensor:
        d = self.sample_grid(self.depth[None], P)[0]
        return torch.bucketize(d, torch.tensor(self.bands, device=self.device))

    def sample_grid(self, F: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Trilinear samples of ``F (C, Z, Y, X)`` at voxel points ``P (..., 3)``: ``(C, ...)``."""
        sz = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
        gn = (2 * P / (sz - 1) - 1).flip(-1).reshape(1, 1, 1, -1, 3)
        out = Fn.grid_sample(F[None].float(), gn, align_corners=True, padding_mode="border")[0, :, 0, 0]
        return out.view(F.shape[0], *P.shape[:-1])

    # ---- curves: bundles, trees, networks, helices
    def _fibre_extras(self, N, S, rad, rng):
        """Beading, striation, hollowness (myelin-like rings), lamellae for a set of fibres."""
        kw = {}
        if rng.p(0.2):                                                      # beading: periodic swellings
            lam = rng.lu(4, 20) * float(rad.mean()); s = torch.arange(S, device=self.device, dtype=torch.float32) * H
            ph = rng.rand(N, 1) * 6.28
            rad = rad * (1 + rng.u(0.2, 0.8) * torch.relu(torch.sin(2 * math.pi * s[None] / lam + ph)) ** 2)
        con = None
        if rng.p(P_VISIBLE_STRANDS):                                        # 29 Oct: each fibre its own material density (packed fibres
            kw["density"] = 1 - rng.u(0.2, 0.5) * rng.rand(N)              # touching still show as fibres, under any physics) and, when
            if float(rad.mean()) >= 1.5:                                    # thick, longitudinal fibrils inside (streaks along the axis)
                kw["fibril"] = torch.stack([torch.full((N,), rng.u(0.3, 0.7), device=self.device), torch.full((N,), rng.lu(1.5, 4.0), device=self.device),
                                            rng.rand(N) * 6.28], 1)
        if rng.p(0.15):                                                     # striation: periodic contrast along the axis
            lam = rng.lu(2, 12); s = torch.arange(S, device=self.device, dtype=torch.float32) * H
            con = 1 + rng.u(0.2, 0.6) * torch.sin(2 * math.pi * s[None] / lam + rng.rand(N, 1) * 6.28)
        if rng.p(0.18) and float(rad.mean()) >= 1.2:                        # hollow fibres
            kw["wall"] = rad * rng.u(0.2, 0.45); kw["lumen_c"] = torch.full_like(rad, rng.u(-40, 40))
            if rng.p(0.3):
                kw["lam"] = torch.full((N,), rng.i(2, 4), device=self.device, dtype=torch.long); kw["wall"] = rad * rng.u(0.08, 0.15)
        if rng.p(0.15):                                                     # angular (square-ish or diamond-ish) sections
            kw["sexp"] = torch.full_like(rad, rng.lu(0.8, 6.0))
        if rng.p(0.15):                                                     # ribbons
            kw["flat"] = torch.full_like(rad, rng.lu(0.25, 0.8)); kw["phi"] = smooth_noise1d(rng, N, S, 20) * 0.3
        return rad, con, kw

    def bundle(self, cross_with: Curves | None = None, mode: str = "angled", axis=None, centre=None, radius=None, curv=None, fibre: dict | None = None,
               contrast: float | None = None, fan: float | None = None):
        """A bundle of fibres (resolved: packed on a jittered hexagonal section; unresolved: the bundle's partial volume).
        ``mode``: ``single``; ``angled`` / ``kissing`` / ``fanning`` (through a point of ``cross_with``); ``split`` (the
        tract divides past a point into two halves diverging by an angle -- returned as two groups, one population each)."""
        r = self.r; S0 = int(2.2 * max(self.shape) / H)
        fibre = fibre or {}
        nano = self.voxel_um < 0.3                                                          # EM scales: filaments, fibrils, axons
        Rb = radius or self.size(0.05 if nano else 20, 3000, 1.5, min(self.shape) / 5)
        rf_um = fibre.get("rf_um") or (r.lu(0.0035, 1.0) if nano else r.lu(0.25, 12.0)); rf = min(rf_um / self.voxel_um, min(self.shape) / 6)
        if Rb is None:
            return None
        Lc = 2.0 * max(self.shape); cross = None
        if cross_with is not None:                                          # through a point of an existing bundle, inside the canvas
            Pw = cross_with.P[0]; sz = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
            inner = ((Pw > 0.2 * sz) & (Pw < 0.8 * sz)).all(1)
            cand = torch.nonzero(inner)[:, 0] if bool(inner.any()) else torch.arange(Pw.shape[0] // 4, 3 * Pw.shape[0] // 4, device=self.device)
            k = int(cand[r.i(0, cand.numel() - 1)]); P0 = Pw[k]
            t0 = tangents(cross_with.P[:1])[0, k]
            p = Fn.normalize(torch.linalg.cross(t0, r.unit()), dim=0)
            if mode == "kissing":
                ang = math.radians(r.u(5, 20)); d = math.cos(ang) * t0 + math.sin(ang) * p
                nrm = Fn.normalize(torch.linalg.cross(d, torch.linalg.cross(t0, d)), dim=0)
                P0 = P0 + nrm * (float(cross_with.rad[0].mean()) + Rb)
            else:
                ang = math.radians(r.u(8, 40) if r.p(P_NARROW) else r.u(5, 90))                   # narrow crossings oversampled
                d = math.cos(ang) * t0 + math.sin(ang) * p
            c = P0; cross = (P0, t0, d)
        else:
            c = centre if centre is not None else self.centre_point(); d = axis if axis is not None else r.unit()
        S = S0                                                              # through c with direction d there (the crossing point and angle)
        tort = curv is None and r.p(P_TORTUOUS)                             # a meandering path: turns on a short correlation length, the
        cu = curv if curv is not None else (min(r.lu(0.03, 0.08), 0.4 / max(Rb, 1.0)) if tort else r.lu(0.001, 0.02))   # curvature radius
        Pc = centred_paths(c[None], d[None], torch.tensor([Lc], device=self.device), S, torch.tensor([cu], device=self.device),   # >= ~3x the
                           r.lu(5, 15) if tort else r.lu(15, 80), r)                                                         # bundle's (no folding;
                                                                                                                     # DiSCo's ~20 deg per 15 um)
        s = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
        fan = fan if fan is not None else (r.u(0.3, 1.5) if (mode == "fanning" or r.p(P_FAN)) else 0.0)
        Rs = Rb * (1 + fan * 0.1 * Fn.softplus((s - 0.5) / 0.05))            # fanning: the bundle widens past its middle (a smooth
                                                                            # knee: edge fibres turned by up to 16 deg at the centre)
        contrast = self.contrast() if contrast is None else float(contrast); bid = int(self.ids(1, Rb)[0])
        side = None
        if rf >= 0.45:                                                      # resolved fibres
            from . import fibres as FB
            nmax = int(fibre.get("max_fibres", 250 if FB.RELAX else 400)); radii = None
            if FB.PACKING == "random":                                          # Monte Carlo: dart throwing with gamma radii (no lattice)
                fvf = r.u(0.15, 0.45) if FB.RELAX else r.u(0.35, 0.6)                # with the relaxation: room for a crossing
                C_, R_ = FB.random_section(fvf, rf, r.u(4, 12), Rb, r.np, max_n=nmax)   # bundle to pass between the fibres
                o = torch.tensor(C_ / Rb, dtype=torch.float32, device=self.device); radii = torch.tensor(R_, dtype=torch.float32, device=self.device)
                sp = 2.4 * rf / Rb
            else:                                                               # packed on a jittered hexagonal section
                sp = max(2 * rf * 1.15 * r.u(1.05, 1.5), 1.0) / Rb                  # spacing clears the largest fibre (radius jitter capped at 1.15)
                n_side = int(2 / sp) + 2
                ii, jj = torch.meshgrid(torch.arange(-n_side, n_side + 1, device=self.device), torch.arange(-n_side, n_side + 1, device=self.device), indexing="ij")
                o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp
                o = o + fibre.get("jitter", 0.1) * sp * (r.rand(*o.shape) - 0.5) * 2   # jitter, kept below the gap: no overlaps
                o = o[o.norm(dim=1) <= 1 - rf / Rb]
                if o.shape[0] > nmax:
                    o = o[torch.randperm(o.shape[0], generator=r.g, device=self.device)[:nmax]]
                if o.shape[0] == 0:
                    o = torch.zeros(1, 2, device=self.device)
            N = o.shape[0]
            T = tangents(Pc); e1, e2 = rmf(Pc, T)
            tau = r.u(-0.03, 0.03); psi = tau * s * Lc
            amp = fibre.get("disp", r.u(0.3, 1.5) if FB.RELAX else 0.25)        # with the relaxation fibres may wander and intertwine;
            wmax = (30 * H) ** 2 / (4.9 * max(rf, 0.5) * Rb)                    # without it, within half a spacing: no overlaps; a
            disp = min(amp * sp, wmax) * smooth_noise1d(r, N, S, 30, 2)         # fibre bends no tighter than ~4 radii (rms; 28 Oct: a
                                                                                # fibre as wide as its bundle zig-zagged, folding its section)
            oo = o[:, None] + disp
            ox = oo[..., 0] * torch.cos(psi) - oo[..., 1] * torch.sin(psi); oy = oo[..., 0] * torch.sin(psi) + oo[..., 1] * torch.cos(psi)
            P = Pc + (Rs[None, :, None] * (ox[..., None] * e1 + oy[..., None] * e2))
            if mode == "split":                                             # the two halves diverge past s0 (smooth onset)
                v = r.unit()[:2]; v = v / v.norm().clamp_min(1e-6); s0 = r.u(0.3, 0.6); th = math.radians(r.u(15, 60))
                side = torch.where((o @ v) >= 0, 1.0, -1.0)
                ramp = smooth1d(torch.relu(s - s0)[None] * Lc, 6.0)[0] / smooth1d(torch.ones(1, S, device=self.device), 6.0)[0]
                lat = side[:, None] * math.tan(th / 2) * ramp[None]            # (N, S) voxels
                P = P + lat[..., None] * (v[0] * e1 + v[1] * e2)
                self.record.setdefault("split", []).append(round(math.degrees(th), 1))
            rad = (radii if radii is not None else rf * torch.exp(0.12 * r.randn(N)).clamp(0.8, 1.15))[:, None].expand(N, S).clone()
            rad, con, kw = self._fibre_extras(N, S, rad, r)
            if fibre.get("wall"):                                                       # hollow: a wall and contents
                kw["wall"] = rad * r.u(*fibre["wall"]); kw["lumen_c"] = torch.full_like(rad, fibre.get("lumen_c", r.u(-40, 40))); kw.pop("lam", None)
            if fibre.get("lam"):                                                        # concentric lamellae
                kw["lam"] = torch.full((N,), r.i(*fibre["lam"]), device=self.device, dtype=torch.long)
            if fibre.get("bands"):                                                      # cross bands in register (one phase for all fibres)
                per, amp, wid = fibre["bands"]; sl_ = torch.arange(S, device=self.device, dtype=torch.float32) * (Lc / max(S - 1, 1))
                fr_ = ((sl_ - r.u(0, per)) / per).remainder(1.0)
                con = 1 + amp * (torch.exp(-0.5 * (torch.minimum(fr_, 1 - fr_) / wid) ** 2) - 0.5 * ((fr_ - 0.5).abs() < 0.25).float())[None].expand(N, S)
            cmap = contrast * (con if con is not None else 1.0) * torch.ones(N, S, device=self.device)
            obj = self.ids(N, rad.amin(1))                                      # the smallest radius along each fibre
            if getattr(self, "deep", False) and N >= 6:                         # fascicles: fibres grouped around seeds on the section
                nf = r.i(2, min(7, N // 3))
                seeds = o[torch.randperm(N, generator=r.g, device=self.device)[:nf]]
                fa = torch.cdist(o, seeds).argmin(1)
                cnt = torch.bincount(fa, minlength=nf).float()
                fid = self.ids(nf, Rb * torch.sqrt(cnt / N))                      # area-equivalent radius (voxels)
                for o_, f_ in zip(obj.tolist(), fa.tolist()):
                    self.parents[o_] = int(fid[f_])
                for f_ in fid.tolist():
                    self.parents[f_] = bid
            else:
                for o_ in obj.tolist():
                    self.parents[o_] = bid
            cv = curves_like(P, rad, r, obj=obj, seg=torch.full((N,), bid, device=self.device), contrast=cmap, **kw)
            g = Group("bundle", "line", float(rad.median()), contrast, curves=[cv])
            g.offsets, g.rf = o, float(rad.max())
            self.record["families"].append({"bundle": {"fibres": N, "fibre_um": round(rf_um, 2), "radius_vox": round(Rb, 1), "fan": round(fan, 2), "mode": mode if (cross_with is not None or mode == "split") else "single"}})
        else:                                                               # unresolved fibres: the bundle's partial volume
            from . import fibres as FB
            phi_ = r.u(0.3, 0.8); one = lambda v: torch.tensor([v], device=self.device)
            vis_ = r.p(P_VISIBLE_STRANDS)                                     # 29 Oct: most draws show their strands (streaks along the
            beta = float(fibre["texture"]) if "texture" in fibre else (r.u(0.6, 0.95) if vis_ else (r.u(0.1, 0.5) if r.p(P_UNRESOLVED_TEXTURE) else 0.0))   # fibres)
            tube = bid
            if STRAND_OBJECTS:                                              # the smooth partial volume: one more child of the bundle
                tube = int(self.ids(1, rf)[0]); self.parents[tube] = bid    # (its radius the fibres': merged at any window)
            cv = curves_like(Pc, Rs[None].clone(), r, obj=one(tube), seg=one(bid), contrast=contrast, density=one(phi_ * (1 - beta)), fibre_r=one(rf))
            curves = [cv]; tex = {}
            if beta > 0:                                                    # the rest in strands: groups of sub-voxel fibres whose packing
                rs = min(r.lu(0.6, 3.0), max(0.6, Rb / 3))                  # varies (fascicles, density fluctuations), wandering -- the
                C_, R_ = FB.random_section(r.u(0.5, 0.6), rs, r.u(3, 8), Rb, r.np, max_n=600)   # streaky partial-volume texture that keeps a
                o = torch.tensor(C_ / Rb, dtype=torch.float32, device=self.device); N = o.shape[0]   # bundle's direction visible where
                fill = float(np.sum(R_ ** 2)) / Rb ** 2                     # its fibres are not (the smooth tube alone showed none)
                xi = torch.exp((r.u(0.3, 0.7) if vis_ else r.u(0.15, 0.5)) * r.randn(N)); xi = xi / xi.mean()
                dens = (beta * phi_ / max(fill, 1e-3) * xi).clamp(0.02, 1.0)
                turn, lam = math.radians(r.lu(2.0, 20.0)), r.lu(6.0, 30.0)    # strand tortuosity (rms turn) over a correlation length (voxels)
                disp = (turn * lam / Rb) * smooth_noise1d(r, N, S, lam / H, 2)
                T_ = tangents(Pc); e1, e2 = rmf(Pc, T_)
                oo = o[:, None] + disp
                P = Pc + Rs[None, :, None] * (oo[..., :1] * e1 + oo[..., 1:] * e2)
                rad = torch.tensor(R_, dtype=torch.float32, device=self.device)[:, None] * torch.exp(0.15 * smooth_noise1d(r, N, S, lam / H))
                if STRAND_OBJECTS:                                          # each strand its own object, a leaf under the bundle (29 Oct)
                    sid = self.ids(N, rad.amin(1))
                    for o_ in sid.tolist():
                        self.parents[o_] = bid
                else:
                    sid = torch.full((N,), bid, device=self.device)
                curves.append(curves_like(P, rad, r, obj=sid, seg=torch.full((N,), bid, device=self.device),
                                          contrast=contrast, density=dens, fibre_r=torch.full((N,), rf, device=self.device)))
                tex = {"texture": round(beta, 2), "strands": N, "strand_vox": round(rs, 2), "strand_turn_deg": round(math.degrees(turn), 1)}
            g = Group("bundle", "line", rf, contrast, curves=curves); g.offsets = None
            self.record["families"].append({"bundle": {"unresolved": True, "fibre_um": round(rf_um, 2), "packing": round(phi_, 2), "radius_vox": round(Rb, 1),
                                                       "mode": mode if cross_with is not None else "single", **tex}})
        g.centreline = curves_like(Pc, Rs[None].clone(), r, obj=torch.tensor([bid], device=self.device), seg=torch.tensor([bid], device=self.device), contrast=contrast)
        g.cross, g.Rb = cross, float(Rs.max())
        if tort:
            self.record["families"][-1]["bundle"]["tortuous"] = round(cu, 3)
        g.centre_pt = c                                                     # where crops centre on it (crossing point, or the fan / split / bend)
        if cross is not None:
            self.record.setdefault("cross_deg", []).append(round(math.degrees(math.acos(min(1.0, abs(float(cross[1] @ cross[2]))))), 1))
        if side is not None:                                                # one population per half
            out = []
            for sg in (1.0, -1.0):
                c_ = curves_subset(g.curves[0], side == sg)
                if c_ is not None:
                    h = Group("bundle", "line", g.radius, contrast, curves=[c_]); h.centreline, h.cross, h.Rb, h.offsets, h.centre_pt = g.centreline, None, g.Rb, None, c
                    out.append(h)
            return out
        return g

    def interleave(self, ga: Group, gb: Group):
        """Two resolved bundles crossing at an angle interdigitate: in the crossing region the fibres of each are
        regrouped into fascicles (slabs of thickness ``w`` across both directions) that alternate -- A's slab k moves to
        the k-th even place, B's to the k-th odd place, a gap of a fibre diameter between them -- so no fibre crosses
        another; the bundles thicken where they cross. A fibre's displacement fades in over a taper along it."""
        P0, tA, dB = gb.cross
        n = torch.linalg.cross(tA, dB); 
        if float(n.norm()) < 1e-3:
            return
        n = n / n.norm(); r = self.r
        rfm = max(getattr(ga, "rf", 0.5), getattr(gb, "rf", 0.5))
        w = r.u(3, 10) * 2 * rfm; gap = 2.2 * rfm
        for g, parity, g_other, R_other in ((ga, 0, gb, gb.Rb), (gb, 1, ga, ga.Rb)):
            cl = g_other.centreline.P[0, ::4] - P0                            # the other bundle's actual centreline (28 Oct review:
            cl = cl - (cl @ n)[:, None] * n                                  # its straight axis put a meandering bundle's crossing
            for c in g.curves:                                               # elsewhere: 12-13 % of the voxels shared)
                rel = c.P - P0
                h = rel @ n
                q = rel - h[..., None] * n                                   # in-plane distance from the other bundle's centreline
                dist = (q[:, :, None, :] - cl[None, None]).norm(dim=-1).amin(-1)
                taper = max(2.0 * w, 0.5 * R_other)
                wt = ((R_other + rfm + taper - dist) / taper).clamp(0, 1)
                wt = wt * wt * (3 - 2 * wt)                                  # smoothstep
                wt = smooth1d(wt, 4.0) / smooth1d(torch.ones_like(wt), 4.0)
                hc = (h * wt).sum(-1, keepdim=True) / wt.sum(-1, keepdim=True).clamp_min(1e-6)   # one slab per fibre: its place
                k = torch.floor(hc / w)                                      # in the crossing (28 Oct: per sample, a fibre
                shift = k * (w + 2 * gap) + parity * (w + gap)               # wandering over a slab's edge jumped 10-24 voxels)
                c.P = c.P + (wt * shift)[..., None] * n
        self.record.setdefault("interleaved", []).append({"slab_vox": round(w, 1)})

    def tree(self, style: str = "vessel"):
        """A branching tree grown generation by generation. ``vessel``: Murray's law (r^3 = r1^3 + r2^3) and its optimal
        angles, a wall and lumen, stenoses and collapses; ``dendrite``: Rall's 3/2 law, thin and tortuous, solid, a
        soma; ``gland``: a narrow epithelial tube (thick wall, open lumen) ending in acini (hollow spheres). Returned as
        two populations: at every branching the smaller child changes side, so a branch point always separates the
        departing branch from the parent and its continuation (the same object id on both sides)."""
        r = self.r
        if style == "dendrite":
            r0 = self.size(0.5, 6, 0.45, 3.0); rmin = max(0.15 / self.voxel_um, 0.3); ex = 1.5; tort = (0.03, 0.15); Lf = (8, 25)
        elif style == "gland":
            r0 = self.size(15, 150, 1.5, min(self.shape) / 10); rmin = max(8.0 / self.voxel_um, 0.8); ex = 3.0; tort = (0.005, 0.03); Lf = (3, 6)
        else:
            r0 = self.size(15, 800, 1.5, min(self.shape) / 7); rmin = max(3.0 / self.voxel_um, 0.45); ex = 3.0; tort = (0.005, 0.04); Lf = (5, 12)
        if r0 is None or r0 < rmin * 1.5:
            return None
        tid = int(self.ids(1, r0)[0]); contrast = self.contrast(); g = Group("tree", "line", r0, contrast)
        g.murray = []
        if style == "dendrite":
            wf, f0, lumen_c = 0.0, 1.0, 0.0
        elif style == "gland":
            wf, f0, lumen_c = r.u(0.3, 0.55), 1.0, r.u(-60, 60)
        else:
            wf = r.u(0.08, 0.3); f0 = 1.0 if r.p(0.6) else r.u(0.5, 1.0)       # arterial-like wall; veins flatter
            lumen_c = r.u(-60, 80) if r.p(0.7) else 0.0
        c = self.centre_point(0.1); d = r.unit()
        start = (c - d * r0 * 4)[None]; dirs = d[None]; rads = torch.tensor([r0], device=self.device)
        side = torch.zeros(1, device=self.device, dtype=torch.long)
        deep = getattr(self, "deep", False)
        if deep:                                                                # branches as objects: the trunk first, under the tree
            br = self.ids(1, r0); self.parents[int(br[0])] = tid
        nseg = 0; segs, sides, ends_all = [], [], []
        for depth in range(12):
            A = start.shape[0]
            if A == 0 or nseg > 300:
                break
            L = torch.tensor([r.lu(*Lf) * float(x) for x in rads.tolist()], device=self.device)
            S = max(8, int(math.ceil(float(L.max()) / H)))
            P = smooth_paths(start, dirs, L, S, torch.full((A,), r.lu(*tort), device=self.device), float(L.mean()) / (3 if style == "dendrite" else 1), r)
            # children where they stay above the smallest radius (r^ex = r1^ex + r2^ex)
            q = torch.tensor([r.u(0.15, 0.5) for _ in range(A)], device=self.device)
            r1 = rads * (1 - q) ** (1 / ex); r2 = rads * q ** (1 / ex)
            grow = r1 >= rmin
            if nseg + 2 * int(grow.sum()) > 300:
                grow = grow & (torch.rand(A, generator=r.g, device=self.device) < 0.5)
            prof = rads[:, None].expand(A, S).clone()
            sfrac = torch.arange(S, device=self.device, dtype=torch.float32) / (S - 1)
            tip = torch.where(grow, r1, rads * (0.5 if style != "gland" else 1.0))   # taper the last quarter to the larger child (or a tip)
            ramp = ((sfrac - 0.75) / 0.25).clamp(0, 1)
            prof = prof * (1 - ramp) + tip[:, None] * ramp
            if style == "vessel":
                prof = prof * events(A, S, 0.3, (0.2, 0.6), (3, 12), r)       # stenoses
                flat = f0 * events(A, S, 0.35, (0.5, 0.95), (4, 15), r)      # collapses: the section flattens, the lumen closes
            else:
                flat = torch.ones(A, S, device=self.device)
            seg_ids = torch.arange(nseg, nseg + A, device=self.device); nseg += A
            kw = {"wall": prof * wf, "lumen_c": torch.full((A, S), lumen_c, device=self.device)} if wf > 0 else {}
            cvs = curves_like(P, prof, r, obj=br.clone() if deep else torch.full((A,), tid, device=self.device), seg=seg_ids, contrast=contrast, flat=flat, **kw)
            segs.append(cvs); sides.append(side)
            ends = P[:, -1]; tend = tangents(P)[:, -1]
            if bool((~grow).any()):
                self.ends.append(ends[~grow]); ends_all.append((ends[~grow], rads[~grow]))
            if not bool(grow.any()):
                break
            self.branch.append(ends[grow])
            r0g, r1g, r2g = rads[grow], r1[grow], r2[grow]
            g.murray.append(torch.stack([r0g, r1g, r2g], 1))                    # each branching's parent and child radii
            if ex == 3.0:                                                       # Murray's optimal angles
                cos1 = ((r0g ** 4 + r1g ** 4 - r2g ** 4) / (2 * r0g ** 2 * r1g ** 2)).clamp(-1, 1)
                cos2 = ((r0g ** 4 + r2g ** 4 - r1g ** 4) / (2 * r0g ** 2 * r2g ** 2)).clamp(-1, 1)
            else:                                                               # dendrites: wide, irregular angles
                cos1 = torch.cos(torch.deg2rad(10 + 30 * torch.rand(int(grow.sum()), generator=r.g, device=self.device)))
                cos2 = torch.cos(torch.deg2rad(30 + 50 * torch.rand(int(grow.sum()), generator=r.g, device=self.device)))
            te = tend[grow]; pe = Fn.normalize(torch.linalg.cross(te, r.unit(int(grow.sum()))), dim=1)
            d1 = Fn.normalize(cos1[:, None] * te + torch.sqrt(1 - cos1 ** 2)[:, None] * pe, dim=1)
            d2 = Fn.normalize(cos2[:, None] * te - torch.sqrt(1 - cos2 ** 2)[:, None] * pe, dim=1)
            start = torch.cat([ends[grow], ends[grow]]); dirs = torch.cat([d1, d2]); rads = torch.cat([r1g, r2g])
            if deep:                                                            # the larger child continues its branch; the smaller is a new one under it
                nb_ = self.ids(int(grow.sum()), r2g)
                for c_, p_ in zip(nb_.tolist(), br[grow].tolist()):
                    self.parents[c_] = p_
                for b_, r_ in zip(br[grow].tolist(), r1g.tolist()):                 # a continuing branch tapers: its smallest radius
                    self.obj_r[b_] = min(self.obj_r.get(b_, r_), r_)
                br = torch.cat([br[grow], nb_])
            side = torch.cat([side[grow], 1 - side[grow]])                       # the larger child keeps its parent's side
        g.curves = segs
        self.obstacles.append((torch.cat([cv.P[:, ::4].reshape(-1, 3) for cv in segs]), torch.cat([cv.rad[:, ::4].reshape(-1) for cv in segs])))
        self.record["families"].append({"tree": {"style": style, "root_vox": round(r0, 1), "segments": nseg, "wall_frac": round(wf, 2)}})
        out = []
        for sd in (0, 1):                                                       # two populations by branch side
            cs = [curves_subset(cv, sv == sd) for cv, sv in zip(segs, sides)]
            cs = [c_ for c_ in cs if c_ is not None]
            if cs:
                h = Group("tree" if style == "vessel" else style, "line", r0, contrast, curves=cs); h.murray = g.murray
                out.append(h)
        if style == "dendrite":                                                 # the soma: a filled ellipsoid at the root
            rs = r0 * r.u(3, 6); out.append(self._solids(start=None, C=c[None] - d[None] * r0 * 4, A=torch.eye(3, device=self.device)[None] * rs,
                                                         name="soma", oid=torch.tensor([tid], device=self.device)))
        if style == "gland" and ends_all:                                       # acini: hollow spheres at the blind ends
            Ce = torch.cat([e for e, _ in ends_all]); Re = torch.cat([rr_ for _, rr_ in ends_all]) * r.u(1.8, 3.0)
            out.append(self._shells(Ce, Re, wall=float(wf) * float(Re.mean()) * 0.8, name="acini", parent=tid))
        return out

    def network(self):
        r = self.r
        rc = self.size(2.5, 6.0, 0.45, 3.0)
        if rc is None:
            return None
        n = r.i(30, 160); m = torch.tensor([0.1 * s for s in self.shape], device=self.device)
        pts = m + r.rand(n, 3) * (torch.tensor(self.shape, device=self.device) - 2 * m)
        D = torch.cdist(pts, pts); D.fill_diagonal_(1e9)
        k = 3; nb = D.topk(k, largest=False).indices
        e = torch.stack([torch.arange(n, device=self.device).repeat_interleave(k), nb.reshape(-1)], 1)
        e = torch.unique(torch.sort(e, 1).values, dim=0)
        a, b = pts[e[:, 0]], pts[e[:, 1]]; L = (b - a).norm(dim=1)
        S = max(8, int(math.ceil(float(L.max()) / H)))
        s = torch.linspace(0, 1, S, device=self.device)[None, :, None]
        wob = smooth_noise1d(r, e.shape[0], S, S / 4, 3) * (0.08 * L[:, None, None]) * torch.sin(math.pi * s)
        P = a[:, None] + (b - a)[:, None] * s + wob
        rad = (rc * torch.exp(0.15 * r.randn(e.shape[0])))[:, None].expand(-1, S).clone()
        nid = int(self.ids(1, rc)[0]); contrast = self.contrast()
        kw = {"wall": rad * 0.35, "lumen_c": torch.full_like(rad, r.u(-40, 60))} if rc >= 2.0 else {}
        cv = curves_like(P, rad, r, obj=torch.full((e.shape[0],), nid, device=self.device), seg=torch.arange(e.shape[0], device=self.device), contrast=contrast, **kw)
        deg = torch.bincount(e.reshape(-1), minlength=n)
        if bool((deg >= 3).any()):
            self.branch.append(pts[deg >= 3])
        self.record["families"].append({"network": {"nodes": n, "edges": int(e.shape[0]), "radius_vox": round(rc, 2)}})
        # one population per dominant axis of the edge: edges meeting at a node at large angles are apart
        dom = ((b - a) / L[:, None].clamp_min(1e-6)).abs().argmax(1)
        out = []
        for j in range(3):
            c_ = curves_subset(cv, dom == j)
            if c_ is not None:
                out.append(Group("network", "line", rc, contrast, curves=[c_]))
        return out

    def helix(self):
        r = self.r
        rf = self.size(0.5, 30, 0.45, 4.0)
        if rf is None:
            return None
        rho = rf * r.u(1.5, 5.0); pitch = rf * r.u(4, 25); n_str = r.i(1, 3)
        c = self.centre_point(); d = r.unit(); Lc = 2.0 * max(self.shape); S = int(Lc / H)
        Pc = centred_paths(c[None], d[None], torch.tensor([Lc], device=self.device), S, torch.tensor([r.lu(0.001, 0.01)], device=self.device), 40.0, r)
        T = tangents(Pc); e1, e2 = rmf(Pc, T)
        sl = torch.arange(S, device=self.device, dtype=torch.float32) * (Lc / (S - 1))
        ph = torch.arange(n_str, device=self.device, dtype=torch.float32)[:, None] * (2 * math.pi / n_str)
        ang = 2 * math.pi * sl[None] / pitch + ph
        P = Pc + rho * (torch.cos(ang)[..., None] * e1 + torch.sin(ang)[..., None] * e2)
        rad = torch.full((n_str, S), rf, device=self.device); contrast = self.contrast()
        cv = curves_like(P, rad, r, obj=self.ids(n_str, rf), seg=torch.zeros(n_str, dtype=torch.long, device=self.device), contrast=contrast)
        self.record["families"].append({"helix": {"strands": n_str, "radius_vox": round(rf, 2), "pitch_vox": round(pitch, 1)}})
        return Group("helix", "line", rf, contrast, curves=[cv])

    def radial(self):
        """Fibres along (radial) or across (tangential) the laminar depth coordinate, confined to one band."""
        if self.depth is None:
            return None
        r = self.r; rf = self.size(0.5, 20, 0.45, 3.0)
        if rf is None:
            return None
        k = r.i(0, len(self.bands)); lo = self.bands[k - 1] if k > 0 else -1e9; hi = self.bands[k] if k < len(self.bands) else 1e9
        radial = r.p(0.6); n = r.i(20, 150)
        F = self.normal_field if radial else Fn.normalize(torch.linalg.cross(self.normal_field, r.unit().view(3, 1, 1, 1).expand_as(self.normal_field), dim=0), dim=0)
        P = (r.rand(n, 3) * torch.tensor(self.shape, device=self.device))
        S = int(1.5 * max(self.shape) / H); pts = [P]
        for _ in range(S - 1):
            d = Fn.normalize(self.sample_grid(F, P).T, dim=1)
            if len(pts) > 1:
                prev = P - pts[-2]; d = torch.where(((d * prev).sum(1, keepdim=True)) < 0, -d, d)
            P = P + H * d; pts.append(P)
        Pc = torch.stack(pts, 1)
        dep = self.sample_grid(self.depth[None], Pc)[0]
        rad = torch.full((n, S), rf, device=self.device) * ((dep > lo) & (dep < hi)).float()
        contrast = self.contrast()
        cv = curves_like(Pc, rad, r, obj=self.ids(n, rf), seg=torch.zeros(n, dtype=torch.long, device=self.device), contrast=contrast)
        self.record["families"].append({"radial": {"radial": radial, "fibres": n, "band": k}})
        return Group("radial" if radial else "tangential", "line", rf, contrast, curves=[cv])

    # ---- surfaces and implicit walls
    def sheet(self, normal=None, centre=None, half_thickness=None, height=None):
        """A folded sheet (fenestrated, bilayer, stacked by chance); ``height = (sf, amp, kx, ky, ph, e1, e2)`` puts it on
        the analytic fold of an epithelium (its basement membrane) instead of a random one."""
        r = self.r
        ht = half_thickness or self.size(0.002, 150, 0.5, min(self.shape) / 10)              # membranes (nm) to fascia (mm)
        if ht is None:
            return None
        n0 = normal if normal is not None else r.unit()
        a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        c = centre if centre is not None else self.centre_point(); Lh = 0.9 * max(self.shape); nu = int(2 * Lh / H)
        uu = (torch.arange(nu, device=self.device, dtype=torch.float32) - nu / 2) * H
        U, V = torch.meshgrid(uu, uu, indexing="ij")
        if height is not None:
            sf, amp, kx, ky, ph, e1, e2 = height
            hgt = amp * torch.sin(2 * math.pi * kx * U + ph) * torch.cos(2 * math.pi * ky * V)
        else:
            sf = r.lu(8, 40); hgt = noise2d(nu, nu, sf / H, r) * (r.u(0, 1.0) * sf)
        gu, gv = torch.gradient(hgt, spacing=H)
        keep = torch.ones_like(hgt, dtype=torch.bool)
        if height is None and r.p(0.25):                                    # fenestrated
            holes = noise2d(nu, nu, r.lu(2, 10) / H, r); keep = holes < float(torch.quantile(holes.flatten()[::97], 1 - r.u(0.05, 0.35)))
        nrm = Fn.normalize(n0[None, None] - gu[..., None] * e1 - gv[..., None] * e2, dim=-1)
        area = torch.sqrt(1 + gu ** 2 + gv ** 2)
        base = c + U[..., None] * e1 + V[..., None] * e2 + hgt[..., None] * n0
        bil = r.u(1.5, 4.0) * ht if (height is None and r.p(0.25)) else 0.0
        n_st = r.i(2, 4) if (height is None and r.p(0.2)) else 1; spacing = r.u(3, 8) * ht
        contrast = self.contrast(); g = Group("sheet", "sheet", ht, contrast)
        nt = max(2, int(math.ceil(2 * ht / H))); tt = (torch.arange(nt, device=self.device, dtype=torch.float32) + 0.5) / nt * 2 * ht - ht
        leaf = [0.0] if bil == 0 else [-0.5 * bil - ht, 0.5 * bil + ht]
        for k in range(n_st):
            oid = self.ids(1, 1e3)                                          # an extended surface: separable at any thickness
            off = (k - (n_st - 1) / 2) * spacing
            for lf in leaf:
                Q = base[keep][:, None] + (off + lf + tt)[None, :, None] * nrm[keep][:, None]
                W = (area[keep] * H * H * (2 * ht / nt))[:, None].expand(-1, nt)
                tens = sheet_tensor(nrm[keep])
                hol = bil / (bil + 2 * ht) if bil else 0.0
                desc = torch.tensor([ht, 0.0, hol, 0.0, 0.0], device=self.device).expand(tens.shape[0], 5)
                V = torch.cat([tens, desc, torch.full((tens.shape[0], 1), contrast, device=self.device)], 1)
                g.points.append((Q.reshape(-1, 3), W.reshape(-1), V[:, None].expand(-1, nt, -1).reshape(-1, 12), oid.expand(Q.shape[0] * nt)))
        self.record["families"].append({"sheet": {"half_thickness_vox": round(ht, 2), "bilayer": round(bil, 1), "stack": n_st, "fenestrated": bool((~keep).any())}})
        return g

    def foam(self, porous: bool = False):
        r = self.r
        cell = self.size(20, 1500, 6.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.5, r.u(0.06, 0.2) * cell); contrast = self.contrast(); oid = int(self.ids(1, 1e3)[0])
        dev = self.device
        if not porous:
            occ, nrm = self._voronoi(cell, hw)
        else:
            phi_ = smooth_noise(self.shape, cell / 3, r.g, dev)
            gz, gy, gx = torch.gradient(phi_); gr = torch.stack([gz, gy, gx]); gm = gr.norm(dim=0).clamp_min(1e-6)
            occ = (0.5 + hw - phi_.abs() / gm).clamp(0, 1); nrm = gr / gm
        if r.p(0.5):                                                        # confined to a region
            reg = gaussian(r.rand(*self.shape), r.lu(10, 30)); occ = occ * (reg > float(torch.quantile(reg.flatten()[::101], r.u(0.2, 0.6)))).float()
        tens = sheet_tensor(nrm.movedim(0, -1)).movedim(-1, 0)
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = hw; desc[2] = max(0.0, 1 - 3 * hw / cell)
        g = Group("porous" if porous else "foam", "sheet", hw, contrast, dense=[(occ, tens, desc, contrast, torch.full(self.shape, oid, device=dev, dtype=torch.long))])
        self.record["families"].append({"porous" if porous else "foam": {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2)}})
        return g

    def _voronoi(self, cell: float, hw: float, metric: torch.Tensor | None = None, warp: float = 0.0, warp_sigma: float | None = None):
        """Voronoi walls of half-width ``hw`` voxels around seeds ``cell`` voxels apart: occupancy ``(Z, Y, X)`` and unit
        wall normals ``(3, Z, Y, X)``. ``metric`` (3 x 3, :func:`_stretch_metric`): distances measured after this map
        (cells elongated or flattened along an axis); ``warp``: query points displaced by a smooth field of this amplitude
        (voxels, scale ``warp_sigma``) -- wavy, interlocking walls. The normals are the bisector's gradient through both
        (the metric's transpose and the warp's Jacobian)."""
        r = self.r; dev = self.device; Z, Y, X = self.shape
        occ = torch.zeros(self.shape, device=dev); nrm = torch.zeros((3,) + self.shape, device=dev)
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        n = max(4, int(Z * Y * X / cell ** 3)); seeds = r.rand(n, 3) * torch.tensor(self.shape, device=dev)
        Wf = torch.stack([smooth_noise(self.shape, warp_sigma or cell / 3, r.g, dev) for _ in range(3)], -1) * warp if warp > 0 else None
        Mt = metric.T if metric is not None else None
        seeds_t = seeds @ Mt if Mt is not None else seeds
        bs = 32; m = 2.5 * cell + 2                                                      # blocks against the seeds near them only (memory ~ block x local seeds)
        for z0 in range(0, Z, bs):
            for y0 in range(0, Y, bs):
                for x0 in range(0, X, bs):
                    blk = (slice(z0, z0 + bs), slice(y0, y0 + bs), slice(x0, x0 + bs))
                    q = g_[blk]; bsh = q.shape[:3]; q = q.reshape(-1, 3)
                    if Wf is not None:
                        q = q + Wf[blk].reshape(-1, 3)
                    if Mt is not None:
                        q = q @ Mt
                    lo = q.min(0).values - m; hi = q.max(0).values + m
                    near = ((seeds_t >= lo) & (seeds_t <= hi)).all(1)
                    sd = seeds_t[near] if int(near.sum()) >= 2 else seeds_t
                    d2, i2 = torch.cdist(q, sd).topk(2, dim=1, largest=False)
                    s1, s2 = sd[i2[:, 0]], sd[i2[:, 1]]; sep = (s2 - s1).norm(dim=1).clamp_min(1e-6)
                    dist = (d2[:, 1] ** 2 - d2[:, 0] ** 2) / (2 * sep)
                    occ[blk] = (0.5 + hw - dist).clamp(0, 1).view(bsh)
                    nrm[(slice(None),) + blk] = ((s2 - s1) / sep[:, None]).T.reshape(3, *bsh)
        if metric is not None or Wf is not None:                                          # the wall's normal in the canvas: grad of u . M (q + W(q))
            v = torch.einsum("ij,i...->j...", metric, nrm) if metric is not None else nrm.clone()
            if Wf is not None:
                gv = v.clone()
                for i in range(3):
                    gi = torch.gradient(Wf[..., i])
                    for j in range(3):
                        gv[j] += v[i] * gi[j]
                v = gv
            nrm = v / v.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return occ, nrm

    def _wall_group(self, name: str, occ: torch.Tensor, nrm: torch.Tensor, hw: float, cell: float, record: dict):
        """A sheet population from wall occupancy and normals (one object: an extended surface)."""
        dev = self.device; contrast = self.contrast(); oid = int(self.ids(1, 1e3)[0])
        tens = sheet_tensor(nrm.movedim(0, -1)).movedim(-1, 0)
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = hw; desc[2] = max(0.0, 1 - 3 * hw / cell)
        self.record["families"].append({name: record})
        return Group(name, "sheet", hw, contrast, dense=[(occ, tens, desc, contrast, torch.full(self.shape, oid, device=dev, dtype=torch.long))])

    def _discs(self, C: torch.Tensor, Nrm: torch.Tensor, R: torch.Tensor, ht: float, name: str, oid: torch.Tensor, contrast=None):
        """Flat discs -- centres ``C (n, 3)``, unit normals ``Nrm (n, 3)``, radii ``R (n,)`` voxels, half thickness ``ht``:
        sheet tensors along the normal (Z discs across myofibrils, cristae across mitochondria)."""
        dev = self.device; n = C.shape[0]; contrast = self.contrast() if contrast is None else contrast
        ns = max(4, int(math.ceil(2 * float(R.max()) / H))); U, du = _template2d(ns, dev); U = U[U.norm(dim=1) <= 1]; m = U.shape[0]
        a = torch.where(Nrm[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=dev), torch.tensor([0, 1.0, 0], device=dev))
        e1 = Fn.normalize(torch.linalg.cross(Nrm, a), dim=1); e2 = torch.linalg.cross(Nrm, e1)
        ht = max(0.25, float(ht)); nt = max(1, int(math.ceil(2 * ht / H))); tt = (torch.arange(nt, device=dev, dtype=torch.float32) + 0.5) / nt * 2 * ht - ht
        Q = (C[:, None, None] + R[:, None, None, None] * (U[None, :, None, :1] * e1[:, None, None] + U[None, :, None, 1:] * e2[:, None, None])
             + tt[None, None, :, None] * Nrm[:, None, None])                                            # (n, m, nt, 3)
        W = (R ** 2 * du * du * (2 * ht / nt))[:, None, None].expand(n, m, nt)
        V = torch.cat([sheet_tensor(Nrm), torch.stack([torch.full((n,), ht, device=dev)] + [torch.zeros(n, device=dev)] * 4, 1),
                       torch.full((n, 1), contrast, device=dev)], 1)
        g = Group(name, "sheet", ht, contrast)
        g.points.append((Q.reshape(-1, 3), W.reshape(-1), V[:, None, None].expand(n, m, nt, 12).reshape(-1, 12), oid[:, None, None].expand(n, m, nt).reshape(-1)))
        return g

    def _place(self, n_max: int, dmin: float) -> torch.Tensor:
        """Up to ``n_max`` random centres on the canvas, none closer than ``dmin`` voxels to an earlier one."""
        dev = self.device; n_c = int(min(4000, 2.5 * np.prod(self.shape) / max(dmin, 1.0) ** 3)) + 1
        C = self.r.rand(n_c, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        D = torch.cdist(C, C); earlier = torch.ones_like(D, dtype=torch.bool).tril(-1)
        return C[~((D < dmin) & earlier).any(1)][:n_max]

    def _wide_radius(self, rf: float, nmax: int = 1500) -> float:
        """A bundle radius (voxels) spanning a good part of the view, at most ``nmax`` fibres of radius ``rf`` across it
        (such bundles are drawn nearly straight: a canvas-wide array that curls leaves the view)."""
        return float(min(self.r.u(0.25, 0.6) * min(self.shape), max(2.5 * rf, 1.05 * rf * math.sqrt(nmax) * 1.2)))

    # ---- electron-microscopy scales: compartments, organelles, myelin, nuclei, myofibrils
    def compartments(self):
        """Cytoplasm and neuropil as electron microscopy sees them: space-filling compartments (neurites, processes, cell
        bodies 0.1-3 um across) bounded by thin membranes -- Voronoi walls about a voxel thick, the compartments often
        stretched along an axis (neurites cut across or along) and wavy."""
        r = self.r
        cell = self.size(0.1, 3.0, 5.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.5, min(1.5, r.u(0.02, 0.08) * cell)); st = r.lu(0.25, 1.0) if r.p(0.6) else 1.0
        M = _stretch_metric(self.r.unit(), st) if st < 0.97 else None; warp = r.u(0.05, 0.25) * cell if r.p(0.5) else 0.0
        occ, nrm = self._voronoi(cell, hw, metric=M, warp=warp)
        return self._wall_group("compartments", occ, nrm, hw, cell, {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2), "stretch": round(1 / st, 2), "warp": round(warp, 1)})

    def organelles(self):
        """A cell's interior at electron-microscopy scales, two to four of: mitochondria (capsules 0.1-0.5 um across and
        1.5-8 times as long, a double membrane, cristae across them), vesicles in clusters (hollow spheres 15-60 nm),
        droplets and granules (solid spheres), stacked ER cisternae (sheets 5-30 nm half thick). Only what resolves."""
        r = self.r; dev = self.device; out = []
        kinds = [k for k in ("mito", "vesicles", "droplets", "er") if r.p(0.6)] or ["mito"]
        if "mito" in kinds:
            rm = self.size(0.1, 0.5, 1.5, min(self.shape) / 10)
            if rm is not None:
                L = r.lu(3, 16) * rm; C = self._place(r.i(3, 60), 2.2 * rm + 0.6 * L)
                n = C.shape[0]
                if n:
                    S = max(8, int(L / H)); d = r.unit(n)
                    P = smooth_paths(C - 0.5 * L * d, d, torch.full((n,), L, device=dev), S, torch.full((n,), r.lu(0.02, 0.2), device=dev), max(L / 3, 1.0), r)
                    sl = torch.arange(S, device=dev, dtype=torch.float32) / (S - 1) * L; de = torch.minimum(sl, L - sl)
                    prof = torch.where(de < rm, torch.sqrt((rm ** 2 - (rm - de) ** 2).clamp_min(0)), torch.full_like(de, rm))   # hemispherical ends
                    rad = (prof[None] * torch.exp(0.15 * r.randn(n, 1))).clamp_min(0.05)
                    T_ = max(0.5, r.u(0.06, 0.12) * rm); oid = self.ids(n, rad.amax(1))
                    cv = curves_like(P, rad, r, obj=oid, seg=torch.zeros(n, dtype=torch.long, device=dev), contrast=self.contrast(),
                                     wall=torch.full_like(rad, T_), lam=torch.full((n,), 2, device=dev, dtype=torch.long))
                    out.append(Group("mitochondria", "line", rm, float(cv.contrast[0, 0]), curves=[cv]))
                    sp = r.u(0.6, 1.5) * rm; k = torch.arange(int(L / sp) + 1, device=dev, dtype=torch.float32) * sp + 0.5 * sp   # cristae every sp along each
                    k = k[k < L - 0.5 * sp]
                    if k.numel():
                        ix = (k / L * (S - 1)).round().long().clamp(0, S - 1)
                        Cc = P[:, ix].reshape(-1, 3); Tc = tangents(P)[:, ix].reshape(-1, 3); Rc = (0.8 * rad[:, ix]).reshape(-1).clamp_min(0.3)
                        out.append(self._discs(Cc, Tc, Rc, max(0.25, 0.5 * T_), "cristae", oid[:, None].expand(n, ix.numel()).reshape(-1)))
                    self.obstacles.append((C, torch.full((n,), 0.5 * L, device=dev)))
        if "vesicles" in kinds:
            rv = self.size(0.015, 0.06, 0.6, 6.0)
            if rv is not None:
                k = r.i(1, 6); cc = r.rand(k, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
                m = r.i(10, 80); C = (cc[:, None] + r.u(2.5, 6.0) * rv * r.randn(k, m, 3)).reshape(-1, 3)
                R = rv * torch.exp(0.2 * r.randn(C.shape[0])).clamp(0.6, 1.6)
                out.append(self._shells(C, R, wall=max(0.5, 0.25 * rv), name="vesicles"))
        if "droplets" in kinds:
            rd = self.size(0.02, 1.0, 0.8, min(self.shape) / 10)
            if rd is not None:
                C = self._place(r.i(3, 200), 2.2 * rd); n = C.shape[0]
                if n:
                    A = torch.eye(3, device=dev)[None] * (rd * torch.exp(0.25 * r.randn(n))).clamp_min(0.5)[:, None, None]
                    out.append(self._solids(C, A, "droplets", self.ids(n, rd)))
        if "er" in kinds:
            ht = self.size(0.005, 0.03, 0.5, 3.0)
            if ht is not None:
                g = self.sheet(half_thickness=ht)
                if g is not None:
                    g.name = "er"; out.append(g)
        self.record["families"].append({"organelles": {"kinds": kinds, "groups": len(out)}})
        return out or None

    def myelin(self):
        """Myelinated axons, electron to light microscopy: a bundle of axons 0.2-3 um in radius, each in a dark sheath of
        g-ratio 0.6-0.8 -- concentric lamellae where the voxel resolves them, one ring where not; the axoplasm a lumen."""
        r = self.r; rf_um = r.lu(0.2, 3.0)
        if rf_um / self.voxel_um < 1.5:
            return None
        rf = min(rf_um / self.voxel_um, min(self.shape) / 6)
        g = self.bundle(None, "single", curv=r.lu(0.0003, 0.003), radius=self._wide_radius(rf, 800), fibre={"rf_um": rf_um, "jitter": 0.3, "disp": 0.4, "max_fibres": 800})
        if g is None or isinstance(g, list) or not g.curves or g.curves[0].N < 1 or g.offsets is None:
            return None
        for cv in g.curves:
            sheath = cv.rad * (1 - r.u(0.6, 0.8)); smean = float(sheath.mean())
            L_ = r.i(1, max(1, min(12, int((smean / 0.6 + 1) // 2))))
            cv.wall = sheath / (2 * L_ - 1) if L_ > 1 else sheath
            cv.lam = torch.full((cv.N,), L_, device=self.device, dtype=torch.long)
            cv.lumen_c = torch.full_like(cv.rad, r.u(-40, 40))
        g.name = "myelin"; self.record["families"].append({"myelin": {"axon_um": round(rf_um, 2), "lamellae": int(L_)}})
        return g

    def nuclei(self):
        """Cell nuclei, electron to light microscopy: one or two large nuclei (radius 2-8 um) partly in view at fine
        scales, a field of them at coarse ones -- a double envelope, heterochromatin clumped against it, a nucleolus."""
        r = self.r; dev = self.device
        R = self.size(2, 8, 4.0, 0.45 * min(self.shape))
        if R is None:
            return None
        if R > min(self.shape) / 5:                                                             # one or two, their envelopes across the view
            C = self.centre_point(0.0) + (r.rand(r.i(1, 2), 3) - 0.5) * R
        else:
            C = self._place(300, 2.1 * R * 1.1)
        n = C.shape[0]
        if n == 0:
            return None
        Rn = R * torch.exp(0.1 * r.randn(n)).clamp(0.85, 1.1); wall = min(3.0, max(0.5, r.u(0.01, 0.03) * R))
        cid = self.ids(n, Rn)
        env = self._shells(C, Rn, wall=wall, name="envelope"); out = [env]
        rc = max(0.6, r.u(0.03, 0.08) * R); m = int(min(1500, r.u(20, 200) * n))
        k = torch.randint(0, n, (m,), generator=r.g, device=dev); d = Fn.normalize(r.randn(m, 3), dim=1)
        Cc = C[k] + d * (Rn[k] * r.u(0.86, 0.95) - rc)[:, None]
        A = torch.eye(3, device=dev)[None] * (rc * torch.exp(0.3 * r.randn(m))).clamp_min(0.5)[:, None, None]
        oc = self.ids(m, rc)
        for a_, b_ in zip(oc.tolist(), cid[k].tolist()):
            self.parents[a_] = b_
        out.append(self._solids(Cc, A, "chromatin", oc))
        rnu = max(0.8, r.u(0.12, 0.3) * R)
        Cn = C + (r.rand(n, 3) - 0.5) * 0.6 * (Rn[:, None] - rnu)
        out.append(self._solids(Cn, torch.eye(3, device=dev)[None].expand(n, 3, 3) * rnu, "nucleolus", self.ids(n, rnu)))
        self.obstacles.append((C, Rn))
        self.record["families"].append({"nuclei": {"n": n, "radius_vox": round(R, 1)}})
        return out

    def myofibrils(self):
        """Striated muscle: myofibrils (radius 0.3-1 um) packed in parallel and cross-banded in register -- a Z disc every
        1.8-2.6 um (a sarcomere: a sheet across the fibres, so the banding is an orientation of its own, under any
        physics) and the A band between (contrast)."""
        r = self.r; rf_um = r.lu(0.3, 1.0); per = r.u(1.8, 2.6) / self.voxel_um
        if per < 3.0 or rf_um / self.voxel_um < 0.45:
            return None
        rf = min(rf_um / self.voxel_um, min(self.shape) / 6)
        g = self.bundle(None, "single", curv=r.lu(0.0003, 0.003), radius=self._wide_radius(rf, 800),
                        fibre={"rf_um": rf_um, "jitter": 0.05, "disp": 0.08, "bands": (per, r.u(0.3, 0.9), r.u(0.04, 0.12)), "max_fibres": 800})
        if g is None or isinstance(g, list) or g.offsets is None:
            return None
        cv = g.curves[0]; N, S = cv.P.shape[:2]; Lc = 2.0 * max(self.shape); ds = Lc / max(S - 1, 1)
        z0 = r.u(0, per); out = [g]
        if z0 >= Lc:                                                                            # a sarcomere longer than the fibres in view: no disc
            g.name = "myofibrils"; self.record["families"].append({"myofibrils": {"fibril_um": round(rf_um, 2), "sarcomere_vox": round(per, 1)}})
            return out
        zs = torch.arange(z0, Lc, per, device=self.device); ix = (zs / ds).round().long().clamp(0, S - 1)
        Cz = cv.P[:, ix].reshape(-1, 3); Tz = tangents(cv.P)[:, ix].reshape(-1, 3); Rz = cv.rad[:, ix].reshape(-1)
        inside = ((Cz > -Rz[:, None]) & (Cz < torch.tensor(self.shape, device=self.device) + Rz[:, None])).all(1)
        if bool(inside.any()):
            zd = self._discs(Cz[inside], Tz[inside], Rz[inside] * 1.05, max(0.3, 0.06 / self.voxel_um), "z_discs",
                             cv.obj[:, None].expand(N, ix.numel()).reshape(-1)[inside])
            out.append(zd)
        g.name = "myofibrils"; self.record["families"].append({"myofibrils": {"fibril_um": round(rf_um, 2), "sarcomere_vox": round(per, 1)}})
        return out

    # ---- air spaces, plant cells, lattices
    def alveoli(self):
        """Thin-walled air spaces: lung alveoli and parabronchi, leaf mesophyll air space -- Voronoi walls 1.5-5 % of
        the cell (30-400 um) thick, the cells empty, often opened by a duct running through."""
        r = self.r
        cell = self.size(30, 400, 8.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.5, r.u(0.015, 0.05) * cell)
        occ, nrm = self._voronoi(cell, hw, warp=r.u(0.05, 0.15) * cell if r.p(0.5) else 0.0)
        duct = r.p(0.5)
        if duct:                                                                                # an alveolar duct: walls opened along a line
            g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=self.device, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
            c = self.centre_point(0.3); d = r.unit(); rel = g_ - c; dist = (rel - (rel @ d)[..., None] * d).norm(dim=-1)
            occ = occ * (dist > r.u(0.8, 2.0) * cell).float()
        return self._wall_group("alveoli", occ, nrm, hw, cell, {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2), "duct": duct})

    def jigsaw(self):
        """Plant epidermal pavement cells: interlocking lobed walls (Voronoi walls under a strong short-range warp) in a
        layer about one cell thick, the cells flattened against it."""
        r = self.r
        cell = self.size(15, 120, 8.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.5, r.u(0.04, 0.1) * cell); n0 = r.unit()
        occ, nrm = self._voronoi(cell, hw, metric=_stretch_metric(n0, r.u(1.5, 3.0)), warp=r.u(0.25, 0.45) * cell, warp_sigma=cell * r.u(0.15, 0.3))
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=self.device, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        depth = (g_ - self.centre_point(0.3)) @ n0; half = r.u(0.4, 1.2) * cell
        occ = occ * torch.sigmoid((half - depth.abs()) / 0.7)                                   # the epidermis: one layer
        return self._wall_group("jigsaw", occ, nrm, hw, cell, {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2), "layer_vox": round(2 * half, 1)})

    def honeycomb(self):
        """Walled plant cells -- meristem, parenchyma, xylem fibres and tracheids: Voronoi walls 4-15 % of the cell
        (5-100 um), the cells round or elongated up to 8 times along an axis."""
        r = self.r
        cell = self.size(5, 100, 5.0, min(self.shape) / 3)
        if cell is None:
            return None
        hw = max(0.6, r.u(0.04, 0.15) * cell); el = r.lu(1.0, 8.0) if r.p(0.5) else 1.0
        occ, nrm = self._voronoi(cell, hw, metric=_stretch_metric(r.unit(), 1.0 / el) if el > 1.05 else None)
        return self._wall_group("honeycomb", occ, nrm, hw, cell, {"cell_vox": round(cell, 1), "wall_vox": round(hw, 2), "elongation": round(el, 2)})

    def xylem(self):
        """Xylem vessels: wide hollow tubes (radius 8-100 um) side by side, thick walls with annular or spiral
        thickenings -- a helical strand on the inside of the wall of some -- and contents of either contrast."""
        r = self.r; dev = self.device
        rf_um = r.lu(8, 100); rf = min(rf_um / self.voxel_um, min(self.shape) / 6)
        if rf < 1.5:
            return None
        g = self.bundle(None, "single", curv=r.lu(0.0003, 0.003), radius=self._wide_radius(rf, 600), fibre={"rf_um": rf_um, "jitter": 0.3, "disp": 0.3, "wall": (0.08, 0.25), "max_fibres": 600})
        if g is None or isinstance(g, list) or g.offsets is None:
            return None
        out = [g]; cv = g.curves[0]
        if r.p(0.6) and float(cv.rad.mean()) >= 2.5:                                            # spiral thickenings
            k = torch.randperm(cv.N, generator=r.g, device=dev)[:min(cv.N, 30)]
            P = cv.P[k]; T = tangents(P); e1, e2 = rmf(P, T); N, S = P.shape[:2]
            rho = cv.rad[k] - cv.wall[k] * 0.5; rf = (cv.wall[k] * r.u(0.4, 0.8)).clamp_min(0.4)
            sl = torch.arange(S, device=dev, dtype=torch.float32) * (2.0 * max(self.shape) / max(S - 1, 1))
            pitch = float(cv.rad.mean()) * r.u(0.3, 1.5); ang = 2 * math.pi * sl[None] / pitch + r.rand(N, 1) * 6.28
            Ph = P + rho[..., None] * (torch.cos(ang)[..., None] * e1 + torch.sin(ang)[..., None] * e2)
            hv = curves_like(Ph, rf, r, obj=cv.obj[k], seg=torch.zeros(N, dtype=torch.long, device=dev), contrast=self.contrast())
            out.append(Group("thickenings", "line", float(rf.mean()), float(hv.contrast[0, 0]), curves=[hv]))
        g.name = "xylem"; self.record["families"].append({"xylem": {"vessels": int(cv.N), "thickenings": len(out) > 1}})
        return out

    def lattice(self):
        """Crystalline arrays of parallel tubes or rods: compound-eye ommatidia, stereocilia, honeycomb tubes, the
        myofilament lattice -- hexagonal packing without jitter, often fanning (radiating) and hollow."""
        r = self.r
        rf = self.size(0.005, 30, 0.8, 6.0)
        if rf is None:
            return None
        fib = {"rf_um": rf * self.voxel_um, "jitter": 0.0, "disp": 0.02, "max_fibres": 1500}
        if r.p(0.5):
            fib["wall"] = (0.15, 0.4)
        g = self.bundle(None, "fanning" if r.p(0.4) else "single", curv=r.lu(0.0003, 0.003), radius=self._wide_radius(rf, 1500), fibre=fib)
        if g is None or isinstance(g, list):
            return None
        g.name = "lattice"; self.record["families"].append({"lattice": {"rod_vox": round(rf, 2), "hollow": "wall" in fib}})
        return g

    # ---- organ scale, and isotropic content
    def _dense_solid(self, C: torch.Tensor, A: torch.Tensor, exps: tuple, name: str, contrast=None):
        """A large solid on the grid (a superquadric ``x = C + A g``, ``exps``), one voxel's soft edge: a blob population
        (its tensor the solid's second moment, constant over it) without sampling its interior point by point."""
        dev = self.device; contrast = self.contrast() if contrast is None else contrast
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        u = (g_ - C) @ torch.linalg.inv(A).T; e1, e2 = exps
        f = ((u[..., 1].abs() ** (2 / e2) + u[..., 2].abs() ** (2 / e2)) ** (e2 / e1) + u[..., 0].abs() ** (2 / e1)).clamp_min(1e-9)
        sv = torch.linalg.svdvals(A); amin = float(sv.min())
        rho = f ** (e1 / 2)                                                                     # ~ the radial coordinate (1 on the surface)
        occ = (0.5 + (1 - rho) * amin).clamp(0, 1)
        tens = (eye6(dev) / 3).view(6, 1, 1, 1).expand(6, *self.shape)                         # solid material: isotropic (29 Oct)
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = float(sv.mean()); desc[3] = float(sv[2] / sv[0])
        oid = int(self.ids(1, float(sv.min()))[0])
        return Group(name, "blob", float(sv.mean()), contrast, dense=[(occ, tens, desc, contrast, torch.full(self.shape, oid, device=dev, dtype=torch.long))])

    def organs(self):
        """Organ-scale anatomy (clinical CT, MRI): three to seven large smooth compartments (deformed superquadrics 3-150 mm),
        a hollow organ or large vessel (a wide tube: wall and contents) and a bony shell -- piecewise-smooth regions with
        sharp edges, the scale of SynthSeg-style label maps."""
        r = self.r; dev = self.device
        Rv = self.size(3000, 150000, 0.12 * min(self.shape), 0.45 * min(self.shape))
        if Rv is None:
            return None
        out = []
        for _ in range(r.i(3, 7)):
            ax = torch.tensor([Rv * r.lu(0.4, 1.0) for _ in range(3)], device=dev)
            A = _rot(r, 1)[0] @ torch.diag(ax); C = r.rand(3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
            out.append(self._dense_solid(C, A, (r.lu(0.5, 2.0), r.lu(0.5, 2.0)), "organ"))
        rt = self.size(1000, 15000, 2.0, min(self.shape) / 6)
        if rt is not None and r.p(0.7):                                                         # bowel, airway, aorta
            c = self.centre_point(); d = r.unit(); Lc = 2.0 * max(self.shape); S = int(Lc / H)
            P = centred_paths(c[None], d[None], torch.tensor([Lc], device=dev), S, torch.tensor([r.lu(0.002, 0.02)], device=dev), r.lu(20, 60), r)
            rad = torch.full((1, S), rt, device=dev)
            cv = curves_like(P, rad, r, obj=self.ids(1, rt), seg=torch.zeros(1, dtype=torch.long, device=dev), contrast=self.contrast(),
                             wall=rad * r.u(0.08, 0.3), lumen_c=torch.full_like(rad, r.u(-80, 80)))
            out.append(Group("hollow_organ", "line", rt, float(cv.contrast[0, 0]), curves=[cv]))
        if r.p(0.5):                                                                            # bone: a cortical shell
            Rb = Rv * r.u(0.3, 0.8)
            out.append(self._shells(self.centre_point(0.2)[None], torch.tensor([Rb], device=dev), wall=max(1.0, min(8.0, r.u(0.05, 0.15) * Rb)), name="bone"))
        self.record["families"].append({"organs": {"radius_vox": round(Rv, 1), "groups": len(out)}})
        return out

    def granules(self):
        """Isotropic granular content -- secretory granules, starch grains, platelets, a dense field of small nuclei:
        solid spheres 0.5-4 voxels in radius, packed at random. Spheres carry no direction: where they are all there is,
        the right answer is an isotropic density."""
        r = self.r; dev = self.device
        rg = self.size(0.01, 20, 0.5, 4.0)
        if rg is None:
            return None
        sp = 2.1 * rg * r.u(1.0, 1.8); dev_ = torch.tensor(self.shape, device=dev, dtype=torch.float32)  # a jittered lattice: dense, never overlapping
        ax = [torch.arange(0.5 * sp, n_ - 0.25 * sp, sp, device=dev) for n_ in self.shape]
        C = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        C = C + (r.rand(*C.shape) - 0.5) * (sp - 2.1 * rg)                                       # jitter within the gap
        C = C[r.rand(C.shape[0]) < r.u(0.3, 1.0)]
        C = C[torch.randperm(C.shape[0], generator=r.g, device=dev)[:150000]]; n = C.shape[0]
        if n == 0:
            return None
        A = torch.eye(3, device=dev)[None] * (rg * torch.exp(0.25 * r.randn(n))).clamp(0.5, 1.2 * rg)[:, None, None]
        self.record["families"].append({"granules": {"n": n, "radius_vox": round(rg, 2)}})
        return self._solids(C, A, "granules", self.ids(n, rg))

    def cells(self):
        r = self.r; dev = self.device
        rc = self.size(3, 40, 1.5, 14.0)
        if rc is None:
            return None
        elong = r.lu(1.0, 2.5); dmin = 2.1 * rc * max(1.0, elong) * 1.12          # bounding spheres (with the size jitter) never overlap
        n_c = int(min(4000, 2.5 * np.prod(self.shape) / dmin ** 3))
        C = r.rand(n_c, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        D = torch.cdist(C, C); earlier = torch.ones_like(D, dtype=torch.bool).tril(-1)
        C = C[~((D < dmin) & earlier).any(1)]                                # random placement, a candidate dropped if it hits an earlier one
        keep = r.rand(C.shape[0]) < r.u(0.3, 1.0)
        if self.depth is not None:                                          # laminar density
            bd = torch.tensor(self.band_density, device=dev)[self.band_of(C)]
            keep = keep & (r.rand(C.shape[0]) < bd)
        C = C[keep]; C = C[torch.randperm(C.shape[0], generator=r.g, device=dev)[:700]]; n = C.shape[0]   # shuffled before the cap
        if n == 0:
            return None
        Rm = _rot(r, n) if r.p(0.5) else _rot(r, 1).expand(n, 3, 3)
        ax = torch.stack([torch.ones(n, device=dev), torch.ones(n, device=dev), torch.full((n,), elong, device=dev)], 1) * rc * torch.exp(0.12 * r.randn(n, 1)).clamp(max=1.12)
        out = self._cell_groups(C, Rm, ax, rc)
        self.record["families"].append({"cells": {"n": n, "radius_vox": round(rc, 2), "elongation": round(elong, 2)}})
        return out

    def _cell_groups(self, C, Rm, ax, rc):
        """Cells at centres ``C (n, 3)``, orientations ``Rm (n, 3, 3)``, semi-axes ``ax (n, 3)``: membranes (thin shells,
        sheet tensors) and nuclei (filled ellipsoids, their tensor the normalised second moment), nucleus -> cell."""
        r = self.r; dev = self.device; n = C.shape[0]
        A = Rm @ torch.diag_embed(ax)                                       # (n, 3, 3): unit sphere -> cell surface
        tm = max(0.5, r.u(0.05, 0.15) * rc); nuc = r.u(0.3, 0.6)
        cmem, cnuc = self.contrast(), self.contrast()
        cid = self.ids(n, ax.min(1).values); nid = self.ids(n, ax.min(1).values * nuc)
        for a_, b_ in zip(cid.tolist(), nid.tolist()):
            self.parents[b_] = a_
        nd = max(32, int(4 * math.pi * float(ax.max()) ** 2 / H ** 2)); Dd = _fib_sphere(nd, dev)            # (nd, 3)
        nt = max(1, int(math.ceil(tm / H)))
        Ainv_T = torch.linalg.inv(A).transpose(1, 2)
        gm = Fn.normalize(torch.einsum("nij,dj->ndi", Ainv_T, Dd), dim=-1)                                   # outward normals
        dA = torch.linalg.det(A).abs()[:, None] * torch.einsum("nij,dj->ndi", Ainv_T, Dd).norm(dim=-1) * (4 * math.pi / nd)
        gmem = Group("membranes", "sheet", tm / 2, cmem); gnuc = Group("nuclei", "blob", nuc * rc, cnuc)
        for k in range(nt):
            rho = 1 - (k + 0.5) * tm / nt / rc
            Q = C[:, None] + rho * torch.einsum("nij,dj->ndi", A, Dd)
            W = dA * (tm / nt)
            V = torch.cat([sheet_tensor(gm), torch.stack([torch.full_like(W, tm / 2), torch.zeros_like(W), torch.full_like(W, (1 - tm / rc) ** 2), torch.zeros_like(W), torch.zeros_like(W)], -1),
                           torch.full_like(W, cmem)[..., None]], -1)
            gmem.points.append((Q.reshape(-1, 3), W.reshape(-1), V.reshape(-1, 12), cid[:, None].expand(-1, nd).reshape(-1)))
        off = (r.rand(n, 3) - 0.5) * 0.4 * rc * (1 - nuc)
        gn = self._solids(C=C + off, A=A * nuc, name="nuclei", oid=nid, contrast=cnuc)
        gnuc.points = gn.points
        self.obstacles.append((C, ax.max(1).values))
        return [gmem, gnuc]

    def _solids(self, C, A, name, oid, contrast=None, start=None, shape: str = "ellipsoid", exps=(2.0, 2.0), normals=None):
        """Filled solids ``x = C + A g`` for ``g`` in a unit shape (``ellipsoid``; ``superquadric`` with exponents
        ``exps``; ``polyhedron``: inside every plane ``n_i . g <= 1`` of ``normals``), their tensor the normalised second
        moment of the solid -- a blob's shape, no category. ``C (n, 3)``, ``A (n, 3, 3)``, ``oid (n,)``."""
        dev = self.device; n = C.shape[0]
        contrast = self.contrast() if contrast is None else contrast
        amax = float(torch.linalg.matrix_norm(A, ord=2).max())
        ns = max(4, int(math.ceil(2 * amax / H)))
        U3 = (torch.arange(ns, device=dev, dtype=torch.float32) + 0.5) / ns * 2 - 1
        G3 = torch.stack(torch.meshgrid(U3, U3, U3, indexing="ij"), -1).reshape(-1, 3)
        if shape == "superquadric":
            e1, e2 = exps
            inside = ((G3[:, 1].abs() ** (2 / e2) + G3[:, 2].abs() ** (2 / e2)) ** (e2 / e1) + G3[:, 0].abs() ** (2 / e1)) <= 1
        elif shape == "polyhedron":
            inside = ((G3 @ normals.T) <= 1).all(1)
        else:
            inside = G3.norm(dim=1) <= 1
        G3 = G3[inside]
        if G3.shape[0] == 0:
            G3 = torch.zeros(1, 3, device=dev)
        Q = C[:, None] + torch.einsum("nij,mj->nmi", A, G3)
        W = (torch.linalg.det(A).abs() * (2 / ns) ** 3)[:, None].expand(-1, G3.shape[0])
        Mg = (G3[:, :, None] * G3[:, None, :]).mean(0)                                                     # the unit shape's second moment
        M2 = torch.eye(3, device=dev).expand(n, 3, 3) / 3                                                   # solid material: isotropic (29 Oct;
                                                                                                            # was the solid's shape moment)
        sv = torch.linalg.svdvals(A)                                                                        # (n, 3) descending
        Vn = torch.cat([_sym6(M2), torch.stack([sv.mean(1), torch.zeros(n, device=dev), torch.zeros(n, device=dev), sv[:, 2] / sv[:, 0].clamp_min(1e-9),
                                                torch.full((n,), float(min(1.0, max(0.0, (exps[0] - 2) / (exps[0] + 2)))) if shape != "polyhedron" else 0.6, device=dev)], -1),
                        torch.full((n, 1), contrast, device=dev)], -1)
        g = Group(name, "blob", float(sv.mean()), contrast)
        g.points.append((Q.reshape(-1, 3), W.reshape(-1), Vn[:, None].expand(-1, G3.shape[0], -1).reshape(-1, 12), oid[:, None].expand(-1, G3.shape[0]).reshape(-1)))
        return g

    def _shells(self, C, R, wall: float, name: str, parent: int | None = None, contrast=None):
        """Hollow spheres (centres ``C (n, 3)``, outer radii ``R (n,)``, wall thickness in voxels): sheet tensors along
        the wall, descriptors (half wall, lumen fraction)."""
        dev = self.device; n = C.shape[0]; contrast = self.contrast() if contrast is None else contrast
        wall = max(0.5, float(wall)); oid = self.ids(n, R)
        if parent is not None:
            for o_ in oid.tolist():
                self.parents[o_] = parent
        nd = max(32, int(4 * math.pi * float(R.max()) ** 2 / H ** 2)); Dd = _fib_sphere(nd, dev)
        nt = max(1, int(math.ceil(wall / H))); g = Group(name, "sheet", wall / 2, contrast)
        tens = sheet_tensor(Dd)
        for k in range(nt):
            rho = R - (k + 0.5) * wall / nt                                                             # (n,)
            Q = C[:, None] + rho[:, None, None] * Dd[None]
            W = (4 * math.pi * rho ** 2 / nd * (wall / nt))[:, None].expand(-1, nd)
            lum = ((R - wall).clamp_min(0) / R) ** 3
            desc = torch.stack([torch.full((n, nd), wall / 2, device=dev), torch.zeros(n, nd, device=dev), lum[:, None].expand(-1, nd),
                                torch.zeros(n, nd, device=dev), torch.zeros(n, nd, device=dev)], -1)
            V = torch.cat([tens[None].expand(n, -1, -1), desc, torch.full((n, nd, 1), contrast, device=dev)], -1)
            g.points.append((Q.reshape(-1, 3), W.reshape(-1), V.reshape(-1, 12), oid[:, None].expand(-1, nd).reshape(-1)))
        self.obstacles.append((C, R))
        return g

    def neuron(self):
        """A real neuron (or glial cell) from NeuroMorpho.org (standardised SWC), at the scene's voxel size, rotated and
        placed at random: its neurites as curves (radius along them from the reconstruction), its soma a solid; two
        populations by branch side (the thinner child changes side at every branching), one object."""
        files = neuron_files()
        if not files or self.voxel_um < 0.15:                                                   # a reconstructed cell spans 10^5 voxels at nm scales
            return None
        r = self.r; fsel = files[r.i(0, len(files) - 1)]; z = np.load(fsel)
        P_um, R_um, par, typ = z["nodes"].astype(np.float64), z["radii"].astype(np.float64), z["parents"], z["types"]
        soma = np.nonzero(typ == 1)[0]
        root = soma[0] if soma.size else int(np.nonzero(par < 0)[0][0])
        Rm = _rot(r, 1)[0].cpu().numpy().astype(np.float64)
        c = self.centre_point(0.3).cpu().numpy()
        P = (P_um - P_um[root]) @ Rm.T / self.voxel_um + c; R = np.maximum(R_um / self.voxel_um, 0.05)
        n = P.shape[0]; nch = np.bincount(par[par >= 0], minlength=n)
        kids = [[] for _ in range(n)]
        for i in range(n):
            if par[i] >= 0:
                kids[par[i]].append(i)
        chain = -np.ones(n, dtype=np.int64); chains, start_par, side = [], [], []
        for i in range(n):                                                                    # SWC lists a parent before its children
            p_ = par[i]
            if typ[i] == 1:
                continue
            if p_ >= 0 and chain[p_] >= 0 and nch[p_] == 1 and typ[p_] == typ[i]:
                chain[i] = chain[p_]; chains[chain[i]].append(i)
            else:
                chain[i] = len(chains); chains.append([i]); start_par.append(p_)
                ps = side[chain[p_]] if (p_ >= 0 and chain[p_] >= 0) else 0
                sib = [j for j in kids[p_] if j != i] if p_ >= 0 else []                           # the thicker child keeps its parent's side
                side.append(ps if not sib or R[i] >= max(R[j] for j in sib) else 1 - ps)
        lo, hi = -8.0, np.array(self.shape) + 8.0
        curves = []
        for ci, ids in enumerate(chains):
            idx = ([start_par[ci]] if start_par[ci] >= 0 else []) + ids
            Q = P[idx]; rr = R[idx]
            if not ((Q > lo) & (Q < hi)).all(1).any() or Q.shape[0] < 2:
                continue
            seg = np.linalg.norm(np.diff(Q, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
            if cum[-1] < 1.0:
                continue
            t = np.arange(0, cum[-1] + 1e-9, H)
            curves.append((np.stack([np.interp(t, cum, Q[:, a]) for a in range(3)], 1), np.interp(t, cum, rr), side[ci]))
        if not curves:
            return None
        curves = sorted(curves, key=lambda c_: -c_[0].shape[0])[:800]
        oid = int(self.ids(1, float(np.median(R)))[0]); contrast = self.contrast(); out = []
        for sd in (0, 1):
            cs = [c_ for c_ in curves if c_[2] == sd]
            if not cs:
                continue
            S = max(c_[0].shape[0] for c_ in cs)
            Pt = np.zeros((len(cs), S, 3)); Rt = np.zeros((len(cs), S))
            for j, (q, rr, _) in enumerate(cs):
                Pt[j, :q.shape[0]] = q; Pt[j, q.shape[0]:] = q[-1]; Rt[j, :q.shape[0]] = rr
            Pt = torch.tensor(Pt, dtype=torch.float32, device=self.device); Rt = torch.tensor(Rt, dtype=torch.float32, device=self.device)
            cv = curves_like(Pt, Rt, r, obj=torch.full((len(cs),), oid, device=self.device), seg=torch.arange(len(cs), device=self.device), contrast=contrast)
            out.append(Group("neuron", "line", float(np.median(R)), contrast, curves=[cv]))
        if soma.size:                                                                         # the soma: a solid of its reconstructed radius
            rs = float(R[soma].max())
            out.append(self._solids(C=torch.tensor(P[root][None], dtype=torch.float32, device=self.device),
                                    A=torch.eye(3, device=self.device)[None] * rs * r.u(0.9, 1.3), name="soma", oid=torch.tensor([oid], device=self.device)))
        self.record["families"].append({"neuron": {"file": fsel.stem, "cell_type": str(z["cell_type"]) if "cell_type" in z else "", "chains": len(curves)}})
        return out

    def tissue(self, grey: bool = False) -> list:
        """Packed tissue (27 Oct): the canvas filled wall to wall, as white matter is -- not bundles on a background.
        Domains (a warped Voronoi partition, each stretched along its own fibre direction as fascicles are) abut with
        no gap, or with a thin extracellular band; a tract domain is one population following a smooth direction field
        at packing 0.3-0.85 (contrast drawn apart from it), often nearly parallel to its neighbour (fascicles of one
        tract); a crossing zone holds two or three populations interleaved fibre by fibre, by fascicles or in layers
        (angles oversampled at 8-40 deg; shares 0.1-0.5; contrasts often related; sometimes three or four in one plane,
        the line populations a sheet must be told from); a cell domain holds packed cells. Elements fill each domain by a
        coverage grid (seeded where it is free, grown along the field until they leave their domain): fibres of radius
        >= 0.45 voxel one by one (each an object), finer ones as strands of their partial volume over a smooth part (as
        unresolved bundles). Then rows of cells along the fibres and vessels through the tissue, sometimes. ``grey``:
        grey-matter-like -- sparse fibres turning in every direction among dense cells and neuropil. Returns the groups."""
        r = self.r; dev = self.device; shp = self.shape; Z = float(max(shp)); vu = self.voxel_um
        from . import compose as CP
        g1 = [torch.arange(n, device=dev, dtype=torch.float32) for n in shp]
        X = torch.stack(torch.meshgrid(*g1, indexing="ij"), -1)                                     # (Z, Y, X, 3)
        rf = r.lu(0.15, 4.0) / vu                                                                   # the fibres' radius (voxels)
        rec = {"grey": bool(grey), "fibre_um": round(rf * vu, 3)}
        pops = []                                                                                   # mask, field grid, packing, contrast, kind

        Xc = X[::2, ::2, ::2]                                                                       # (smooth over >= 13 voxels: evaluated every
        def dir_grid(d0, amp, L):                                                                   # second voxel, interpolated)
            f = CP.Smooth(d0, amp, L, r); Xf = Xc.reshape(-1, 3)
            dc = torch.cat([f.dirs(Xf[i:i + (1 << 20)]) for i in range(0, Xf.shape[0], 1 << 20)]).view(*Xc.shape[:3], 3)
            d = Fn.interpolate(dc.movedim(-1, 0)[None], size=shp, mode="trilinear", align_corners=False)[0].movedim(0, -1)
            return Fn.normalize(d, dim=-1)

        def narrow_angle():
            return r.u(8, 40) if r.p(P_NARROW) else r.u(5, 90)

        def turned(d, deg):
            p = Fn.normalize(torch.linalg.cross(d, r.unit()), dim=0); t = math.radians(deg)
            return Fn.normalize(math.cos(t) * d + math.sin(t) * p, dim=0)

        if grey:
            K = 1; lab = torch.zeros(shp, dtype=torch.long, device=dev); kinds = ["grey"]; gap = torch.zeros(shp, dtype=torch.bool, device=dev)
        else:
            K = r.i(2, 6); C = r.rand(K, 3) * torch.tensor(shp, device=dev, dtype=torch.float32)
            kinds = ["cross" if r.p(0.3) else "cells" if r.p(0.08) else "tract" for _ in range(K)]
            if "cross" not in kinds and r.p(0.5):
                kinds[r.i(0, K - 1)] = "cross"
            dirs, prev = [], None
            for k in range(K):                                                                      # neighbours often one tract's fascicles
                d = turned(prev, r.lu(3, 25)) if (prev is not None and r.p(0.5)) else r.unit(); dirs.append(d); prev = d
            Dk = []
            for k in range(K):                                                                      # distance stretched along the domain's
                v = X - C[k]; s_ = 1.0 if kinds[k] == "cross" else r.lu(2.0, 6.0)                 # direction: domains elongated as fascicles
                along = v @ dirs[k]
                Dk.append(((v - along[..., None] * dirs[k]).pow(2).sum(-1) + (along / s_) ** 2).sqrt()
                          + r.u(0.0, 0.2) * Z * smooth_noise(shp, r.lu(8, 30), r.g, dev))
            Dk = torch.stack(Dk); two = Dk.topk(2, dim=0, largest=False).values; lab = Dk.argmin(0); del Dk
            gap = (two[1] - two[0]) < (r.u(0.5, 2.0) if r.p(0.3) else 0.0)                          # thin bands between domains
            rec["gap_band"] = bool(gap.any())
        for k in range(K):
            dom = (lab == k) & ~gap
            if int(dom.sum()) < 200:
                continue
            if kinds[k] == "grey":
                for _ in range(r.i(1, 2)):
                    pops.append(dict(mask=dom, dg=dir_grid(r.unit(), r.lu(0.8, 2.0), r.lu(0.1, 0.4) * Z), phi=r.u(0.03, 0.15), contrast=self.contrast(), kind="grey"))
            elif kinds[k] == "tract":
                pops.append(dict(mask=dom, dg=dir_grid(dirs[k], r.lu(0.02, 0.4), r.lu(0.3, 2.0) * Z), phi=r.u(0.3, 0.85), contrast=self.contrast(), kind="tract"))
            elif kinds[k] == "cells":
                pops.append(dict(mask=dom, kind="cells"))
            else:                                                                                   # a crossing zone
                planar = r.p(0.15); M = r.i(3, 4) if planar else (3 if r.p(0.35) else 2)
                A = dirs[k]; n_pl = Fn.normalize(torch.linalg.cross(A, r.unit()), dim=0)
                if planar:                                                                          # M lines in one plane, spread
                    th0 = r.u(0, 180); step = r.u(25, 180 / M)
                    ds = [Fn.normalize(math.cos(math.radians(th0 + j * step)) * A + math.sin(math.radians(th0 + j * step)) * torch.linalg.cross(n_pl, A), dim=0) for j in range(M)]
                    angs = [round(step, 1)] * (M - 1)
                else:
                    a2 = narrow_angle(); ds = [A, turned(A, a2)]; angs = [round(a2, 1)]
                    if M == 3:
                        ds.append(r.unit()); angs.append(round(math.degrees(math.acos(min(1.0, abs(float(ds[2] @ A))))), 1))
                sh = np.array([1.0] + [r.u(0.1, 0.5) / (1 - r.u(0.1, 0.5)) for _ in range(M - 1)]); sh = sh / sh.sum()   # minority shares 0.1-0.5
                grain = "layers" if planar else ["fibre", "fascicle", "layers"][r.i(0, 2)]
                phi = r.u(0.3, 0.85); c0 = self.contrast(); related = r.p(P_RELATED_CONTRAST)
                cons = [c0 * (r.lu(0.9, 1.1) if related else 1.0) if (related or j == 0) else self.contrast() for j in range(M)]
                if grain == "fibre":                                                                # each population over the whole zone
                    subs = [dom] * M; phis = [phi * s_ for s_ in sh]
                elif grain == "layers":                                                             # slabs along the planes' common normal
                    nrm = Fn.normalize(torch.linalg.cross(ds[0], ds[1]), dim=0); t_ = r.lu(2.0, 12.0) * max(rf, 1.0)
                    li = ((X @ nrm) / t_).floor().long(); u_ = torch.rand(int(li.max() - li.min()) + 1, generator=r.g, device=dev)
                    which = torch.searchsorted(torch.tensor(np.cumsum(sh)[:-1], device=dev, dtype=torch.float32), u_[li - li.min()])
                    subs = [dom & (which == j) for j in range(M)]; phis = [phi] * M
                else:                                                                               # fascicles: the first population in tubes
                    e1_ = Fn.normalize(torch.linalg.cross(ds[0], r.unit()), dim=0); e2_ = torch.linalg.cross(ds[0], e1_)   # along its own direction,
                    g_ = r.lu(3.0, 15.0) * max(rf, 1.0); wq = r.u(0.0, 0.4) * g_                      # the others between them: fascicle
                    uv = torch.stack([X @ e1_, X @ e2_], -1) / g_ + (wq / g_) * torch.stack([smooth_noise(shp, g_, r.g, dev) for _ in range(2)], -1)
                    ij = uv.floor().long(); key = ij[..., 0] * 100003 + ij[..., 1]                    # sections on a warped grid across it
                    uk, inv = torch.unique(key, return_inverse=True)
                    tube = (torch.rand(uk.shape[0], generator=r.g, device=dev) < float(sh[0]))[inv]
                    rest = dom & ~tube; subs = [dom & tube] + [rest] * (M - 1)
                    phis = [phi] + [phi * s_ / max(1 - sh[0], 1e-3) for s_ in sh[1:]]
                cz = torch.nonzero(dom).float().mean(0)                                              # (crops may centre on the zone)
                for j in range(M):
                    pops.append(dict(mask=subs[j], dg=dir_grid(ds[j], r.lu(0.02, 0.3), r.lu(0.3, 2.0) * Z), phi=phis[j], contrast=cons[j], kind="cross", centre=cz))
                rec.setdefault("crossing", []).append({"angles": angs, "grain": grain, "shares": [round(float(x), 2) for x in sh], "planar": planar,
                                                       "related_contrast": related})
        rec["domains"] = K; rec["kinds"] = kinds
        lines = [p for p in pops if p["kind"] != "cells"]
        groups = []
        if lines:                                                                                   # fill: fibres one by one, or strands
            vol = sum(float(p["mask"].sum()) for p in lines); L_ = min(2.0 * Z, max(20.0, Z * 0.8))
            cap = 3000 if rf >= 0.45 else 4000                                                    # (resolved: the swept renderer's cost ~
                                                                                                  # elements x length -- CPU workers)
            re_min = math.sqrt(0.55 * vol / (math.pi * L_ * cap))
            fibres = rf >= 0.45 and rf >= re_min
            re = rf if fibres else max(0.6, rf, re_min, r.lu(0.6, 2.5))
            P_all, pid_all, cov_frac = self._fill_domains(lines, re, cap, fibres)
            rec.update({"resolved": bool(fibres), "element_vox": round(re, 2), "elements": int(P_all.shape[0]) if P_all is not None else 0,
                        "coverage": [round(float(c_), 2) for c_ in cov_frac], "packing": [round(float(p["phi"]), 2) for p in lines]})
            for j, p in enumerate(lines):
                P = _trim_paths(P_all[pid_all == j]) if P_all is not None else None
                pid = int(self.ids(1, float(p["mask"].sum()) ** (1 / 3))[0])
                g = Group("tract" if p["kind"] != "grey" else "grey fibres", "line", re, p["contrast"])
                tens = spread_tensor(p["dg"], float(spread_e(draw_spread(1, r.g, dev))[0])).movedim(-1, 0)   # (6, Z, Y, X) along the field,
                                                                                                    # the smooth fill's own spread (29 Oct)
                if P is not None and P.shape[0]:
                    N, S = P.shape[:2]
                    if fibres:                                                                      # resolved: each fibre an object
                        rad = torch.full((N, S), re, device=dev) * torch.exp(0.12 * r.randn(N, 1)).clamp(0.8, 1.2)
                        obj = self.ids(N, rad.amin(1))
                        for o_ in obj.tolist():
                            self.parents[o_] = pid
                        g.curves.append(curves_like(P, rad, r, obj=obj, seg=torch.full((N,), pid, device=dev), contrast=p["contrast"]))
                    else:                                                                           # strands of the partial volume, drawn as a
                        vis_ = r.p(P_VISIBLE_STRANDS)                                               # field: their centrelines' material and
                        beta = r.u(0.6, 0.95) if vis_ else r.u(0.2, 0.5)                            # (29 Oct: mostly visible streaks)
                        seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1).sum()                              # direction splatted and blurred by their
                        fill = max(float(seg) * math.pi * re * re / max(float(p["mask"].sum()), 1.0), 0.05)   # radius (soft-edged, as a
                        xi = torch.exp((r.u(0.3, 0.7) if vis_ else r.u(0.15, 0.5)) * r.randn(N)); xi = xi / xi.mean()   # group of fine fibres is; the swept
                        cap_ = 0.7 * (1 - p["phi"] * (1 - beta))                                     # renderer took minutes on a CPU). A strand's
                        dens = (beta * p["phi"] / fill * xi).clamp(0.02, max(cap_, 0.05))           # blurred peak is ~1.39 dens: capped below
                        occ_s, ten_s = self._strand_field(P, re, dens, spread=draw_spread(P.shape[0], r.g, dev))   # saturation, the shortfall in the smooth part
                        phi_b = max(p["phi"] - fill * float(dens.mean()), 0.0)
                        m_in = p["mask"]
                        for _ in range(2):                                                          # overlapping strands: the field's 99th
                            v_ = occ_s[m_in]                                                        # percentile kept under the smooth part's
                            s99 = float(torch.quantile(v_[:: max(1, v_.numel() // 200_000)], 0.99)) if v_.numel() else 0.0   # room, the
                            room = 0.98 - phi_b                                                     # mass taken off spread into the smooth part
                            if s99 <= room or room <= 0.05:
                                break
                            k_ = room / s99; ms = float(v_.mean())
                            occ_s = occ_s * k_; ten_s = ten_s * k_; phi_b = min(phi_b + (1 - k_) * ms, 0.9)
                        occ_b = p["mask"].float() * phi_b                                           # (the rest of the fill)
                        occ = occ_s + occ_b
                        tens = torch.where((occ > 1e-6)[None], (ten_s + occ_b[None] * tens) / occ.clamp_min(1e-6)[None], tens)
                        on = (occ > 1e-4).float()
                        desc = torch.zeros((5,) + shp, device=dev); desc[0] = rf * on; desc[3] = on
                        g.dense.append((occ.half(), tens.half(), desc.half(), p["contrast"], pid))  # (half precision, one object)
                if P is None or P.shape[0] == 0:                                                    # (nothing placed: no population)
                    continue
                g.centre_P = P[P.shape[0] // 2]
                if "centre" in p:
                    g.centre_pt = p["centre"]
                g.paths = P
                groups.append(g)
        for p in pops:                                                                              # packed cells
            if p["kind"] == "cells":
                rc = max(1.2, (self.size(3, 12, 1.2, 10.0) or 2.0)); idx = torch.nonzero(p["mask"]).float()
                if idx.shape[0] == 0:
                    continue
                Cc = idx[torch.randperm(idx.shape[0], generator=r.g, device=dev)[:4000]] + r.rand(min(4000, idx.shape[0]), 3) - 0.5
                cell_ = (Cc / (2.2 * rc)).floor().long(); key = (cell_[:, 0] * 4099 + cell_[:, 1]) * 4099 + cell_[:, 2]
                _, inv = torch.unique(key, return_inverse=True)
                first = torch.full((int(inv.max()) + 1,), Cc.shape[0], device=dev, dtype=torch.long).scatter_reduce_(0, inv, torch.arange(Cc.shape[0], device=dev), "amin")
                Cc = Cc[first]; Cc = Cc[torch.randperm(Cc.shape[0], generator=r.g, device=dev)[:1500]]; n = Cc.shape[0]   # (at most 1500, as
                A = _rot(r, n) * (rc * torch.exp(0.12 * r.randn(n, 1, 1)).clamp(0.8, 1.15))                              # the cells family's 700)
                gc = self._solids(Cc, A, "cells", self.ids(n, rc))
                if gc is not None:
                    groups.append(gc); rec["cells_packed"] = rec.get("cells_packed", 0) + n
        tracts = [g for g in groups if g.name == "tract" and getattr(g, "paths", None) is not None and g.paths.shape[0] >= 4]
        if tracts and not grey and r.p(0.5):                                                         # rows of cells along the fibres
            g_ = tracts[r.i(0, len(tracts) - 1)]; P = g_.paths; rc = max(1.0, (self.size(3, 8, 1.0, 6.0) or 1.5))
            pick = torch.randperm(P.shape[0], generator=r.g, device=dev)[:r.i(5, 40)]; sp = r.lu(2.5, 6.0) * rc
            Cs, Ts = [], []
            for q in pick.tolist():
                Pq = P[q]; seg = (Pq[1:] - Pq[:-1]).norm(dim=-1); s_ = torch.cat([torch.zeros(1, device=dev), seg.cumsum(0)])
                if float(s_[-1]) < 2 * sp:
                    continue
                at = torch.arange(float(r.u(0, sp)), float(s_[-1]), sp, device=dev); i_ = torch.searchsorted(s_, at).clamp(1, len(s_) - 1)
                Cs.append(Pq[i_]); Ts.append(Fn.normalize(Pq[i_] - Pq[i_ - 1], dim=-1))
            if Cs:
                Cc = torch.cat(Cs); Tt = torch.cat(Ts); n = Cc.shape[0]; el = r.lu(1.2, 2.0)
                e1_ = Fn.normalize(torch.linalg.cross(Tt, r.unit()[None].expand(n, 3)), dim=-1); e2_ = torch.linalg.cross(Tt, e1_)
                Cc = Cc + e1_ * (rc * r.u(1.0, 2.0) + 0.5)                                          # beside the fibre, not on its path
                A = torch.stack([e1_ * rc, e2_ * rc, Tt * rc * el], -1)                             # long axis along the fibres
                gc = self._solids(Cc, A, "cell rows", self.ids(n, rc))
                if gc is not None:
                    groups.append(gc); rec["cell_rows"] = n
        if grey:                                                                                    # among dense cells and neuropil
            for fam_ in (self.cells, self.neuropil):
                out_ = fam_()
                for g_ in (out_ if isinstance(out_, list) else [out_] if out_ is not None else []):
                    groups.append(g_)
        elif r.p(0.4):                                                                              # vessels through the tissue
            out_ = self.tree("vessel")
            for g_ in (out_ if isinstance(out_, list) else [out_] if out_ is not None else []):
                groups.append(g_)
            rec["vessels"] = True
        self.record["tissue"] = rec
        return groups

    def _strand_field(self, P: torch.Tensor, re: float, dens: torch.Tensor, spread: torch.Tensor | None = None):
        """Strands ``P (N, S, 3)`` of radius ``re`` and density ``dens (N,)`` as fields: each segment's material volume
        (pi re^2 ds dens) and its direction dyad (with ``spread (N,)`` degrees: the strand's material ellipse) splatted
        trilinearly at its midpoint, then blurred (sigma 0.6 ``re``) --
        ``occ (Z, Y, X)`` (mass kept) and the occupancy-weighted tensor ``(6, Z, Y, X)``."""
        shp = self.shape; dev = self.device
        sg = P[:, 1:] - P[:, :-1]; ds = sg.norm(dim=-1); keep = ds > 1e-6
        X = (0.5 * (P[:, 1:] + P[:, :-1]))[keep]; T = (sg / ds.clamp_min(1e-9)[..., None])[keep]
        W = (math.pi * re * re * ds * dens[:, None])[keep]
        Tt = line_tensor(T) if spread is None else spread_tensor(T, spread_e(spread)[:, None].expand_as(ds)[keep])
        V = torch.cat([W[:, None], W[:, None] * Tt], 1)                                              # (n, 7)
        G = torch.zeros(int(np.prod(shp)), 7, device=dev); Z_, Y_, X_ = shp
        p0 = X.floor(); fr = X - p0; i0 = p0.long()
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    iz, iy, ix = i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx
                    w = (fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dx else 1 - fr[:, 2])
                    ok = (iz >= 0) & (iz < Z_) & (iy >= 0) & (iy < Y_) & (ix >= 0) & (ix < X_)
                    G.index_add_(0, ((iz * Y_ + iy) * X_ + ix)[ok], V[ok] * w[ok, None])
        G = gaussian(G.T.reshape(7, *shp), max(0.5, 0.6 * re))
        return G[0], G[1:]

    def _fill_domains(self, pops: list, re: float, cap: int, fibres: bool, rounds: int = 3):
        """Elements of radius ``re`` filling each population's mask along its field grid (``pops[j]["mask"]``,
        ``["dg"] (Z, Y, X, 3)``, ``["phi"]``), all grown in one batch: per round, seeds where the shared coverage grid is
        still free (one per cell of side 2 ``re``, at most ``cap`` in all), each grown both ways along its own field
        (nearest voxel, axial sign kept) until it leaves its own mask; the grid then marks every voxel within ``re`` of
        a path. A fibre-grain crossing's populations share their zone's grid, so they do not overlap where seeded.
        Returns paths ``(N, S, 3)``, each path's population, and each population's covered fraction."""
        r = self.r; dev = self.device; shp = self.shape; lim = torch.tensor(shp, device=dev)
        Mk = torch.stack([p["mask"] for p in pops]); DG = torch.stack([p["dg"].half() for p in pops])   # (J, Z, Y, X), (J, Z, Y, X, 3)
        Mc = Mk[:, ::2, ::2, ::2]; shc = tuple(Mc.shape[1:]); limc = torch.tensor(shc, device=dev)     # the coverage at half resolution
        vol = Mc.flatten(1).sum(1).float().clamp_min(1)                                                # (enough to choose seeds; 8x cheaper)
        target = torch.tensor([min(p["phi"], 0.6) if fibres else min(0.6, max(0.1, 2 * p["phi"])) for p in pops], device=dev)   # fibres: their
        cov = torch.zeros(shc, dtype=torch.bool, device=dev); k = 2 * int(re // 2) + 1                  # packing (random addition jams
        h = 0.75; n_st = int(min(2.0 * max(shp), 400) / h / 2); paths, pids = [], []                    # ~0.6); strands tile, densities carry it
        for rnd in range(rounds):
            frac = (Mc & cov[None]).flatten(1).sum(1).float() / vol
            need = (target - frac).clamp_min(0)
            n_have = sum(int(x.shape[0]) for x in paths)
            if float(need.max()) <= 0.02 or n_have >= cap:
                break
            seeds, sp = [], []; left = cap - n_have                                                # one budget for all the populations
            want = [max(4.0, 1.6 * float(need[j]) * 8 * float(vol[j]) / (math.pi * re * re * 0.6 * max(shp))) if float(need[j]) > 0.02 else 0.0   # (vol: coarse
                    for j in range(len(pops))]                                                                                                # cells, 8 voxels)
            scale = min(1.0, left / max(sum(want), 1.0))
            for j in range(len(pops)):
                if want[j] <= 0:
                    continue
                free = torch.nonzero(Mc[j] & ~cov)
                if free.shape[0] == 0:
                    continue
                n_j = max(1, int(want[j] * scale))
                pick = 2 * free[torch.randperm(free.shape[0], generator=r.g, device=dev)[:4 * n_j]].float()
                pick = (pick + 2 * r.rand(pick.shape[0], 3) - 0.5).clamp(max=lim.float() - 1)
                ii = pick.round().long().clamp_min(0); pick = pick[Mk[j, ii[:, 0], ii[:, 1], ii[:, 2]]]   # (inside the fine mask)
                if pick.shape[0] == 0:
                    continue
                cl = (pick / max(2 * re, 1.0)).floor().long(); key = (cl[:, 0] * 4099 + cl[:, 1]) * 4099 + cl[:, 2]
                _, inv = torch.unique(key, return_inverse=True)
                first = torch.full((int(inv.max()) + 1,), pick.shape[0], device=dev, dtype=torch.long).scatter_reduce_(0, inv, torch.arange(pick.shape[0], device=dev), "amin")
                s_ = pick[first[torch.randperm(first.shape[0], generator=r.g, device=dev)[:n_j]]]
                seeds.append(s_); sp.append(torch.full((s_.shape[0],), j, device=dev, dtype=torch.long))
            if not seeds:
                break
            x0 = torch.cat(seeds); pj = torch.cat(sp); N = x0.shape[0]

            def look(x):
                i = torch.minimum(x.round().long().clamp_min(0), lim - 1)
                inside = ((x >= -0.5) & (x <= lim - 0.5)).all(-1) & Mk[pj, i[:, 0], i[:, 1], i[:, 2]]
                return DG[pj, i[:, 0], i[:, 1], i[:, 2]].float(), inside

            halves = []
            for sgn in (1.0, -1.0):                                                                 # both ways from each seed
                x = x0.clone(); d_prev = look(x)[0] * sgn; alive = torch.ones(N, dtype=torch.bool, device=dev); out = []
                for _ in range(n_st):
                    d, _ = look(x); d = d * torch.sign((d * d_prev).sum(-1, keepdim=True) + 1e-9)
                    xm = x + 0.5 * h * d; dm, _ = look(xm); dm = dm * torch.sign((dm * d).sum(-1, keepdim=True) + 1e-9)
                    xn = x + h * dm; _, ok = look(xn); ok = ok & alive
                    x = torch.where(ok[:, None], xn, x); d_prev = torch.where(ok[:, None], dm, d_prev); alive = ok
                    out.append(x.clone())
                    if not bool(alive.any()):
                        break
                halves.append(torch.stack(out, 1))
            P = torch.cat([halves[1].flip(1), x0[:, None], halves[0]], 1)
            moved = (P[:, -1] - P[:, 0]).norm(dim=-1) > 2.0                                         # (seeds stuck at a border: dropped)
            P, pj = P[moved], pj[moved]
            if P.shape[0]:
                for j in torch.unique(pj).tolist():                                                  # each population's coverage only within
                    Q = (P[pj == j][:, ::2].reshape(-1, 3) / 2).round().long(); okq = ((Q >= 0) & (Q < limc)).all(-1); Q = Q[okq]   # its own mask
                    hit = torch.zeros(shc, device=dev); hit[Q[:, 0], Q[:, 1], Q[:, 2]] = 1          # (thin layers: a neighbour's spill
                    cov |= (_dilate(hit, k) > 0) & Mc[j]                                             # must not fill them)
                paths.append(P); pids.append(pj)
        if not paths:
            return None, None, [0.0] * len(pops)
        S = max(p_.shape[1] for p_ in paths)
        pad = lambda p_: torch.cat([p_, p_[:, -1:].expand(-1, S - p_.shape[1], -1)], 1) if p_.shape[1] < S else p_   # (repeated ends: no material)
        frac = ((Mc & cov[None]).flatten(1).sum(1).float() / vol).tolist()
        return torch.cat([pad(p_) for p_ in paths]), torch.cat(pids), frac

    # ---- oriented texture fields (29 Oct): orientation as a statistical property of a dense texture
    def _texture_field(self) -> torch.Tensor:
        """A smooth orientation field ``(3, Z, Y, X)``: 1-4 control directions at random points blended by Gaussian
        weights (signs aligned to the first), plus a random smooth warp of the directions."""
        r = self.r; dev = self.device; shp = self.shape
        X = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in shp), indexing="ij"), 0)
        K = r.i(1, 4); w = r.lu(20, 80); d0 = r.unit(); v = torch.zeros((3,) + shp, device=dev)
        for k in range(K):
            c = r.rand(3) * torch.tensor(shp, device=dev, dtype=torch.float32); d = d0 if k == 0 else r.unit()
            d = d if float(d @ d0) >= 0 else -d
            v += torch.exp(-0.5 * ((X - c.view(3, 1, 1, 1)) ** 2).sum(0) / w ** 2)[None] * d.view(3, 1, 1, 1)
        v = v / v.norm(dim=0, keepdim=True).clamp_min(1e-6)
        a = r.u(0.0, 0.7)                                                                        # the warp of the directions
        if a > 0:
            v = v + a * direction_field(shp, r.lu(8, 40), r.g, dev)
        return v / v.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def _trace(self, seeds: torch.Tensor, f: torch.Tensor, delta: torch.Tensor, h: float, S: int) -> torch.Tensor:
        """Streamlines ``(N, S, 3)`` through ``seeds (N, 3)`` along the field ``f (3, Z, Y, X)`` (both ways, step ``h``),
        each strand's direction offset by its own ``delta (N, 3)`` (its dispersion about the field)."""
        sz = torch.tensor(self.shape, device=seeds.device, dtype=torch.float32)
        def at(q):
            g = (2 * q / (sz - 1).clamp_min(1) - 1).flip(-1).view(1, 1, 1, -1, 3)
            return Fn.grid_sample(f[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0].T
        half = (S - 1) // 2; out = [seeds]
        for sgn in (1.0, -1.0):
            q = seeds.clone(); prev = None; pts = []
            for _ in range(half if sgn > 0 else S - 1 - half):
                d = at(q) + delta
                d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                if prev is None:
                    d = d * sgn
                else:
                    d = torch.where(((d * prev).sum(-1, keepdim=True) < 0), -d, d)
                q = q + h * d; prev = d; pts.append(q)
            if sgn > 0:
                out = out + pts
            else:
                out = pts[::-1] + out
        return torch.stack(out, 1)

    def _texture_population(self, f: torch.Tensor, mask: torch.Tensor, rec: dict):
        """One texture population: strands (width 0.5-4 voxels, elongation 2-20 -- at least 4 voxels long --, per-strand
        dispersion 2-15 deg (a per-strand direction offset of that rms per component), random
        packing) traced along ``f`` from seeds in ``mask``, splatted as a field (:meth:`_strand_field`); one object,
        its tensor the strands' own directions (the field plus their dispersion)."""
        r = self.r; dev = self.device; shp = self.shape
        width = r.lu(0.5, 4.0); re = width / 2; elong = r.lu(2, 20)
        L = max(4.0, min(width * elong, 0.5 * max(shp))); phi = r.lu(0.03, 0.6); spread = math.radians(r.lu(2, 15))   # (shorter than ~4
                                                                                                 # voxels a strand renders as a blob)
        vol = float(mask.sum())
        if vol < 64:
            return None
        per = math.pi * re * re * L
        N = int(max(20, min(TEXTURE_MAX_STRANDS, phi * vol / per)))
        h = max(0.5, L / 24); S = max(3, int(round(L / h)) + 1)
        idx = torch.nonzero(mask > 0.5)
        if idx.shape[0] == 0:
            return None
        seeds = idx[torch.randint(0, idx.shape[0], (N,), generator=r.g, device=dev)].float() + r.rand(N, 3) - 0.5
        delta = spread * r.randn(N, 3)
        P = self._trace(seeds, f, delta, h, S)
        dens = (phi * vol / (N * per)) * torch.exp(0.3 * r.randn(N)).clamp(0.3, 3.0)
        dens = dens.clamp(0.02, 0.7)                                                             # (a blurred strand's peak ~1.4 dens)
        occ_s, ten_s = self._strand_field(P, re, dens)
        T = torch.where((occ_s > 1e-6)[None], ten_s / occ_s.clamp_min(1e-6)[None], line_tensor(f.movedim(0, -1)).movedim(-1, 0))
        T = T / T[:3].sum(0, keepdim=True).clamp_min(1e-6)
        occ = occ_s * mask
        q99 = float(torch.quantile(occ[mask > 0.5][:: max(1, int((mask > 0.5).sum()) // 200_000)], 0.99)) if bool((mask > 0.5).any()) else 0.0
        if q99 > 0.9:
            occ = occ * (0.9 / q99)
        occ = occ.clamp(0, 1)
        on = (occ > 1e-4).float(); desc = torch.zeros((5,) + shp, device=dev); desc[0] = re * on
        contrast = self.contrast(25, 100) if r.p(0.8) else self.contrast(5, 25); oid = self.ids(1, max(re, 0.25))   # 80 % clear, 20 % faint
        obj_map = torch.full(shp, int(oid[0]), device=dev, dtype=torch.long); n_cl = 1
        if TEXTURE_CLUSTERS is not None:                                                         # compact patches of strands as objects
            obj_map, n_cl = self._texture_clusters(seeds, int(oid[0]), max(re, 0.25))            # under the field (29 Oct)
        rec.setdefault("pops", []).append({"width": round(width, 2), "elong": round(elong, 1), "phi": round(phi, 3), "spread_deg": round(math.degrees(spread), 1),
                                           "strands": N, "contrast": round(contrast, 1), "clusters": n_cl})
        g = Group("texture", "line", re, contrast, dense=[(occ, T, desc, contrast, obj_map)])
        return g

    def _texture_clusters(self, seeds: torch.Tensor, field: int, radius: float):
        """A texture field's objects: k-means (10 rounds) on its strands' seeds into ``K`` clusters (one per ~100 strands,
        within :data:`TEXTURE_CLUSTERS`), each a new object under ``field``; every voxel goes to its nearest centre (a
        Voronoi map -- compact patches; a strand crossing a patch edge is shared between them). ``radius``: the clusters'
        (the strands'), so the resolution rule merges them into the field as it would the strands. Returns the voxel ->
        object map and K."""
        r = self.r; dev = self.device; shp = self.shape; N = seeds.shape[0]
        lo, hi = TEXTURE_CLUSTERS; K = int(min(N, max(lo, min(hi, N // 100))))
        Cc = seeds[torch.randperm(N, generator=r.g, device=dev)[:K]].clone()
        for _ in range(10):
            lab = torch.cdist(seeds, Cc).argmin(1)
            cnt = torch.bincount(lab, minlength=K).float()
            sm = torch.zeros(K, 3, device=dev).index_add_(0, lab, seeds)
            Cc = torch.where(cnt[:, None] > 0, sm / cnt.clamp_min(1)[:, None], Cc)
        cid = self.ids(K, radius)
        for o_ in cid.tolist():
            self.parents[o_] = field
        X = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in shp), indexing="ij"), -1).reshape(-1, 3)
        lab = torch.cat([torch.cdist(X[i:i + 262144], Cc).argmin(1) for i in range(0, X.shape[0], 262144)])
        return cid[lab].view(shp).long(), K

    def texture(self):
        """Oriented texture fields: orientation as a statistical property of a dense, fibre-like texture -- a smooth
        orientation field (control directions, smooth interpolation, a random warp), strands of 0.5-4 voxels and
        elongation 2-20 along it with their own dispersion, packing from clear to barely above the background; over
        the whole crop or a region; sometimes 2-3 superposed fields at 35-90 degrees (one population each: two or three
        lobes), sometimes a field turning sharply across a boundary (two populations, complementary masks). Each field is
        one population with one object: its target is the strands' own direction distribution."""
        r = self.r; dev = self.device; shp = self.shape
        region = torch.ones(shp, device=dev)
        if r.p(0.5):                                                                            # a region, not the whole canvas
            region = (smooth_noise(shp, r.lu(15, 50), r.g, dev) > r.u(-0.8, 0.5)).float()
            region = gaussian(region, 1.0)
        d1 = self._texture_field(); rec = {}
        u_ = r.u(0, 1)
        if u_ < 0.35:                                                                           # superposed fields
            k = 2 if r.p(0.7) else 3
            fields = [d1] + [companion_field(d1, r.u(35, 90), r.lu(15, 60), r.g, dev) for _ in range(k - 1)]
            masks = [region] * k; rec["mode"] = f"superposed{k}"
        elif u_ < 0.55:                                                                         # a sharp turn across a boundary
            d2 = companion_field(d1, r.u(30, 90), r.lu(15, 60), r.g, dev)
            X = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in shp), indexing="ij"), 0)
            c = self.centre_point(); nrm = r.unit()
            sd = ((X - c.view(3, 1, 1, 1)) * nrm.view(3, 1, 1, 1)).sum(0) + r.u(0, 8) * smooth_noise(shp, r.lu(10, 40), r.g, dev)
            m = torch.sigmoid(sd / 0.7); fields = [d1, d2]; masks = [region * m, region * (1 - m)]; rec["mode"] = "boundary"
        else:
            fields, masks = [d1], [region]; rec["mode"] = "single"
        out = [g for g in (self._texture_population(f_, m_, rec) for f_, m_ in zip(fields, masks)) if g is not None]
        self.record["families"].append({"texture": rec})
        return out or None

    def neuropil(self, fill: float | None = None, name: str = "neuropil", r_min: float = 0.0, somata: bool = True):
        """Neuropil: dense, short, tortuous thin fibres (axons and dendrites, 0.1-1.5 um radius) in random directions and
        small somata (3-8 um), filling the canvas -- resolvable at a micron or two, partial-volume texture at 20 um;
        isotropic in aggregate, locally oriented where a fibre is resolved. ``fill``: the fibres' share of the canvas's volume
        (the matrix, 29 Oct: dense, as neuropil is; at most :data:`MATRIX_MAX_FIBRES` fibres); default, a sparse underlay.
        ``r_min``: no fibre thinner (voxels): at coarse voxels each is then a micro-fascicle of unresolved neurites, still one
        object with its own ODF. ``name``: the population's name; ``somata``: somata among the fibres."""
        r = self.r; dev = self.device
        rf = max(min(r.lu(0.1, 1.5) / self.voxel_um, min(self.shape) / 16), r_min)               # nm scenes: a neurite fills the view, no wider
        L = r.lu(20, 200) / self.voxel_um
        if L < 3:
            return None
        L = min(L, 0.5 * max(self.shape))
        n = int(min(700, r.u(0.3, 1.0) * 700, 1.5 * np.prod(self.shape) / (math.pi * rf * rf * L) + 1))   # thick neurites (nm scales): about 1.5 canvases of them
        if fill is not None:                                                                    # the matrix: this share of the volume
            n = int(min(MATRIX_MAX_FIBRES, fill * np.prod(self.shape) / (math.pi * max(rf, 0.05) ** 2 * L) + 1))
        S = max(8, int(L / H))
        start = r.rand(n, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        P = smooth_paths(start, r.unit(n), torch.full((n,), L, device=dev), S, torch.full((n,), r.lu(0.05, 0.4), device=dev), max(L / 4, 1.0), r)
        rad = torch.full((n, S), max(rf, 0.05), device=dev) * torch.exp(0.2 * r.randn(n, 1))
        cv = curves_like(P, rad, r, obj=self.ids(n, rad[:, 0]), seg=torch.zeros(n, dtype=torch.long, device=dev), contrast=self.contrast(10, 90))
        out = [Group(name, "line", float(rad.median()), float(cv.contrast[0, 0]), curves=[cv])]
        rs = r.lu(3, 8) / self.voxel_um
        if somata and 0.4 <= rs <= min(self.shape) / 6:                                                    # somata among the fibres
            m = int(min(400, r.u(0.2, 1.0) * np.prod(self.shape) / (40 * rs) ** 3 * 8 + 1))
            C = r.rand(m, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
            A = _rot(r, m) @ torch.diag_embed(torch.stack([torch.full((m,), rs, device=dev)] * 2 + [torch.full((m,), rs * r.u(1.0, 1.6), device=dev)], 1))
            out.append(self._solids(C, A, "somata", self.ids(m, rs)))
        self.record["families"].append({"neuropil": {"fibres": n, "fibre_um": round(rf * self.voxel_um, 2), "length_um": round(L * self.voxel_um), "somata_um": round(rs * self.voxel_um, 1)}})
        return out

    def crystals(self):
        """Compact solids (polyhedra, shards, superquadrics), placed without overlap, each its own object: a blob
        population whose tensors are the solids' second moments (rendered as points: no device synchronisation)."""
        r = self.r; dev = self.device; rad = self.size(0.01, 200, 1.0, 8.0)
        if rad is None:
            return None
        prim = ["polyhedron", "shard", "superquadric"][r.i(0, 2)]
        ax = torch.tensor([r.lu(0.6, 2.0), r.lu(0.6, 2.0), r.lu(0.6, 3.0)], device=dev)
        if prim == "shard":
            ax[r.i(0, 2)] = r.u(0.15, 0.35)
        ax = ax / ax.prod() ** (1 / 3) * rad
        dmin = 2.05 * float(ax.max()) * 1.15
        n_c = int(min(3000, r.u(0.3, 1.2) * np.prod(self.shape) / dmin ** 3))
        C = r.rand(max(n_c, 1), 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        D = torch.cdist(C, C); earlier = torch.ones_like(D, dtype=torch.bool).tril(-1)
        C = C[~((D < dmin) & earlier).any(1)][:600]; n = C.shape[0]
        sc = torch.exp(0.15 * r.randn(n, 1)).clamp(0.75, 1.15)
        A = _rot(r, n) @ torch.diag_embed(ax[None] * sc)
        nrm = Fn.normalize(torch.tensor(r.np.normal(size=(r.i(5, 12), 3)), dtype=torch.float32, device=dev), dim=1) if prim == "polyhedron" else None
        exps = (r.lu(0.3, 2.0), r.lu(0.3, 2.0))
        g = self._solids(C, A, "crystals", self.ids(n, float(ax.min()) * sc.reshape(-1)), shape={"shard": "superquadric"}.get(prim, prim), exps=exps, normals=nrm)
        self.obstacles.append((C, sc.reshape(-1) * float(ax.max())))
        self.record["families"].append({"crystals": {"primitive": prim, "n": n, "radius_vox": round(rad, 2)}})
        return g

    # ---- the added families: plywood lamellae, fascicle hierarchies, trabeculae, tubules, epithelia
    def plywood(self):
        """Stacked lamellae of parallel fibres whose direction turns from one lamella to the next (orthogonal plywood,
        90 degrees; twisted / Bouligand, a constant small angle): lamellar bone, cornea, cuticle. Unresolved fibrils:
        each lamella a slab of line tensors; alternate lamellae are separate populations."""
        r = self.r; dev = self.device
        tl = self.size(2, 40, 1.5, min(self.shape) / 8)
        if tl is None:
            return None
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        g_ = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in self.shape], indexing="ij"), -1)
        rel = g_ - torch.tensor(self.shape, device=dev, dtype=torch.float32) / 2
        L = int(2 * max(self.shape)); hgt = noise2d(L, L, r.lu(15, 50), r) * r.u(0, 6)
        gu = (rel @ e1 / (L / 2)).clamp(-1, 1); gv = (rel @ e2 / (L / 2)).clamp(-1, 1)
        hs = Fn.grid_sample(hgt[None, None], torch.stack([gv, gu], -1).view(1, 1, -1, 2), align_corners=True).view(self.shape)
        depth = rel @ n0 - hs
        k = torch.floor(depth / tl); frac = depth / tl - k
        dth = math.radians(90.0 if r.p(0.5) else r.u(10, 45)); th0 = r.u(0, math.pi)
        theta = th0 + k * dth
        d = torch.cos(theta)[..., None] * e1 + torch.sin(theta)[..., None] * e2                     # (Z, Y, X, 3)
        gap = r.u(0.05, 0.2); edge = 0.5 / tl
        occ = (((frac - gap / 2) / edge).clamp(0, 1) * ((1 - gap / 2 - frac) / edge).clamp(0, 1)) * r.u(0.4, 0.85)
        tens = line_tensor(d).movedim(-1, 0)
        rf_um = r.lu(0.05, 1.0); rf = rf_um / self.voxel_um
        desc = torch.zeros((5,) + self.shape, device=dev); desc[0] = rf
        contrast = self.contrast(); oid = self.ids(1, tl)
        out = []
        for par in (0, 1):
            m = (k.remainder(2) == par).float()
            out.append(Group("lamellae", "line", rf, contrast, dense=[(occ * m, tens, desc, contrast, torch.full(self.shape, int(oid[0]), device=dev, dtype=torch.long))]))
        self.record["families"].append({"plywood": {"lamella_vox": round(tl, 1), "turn_deg": round(math.degrees(dth), 1)}})
        return out

    def fascicles(self):
        """A hierarchy: fascicles (each a packed bundle of fibres, crimped) inside sheaths, packed side by side along a
        common axis -- tendon, muscle, nerve. Fibres -> fascicle -> the whole; the sheaths a hollow-tube population."""
        r = self.r; dev = self.device
        Rt = self.size(100, 6000, 4.0, min(self.shape) / 3)
        if Rt is None:
            return None
        nf = r.i(3, 12); Rf = Rt / math.sqrt(nf) * r.u(0.6, 0.85)
        d = r.unit(); c = self.centre_point(0.3)
        a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(d[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(d, a), dim=0); e2 = torch.linalg.cross(d, e1)
        # fascicle centres: a jittered hexagonal packing in the section
        sp = 2.15 * Rf; nn_ = int(Rt / sp) + 2
        ii, jj = torch.meshgrid(torch.arange(-nn_, nn_ + 1, device=dev), torch.arange(-nn_, nn_ + 1, device=dev), indexing="ij")
        o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp
        o = o[o.norm(dim=1) <= max(Rt - Rf, 0.0) + 1e-3][:nf]
        crimp_a, crimp_l = r.u(0, 0.15) * Rf, r.lu(4, 15) * Rf
        groups, sheath_curves = [], []
        wall = max(0.5, r.u(0.04, 0.12) * Rf); cs = self.contrast()
        tid = int(self.ids(1, Rt)[0])
        fib_curves = []
        for j in range(o.shape[0]):
            cen = c + o[j, 0] * e1 + o[j, 1] * e2
            g = self.bundle(None, "single", axis=d, centre=cen, radius=Rf - wall)
            if g is None or isinstance(g, list):
                continue
            # crimp: a sinusoidal wave of the whole fascicle in a random section direction
            ph = r.u(0, 6.28); wv = Fn.normalize(r.unit() - (r.unit() @ d) * d, dim=0)
            for cv in g.curves:
                sl = (cv.P - c) @ d
                cv.P = cv.P + (crimp_a * torch.sin(2 * math.pi * sl / crimp_l + ph))[..., None] * wv
                for o_ in cv.obj.tolist() + torch.unique(cv.seg).tolist():                     # (the fascicle's own node under the
                    self.parents.setdefault(o_, tid)                                            # whole: fibre -> fascicle -> tendon)
            fib_curves += g.curves
            Pc = g.centreline.P + (crimp_a * torch.sin(2 * math.pi * ((g.centreline.P - c) @ d) / crimp_l + ph))[..., None] * wv
            sid = self.ids(1, Rf); self.parents[int(sid[0])] = tid
            sheath_curves.append(curves_like(Pc, torch.full_like(Pc[..., 0], Rf), r, obj=sid, seg=sid, contrast=cs, wall=torch.full_like(Pc[..., 0], wall)))
        if not fib_curves:
            return None
        out = [Group("fascicle_fibres", "line", float(np.median([float(cv.rad.median()) for cv in fib_curves])), self.contrast(), curves=fib_curves)]
        if sheath_curves:
            out.append(Group("sheaths", "line", Rf, cs, curves=sheath_curves))
        self.record["families"].append({"fascicles": {"n": len(sheath_curves), "fascicle_vox": round(Rf, 1), "crimp": round(crimp_a / Rf, 2)}})
        return out

    def trabecular(self):
        """Trabecular bone-like walls from two smooth level sets: plates (near the zero set of one: sheets) and rods (near
        the intersection of two: lines along the cross product of their gradients), in complementary regions."""
        r = self.r; dev = self.device
        cell = self.size(200, 2000, 8.0, min(self.shape) / 2)
        if cell is None:
            return None
        p1 = smooth_noise(self.shape, cell / 3, r.g, dev); p2 = smooth_noise(self.shape, cell / 3, r.g, dev)
        g1 = torch.stack(torch.gradient(p1)); g2 = torch.stack(torch.gradient(p2))
        n1 = g1.norm(dim=0).clamp_min(1e-6); n2 = g2.norm(dim=0).clamp_min(1e-6)
        d1 = p1.abs() / n1; d2 = p2.abs() / n2
        hp = max(0.6, r.u(0.04, 0.12) * cell); hr = max(0.7, r.u(0.05, 0.15) * cell)
        reg = gaussian(r.rand(*self.shape), cell / 2); reg = (reg - reg.mean()) / reg.std().clamp_min(1e-6)
        plate_share = r.u(0.2, 0.8); thr = float(torch.quantile(reg.flatten()[::97], plate_share))
        plate_reg = torch.sigmoid((thr - reg) * 6)
        occ_p = (0.5 + hp - d1).clamp(0, 1) * plate_reg
        occ_r = (0.5 + hr - torch.sqrt(d1 ** 2 + d2 ** 2)).clamp(0, 1) * (1 - plate_reg)
        tp = sheet_tensor((g1 / n1).movedim(0, -1)).movedim(-1, 0)
        tr = line_tensor(Fn.normalize(torch.linalg.cross(g1, g2, dim=0), dim=0).movedim(0, -1)).movedim(-1, 0)
        dp = torch.zeros((5,) + self.shape, device=dev); dp[0] = hp
        dr = torch.zeros((5,) + self.shape, device=dev); dr[0] = hr
        oid = self.ids(1, hp); cst = self.contrast(); om = torch.full(self.shape, int(oid[0]), device=dev, dtype=torch.long)
        self.record["families"].append({"trabecular": {"cell_vox": round(cell, 1), "plate_share": round(plate_share, 2)}})
        return [Group("plates", "sheet", hp, cst, dense=[(occ_p, tp, dp, cst, om)]), Group("rods", "line", hr, cst, dense=[(occ_r, tr, dr, cst, om)])]

    def tubules(self):
        """Convoluted tubules (kidney-like): many tortuous epithelial tubes with a thick wall and an open lumen."""
        r = self.r; dev = self.device
        rt = self.size(8, 60, 1.2, 8.0)
        if rt is None:
            return None
        n = int(min(120, r.u(0.2, 0.6) * np.prod(self.shape) / (math.pi * rt ** 2 * 2 * max(self.shape)) + 3))
        L = torch.full((n,), 1.5 * max(self.shape), device=dev); S = int(1.5 * max(self.shape) / H)
        start = r.rand(n, 3) * torch.tensor(self.shape, device=dev, dtype=torch.float32)
        P = smooth_paths(start - 0.5 * L[:, None] * r.unit(n), r.unit(n), L, S, torch.full((n,), r.lu(0.05, 0.25) / rt ** 0.5, device=dev), r.lu(3, 8) * rt, r)
        rad = torch.full((n, S), rt, device=dev) * torch.exp(0.1 * r.randn(n, 1))
        cv = curves_like(P, rad, r, obj=self.ids(n, rad[:, 0]), seg=torch.zeros(n, dtype=torch.long, device=dev), contrast=self.contrast(),
                         wall=rad * r.u(0.3, 0.6), lumen_c=torch.full_like(rad, r.u(-50, 50)))
        cv = self.deflect(cv, extra=[(P[:, ::6].reshape(-1, 3), rad[:, ::6].reshape(-1))], self_ids=True)
        self.record["families"].append({"tubules": {"n": n, "radius_vox": round(rt, 2)}})
        return Group("tubules", "line", rt, float(cv.contrast[0, 0]), curves=[cv])

    def epithelium(self):
        """An epithelium: columnar cells packed on a folded surface (a basement membrane), elongated along its normal,
        with nuclei; the membrane a sheet population."""
        r = self.r; dev = self.device
        rc = self.size(3, 15, 1.2, 6.0)
        if rc is None:
            return None
        n0 = r.unit(); a = torch.tensor([1.0, 0, 0], device=dev) if abs(float(n0[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        e1 = Fn.normalize(torch.linalg.cross(n0, a), dim=0); e2 = torch.linalg.cross(n0, e1)
        c = self.centre_point(0.35); Lh = 0.75 * max(self.shape); sp = 2.1 * rc
        nu = int(2 * Lh / sp)
        ii, jj = torch.meshgrid(torch.arange(nu, device=dev), torch.arange(nu, device=dev), indexing="ij")
        uv = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * sp - Lh
        sf = r.lu(10, 40); amp = r.u(0, 0.3) * sf                                                   # folds up to ~60 deg steep
        kx, ky, ph = r.u(0.5, 1.5) / sf, r.u(0.5, 1.5) / sf, r.u(0, 6.28)
        hgt = amp * torch.sin(2 * math.pi * kx * uv[:, 0] + ph) * torch.cos(2 * math.pi * ky * uv[:, 1])
        dhu = amp * 2 * math.pi * kx * torch.cos(2 * math.pi * kx * uv[:, 0] + ph) * torch.cos(2 * math.pi * ky * uv[:, 1])
        dhv = -amp * 2 * math.pi * ky * torch.sin(2 * math.pi * kx * uv[:, 0] + ph) * torch.sin(2 * math.pi * ky * uv[:, 1])
        nrm = Fn.normalize(n0[None] - dhu[:, None] * e1 - dhv[:, None] * e2, dim=1)
        hc = rc * r.u(1.5, 4.0)                                                                     # half height of the columnar cells
        base = c + uv[:, :1] * e1 + uv[:, 1:] * e2 + hgt[:, None] * n0
        C = base + nrm * hc
        ok = ((C > -hc) & (C < torch.tensor(self.shape, device=dev) + hc)).all(1)
        C, nrm = C[ok][:700], nrm[ok][:700]; n = C.shape[0]
        if n == 0:
            return None
        t1 = Fn.normalize(torch.linalg.cross(nrm, e1[None].expand_as(nrm)), dim=1); t2 = torch.linalg.cross(nrm, t1)
        Rm = torch.stack([t1, t2, nrm], -1)                                                          # columns: in-plane, in-plane, normal
        ax = torch.stack([torch.full((n,), rc * 0.95, device=dev), torch.full((n,), rc * 0.95, device=dev), torch.full((n,), hc, device=dev)], 1)
        out = self._cell_groups(C, Rm, ax, rc)
        bm = self.sheet(normal=n0, centre=c, half_thickness=max(0.5, 0.15 * rc), height=(sf, amp, kx, ky, ph, e1, e2))
        if bm is not None:
            bm.name = "basement"; out.append(bm)
        self.record["families"].append({"epithelium": {"cells": n, "radius_vox": round(rc, 2), "height_vox": round(2 * hc, 1)}})
        return out

    # ---- fibres yield to cells and vessels
    def deflect(self, cv: Curves, iters: int = 4, margin: float = 0.5, extra=None, self_ids: bool = False):
        """Curves yield to obstacles (spheres ``(centres, radii)``): the scene's (cells, vessels) or ``extra`` ones only
        (a bundle kissing another; tubules among themselves, with ``self_ids`` -- ``extra`` sampled from these curves,
        every 6th point -- so no curve yields to itself)."""
        obs = extra if extra is not None else self.obstacles
        if not obs:
            return cv
        C = torch.cat([o[0] for o in obs]); R = torch.cat([o[1] for o in obs])
        P = cv.P.clone(); N, S, _ = P.shape
        rr = cv.rad.reshape(-1)
        owner_q = torch.arange(N, device=P.device).repeat_interleave(S)
        owner_c = torch.arange(N, device=P.device).repeat_interleave((S + 5) // 6) if self_ids else None
        for _ in range(iters):
            flat = P.reshape(-1, 3); push = torch.zeros_like(flat)
            if self_ids:
                C = P[:, ::6].reshape(-1, 3)
            rows = max(2048, int(1.5e8 / max(C.shape[0], 1)))                # ~0.6 GB per distance block, whatever the obstacle count
            for i0 in range(0, flat.shape[0], rows):
                q = flat[i0:i0 + rows]; d = torch.cdist(q, C)
                ov = (R[None] + rr[i0:i0 + rows, None] + margin - d).clamp_min(0)
                if self_ids:
                    ov = ov * (owner_q[i0:i0 + rows, None] != owner_c[None]).float()
                w = ov / d.clamp_min(1e-6)                                   # push = sum_c ov_c (q - C_c) / d_c
                push[i0:i0 + rows] = q * w.sum(1, keepdim=True) - w @ C
            P = P + push.view(N, S, 3)
            sm = smooth1d(P.reshape(N, S, 3), 1.5)                          # smooth along the curve (unit-variance kernel: re-normalise the mean)
            P = sm / smooth1d(torch.ones(N, S, 3, device=P.device), 1.5)
        cv.P = P
        return cv

    # ---- the scene
    def build(self, families: dict | None = None, n_groups: tuple = (1, 5), layout_p: float = 0.3, underlay_p: float = 0.4,
              cross_p: float = 0.0, deep_tree: bool = False, dense_p: float = 0.0):
        """The scene's families; with probability ``underlay_p`` a neuropil underlay as well (real tissue is full of it),
        unless the families given exclude neuropil. With probability ``cross_p`` a crossing scene: two or three bundles,
        each crossing (or kissing, or fanning through) the one before, beside the other families drawn. ``deep_tree``:
        objects in deeper hierarchies -- a resolved bundle's fibres in fascicles (fibre -> fascicle -> bundle), a tree's
        branches as objects under the branch they leave (twig -> branch -> ... -> trunk -> tree)."""
        r = self.r
        self.deep = bool(deep_tree)                                            # hierarchies below the population: fascicles, branches
        self.dense = dense_p > 0 and r.p(dense_p)                               # a dense scene: many interacting populations, heavier packing
        if self.dense:
            n_groups = (max(4, n_groups[0]), max(8, n_groups[1])); self.record["dense"] = True
        crossing = cross_p > 0 and r.p(cross_p)
        if r.p(layout_p):
            self.layout()
        fam = dict(self.FAMILIES); fam.update(families or {})
        names = [k for k, v in fam.items() if v > 0 and k != "tissue" and (k != "radial" or self.depth is not None)]   # (tissue: a scene, not a group)
        iso = r.p(ISO_SCENE_P) and fam.get("granules", 0) > 0                                  # nothing with a direction: granules (and crystals)
        if iso:
            names = [k for k in ("granules", "crystals") if fam.get(k, 0) > 0]
            self.record["iso_scene"] = True
        if not iso and (families is None or families.get("tissue", 0) > 0) and r.p(P_TISSUE * scale_weight("tissue", self.voxel_um)):
            for g_ in self.tissue(grey=r.p(P_GREY_TISSUE)):                                     # packed tissue (an explicit family list leaves
                self.groups.append(g_)                                                         # it out unless it names it); its crossings are
            n_groups = (0, 1); crossing = False; underlay_p = 0.0                              # inside it: one family or none on top
        p = np.array([fam[k] * scale_weight(k, self.voxel_um) for k in names], dtype=float); p /= p.sum()   # what belongs at this voxel size
        n = r.i(*n_groups)
        chosen = [str(c_) for c_ in r.np.choice(names, size=n, p=p)]
        order = {"cells": 0, "crystals": 0, "epithelium": 0, "nuclei": 0, "organs": 0, "organelles": 0, "granules": 0,
                 "tree": 1, "gland": 1, "network": 2}                                           # obstacles first, fibres after
        line_sheet = False
        if crossing:
            line_sheet = r.p(0.3)                                                  # also a line across a sheet (fibres through a membrane, a lamella)
            nb = r.i(1, 2) if line_sheet else r.i(2, 3); chosen = chosen[:max(0, n - nb)] + ["bundle"] * nb
            self.record["crossing_scene"] = nb + (2 if line_sheet else 0)
        chosen.sort(key=lambda k: order.get(k, 5))
        mat = []
        if not iso and MATRIX_P > 0 and r.p(MATRIX_P):                          # the matrix (29 Oct): the tissue around the
            mat = [str(m_) for m_ in r.np.choice(MATRIX_TYPES, size=2 if r.p(MATRIX_TWO_P) else 1, replace=False)]   # families, built
            self.record["matrix"] = mat                                         # of objects with their own ODFs (no painted
            if "cells" in mat:                                                  # background); cells first, so fibres avoid them
                for g_ in (self.cells() or []):
                    g_.matrix = True; self.groups.append(g_)
        bundles = []
        from .compose import population as _compose_population
        for name in chosen:
            if name == "bundle":
                prev = bundles[-1] if bundles and (crossing or r.p(0.6)) else None
                mode = ["angled", "angled", "kissing", "fanning"][r.i(0, 3)] if prev is not None else ("split" if r.p(0.15) else "single")
                rel_c = prev.contrast * r.lu(0.9, 1.1) if (prev is not None and r.p(P_RELATED_CONTRAST)) else None   # one material, one contrast
                gs = self.bundle(prev.centreline if prev is not None else None, mode, contrast=rel_c)
                from . import fibres as FB
                if gs is not None and FB.RELAX:                                    # fibres interweave in the relaxation (below)
                    bundles.extend(gs if isinstance(gs, list) else [gs])
                elif gs is not None:
                    for g_ in (gs if isinstance(gs, list) else [gs]):
                        if mode == "angled" and prev is not None and prev.offsets is not None and g_.offsets is not None and r.p(0.7):
                            self.interleave(prev, g_)                              # fascicles interdigitate where the bundles cross
                        if mode == "kissing" and prev is not None:                 # the newer bundle flattens against the other
                            cl = prev.centreline
                            g_.curves = [self.deflect(c, extra=[(cl.P[0, ::4], cl.rad[0, ::4])]) for c in g_.curves]
                        g_.curves = [self.deflect(c) for c in g_.curves]
                        bundles.append(g_)
            elif name == "compose":                                           # related to any earlier population with a centreline
                prev = [g_ for g_ in self.groups if getattr(g_, "centre_P", None) is not None or getattr(g_, "sdf", None) is not None]
                for g_ in self.groups:
                    if getattr(g_, "centre_P", None) is None and getattr(g_, "centreline", None) is not None:
                        g_.centre_P = g_.centreline.P[0]; prev.append(g_)
                gs = _compose_population(self, prev, dense=self.dense)
            elif name == "foam":
                gs = self.foam(False)
            elif name == "porous":
                gs = self.foam(True)
            elif name in ("dendrite", "gland"):
                gs = self.tree(name)
            else:
                gs = getattr(self, name)()
            if gs is None:
                continue
            for g_ in (gs if isinstance(gs, list) else [gs]):
                self.groups.append(g_)
        if crossing and r.p(0.4):                                               # a fan beside the crossing: the spread a readout that
            gf = self.bundle(None, "single", fan=r.u(0.4, 1.5))                 # splits crossings must not split
            for g_ in (gf if isinstance(gf, list) else [gf] if gf is not None else []):
                g_.curves = [self.deflect(c) for c in g_.curves]; self.groups.append(g_)
            self.record["fan_beside_crossing"] = True
        if line_sheet:                                                          # a sheet, and a bundle through it at 5-90 deg to its plane
            c = self.centre_point(); nrm = r.unit()
            sh = self.sheet(normal=nrm, centre=c)
            if sh is not None:
                for g_ in (sh if isinstance(sh, list) else [sh]):
                    self.groups.append(g_)
                a = torch.tensor([1.0, 0, 0], device=self.device) if abs(float(nrm[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=self.device)
                e1 = Fn.normalize(torch.linalg.cross(nrm, a), dim=0)
                beta = math.radians(r.u(5, 90)); ax = math.cos(beta) * e1 + math.sin(beta) * nrm
                gb = self.bundle(None, "single", axis=ax, centre=c)
                for g_ in (gb if isinstance(gb, list) else [gb] if gb is not None else []):
                    g_.cross = (c, e1, ax); g_.centre_pt = c; self.groups.append(g_)
                self.record["line_sheet_deg"] = round(math.degrees(beta), 1)
        if "neuropil" in mat:                                                   # neuropil: drawn first, so what lies in it
            gs = [g_ for g_ in (self.neuropil(fill=r.u(*MATRIX_FILL), r_min=MATRIX_R_MIN) or [])]   # displaces it (it fills the space between)
            for g_ in gs:
                g_.matrix = True
            self.groups[0:0] = gs
        if mat and GROUND_FILL > 0:                                             # the ground: fine fibrils filling the volume under
            gs = [g_ for g_ in (self.neuropil(fill=GROUND_FILL, name="ground", r_min=MATRIX_R_MIN, somata=False) or [])]   # everything,
            for g_ in gs:                                                       # drawn first (all else displaces it): no voxel of
                g_.matrix = True                                                # the specimen without material and an ODF
            self.groups[0:0] = gs; self.record["ground"] = True
        elif not iso and (families is None or families.get("neuropil", self.FAMILIES["neuropil"]) > 0) and r.p(underlay_p * scale_weight("neuropil", self.voxel_um)):
            gs = self.neuropil()                                                       # the underlay
            for g_ in (gs or []):
                self.groups.append(g_)
        if not self.groups:                                                    # at least one population
            g_ = self.bundle()
            if g_ is not None:
                self.groups.append(g_)
        from . import fibres as FB
        if FB.RELAX:                                                           # Monte Carlo fibre architecture: collisions resolved
            FB.relax_scene(self)
        return self

    def render(self, half: bool = False) -> dict:
        """All populations on the canvas: per group occupancy / tensor / desc (5, voxels and 0-1) / contrast map; the
        lumen contents; identities; branch and endpoint maps. ``half``: each population's fields kept in half precision
        as soon as it is rendered (the cache's form: a 256^3 canvas's float32 fields are ~0.9 GB a population)."""
        cv = Canvas(self.shape, self.device, getattr(self, "deform", None)); out = []
        assert len(self.groups) <= Canvas.NG, f"{len(self.groups)} populations > Canvas.NG"
        for gi, g_ in enumerate(self.groups):
            cv.group = gi
            if g_.layer is not None:
                occ, tens = render_layer(g_.layer, self.shape, self.r.g, self.device)
                d3 = g_.layer.desc
                desc = torch.stack([d3[0], torch.zeros_like(d3[0]), d3[1], torch.ones_like(d3[0]) * (occ > 0), d3[2]])
                cv.add_dense(occ, tens, desc, g_.contrast, torch.full(self.shape, int(self.ids(1)[0]), device=self.device, dtype=torch.long))
            for c in g_.curves:
                if c.spread is not None:
                    self.obj_spread.update(zip(c.obj.tolist(), [round(v, 2) for v in c.spread.tolist()]))
                render_curves(cv, c, max_points=2e6 if c.N > 1000 else 4e6)                          # (chunks: the splat's peak ~2 GB per 1e6 points;
                                                                                                     # the trainer's producer thread shares the GPU)
            for Q, W, V, obj in g_.points:                                      # (in chunks: a field of cells can be 10^7 points,
                for i in range(0, Q.shape[0], 4_000_000):                       # 8 corner keys each)
                    cv.splat(Q[i:i + 4_000_000], W[i:i + 4_000_000], V[i:i + 4_000_000], obj=obj[i:i + 4_000_000] if obj is not None else None)
            for occ, tens, desc, con, om in g_.dense:
                cv.add_dense(occ, tens, desc, con, om)
            res = cv.finish()
            if half:
                res = {k: v.half() for k, v in res.items()}
            res["group"] = g_; out.append(res)
        if self.obj_spread:                                                                     # the fibres' spread angles (degrees), inspectable:
            sv_ = np.array(list(self.obj_spread.values()))                                      # per object in ``obj_spread``, summarised here
            self.record["spread_deg"] = {"n": int(sv_.size), "q10": round(float(np.percentile(sv_, 10)), 2), "median": round(float(np.median(sv_)), 2),
                                         "q90": round(float(np.percentile(sv_, 90)), 2)}
        ids, frac = cv.identities(2)
        ent = cv.entries()                                                                      # (voxel, population, object): the entities' targets
        lum = cv.lumen.view(self.shape).clamp(0, 1); lum_c = cv.lumen_c.view(self.shape) / cv.lumen.view(self.shape).clamp_min(1e-9)
        maps = torch.zeros((2,) + self.shape, device=self.device)
        dfm = getattr(self, "deform", None)
        moved = [(dfm.forward(torch.cat(pts))[0] if (dfm is not None and pts) else (torch.cat(pts) if pts else None)) for pts in (self.branch, self.ends)]
        for k, Pp in enumerate(moved):                                                          # (branch points and ends move with the specimen)
            if Pp is not None:
                idx = Pp.round().long()
                ok = ((idx >= 0) & (idx < torch.tensor(self.shape, device=self.device))).all(1)
                if bool(ok.any()):
                    maps[k].index_put_((idx[ok, 0], idx[ok, 1], idx[ok, 2]), torch.ones(int(ok.sum()), device=self.device), accumulate=True)
                    maps[k] = gaussian(maps[k], 1.0) * (2 * math.pi) ** 1.5
        return {"groups": out, "ids": ids, "id_frac": frac, "lumen": lum, "lumen_c": lum_c, "points": maps, "entries": ent,
                "point_lists": tuple(m_ if m_ is not None else torch.zeros(0, 3, device=self.device) for m_ in moved)}


# ------------------------------------------------------------------------------------------------------------------
# a training pair: a crop of a rendered scene, composited, acquired
# ------------------------------------------------------------------------------------------------------------------

import threading

_TLS = threading.local()                   # one scene cache per thread: the background producer and the main thread render apart
VIS_LADDER = (1, 2, 4, 8)                  # visibility evidence at the shapes' scale and up to 8x it (64 voxels): interiors count
VIS_WINDOWS = (1, 2, 4)                    # ... in windows of 5, 10 and 20 voxels: a crossing's featureless core counts (context)
CACHE_SCENES = 2                           # a 256^3 canvas's render is several GB: a scene's crops are drawn in sequence anyway


def _scene_cache() -> dict:
    if not hasattr(_TLS, "cache"):
        _TLS.cache = {}
    return _TLS.cache


class _CacheProxy:
    """``_CACHE.clear()`` clears the calling thread's scene cache (kept for scripts that call it)."""
    def clear(self):
        _scene_cache().clear()


_CACHE = _CacheProxy()


def _compact(ren: dict) -> dict:
    """A render for the cache: fields in half precision, identities as int32 (a 256^3 canvas's render is several GB in
    float32); :func:`draw_scene` restores the types on the crop."""
    out = dict(ren)
    out["groups"] = [{k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in g.items()} for g in ren["groups"]]
    out["ids"], out["id_frac"] = ren["ids"].int(), ren["id_frac"].half()
    out["lumen"], out["lumen_c"] = ren["lumen"].half(), ren["lumen_c"].half()
    vox, grp, obj, w, v = ren["entries"]
    out["entries"] = (vox.int(), grp.to(torch.int8), obj.int(), w.half(), v.half())
    return out


def render_scene(canvas, scene_seed: int, device, voxel_um=None, families=None, n_groups=(1, 5), layout_p=0.3, content="general", style=None,
                 underlay_p: float = 0.4, cross_p: float = 0.0, deep_tree: bool = False, dense_p: float = 0.0, p_warp: float = 0.0,
                 p_affine: float = 0.0, legacy_deform: bool = False):
    """The scene (cached: crops of one scene are drawn in sequence) and its background on the whole canvas -- a
    texture, regions and possibly a cell stain, so crops at any zoom of one scene share them. With probability
    ``p_warp`` / ``p_affine`` the specimen is deformed (:class:`Deform`): the scene renders deformed, its background
    and outside mask pulled back with it."""
    key = (tuple(canvas), scene_seed, str(device), voxel_um, str(families), tuple(n_groups), layout_p, content, style, underlay_p, cross_p, deep_tree, dense_p,
           p_warp, p_affine, legacy_deform)
    _CACHE = _scene_cache()
    if key in _CACHE:
        _CACHE[key] = _CACHE.pop(key)                                       # most recently used last
    else:
        while len(_CACHE) >= CACHE_SCENES:                                  # a few scenes (training draws and paired renders interleave)
            _CACHE.pop(next(iter(_CACHE)))
        sc = Scene(canvas, scene_seed, device, voxel_um, content).build(families, n_groups, layout_p, underlay_p, cross_p, deep_tree, dense_p)
        st = style or ("flat" if sc.record.get("matrix") else str(sc.r.np.choice(["flat", "regions", "tissue", "cells"], p=BG_STYLE_P)))   # (a matrix:
                                                                            # the structure is its objects', the background plain)
        bgp = background(tuple(canvas), st, sc.r.g, device, 1.0, content, nprng=sc.r.np, natural=True)
        dw, da = (p_warp > 0 and sc.r.p(p_warp)), (p_affine > 0 and sc.r.p(p_affine))
        sc.deform = Deform(tuple(canvas), sc.r, dw, da, device, cfg=DEFORM_LEGACY if legacy_deform else None) if (dw or da) else None
        if sc.deform is not None:                                           # the background deforms with the specimen
            dfm = sc.deform; bg0, lab0, R0, extra0 = bgp
            for E in extra0:
                E.occupancy = dfm.pull(E.occupancy.float())
                tt = dfm.pull(E.tensor.float())
                E.tensor = dfm.push_tensor(tt.reshape(6, -1).T, dfm.F_src).T.reshape(6, *tuple(canvas))
            bgp = (dfm.pull(bg0.float(), padding="border"), dfm.pull(lab0.float(), mode="nearest", padding="border").to(lab0.dtype),
                   dfm.pull(R0.float(), padding="border"), extra0)                         # (the background goes on past the canvas)
            if sc.outside is not None:
                sc.outside = dfm.pull(sc.outside.float(), padding="border") > 0.5
            sc.record["deform"] = dfm.record
            for g_ in sc.groups:                                                    # crops centre where the deformation took them
                if getattr(g_, "centre_pt", None) is not None:
                    g_.centre_pt_def = dfm.forward(g_.centre_pt.float().view(1, 3))[0][0]
        _CACHE[key] = (sc, _compact(sc.render(half=True)), bgp, st)
        for g_ in sc.groups:                                                    # the cache keeps the render, not what made it: a packed
            g_.dense = []; g_.paths = None                                      # scene's dense parts (~0.9 GB each at 256^3) and the
        if sc.deform is not None:                                               # deformation's fields (the crop choice needs only `valid`)
            sc.deform.W = sc.deform.G = sc.deform.src = sc.deform.F_src = sc.deform.st_src = None
    return _CACHE[key]


def _pool_fields(occ, fields: list, zoom: int):
    """Average ``occ (Z, Y, X)`` over ``zoom``³ blocks, and occupancy-weighted averages of ``fields`` (each ``(C, Z, Y, X)``):
    the same population rendered at a voxel ``zoom`` times larger (partial volume is linear)."""
    if zoom == 1:
        return occ, fields
    po = Fn.avg_pool3d(occ[None, None], zoom)[0, 0]
    out = [Fn.avg_pool3d((f * occ[None])[None], zoom)[0] / po.clamp_min(1e-9)[None] for f in fields]
    return po, out


def _crop(x, sl, zoom: int, mode: str = "mean"):
    """A crop of ``x (..., Z, Y, X)`` at slices ``sl`` reduced by ``zoom`` (``mean``: block average; ``pick``: the block's
    first voxel -- labels)."""
    x = x[(Ellipsis,) + tuple(sl)]
    if zoom == 1:
        return x
    if mode == "pick":
        return x[(Ellipsis,) + (slice(None, None, zoom),) * 3]
    lead = x.shape[:-3]
    return Fn.avg_pool3d(x.reshape(1, -1, *x.shape[-3:]).float(), zoom)[0].reshape(*lead, *[n // zoom for n in x.shape[-3:]])


def _point_map(pts: torch.Tensor, origin, zoom: int, shape, device) -> torch.Tensor:
    """Branch points / endpoints within a crop as unit-peak Gaussians (sigma 1 voxel of the crop)."""
    m = torch.zeros(shape, device=device)
    if pts.shape[0]:
        idx = ((pts - torch.tensor(origin, device=device, dtype=torch.float32)) / zoom).round().long()
        ok = ((idx >= 0) & (idx < torch.tensor(shape, device=device))).all(1)
        if bool(ok.any()):
            m.index_put_((idx[ok, 0], idx[ok, 1], idx[ok, 2]), torch.ones(int(ok.sum()), device=device), accumulate=True)
            m = gaussian(m, 1.0) * (2 * math.pi) ** 1.5
    return m


def microstructure_texture(shape, rng: Rng, device) -> torch.Tensor:
    """Unresolved microstructure (cells, neuropil, capillaries below the voxel) as image texture: a dense, isotropic,
    fine-grained field -- sub-voxel blobs (a Poisson process of random density and brightness, blurred by a fraction
    of a voxel to a voxel and a half) mixed with fine-scale noise; zero mean, unit spread, random polarity. Real 20 um
    HiP-CT cortex is this kind of texture everywhere (locally incoherent, power at high frequencies); smooth noise
    backgrounds are not (the domain-gap statistics, 23 Oct)."""
    rho = rng.lu(0.003, 0.3)                                                                   # up to a dense, continuous speckle
    pts = (rng.rand(*shape) < rho).float() * torch.exp(0.5 * rng.randn(*shape))
    blobs = gaussian(pts, rng.u(0.4, 1.2), truncate=3.0)
    blobs = (blobs - blobs.mean()) / blobs.std().clamp_min(1e-6)
    fine = smooth_noise(shape, rng.u(0.0, 1.0), rng.g, device)
    fine = (fine - fine.mean()) / fine.std().clamp_min(1e-6)
    w = rng.u(0, 1)
    t = w * blobs + (1 - w) * fine
    if rng.p(0.4):                                                                              # or power-law (1/f^beta) texture, beta
        t = rng.u(0.3, 1.0) * power_law_noise(shape, rng.u(0.5, 3.5), rng, device) + (1 - rng.u(0.3, 1.0)) * t   # 0.5-3.5: from speckle to
    t = (t - t.mean()) / t.std().clamp_min(1e-6)                                                # cloud (synthetic, every spectrum
    return t if rng.p(0.5) else -t                                                              # in between -- 27 Oct)


def power_law_noise(shape, beta: float, rng: Rng, device) -> torch.Tensor:
    """Isotropic Gaussian noise with power spectrum ~ |k|^-beta (white noise filtered in Fourier space), zero mean, unit
    spread."""
    k = [torch.fft.fftfreq(n, device=device) for n in shape[:-1]] + [torch.fft.rfftfreq(shape[-1], device=device)]
    K = torch.sqrt(sum(g ** 2 for g in torch.meshgrid(*k, indexing="ij"))).clamp_min(1.0 / max(shape))
    t = torch.fft.irfftn(torch.fft.rfftn(rng.randn(*shape)) * K ** (-beta / 2), s=shape)
    return (t - t.mean()) / t.std().clamp_min(1e-6)


def sample_mask(shape, rng: Rng, device) -> torch.Tensor:
    """A specimen's edge across a crop: a gently curved surface (a plane with smooth relief, or a sphere's cap) with
    10-60 % of the crop beyond it; 1 inside the tissue, 0 outside, a one-voxel soft edge."""
    g = torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape], indexing="ij"), -1)
    c = torch.tensor([n / 2 for n in shape], device=device)
    n = rng.unit(); out_frac = rng.u(0.1, 0.6)
    if rng.p(0.6):
        h = (g - c) @ n
        relief = rng.u(0, 8) * smooth_noise(shape, rng.lu(10, 40), rng.g, device)
        d = h + relief
    else:                                                                    # the cap of a large sphere
        R0 = rng.lu(1.0, 6.0) * max(shape); ctr = c - n * R0
        d = R0 - (g - ctr).norm(dim=-1)
    thr = float(torch.quantile(d.flatten()[:: max(1, d.numel() // 100000)], 1 - out_frac))
    return torch.sigmoid((thr - d) / 0.5)


def cutout_mask(shape, rng: Rng, device) -> torch.Tensor:
    """Tissue removed from the crop -- one to four regions: a block cut away or a missing tile (a rotated cuboid), a
    bubble or a hole (an ellipsoid), a knife's gap or a section lost from a block (a planar slab through the crop).
    1 where tissue remains, 0 where it was removed, a soft one-voxel edge. The medium fills the gap and the targets go
    with the tissue (:func:`draw_scene`)."""
    g = torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape], indexing="ij"), -1)
    sz = torch.tensor(shape, device=device, dtype=torch.float32); keep = torch.ones(shape, device=device); kinds = []
    for _ in range(rng.i(1, 4)):
        kind = ["cuboid", "ellipsoid", "slab"][int(rng.np.choice(3, p=[0.4, 0.35, 0.25]))]
        c = rng.rand(3) * sz; Rm = _rot(rng, 1)[0]; rel = (g - c) @ Rm
        if kind == "cuboid":
            h = torch.tensor([rng.u(0.05, 0.3) * float(n) for n in shape], device=device)
            q = rel.abs() - h; d = q.clamp_min(0).norm(dim=-1) + q.max(-1).values.clamp_max(0)    # signed distance to the box
        elif kind == "ellipsoid":
            ra = torch.tensor([rng.u(0.05, 0.3) * float(n) for n in shape], device=device)
            d = ((rel / ra).norm(dim=-1) - 1) * float(ra.min())
        else:
            d = rel[..., 0].abs() - rng.u(1.0, 0.08 * float(min(shape)))
        keep = keep * torch.sigmoid(d / 0.5); kinds.append(kind)
    return keep, kinds


def _slice_profile(img, rng: Rng, rec: dict):
    """Native anisotropic acquisition: slices of thickness ``t`` (a Gaussian or box profile) every ``sp`` voxels along a
    random axis (a gap when ``sp > t``), at a random phase, interpolated back to the grid."""
    ax = rng.i(0, 2); t = rng.lu(1.5, 6.0); sp = t * rng.u(0.7, 1.6); n = img.shape[ax]
    x = img.movedim(ax, -1)
    if rng.p(0.5):                                                          # Gaussian profile (FWHM t)
        x = gaussian(x, t / 2.355, truncate=3.0, axes=(-1,))
        prof = "gauss"
    else:                                                                   # box profile
        k = max(1, int(round(t))); x = Fn.avg_pool1d(Fn.pad(x.reshape(-1, 1, n), (k // 2, (k - 1) // 2), mode="replicate"), k, stride=1).reshape(x.shape)
        prof = "box"
    pos = torch.arange(rng.u(0, sp), n - 1, sp, device=img.device)
    if pos.numel() < 2:
        return img
    i0 = pos.floor().long().clamp(0, n - 2); w = pos - i0
    samp = x[..., i0] * (1 - w) + x[..., i0 + 1] * w                         # (..., m) the acquired slices
    grid = torch.arange(n, device=img.device, dtype=torch.float32)
    j = torch.searchsorted(pos, grid).clamp(1, pos.numel() - 1)
    if rng.p(0.3):                                                          # nearest-slice resampling
        near = torch.where((grid - pos[j - 1]).abs() < (pos[j] - grid).abs(), j - 1, j)
        y = samp[..., near]
    else:
        ww = ((grid - pos[j - 1]) / (pos[j] - pos[j - 1])).clamp(0, 1)
        y = samp[..., j - 1] * (1 - ww) + samp[..., j] * ww
    rec["slice_profile"] = {"axis": ax, "thick": round(t, 2), "spacing": round(sp, 2), "profile": prof}
    return y.movedim(-1, ax).contiguous()


def _extra_acquisition(img, layers, rng: Rng, modality: str, rec: dict, strength: float = 1.0):
    """Artefacts the acquisition model lacks: a CT scan of the content (:mod:`~shapetensor.learn.ctsim`: streaks from
    dense material, dark bands, halos, photon starvation -- CT / HiP-CT, rarely elsewhere), native slice profiles (MRI,
    histology, EM sections, fluorescence's axial sampling), section tears (tissue missing along a crack, labels removed
    with it) and folds (a band of a section doubled), local-tomography truncation (low frequencies lost) for HiP-CT /
    CT. Each scaled by the acquisition ``strength`` (0: none -- the curriculum's clean start)."""
    shp = img.shape; sg = float(min(max(strength, 0.0), 1.0))
    if modality in ("hipct", "ct", "mixed") and rng.p(0.3 * sg) or rng.p(0.03 * sg):        # a scan from the content: streaks, bands, halos
        from .ctsim import scan
        img = scan(img, rng, rec, sg)
    if modality in ("mri", "histology", "mixed", "em", "fluorescence") and rng.p(0.35 * sg):
        img = _slice_profile(img, rng, rec)
    if modality in ("histology", "mixed", "em") and rng.p(0.3 * sg):
        k = rng.i(1, 3); a = rng.i(0, shp[0] - 1); n = noise2d(shp[1], shp[2], rng.lu(6, 20), rng)
        crack = (n.abs() < rng.u(0.02, 0.08)).float()
        sl = slice(max(0, a - k), min(shp[0], a + k))
        img[sl] = img[sl] * (1 - crack) + float(img.min()) * crack
        for L in layers:
            L.occupancy[sl] = L.occupancy[sl] * (1 - crack)
        rec["tear"] = (a, k)
    if modality in ("histology", "mixed", "em") and rng.p(0.2 * sg):
        a = rng.i(1, shp[0] - 2); b0 = rng.i(0, shp[1] - 8); w = rng.i(3, 8)
        if b0 + 2 * w <= shp[1]:                                            # the neighbouring band folded over it
            img[a, b0:b0 + w] = img[a, b0:b0 + w] + 0.6 * img[a, b0 + w:b0 + 2 * w]
        rec["fold"] = (a, b0, w)
    if modality in ("hipct", "ct", "mixed") and rng.p(0.3 * sg):
        s = rng.lu(10, 40); alpha = rng.u(0.2, 0.8)
        img = img - alpha * (gaussian(img, s) - img.mean())
        rec["local_tomography"] = (round(s, 1), round(alpha, 2))
    return img


#: the HiP-CT profile's grain (unresolved microstructure) amplitude, in units of the image's 5-95 % range: drawn in
#: [lo x strength, lo + span x strength] (0.1-0.4 at full strength, 0-0.1 at strength 0)
HIPCT_GRAIN = {"lo": 0.1, "span": 0.3}


BG_TEX_NORM = 40.0                # grey levels: the background's texture is normalised by max(its spread, this) before it modulates
                                  # the background material (29 Oct)


def _material_contrast(mat: dict, bg: dict, ra: "Rng", stain: bool = False) -> dict:
    """A population's material made to stand out (29 Oct): with probability :data:`P_VISIBLE_STRANDS` its largest relative
    difference from the background material (over the keys both carry, the background's nonzero ones) is raised to at
    least 0.3-1.0; a background stain (``stain``) is brought within 10-30 % of the background (a stain, not the brightest
    thing there)."""
    keys = [k for k, v in mat.items() if k in bg and not isinstance(v, (bool, str))]
    if not keys:
        return mat
    arr = lambda v: np.atleast_1d(np.asarray(v, dtype=float))
    out = dict(mat)
    rel = max(float(np.max(np.abs(arr(mat[k]) - arr(bg[k])) / np.maximum(np.abs(arr(bg[k])), 1e-3))) for k in keys if np.any(arr(bg[k]) != 0)) \
        if any(np.any(arr(bg[k]) != 0) for k in keys) else 1.0
    if stain:                                                                                  # a stain: at most 10-30 % from the background
        scale = min(1.0, ra.u(0.1, 0.3) / max(rel, 1e-3))
    elif ra.p(P_VISIBLE_STRANDS):
        want = ra.u(0.3, 1.0)
        if rel >= want:
            return mat
        scale = want / max(rel, 1e-3)
    else:
        return mat
    for k in keys:
        b, v = arr(bg[k]), arr(mat[k])
        if not np.any(b != 0):
            continue
        nv = b + scale * (v - b)
        nv = np.where(b > 0, np.maximum(nv, 0.02 * b), nv)                                  # (densities stay positive)
        out[k] = float(nv[0]) if np.ndim(mat[k]) == 0 else type(mat[k])(nv.tolist()) if isinstance(mat[k], (list, tuple)) else nv
    return out


def _hipct_acquisition(img, rng: Rng, rec: dict, strength: float = 1.0):
    """The HiP-CT profile's extra artefact: local-tomography truncation (the scan's field of view inside the specimen:
    low frequencies lost), its probability and depth scaled by the strength."""
    sg = float(min(max(strength, 0.0), 1.0))
    if rng.p(0.4 * sg):
        s = rng.lu(10, 40); alpha = rng.u(0.1, 0.6) * sg
        img = img - alpha * (gaussian(img, s) - img.mean())
        rec["local_tomography"] = (round(s, 1), round(alpha, 2))
    return img


def _log_desc(d5: torch.Tensor) -> torch.Tensor:
    """The scene's 5 descriptors (size, wall in voxels; lumen fraction, flattening, angularity) as learned:
    ``log2(1 + size)``, the wall relative to the size, then the three fractions."""
    return torch.stack([torch.log2(1 + d5[0].clamp_min(0)), (d5[1] / d5[0].clamp_min(1e-3)).clamp(0, 1), d5[2], d5[3], d5[4]])


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """A cube max filter of side ``k`` on ``x (Z, Y, X)``, separably (three 1-D passes: max_pool3d is slow on a CPU)."""
    if k <= 1:
        return x
    r = k // 2
    for ax in range(3):
        xp = Fn.pad(x.movedim(ax, -1), (r, r)); y = xp[..., :x.shape[ax]]
        for s_ in range(1, k):
            y = torch.maximum(y, xp[..., s_:s_ + x.shape[ax]])
        x = y.movedim(-1, ax)
    return x


def _trim_paths(P: torch.Tensor) -> torch.Tensor:
    """Paths ``(N, S, 3)`` that froze (repeated points: no material) shifted to start at their first moving sample and cut
    to the longest moving stretch -- the renderer sweeps every sample, frozen or not."""
    if P is None or P.shape[0] == 0:
        return P
    mv = (P[:, 1:] - P[:, :-1]).norm(dim=-1) > 1e-6                                              # (N, S - 1)
    if not bool(mv.any()):
        return P[:, :2]
    S = P.shape[1]; ar = torch.arange(S - 1, device=P.device)
    first = torch.where(mv, ar, torch.full_like(ar, S)).amin(1).clamp(max=S - 2)
    last = torch.where(mv, ar, torch.full_like(ar, -1)).amax(1) + 1
    L = int((last - first).max()) + 1
    idx = (first[:, None] + torch.arange(L, device=P.device)[None]).clamp(max=S - 1)
    idx = torch.minimum(idx, last[:, None].clamp_min(first[:, None]))                            # (repeats past the end: no material)
    return P.gather(1, idx[..., None].expand(-1, -1, 3))


#: crop strata by oriented-tissue fill (the share of the crop within ~2 voxels of oriented material): packed, partial,
#: sparse, none. ``draw_scene(fill_quota=(0.3, 0.3, 0.2, 0.2))`` draws a stratum per crop and picks, of 8 random crops of
#: the scene, the first in it (else the nearest) -- white matter is fibres wall to wall, which crops of isolated bundles
#: never showed (27 Oct: oriented fill median 0 % of a crop, 8 % of crops over 70 %).
FILL_STRATA = ((0.7, 1.01), (0.3, 0.7), (0.03, 0.3), (0.0, 0.03))
#: the training's default quota (train.py passes it with the scene generator)
FILL_QUOTA = (0.35, 0.35, 0.2, 0.1)   # 29 Oct: was (0.3, 0.3, 0.2, 0.2)
BG_STYLE_P = (0.4, 0.15, 0.25, 0.2)   # the scene background's style: flat, regions (compartments), tissue, cells (29 Oct; was uniform)
CENTRE_FILL = 0.3                     # a centred crop is moved (within a third of the crop) to hold at least this much object material


def _oriented_extent(sc, ren, device) -> torch.Tensor:
    """Where oriented material lies on the canvas (bool), from the rendered populations of kind line or sheet."""
    occ = None
    for res in ren["groups"]:
        g_ = res["group"]
        if getattr(g_, "kind", None) in ("line", "sheet"):
            o_ = res["occupancy"].float(); occ = o_ if occ is None else occ + o_
    if occ is None:
        return torch.zeros(sc.shape, dtype=torch.bool, device=device)
    if getattr(sc, "outside", None) is not None:                                                  # (beyond a cortical layout's pial surface:
        occ = occ * (~sc.outside).float()                                                         # nothing, as the crop's mask makes it)
    return gaussian(occ[None, None], 2.0)[0, 0] > 0.15


def draw_scene(shape=(64, 64, 64), seed: int = 0, device="cuda", modality: str | None = None, strength: float = 1.0,
               canvas: int = 128, crops_per_scene: int = 8, voxel_um: float | None = None, families: dict | None = None,
               n_groups=(1, 5), layout_p: float = 0.3, content: str = "general", style: str | None = None, second_contrast: bool = False,
               zoom: int = 1, physics: str | None = None, p_physics: float = 0.75, p_warp: float = 0.6, acq_seed: int | None = None,
               crop: tuple | None = None, identity_min_r: float = 1.5, p_blend: float = 0.35, p_aniso: float = 0.25, p_scramble: float = 0.3,
               p_texture: float = 0.5, p_merge: float = 0.25, p_boundary: float = 0.2, p_affine: float = 0.5, p_grain: float = 0.8,
               underlay_p: float = 0.4, p_sparse: float = 0.12, p_cutout: float = 0.15, entity_ctx: bool = False, cross_p: float = 0.0,
               p_centre: float = 0.0, deep_tree: bool = False, dense_p: float = 0.0, scene_seed: int | None = None,
               fill_quota: tuple | None = None, profile: str | None = None):
    """A training pair from an object scene, in the format of :func:`~shapetensor.learn.generator.draw`: ``(Sample,
    drawn)``. Scenes are rendered on a ``canvas``³ grid and ``crops_per_scene`` consecutive seeds share one (cut at
    different places), so a scene's cost is spread over its crops.

    * ``zoom``: the crop covers ``zoom`` times the extent, every field block-averaged (the same scene at a voxel
      ``zoom`` times larger -- multi-scale crops; paired renders with :func:`draw_pair`);
    * ``physics``: the image formation (:mod:`~shapetensor.learn.physics`; None: drawn for the modality with
      probability ``p_physics``, else additive);
    * ``p_warp`` / ``p_affine``: the specimen deformed (per scene, :class:`Deform`) -- an elastic warp / an affine of
      the geometry itself, so labels, identities, points, tensors and descriptors move with it exactly (the crop-level
      warp it replaces mirrored the crop's edge into fake crossings);
    * ``acq_seed``: the acquisition's seed (modality, physics, materials, nuisance) apart from the crop's;
    * contrast routes at image formation: ``p_blend`` -- a blend with a second physics (either polarity), ``p_aniso`` -- an
      orientation-dependent term (``sin^2`` or ``cos^2`` of each population's angle to a random direction, its own
      amplitude and sign) under any physics, ``p_scramble`` -- each population and background region its own offset.
      With :mod:`~shapetensor.learn.augment` at batch time, one geometry reaches the network through many contrasts;
    * before visibility is measured (so the targets ask only for what shows): ``p_texture`` -- each population's material
      varies inside it (its own grain and depth); ``p_merge`` -- populations as intense as the background or as another
      (SynthSeg's shared classes); ``p_boundary`` -- the specimen's edge (medium beyond it, nothing to find there);
      ``p_affine`` -- a global shear and anisotropic scale with the warp (tensors, identities and descriptors follow);
      ``p_grain`` -- unresolved microstructure as dense isotropic fine texture under the tissue (no orientation: the
      targets do not change, the visibility measured after it); ``p_sparse`` -- sparse labelling: 1-30 % of the objects
      visible, the rest gone from image and targets alike (fluorescent reporters, Golgi stains, contrast-filled vessels);
      ``p_cutout`` -- chunks of tissue removed (:func:`cutout_mask`: blocks, holes, a knife's gap), the medium in their
      place and nothing to find there.

    ``profile="hipct"`` (opt-in; None: everything above, unchanged): the acquisition is synchrotron HiP-CT only -- X-ray
    phase-contrast image formation (tied attenuation and phase, a Paganin low-pass of 0.5-2 voxels, small residual
    fringes; population contrast :data:`~shapetensor.learn.physics.HIPCT_CONTRAST`), no blends with other physics, no
    orientation-dependent contrast, no per-population offsets (scramble); texture inside populations (at most 0.2), shared
    intensities (merge) and the background's texture scaled by the strength; grain at 0.1-0.4 of the range (0-0.1 at
    strength 0, :data:`HIPCT_GRAIN`); in :func:`~shapetensor.learn.generator.nuisance` an isotropic PSF, low-pass noise,
    rings, fringes, seams and a bias (:data:`~shapetensor.learn.generator.HIPCT_PROFILE`); of the extra artefacts only
    local-tomography truncation. Strength 0 is nearly clean. Batch-time routes: ``augment.PROFILE = "hipct"``.

    With ``entity_ctx``, ``drawn["entity_ctx"]`` holds what :func:`~shapetensor.learn.entities.entity_targets` needs: the
    canvas's (voxel, population, object) entries, its tree, the crop's origin / zoom, each population's layer, the warp.

    ``drawn`` also carries ``ids``/``id_frac`` (the two heaviest objects per voxel), ``identity`` (the object at the level
    the resolution separates: smaller than ``identity_min_r`` voxels -> its parent), ``descriptors5`` (per layer:
    log2(1 + size), relative wall, lumen fraction, flattening, angularity), ``desc_um``, ``points`` (branch points,
    endpoints), ``recoverability``, ``parents``, the voxel size and the physics."""
    sid = seed // max(1, crops_per_scene) if scene_seed is None else int(scene_seed)            # ``scene_seed``: another crop of that scene
    if profile == "hipct":                                                    # the HiP-CT profile: no oriented-texture family (29 Oct)
        families = {**(families if families is not None else {**Scene.FAMILIES, "tissue": 1.0}), "texture": 0.0}
    if profile == "hipct" and p_warp == 0.6 and p_affine == 0.5:              # the HiP-CT profile keeps the deformation of before
        p_warp, p_affine = 0.5, 0.3                                           # 29 Oct (the defaults' widening is the universal path's)
    sc, ren, bgp, st = render_scene((canvas,) * 3, 7_000_000 + sid, device, voxel_um, families, n_groups, layout_p, content, style, underlay_p, cross_p, deep_tree, dense_p,
                                    p_warp, p_affine, profile == "hipct")    # the specimen's deformation, per scene (Deform)
    dfm = getattr(sc, "deform", None)
    vfrac = (lambda oc: float(dfm.valid[tuple(slice(a, a + e) for a, e in zip(oc, ext))].float().mean())) if dfm is not None else (lambda oc: 1.0)
    r = Rng(seed, device); ra = Rng(acq_seed if acq_seed is not None else seed + 31337, device)
    rg = Rng(seed + 90001, device)                                            # specimen geometry (sparse labels, edges, cut-outs): the scene's,
                                                                              # never the acquisition's, so an acquisition only changes appearance
    from .generator import MODALITIES
    names = list(MODALITIES); modality = modality or names[ra.i(0, len(names) - 1)]          # X-ray, MRI, histology, EM, fluorescence, cryo-ET, mixed
    hip = profile == "hipct"; sg_ = float(min(max(strength, 0.0), 1.0))
    if profile not in (None, "hipct"):
        raise ValueError(f"unknown acquisition profile {profile!r}")
    if hip:                                                                   # HiP-CT only: texture inside populations and shared
        modality = "hipct"                                                    # intensities scale with the strength; no blends, no
        p_texture, p_merge, p_scramble, p_blend, p_aniso = p_texture * sg_, 0.5 * p_merge * sg_, 0.0, 0.0, 0.0   # sin^2 contrast, no offsets
                                                                              # (scramble) -- one phase map
    ext = [s_ * zoom for s_ in shape]
    assert all(e <= canvas for e in ext), "the zoomed crop must fit the canvas"
    o = list(crop) if crop is not None else [r.i(0, canvas - e) for e in ext]
    if crop is None and dfm is not None:                                      # inside the deformed canvas (no material was drawn
        best_o = (vfrac(o), o)                                                # beyond it)
        for _ in range(8):
            if best_o[0] >= 0.999:
                break
            oc = [r.i(0, canvas - e) for e in ext]; v_ = vfrac(oc)
            if v_ > best_o[0]:
                best_o = (v_, oc)
        o = best_o[1]
    centred = False
    if crop is None and p_centre > 0 and r.p(p_centre):                        # centred on a crossing (or a bundle's fan / split / bend)
        cr = [g_.centre_pt for g_ in sc.groups if getattr(g_, "cross", None) is not None]
        ce = [g_.centre_pt for g_ in sc.groups if getattr(g_, "centre_pt", None) is not None]
        pick = cr if (cr and (not ce or r.p(0.75))) else ce
        if pick:
            P0 = pick[r.i(0, len(pick) - 1)]
            if dfm is not None:                                               # where the deformation took it
                P0 = next((g_.centre_pt_def for g_ in sc.groups if getattr(g_, "centre_pt", None) is P0 and hasattr(g_, "centre_pt_def")), P0)
            o = [int(min(max(round(float(P0[a]) - e / 2 + r.u(-e / 6, e / 6)), 0), canvas - e)) for a, e in enumerate(ext)]; centred = True
            if "_objfill" not in ren:                                         # 29 Oct: the crop holds the populations' material --
                ren["_objfill"] = sum((res_["occupancy"].float() for res_ in ren["groups"]), torch.zeros((canvas,) * 3, device=device)).clamp(0, 1)
            ofill = lambda oc: float(ren["_objfill"][tuple(slice(a, a + e) for a, e in zip(oc, ext))].mean())
            if ofill(o) < CENTRE_FILL:                                        # moved within a third of the crop toward more of it
                best_c = (ofill(o), o); P0f = [float(P0[a]) for a in range(3)]
                for _ in range(12):
                    oc = [int(min(max(round(P0f[a] - e / 2 + r.u(-e / 3, e / 3)), 0), canvas - e)) for a, e in enumerate(ext)]
                    f_ = ofill(oc) if (dfm is None or vfrac(oc) >= 0.999) else -1.0
                    if f_ > best_c[0]:
                        best_c = (f_, oc)
                    if best_c[0] >= CENTRE_FILL:
                        break
                o = best_c[1]
            if dfm is not None and vfrac(o) < 0.999:                          # nearest shift that stays inside the deformed canvas
                best_o = (vfrac(o), 0.0, o)
                for _ in range(16):
                    oc = [int(min(max(a_ + round(r.u(-e / 3, e / 3)), 0), canvas - e)) for a_, e in zip(o, ext)]
                    v_, d_ = vfrac(oc), sum((x_ - y_) ** 2 for x_, y_ in zip(oc, o))
                    if (v_, -d_) > (best_o[0], -best_o[1]):
                        best_o = (v_, d_, oc)
                o = best_o[2]
    fill_rec = None
    if crop is None and not centred and fill_quota:                            # crops stratified by how much oriented tissue they hold
        lo_hi = FILL_STRATA; q = np.array(fill_quota, dtype=float); q = q / q.sum()
        want = int(r.np.choice(len(q), p=q))
        ext_map = _oriented_extent(sc, ren, device)                             # (canvas) 1 where oriented tissue lies within ~2 voxels
        best = None
        for _ in range(8):
            oc = [r.i(0, canvas - e) for e in ext]
            f_ = float(ext_map[tuple(slice(a, a + e) for a, e in zip(oc, ext))].float().mean())
            lo_, hi_ = lo_hi[want]; miss = 0.0 if lo_ <= f_ < hi_ else min(abs(f_ - lo_), abs(f_ - hi_))
            miss += 10.0 * (1.0 - vfrac(oc))                                    # (a crop reaching beyond the deformed canvas: last)
            if best is None or miss < best[0]:
                best = (miss, oc, f_)
            if miss == 0.0:
                break
        o = best[1]; fill_rec = {"stratum": ["packed", "partial", "sparse", "none"][want], "fill": round(best[2], 3), "hit": best[0] == 0.0}
    sl = [slice(a, a + e) for a, e in zip(o, ext)]
    shape = tuple(shape)
    # the background, cut from the scene's
    bg0, lab0, R0, extra0 = bgp
    bg = _crop(bg0, sl, zoom); R = _crop(R0, sl, zoom); lab = _crop(lab0, sl, zoom, "pick")
    extra = []
    for E in extra0:
        oc, (tc,) = _pool_fields(E.occupancy[tuple(sl)], [E.tensor[(slice(None),) + tuple(sl)]], zoom)
        if E.kind == "blob":                                                 # a cell stain's cells: solid material, isotropic (29 Oct)
            tc = (eye6(device) / 3).view(6, 1, 1, 1).expand(6, *tc.shape[1:]).clone()
        L = Layer(E.kind, torch.zeros(3, 1, 1, 1, device=device), E.radius / zoom, E.length / zoom, 0.0, torch.ones(shape, dtype=torch.bool, device=device),
                  E.contrast, occupancy=oc.clone(), tensor=tc.clone())
        L.desc = torch.tensor([E.radius / zoom, 0.0, 0.0], device=device).view(3, 1, 1, 1) * (oc > 0)[None]
        extra.append(L)
    layers = []; group_layer = []
    for res in ren["groups"]:
        g_ = res["group"]; group_layer.append(-1)
        occ, (tens, d5, cm) = _pool_fields(res["occupancy"][tuple(sl)].float(), [res["tensor"][(slice(None),) + tuple(sl)].float(), res["desc"][(slice(None),) + tuple(sl)].float(),
                                                                          res["contrast"][tuple(sl)][None].float()], zoom)
        if float(occ.max()) <= 1e-3:
            continue
        if zoom > 1:
            d5 = d5.clone(); d5[0] = d5[0] / zoom; d5[1] = d5[1] / zoom                    # sizes in the crop's voxels
        L = Layer(g_.kind, torch.zeros(3, 1, 1, 1, device=device), max(g_.radius / zoom, 0.5), 2 * max(g_.radius / zoom, 0.5), 0.0,
                  torch.ones(shape, dtype=torch.bool, device=device), g_.contrast, occupancy=occ.clone(), tensor=tens.clone())
        L.desc = torch.stack([d5[0], d5[2], d5[4]])                          # the layer descriptors: thickness, hollowness, angularity
        L.desc5 = d5.clone(); L.cmap = cm[0]; L.name = g_.name
        group_layer[-1] = len(layers); layers.append(L)
    shapes_ = layers + extra
    lum = _crop(ren["lumen"].float(), sl, zoom); lum_c = _crop(ren["lumen"].float() * ren["lumen_c"].float(), sl, zoom) / lum.clamp_min(1e-9)
    rec = {"scene": sid, "crop": o, "zoom": zoom, "style": st, **sc.record}
    keep_lut, kv = None, None
    if layers and rg.p(p_sparse):                                            # sparse labelling: most objects unlabelled -- gone from image and targets
        fk = rg.lu(0.01, 0.3); keep_lut = rg.rand(sc.next_obj + 1) < fk
        ids0 = _crop(ren["ids"][0], sl, zoom, "pick").long(); lab0_ = keep_lut[ids0.clamp_min(0)]
        hidden = (ids0 >= 0) & ~lab0_; kept = (ids0 >= 0) & lab0_
        kv = (~((Fn.max_pool3d(hidden.float()[None, None], 3, 1, 1)[0, 0] > 0) & ~kept)).float()   # the hidden objects and their partial-volume rim
        for L in layers:
            L.occupancy = L.occupancy * kv
        lum = lum * kv; rec["sparse"] = round(fk, 3)
    M = None
    if rg.p(p_boundary):                                                     # the specimen's edge: tissue on one side, medium on the other
        M = sample_mask(shape, rg, device)
    if rg.p(p_cutout):                                                       # chunks of tissue removed: blocks, holes, a knife's gap
        Mc, kinds = cutout_mask(shape, rg, device); M = Mc if M is None else M * Mc; rec["cutout"] = kinds
    if sc.outside is not None:                                               # a real cortical layout: nothing beyond the pial surface
        Mo = 1.0 - _crop(sc.outside.float(), sl, zoom)
        M = Mo if M is None else M * Mo
    medium = None
    if M is not None and float(M.min()) < 0.5:
        for L in shapes_:
            L.occupancy = L.occupancy * M
        lum = lum * M
        kind = ["dark", "bright", "mid"][int(ra.np.choice(3, p=[0.4, 0.4, 0.2]))]             # air / a dark mount, glass / a white slide, agar / paraffin
        med = {"dark": ra.u(0, 35), "bright": ra.u(220, 255), "mid": ra.u(35, 220)}[kind]
        medium = (med, ra.u(0, 6), ra.u(0, 8), kind)                                           # level, noise, faint texture
        R = M * R + (1 - M) * med                                                              # the specimen's surface is a step (an interface)
        lab = torch.where(M > 0.5, lab, torch.full_like(lab, int(lab.max()) + 1))
        rec["boundary"] = (round(float(1 - M.mean()), 2), kind)
    pre = [L.occupancy.clone() for L in layers] if entity_ctx else None
    cstore = {}
    img = _composite(bg, R, shapes_, device, store=cstore)                   # normalises the occupancies, adds the interface and isotropic states
    from . import generator as _gn
    after = [cstore["occluded"][i] for i in range(len(layers))] if (entity_ctx and layers and not _gn.TARGETS_BUDGET) else [L.occupancy for L in layers]
    comp = (torch.stack([torch.where(p_ > 1e-6, a_ / p_.clamp_min(1e-6), torch.ones_like(p_)) for p_, a_ in zip(pre, after)])
            if (entity_ctx and layers) else None)                            # occlusion (and the volume budget if TARGETS_BUDGET), per population
    mode = (physics or "xray") if hip else (physics or (ra.np.choice(PHYS_BY_MODALITY[modality]) if ra.p(p_physics) else "additive"))
    n_sh = len(layers) + len(extra)
    filled = sum((L.occupancy for L in shapes_[:n_sh]), torch.zeros(shape, device=device))
    occs = torch.stack([L.occupancy for L in shapes_[:n_sh]] + [torch.minimum(lum, (1 - filled).clamp_min(0))])
    tens = torch.stack([L.tensor for L in shapes_[:n_sh]] + [eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3])
    mtx = bool(sc.record.get("matrix"))                                     # a matrix: no free texture (structure only from objects)
    texture = (bg - bg.mean()) / bg.std().clamp_min(1e-6 if (hip and not mtx) else BG_TEX_NORM)          # 29 Oct: the background's texture as strong as
                                                                                              # it is (a flat background's 2-grey-level noise
                                                                                              # was normalised to full strength)
    tex = None
    if not mtx and ra.p(p_texture):                                          # texture inside each population (its own grain and depth)
        tex = torch.stack([(1 + (ra.u(0.05, 0.5) if not hip else ra.u(0.05, 0.05 + 0.15 * sg_)) * smooth_noise(shape, ra.lu(0.6, 6.0), ra.g, device)).clamp_min(0.05) if ra.p(0.7)
                           else torch.ones(shape, device=device) for _ in range(occs.shape[0])])
        routes_pre = ["texture"]
    else:
        routes_pre = []
    merge = {}
    if n_sh and ra.p(p_merge):                                               # shared intensities: populations as dark as the background or as another
        for k in range(n_sh):
            if ra.p(0.4):
                merge[k] = -1 if (n_sh == 1 or ra.p(0.5)) else ra.i(0, n_sh - 1)
        if merge:
            routes_pre.append("merge")
    def physics_image(m):
        if m == "additive":
            out = img.clone()
            for k, L in enumerate(layers):                                   # contrast varying along objects (striation, profiles)
                out = out + (L.cmap - L.contrast) * L.occupancy
            for k, L in enumerate(shapes_[:n_sh]):
                c_eff = 0.0 if merge.get(k, None) == -1 else (shapes_[merge[k]].contrast if k in merge else L.contrast)
                out = out + (c_eff - L.contrast) * L.occupancy
                if tex is not None:
                    out = out + c_eff * (tex[k] - 1) * L.occupancy
            return out + lum * lum_c
        if hip and m == "xray":                                               # tied attenuation and phase, contrast by strength
            from .physics import draw_background_hipct, draw_material_hipct
            bgm = draw_background_hipct(ra.np); mats = [draw_material_hipct(ra.np, bgm, sg_) for _ in range(occs.shape[0])]
        else:
            mats = [draw_material(ra.np, m) for _ in range(occs.shape[0])]; bgm = draw_material(ra.np, m, background=True)
            nl_ = len(layers)                                                 # 29 Oct: most populations stand out from the background,
            mats = [_material_contrast(mats[k], bgm, ra, stain=(nl_ <= k < n_sh)) if k < n_sh else mats[k] for k in range(len(mats))]   # stains faint
        for k, t in merge.items():
            mats[k] = dict(bgm) if t == -1 else dict(mats[t])
            if t == -1:                                                      # the background material carries no orientation terms
                mats[k].update({kk: vv for kk, vv in draw_material(ra.np, m).items() if kk not in bgm})
        if hip and m == "xray":
            from .physics import HIPCT_FORMATION as HF, draw_sequence_hipct
            return form_image(m, occs, tens, texture, mats, bgm, draw_sequence_hipct(ra.np, sg_), tex=tex, bg_mod=HF["bg_mod"] + HF["bg_mod_s"] * sg_)
        return form_image(m, occs, tens, texture, mats, bgm, draw_sequence(ra.np, m), tex=tex)
    img_main = physics_image(mode)
    routes = [mode] + routes_pre
    if ra.p(p_blend):                                                        # physics as a continuum: a blend with another (either polarity)
        m2 = str(ra.np.choice([m_ for m_ in PHYS_MODES if m_ != mode])); w = ra.u(0.3, 0.9)
        a = _grey(img_main); b = _grey(physics_image(m2)); b = 255 - b if ra.p(0.5) else b
        img_main = w * a + (1 - w) * b; routes.append(f"{m2}:{1 - w:.2f}")
    if ra.p(p_aniso):                                                        # orientation-dependent contrast under any physics
        gd = ra.unit(); gq = torch.stack([tens[k, 0] * gd[0] ** 2 + tens[k, 1] * gd[1] ** 2 + tens[k, 2] * gd[2] ** 2
                                          + 2 * (tens[k, 3] * gd[0] * gd[1] + tens[k, 4] * gd[0] * gd[2] + tens[k, 5] * gd[1] * gd[2]) for k in range(n_sh)]) if n_sh else None
        if gq is not None:
            amp = torch.tensor([ra.u(10, 80) * (1 if ra.p(0.5) else -1) for _ in range(n_sh)], device=device).view(-1, 1, 1, 1)
            sin2 = (1 - gq).clamp(0, 1) if ra.p(0.5) else gq.clamp(0, 1)
            img_main = _grey(img_main) + (amp * occs[:n_sh] * sin2).sum(0); routes.append("aniso")
    if ra.p(p_scramble):                                                     # each population and region its own offset, on top of the physics
        g0 = _grey(img_main)
        off = sum((ra.u(5, 60) * (1 if ra.p(0.5) else -1)) * occs[k] for k in range(n_sh)) if n_sh else 0.0
        nl = int(lab.max()) + 1; reg = torch.tensor([ra.u(-40, 40) for _ in range(nl)], device=device)[lab]
        img_main = g0 + off + reg; routes.append("scramble")
    img = img_main
    if ra.p(p_grain):                                                                          # unresolved microstructure under the tissue
        flat_ = img.flatten()[:: max(1, img.numel() // 100000)]
        q5, q95 = torch.quantile(flat_.float(), torch.tensor([0.05, 0.95], device=device))
        amp = (ra.lu(0.3, 0.3 + 3.7 * strength) * float(q95 - q5) / 4 if not hip
               else ra.u(HIPCT_GRAIN["lo"] * sg_, HIPCT_GRAIN["lo"] + HIPCT_GRAIN["span"] * sg_) * float(q95 - q5))   # HiP-CT: 0.1-0.4 of the range                          # real 20 um HiP-CT: texture over faint structures (domain gap, 23 Oct);
                                                                                                # with the strength, at most the image's range (28 Oct: up to 2x
                                                                                                # the range at any strength, under 80 % of crops, the density's
                                                                                                # targets went unseen and its best answer was flat)
        tex_ = microstructure_texture(shape, ra, device) * amp
        img = img + (tex_ * M if M is not None else tex_)
        routes.append("grain")
    if medium is not None:                                                                     # the medium set in grey levels after the physics:
        med, sd, tx, _ = medium                                                                # dark or bright whatever the image formation
        g0 = _grey(img) if mode != "additive" else img
        fill = med + sd * ra.randn(*shape) + tx * smooth_noise(shape, ra.lu(8, 30), ra.g, device)
        img = M * g0 + (1 - M) * fill
    rec["physics"] = mode; rec["routes"] = routes
    img = _extra_acquisition(img, layers, ra, modality, rec, strength) if not hip else _hipct_acquisition(img, ra, rec, strength)
    s = Sample(img, shapes_, bg, lab, st, 1.0, content, rec, R, budget=cstore.get("over") if entity_ctx else None)
    d = nuisance(s, seed=acq_seed if acq_seed is not None else seed, device=device, strength=strength, modality=modality, warp=False, fast=True, hp_max=12.0,
                 vis_opts={"per_layer_hp": True, "local_corr": True, "ladder": VIS_LADDER, "window_ladder": VIS_WINDOWS}, affine=False,   # (deformed
                                                                                                                                          # with the scene)
                 orient_from_texture=True if "tissue" in sc.record else None,            # packed tissue: evidence from texture, not edges
                 profile=profile)
    if fill_rec is not None:
        d["record"]["crop_fill"] = fill_rec
    if medium is not None and ra.p(0.5):                                                       # the tissue's range stretched to the display, the medium
        tis = d["image"].float()[M > 0.5]                                                      # clipped -- as a real crop normalised by its tissue
        if tis.numel() > 100:
            lo_, hi_ = torch.quantile(tis[:: max(1, tis.numel() // 100000)], torch.tensor([0.005, 0.995], device=device))
            d["image"] = ((d["image"].float() - lo_) / (hi_ - lo_).clamp_min(1e-6) * 255).clamp(0, 255).round()
            rec["tissue_window"] = True
    ids = _crop(ren["ids"], sl, zoom, "pick").long(); frac = _crop(ren["id_frac"], sl, zoom, "pick").float()
    pts = torch.stack([_point_map(pl, o, zoom, shape, device) for pl in ren["point_lists"]])
    if keep_lut is not None:                                                 # unlabelled objects: no identity, no points
        vis_ = (ids >= 0) & keep_lut[ids.clamp_min(0)]
        ids = torch.where(vis_, ids, torch.full_like(ids, -1)); frac = frac * vis_; pts = pts * kv[None]
    if M is not None:                                                        # nothing beyond the specimen's edge
        ids = torch.where((M > 0.5)[None], ids, torch.full_like(ids, -1)); frac = frac * (M > 0.5)[None]; pts = pts * M[None]
    D5 = torch.zeros(len(s.layers), 5, *shape, device=device)
    for k, L in enumerate(s.layers):
        if hasattr(L, "desc5"):
            D5[k] = _log_desc(L.desc5)
    grid = d.get("warp_grid")
    if grid is not None:                                                     # the warp moves the identities, points and descriptors too
        from .generator import _resample
        pad = d["record"].get("warp_padding", "border")                                       # the same padding as the image's
        ids = _resample(ids.float(), grid, mode="nearest", padding=pad).round().long(); frac = _resample(frac, grid, padding=pad)
        pts = _resample(pts, grid, padding=pad)
        D5 = _resample(D5.reshape(-1, *shape), grid, padding=pad).reshape(D5.shape)
        D5[:, 0] = torch.log2(1 + (2 ** D5[:, 0] - 1) * d["stretch"][None])     # sizes scale with the local stretch
    lut = sc.identity_lut(identity_min_r)
    d["identity"] = torch.where(ids[0] >= 0, lut[ids[0].clamp(0, lut.numel() - 1)], ids[0])
    d["ids"], d["id_frac"], d["points"], d["descriptors5"] = ids, frac, pts, D5
    vu = sc.voxel_um * zoom
    d["desc_um"] = [torch.stack([L.desc5[0] * vu, L.desc5[1] * vu, L.desc5[2], L.desc5[3], L.desc5[4]]) for L in layers]
    d["recoverability"] = d["orientation_visibility"]; d["parents"] = sc.parents; d["voxel_um"] = vu
    d["record"].update({k: v for k, v in rec.items() if k not in d["record"]})
    if entity_ctx:                                                           # the entities' targets are computed from these
        if "tree" not in ren:
            from .entities import build_tree
            ren["tree"] = build_tree(sc, ren["entries"][1], ren["entries"][2])
        d["entity_ctx"] = {"entries": ren["entries"], "tree": ren["tree"], "canvas_shape": (canvas,) * 3, "origin": o, "zoom": zoom,
                           "group_layer": group_layer, "grid": d.get("warp_grid"), "keep": keep_lut, "mask": M, "comp": comp}
    if second_contrast:                                                      # a second contrast of the same scene (multi-contrast MRI)
        img2 = bg * 0.5 + 60
        for L in layers:
            img2 = img2 + ra.u(-120, 120) * L.occupancy
        d["image2"] = img2
    return s, d


def draw_pair(shape=(64, 64, 64), seed: int = 0, device="cuda", kind: str = "zoom", **kw):
    """Two renders of one scene for consistency: ``zoom`` -- the crop and the region twice its extent around it at
    half the resolution (the fine crop's lattice cells 2³ -> one coarse cell; ``offset``: the coarse cell of the fine
    crop's first cell pair); ``modality`` -- the same crop under another acquisition (modality, physics, materials,
    noise); ``shift`` -- two overlapping windows of the scene, offset by multiples of 8 voxels (``info["shift"]``, crop 2's
    origin minus crop 1's), under the same acquisition or another. Returns ``(s1, d1), (s2, d2), info``."""
    canvas = kw.get("canvas", 128); r = Rng(seed + 99991, device)
    kw = {**kw, "p_boundary": 0.0, "p_affine": 0.0, "p_warp": 0.0, "p_sparse": 0.0, "p_cutout": 0.0}   # one geometry: the renders differ in acquisition only
    if kind == "zoom":
        ext = [2 * n for n in shape]; assert all(e <= canvas for e in ext)
        oc = [8 * r.i(0, (canvas - e) // 8) for e in ext]                    # the coarse crop's origin (multiple of 8: lattice-aligned)
        of = [a + 8 * r.i(0, n // 8) for a, n in zip(oc, shape)]             # the fine crop inside it
        a = draw_scene(shape, seed, device, crop=of, zoom=1, **kw)
        b = draw_scene(shape, seed, device, crop=oc, zoom=2, acq_seed=seed + 4242, **kw)
        return a, b, {"kind": "zoom", "offset_cells": [(f - c) // 8 for f, c in zip(of, oc)], "scale": 2}
    if kind == "shift":                                                      # two overlapping windows (multiples of 8: lattice-aligned)
        assert int(kw.get("zoom", 1)) == 1, "shift pairs are drawn at zoom 1 (offsets in crop voxels)"
        sh = [8 * r.i(-(n // 2) // 8, (n // 2) // 8) for n in shape]
        if not any(sh):
            sh[r.i(0, 2)] = 8
        o1 = [r.i(max(0, -d_), canvas - n - max(0, d_)) for n, d_ in zip(shape, sh)]; o2 = [a_ + d_ for a_, d_ in zip(o1, sh)]
        same = r.p(0.5)                                                      # the same acquisition, or another
        a = draw_scene(shape, seed, device, crop=o1, **kw)
        b = draw_scene(shape, seed, device, crop=o2, acq_seed=None if same else seed + 4242, **kw)
        return a, b, {"kind": "shift", "shift": sh, "same_acq": same, "offset_cells": [0, 0, 0], "scale": 1}
    o = [r.i(0, canvas - n) for n in shape]
    a = draw_scene(shape, seed, device, crop=o, **kw)
    b = draw_scene(shape, seed, device, crop=o, acq_seed=seed + 4242, **kw)
    return a, b, {"kind": "modality", "offset_cells": [0, 0, 0], "scale": 1}
