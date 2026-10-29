"""Synthesis for the population model: layered random shapes, each layer one population with a shape tensor.

The grammar (anatomix's recipe generalised):

* a **background**: ``regions`` (anatomix: a random label map, an intensity per label), ``tissue`` (two or three broad
  regions with texture), ``cells`` (a cell-stain field of small blobs, dark or bright, optionally laminar);
* ``K`` **layers**, each a population made by one process: a smooth orientation field (or an arc field, for bends of
  a drawn radius), a shape class -- ``line`` (extrusions along the field), ``sheet`` (discs perpendicular to it),
  ``blob`` (extrusions about as long as their radius) -- a radius, a length, a density, a region and a contrast; lines
  may be hollow (vessel walls); some layers occlude what was drawn before them;
* **crossing groups**: two or three line layers sharing one region at a drawn crossing angle (20-90°), their densities
  splitting one total, later fibres yielding where earlier ones already are (interdigitated, not stacked).

Every layer is rendered by trilinear splatting of its shapes' sample points and a Gaussian of the shape's radius. Its
**occupancy** is a volume fraction (1 at the centre of an isolated shape, normalised analytically, capped at 1); its
**shape tensor** ``(6, Z, Y, X)`` (trace 1; components zz, yy, xx, zy, zx, yx) is the local mean of the deposited
shapes' tensors: ``t tᵀ`` along a line, ``(I - n nᵀ) / 2`` on a disc of normal ``n``, the normalised second moment of a
blob's mass. The layers are composited as volume fractions: an occluder hides what is under it (the labels under it are
attenuated), overlaps never fill more than the voxel; the **isotropic** remainder fills the rest of the volume. The
**interface** (a sheet at the background's region boundaries, weighted by the step) is a boundary, not a volume: it is
outside the volume budget and hidden only by occluders, as the step is in the image.

:func:`nuisance` applies one modality's acquisition (``hipct`` / ``ct`` / ``mri`` / ``histology`` / ``em`` /
``fluorescence`` / ``cryoet`` / ``mixed``): an elastic warp (shape tensors transformed as ``J⁻¹ S J⁻ᵀ``), intensity
remaps, a bias field, the PSF, noise, and the modality's artefacts along a randomly drawn acquisition axis (serial
sections: shifts, drift, lost sections and slabs, knife chatter; light-sheet shadows, depth attenuation, the cryo-ET
missing wedge and CTF, photon-starved doses, saturation), then a display window; it measures each layer's
**visibility** (contrast to noise at the shapes' scale after the whole chain), which scales the layer's mass in the
target: its local **presence** visibility scales the layer's mass, its **orientation** visibility blends the layer's
shape tensor towards isotropic (seen but orientation not recoverable). :func:`draw` makes one training pair: content
matched to the modality, then its nuisances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from ..filters import gaussian, kernel1d


# ---------------------------------------------------------------------------------------------------------------
# random fields
# ---------------------------------------------------------------------------------------------------------------

def smooth_noise(shape, sigma: float, g: torch.Generator, device) -> torch.Tensor:
    """Stationary smoothed Gaussian noise ``(Z, Y, X)``, unit variance in expectation. Drawn on a grid padded beyond
    the blur's reach (its edge handling never touches the block) and, for large ``sigma``, on a coarse grid (spacing
    ``sigma // 2``) then trilinearly upsampled; normalised by the kernel's exact variance, not by the block's std."""
    if sigma <= 0:
        return torch.randn(shape, device=device, generator=g)
    f = max(1, int(sigma // 2)); sc = sigma / f
    pad = int(math.ceil(3 * sc)) + 1
    m = [int(math.ceil((n - 1) / f)) + 1 for n in shape]                                       # coarse samples covering the block
    x = torch.randn([k + 2 * pad for k in m], device=device, generator=g)
    x = gaussian(x, sc, truncate=3.0)[pad:-pad, pad:-pad, pad:-pad]
    x = x / math.sqrt(float((kernel1d(sc, 0, 3.0) ** 2).sum()) ** 3)
    if f > 1:
        x = Fn.interpolate(x[None, None], size=[(k - 1) * f + 1 for k in m], mode="trilinear", align_corners=True)[0, 0]
    return x[: shape[0], : shape[1], : shape[2]].contiguous()


def direction_field(shape, sigma: float, g: torch.Generator, device) -> torch.Tensor:
    """A smooth random unit-vector field ``(3, Z, Y, X)`` (z, y, x); ``sigma`` sets how far it stays coherent."""
    v = torch.stack([smooth_noise(shape, sigma, g, device) for _ in range(3)])
    return v / v.norm(dim=0, keepdim=True).clamp_min(1e-6)


def _unit(g: torch.Generator, device) -> torch.Tensor:
    v = torch.randn(3, device=device, generator=g)
    return v / v.norm().clamp_min(1e-6)


def _coords(shape, device) -> torch.Tensor:
    return torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32) for n in shape), indexing="ij"))


def arc_field(shape, radius: float, g: torch.Generator, device) -> torch.Tensor:
    """Tangents of circles about a random axis placed ``radius`` voxels from the block centre: fibres there bend with
    that radius of curvature (a controlled bend). ``(3, Z, Y, X)``."""
    a = _unit(g, device); e = _unit(g, device); e = e - (e @ a) * a; e = e / e.norm().clamp_min(1e-6)
    c = torch.tensor([(n - 1) / 2 for n in shape], device=device) + radius * e
    r = _coords(shape, device) - c.view(3, 1, 1, 1)
    t = torch.linalg.cross(a.view(3, 1, 1, 1).expand_as(r), r, dim=0)
    n = t.norm(dim=0, keepdim=True)
    return torch.where(n > 1e-3, t / n.clamp_min(1e-6), a.view(3, 1, 1, 1).expand_as(t))


def companion_field(d1: torch.Tensor, theta_deg: float, coherence: float, g: torch.Generator, device) -> torch.Tensor:
    """A field at exactly ``theta_deg`` to ``d1`` everywhere, turning about it smoothly (a crossing population that
    bends with the first). ``(3, Z, Y, X)``."""
    e = direction_field(d1.shape[1:], coherence, g, device)
    e = e - (e * d1).sum(0, keepdim=True) * d1
    n = e.norm(dim=0, keepdim=True)
    alt = torch.linalg.cross(d1, torch.roll(d1, 1, dims=0) + 0.3, dim=0)                       # any perpendicular, where e vanishes
    e = torch.where(n > 1e-3, e / n.clamp_min(1e-6), alt / alt.norm(dim=0, keepdim=True).clamp_min(1e-6))
    t = math.radians(theta_deg)
    return math.cos(t) * d1 + math.sin(t) * e


def region_mask(shape, sigma: float, frac: float, g: torch.Generator, device) -> torch.Tensor:
    """A random region covering about ``frac`` of the block (a smoothed-noise threshold), or the whole block."""
    if frac >= 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    n = smooth_noise(shape, sigma, g, device)
    return n >= torch.quantile(n.flatten()[:: max(1, n.numel() // 200_000)], 1 - frac)


def label_map(shape, n_labels: int, sigma: float, g: torch.Generator, device) -> torch.Tensor:
    """anatomix's random label map: the argmax of ``n_labels`` smoothed noises, ``(Z, Y, X)`` long."""
    return torch.stack([smooth_noise(shape, sigma, g, device) for _ in range(n_labels)]).argmax(0)


# ---------------------------------------------------------------------------------------------------------------
# shape tensors (6 components: zz, yy, xx, zy, zx, yx)
# ---------------------------------------------------------------------------------------------------------------

SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def outer6(v: torch.Tensor) -> torch.Tensor:
    """``v (..., 3)`` -> ``v vᵀ`` as ``(..., 6)``."""
    return torch.stack([v[..., i] * v[..., j] for i, j in SYM], -1)


def eye6(device, lead=()) -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device).expand(*lead, 6)


def full3(T: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` -> ``(..., 3, 3)``."""
    zz, yy, xx, zy, zx, yx = T.unbind(-1)
    return torch.stack([torch.stack([zz, zy, zx], -1), torch.stack([zy, yy, yx], -1), torch.stack([zx, yx, xx], -1)], -2)


def sym6(M: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 6)``."""
    return torch.stack([M[..., i, j] for i, j in SYM], -1)


def line_tensor(d: torch.Tensor) -> torch.Tensor:
    return outer6(d)


def sheet_tensor(n: torch.Tensor) -> torch.Tensor:
    return (eye6(n.device, n.shape[:-1]) - outer6(n)) / 2


def spread_tensor(t: torch.Tensor, e) -> torch.Tensor:
    """A fibre's local material ODF as an ellipse (29 Oct): ``(1 - e) t t^T + e (I - t t^T) / 2`` for directions
    ``t (..., 3)`` and spread ``e = sin^2 s`` (scalar or ``(...)``; 0: the tangent's dyad, 2/3: isotropic). Trace 1."""
    e = torch.as_tensor(e, dtype=t.dtype, device=t.device)
    e = e[..., None] if e.dim() else e
    d = outer6(t)
    return (1 - e) * d + e * (eye6(t.device, t.shape[:-1]) - d) / 2


SPREAD_DEG = (1.0, 15.0)            # each fibre-like object's own spread angle s (degrees, log-uniform): e = sin^2 s (29 Oct)


def draw_spread(n: int, g: torch.Generator, device) -> torch.Tensor:
    """``n`` spread angles (degrees), log-uniform in :data:`SPREAD_DEG`."""
    lo, hi = math.log(SPREAD_DEG[0]), math.log(SPREAD_DEG[1])
    return torch.exp(lo + (hi - lo) * torch.rand(n, generator=g, device=device))


def spread_e(spread_deg: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.deg2rad(spread_deg)) ** 2


def blob_tensor(d: torch.Tensor, radius: float, length: float, step: float = 0.5) -> torch.Tensor:
    """The normalised second moment of a blob's rendered mass: its points (every ``step`` over ``length`` along
    ``d``: variance ``step² h (h + 1) / 3``) blurred by a Gaussian of ``radius`` plus the splat (variance
    ``radius² + 1/6`` per axis)."""
    h = _line_half(length, step)
    a, b = max(radius, 0.5) ** 2 + 1.0 / 6.0, step ** 2 * h * (h + 1) / 3.0
    return (a * eye6(d.device, d.shape[:-1]) + b * outer6(d)) / (3 * a + b)


def eig3(T: torch.Tensor):
    """Eigen-decomposition of a tensor field ``T (6, ...)`` (zz, yy, xx, zy, zx, yx) in closed form (no batch limit,
    unlike cusolver): eigenvalues ``(3, ...)`` ascending and eigenvectors ``(3, 3, ...)`` (``vec[i]`` for ``val[i]``)."""
    from ..st import _eig3_sym
    shp = T.shape[1:]
    val, vec = _eig3_sym(T.reshape(6, -1).float(), full=True)
    return val.reshape(3, *shp).to(T.dtype), vec.reshape(3, 3, *shp).to(T.dtype)


def principal(T: torch.Tensor, which: str = "max") -> torch.Tensor:
    """The eigenvector of a tensor field ``(6, ...)`` for its largest (``max``: a line's direction) or smallest
    (``min``: a sheet's normal) eigenvalue, ``(3, ...)``, and the eigenvalues ``(3, ...)`` descending."""
    val, vec = eig3(T)
    v = vec[2] if which == "max" else vec[0]
    return v, val.flip(0)


# ---------------------------------------------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Layer:
    kind: str                      # "line" | "sheet" | "blob"
    field: torch.Tensor            # (3, Z, Y, X): the generating field -- direction (line, blob) or normal (sheet)
    radius: float                  # voxels (the shapes' Gaussian cross-section)
    length: float                  # voxels (the extrusion length; a disc's radius for sheets; ~radius for blobs)
    density: float                 # seeds per 1000 voxels of the region
    region: torch.Tensor           # (Z, Y, X) bool
    contrast: float                # grey levels over the local background where the layer fills a voxel; an occluder's absolute grey level
    occluding: bool = False        # the layer hides what was drawn before it
    hollow: bool = False           # lines: a wall, the lumen (0.55 of the radius) empty
    coherence: float = 0.0         # the field's coherence length (voxels); 0 for an arc or a fixed field
    bend_radius: float = 0.0       # the arc field's radius of curvature (voxels); 0 for other fields
    role: str = "shape"            # "shape" | "interface" | "isotropic"
    group: int = -1                # crossing group id (-1: none)
    occupancy: torch.Tensor | None = None   # (Z, Y, X) volume fraction 0-1
    tensor: torch.Tensor | None = None      # (6, Z, Y, X) shape tensor, trace 1
    seeds: torch.Tensor | None = None       # (n, 3) explicit seed positions (None: drawn at ``density``)
    # the shape grammar (see render_layer): a line's cross-section, a sheet's leaflets, a compact solid's primitive
    section: str = "disc"          # lines: "disc" (Gaussian profile) | "ring" (hollow: a wall) | "ellipse" (ribbon) | "superellipse" (square-ish)
    section_aspect: float = 1.0    # ellipse: minor / major axis
    section_exp: float = 2.0       # superellipse exponent (2: disc; > 2 square-ish; < 2 diamond-ish)
    wall: float = 0.4              # ring: wall thickness as a fraction of the radius
    bilayer: float = 0.0           # sheets: the gap between two leaflets (voxels); 0: one solid sheet
    primitive: str = "superquadric"  # solids: "superquadric" | "polyhedron" | "shard"
    axes: tuple = (1.0, 1.0, 1.0)  # solids: semi-axes as multiples of the radius, along (u, v, the field)
    exps: tuple = (1.0, 1.0)       # superquadric exponents (1, 1: ellipsoid; -> 0: box-like; 2: octahedral)
    shell: float = 0.0             # solids: shell thickness (voxels); 0: solid
    desc: torch.Tensor | None = None        # (3, Z, Y, X) per-voxel descriptors: thickness (voxels), hollowness, angularity (0-1)


def _at(pos: torch.Tensor, fld: torch.Tensor, shape) -> torch.Tensor:
    """The field at positions ``pos (n, 3)`` (nearest voxel), ``(n, 3)``."""
    idx = pos.round().long()
    for i, n in enumerate(shape):
        idx[:, i] = idx[:, i].clamp(0, n - 1)
    return fld[:, idx[:, 0], idx[:, 1], idx[:, 2]].T


def _splat(P: torch.Tensor, V: torch.Tensor, shape, device):
    """Trilinear splat of points ``P (n, 3)`` carrying tensors ``V (n, 6)``: the density ``(Z, Y, X)`` and the
    density-weighted tensor sum ``(6, Z, Y, X)``."""
    dens = torch.zeros(shape, device=device); T = torch.zeros(tuple(shape) + (6,), device=device)
    p0 = P.floor(); fr = P - p0; i0 = p0.long(); lim = torch.tensor(shape, device=device)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                off = torch.tensor([dz, dy, dx], device=device)
                idx = i0 + off
                w = torch.where(off.bool(), fr, 1 - fr).prod(1)
                ok = ((idx >= 0) & (idx < lim)).all(1)
                ii = idx[ok]; ww = w[ok]
                dens.index_put_((ii[:, 0], ii[:, 1], ii[:, 2]), ww, accumulate=True)
                T.index_put_((ii[:, 0], ii[:, 1], ii[:, 2]), ww[:, None] * V[ok], accumulate=True)
    return dens, T.movedim(-1, 0)


def _sheet_offsets(length: float, step: float, device) -> torch.Tensor:
    """The (u, w) offsets of a disc's sample points, ``(m, 2)``: a grid of spacing ~``step`` inside radius ``length``."""
    m = max(4, int(round(2 * length / step)))
    c = (torch.arange(m, device=device, dtype=torch.float32) + 0.5) / m * 2 - 1
    U, W = torch.meshgrid(c, c, indexing="ij"); keep = U ** 2 + W ** 2 <= 1
    return length * torch.stack([U[keep], W[keep]], 1)


def _line_half(length: float, step: float) -> int:
    return max(1, int(round(length / 2 / step)))


def _peak(kind: str, radius: float, length: float, step: float, device) -> float:
    """The density at the centre of one isolated straight shape (the occupancy's normaliser): the deposited points of
    the shape summed under the blur (Gaussian of ``radius``, plus the trilinear splat's variance 1/6 per axis)."""
    s2 = radius ** 2 + 1.0 / 6.0; c3 = (2 * math.pi * s2) ** -1.5
    if kind == "sheet":
        off = _sheet_offsets(length, step, device)
        return float(c3 * torch.exp(-(off ** 2).sum(1) / (2 * s2)).sum())
    h = _line_half(length, step); k = torch.arange(-h, h + 1, device=device, dtype=torch.float32) * step
    return float(c3 * torch.exp(-k ** 2 / (2 * s2)).sum())


def _wall_peak(ro: float, ri: float) -> float:
    """The maximum of the hollow profile ``exp(-ρ²/2a) - exp(-ρ²/2b)`` (a = ro² + 1/6, b = ri² + 1/6)."""
    a, b = ro ** 2 + 1 / 6, ri ** 2 + 1 / 6
    x = math.log(a / b) / (1 / b - 1 / a)
    return math.exp(-x / a) - math.exp(-x / b)


def _seeds(layer: Layer, shape, g: torch.Generator, device) -> torch.Tensor:
    """Seeds at ``layer.density`` per 1000 voxels, uniform over the block padded by the shapes' reach (half a line's
    length, a disc's radius; at most 24 voxels): shapes enter from outside as they would in a larger volume (no
    fall-off at the faces). The region is applied to what is deposited, not to the seeds (see :func:`render_layer`),
    so the region's volume fraction of shapes is the drawn one wherever the region is."""
    if layer.seeds is not None:
        return layer.seeds.to(device).float()
    pad = min(24.0, layer.length / 2 if layer.kind != "sheet" else layer.length)
    lo = torch.full((3,), -pad, device=device); span = torch.tensor(shape, device=device, dtype=torch.float32) - 1 + 2 * pad
    n = int(torch.poisson(torch.tensor(layer.density * float(span.prod()) / 1000.0, device=device), generator=g))
    return lo + torch.rand((max(n, 1), 3), device=device, generator=g) * span


def shape_volume(kind: str, radius: float, length: float, step: float = 0.5, device="cpu") -> float:
    """The occupancy integral of one isolated shape (voxels): its deposited points over its peak density."""
    r = max(radius, 0.5)
    n = _sheet_offsets(length, step, device).shape[0] if kind == "sheet" else 2 * _line_half(length, step) + 1
    return n / _peak(kind, r, length, step, device)


def density_for(kind: str, radius: float, length: float, fraction: float, step: float = 0.5) -> float:
    """Seeds per 1000 voxels giving a volume ``fraction`` of shapes (before overlaps)."""
    return 1000.0 * fraction / shape_volume(kind, radius, length, step)


def layer_density(layer: "Layer", fraction: float, step: float = 0.5) -> float:
    """Seeds per 1000 voxels for a volume ``fraction`` of the layer's shapes, whatever their grammar."""
    if layer.kind == "solid":
        return 1000.0 * fraction / solid_volume(layer)
    if layer.kind == "line" and layer.section != "disc":
        return 1000.0 * fraction / max(section_area(layer) * layer.length, 1.0)
    vol = shape_volume(layer.kind, layer.radius, layer.length, step)
    return 1000.0 * fraction / (vol * (2 if (layer.kind == "sheet" and layer.bilayer > 0) else 1))


DESCRIPTORS = ("thickness", "hollowness", "angularity")


def _section_offsets(layer: Layer, h: float, device) -> torch.Tensor:
    """The (u, w) offsets filling a line's cross-section on a grid of spacing ``h``, ``(m, 2)``."""
    r = max(layer.radius, 0.5)
    n = int(math.ceil(r / h))
    c = torch.arange(-n, n + 1, device=device, dtype=torch.float32) * h
    U, W = torch.meshgrid(c, c, indexing="ij"); U, W = U.flatten(), W.flatten()
    if layer.section == "ring":
        rr = (U ** 2 + W ** 2).sqrt(); keep = (rr <= r) & (rr >= r * (1 - layer.wall))
    elif layer.section == "ellipse":
        keep = (U / r) ** 2 + (W / (r * layer.section_aspect)) ** 2 <= 1
    elif layer.section == "superellipse":
        keep = (U / r).abs() ** layer.section_exp + (W / r).abs() ** layer.section_exp <= 1
    else:
        keep = U ** 2 + W ** 2 <= r * r
    return torch.stack([U[keep], W[keep]], 1)


def section_area(layer: Layer) -> float:
    """The cross-section's area (voxels²)."""
    r = max(layer.radius, 0.5)
    if layer.section == "ring":
        return math.pi * r * r * (1 - (1 - layer.wall) ** 2)
    if layer.section == "ellipse":
        return math.pi * r * r * layer.section_aspect
    if layer.section == "superellipse":
        p = layer.section_exp
        return 4 * r * r * math.gamma(1 + 1 / p) ** 2 / math.gamma(1 + 2 / p)
    return math.pi * r * r


def section_angularity(layer: Layer) -> float:
    """0 for round sections; towards 1 for square (superellipse exponent >> 2) or diamond (<< 2) sections."""
    if layer.section != "superellipse":
        return 0.0
    return float(min(1.0, abs(math.log(layer.section_exp / 2.0)) / math.log(4.0)))


def _paths(layer: Layer, seeds: torch.Tensor, shape, step: float):
    """Points along each seed's streamline of the field (both ways, ``length`` in all) and their tangents."""
    d_seed = _at(seeds, layer.field, shape)
    P = [seeds]; D = [d_seed]
    for sgn in (1.0, -1.0):
        p = seeds.clone(); d0 = d_seed
        for _ in range(_line_half(layer.length, step)):
            d = _at(p, layer.field, shape)
            d = torch.where((d * d0).sum(1, keepdim=True) < 0, -d, d)                           # keep the sense of travel along an axial field
            d0 = d
            p = p + sgn * step * d; P.append(p.clone()); D.append(d)
    return torch.stack(P, 1), torch.stack(D, 1)                                                 # (n, steps, 3)


def _frame(d: torch.Tensor, ref: torch.Tensor):
    """Unit vectors e1, e2 perpendicular to ``d (..., 3)``, from a reference ``ref`` (falls back where parallel)."""
    e1 = torch.linalg.cross(d, ref.expand_as(d))
    alt = torch.linalg.cross(d, torch.roll(ref, 1, dims=-1).expand_as(d))
    e1 = torch.where(e1.norm(dim=-1, keepdim=True) > 1e-3, e1, alt)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return e1, torch.linalg.cross(d, e1)


def _keep(P: torch.Tensor, layer: Layer, shape, exclude):
    idx = P.round().long()
    inside = ((idx >= 0) & (idx < torch.tensor(shape, device=P.device))).all(-1)
    for i, n_ in enumerate(shape):
        idx[..., i] = idx[..., i].clamp(0, n_ - 1)
    keep = inside & layer.region[idx[..., 0], idx[..., 1], idx[..., 2]]
    if exclude is not None:
        keep &= ~exclude[idx[..., 0], idx[..., 1], idx[..., 2]]
    return keep


def _sdf(prim: str, xs: torch.Tensor, exps, N: torch.Tensor | None, cut: torch.Tensor | None) -> torch.Tensor:
    """Signed distance (in units of the smallest semi-axis, approximately) of points ``xs (..., 3)`` already scaled
    by the semi-axes: a superquadric of ``exps``, a convex polyhedron of face normals ``N (n_obj, k, 3)`` at unit-ish
    offsets (stored in ``N``'s norm), a shard (the polyhedron cut by the plane ``cut (n_obj, 4)``)."""
    if prim == "superquadric":
        e1, e2 = exps
        a = (xs[..., 0].abs() + 1e-9) ** (2 / e2) + (xs[..., 1].abs() + 1e-9) ** (2 / e2)
        F = a ** (e2 / e1) + (xs[..., 2].abs() + 1e-9) ** (2 / e1)
        return F ** (e1 / 2) - 1
    off = N.norm(dim=-1)                                                                        # (n_obj, k) face offsets
    nn_ = N / off[..., None]
    d = torch.bmm(xs, nn_.transpose(1, 2)) - off[:, None, :]                                    # (n_obj, b, k): every face at once
    sd = torch.maximum(d.amax(-1), xs.norm(dim=-1) - 1.5)                                     # bounded: random faces need not enclose the centre
    if prim == "shard" and cut is not None:
        sd = torch.maximum(sd, torch.bmm(xs, cut[:, :3, None])[..., 0] - cut[:, 3:4])
    return sd


def render_solids(layer: Layer, shape, g: torch.Generator, device, exclude: torch.Tensor | None = None, edge: float = 0.7):
    """Compact solids, one per seed, rendered from signed distance functions: superquadrics (smooth ellipsoids to
    box-like), convex polyhedra (crystals with facets and sharp edges) and shards (polyhedra cut by a plane), solid or
    as shells. Each object's long axis follows the layer's field at its centre (the population's orientation); its
    semi-axes are ``radius * axes``. Its label is the normalised second moment of its own rendered mass (any shape:
    a needle is linear, a platelet planar, a cube or sphere isotropic); its descriptors are its smallest semi-axis
    (thickness), its shell (hollowness: 1 - shell / smallest semi-axis) and its angularity (0 for an ellipsoid; 1 for
    a polyhedron or shard; a superquadric by its exponents). Returns occupancy, tensor, descriptor maps."""
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    seeds = _seeds(layer, shape, g, device); n = seeds.shape[0]
    r = max(layer.radius, 0.5); ax = torch.tensor(layer.axes, device=device, dtype=torch.float32) * r   # (3,) semi-axes in the object frame (u, v, w)
    cap = max(4.0, min(shape) / 6.0)                                                             # objects larger than this are regions, not shapes
    if float(ax.max()) > cap:
        ax = ax * (cap / float(ax.max()))
    H = int(math.ceil(float(ax.max()) * (1.5 if layer.primitive != "superquadric" else 1.05) + 1.5 + layer.shell))
    o = torch.arange(-H, H + 1, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)              # (b, 3) box offsets
    w = _at(seeds, layer.field, shape)                                                           # each object's long axis
    ref = torch.randn((n, 3), device=device, generator=g)
    e1, e2 = _frame(w, ref)
    R = torch.stack([e1, e2, w], -1)                                                             # (n, 3, 3): columns u, v, w
    N = cut = None
    if layer.primitive in ("polyhedron", "shard"):
        k = int(u(6, 10))
        N = torch.nn.functional.normalize(torch.randn((n, k, 3), device=device, generator=g), dim=-1) * (1 + 0.25 * torch.rand((n, k, 1), device=device, generator=g))
        if layer.primitive == "shard":
            cut = torch.cat([torch.nn.functional.normalize(torch.randn((n, 3), device=device, generator=g), dim=-1),
                             (-0.4 + 0.6 * torch.rand((n, 1), device=device, generator=g))], 1)
    smin = float(ax.min())
    dens = torch.zeros(shape, device=device); T = torch.zeros((6,) + tuple(shape), device=device); D = torch.zeros((3,) + tuple(shape), device=device)
    hol = (1 - layer.shell / smin) if layer.shell > 0 else 0.0
    ang = 1.0 if layer.primitive != "superquadric" else float(min(1.0, max(abs(math.log(e)) for e in layer.exps) / math.log(5.0)))
    batch = max(1, int(2e7 // (grid.shape[0] * (N.shape[1] if N is not None else 1))))
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch)); c = seeds[sl]; m = c.shape[0]
        base = c.round()
        X = base[:, None] + grid[None]                                                            # (m, b, 3) voxel positions
        rel = X - c[:, None]
        xo = torch.einsum("mbi,mij->mbj", rel, R[sl])                                             # object frame
        xs = xo / ax
        sd = _sdf(layer.primitive, xs, layer.exps, None if N is None else N[sl], None if cut is None else cut[sl]) * smin
        if layer.shell > 0:
            sd = sd.abs() - layer.shell / 2
        occ = (0.5 - sd / edge).clamp(0, 1)                                                       # partial volume over ~ edge voxels
        occ = occ * _keep(X, layer, shape, exclude)
        mass = occ.sum(1).clamp_min(1e-6)
        mu = (occ[..., None] * rel).sum(1) / mass[:, None]
        dx = rel - mu[:, None]
        C = torch.einsum("mb,mbi,mbj->mij", occ, dx, dx) / mass[:, None, None]
        C = C / C.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None].clamp_min(1e-9)
        t6 = sym6(C)                                                                              # (m, 6)
        idx = X.round().long(); ok = ((idx >= 0) & (idx < torch.tensor(shape, device=device))).all(-1) & (occ > 0)
        ii = idx[ok]; ww = occ[ok]; tt = t6[:, None].expand(-1, X.shape[1], -1)[ok]
        dens.index_put_((ii[:, 0], ii[:, 1], ii[:, 2]), ww, accumulate=True)
        Tf = T.view(6, -1); flat = (ii[:, 0] * shape[1] + ii[:, 1]) * shape[2] + ii[:, 2]
        Tf.index_add_(1, flat, (ww[:, None] * tt).T)
        dval = torch.tensor([smin, hol, ang], device=device)
        D.view(3, -1).index_add_(1, flat, ww[None] * dval[:, None])
    occ = dens.clamp(0, 1)
    tensor = torch.where(dens[None] > 1e-6, T / dens.clamp_min(1e-6)[None], eye6(device).view(6, 1, 1, 1) / 3)
    desc = torch.where(dens[None] > 1e-6, D / dens.clamp_min(1e-6)[None], torch.zeros_like(D))
    return occ, tensor, desc


def solid_volume(layer: Layer, device="cpu") -> float:
    """The volume of one of the layer's solids (voxels): an ellipsoid estimate scaled by the primitive's fill."""
    r = max(layer.radius, 0.5); a = [x * r for x in layer.axes]
    v = 4 / 3 * math.pi * a[0] * a[1] * a[2]
    if layer.primitive == "superquadric":
        e = sum(layer.exps) / 2; v *= min(1.9, max(0.5, 1.0 + 0.9 * (1 - e)))                     # box-like fills more, pinched less
    elif layer.primitive == "shard":
        v *= 0.7
    if layer.shell > 0:
        v *= min(1.0, 3 * layer.shell / max(min(a), 1e-3))
    return max(v, 1.0)


def render_layer(layer: Layer, shape, g: torch.Generator, device, step: float = 0.5, exclude: torch.Tensor | None = None):
    """The layer's occupancy ``(Z, Y, X)`` (a volume fraction: 1 at the centre of an isolated shape, overlaps capped at
    1) and shape tensor ``(6, Z, Y, X)`` (trace 1; the local mean of the deposited shapes' tensors). Shapes grow from
    random seeds in the region along the field (lines, blobs) or as discs perpendicular to it (sheets); points in
    ``exclude`` are not deposited (a fibre yielding to another population's fibre)."""
    if layer.kind == "solid":
        occ, tensor, layer.desc = render_solids(layer, shape, g, device, exclude)
        return occ, tensor
    if layer.hollow and layer.kind == "line":
        layer.section = "ring"
    if layer.kind == "line" and layer.section != "disc":
        return _render_section(layer, shape, g, device, step, exclude)
    seeds = _seeds(layer, shape, g, device)
    if layer.kind in ("line", "blob"):
        tens = line_tensor if layer.kind == "line" else (lambda d: blob_tensor(d, layer.radius, layer.length, step))
        d_seed = _at(seeds, layer.field, shape)
        P = [seeds]; V = [tens(d_seed)]
        for sgn in (1.0, -1.0):
            p = seeds.clone(); d0 = d_seed
            for _ in range(_line_half(layer.length, step)):
                d = _at(p, layer.field, shape)
                d = torch.where((d * d0).sum(1, keepdim=True) < 0, -d, d)                       # keep the sense of travel along an axial field
                d0 = d
                p = p + sgn * step * d; P.append(p.clone()); V.append(tens(d))
        P = torch.cat(P); V = torch.cat(V)
    else:
        n = _at(seeds, layer.field, shape)
        a = torch.where(n[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device=device).expand_as(n), torch.tensor([[0, 1.0, 0]], device=device).expand_as(n))
        e1 = torch.linalg.cross(n, a); e1 = e1 / e1.norm(dim=1, keepdim=True).clamp_min(1e-6); e2 = torch.linalg.cross(n, e1)
        off = _sheet_offsets(layer.length, step, device)                                       # the seed itself is not deposited twice
        P = (seeds[:, None] + off[None, :, :1] * e1[:, None] + off[None, :, 1:] * e2[:, None]).reshape(-1, 3)
        V = sheet_tensor(n)[:, None].expand(-1, off.shape[0], -1).reshape(-1, 6)
        if layer.bilayer > 0:                                                                  # two leaflets, the gap between them hollow
            nn_ = n[:, None].expand(-1, off.shape[0], -1).reshape(-1, 3)
            P = torch.cat([P + 0.5 * layer.bilayer * nn_, P - 0.5 * layer.bilayer * nn_]); V = torch.cat([V, V])
    idx = P.round().long()
    inside = ((idx >= 0) & (idx < torch.tensor(shape, device=device))).all(1)
    for i, n_ in enumerate(shape):
        idx[:, i] = idx[:, i].clamp(0, n_ - 1)
    keep = inside & layer.region[idx[:, 0], idx[:, 1], idx[:, 2]]                                 # shapes are confined to the layer's region
    if exclude is not None:
        keep &= ~exclude[idx[:, 0], idx[:, 1], idx[:, 2]]                                       # and yield where another population's fibre is
    P, V = P[keep], V[keep]
    dens, T = _splat(P, V, shape, device)
    r = max(layer.radius, 0.5)
    db = gaussian(dens, r, truncate=3.0); Tb = gaussian(T, r, truncate=3.0)
    tensor = torch.where(db[None] > 1e-6, Tb / db.clamp_min(1e-6)[None], eye6(device).view(6, 1, 1, 1) / 3)
    occ = (db / _peak(layer.kind, r, layer.length, step, device)).clamp(0, 1)
    hol = layer.bilayer / (layer.bilayer + 2 * r) if (layer.kind == "sheet" and layer.bilayer > 0) else 0.0
    layer.desc = torch.tensor([r, hol, 0.0], device=device).view(3, 1, 1, 1) * (occ > 0)[None]
    return occ, tensor


def _render_section(layer: Layer, shape, g, device, step: float, exclude):
    """A line population with a non-round or hollow cross-section: the section (ring, ellipse, superellipse) filled
    with points on a grid of spacing ``h`` and swept along each streamline, its frame turning with the path; occupancy
    = the points' density over the sampling density (1 inside a shape, an anti-aliased edge), label ``t tᵀ``."""
    seeds = _seeds(layer, shape, g, device)
    r = max(layer.radius, 0.5); h = 0.5 if r <= 3 else r / 6
    if layer.section == "ring":
        layer.wall = max(layer.wall, min(1.0, 1.0 / r))                                          # a wall of a voxel or more: thinner cannot be drawn on the grid
    P, D = _paths(layer, seeds, shape, step)                                                     # (n, s, 3)
    off = _section_offsets(layer, h, device)                                                    # (m, 2)
    ref = torch.nn.functional.normalize(torch.randn((P.shape[0], 1, 3), device=device, generator=g), dim=-1)
    e1, e2 = _frame(D, ref)                                                                     # (n, s, 3)
    n_pts = P.shape[0] * P.shape[1] * off.shape[0]
    stride = max(1, int(math.ceil(n_pts / 3e7)))                                                # a cap on the points: sparser sampling of the path
    P, D, e1, e2 = P[:, ::stride], D[:, ::stride], e1[:, ::stride], e2[:, ::stride]
    Q = P[:, :, None] + off[None, None, :, :1] * e1[:, :, None] + off[None, None, :, 1:] * e2[:, :, None]   # (n, s, m, 3)
    V = line_tensor(D)[:, :, None].expand(-1, -1, off.shape[0], -1)
    Q, V = Q.reshape(-1, 3), V.reshape(-1, 6)
    keep = _keep(Q, layer, shape, exclude); Q, V = Q[keep], V[keep]
    dens, T = _splat(Q, V, shape, device)
    db = gaussian(dens, 0.6, truncate=3.0); Tb = gaussian(T, 0.6, truncate=3.0)
    occ = (db * (step * stride) * h * h).clamp(0, 1)                                            # points per voxel at full sampling: 1 / (step h²)
    tensor = torch.where(db[None] > 1e-6, Tb / db.clamp_min(1e-6)[None], eye6(device).view(6, 1, 1, 1) / 3)
    hol = (1 - layer.wall) if layer.section == "ring" else 0.0
    thick = r * (layer.section_aspect if layer.section == "ellipse" else 1.0)
    layer.desc = torch.tensor([thick, hol, section_angularity(layer)], device=device).view(3, 1, 1, 1) * (occ > 0)[None]
    return occ, tensor


CONTENT = ("general", "mri", "histology")


KINDS_DRAWN = ("line", "sheet", "blob", "solid")


def random_layer(shape, g: torch.Generator, device, kind: str | None = None, scale: float = 1.0, content: str = "general",
                 field: torch.Tensor | None = None, density: float | None = None, region: torch.Tensor | None = None) -> Layer:
    """A random population from the shape grammar. ``scale`` multiplies every size (the voxel size within the trained
    range); bends and the field's coherence are drawn in voxels, independent of the scale, so tight bends occur at
    every scale.

    * ``line``: swept along the field -- a solid Gaussian-profile disc (fibres), or a sharp cross-section: a ring
      (hollow fibres, vessels; wall 0.2-0.6 of the radius), an ellipse (ribbons), a superellipse (square-ish);
    * ``sheet``: discs perpendicular to the field, one leaflet or two (bilayer: a hollow membrane);
    * ``blob``: short Gaussian extrusions (cells, nuclei, grains);
    * ``solid``: compact objects from signed distance functions -- superquadrics (smooth ellipsoids to box-like),
      convex polyhedra (crystals), shards; isotropic, needle or platelet proportions; solid or shells (vesicles)."""
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    pick = lambda opts, w: opts[int(torch.multinomial(torch.tensor(w, device=device, dtype=torch.float32), 1, generator=g))]
    if kind is None:
        kind = pick(KINDS_DRAWN, [0.4, 0.4, 0.2, 0.0] if content == "mri" else [0.35, 0.25, 0.15, 0.25])
    radius = max(lu(0.6, 3.0) * scale, 1.2 if content == "mri" else 0.6)
    coh, bend = 0.0, 0.0
    if field is None:
        if kind == "line" and u(0, 1) < 0.3:
            bend = lu(8, 60); field = arc_field(shape, bend, g, device)
        else:
            coh = max(lu(6, 60), 3 * radius); field = direction_field(shape, coh, g, device)
    extra = {}
    if kind == "line":
        length = lu(20, 120) * scale
        sec = pick(("disc", "ring", "ellipse", "superellipse"), [0.55, 0.25, 0.1, 0.1] if content != "mri" else [0.5, 0.4, 0.05, 0.05])
        if sec != "disc":
            radius = max(radius, lu(1.2, 4.0) * scale)                                               # a resolvable section
            extra = {"section": sec, "wall": max(u(0.2, 0.6), 1.0 / radius), "section_aspect": lu(0.25, 0.7), "section_exp": lu(0.6, 8.0)}
    elif kind == "sheet":
        length = lu(6, 30) * scale
        if u(0, 1) < 0.2:
            extra = {"bilayer": u(1.5, 4.0) * radius}
    elif kind == "blob":
        length = radius * lu(0.5, 2.5)
    else:
        radius = lu(1.0, 6.0) * scale
        shape_kind = pick(("iso", "needle", "platelet", "any"), [0.35, 0.25, 0.2, 0.2])
        axes = {"iso": (1.0, 1.0, 1.0), "needle": (1.0, 1.0, lu(2.0, 6.0)), "platelet": (lu(2.0, 5.0), lu(2.0, 5.0), 1.0),
                "any": (lu(0.6, 2.0), lu(0.6, 2.0), lu(0.6, 3.0))}[shape_kind]
        prim = pick(("superquadric", "polyhedron", "shard"), [0.5, 0.3, 0.2])
        exps = (lu(0.2, 2.0), lu(0.2, 2.0)) if u(0, 1) < 0.6 else (1.0, 1.0)
        shell = u(0.8, 0.45 * radius * min(axes)) if (u(0, 1) < 0.25 and radius * min(axes) > 2.5) else 0.0
        length = 2 * radius * max(axes)
        extra = {"primitive": prim, "axes": axes, "exps": exps, "shell": shell}
    phi = lu(0.02, 0.3 if kind == "solid" else 0.5) * (0.3 if content == "mri" and kind == "line" else 1.0)   # the layer's volume fraction (MRI: sparse tubes)
    if region is None:
        region = region_mask(shape, lu(8, 40) * scale, u(0.3, 1.0) if u(0, 1) < 0.7 else 1.0, g, device)
    contrast = u(15, 140) * (1 if u(0, 1) < 0.5 else -1)                                       # faint to strong
    occluding = u(0, 1) < 0.2
    if occluding:
        contrast = u(30, 225)
    L = Layer(kind, field, radius, length, 0.0, region, contrast, occluding, False, coh, bend, **extra)
    L.density = layer_density(L, phi) if density is None else density
    return L


# ---------------------------------------------------------------------------------------------------------------
# a sample
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Sample:
    image: torch.Tensor            # (Z, Y, X) float, grey levels (clean)
    layers: list                   # Layer, each with .occupancy and .tensor; the last two are the interface and isotropic states
    background: torch.Tensor       # (Z, Y, X) the background before the layers
    labels: torch.Tensor           # (Z, Y, X) the background's label map
    style: str = "regions"         # the background style
    scale: float = 1.0             # the size multiplier the sample was drawn at
    content: str = "general"       # the content prior (general / mri / histology)
    record: dict = field(default_factory=dict)
    regions: torch.Tensor | None = None     # (Z, Y, X) the background's region means (no texture): what the interface is a step of
    budget: torch.Tensor | None = None      # (Z, Y, X) the volume budget's divisor (>= 1): the targets undo it (TARGETS_BUDGET)


BACKGROUNDS = ("regions", "tissue", "cells")
STEP_FULL = 15.0                   # an interface step of this many grey levels or more is a full interface (the layers' contrast floor)


def interface(R: torch.Tensor, sigma: float = 1.5):
    """The interface state of a piecewise-constant region image ``R`` (the region means, no texture): the normal from
    the gradient of ``R`` blurred by ``sigma`` voxels, the weight from its magnitude -- at a planar step of ``Δ`` grey
    levels it is ``min(1, Δ / STEP_FULL) * exp(-d² / 2σ²)`` at distance ``d``, whatever the step's orientation or the
    sample's scale."""
    sm = gaussian(R, sigma, truncate=3.0); gr = torch.stack(torch.gradient(sm, dim=(0, 1, 2)), 0); mag = gr.norm(dim=0)
    w = (mag * math.sqrt(2 * math.pi) * sigma / STEP_FULL).clamp(0, 1)
    ez = torch.zeros_like(gr); ez[0] = 1.0                                                      # any unit normal where there is no step (weight 0)
    return torch.where(mag[None] > 1e-6, gr / mag.clamp_min(1e-9)[None], ez), w


def background(shape, style: str, g: torch.Generator, device, scale: float = 1.0, content: str = "general", nprng=None, natural: bool = False):
    """The background before the layers: ``regions`` (anatomix: a random label map, an intensity per label),
    ``tissue`` (two or three broad regions -- grey/white matter -- with texture), ``cells`` (a cell stain: small blobs,
    dark or bright, their density optionally laminar along a random direction). Returns the image, the label map, the
    region means, and the extra layers the style brings (the cells as a blob population). With ``nprng`` (a numpy
    generator) every scalar is drawn on the CPU: no device synchronisation. ``natural`` (the scene generator, 29 Oct):
    compartments larger (15-50 voxels), their contrast a spread of 10-60 grey levels about a common level (was 40-210),
    their edges softened by 1-2 voxels and textured inside; tissue regions likewise (spread 15-80)."""
    if nprng is not None:
        u = lambda a, b: float(nprng.uniform(a, b))
        ri = lambda a, b: int(nprng.integers(a, b))
    else:
        u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
        ri = lambda a, b: int(torch.randint(a, b, (1,), generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    extra = []
    if style == "flat":                                                                        # a uniform background (feasibility tests)
        lab = torch.zeros(shape, dtype=torch.long, device=device); R = torch.full(shape, u(70, 190), device=device)
        img = R + 2 * smooth_noise(shape, 1.0, g, device)
    elif style == "regions" and natural:
        nl = ri(3, 9); lab = label_map(shape, nl, u(15, 50) * scale, g, device); mid = u(70, 190); spn = lu(10, 60)
        R = (mid + spn * (torch.rand(nl, device=device, generator=g) - 0.5))[lab]; R = gaussian(R, u(1.0, 2.0))
        img = R + u(4, 15) * smooth_noise(shape, u(1.0, 3.0) * scale, g, device) + u(2, 8) * smooth_noise(shape, u(4.0, 10.0) * scale, g, device)
    elif style == "regions":
        nl = ri(3, 9); lab = label_map(shape, nl, u(6, 25) * scale, g, device)
        R = (40 + 170 * torch.rand(nl, device=device, generator=g))[lab]; img = R + 6 * smooth_noise(shape, 2.0 * scale, g, device)
    elif style == "tissue":
        nl = ri(2, 4); lab = label_map(shape, nl, u(25, 60) * scale, g, device)
        if natural:
            mid = u(70, 190); spn = lu(15, 80)
            R = (mid + spn * (torch.rand(nl, device=device, generator=g) - 0.5))[lab]; R = gaussian(R, u(1.0, 2.0))
        else:
            R = (60 + 150 * torch.rand(nl, device=device, generator=g))[lab]
        img = R + u(8, 25) * smooth_noise(shape, u(1.0, 3.0) * scale, g, device)
    else:                                                                                      # cells: a blob population on a flat background
        lab = torch.zeros(shape, dtype=torch.long, device=device); R = torch.full(shape, u(70, 190), device=device)
        img = R + 5 * smooth_noise(shape, 1.0, g, device)
        n = smooth_noise(shape, u(1.2, 2.5) * scale, g, device)
        if u(0, 1) < (0.6 if content == "histology" else 0.25):                                # laminae: the cell density varies along one direction
            d = _unit(g, device); x = (_coords(shape, device) * d.view(3, 1, 1, 1)).sum(0)
            n = n + u(0.5, 1.5) * torch.sin(2 * math.pi * x / lu(20, 80) + u(0, 2 * math.pi))
        cells = gaussian((n > u(0.8, 1.6)).float(), 0.8 * scale).clamp(0, 1)
        r = 1.2 * scale
        L = Layer("blob", direction_field(shape, 8.0, g, device), r, r, 0.0, torch.ones(shape, dtype=torch.bool, device=device),
                  (u(15, 60) if natural else u(50, 140)) * (1 if u(0, 1) < 0.35 else -1), occupancy=cells,   # (natural: a stain, not
                  tensor=eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3)                                # the brightest thing there)
        L.desc = torch.tensor([r, 0.0, 0.0], device=device).view(3, 1, 1, 1) * (cells > 0)[None]
        extra.append(L)
    return img, lab, R, extra


#: the volume budget in the targets: where populations overlap past one voxel of material, the image shares the voxel
#: between them (each scaled by the overlap) -- the targets need not. False: each population keeps its own occupancy in
#: the targets (occlusion kept: a solid displaces what it covers), so crossing populations run unbroken through each
#: other and the mass of a crossing is the sum of both; True: the targets share the voxel as the image does (before:
#: each population carved out of the other -- a hole in A in B's shape, B faint inside A -- the crossing removed)
TARGETS_BUDGET = False


def _composite(bg: torch.Tensor, R: torch.Tensor, layers: list, device, store: dict | None = None):
    """Composite the shape layers over the background as volume fractions, in drawing order: an occluder hides what is
    under it (those layers' occupancies are attenuated by ``1 - o``), then any voxel holding more than one voxel of
    shapes is scaled back to one. Appends the interface and the isotropic states; returns the image."""
    shape = bg.shape; vis = torch.ones(shape, device=device)
    for j, L in enumerate(layers):
        if L.occluding:
            for P in layers[:j]:
                P.occupancy = P.occupancy * (1 - L.occupancy)
            vis = vis * (1 - L.occupancy)
    tot = torch.stack([L.occupancy for L in layers]).sum(0) if layers else torch.zeros(shape, device=device)
    over = tot.clamp_min(1.0)
    if store is not None:
        store["over"] = over; store["occluded"] = [L.occupancy.clone() for L in layers]               # occlusion only (no budget)
    for L in layers:
        L.occupancy = L.occupancy / over
    occl = sum((L.occupancy for L in layers if L.occluding), torch.zeros(shape, device=device))
    img = bg * (1 - occl)
    for L in layers:
        img = img + L.contrast * L.occupancy
    rest = (1 - torch.stack([L.occupancy for L in layers]).sum(0)).clamp(0, 1) if layers else torch.ones(shape, device=device)
    normal, bw = interface(R)
    ones = torch.ones(shape, dtype=torch.bool, device=device)
    # the interface is a boundary, not a volume: its weight is the step's, hidden only where an occluder hides the step
    # (as in the image, where shapes add to the background and the step shows through them); the volume budget is the
    # shapes' and the isotropic remainder's, summing to one
    layers.append(Layer("sheet", normal, 1.0, 1.0, 0.0, ones, 0.0, role="interface", occupancy=bw * vis, tensor=sheet_tensor(normal.movedim(0, -1)).movedim(-1, 0)))
    layers.append(Layer("blob", normal, 1.0, 1.0, 0.0, ones, 0.0, role="isotropic", occupancy=rest, tensor=eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3))
    for L in layers:
        if L.desc is None:                                                                        # layers made outside render_layer
            L.desc = torch.tensor([max(L.radius, 0.5) if L.role == "shape" else 0.0, 0.0, 0.0], device=device).view(3, 1, 1, 1) * (L.occupancy > 0)[None]
    return img


def sample(shape=(128, 128, 128), k_layers: int | None = None, seed: int = 0, device="cuda", scale: float | None = None, style: str | None = None,
           content: str = "general", crossing: bool | None = None, cross_deg: float | None = None, kinds: tuple | None = None) -> Sample:
    """A clean sample: a background, an optional crossing group (``crossing``; None draws it with p 0.5: two or three
    line layers sharing one region, the first two at ``cross_deg``, None drawing 20-90°), ``k_layers`` further random
    layers (None draws 0-3; ``kinds`` restricts their kinds), composited as volume fractions, plus the interface and
    isotropic states so that every voxel's state is defined. ``style`` ``"flat"``: a uniform background (tests). ``scale`` multiplies every size (None draws it: 0.5-2.5, 1.5-4 for ``content="mri"``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    if scale is None:
        scale = lu(1.5, 4.0) if content == "mri" else lu(0.5, 2.5)
    if style is None:
        w = {"mri": [0.5, 0.5, 0.0], "histology": [0.2, 0.2, 0.6]}.get(content, [1 / 3, 1 / 3, 1 / 3])
        style = BACKGROUNDS[int(torch.multinomial(torch.tensor(w, device=device), 1, generator=g))]
    bg, lab, R, layers = background(shape, style, g, device, scale, content)
    rec = {"style": style, "scale": round(scale, 3), "content": content}
    if crossing is None:
        crossing = u(0, 1) < 0.5
    if crossing:
        n = 2 if u(0, 1) < 0.75 else 3
        region = region_mask(shape, lu(15, 50) * scale, u(0.4, 1.0) if u(0, 1) < 0.6 else 1.0, g, device)
        phi = u(0.1, 0.5) * (0.4 if content == "mri" else 1.0)                                # the fibres' volume fraction in the region, shared by the populations
        w = torch.rand(n, device=device, generator=g) + 0.05; w = (w / w.sum()).tolist()      # its split: dominant and non-dominant populations (to ~1:20)
        independent = cross_deg is None and u(0, 1) < 0.4                                      # independent fields: the angle varies over the region
        theta = cross_deg if cross_deg is not None else u(20, 90)
        coh = lu(10, 60)
        L0 = random_layer(shape, g, device, kind="line", scale=scale, content=content, region=region)
        if independent:
            fields = [L0.field] + [direction_field(shape, lu(10, 60), g, device) for _ in range(1, n)]
        else:
            fields = [L0.field, companion_field(L0.field, theta, coh, g, device)]
            if n == 3:
                fields.append(companion_field(L0.field, u(20, 90), coh, g, device))
        group = [L0] + [random_layer(shape, g, device, kind="line", scale=scale, content=content, field=fields[k], region=region) for k in range(1, n)]
        for Lk, wk in zip(group, w):                                                            # seeds per 1000 voxels for the layer's share of the volume
            Lk.density = layer_density(Lk, phi * wk)
        con = u(15, 140) * (1 if u(0, 1) < 0.5 else -1)
        same = u(0, 1) < 0.8                                                                     # usually one material
        for Lk in group:
            Lk.occluding = False; Lk.group = 0
            Lk.contrast = con * u(0.7, 1.3) if same else u(15, 140) * (1 if u(0, 1) < 0.5 else -1)
        order = torch.randperm(n, generator=g, device=device).tolist(); have = torch.zeros(shape, device=device)
        for k in order:
            Lk = group[k]
            Lk.occupancy, Lk.tensor = render_layer(Lk, shape, g, device, exclude=have > 0.5)      # later fibres yield where earlier ones are
            have = have + Lk.occupancy
        layers += group
        rec["crossing"] = {"n": n, "theta_deg": None if independent else round(theta, 1), "independent": independent, "weights": [round(x, 3) for x in w], "fraction": round(phi, 3)}
    K = k_layers if k_layers is not None else int(torch.randint(0, 4, (1,), generator=g, device=device))
    for _ in range(K):
        kind = None if kinds is None else kinds[int(torch.randint(0, len(kinds), (1,), generator=g, device=device))]
        L = random_layer(shape, g, device, kind=kind, scale=scale, content=content)
        L.occupancy, L.tensor = render_layer(L, shape, g, device)
        layers.append(L)
    img = _composite(bg, R, layers, device)
    return Sample(img, layers, bg, lab, style, scale, content, rec, R)


def boundary_sample(shape=(96, 96, 96), step: float = 60.0, contrast: float = 20.0, angle_deg: float = 30.0, seed: int = 0, device="cuda",
                    radius: float = 1.0, density: float = 1.5) -> Sample:
    """The boundary case, controlled: two half-spaces (an intensity ``step`` across z = Z/2) and one line population of
    ``contrast`` whose direction is ``angle_deg`` from the boundary normal, crossing it. Built with the same
    compositing and interface definition as :func:`sample`: at the step the target has both states."""
    g = torch.Generator(device=device).manual_seed(seed)
    Z, Y, X = shape
    lab = (torch.arange(Z, device=device) >= Z // 2).long().view(Z, 1, 1).expand(shape).contiguous()
    base = 90.0; R = base + step * lab.float()
    bg = R + 4 * smooth_noise(shape, 2.0, g, device)
    t = math.radians(angle_deg); d = torch.tensor([math.cos(t), 0.0, math.sin(t)], device=device).view(3, 1, 1, 1).expand((3,) + tuple(shape)).contiguous()
    L = Layer("line", d, radius, float(Z) * 1.5, density, torch.ones(shape, dtype=torch.bool, device=device), contrast)
    L.occupancy, L.tensor = render_layer(L, shape, g, device)
    layers = [L]
    img = _composite(bg, R, layers, device)
    return Sample(img, layers, bg, lab, "boundary", 1.0, "general", {"step": step, "contrast": contrast, "angle_deg": angle_deg}, R)


# ---------------------------------------------------------------------------------------------------------------
# nuisances
# ---------------------------------------------------------------------------------------------------------------

def _warp_grid(disp: torch.Tensor):
    """``disp (3, Z, Y, X)`` voxel displacements -> a grid_sample grid ``(1, Z, Y, X, 3)`` in (x, y, z) order."""
    _, Z, Y, X = disp.shape
    p = _coords((Z, Y, X), disp.device) + disp
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=disp.device).view(3, 1, 1, 1)
    return (p * scale - 1.0).flip(0).movedim(0, -1)[None]


def _resample(vol: torch.Tensor, grid: torch.Tensor, mode="bilinear", padding: str = "border") -> torch.Tensor:
    v = vol[None, None] if vol.dim() == 3 else vol[None]
    out = Fn.grid_sample(v, grid, mode=mode, padding_mode=padding, align_corners=True)
    return out[0, 0] if vol.dim() == 3 else out[0]


def _jacobian(disp: torch.Tensor) -> torch.Tensor:
    """``I + ∂disp/∂x`` as ``(Z, Y, X, 3, 3)``."""
    J = torch.stack([torch.stack(torch.gradient(disp[i], dim=(0, 1, 2)), 0) for i in range(3)])
    return J.permute(2, 3, 4, 0, 1) + torch.eye(3, device=disp.device)


def gibbs(img: torch.Tensor, axes, cutoff: float) -> torch.Tensor:
    """k-space truncation along ``axes`` at ``cutoff`` cycles/voxel (Nyquist 0.5): an acquisition at lower resolution
    reconstructed by zero filling. Each axis is mirror-padded first, so the block's faces do not ring."""
    for a in axes:
        n = img.shape[a]
        x = torch.cat([img, img.flip(a)], dim=a)
        F = torch.fft.rfft(x, dim=a)
        k = torch.fft.rfftfreq(2 * n, device=img.device)
        shp = [1] * img.dim(); shp[a] = -1
        x = torch.fft.irfft(F * (k <= cutoff).view(shp).to(F.dtype), n=2 * n, dim=a)
        img = x.narrow(a, 0, n)
    return img


def _shift2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """A 2-D slice shifted by whole voxels, edges replicated (no wrap-around)."""
    p = max(abs(dy), abs(dx))
    if p == 0:
        return x
    xp = Fn.pad(x[None, None], (p, p, p, p), mode="replicate")[0, 0]
    H, W = x.shape
    return xp[p - dy: p - dy + H, p - dx: p - dx + W]


#: 29 Oct: a wider corruption pool on the universal path (real crops often look like oriented noise at many scales under
#: confounding gradients and structures): per key the probability at full strength (scaled by the strength, as every
#: destructive nuisance), the max of the default here and the modality's entry in :data:`EXTREME_MOD`. Never drawn under
#: the HiP-CT profile.
#: 29 Oct: the strength each modality's whole profile is drawn at, per unit of the requested strength -- calibrated so a
#: profile keeps >= ~0.85 of the clean image's local structure (|corr|, scripts/check_artefact_strength.py MODALITY=1) at 0.3 and
#: ~0.5-0.6 at 1.0 where it went below
MOD_STRENGTH = {"fluorescence": 0.7, "ct": 0.85, "histology": 0.85, "mri": 0.9, "mixed": 0.75, "em": 0.75, "cryoet": 0.75, "hipct": 1.0}   # (measured on local structure, a bias field's shading not counted;
                                                                             # the rest at 1: they keep >= 0.95 at 0.3 and ~0.5-0.9 at 1.0)
EXTREME_P = {"bh_streaks": 0.05, "scatter": 0.06, "metal": 0.04, "us_speckle": 0.08, "oct_speckle": 0.05, "oriented_noise": 0.15,
             "motion": 0.10, "slice_motion": 0.08, "fold_lut": 0.10, "posterise": 0.08, "clip": 0.08, "local_reversal": 0.08}
EXTREME_MOD = {"ct": {"bh_streaks": 0.3, "scatter": 0.3, "metal": 0.15}, "hipct": {"bh_streaks": 0.1, "scatter": 0.2, "metal": 0.05},
               "mixed": {k: 0.15 for k in EXTREME_P}, "histology": {"slice_motion": 0.25}, "em": {"slice_motion": 0.25},
               "fluorescence": {"motion": 0.15, "slice_motion": 0.15}, "mri": {"motion": 0.2, "slice_motion": 0.15}}


def _rand_rot(g: torch.Generator, device) -> torch.Tensor:
    """A random rotation (3, 3) (columns: its axes)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, device=device))
    return q * torch.sign(torch.diagonal(r))[None]


def aniso_noise(shape, sig3, Rm: torch.Tensor, g: torch.Generator, device) -> torch.Tensor:
    """Gaussian noise correlated by an anisotropic Gaussian kernel (sigmas ``sig3`` along the columns of ``Rm``): an
    oriented grain at any direction (Fourier filtered); zero mean, unit spread."""
    ks = [torch.fft.fftfreq(n, device=device) for n in shape]
    K = torch.stack(torch.meshgrid(*ks, indexing="ij"), -1) @ Rm                                  # frequencies along the axes
    H = torch.exp(-2 * math.pi ** 2 * sum((K[..., i] * float(sig3[i])) ** 2 for i in range(3)))
    n = torch.fft.ifftn(torch.fft.fftn(torch.randn(shape, generator=g, device=device)) * H).real
    return (n - n.mean()) / n.std().clamp_min(1e-9)


def line_blur(img: torch.Tensor, d, length: float, n: int | None = None) -> torch.Tensor:
    """Motion blur: the mean of ``img`` (2-D or 3-D) along a segment of ``length`` voxels in direction ``d`` (border
    padding)."""
    nd = img.dim(); sh = img.shape; dev = img.device
    n = n or max(3, int(2 * length) + 1)
    base = torch.stack(torch.meshgrid(*(torch.arange(m, device=dev, dtype=torch.float32) for m in sh), indexing="ij"), -1)
    sz = torch.tensor(sh, device=dev, dtype=torch.float32); dv = torch.as_tensor(d, device=dev, dtype=torch.float32)
    acc = torch.zeros_like(img, dtype=torch.float32); x = img.float()[None, None]
    for t in torch.linspace(-length / 2, length / 2, n).tolist():
        gr = (2 * (base + t * dv) / (sz - 1).clamp_min(1) - 1).flip(-1)[None]
        acc += Fn.grid_sample(x, gr, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    return acc / n


def _seg_dist(yy, xx, p, q):
    """In-plane distance of the grid (yy, xx) to the segment p-q (2-vectors)."""
    py, px = float(p[0]), float(p[1]); dy, dx = float(q[0]) - py, float(q[1]) - px
    L2 = max(dy * dy + dx * dx, 1e-6)
    t = (((yy - py) * dy + (xx - px) * dx) / L2).clamp(0, 1)
    return torch.sqrt((yy - py - t * dy) ** 2 + (xx - px - t * dx) ** 2)


def _bright(img: torch.Tensor, n: int, q: float, g: torch.Generator) -> torch.Tensor:
    """``n`` random voxels among the brightest ``q`` of ``img`` (the dense objects of a CT): ``(n, 3)``."""
    idx = torch.nonzero(img >= _quantile(img, 1 - q))
    if idx.shape[0] == 0:
        idx = torch.nonzero(img >= img.max())
    return idx[torch.randint(0, idx.shape[0], (n,), generator=g, device=img.device)].float()


def _quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten()[:: max(1, x.numel() // 1_000_000)], q))


def _corr_volume(r: torch.Tensor) -> float:
    """The noise's correlation volume (voxels) from the lag-1 autocorrelation ``ρ`` of a residual along each axis, for
    Gaussian-correlated noise (white noise blurred by σ: ``ρ(d) = exp(-d² / 4σ²)``, volume ``(2√π σ)³``, at least one
    voxel per axis): 1 for white noise, larger after a PSF, thick slices or smooth clutter."""
    v = 1.0
    for a in range(3):
        x0, x1 = r.narrow(a, 0, r.shape[a] - 1), r.narrow(a, 1, r.shape[a] - 1)
        rho = float(((x0 - x0.mean()) * (x1 - x1.mean())).mean() / (x0.std() * x1.std()).clamp_min(1e-9))
        rho = min(max(rho, 1e-6), 0.99)
        sig = math.sqrt(-1.0 / (4.0 * math.log(rho)))
        v *= max(1.0, 2.0 * math.sqrt(math.pi) * sig)
    return v


def _corr_volume_local(r: torch.Tensor, pool) -> torch.Tensor:
    """:func:`_corr_volume` per window: the lag-1 autocorrelation of the residual along each axis pooled with the
    target's window (``pool``), so a thick-sliced or blurred part of a block is not credited with white-noise evidence
    and a clean part is not penalised for another part's correlation. ``(c...)``."""
    v = None; rr = pool((r * r)[None])[0].clamp_min(1e-12)
    for a in range(3):
        x1 = torch.roll(r, -1, dims=a)
        rho = (pool((r * x1)[None])[0] / rr).clamp(1e-6, 0.99)
        sig = torch.sqrt(-1.0 / (4.0 * torch.log(rho)))
        f = (2.0 * math.sqrt(math.pi) * sig).clamp_min(1.0)
        v = f if v is None else v * f
    return v


def _vis_t(img, regressors, idx, s_hp, sigma_v, step, local_corr):
    """The t-statistics ``(c..., len(idx))`` of a joint regression of the high-passed image on the high-passed
    regressors ``idx`` in a Gaussian window at each lattice cell (see :func:`visibility`)."""
    dev = img.device; shp = img.shape
    hp = lambda x: x - gaussian(x, s_hp, truncate=3.0)
    y = hp(img); A = hp(regressors[idx]); n = len(idx)
    pool = lambda x: Fn.avg_pool3d(gaussian(x, sigma_v, truncate=3.0)[None], step)[0]
    Gm = torch.zeros((n, n) + tuple(s // step for s in shp), device=dev)
    for i in range(n):
        for j in range(i, n):
            Gm[i, j] = Gm[j, i] = pool((A[i] * A[j])[None])[0]
    bm = pool(A * y[None]); yy = pool((y * y)[None])[0]
    G = Gm.movedim(0, -1).movedim(0, -1)                                                          # (c..., n, n)
    ridge = 1e-6 * G.diagonal(dim1=-2, dim2=-1).amax(-1).clamp_min(1e-12)[..., None, None] * torch.eye(n, device=dev)
    Gi = torch.linalg.inv(G + ridge)
    c = (Gi @ bm.movedim(0, -1)[..., None])[..., 0]                                                # (c..., n)
    res = (yy - (c * bm.movedim(0, -1)).sum(-1)).clamp_min(1e-9)                                  # the window's residual mean square
    c_vox = Fn.interpolate(c.movedim(-1, 0)[None], scale_factor=step, mode="trilinear", align_corners=False)[0][(slice(None),) + tuple(slice(0, s_) for s_ in shp)]
    resid = y - (c_vox * A).sum(0)                                                                  # the image less the layers' fitted signal: noise and clutter
    if local_corr:
        neff = ((2 * math.sqrt(math.pi) * sigma_v) ** 3 / _corr_volume_local(resid, pool))[..., None]
    else:
        neff = (2 * math.sqrt(math.pi) * sigma_v) ** 3 / _corr_volume(resid)                      # independent voxels in the window
    se = (res[..., None] * Gi.diagonal(dim1=-2, dim2=-1).clamp_min(0) / neff).sqrt()
    return c.abs() / se.clamp_min(1e-12)


#: a larger evidence window pools more voxels (t grows as their root): its t can be divided by this (e.g. sqrt(fw)) so
#: the larger window credits only a pattern that shows nearby -- off: the penalty gave crossings back to the background
#: (orientation visibility at crossing cores 0.33, target crossings resolved 12 % vs 88 % without it)
WINDOW_PENALTY = lambda fw: 1.0


def visibility(img: torch.Tensor, regressors: torch.Tensor, units, radii=None, sigma_v: float = 5.0, step: int = 4,
               t_lo: float = 3.0, t_hi: float = 8.0, return_t: bool = False, s_hp_max: float | None = None,
               per_layer_hp: bool = False, local_corr: bool = False, ladder=(1,), ladder_max: float = 64.0, window_ladder=(1,)) -> torch.Tensor:
    """Each layer's **local** visibility ``(K, Z, Y, X)``: how clearly its shapes show in the final image around each
    point. The image and the layers' image signatures (``regressors``: a shape layer's occupancy, an interface's
    region-means image) are high-passed at a scale matched to the shapes (σ = max(4, 2.5 × the largest radius in
    ``radii``; with ``per_layer_hp``, each layer at its own radius's scale -- layers grouped by scale, each group's
    t from a regression at its scale with every layer as a covariate -- so a large object does not blur a fine one's
    evidence)); in a Gaussian window at each lattice cell (``sigma_v``: the target's own window, so the evidence is the
    window's -- a featureless interior is not credited with its distant edges) the image is regressed on them jointly.
    Layer k's evidence is the t-statistic ``|c_k| / se_k`` of its fit: its contrast against the noise, times the root
    of its structure energy in the window, the standard error counting the window's voxels over the noise's correlation
    volume (``local_corr``: per window, else over the block) -- scale-free (a thick shape or an extended step gathers
    more evidence than a thin fibre at the same contrast to noise). Visibility = (t - ``t_lo``) / (``t_hi`` - ``t_lo``),
    clipped to [0, 1], interpolated to the voxels: 0 where the layer's shapes do not show (a featureless slab, a noisy
    or dark part of the block), 1 where they do.
    Two uses, two ranges (:data:`PRESENCE_T`, :data:`ORIENTATION_T`): whether the shapes are there at all -- a
    detection, t 3 to 8 -- and whether their orientation is recoverable -- measured on drawn samples against a local
    structure tensor's orientation error on single-population fibre voxels (about 40° (chance ~60°) up to t ≈ 10,
    24-32° at t 15-25, 17° beyond 25) and set somewhat below it, t 6 to 18.
    ``ladder``: further high-pass scales, multiples of each scale up to ``ladder_max`` voxels, the evidence the largest
    t over them -- a shape shows at *some* scale: the interior of a uniform band, of a densely packed bundle, is visible
    as material at the band's own scale although no edge or texture lies in the window (a single scale at the fibres'
    size credited only the edges: mass targets ran along both sides of bundles).
    ``window_ladder``: further evidence windows, multiples of ``sigma_v``, the evidence the largest t over them too -- a
    population shows *somewhere near*: where two packed bundles cross, the core is a featureless block of material (each
    layer's own pattern gone), yet both bundles show as they enter and leave it, so the crossing is inferable from that
    context; a single window of the target's size called the core invisible and the targets dropped the crossing.
    ``units[k] = None`` marks a layer not measured (visibility 1)."""
    K = regressors.shape[0]; dev = img.device; shp = img.shape
    v = torch.ones((K,) + tuple(shp), device=dev)
    idx = [k for k in range(K) if units[k] is not None]
    if not idx:
        return v
    cap = (lambda x: min(x, s_hp_max)) if s_hp_max is not None else (lambda x: x)
    if per_layer_hp and radii is not None:
        scale_of = {k: cap(max(4.0, 2.5 * float(radii[k]))) for k in idx}
        bins = {}
        for k in idx:                                                                             # scales within a factor of ~1.4 share a regression
            bins.setdefault(round(math.log2(scale_of[k]) * 2), []).append(k)
        t = torch.zeros(tuple(s // step for s in shp) + (len(idx),), device=dev)
        for key, ks in bins.items():
            s_hp = max(scale_of[k] for k in ks)
            tg = None
            for fw in window_ladder:
                for f_ in ladder:
                    if f_ == 1 or s_hp * f_ <= ladder_max * 1.001:
                        t_ = _vis_t(img, regressors, idx, s_hp * f_, sigma_v * fw, step, local_corr) / WINDOW_PENALTY(fw)
                        tg = t_ if tg is None else torch.maximum(tg, t_)
            for k in ks:
                j = idx.index(k); t[..., j] = tg[..., j]
    else:
        rmax = max([float(radii[k]) for k in idx] if radii is not None else [1.0])
        s0 = cap(max(4.0, 2.5 * rmax))
        t = None
        for fw in window_ladder:
            for f_ in ladder:
                if f_ == 1 or s0 * f_ <= ladder_max * 1.001:
                    t_ = _vis_t(img, regressors, idx, s0 * f_, sigma_v * fw, step, local_corr) / WINDOW_PENALTY(fw)
                    t = t_ if t is None else torch.maximum(t, t_)
    if return_t:                                                                                  # the evidence itself, upsampled (for calibration)
        tv = Fn.interpolate(t.movedim(-1, 0)[None], scale_factor=step, mode="trilinear", align_corners=False)[0]
        out = torch.zeros((K,) + tuple(shp), device=dev); out[idx] = tv[(slice(None),) + tuple(slice(0, s_) for s_ in shp)]
        return out
    vis = ((t - t_lo) / (t_hi - t_lo)).clamp(0, 1).movedim(-1, 0)                                 # (n, c...)
    up = Fn.interpolate(vis[None], scale_factor=step, mode="trilinear", align_corners=False)[0]   # cell i at voxel step*i + (step-1)/2
    v[idx] = up[(slice(None),) + tuple(slice(0, s) for s in shp)]
    return v


PRESENCE_T = (3.0, 8.0)            # t range over which a layer's shapes go from undetected to detected (its mass)
ORIENT_VIS = True                  # the generator's own orientation visibility (off: the trainer measures it on the image per query)
ORIENTATION_T = (6.0, 18.0)        # t range over which their orientation goes from unrecoverable to recoverable (its tensor): set below a
#: orientation visibility from the single-scale evidence (no high-pass ``ladder``) its range was calibrated on: the
#: ladder credits a uniform band's interior as visible material at coarse scales -- material, not orientation -- and
#: with it ``ovis`` saturated wherever a region shows at any scale, asking for kappa up to 600 where a structure tensor
#: finds no orientation (generator crossing arms 2026-10-27: 7 of 10 off by 30-71 deg). The window ladder is kept
#: (orientation visible nearby is context). Off: the shared t (as before).
ORIENT_SINGLE_SCALE = False
#: orientation visibility from each shape layer's fibre-scale texture (its occupancy less its own blur at 2x its
#: radius: the part of a population that carries its objects' orientation; a uniformly filled region has none inside),
#: regressed on the image at the single scale with the window ladder -- material that shows only as a region is not
#: credited with a visible orientation. Overrides ORIENT_SINGLE_SCALE. Off: the shared t.
ORIENT_FROM_TEXTURE = False
#: the entity targets from the geometry: every layer fully visible (presence and orientation), so the targets are the true
#: occupancy and the true tensors, whatever the image shows. The visibility rules were measured on the image *before* the
#: batch-time contrast routes (~95 % of images then get a destructive route), removed mass without giving it to the
#: background, and moved tensors toward isotropic -- changing a population's type, not only its confidence (26-32 % of
#: line populations' visible mass got girdle targets: generator review, 27 Oct). Evidence belongs in its own output.
TARGETS_FROM_GEOMETRY = False
                                   # local structure tensor's (8-25; sigma_int 3), which is a lower bound on what a 48-voxel receptive
                                   # field can integrate along a fibre -- to be re-measured with the trained model's own error against t


#: per modality, the probability of each nuisance; a draw picks a modality (or "mixed") and then also applies every other
#: nuisance with a small probability, so typical and atypical combinations are both seen
MODALITIES = {
    "hipct":     {"warp": 0.7, "invert": 0.5, "bend": 0.7, "bias": 0.6, "aniso_psf": 0.6, "resolution": 0.3, "poisson": 0.8, "gaussian": 0.9, "speckle": 0.1,
                  "rings": 0.6, "streaks": 0.5, "slice_jitter": 0.5, "fringes": 0.7, "cupping": 0.5, "seams": 0.3, "zingers": 0.4,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.1, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.1, "dust": 0.1, "mosaic": 0.0,
                  "slabs": 0.05, "drift": 0.0, "chatter": 0.0, "lsstripes": 0.05, "depth": 0.0, "lowdose": 0.0, "wedge": 0.0, "ctf": 0.0, "saturate": 0.1},
    "ct":        {"warp": 0.7, "invert": 0.4, "bend": 0.7, "bias": 0.5, "aniso_psf": 0.5, "resolution": 0.5, "poisson": 0.8, "gaussian": 0.9, "speckle": 0.2,
                  "rings": 0.5, "streaks": 0.6, "slice_jitter": 0.3, "fringes": 0.2, "cupping": 0.7, "seams": 0.2, "zingers": 0.3,
                  "rician": 0.0, "gibbs": 0.1, "ghosting": 0.0, "thick": 0.2, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.1, "dust": 0.1, "mosaic": 0.0,
                  "slabs": 0.05, "drift": 0.0, "chatter": 0.0, "lsstripes": 0.0, "depth": 0.0, "lowdose": 0.0, "wedge": 0.0, "ctf": 0.0, "saturate": 0.15},
    "mri":       {"warp": 0.8, "invert": 0.5, "bend": 0.8, "bias": 0.9, "aniso_psf": 0.5, "resolution": 0.6, "poisson": 0.0, "gaussian": 0.3, "speckle": 0.0,
                  "rings": 0.0, "streaks": 0.0, "slice_jitter": 0.2, "fringes": 0.0, "cupping": 0.0, "seams": 0.0, "zingers": 0.1,
                  "rician": 0.9, "gibbs": 0.6, "ghosting": 0.4, "thick": 0.6, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.0, "dust": 0.0, "mosaic": 0.0,
                  "slabs": 0.05, "drift": 0.0, "chatter": 0.0, "lsstripes": 0.0, "depth": 0.0, "lowdose": 0.0, "wedge": 0.0, "ctf": 0.0, "saturate": 0.05},
    "histology": {"warp": 0.9, "invert": 0.6, "bend": 0.8, "bias": 0.7, "aniso_psf": 0.6, "resolution": 0.5, "poisson": 0.3, "gaussian": 0.7, "speckle": 0.3,
                  "rings": 0.0, "streaks": 0.1, "slice_jitter": 0.8, "fringes": 0.0, "cupping": 0.0, "seams": 0.3, "zingers": 0.2,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.7, "misalign": 0.7, "missing": 0.3, "stain": 0.7, "stripes": 0.5, "dust": 0.5, "mosaic": 0.5,
                  "slabs": 0.2, "drift": 0.3, "chatter": 0.3, "lsstripes": 0.1, "depth": 0.0, "lowdose": 0.0, "wedge": 0.0, "ctf": 0.0, "saturate": 0.1},
    # serial-section TEM / SBEM / FIB-SEM: sections cut and imaged one by one (shifts, drift, flicker, lost sections and slabs,
    # knife chatter, curtaining), thick sections along the cutting axis, electron shot noise, a little detector saturation
    "em":        {"warp": 0.5, "invert": 0.5, "bend": 0.5, "bias": 0.4, "aniso_psf": 0.3, "resolution": 0.3, "poisson": 0.7, "gaussian": 0.8, "speckle": 0.1,
                  "rings": 0.0, "streaks": 0.0, "slice_jitter": 0.6, "fringes": 0.1, "cupping": 0.0, "seams": 0.2, "zingers": 0.1,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.5, "misalign": 0.5, "missing": 0.3, "stain": 0.5, "stripes": 0.2, "dust": 0.2, "mosaic": 0.2,
                  "slabs": 0.15, "drift": 0.4, "chatter": 0.3, "lsstripes": 0.3, "depth": 0.1, "lowdose": 0.05, "wedge": 0.0, "ctf": 0.0, "saturate": 0.1},
    # confocal / light-sheet / two-photon: an axially elongated PSF, photon-limited noise, attenuation and bleaching with
    # depth, light-sheet shadows, tile vignetting and seams, thick optical sections, saturated bright structures
    "fluorescence": {"warp": 0.6, "invert": 0.15, "bend": 0.5, "bias": 0.6, "aniso_psf": 0.9, "resolution": 0.3, "poisson": 0.9, "gaussian": 0.7, "speckle": 0.1,
                  "rings": 0.0, "streaks": 0.0, "slice_jitter": 0.3, "fringes": 0.0, "cupping": 0.0, "seams": 0.2, "zingers": 0.1,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.4, "misalign": 0.1, "missing": 0.05, "stain": 0.0, "stripes": 0.1, "dust": 0.2, "mosaic": 0.3,
                  "slabs": 0.05, "drift": 0.05, "chatter": 0.0, "lsstripes": 0.4, "depth": 0.6, "lowdose": 0.3, "wedge": 0.0, "ctf": 0.0, "saturate": 0.2},
    # cryo-electron tomography: the missing wedge, the contrast transfer function, a dose of a few electrons per voxel
    "cryoet":    {"warp": 0.3, "invert": 0.5, "bend": 0.3, "bias": 0.2, "aniso_psf": 0.2, "resolution": 0.2, "poisson": 0.5, "gaussian": 0.8, "speckle": 0.0,
                  "rings": 0.0, "streaks": 0.1, "slice_jitter": 0.1, "fringes": 0.2, "cupping": 0.0, "seams": 0.0, "zingers": 0.2,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.0, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.0, "dust": 0.1, "mosaic": 0.0,
                  "slabs": 0.0, "drift": 0.0, "chatter": 0.0, "lsstripes": 0.0, "depth": 0.1, "lowdose": 0.8, "wedge": 0.9, "ctf": 0.9, "saturate": 0.05},
}
NEW_KEYS = ("slabs", "drift", "chatter", "lsstripes", "depth", "lowdose", "wedge", "ctf", "saturate")
MODALITIES["mixed"] = {k: (0.1 if k in ("wedge", "ctf", "lowdose") else 0.15 if k in NEW_KEYS else 0.35) for k in MODALITIES["hipct"]}
CROSS_MIX = 0.08                   # every nuisance outside the chosen modality's profile is still drawn with this probability
CROSS_KEY = {"wedge": 0.03, "ctf": 0.03, "lowdose": 0.03, "slabs": 0.03}   # the most destructive ones less often
#: the HiP-CT-only profile (``nuisance(profile="hipct")``, opt-in from ``scene.draw_scene(profile="hipct")``; 28 Oct review):
#: only what a synchrotron propagation phase-contrast scan of fixed tissue shows -- an isotropic PSF (0.5-1.2 voxels),
#: low-pass (Paganin-filtered) noise instead of white Poisson / Gaussian, rings, residual edge fringes, stitching seams, a
#: flat-field bias, monotone remaps and either polarity; no cross-mixing (no section, MRI, EM, fluorescence, cryo-ET
#: effects, no anisotropic PSF, no thick slices). Strength 0: nearly clean; 1: the hardest plausible HiP-CT.
HIPCT_PROFILE = {k: 0.0 for k in MODALITIES["hipct"]}
HIPCT_PROFILE.update({"invert": 0.5, "bend": 0.5, "bias": 0.5, "rings": 0.6, "fringes": 0.7, "seams": 0.3})
#: the HiP-CT profile's noise at full strength (fractions of the image's range): the low-pass part's sd drawn in
#: [lo, hi], the white part's up to ``white``; scaled by the strength
HIPCT_NOISE = {"lo": 0.005, "hi": 0.02, "white": 0.01}
HIPCT_PSF = (0.5, 1.2)              # the HiP-CT profile's isotropic PSF sigma (voxels): from the first at strength 0 up to the second
#: record keys the HiP-CT profile never writes (the removed nuisances; tests check them)
HIPCT_REMOVED = ("misalign", "drift", "missing_sections", "chatter", "stain", "lsstripes", "depth", "cupping", "mosaic", "heavy_psf",
                 "resolution", "gibbs", "ghosting", "ctf", "poisson_photons", "gaussian_sd", "rician_sd", "speckle", "lowdose", "wedge",
                 "thick_slices", "streaks", "stripes", "zingers", "dust", "slice_jitter", "saturate", "slabs", "warp", "affine",
                 "slice_profile", "tear", "fold", "ct_scan", "scan")


def _nuisance_once(s: Sample, seed: int = 0, device="cuda", strength: float = 1.0, modality: str | None = None, warp: bool | None = None,
             fast: bool = False, hp_max: float | None = None, vis_opts: dict | None = None, affine: bool = False,
             orient_from_texture: bool | None = None, profile: str | None = None) -> dict:
    # ``strength`` (0-1) scales the probability and the magnitude of every nuisance that destroys information (PSF,
    # resolution, thick slices, k-space truncation, noise, artefacts); geometric and monotone remaps are not scaled
    """One draw of a modality's acquisition on a clean sample (``hipct`` / ``ct`` / ``mri`` / ``histology`` / ``em`` /
    ``fluorescence`` / ``cryoet`` / ``mixed``; None picks one), with cross-mixing. The geometry (``warp``; None draws
    it) moves the labels with the image; every photometric nuisance acts along a randomly drawn **acquisition axis**
    (slices, thick slices, anisotropic PSF, rings and cupping about it, streaks in planes across it); the image is kept
    in float to the end, where a display window maps it to 0-255. Returns the image, the warped occupancies
    ``(K, Z, Y, X)``, the warped shape tensors ``(K, 6, Z, Y, X)``, the layers' ``visibility (K,)`` and the draw's record.

    Sections (along the axis): misalignment, a cumulative **drift**, lost sections (the previous or next repeated, the
    mean, blank), stain flicker, **knife chatter** (a sinusoid in each section, its phase drifting); illumination: the
    bias field, **light-sheet shadows / curtaining** (lines sharp across one in-plane axis, constant along the other),
    **depth** attenuation or bleaching with a blur that grows with depth; transfer: the PSF, resolution, Gibbs,
    ghosting, the **CTF** (rings of either sign, possibly phase-flipped); noise: Poisson, Gaussian, Rician, speckle, a
    **low dose** (0.5-20 photons per voxel, read noise, offset), the cryo-ET **missing wedge** (beam along the axis);
    then thick slices, the CT artefacts, **saturation** at one end and lost **slabs** of 2-16 sections (a constant,
    held, interpolated or noise). Everything destructive happens before the visibility is measured, so a lost slab or
    a region drowned in noise asks for nothing in the targets."""
    g = torch.Generator(device=device).manual_seed(seed + 7919)
    if fast:                                                                                     # scalars from numpy: no device synchronisation per draw
        _np = np.random.default_rng(seed + 7919)
        u = lambda a, b: float(_np.uniform(a, b))
        _int = lambda lo, hi: int(_np.integers(lo, hi))
    else:
        u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
        _int = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    names = list(MODALITIES); modality = modality or names[_int(0, len(names))]
    P = MODALITIES[modality]
    hip = profile == "hipct"                                                                    # the HiP-CT-only profile (opt-in)
    if hip:
        modality, P = "hipct", HIPCT_PROFILE
    sg = float(min(max(strength, 0.0), 1.0))                                                     # 0: clean; 1: the full ranges (a training curriculum ramps it)
    sg = sg * MOD_STRENGTH.get(modality, 1.0)                                                    # (29 Oct: each modality's whole profile calibrated)
    KEEP = ("warp", "invert", "bend", "bias")                                                   # remaps that keep the information: not scaled by strength
    on = lambda key: u(0, 1) < max(P.get(key, 0.0), CROSS_KEY.get(key, CROSS_MIX)) * (1.0 if key in KEEP else sg)
    if hip:                                                                                      # no cross-mixing
        on = lambda key: u(0, 1) < P.get(key, 0.0) * (1.0 if key in KEEP else sg)
    img = s.image.clone().float(); Z, Y, X = img.shape; rec = {"modality": modality}
    occ = torch.stack([L.occupancy for L in s.layers]); T = torch.stack([L.tensor for L in s.layers])
    Dsc = torch.stack([L.desc if L.desc is not None else torch.zeros((3,) + tuple(img.shape), device=device) for L in s.layers])   # (K, 3, Z, Y, X)
    R = s.regions if s.regions is not None else torch.zeros_like(img)
    K = occ.shape[0]; stretch = torch.ones_like(img); grid = None
    # ---- geometry: an elastic warp; labels follow; shape tensors by the Jacobian (S -> J⁻¹ S J⁻ᵀ), never folding
    do_warp = bool(warp if warp is not None else on("warp"))
    if do_warp or affine:
        # a global affine (shear, anisotropic scale: shrinkage, section compression, voxel calibration) about the centre,
        # plus the elastic field; the tensors follow the Jacobian of the whole displacement
        aff = torch.zeros((3, Z, Y, X), device=device)
        if affine:
            A = np.eye(3) + np.diag([u(-0.2, 0.2) for _ in range(3)])
            for (i, j) in ((0, 1), (0, 2), (1, 2)):
                if u(0, 1) < 0.5:
                    A[i, j] = u(-0.15, 0.15) if u(0, 1) < 0.5 else 0.0
            At = torch.tensor(A - np.eye(3), dtype=torch.float32, device=device)
            c = torch.tensor([(Z - 1) / 2, (Y - 1) / 2, (X - 1) / 2], device=device).view(3, 1, 1, 1)
            aff = torch.einsum("ij,j...->i...", At, _coords((Z, Y, X), device) - c)
            rec["affine"] = [round(float(v), 3) for v in A.flatten()]
        sig = lu(8, 30); amp = u(0, 6) if do_warp else 0.0
        n3 = torch.stack([smooth_noise((Z, Y, X), sig, g, device) for _ in range(3)]) if do_warp else torch.zeros_like(aff)
        for _ in range(10):
            J = _jacobian(n3 * amp + aff); det = float(torch.linalg.det(J).min())
            if det > 0.2:
                break
            amp *= 0.7
        else:
            amp = 0.0; J = _jacobian(aff); det = float(torch.linalg.det(J).min())
        disp = n3 * amp + aff; grid = _warp_grid(disp)
        pad = "reflection" if affine else "border"                                             # an affine reaches past the faces: tissue continues
        img = _resample(img, grid, padding=pad); occ = _resample(occ, grid, padding=pad); R = _resample(R, grid, padding=pad)
        Tw = _resample(T.reshape(K * 6, Z, Y, X), grid, padding=pad).reshape(K, 6, Z, Y, X)
        rec["warp_padding"] = pad
        Ji = torch.linalg.inv(J)
        S = Ji[None] @ full3(Tw.permute(0, 2, 3, 4, 1)) @ Ji.transpose(-1, -2)[None]
        S = S / S.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-9)
        T = sym6(S).permute(0, 4, 1, 2, 3).contiguous(); rec["warp"] = (round(amp, 2), round(sig, 1), round(det, 2))
        stretch = torch.linalg.det(J).clamp_min(1e-3) ** (-1.0 / 3.0)                            # shapes appear this much larger in the warped image
        Dsc = _resample(Dsc.reshape(K * 3, Z, Y, X), grid, padding=pad).reshape(K, 3, Z, Y, X)
        Dsc[:, 0] = Dsc[:, 0] * stretch[None]
    if getattr(s, "budget", None) is not None and not TARGETS_BUDGET:                           # each population its own occupancy
        bw = s.budget[None]                                                                      # in the targets (the image keeps the
        if do_warp or affine:                                                                    # shared voxel)
            bw = _resample(bw, grid, padding=pad)
        obj = torch.tensor([L.role == "shape" for L in s.layers], device=device).view(-1, 1, 1, 1)
        occ = torch.where(obj, (occ * bw.clamp_min(1.0)).clamp(max=1.0), occ)
    # ---- the acquisition frame: the acquisition axis becomes dim 0 for every photometric nuisance
    ax = _int(0, 3); rec["axis"] = ax
    img = img.movedim(ax, 0).contiguous(); A, B, C = img.shape
    cy, cx = None, None
    if True:                                                                                    # the rotation axis's in-plane position (rings, cupping)
        phi = u(0, 2 * math.pi); dist = (u(0, 0.5) if u(0, 1) < 0.3 else lu(0.5, 5.0)) * max(B, C)
        cy, cx = (B - 1) / 2 + dist * math.sin(phi), (C - 1) / 2 + dist * math.cos(phi)
    present = torch.ones(A, device=device)                                                      # how much of each section's data is still its own (lost: 0)
    yy, xx = torch.meshgrid(torch.arange(B, device=device, dtype=torch.float32), torch.arange(C, device=device, dtype=torch.float32), indexing="ij")
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # sections: misalignment (no wrap-around), missing sections, stain variation
    if on("misalign"):
        sd = u(0.3, 1.5) * sg; sh = (sd * torch.randn(A, 2, device=device, generator=g)).round().long()
        img = torch.stack([_shift2d(img[a], int(sh[a, 0]), int(sh[a, 1])) for a in range(A)]); rec["misalign"] = round(sd, 2)
    if on("drift"):                                                                             # a cumulative random walk of the sections (no wrap-around)
        sd = u(0.2, 1.0) * sg; walk = torch.cumsum(sd * torch.randn(A, 2, device=device, generator=g), 0)
        walk = (walk - walk[A // 2]).clamp(-6, 6).round().long()
        img = torch.stack([_shift2d(img[a], int(walk[a, 0]), int(walk[a, 1])) for a in range(A)]); rec["drift"] = round(sd, 2)
    if on("missing"):                                                                           # single sections lost: the previous or the next repeated, the mean, blank
        k = int(u(1, 4)); modes = []
        for _ in range(k):
            a = int(u(1, A - 2)); m = ("previous", "next", "mean", "blank")[_int(0, 4)]; modes.append(m)
            img[a] = {"previous": lambda: img[a - 1].clone(), "next": lambda: img[a + 1].clone(), "mean": lambda: torch.full_like(img[a], float(img.mean())),
                      "blank": lambda: torch.full_like(img[a], float(img.min()))}[m]()
            present[a] = 0.5 if m in ("previous", "next") else 0.0
        rec["missing_sections"] = (k, modes)
    if on("chatter"):                                                                           # knife chatter: a sinusoid in each section, its phase drifting
        j = _int(1, 3); per = lu(3, 20); amp = u(0.03, 0.2) * sg
        ph = u(0, 2 * math.pi) + torch.cumsum(u(0.1, 1.0) * torch.randn(A, device=device, generator=g), 0)
        pos = torch.arange(img.shape[j], device=device, dtype=torch.float32)
        wave = 1 + amp * torch.sin(2 * math.pi * pos[None] / per + ph[:, None])                 # (A, n_j)
        img = img * (wave[:, :, None] if j == 1 else wave[:, None, :]); rec["chatter"] = (j, round(per, 1), round(amp, 3))
    if on("stain"):
        img = img * (1 + u(0.03, 0.2) * sg * torch.randn(A, device=device, generator=g)).view(A, 1, 1) + u(0, 15) * sg * torch.randn(A, device=device, generator=g).view(A, 1, 1); rec["stain"] = True
    # intensity: polarity, gamma and a bend on the normalised range (no clipping), a log-normal bias field
    if on("invert"):
        img = 255 - img; rec["inverted"] = True
    lo, hi = _quantile(img, 0.005), _quantile(img, 0.995); rng = max(hi - lo, 1e-3)
    gam = lu(0.5, 2.0); x = ((img - lo) / rng).clamp_min(0) ** gam; rec["gamma"] = round(gam, 2)
    if on("bend"):
        a, b = u(0.2, 0.8), u(0.2, 0.8); x = torch.where(x < a, b * x / a, b + (1 - b) * (x - a) / (1 - a)); rec["bend"] = (round(a, 2), round(b, 2))
    img = lo + rng * x
    if on("bias"):
        amp = u(0.05, 0.35) * (1.5 if modality == "mri" else 1.0)
        img = img * torch.exp(amp * smooth_noise((A, B, C), lu(20, 60), g, device)); rec["bias"] = round(amp, 2)
    if on("lsstripes"):                                                                         # light-sheet shadows / FIB-SEM curtaining: lines sharp across one
        j = _int(1, 3); n_j = img.shape[j]; S = torch.zeros(n_j, device=device)                 # in-plane axis, constant along the other, slowly varying
        n = _int(2, 13)
        for _ in range(n):
            c0 = _int(0, n_j); w = _int(1, 5); S[c0:c0 + w] += u(0.3, 1.0)
        S = S.clamp(max=1.0)
        env = (1 + u(0, 0.5) * smooth_noise((A, B, C), lu(20, 60), g, device)).clamp(0, 2)
        a = u(0.1, 0.6) * sg * (1 if u(0, 1) < 0.6 else -1)                                    # darker (shadows) or brighter
        Sf = S.view(1, -1, 1) if j == 1 else S.view(1, 1, -1)
        img = img * (1 - a * Sf * env).clamp_min(0.05); rec["lsstripes"] = (j, n, round(a, 2))
    if on("depth"):                                                                             # attenuation or bleaching along the axis, blur growing with depth
        z = torch.arange(A, device=device, dtype=torch.float32) / max(A - 1, 1)
        z = z if u(0, 1) < 0.5 else 1 - z
        if u(0, 1) < 0.5:
            al = u(0.3, 2.5) * sg; att = torch.exp(-al * z); kind = ("attenuation", round(al, 2))
        else:
            be = u(0.1, 0.6) * sg; att = 1 - be * z; kind = ("bleaching", round(be, 2))
        sb = u(0.5, 2.5) * sg; wz = (z ** u(0.5, 2.0)).view(A, 1, 1)
        img = ((1 - wz) * img + wz * gaussian(img, sb)) if sb > 0.2 else img
        img = img * att.view(A, 1, 1); rec["depth"] = kind + (round(sb, 2),)
    # CT: beam-hardening cupping (darker towards the rotation axis), phase-contrast fringes (an edge pair, either sign)
    if on("cupping"):
        c = u(0, 0.25) * sg; rmax = float(rr.max()) + 1.0
        img = img * (1 - c * (1 - (rr / rmax) ** 2))[None]; rec["cupping"] = round(c, 2)
    if on("fringes"):
        a = u(-0.3, 0.8) * sg; sb = lu(0.7, 1.5); img = img + a * (img - gaussian(img, sb)); rec["fringes"] = (round(a, 2), round(sb, 2))
    # histology tiles: a periodic illumination in the section plane
    if on("mosaic"):
        Tp = int(lu(24, 64)); oy, ox = int(u(0, Tp)), int(u(0, Tp))
        py = ((torch.arange(B, device=device) + oy) % Tp).float(); px = ((torch.arange(C, device=device) + ox) % Tp).float()
        vy = 1 - u(0.05, 0.2) * sg * ((py - Tp / 2).abs() / (Tp / 2)) ** 2; vx = 1 - u(0.05, 0.2) * sg * ((px - Tp / 2).abs() / (Tp / 2)) ** 2
        img = img * (vy[:, None] * vx[None, :])[None]; rec["mosaic"] = Tp
    # PSF (longer along the acquisition axis when anisotropic), resolution loss, k-space truncation, ghosting
    sig = [lu(0.3, 0.3 + 1.7 * sg) for _ in range(3)] if not hip else [lu(HIPCT_PSF[0], HIPCT_PSF[0] + (HIPCT_PSF[1] - HIPCT_PSF[0]) * sg)] * 3   # HiP-CT: isotropic
    if on("heavy_psf"):                                                                          # a poor resolution: the PSF 1.5-3.5x wider
        k_ = lu(1.5, 1.5 + 2.0 * sg); sig = [min(v * k_, 6.0) for v in sig]; rec["heavy_psf"] = round(k_, 2)   # (to ~6 voxels; targets merge
    if on("aniso_psf"):                                                                          # what it cannot resolve: acquisition_blur)
        sig[0] = sig[0] * lu(1.5, 1.5 + 2.5 * sg)
    img = gaussian(img, tuple(sig), truncate=3.0); rec["psf"] = [round(v, 2) for v in sig]
    if on("resolution"):
        f = lu(1.2, 1.2 + 1.3 * sg); small = Fn.interpolate(img[None, None], scale_factor=1 / f, mode="trilinear", align_corners=False)
        img = Fn.interpolate(small, size=(A, B, C), mode="trilinear", align_corners=False)[0, 0]; rec["resolution"] = round(f, 2)
    if on("gibbs"):
        axes = [1, 2] if u(0, 1) < 0.5 else [int(u(1, 3))]
        if u(0, 1) < 0.3:
            axes.append(0)                                                                       # a 3-D acquisition
        c = u(0.45 - 0.2 * sg, 0.45); img = gibbs(img, axes, c); rec["gibbs"] = (axes, round(c, 2))
    if on("ghosting"):
        a = int(u(0, 3)); n = img.shape[a]; sh = int(u(n // 6, n // 2)); img = img + u(0.03, 0.15) * sg * torch.roll(img, shifts=sh, dims=a); rec["ghosting"] = (a, sh)
    if on("ctf"):                                                                               # the contrast transfer function: rings of either sign in frequency
        kk = torch.sqrt(sum(torch.fft.fftfreq(n_, device=device).view([-1 if b == a else 1 for b in range(3)]) ** 2 for a, n_ in enumerate(img.shape)))
        per = lu(3, 15); Aa = u(0.05, 0.15); kB = u(0.15, 0.5)
        chi = math.pi * (kk * per) ** 2
        ctf = -(math.sqrt(1 - Aa * Aa) * torch.sin(chi) + Aa * torch.cos(chi)) * torch.exp(-(kk / kB) ** 2)
        flip = u(0, 1) < 0.5
        if flip:                                                                                 # a phase-flipped (corrected) tomogram
            ctf = ctf.abs()
        m0 = float(img.mean()); img = m0 + torch.fft.ifftn(torch.fft.fftn(img - m0) * ctf).real
        rec["ctf"] = (round(per, 1), round(Aa, 2), round(kB, 2), flip)
    # 29 Oct: the wider pool (EXTREME_P) -- CT physics in the image domain, motion, speckle, oriented noise at many scales
    onx = lambda key: (not hip) and u(0, 1) < max(EXTREME_MOD.get(modality, {}).get(key, 0.0), EXTREME_P[key]) * sg
    lo, hi = _quantile(img, 0.001), _quantile(img, 0.999); rng = max(hi - lo, 1e-3)
    if onx("scatter"):                                                                          # scatter: a low-frequency haze of the content
        a = u(0.2, 1.0) * sg; s_ = lu(8, 30); img = img + a * gaussian(img - lo, s_); rec["scatter"] = (round(a, 2), round(s_, 1))
    if onx("bh_streaks"):                                                                       # beam hardening: dark bands between dense objects
        n = _int(2, 5); P_ = _bright(img, n, 0.01, g); amp = u(0.1, 0.5) * sg * rng; w = u(0.7, 2.5); h = lu(2, max(3, A / 2))
        zs = torch.arange(A, device=device, dtype=torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                band = torch.exp(-0.5 * (_seg_dist(yy, xx, P_[i, 1:], P_[j, 1:]) / w) ** 2)
                wz = torch.exp(-0.5 * ((zs - 0.5 * float(P_[i, 0] + P_[j, 0])) / h) ** 2)
                img = img - amp * wz[:, None, None] * band[None]
        rec["bh_streaks"] = (n, round(amp / rng, 2))
    if onx("metal"):                                                                            # metal / photon starvation: radial streaks
        n = _int(1, 3); P_ = _bright(img, n, 0.002, g); zs = torch.arange(A, device=device, dtype=torch.float32)
        for i in range(n):
            K = _int(6, 25); amp = u(0.2, 0.8) * sg * rng; w = u(0.5, 1.5); h = lu(1, max(2, A / 3)); dec = lu(10, 80)
            dy, dx = yy - float(P_[i, 1]), xx - float(P_[i, 2]); r_ = torch.sqrt(dy ** 2 + dx ** 2)
            st = torch.zeros_like(yy)
            for _ in range(K):
                ph = u(0, math.pi); perp = (dy * math.cos(ph) - dx * math.sin(ph)).abs()
                st += (1 if u(0, 1) < 0.4 else -1) * u(0.3, 1.0) * torch.exp(-0.5 * (perp / w) ** 2)
            st = st * torch.exp(-r_ / dec)
            img = img + amp * torch.exp(-0.5 * ((zs - float(P_[i, 0])) / h) ** 2)[:, None, None] * st[None]
        rec["metal"] = n
    if onx("motion"):                                                                           # motion blur: a line kernel in any direction
        d = torch.randn(3, generator=g, device=device); d = d / d.norm().clamp_min(1e-6); L = lu(2, 2 + 8 * sg)
        img = line_blur(img, d, L); rec["motion"] = round(L, 1)
    if onx("slice_motion"):                                                                     # sequential acquisition: sections blurred / jittered
        q = u(0.2, 0.7); hit = 0
        for a in range(A):
            if u(0, 1) < q:
                ph = u(0, 2 * math.pi); L = lu(1.5, 1.5 + 6 * sg)
                if u(0, 1) < 0.5:
                    img[a] = line_blur(img[a], (math.sin(ph), math.cos(ph)), L)
                else:                                                                           # a sub-voxel / few-voxel jitter of the section
                    img[a] = line_blur(img[a], (math.sin(ph), math.cos(ph)), 2 * u(0.3, 1.0) * L / 2, n=1)
                hit += 1
        rec["slice_motion"] = hit
    us_, oc_ = onx("us_speckle"), onx("oct_speckle")
    if us_ or oc_:                                                                              # fully developed speckle: an oriented grain
        oct_ = oc_ and (not us_ or u(0, 1) < 0.5)                                                # (Rayleigh amplitude / gamma intensity)
        Rm = torch.eye(3, device=device) if oct_ else _rand_rot(g, device)
        s3 = (lu(0.5, 1.2), lu(1.0, 4.0), lu(1.0, 4.0)) if not oct_ else (lu(0.5, 1.0), lu(1.0, 3.0), lu(1.0, 3.0))
        L_ = _int(1, 4); S_ = sum(aniso_noise((A, B, C), s3, Rm, g, device) ** 2 + aniso_noise((A, B, C), s3, Rm, g, device) ** 2 for _ in range(L_)) / L_
        S_ = S_ if u(0, 1) < 0.5 else S_.sqrt()                                                  # intensity (gamma) or amplitude (Rayleigh)
        S_ = S_ / S_.mean().clamp_min(1e-9); m = u(0.5, 1.0) * sg
        x = ((img - lo) / rng).clamp_min(0) * ((1 - m) + m * S_)
        if oct_:                                                                                 # OCT: depth decay along the beam, log display
            zz = torch.arange(A, device=device, dtype=torch.float32).view(A, 1, 1) / max(A - 1, 1)
            cl = lu(10, 1000); x = torch.log1p(cl * x * torch.exp(-u(0, 3) * sg * zz)) / math.log1p(cl)
        img = lo + rng * x; rec["oct_speckle" if oct_ else "us_speckle"] = (L_, [round(v, 2) for v in s3], round(m, 2))
    if onx("oriented_noise"):                                                                   # oriented noise at many scales (additive)
        k = _int(1, 4); nz = torch.zeros_like(img)
        for _ in range(k):
            b = lu(0.7, 6.0); e = lu(2.0, 8.0)
            nz += u(0.3, 1.0) * aniso_noise((A, B, C), (b * e, b, b * lu(1.0, e)), _rand_rot(g, device), g, device)
        amp = u(0.05, 0.4) * sg * rng; img = img + amp * nz / nz.std().clamp_min(1e-9); rec["oriented_noise"] = (k, round(amp / rng, 2))
    # noise: Poisson (on the current range), Gaussian, Rician, speckle, a starved dose
    lo, hi = _quantile(img, 0.001), _quantile(img, 0.999); rng = max(hi - lo, 1e-3)
    if on("poisson"):
        Pn = lu(15 / max(sg, 0.01), 1500 / max(sg, 0.01)); img = lo + rng * torch.poisson(((img - lo) / rng).clamp_min(0) * Pn, generator=g) / Pn; rec["poisson_photons"] = round(Pn)
    if on("gaussian"):
        sd = u(0, 30) * sg * rng / 255; img = img + sd * torch.randn(img.shape, device=device, generator=g); rec["gaussian_sd"] = round(sd, 1)
    if on("rician"):
        sd = u(5, 45) * sg * rng / 255
        img = lo + torch.sqrt((img - lo + sd * torch.randn(img.shape, device=device, generator=g)) ** 2 + (sd * torch.randn(img.shape, device=device, generator=g)) ** 2); rec["rician_sd"] = round(sd, 1)
    if on("speckle"):
        img = img * (1 + u(0, 0.2) * sg * torch.randn(img.shape, device=device, generator=g)); rec["speckle"] = True
    if on("lowdose"):                                                                           # a few photons / electrons per voxel, a read noise and an offset
        Pn = lu(0.5, 20.0) / max(sg, 0.01) ** 2; rd = u(0.3, 2.0); off = u(0, 5)
        cnt = torch.poisson(((img - lo) / rng).clamp_min(0) * Pn, generator=g) + off + rd * torch.randn(img.shape, device=device, generator=g)
        img = lo + rng * (cnt - off) / Pn; rec["lowdose"] = (round(Pn, 2), round(rd, 2))
    if hip and sg > 0:                                                                          # HiP-CT: low-pass noise (the Paganin filter's
        cs = lu(0.6, 1.5); sd = u(HIPCT_NOISE["lo"], HIPCT_NOISE["lo"] + HIPCT_NOISE["hi"] * sg) * rng   # spectrum) and a little white noise
        nz = gaussian(torch.randn(img.shape, device=device, generator=g), cs, truncate=3.0)
        nz = nz / nz.std().clamp_min(1e-9)
        sw = u(0.0, HIPCT_NOISE["white"]) * sg * rng
        img = img + sd * nz + sw * torch.randn(img.shape, device=device, generator=g)
        rec["corr_noise"] = (round(sd / rng * 255, 1), round(cs, 2), round(sw / rng * 255, 1))
    if on("wedge"):                                                                             # the missing wedge: beam along the axis, tilt axis the last in-plane
        th = u(40, 70); k0 = torch.fft.fftfreq(A, device=device).view(-1, 1, 1); k1 = torch.fft.fftfreq(B, device=device).view(1, -1, 1)
        keep = (k0.abs() <= math.tan(math.radians(th)) * k1.abs()) | (k0 == 0)                  # the planes reached by tilts up to +-th
        keep = keep.expand(A, B, C).float()
        if u(0, 1) < 0.5:                                                                        # a soft edge (weighted back-projection)
            keep = gaussian(keep, (0.7, 0.7, 0.0), truncate=3.0); keep[0, 0, 0] = 1.0
        m0 = float(img.mean()); img = m0 + torch.fft.ifftn(torch.fft.fftn(img - m0) * keep).real; rec["wedge"] = round(th, 1)
    # thick slices, after the noise: acquired thick, resampled to isotropic (noise correlated along the axis)
    if on("thick"):
        f = lu(2, 2 + 4 * sg); small = Fn.interpolate(img[None, None], size=(max(2, int(A / f)), B, C), mode="area")
        img = Fn.interpolate(small, size=(A, B, C), mode="nearest" if u(0, 1) < 0.3 else "trilinear")[0, 0]; rec["thick_slices"] = round(f, 1)
    # CT artefacts in float: rings about the rotation axis, streaks in the slice planes at a random angle, seams, zingers
    unit = rng / 255 * sg
    if on("rings"):
        art = torch.zeros((B, C), device=device); amp = u(3, 30) * unit; n = int(lu(3, 30)); w = u(0.5, 1.5)
        r0, r1 = float(rr.min()), float(rr.max())
        for _ in range(n):
            rad = u(r0, r1); art += amp * u(0.5, 1.5) * (1 if u(0, 1) < 0.5 else -1) * torch.exp(-0.5 * ((rr - rad) / w) ** 2)
        img = img + art[None]; rec["rings"] = {"n": n, "centre": (round(cy, 1), round(cx, 1))}
    if on("streaks"):
        amp = u(5, 40) * unit; ang = u(0, 180); spread = u(5, 60); half = 0.5 * math.hypot(B, C)
        yc, xc = yy - (B - 1) / 2, xx - (C - 1) / 2; zs = torch.arange(A, device=device, dtype=torch.float32)
        n = int(_np.poisson(0.5 * A)) if fast else int(torch.poisson(torch.tensor(0.5 * A, device=device), generator=g))
        for _ in range(n):
            z0 = u(0, A - 1); p = math.radians(ang + u(-spread, spread)); off = u(-half, half)
            d = (yc * math.cos(p) - xc * math.sin(p) - off).abs()
            a0, a1 = max(0, int(z0) - 4), min(A, int(z0) + 5)
            wz = torch.exp(-0.5 * (zs[a0:a1] - z0) ** 2)
            img[a0:a1] += amp * u(0.5, 1.5) * (1 if u(0, 1) < 0.5 else -1) * wz[:, None, None] * torch.exp(-0.5 * (d / 0.8) ** 2)[None]
        rec["streaks"] = {"n": n, "angle_deg": round(ang, 1)}
    if on("stripes"):
        a = int(u(1, 3)); n = int(u(2, 12)); amp = u(5, 30) * unit
        for _ in range(n):
            i = int(u(0, img.shape[a] - 1)); w = int(u(1, 3)); sl = [slice(None)] * 3; sl[a] = slice(i, i + w)
            img[tuple(sl)] += amp * (1 if u(0, 1) < 0.5 else -1)
        rec["stripes"] = n
    if on("seams"):
        for a in range(3):
            if u(0, 1) < 0.5:
                n = img.shape[a]; i = int(u(n // 4, 3 * n // 4)); sl = [slice(None)] * 3; sl[a] = slice(i, None); img[tuple(sl)] += u(-20, 20) * unit
        rec["seams"] = True
    if on("zingers"):
        n = int(lu(5, 200)); idx = (torch.rand(n, 3, device=device, generator=g) * torch.tensor([A, B, C], device=device)).long()
        img[idx[:, 0], idx[:, 1], idx[:, 2]] = lo + rng * u(0, 1.2); rec["zingers"] = n
    if on("dust"):
        n = int(lu(2, 30)); idx = (torch.rand(n, 3, device=device, generator=g) * torch.tensor([A, B, C], device=device)).long()
        spots = torch.zeros_like(img); spots[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0; spots = gaussian(spots, u(1.0, 3.0))
        img = img + u(-0.8, 0.8) * rng * spots / spots.max().clamp_min(1e-6); rec["dust"] = n
    if on("slice_jitter"):
        img = img * (1 + u(0.02, 0.1) * sg * torch.randn(A, device=device, generator=g)).view(A, 1, 1); rec["slice_jitter"] = True
    if on("saturate"):                                                                          # the detector clipped at one end
        fr = u(0.1, 0.4) * sg; top = u(0, 1) < 0.6
        img = img.clamp(max=_quantile(img, 1 - fr)) if top else img.clamp(min=_quantile(img, fr)); rec["saturate"] = (round(fr, 3), "top" if top else "bottom")
    if on("slabs") and A >= 8:                                                                  # runs of sections lost: a constant, held, interpolated or noise
        n = _int(1, 4); wmax = max(2, min(16, int(0.12 * A))); lost = []
        for _ in range(n):
            w = _int(2, wmax + 1); a0 = _int(1, A - w); a1 = a0 + w; m = ("mean", "low", "high", "held", "interp", "noise")[_int(0, 6)]
            if m == "mean":
                img[a0:a1] = float(img.mean())
            elif m in ("low", "high"):
                img[a0:a1] = _quantile(img, 0.001 if m == "low" else 0.999)
            elif m == "held":
                img[a0:a1] = img[a0 - 1].clone()
            elif m == "interp":
                t = (torch.arange(1, w + 1, device=device, dtype=torch.float32) / (w + 1)).view(-1, 1, 1)
                nxt = img[a1] if a1 < A else img[a0 - 1]
                img[a0:a1] = (1 - t) * img[a0 - 1] + t * nxt
            else:
                img[a0:a1] = float(img.mean()) + float(img.std()) * torch.randn((w, B, C), device=device, generator=g)
            present[a0:a1] = 0.5 if m in ("held", "interp") else 0.0                            # a neighbour's data (displaced) or none
            lost.append((a0, w, m))
        rec["slabs"] = lost
    # 29 Oct: extreme intensity transforms (EXTREME_P): strongly non-monotone curves, posterisation, clipping, local
    # contrast reversal
    if not hip:
        lo, hi = _quantile(img, 0.001), _quantile(img, 0.999); rng = max(hi - lo, 1e-3)
        x = ((img - lo) / rng).clamp(0, 1); x0 = x
        if onx("fold_lut"):
            f = u(1.0, 1.0 + 3.0 * sg); ph = u(0, 2 * math.pi); w = u(0.5, 1.0) * sg             # (29 Oct: its weight by the strength too)
            x = w * (0.5 + 0.5 * torch.sin(2 * math.pi * f * x + ph)) + (1 - w) * x; rec["fold_lut"] = (round(f, 2), round(w, 2))
        if onx("local_reversal"):
            m_ = gaussian(x, lu(3, 12)); fld = torch.tanh(u(1, 4) * smooth_noise((A, B, C), lu(8, 30), g, device))
            fld = 1 - sg * (1 - fld)                                                             # (29 Oct: by the strength, as every
            x = m_ + fld * (x - m_); rec["local_reversal"] = True                                # destructive nuisance; was full at any)
        if onx("clip"):
            a_, b_ = u(0, 0.3 * sg), u(0, 0.3 * sg); x = x.clamp(_quantile(x, a_), _quantile(x, 1 - b_)); rec["clip"] = (round(a_, 2), round(b_, 2))
        if onx("posterise"):
            nl = _int(2, 9); x = torch.round(x * (nl - 1)) / (nl - 1); rec["posterise"] = nl
        if x is not x0:
            img = lo + rng * x
    img = img.movedim(0, ax).contiguous()
    # ---- the display window: a little clipping at either end, then 8 bits
    clip = u(0.0, 0.05) if u(0, 1) < 0.1 else u(0.0, 0.015)
    lo, hi = _quantile(img, clip * u(0, 1)), _quantile(img, 1 - clip * u(0, 1))
    img = ((img - lo) / max(hi - lo, 1e-6) * 255).clamp(0, 255).round(); rec["window"] = round(clip, 3)
    occ = occ.clamp(0, 1)
    regs = torch.stack([R if L.role == "interface" else occ[k] for k, L in enumerate(s.layers)])
    t = visibility(img, regs, [None if L.role == "isotropic" else 1.0 for L in s.layers], radii=[L.radius if L.role == "shape" else 1.0 for L in s.layers], return_t=True,
                   s_hp_max=hp_max, **(vis_opts or {}))
    iso = torch.tensor([L.role == "isotropic" for L in s.layers], device=device).view(-1, 1, 1, 1)
    vis = torch.where(iso, torch.ones_like(t), ((t - PRESENCE_T[0]) / (PRESENCE_T[1] - PRESENCE_T[0])).clamp(0, 1))
    t_o = t
    if not ORIENT_VIS:                                                                               # the orientation's visibility measured on the
        pass                                                                                         # image at the queries instead (28 Oct)
    elif ORIENT_FROM_TEXTURE if orient_from_texture is None else orient_from_texture:               # orientation: the layers' oriented texture
                                                                                                     # (``orient_from_texture``: per draw -- packed
                                                                                                     # tissue has no edges to credit, 27 Oct)
        tex = torch.stack([(occ[k] - gaussian(occ[k][None, None], max(1.5, 2.0 * float(L.radius)))[0, 0]) if L.role == "shape" else regs[k]
                           for k, L in enumerate(s.layers)])
        t_o = visibility(img, tex, [None if L.role == "isotropic" else 1.0 for L in s.layers], radii=[L.radius if L.role == "shape" else 1.0 for L in s.layers],
                         return_t=True, s_hp_max=hp_max, **{**(vis_opts or {}), "ladder": (1,)})
    elif ORIENT_SINGLE_SCALE and (vis_opts or {}).get("ladder", (1,)) != (1,):                     # orientation: the calibrated single scale
        t_o = visibility(img, regs, [None if L.role == "isotropic" else 1.0 for L in s.layers], radii=[L.radius if L.role == "shape" else 1.0 for L in s.layers],
                         return_t=True, s_hp_max=hp_max, **{**(vis_opts or {}), "ladder": (1,)})
    ovis = torch.where(iso, torch.ones_like(t), ((t_o - ORIENTATION_T[0]) / (ORIENTATION_T[1] - ORIENTATION_T[0])).clamp(0, 1))
    if not ORIENT_VIS:                                                                               # (28 Oct review: the ``pass`` above left the
        ovis = torch.ones_like(t)                                                                    # presence t on the orientation thresholds -- a
                                                                                                     # second discount under the image-measured one)
    if bool((present < 1).any()):                                                               # lost sections: the visibility window (sigma 5) would
        pm = gaussian(present.view(A, 1, 1), (1.5, 0.0, 0.0), truncate=3.0).view([-1 if b == ax else 1 for b in range(3)])   # bridge them; a gap of
        vis = torch.where(iso, vis, vis * pm[None]); ovis = torch.where(iso, ovis, ovis * pm[None])                            # 1-2 sections still is
    if TARGETS_FROM_GEOMETRY:                                                                    # the true geometry as the targets
        vis = torch.ones_like(vis); ovis = torch.ones_like(ovis)
    wmean = lambda v: [round(float((x * o).sum() / o.sum().clamp_min(1e-6)), 2) for x, o in zip(v, occ)]   # occupancy-weighted means, for the record
    rec["visibility"] = wmean(vis); rec["orientation_visibility"] = wmean(ovis)
    return {"image": img, "occupancy": occ, "tensors": T, "descriptors": Dsc, "visibility": vis, "orientation_visibility": ovis, "stretch": stretch, "record": rec,
            "warp_grid": grid}


def draw(shape=(128, 128, 128), seed: int = 0, device="cuda", modality: str | None = None, strength: float = 1.0, **kw):
    """One training pair: a modality (None draws one), content matched to it (MRI: thick tubes and sheets, no
    sub-voxel fibres; histology: cell stains, laminae), the clean sample, and one draw of its acquisition."""
    g = torch.Generator(device=device).manual_seed(seed + 104729)
    names = list(MODALITIES)
    modality = modality or names[int(torch.randint(0, len(names), (1,), generator=g, device=device))]
    content = {"mri": "mri", "histology": "histology"}.get(modality, "general")
    s = sample(shape, seed=seed, device=device, content=content, **kw)
    return s, nuisance(s, seed=seed, device=device, modality=modality, strength=strength)


#: the share of the clean render's structure an acquisition must keep (|correlation| of the lightly blurred images);
#: below it the acquisition is redrawn at half strength with probability ``SIGNAL_REDRAW``, so most crops keep some
#: signal while a minority of hopeless ones stay for robustness. The targets are never affected.
SIGNAL_FLOOR = 0.15
SIGNAL_REDRAW = 0.8


def _signal_corr(clean: torch.Tensor, img: torch.Tensor) -> float:
    a = gaussian(clean.float()[None, None], 1.0)[0, 0].flatten(); b = gaussian(img.float()[None, None], 1.0)[0, 0].flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / ((a.norm() * b.norm()).clamp_min(1e-9)))


def nuisance(s: Sample, seed: int = 0, device="cuda", strength: float = 1.0, **kw) -> dict:
    """:func:`_nuisance_once`, with a floor on the structure that survives: where the corrupted image keeps almost
    nothing of the clean render, the acquisition is usually redrawn at half the strength (twice at most)."""
    d = _nuisance_once(s, seed=seed, device=device, strength=strength, **kw)
    if SIGNAL_FLOOR <= 0 or getattr(s, "image", None) is None:
        return d
    rng = np.random.default_rng(seed + 555); st = strength; tries = 0
    c = abs(_signal_corr(s.image, d["image"]))
    while c < SIGNAL_FLOOR and tries < 2 and rng.random() < SIGNAL_REDRAW:
        st *= 0.5; tries += 1
        d = _nuisance_once(s, seed=seed + 7919 * tries, device=device, strength=st, **kw)
        c = abs(_signal_corr(s.image, d["image"]))
    d["record"]["signal_corr"] = round(c, 3)
    if tries:
        d["record"]["signal_redraw"] = (tries, round(st, 3))
    return d
