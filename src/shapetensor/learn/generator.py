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
attenuated), overlaps never fill more than the voxel. Two more states fill every voxel: the **interface** (a sheet at
the background's region boundaries, weighted by the step) and the **isotropic** remainder.

:func:`nuisance` applies one modality's acquisition (``hipct`` / ``ct`` / ``mri`` / ``histology`` / ``mixed``): an
elastic warp (shape tensors transformed as ``J⁻¹ S J⁻ᵀ``), intensity remaps, a bias field, the PSF, noise, and the
modality's artefacts along a randomly drawn acquisition axis, then a display window; it measures each layer's
**visibility** (contrast to noise at the shapes' scale after the whole chain), which scales the layer's mass in the
target. :func:`draw` makes one training pair: content matched to the modality, then its nuisances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def blob_tensor(d: torch.Tensor, radius: float, length: float) -> torch.Tensor:
    """The normalised second moment of a blob's mass: a Gaussian of ``radius`` extruded ``length`` along ``d``."""
    a, b = radius ** 2, length ** 2 / 12.0
    return (a * eye6(d.device, d.shape[:-1]) + b * outer6(d)) / (3 * a + b)


def principal(T: torch.Tensor, which: str = "max") -> torch.Tensor:
    """The eigenvector of a tensor field ``(6, ...)`` for its largest (``max``: a line's direction) or smallest
    (``min``: a sheet's normal) eigenvalue, ``(3, ...)``, and the eigenvalues ``(3, ...)`` descending."""
    M = full3(T.movedim(0, -1))
    ev, V = torch.linalg.eigh(M)
    v = V[..., -1] if which == "max" else V[..., 0]
    return v.movedim(-1, 0), ev.flip(-1).movedim(-1, 0)


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


def render_layer(layer: Layer, shape, g: torch.Generator, device, step: float = 0.5, exclude: torch.Tensor | None = None):
    """The layer's occupancy ``(Z, Y, X)`` (a volume fraction: 1 at the centre of an isolated shape, overlaps capped at
    1) and shape tensor ``(6, Z, Y, X)`` (trace 1; the local mean of the deposited shapes' tensors). Shapes grow from
    random seeds in the region along the field (lines, blobs) or as discs perpendicular to it (sheets); points in
    ``exclude`` are not deposited (a fibre yielding to another population's fibre)."""
    n_seeds = max(1, int(layer.density * float(layer.region.sum()) / 1000.0))
    cand = layer.region.nonzero().float()
    seeds = cand[torch.randint(0, cand.shape[0], (n_seeds,), device=device, generator=g)] + torch.rand((n_seeds, 3), device=device, generator=g) - 0.5
    if layer.kind in ("line", "blob"):
        tens = line_tensor if layer.kind == "line" else (lambda d: blob_tensor(d, layer.radius, layer.length))
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
    if exclude is not None:
        idx = P.round().long()
        for i, s in enumerate(shape):
            idx[:, i] = idx[:, i].clamp(0, s - 1)
        keep = ~exclude[idx[:, 0], idx[:, 1], idx[:, 2]]; P, V = P[keep], V[keep]
    dens, T = _splat(P, V, shape, device)
    r = max(layer.radius, 0.5)
    db = gaussian(dens, r, truncate=3.0); Tb = gaussian(T, r, truncate=3.0)
    tensor = torch.where(db[None] > 1e-6, Tb / db.clamp_min(1e-6)[None], eye6(device).view(6, 1, 1, 1) / 3)
    occ = db / _peak(layer.kind, r, layer.length, step, device)
    if layer.hollow and layer.kind == "line":
        ri = 0.55 * r
        inner = gaussian(dens, ri, truncate=3.0) / _peak("line", ri, layer.length, step, device)
        occ = (occ - inner) / _wall_peak(r, ri)
    return occ.clamp(0, 1), tensor


CONTENT = ("general", "mri", "histology")


def random_layer(shape, g: torch.Generator, device, kind: str | None = None, scale: float = 1.0, content: str = "general",
                 field: torch.Tensor | None = None, density: float | None = None, region: torch.Tensor | None = None) -> Layer:
    """A random population. ``scale`` multiplies every size (the voxel size within the trained range); bends and the
    field's coherence are drawn in voxels, independent of the scale, so tight bends occur at every scale."""
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    if kind is None:
        w = torch.tensor([0.4, 0.4, 0.2] if content == "mri" else [1 / 3, 1 / 3, 1 / 3], device=device)
        kind = ("line", "sheet", "blob")[int(torch.multinomial(w, 1, generator=g))]
    radius = max(lu(0.6, 3.0) * scale, 1.2 if content == "mri" else 0.6)
    coh, bend = 0.0, 0.0
    if field is None:
        if kind == "line" and u(0, 1) < 0.3:
            bend = lu(8, 60); field = arc_field(shape, bend, g, device)
        else:
            coh = max(lu(6, 60), 3 * radius); field = direction_field(shape, coh, g, device)
    if kind == "line":
        length, dens = lu(20, 120) * scale, lu(0.3, 4.0) / scale ** 2
    elif kind == "sheet":
        length, dens = lu(6, 30) * scale, lu(0.02, 0.3) / scale ** 2
    else:
        length, dens = radius * lu(0.5, 2.5), lu(0.2, 3.0) / scale ** 3
    if content == "mri" and kind == "line":
        dens *= 0.3                                                                             # sparse tubes: vessels, tracts
    density = dens if density is None else density
    if region is None:
        region = region_mask(shape, lu(8, 40) * scale, u(0.3, 1.0) if u(0, 1) < 0.7 else 1.0, g, device)
    contrast = u(15, 140) * (1 if u(0, 1) < 0.5 else -1)                                       # faint to strong
    occluding = u(0, 1) < 0.2
    if occluding:
        contrast = u(30, 225)
    hollow = kind == "line" and u(0, 1) < (0.3 if content == "mri" else 0.15)
    return Layer(kind, field, radius, length, density, region, contrast, occluding, hollow, coh, bend)


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


def background(shape, style: str, g: torch.Generator, device, scale: float = 1.0, content: str = "general"):
    """The background before the layers: ``regions`` (anatomix: a random label map, an intensity per label),
    ``tissue`` (two or three broad regions -- grey/white matter -- with texture), ``cells`` (a cell stain: small blobs,
    dark or bright, their density optionally laminar along a random direction). Returns the image, the label map, the
    region means, and the extra layers the style brings (the cells as a blob population)."""
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    extra = []
    if style == "regions":
        nl = int(torch.randint(3, 9, (1,), generator=g, device=device)); lab = label_map(shape, nl, u(6, 25) * scale, g, device)
        R = (40 + 170 * torch.rand(nl, device=device, generator=g))[lab]; img = R + 6 * smooth_noise(shape, 2.0 * scale, g, device)
    elif style == "tissue":
        nl = int(torch.randint(2, 4, (1,), generator=g, device=device)); lab = label_map(shape, nl, u(25, 60) * scale, g, device)
        R = (60 + 150 * torch.rand(nl, device=device, generator=g))[lab]; img = R + u(8, 25) * smooth_noise(shape, u(1.0, 3.0) * scale, g, device)
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
                  u(50, 140) * (1 if u(0, 1) < 0.35 else -1), occupancy=cells, tensor=eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3)
        extra.append(L)
    return img, lab, R, extra


def _composite(bg: torch.Tensor, R: torch.Tensor, layers: list, device):
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
    for L in layers:
        L.occupancy = L.occupancy / over
    occl = sum((L.occupancy for L in layers if L.occluding), torch.zeros(shape, device=device))
    img = bg * (1 - occl)
    for L in layers:
        img = img + L.contrast * L.occupancy
    rest = (1 - torch.stack([L.occupancy for L in layers]).sum(0)).clamp(0, 1) if layers else torch.ones(shape, device=device)
    normal, bw = interface(R)
    bw = bw * vis
    ones = torch.ones(shape, dtype=torch.bool, device=device)
    layers.append(Layer("sheet", normal, 1.0, 1.0, 0.0, ones, 0.0, role="interface", occupancy=rest * bw, tensor=sheet_tensor(normal.movedim(0, -1)).movedim(-1, 0)))
    layers.append(Layer("blob", normal, 1.0, 1.0, 0.0, ones, 0.0, role="isotropic", occupancy=rest * (1 - bw), tensor=eye6(device).view(6, 1, 1, 1).expand(6, *shape) / 3))
    return img


def sample(shape=(128, 128, 128), k_layers: int | None = None, seed: int = 0, device="cuda", scale: float | None = None, style: str | None = None,
           content: str = "general", crossing: bool | None = None, cross_deg: float | None = None) -> Sample:
    """A clean sample: a background, an optional crossing group (``crossing``; None draws it with p 0.5: two or three
    line layers sharing one region, the first two at ``cross_deg``, None drawing 20-90°), ``k_layers`` further random
    layers (None draws 0-3), composited as volume fractions, plus the interface and isotropic states so that every
    voxel's state is defined. ``scale`` multiplies every size (None draws it: 0.5-2.5, 1.5-4 for ``content="mri"``)."""
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
        w = torch.rand(n, device=device, generator=g) + 0.25; w = (w / w.sum()).tolist()      # its split: dominant and non-dominant populations
        theta = cross_deg if cross_deg is not None else u(20, 90)
        coh = lu(10, 60)
        L0 = random_layer(shape, g, device, kind="line", scale=scale, content=content, region=region)
        fields = [L0.field, companion_field(L0.field, theta, coh, g, device)]
        if n == 3:
            fields.append(companion_field(L0.field, u(20, 90), coh, g, device))
        group = [L0] + [random_layer(shape, g, device, kind="line", scale=scale, content=content, field=fields[k], region=region) for k in range(1, n)]
        for Lk, wk in zip(group, w):                                                            # seeds per 1000 voxels for the layer's share of the volume
            Lk.hollow = False
            Lk.density = 1000.0 * phi * wk / (Lk.length * 2 * math.pi * (max(Lk.radius, 0.5) ** 2 + 1 / 6))
        con = u(15, 140) * (1 if u(0, 1) < 0.5 else -1)
        same = u(0, 1) < 0.8                                                                     # usually one material
        for Lk in group:
            Lk.occluding = False; Lk.group = 0
            Lk.contrast = con * u(0.7, 1.3) if same else u(15, 140) * (1 if u(0, 1) < 0.5 else -1)
        order = torch.randperm(n, generator=g, device=device).tolist()
        for attempt in range(2):                                                                # once more if lines leaving the region left it short of the drawn fraction
            have = torch.zeros(shape, device=device)
            for k in order:
                Lk = group[k]
                Lk.occupancy, Lk.tensor = render_layer(Lk, shape, g, device, exclude=have > 0.5)  # later fibres yield where earlier ones are
                have = have + Lk.occupancy
            got = float(have[region].mean())
            if attempt == 1 or got > 0.85 * phi:
                break
            for Lk in group:
                Lk.density *= min(phi / max(got, 1e-3), 3.0)
        layers += group
        rec["crossing"] = {"n": n, "theta_deg": round(theta, 1), "weights": [round(x, 3) for x in w], "fraction": round(phi, 3)}
    K = k_layers if k_layers is not None else int(torch.randint(0, 4, (1,), generator=g, device=device))
    for _ in range(K):
        L = random_layer(shape, g, device, scale=scale, content=content)
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


def _resample(vol: torch.Tensor, grid: torch.Tensor, mode="bilinear") -> torch.Tensor:
    v = vol[None, None] if vol.dim() == 3 else vol[None]
    out = Fn.grid_sample(v, grid, mode=mode, padding_mode="border", align_corners=True)
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


def _quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten()[:: max(1, x.numel() // 1_000_000)], q))


def visibility(img: torch.Tensor, regressors: torch.Tensor, units, sigma_hp: float = 4.0, lo: float = 0.05, hi: float = 0.2) -> torch.Tensor:
    """Each layer's visibility in the final image: the contrast-to-noise of what it puts in the image, at its own scale.
    The image and the layers' image signatures (``regressors (K, Z, Y, X)``: a shape layer's occupancy; an interface's
    region-means image, of which it is the step) are high-passed (``x - G_σ x``, σ = ``sigma_hp``) and the image is
    regressed on them jointly; layer k's contrast is ``(|c_k| - 2 se_k) * units[k]`` (grey levels for a full voxel of the
    shape, or for a full-weight step of ``STEP_FULL``; less two standard errors, so a layer too sparse to show its
    contrast is not credited with it), the noise the residual's std (noise and unexplained clutter). Visibility =
    (cnr - lo) / (hi - lo), clipped to [0, 1]; ``units[k] = None`` marks a layer not measured (visibility 1)."""
    K = regressors.shape[0]; v = torch.ones(K, device=img.device)
    idx = [k for k in range(K) if units[k] is not None]
    if not idx:
        return v
    hp = lambda x: (x - gaussian(x, sigma_hp, truncate=3.0))[..., ::2, ::2, ::2]
    y = hp(img).flatten()
    A = hp(regressors[idx]).reshape(len(idx), -1)
    energy = (A ** 2).sum(1)
    G = A @ A.T + 1e-6 * energy.max().clamp_min(1e-9) * torch.eye(len(idx), device=img.device)
    Gi = torch.linalg.inv(G); c = Gi @ (A @ y)
    noise = (y - c @ A).std().clamp_min(1e-6)
    c_abs = (c.abs() - 2 * noise * Gi.diagonal().clamp_min(0).sqrt()).clamp_min(0)              # the contrast the image demonstrates: |c| less two standard errors
    cnr = c_abs * torch.tensor([float(units[k]) for k in idx], device=img.device) / noise
    vis = ((cnr - lo) / (hi - lo)).clamp(0, 1)
    vis = torch.where(energy > 1.0, vis, torch.zeros_like(vis))                                  # a layer with (almost) no signal left
    v[idx] = vis
    return v


#: per modality, the probability of each nuisance; a draw picks a modality (or "mixed") and then also applies every other
#: nuisance with a small probability, so typical and atypical combinations are both seen
MODALITIES = {
    "hipct":     {"warp": 0.7, "invert": 0.5, "bend": 0.7, "bias": 0.6, "aniso_psf": 0.6, "resolution": 0.3, "poisson": 0.8, "gaussian": 0.9, "speckle": 0.1,
                  "rings": 0.6, "streaks": 0.5, "slice_jitter": 0.5, "fringes": 0.7, "cupping": 0.5, "seams": 0.3, "zingers": 0.4,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.1, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.1, "dust": 0.1, "mosaic": 0.0},
    "ct":        {"warp": 0.7, "invert": 0.4, "bend": 0.7, "bias": 0.5, "aniso_psf": 0.5, "resolution": 0.5, "poisson": 0.8, "gaussian": 0.9, "speckle": 0.2,
                  "rings": 0.5, "streaks": 0.6, "slice_jitter": 0.3, "fringes": 0.2, "cupping": 0.7, "seams": 0.2, "zingers": 0.3,
                  "rician": 0.0, "gibbs": 0.1, "ghosting": 0.0, "thick": 0.2, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.1, "dust": 0.1, "mosaic": 0.0},
    "mri":       {"warp": 0.8, "invert": 0.5, "bend": 0.8, "bias": 0.9, "aniso_psf": 0.5, "resolution": 0.6, "poisson": 0.0, "gaussian": 0.3, "speckle": 0.0,
                  "rings": 0.0, "streaks": 0.0, "slice_jitter": 0.2, "fringes": 0.0, "cupping": 0.0, "seams": 0.0, "zingers": 0.1,
                  "rician": 0.9, "gibbs": 0.6, "ghosting": 0.4, "thick": 0.6, "misalign": 0.0, "missing": 0.0, "stain": 0.0, "stripes": 0.0, "dust": 0.0, "mosaic": 0.0},
    "histology": {"warp": 0.9, "invert": 0.6, "bend": 0.8, "bias": 0.7, "aniso_psf": 0.6, "resolution": 0.5, "poisson": 0.3, "gaussian": 0.7, "speckle": 0.3,
                  "rings": 0.0, "streaks": 0.1, "slice_jitter": 0.8, "fringes": 0.0, "cupping": 0.0, "seams": 0.3, "zingers": 0.2,
                  "rician": 0.0, "gibbs": 0.0, "ghosting": 0.0, "thick": 0.7, "misalign": 0.7, "missing": 0.3, "stain": 0.7, "stripes": 0.5, "dust": 0.5, "mosaic": 0.5},
}
MODALITIES["mixed"] = {k: 0.35 for k in MODALITIES["hipct"]}
CROSS_MIX = 0.08                   # every nuisance outside the chosen modality's profile is still drawn with this probability


def nuisance(s: Sample, seed: int = 0, device="cuda", strength: float = 1.0, modality: str | None = None, warp: bool | None = None) -> dict:
    """One draw of a modality's acquisition on a clean sample (``hipct`` / ``ct`` / ``mri`` / ``histology`` /
    ``mixed``; None picks one), with cross-mixing. The geometry (``warp``; None draws it) moves the labels with the
    image; every photometric nuisance acts along a randomly drawn **acquisition axis** (slices, thick slices,
    anisotropic PSF, rings and cupping about it, streaks in planes across it); the image is kept in float to the end,
    where a display window maps it to 0-255. Returns the image, the warped occupancies ``(K, Z, Y, X)``, the warped
    shape tensors ``(K, 6, Z, Y, X)``, the layers' ``visibility (K,)`` and the draw's record."""
    g = torch.Generator(device=device).manual_seed(seed + 7919)
    u = lambda a, b: float(a + (b - a) * torch.rand(1, generator=g, device=device))
    lu = lambda a, b: float(math.exp(u(math.log(a), math.log(b))))
    names = list(MODALITIES); modality = modality or names[int(torch.randint(0, len(names), (1,), generator=g, device=device))]
    P = MODALITIES[modality]
    on = lambda key: u(0, 1) < max(P.get(key, 0.0), CROSS_MIX)
    img = s.image.clone().float(); Z, Y, X = img.shape; rec = {"modality": modality}
    occ = torch.stack([L.occupancy for L in s.layers]); T = torch.stack([L.tensor for L in s.layers])
    R = s.regions if s.regions is not None else torch.zeros_like(img)
    K = occ.shape[0]
    # ---- geometry: an elastic warp; labels follow; shape tensors by the Jacobian (S -> J⁻¹ S J⁻ᵀ), never folding
    if (warp if warp is not None else on("warp")):
        sig = lu(8, 30); amp = u(0, 6) * strength
        n3 = torch.stack([smooth_noise((Z, Y, X), sig, g, device) for _ in range(3)])
        for _ in range(10):
            J = _jacobian(n3 * amp); det = float(torch.linalg.det(J).min())
            if det > 0.2:
                break
            amp *= 0.7
        else:
            amp = 0.0; J = _jacobian(n3 * amp); det = 1.0
        disp = n3 * amp; grid = _warp_grid(disp)
        img = _resample(img, grid); occ = _resample(occ, grid); R = _resample(R, grid)
        Tw = _resample(T.reshape(K * 6, Z, Y, X), grid).reshape(K, 6, Z, Y, X)
        Ji = torch.linalg.inv(J)
        S = Ji[None] @ full3(Tw.permute(0, 2, 3, 4, 1)) @ Ji.transpose(-1, -2)[None]
        S = S / S.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-9)
        T = sym6(S).permute(0, 4, 1, 2, 3).contiguous(); rec["warp"] = (round(amp, 2), round(sig, 1), round(det, 2))
    # ---- the acquisition frame: the acquisition axis becomes dim 0 for every photometric nuisance
    ax = int(torch.randint(0, 3, (1,), generator=g, device=device)); rec["axis"] = ax
    img = img.movedim(ax, 0).contiguous(); A, B, C = img.shape
    cy, cx = None, None
    if True:                                                                                    # the rotation axis's in-plane position (rings, cupping)
        phi = u(0, 2 * math.pi); dist = (u(0, 0.5) if u(0, 1) < 0.3 else lu(0.5, 5.0)) * max(B, C)
        cy, cx = (B - 1) / 2 + dist * math.sin(phi), (C - 1) / 2 + dist * math.cos(phi)
    yy, xx = torch.meshgrid(torch.arange(B, device=device, dtype=torch.float32), torch.arange(C, device=device, dtype=torch.float32), indexing="ij")
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # sections: misalignment (no wrap-around), missing sections, stain variation
    if on("misalign"):
        sd = u(0.3, 1.5); sh = (sd * torch.randn(A, 2, device=device, generator=g)).round().long()
        img = torch.stack([_shift2d(img[a], int(sh[a, 0]), int(sh[a, 1])) for a in range(A)]); rec["misalign"] = round(sd, 2)
    if on("missing"):
        k = int(u(1, 4))
        for _ in range(k):
            a = int(u(1, A - 2)); img[a] = img[a - 1] if u(0, 1) < 0.5 else torch.full_like(img[a], float(img.mean()))
        rec["missing_sections"] = k
    if on("stain"):
        img = img * (1 + u(0.03, 0.2) * torch.randn(A, device=device, generator=g)).view(A, 1, 1) + u(0, 15) * torch.randn(A, device=device, generator=g).view(A, 1, 1); rec["stain"] = True
    # intensity: polarity, gamma and a bend on the normalised range (no clipping), a log-normal bias field
    if on("invert"):
        img = 255 - img; rec["inverted"] = True
    lo, hi = _quantile(img, 0.005), _quantile(img, 0.995); rng = max(hi - lo, 1e-3)
    gam = lu(0.5, 2.0); x = ((img - lo) / rng).clamp_min(0) ** gam; rec["gamma"] = round(gam, 2)
    if on("bend"):
        a, b = u(0.2, 0.8), u(0.2, 0.8); x = torch.where(x < a, b * x / a, b + (1 - b) * (x - a) / (1 - a)); rec["bend"] = (round(a, 2), round(b, 2))
    img = lo + rng * x
    if on("bias"):
        amp = u(0.05, 0.35) * (1.5 if modality == "mri" else 1.0) * strength
        img = img * torch.exp(amp * smooth_noise((A, B, C), lu(20, 60), g, device)); rec["bias"] = round(amp, 2)
    # CT: beam-hardening cupping (darker towards the rotation axis), phase-contrast fringes (an edge pair, either sign)
    if on("cupping"):
        c = u(0, 0.25); rmax = float(rr.max()) + 1.0
        img = img * (1 - c * (1 - (rr / rmax) ** 2))[None]; rec["cupping"] = round(c, 2)
    if on("fringes"):
        a = u(-0.3, 0.8); sb = lu(0.7, 1.5); img = img + a * (img - gaussian(img, sb)); rec["fringes"] = (round(a, 2), round(sb, 2))
    # histology tiles: a periodic illumination in the section plane
    if on("mosaic"):
        Tp = int(lu(24, 64)); oy, ox = int(u(0, Tp)), int(u(0, Tp))
        py = ((torch.arange(B, device=device) + oy) % Tp).float(); px = ((torch.arange(C, device=device) + ox) % Tp).float()
        vy = 1 - u(0.05, 0.2) * ((py - Tp / 2).abs() / (Tp / 2)) ** 2; vx = 1 - u(0.05, 0.2) * ((px - Tp / 2).abs() / (Tp / 2)) ** 2
        img = img * (vy[:, None] * vx[None, :])[None]; rec["mosaic"] = Tp
    # PSF (longer along the acquisition axis when anisotropic), resolution loss, k-space truncation, ghosting
    sig = [lu(0.3, 2.0) for _ in range(3)]
    if on("aniso_psf"):
        sig[0] = sig[0] * lu(1.5, 4.0)
    img = gaussian(img, tuple(sig), truncate=3.0); rec["psf"] = [round(v, 2) for v in sig]
    if on("resolution"):
        f = lu(1.2, 2.5); small = Fn.interpolate(img[None, None], scale_factor=1 / f, mode="trilinear", align_corners=False)
        img = Fn.interpolate(small, size=(A, B, C), mode="trilinear", align_corners=False)[0, 0]; rec["resolution"] = round(f, 2)
    if on("gibbs"):
        axes = [1, 2] if u(0, 1) < 0.5 else [int(u(1, 3))]
        if u(0, 1) < 0.3:
            axes.append(0)                                                                       # a 3-D acquisition
        c = u(0.25, 0.45); img = gibbs(img, axes, c); rec["gibbs"] = (axes, round(c, 2))
    if on("ghosting"):
        a = int(u(0, 3)); n = img.shape[a]; sh = int(u(n // 6, n // 2)); img = img + u(0.03, 0.15) * torch.roll(img, shifts=sh, dims=a); rec["ghosting"] = (a, sh)
    # noise: Poisson (on the current range), Gaussian, Rician, speckle
    lo, hi = _quantile(img, 0.001), _quantile(img, 0.999); rng = max(hi - lo, 1e-3)
    if on("poisson"):
        Pn = lu(15, 1500); img = lo + rng * torch.poisson(((img - lo) / rng).clamp_min(0) * Pn, generator=g) / Pn; rec["poisson_photons"] = round(Pn)
    if on("gaussian"):
        sd = u(0, 30) * strength * rng / 255; img = img + sd * torch.randn(img.shape, device=device, generator=g); rec["gaussian_sd"] = round(sd, 1)
    if on("rician"):
        sd = u(5, 45) * strength * rng / 255
        img = lo + torch.sqrt((img - lo + sd * torch.randn(img.shape, device=device, generator=g)) ** 2 + (sd * torch.randn(img.shape, device=device, generator=g)) ** 2); rec["rician_sd"] = round(sd, 1)
    if on("speckle"):
        img = img * (1 + u(0, 0.2) * torch.randn(img.shape, device=device, generator=g)); rec["speckle"] = True
    # thick slices, after the noise: acquired thick, resampled to isotropic (noise correlated along the axis)
    if on("thick"):
        f = lu(2, 6); small = Fn.interpolate(img[None, None], size=(max(2, int(A / f)), B, C), mode="area")
        img = Fn.interpolate(small, size=(A, B, C), mode="nearest" if u(0, 1) < 0.3 else "trilinear")[0, 0]; rec["thick_slices"] = round(f, 1)
    # CT artefacts in float: rings about the rotation axis, streaks in the slice planes at a random angle, seams, zingers
    unit = rng / 255
    if on("rings"):
        art = torch.zeros((B, C), device=device); amp = u(3, 30) * unit; n = int(lu(3, 30)); w = u(0.5, 1.5)
        r0, r1 = float(rr.min()), float(rr.max())
        for _ in range(n):
            rad = u(r0, r1); art += amp * u(0.5, 1.5) * (1 if u(0, 1) < 0.5 else -1) * torch.exp(-0.5 * ((rr - rad) / w) ** 2)
        img = img + art[None]; rec["rings"] = {"n": n, "centre": (round(cy, 1), round(cx, 1))}
    if on("streaks"):
        amp = u(5, 40) * unit; ang = u(0, 180); spread = u(5, 60); half = 0.5 * math.hypot(B, C)
        yc, xc = yy - (B - 1) / 2, xx - (C - 1) / 2; zs = torch.arange(A, device=device, dtype=torch.float32)
        n = int(torch.poisson(torch.tensor(0.5 * A, device=device), generator=g))
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
        img = img * (1 + u(0.02, 0.1) * torch.randn(A, device=device, generator=g)).view(A, 1, 1); rec["slice_jitter"] = True
    img = img.movedim(0, ax).contiguous()
    # ---- the display window: a little clipping at either end, then 8 bits
    clip = u(0.0, 0.05) if u(0, 1) < 0.1 else u(0.0, 0.015)
    lo, hi = _quantile(img, clip * u(0, 1)), _quantile(img, 1 - clip * u(0, 1))
    img = ((img - lo) / max(hi - lo, 1e-6) * 255).clamp(0, 255).round(); rec["window"] = round(clip, 3)
    occ = occ.clamp(0, 1)
    regs = torch.stack([R if L.role == "interface" else occ[k] for k, L in enumerate(s.layers)])
    vis = visibility(img, regs, [None if L.role == "isotropic" else (STEP_FULL if L.role == "interface" else 1.0) for L in s.layers])
    rec["visibility"] = [round(float(v), 2) for v in vis]
    return {"image": img, "occupancy": occ, "tensors": T, "visibility": vis, "record": rec}


def draw(shape=(128, 128, 128), seed: int = 0, device="cuda", modality: str | None = None, **kw):
    """One training pair: a modality (None draws one), content matched to it (MRI: thick tubes and sheets, no
    sub-voxel fibres; histology: cell stains, laminae), the clean sample, and one draw of its acquisition."""
    g = torch.Generator(device=device).manual_seed(seed + 104729)
    names = list(MODALITIES)
    modality = modality or names[int(torch.randint(0, len(names), (1,), generator=g, device=device))]
    content = {"mri": "mri", "histology": "histology"}.get(modality, "general")
    s = sample(shape, seed=seed, device=device, content=content, **kw)
    return s, nuisance(s, seed=seed, device=device, modality=modality)
