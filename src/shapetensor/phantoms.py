"""Phantoms with truth, at 20 µm voxels with the texture calibrated to a real HiP-CT crop:

* ``bundle`` — one fibre population (the clear-tissue floor),
* ``crossing`` — two populations crossing at ``cross_deg`` (where a single tensor can only report the bisector),
* ``interface`` — one population crossing a tissue step at 90 / 60 / 30° (where the structure tensor is dragged into
  the plane of the boundary),
* ``bend`` — three fascicles turning through 90° in grey-matter-level tissue (where a population defined by one
  direction splits a single bundle); ``straight=True`` is its control, the same fascicles without the turn,
* ``band_crossing`` — two straight fascicles crossing at ``cross_deg`` in grey-matter-level tissue (two populations
  that must stay apart where they overlap and each stay whole on either side).
  Both are prototypes' phantoms, not yet part of the validation.

Conventions. Arrays are ``(Z, Y, X)``; vectors are ``(z, y, x)``. Fibres are straight lines in x–z planes at random
y (the tissue is extruded along y), splatted into a density that darkens the tissue, with a mixture of widths,
per-fibre dispersion, and correlated Gaussian noise. The "angle" of a fibre is its angle to the z = const plane
within the x–z plane (90 = along z, 0 = along x); its direction is ``(sin a, 0, cos a)``.

Truth, on the (Z, X) plane for every y: ``dir1`` / ``dir2`` unit vectors (NaN where absent), ``npop``, ``dist`` (mm
to the nearest interface; NaN where there is none), ``region`` (2 white matter) and ``cross_deg``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from . import DATA_ROOT
from .filters import gaussian

PHANTOM_DIR = Path(DATA_ROOT) / "phantoms"


@dataclass
class Grid:
    vox: float = 0.02                                  # mm per voxel
    core: tuple = (128, 32, 128)                       # (nz, ny, nx) scored core, voxels
    margin: tuple = (32, 32, 32)                       # voxels added on each face (context for rho and the network)

    @property
    def shape(self):
        return tuple(c + 2 * m for c, m in zip(self.core, self.margin))

    def coords(self, axis: int) -> np.ndarray:
        """Block index -> mm, with the core's origin at 0."""
        n, m = self.shape[axis], self.margin[axis]
        return (np.arange(n) - m + 0.5) * self.vox

    @property
    def core_mm(self):
        return tuple(c * self.vox for c in self.core)

    @property
    def inner(self):
        """The slices of the scored core."""
        return tuple(slice(m, m + c) for c, m in zip(self.core, self.margin))


@dataclass
class TextureParams:
    """Fibre texture and noise, matched to a 20 µm HiP-CT crop (S-21-46): ``amp`` the darkening at the 99th
    percentile of the fibre density (grey levels); ``widths`` (sigma in voxels, probability) of the fibre width
    mixture; ``density`` fibres per mm² of cross-section; white noise of ``noise_sd`` blurred by ``noise_sigma``;
    per-fibre direction jitter ``dispersion_deg`` (in-plane SD) and ``pitch_deg`` (out of plane)."""
    amp: float = 100.0
    noise_sd: float = 95.7
    noise_sigma: float = 1.1
    density: float = 120.0
    widths: tuple = ((1.0, 0.45), (2.0, 0.3), (3.0, 0.15), (6.0, 0.1))
    pv_sigma: float = 1.0
    levels: dict = field(default_factory=lambda: {"wm": 164.0, "gm": 126.0, "csf": 60.0})
    dispersion_deg: float = 20.0
    pitch_deg: float = 10.0


def direction(angle_deg: float) -> np.ndarray:
    """The (z, y, x) unit vector of a fibre at ``angle_deg``."""
    a = math.radians(angle_deg)
    return np.array([math.sin(a), 0.0, math.cos(a)], np.float32)


# ---------------------------------------------------------------------------------------------------------------
# fibre splatting
# ---------------------------------------------------------------------------------------------------------------

class Splat:
    """Accumulates fibre densities per width group over the block, then blurs each group and sums (GPU)."""

    def __init__(self, grid: Grid, tex: TextureParams, rng: np.random.Generator, device="cuda"):
        self.g, self.tex, self.rng, self.dev = grid, tex, rng, device
        self.dens = [np.zeros(grid.shape, np.float32) for _ in tex.widths]
        self.probs = np.array([p for _, p in tex.widths]); self.probs /= self.probs.sum()

    def add_line(self, p0_mm: np.ndarray, d: np.ndarray, length_mm: float, mask2d=None):
        step = 0.5 * self.g.vox
        s = np.arange(-length_mm, length_mm, step)
        p = p0_mm[None] + s[:, None] * d[None]
        idx = np.round(p / self.g.vox + np.array(self.g.margin) - 0.5).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(self.g.shape)), axis=1)
        idx = idx[ok]
        if mask2d is not None and len(idx):
            idx = idx[mask2d[idx[:, 0], idx[:, 2]]]
        if not len(idx):
            return
        k = self.rng.choice(len(self.tex.widths), p=self.probs)
        np.add.at(self.dens[k], (idx[:, 0], idx[:, 1], idx[:, 2]), np.float32(1.0))

    def random_y(self) -> float:
        y0, y1 = self.g.coords(1)[[0, -1]]
        return float(self.rng.uniform(y0, y1))

    def jitter(self, d: np.ndarray) -> np.ndarray:
        """``d`` rotated by the texture's dispersion; the population's mean direction (the truth) is unchanged."""
        t = self.tex
        if t.dispersion_deg <= 0 and t.pitch_deg <= 0:
            return d
        a = math.radians(self.rng.normal(0, t.dispersion_deg)) if t.dispersion_deg > 0 else 0.0
        p = math.radians(self.rng.normal(0, t.pitch_deg)) if t.pitch_deg > 0 else 0.0
        dz, dx = d[0], d[2]
        dz, dx = dz * math.cos(a) - dx * math.sin(a), dz * math.sin(a) + dx * math.cos(a)
        out = np.array([dz * math.cos(p), math.sin(p), dx * math.cos(p)], np.float32)
        return out / np.linalg.norm(out)

    def render(self) -> torch.Tensor:
        """Blurred, 99th-percentile-normalised density ``(Z, Y, X)`` float32 on the device."""
        acc = None
        for (sig, _), d in zip(self.tex.widths, self.dens):
            t = gaussian(torch.as_tensor(d, device=self.dev), sig, truncate=3.0)
            acc = t if acc is None else acc.add_(t)
        pos = acc[acc > 0]
        if pos.numel():
            q = torch.quantile(pos[torch.randperm(pos.numel(), device=self.dev)[:2_000_000]], 0.99)
            acc /= q.clamp_min(1e-9)
        return acc


def add_population(sp: Splat, angle: float, frac: float, region2d=None):
    """Straight fibres at ``angle`` filling the block (restricted to ``region2d`` if given), density × ``frac``."""
    g = sp.g
    d = direction(angle)
    zc, xc, yc = g.coords(0), g.coords(2), g.coords(1)
    diag = math.hypot(zc[-1] - zc[0], xc[-1] - xc[0])
    n = int(sp.tex.density * frac * diag * (yc[-1] - yc[0]))
    nrm = np.array([d[2], 0.0, -d[0]], np.float32)
    centre = np.array([zc.mean(), 0.0, xc.mean()], np.float32)
    for _ in range(n):
        p0 = centre + sp.rng.uniform(-diag / 2, diag / 2) * nrm
        p0[1] = sp.random_y()
        sp.add_line(p0, sp.jitter(d), diag, mask2d=region2d)


def finish(grid: Grid, tex: TextureParams, base2d: np.ndarray, amp2d: np.ndarray, dens: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    """image = PV-blurred base (Z, X) − amp (Z, X) × clip(dens, 0, 1.5) + noise -> uint8 (Z, Y, X)."""
    dev = dens.device
    base = torch.as_tensor(ndi.gaussian_filter(base2d.astype(np.float32), tex.pv_sigma, mode="nearest"), device=dev)
    amp = torch.as_tensor(amp2d.astype(np.float32), device=dev)
    vol = base[:, None, :] - amp[:, None, :] * dens.clamp(0, 1.5)
    noise = torch.as_tensor(rng.normal(0, tex.noise_sd, size=vol.shape).astype(np.float32), device=dev)
    if tex.noise_sigma > 0:
        noise = gaussian(noise, tex.noise_sigma, truncate=3.0)
    return (vol + noise).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


def blank_truth(grid: Grid) -> dict:
    Z, X = grid.shape[0], grid.shape[2]
    nan3 = np.full((Z, X, 3), np.nan, np.float32)
    return {"dir1": nan3.copy(), "dir2": nan3.copy(), "npop": np.zeros((Z, X), np.int8), "dist": np.full((Z, X), np.nan, np.float32),
            "region": np.full((Z, X), 2, np.int8), "cross_deg": np.full((Z, X), np.nan, np.float32)}


# ---------------------------------------------------------------------------------------------------------------
# the three phantoms
# ---------------------------------------------------------------------------------------------------------------

def bundle(grid: Grid, tex: TextureParams, rng, angle: float = 60.0, device="cuda"):
    """One population at ``angle``, filling the block."""
    sp = Splat(grid, tex, rng, device)
    add_population(sp, angle, 1.0)
    tr = blank_truth(grid)
    tr["dir1"][:] = direction(angle); tr["npop"][:] = 1
    Z, X = grid.shape[0], grid.shape[2]
    img = finish(grid, tex, np.full((Z, X), tex.levels["wm"], np.float32), np.full((Z, X), tex.amp, np.float32), sp.render(), rng)
    return img, tr, {"angle": angle}


def crossing(grid: Grid, tex: TextureParams, rng, cross_deg: float = 60.0, ratio: float = 1.0, device="cuda"):
    """Two populations symmetric about the diagonal (angles 45 ± cross/2, so a 90° crossing is z against x and the
    two have distinct DEC colours), densities in the ratio ``ratio`` : 1."""
    a1, a2 = 45.0 + cross_deg / 2, 45.0 - cross_deg / 2
    f1 = ratio / (1 + ratio)
    sp = Splat(grid, tex, rng, device)
    add_population(sp, a1, 2 * f1); add_population(sp, a2, 2 * (1 - f1))
    tr = blank_truth(grid)
    tr["dir1"][:] = direction(a1); tr["dir2"][:] = direction(a2); tr["npop"][:] = 2; tr["cross_deg"][:] = cross_deg
    Z, X = grid.shape[0], grid.shape[2]
    img = finish(grid, tex, np.full((Z, X), tex.levels["wm"], np.float32), np.full((Z, X), tex.amp, np.float32), sp.render(), rng)
    return img, tr, {"cross_deg": cross_deg, "ratio": ratio, "angles": [a1, a2]}


def interface(grid: Grid, tex: TextureParams, rng, step: float = 40.0, angles=(90.0, 60.0, 30.0), device="cuda"):
    """A tissue step of ``step`` grey levels across z at the core's middle, crossed by fibres at ``angles`` (one
    angle per band along x). Truth ``dist`` is the distance to the step."""
    zc, xc = grid.coords(0), grid.coords(2)
    Z, X = np.meshgrid(zc, xc, indexing="ij")
    z_mid = grid.core_mm[0] / 2
    bands = np.linspace(0, grid.core_mm[2], len(angles) + 1); bands[0], bands[-1] = -1e9, 1e9
    sp = Splat(grid, tex, rng, device)
    tr = blank_truth(grid)
    for (x0, x1), a in zip(zip(bands[:-1], bands[1:]), angles):
        sel = (X >= x0) & (X < x1)
        add_population(sp, a, 1.0, region2d=sel)
        tr["dir1"][sel] = direction(a)
    tr["npop"][:] = 1; tr["dist"][:] = np.abs(Z - z_mid)
    wm = tex.levels["wm"]
    base2d = np.where(Z >= z_mid, wm + step, wm).astype(np.float32)
    img = finish(grid, tex, base2d, np.full(Z.shape, tex.amp, np.float32), sp.render(), rng)
    return img, tr, {"step": step, "angles": list(angles), "z_mid_mm": z_mid}


#: The bend phantom's layout, in mm on its own grid (core 192 x 32 x 192 voxels): the turns' common centre (z, x),
#: the fascicles' inner radii and their width.
BEND_GRID = Grid(core=(192, 32, 192))
BEND_CENTRE, BEND_RADII, BEND_WIDTH = (1.0, 2.6), (0.3, 0.8, 1.3), 0.3


def bend_geometry(grid: Grid, centre=BEND_CENTRE, radii=BEND_RADII, width=BEND_WIDTH, straight: bool = False):
    """On the (Z, X) plane: ``fid`` (0 none, 1.. by radius), the fibres' tangent ``tan (Z, X, 2)`` as (dz, dx), the
    position ``s`` (mm) along each fascicle's mid-line (negative on the way in, 0 to r·π/2 through the turn, beyond on
    the way out) and ``bend`` (the quarter circles). Each fascicle runs up z, turns about ``centre`` and leaves along x;
    ``straight``: the way-in arm continued through the block."""
    Z, X = np.meshgrid(grid.coords(0), grid.coords(2), indexing="ij")
    zc, xc = centre
    fid = np.zeros(Z.shape, np.int8); tan = np.full(Z.shape + (2,), np.nan, np.float32)
    s = np.full(Z.shape, np.nan, np.float32); bend = np.zeros(Z.shape, bool)
    rad = np.hypot(Z - zc, xc - X); phi = np.arctan2(Z - zc, xc - X)
    for k, r0 in enumerate(radii, 1):
        r1, rm = r0 + width, r0 + width / 2
        if straight:
            band = (X >= xc - r1) & (X < xc - r0)
            fid[band] = k; tan[band] = (1, 0); s[band] = Z[band] - zc
            continue
        arm_in = (Z < zc) & (X >= xc - r1) & (X < xc - r0)
        arc = (Z >= zc) & (X <= xc) & (rad >= r0) & (rad < r1)
        arm_out = (X > xc) & (Z >= zc + r0) & (Z < zc + r1)
        fid[arm_in | arc | arm_out] = k
        tan[arm_in] = (1, 0); tan[arm_out] = (0, 1)
        tan[arc] = np.stack([np.cos(phi[arc]), np.sin(phi[arc])], -1)
        s[arm_in] = Z[arm_in] - zc; s[arc] = rm * phi[arc]; s[arm_out] = rm * math.pi / 2 + (X[arm_out] - xc)
        bend |= arc
    return fid, tan, s, bend


def _bend_path(r, z0, x1, step, centre, straight, z1):
    """One fibre at radius ``r``: up z from ``z0``, the quarter circle, along x to ``x1`` (``straight``: up z to ``z1``):
    ``(n, 2)`` (z, x) mm and its arc length."""
    zc, xc = centre
    if straight:
        a = np.arange(z0, z1, step); p = np.stack([a, np.full_like(a, xc - r)], 1)
    else:
        a = np.arange(z0, zc, step); ph = np.arange(0, math.pi / 2, step / r); b = np.arange(xc, x1, step)
        p = np.concatenate([np.stack([a, np.full_like(a, xc - r)], 1), np.stack([zc + r * np.sin(ph), xc - r * np.cos(ph)], 1),
                            np.stack([np.full_like(b, zc + r), b], 1)])
    return p, np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def bend(grid: Grid, tex: TextureParams, rng, centre=BEND_CENTRE, radii=BEND_RADII, width=BEND_WIDTH, straight: bool = False,
         device="cuda"):
    """Three fascicles ``width`` mm wide turning through 90° about ``centre`` (inner radii ``radii``) in grey-matter-level
    tissue; ``straight``: the control without the turn. Each fibre takes a random radius in its fascicle and deviates
    from it by the texture's dispersion (rotated about a random point of its path) and pitch, clipped to the fascicle.
    Truth: ``dir1`` the tangent, ``npop`` 1 in the fascicles, ``region`` 2 in them and 1 elsewhere, and ``fid``, ``s``,
    ``bend`` (see :func:`bend_geometry`)."""
    sp = Splat(grid, tex, rng, device)
    fid, tan, s, arc = bend_geometry(grid, centre, radii, width, straight)
    zr, xr, yr = grid.coords(0), grid.coords(2), grid.coords(1)
    step = 0.5 * grid.vox
    disp, pitch = math.radians(tex.dispersion_deg), math.radians(tex.pitch_deg)
    span = min(4.0, width / max(math.sin(disp), 1e-3))                 # a dispersed fibre's typical run inside the band
    for k, r0 in enumerate(radii, 1):
        region = fid == k
        length = (zr[-1] - zr[0]) if straight else (centre[0] - zr[0]) + (r0 + width / 2) * math.pi / 2 + (xr[-1] - centre[1])
        for _ in range(int(tex.density * width * (yr[-1] - yr[0]) * length / span)):
            p, arcl = _bend_path(rng.uniform(r0, r0 + width), zr[0], xr[-1], step, centre, straight, zr[-1])
            j = rng.integers(len(p)); a = rng.normal(0, disp); c, sn = math.cos(a), math.sin(a)
            q = p - p[j]; q = np.stack([q[:, 0] * c - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * c], 1) + p[j]
            y = rng.uniform(yr[0], yr[-1]) + (arcl - arcl[j]) * math.tan(rng.normal(0, pitch))
            idx = np.round(np.stack([q[:, 0], y, q[:, 1]], 1) / grid.vox + np.array(grid.margin) - 0.5).astype(np.int64)
            idx = idx[np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=1)]
            idx = idx[region[idx[:, 0], idx[:, 2]]]
            if len(idx):
                np.add.at(sp.dens[rng.choice(len(tex.widths), p=sp.probs)], (idx[:, 0], idx[:, 1], idx[:, 2]), np.float32(1.0))
    tr = blank_truth(grid)
    inside = fid > 0
    tr["dir1"][inside] = np.stack([tan[inside][:, 0], np.zeros(int(inside.sum()), np.float32), tan[inside][:, 1]], 1)
    tr["npop"][inside] = 1; tr["region"][~inside] = 1
    tr.update({"fid": fid, "s": s, "bend": arc})
    img = finish(grid, tex, np.where(inside, tex.levels["wm"], tex.levels["gm"]).astype(np.float32),
                 np.where(inside, tex.amp, 0.0).astype(np.float32), sp.render(), rng)
    return img, tr, {"centre_mm": list(centre), "radii_mm": list(radii), "width_mm": width, "straight": straight}


def band_crossing(grid: Grid, tex: TextureParams, rng, cross_deg: float = 90.0, width: float = 0.4, t_junction: bool = False, device="cuda"):
    """Two straight fascicles ``width`` mm wide through the core's centre, at 45 ± ``cross_deg`` / 2 (as ``crossing``),
    in grey-matter-level tissue. Truth: ``dir1`` / ``dir2`` the two directions where each fascicle is (``dir1`` the only
    one where a single fascicle is), ``npop`` 1 or 2, ``region`` 2 in the fascicles and 1 elsewhere, and the masks
    ``band1`` / ``band2`` on the (Z, X) plane. ``t_junction``: the second fascicle stops at the first one's centre line
    (a T: its fibres end inside the first band), and ``corridor2`` is the strip it would have continued along."""
    a1, a2 = 45.0 + cross_deg / 2, 45.0 - cross_deg / 2
    Z, X = np.meshgrid(grid.coords(0), grid.coords(2), indexing="ij")
    zc, xc = grid.core_mm[0] / 2, grid.core_mm[2] / 2
    bands, signed = [], []
    for a in (a1, a2):
        d = direction(a)
        signed.append((Z - zc) * d[2] - (X - xc) * d[0])                                 # signed distance from the centre line
        bands.append(np.abs(signed[-1]) < width / 2)
    corridor2 = bands[1].copy()
    if t_junction:
        bands[1] = bands[1] & (signed[0] <= 0)
    sp = Splat(grid, tex, rng, device)
    for a, m in zip((a1, a2), bands):
        add_population(sp, a, 1.0, region2d=m)
    tr = blank_truth(grid)
    b1, b2 = bands
    tr["dir1"][b1] = direction(a1); tr["dir1"][b2 & ~b1] = direction(a2); tr["dir2"][b1 & b2] = direction(a2)
    tr["npop"][b1 | b2] = 1; tr["npop"][b1 & b2] = 2; tr["region"][~(b1 | b2)] = 1; tr["cross_deg"][b1 & b2] = cross_deg
    tr.update({"band1": b1, "band2": b2, "corridor2": corridor2})
    inside = b1 | b2
    img = finish(grid, tex, np.where(inside, tex.levels["wm"], tex.levels["gm"]).astype(np.float32),
                 np.where(inside, tex.amp, 0.0).astype(np.float32), sp.render(), rng)
    return img, tr, {"cross_deg": cross_deg, "angles": [a1, a2], "width_mm": width, "t_junction": t_junction}


PHANTOMS = {"bundle": bundle, "crossing": crossing, "interface": interface, "bend": bend, "band_crossing": band_crossing}


def build(name: str, seed: int = 0, grid: Grid | None = None, tex: TextureParams | None = None, texture: float = 1.0, device="cuda", **kwargs):
    """``(image uint8 (Z, Y, X), truth, meta)`` of phantom ``name`` ("bundle", "crossing", "interface") at ``seed``;
    ``texture`` scales the fibre darkening (1 = the calibrated coherent-bundle level; the real crop lies at 0.25–0.5)."""
    grid = grid or (BEND_GRID if name in ("bend", "band_crossing") else Grid()); tex = tex or TextureParams()
    if texture != 1.0:
        tex = TextureParams(**{**asdict(tex), "amp": tex.amp * texture})
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    img, tr, meta = PHANTOMS[name](grid, tex, rng, device=device, **kwargs)
    return img, tr, {"name": name, "seed": seed, "grid": asdict(grid), "texture": asdict(tex), **meta}


def save(path, img, tr, meta) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, image=img, meta=json.dumps(meta), **tr)
    return p


def load(path):
    with np.load(path) as z:
        img = z["image"]; meta = json.loads(str(z["meta"]))
        tr = {k: z[k] for k in z.files if k not in ("image", "meta")}
    g = meta["grid"]
    return img, tr, meta, Grid(vox=g["vox"], core=tuple(g["core"]), margin=tuple(g["margin"]))


def truth_field(tr: dict, key: str, ny: int) -> torch.Tensor:
    """The truth direction ``key`` ("dir1" / "dir2") as a ``(3, Z, Y, X)`` field (NaN where absent)."""
    d = torch.as_tensor(tr[key]).permute(2, 0, 1)                                       # (3, Z, X)
    return d[:, :, None, :].expand(3, d.shape[1], ny, d.shape[2])
