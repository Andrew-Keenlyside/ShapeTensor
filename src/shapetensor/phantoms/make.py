"""The phantom suite (docs/TCA_FIELD_REVIEW.md §7), built at a chosen voxel size with the structure continued into
the margins, so orientations near the scored core are never computed against a block face.

Conventions. Arrays are ``(Z, Y, X)``; vectors are ``(z, y, x)``. Fibres are straight lines in x-z planes at random
y (the tissue is extruded along y, so one y row of sites is representative), splatted into a density that DARKENS the
tissue (dehydrated-WM convention) with a mixture of widths, plus Gaussian noise. "Angle" of a fibre is its angle to
the z = const plane within the x-z plane: 90 = along z, 0 = along x; direction ``(sin a, 0, cos a)``.

Truth, on the mid-y plane, ``(Z, X, ...)`` over the whole block: ``dir1``/``dir2`` unit vectors (NaN where absent),
``frac1``, ``npop``, ``dist`` (mm to the nearest interface; NaN where there is none), ``bnorm`` (its normal),
``region`` (0 CSF/background, 1 grey matter, 2 white matter, 3 vessel/other), and for crossings ``cross_deg``.

| id | phantom | twin | what it tests |
|---|---|---|---|
| P1 | gyrus: a WM blade, 2.5 mm GM, CSF; a fan crossing the boundary | ``P1t`` no step | the grey/white shell, the hybrid |
| P2 | P1 with U-fibres parallel to the boundary in the superficial band | ``P2t`` | true boundary-parallel fibres must be kept |
| P3-<d> | WM interface of step d (0/8/15/30); fibres cross it at 90/60/30 deg | ``P3-0`` | the headline result, the rho sweep |
| P4-<kind>-c<deg>-f<ratio> | two populations crossing at deg, fraction ratio; single | ``P4-...-single`` | N, directions, fractions |
| P5 | a fan crossed by a straight bundle (continuously varying angle) | | N where the angle varies |
| P6 | fibres plus dark vessels at random 3-D orientations | ``P6t`` no vessels | the vessel-halo failure |
| P7 | a 1 mm laminar slab with internal fibres, darker tissue either side | | the callosum case |
| P8 | homogeneous oblique fibre texture | | the clear-tissue floor; null calibration |
| P9 | noise only | | false populations; detector nulls |
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .. import DATA_ROOT
from ..filters import gaussian

PHANTOM_DIR = Path(DATA_ROOT) / "phantoms"


@dataclass
class Grid:
    vox: float = 0.02                                  # mm per voxel
    core: tuple = (512, 64, 384)                       # (nz, ny, nx) scored core, voxels
    margin: tuple = (128, 128, 128)                    # voxels added on each face (context for rho and the network)

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
    def core_zx(self):
        return ((self.margin[0], self.margin[0] + self.core[0]), (self.margin[2], self.margin[2] + self.core[2]))

    @property
    def y_mid(self) -> int:
        return self.shape[1] // 2


def default_grid(vox: float = 0.02, rho_max: float = 32.0, truncate: float = 3.0, net_margin: int = 32,
                 core=(512, 64, 384)) -> Grid:
    m = int(truncate * rho_max + 0.5) + net_margin
    m = int(math.ceil(m / 8) * 8)
    return Grid(vox=vox, core=tuple(core), margin=(m, m, m))


@dataclass
class TextureParams:
    """Fibre texture and noise. ``amp`` is the darkening at the 99th percentile of the fibre density (grey levels);
    ``widths`` are (sigma in voxels, probability) of the fibre width mixture; ``density`` fibres per mm^2 of
    cross-section; ``noise_sd`` Gaussian noise (grey levels); ``pv_sigma`` partial-volume blur of the tissue map."""
    amp: float = 12.0
    noise_sd: float = 8.0                              # SD of the white noise BEFORE the correlation blur
    noise_sigma: float = 0.0                           # Gaussian blur of the noise (voxels): 0 = white; HiP-CT noise is correlated
    density: float = 60.0
    widths: tuple = ((1.0, 0.5), (2.0, 0.35), (3.0, 0.15))
    pv_sigma: float = 1.0
    gm_amp_frac: float = 0.33                          # fibre darkening inside grey matter relative to WM
    levels: dict = field(default_factory=lambda: {"wm": 190.0, "gm": 150.0, "csf": 60.0})
    step_scale: float = 1.0                            # multiplies the P3/P7 steps (grey levels)
    dispersion_deg: float = 0.0                        # per-fibre direction jitter (Gaussian SD, in the x-z plane)
    pitch_deg: float = 0.0                             # per-fibre out-of-plane jitter (towards y), Gaussian SD


def _dir(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.sin(a), 0.0, math.cos(a)], np.float32)


# ------------------------------------------------------------------------------------------------------------------
# fibre splatting
# ------------------------------------------------------------------------------------------------------------------

class Splat:
    """Accumulates fibre densities per width group over the block, then blurs each group and sums (GPU)."""

    def __init__(self, grid: Grid, tex: TextureParams, rng: np.random.Generator, device="cuda"):
        self.g, self.tex, self.rng, self.dev = grid, tex, rng, device
        self.dens = [np.zeros(grid.shape, np.float32) for _ in tex.widths]
        self.probs = np.array([p for _, p in tex.widths]); self.probs /= self.probs.sum()

    def add_line(self, p0_mm: np.ndarray, d: np.ndarray, length_mm: float, mask2d=None, weight: float = 1.0):
        """A straight fibre through ``p0`` (z, y, x mm) along unit ``d``, spanning ±length; points every 0.5 voxel.
        ``mask2d (Z, X)`` restricts deposition (e.g. to WM); ``weight`` scales its density."""
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
        np.add.at(self.dens[k], (idx[:, 0], idx[:, 1], idx[:, 2]), np.float32(weight))

    def random_y(self) -> float:
        z0, z1 = self.g.coords(1)[[0, -1]]
        return float(self.rng.uniform(z0, z1))

    def jitter(self, d: np.ndarray) -> np.ndarray:
        """``d`` rotated by the texture's dispersion: a Gaussian angle in the x-z plane and a Gaussian pitch
        towards y. The population's MEAN direction (the truth) is unchanged; single fibres disperse about it."""
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


def _add_population(sp: Splat, angle: float, frac: float, region2d=None, amp_scale: float = 1.0):
    """Straight fibres at ``angle`` filling the block (restricted to ``region2d`` if given), density x ``frac``."""
    g = sp.g
    d = _dir(angle)
    zc, xc = g.coords(0), g.coords(2)
    Lz, Lx = zc[-1] - zc[0], xc[-1] - xc[0]
    # cross-section perpendicular to d within the x-z plane has width w = |Lz cos a| + |Lx sin a| ... use the block diagonal
    diag = math.hypot(Lz, Lx)
    yc = g.coords(1)
    Ly = yc[-1] - yc[0]
    n = int(sp.tex.density * frac * diag * Ly)
    nrm = np.array([d[2], 0.0, -d[0]], np.float32)                    # in-plane normal
    centre = np.array([zc.mean(), 0.0, xc.mean()], np.float32)
    for _ in range(n):
        off = sp.rng.uniform(-diag / 2, diag / 2)
        p0 = centre + off * nrm
        p0[1] = sp.random_y()
        sp.add_line(p0, sp.jitter(d), diag, mask2d=region2d, weight=amp_scale)


def _finish(grid: Grid, tex: TextureParams, base2d: np.ndarray, amp2d: np.ndarray, dens: torch.Tensor,
            rng: np.random.Generator, extra: torch.Tensor | None = None) -> np.ndarray:
    """image = PV-blurred base (Z, X) - amp (Z, X) * clip(dens, 0, 1.5) + extra + noise -> uint8 (Z, Y, X)."""
    dev = dens.device
    base = torch.as_tensor(ndi.gaussian_filter(base2d.astype(np.float32), tex.pv_sigma, mode="nearest"), device=dev)
    amp = torch.as_tensor(amp2d.astype(np.float32), device=dev)
    vol = base[:, None, :] - amp[:, None, :] * dens.clamp(0, 1.5)
    if extra is not None:
        vol = vol + extra
    noise = torch.as_tensor(rng.normal(0, tex.noise_sd, size=vol.shape).astype(np.float32), device=dev)
    if tex.noise_sigma > 0:
        noise = gaussian(noise, tex.noise_sigma, truncate=3.0)
    img = (vol + noise).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
    return img


def _blank_truth(grid: Grid) -> dict:
    Z, X = grid.shape[0], grid.shape[2]
    nan3 = np.full((Z, X, 3), np.nan, np.float32)
    return {"dir1": nan3.copy(), "dir2": nan3.copy(), "frac1": np.full((Z, X), np.nan, np.float32),
            "npop": np.zeros((Z, X), np.int8), "dist": np.full((Z, X), np.nan, np.float32), "bnorm": nan3.copy(),
            "region": np.full((Z, X), 2, np.int8), "cross_deg": np.full((Z, X), np.nan, np.float32)}


# ------------------------------------------------------------------------------------------------------------------
# P3: tract interface
# ------------------------------------------------------------------------------------------------------------------

def interface(grid: Grid, tex: TextureParams, rng, step: float = 15.0, angles=(90.0, 60.0, 30.0), device="cuda"):
    zc, xc = grid.coords(0), grid.coords(2)
    Z, X = np.meshgrid(zc, xc, indexing="ij")
    z_mid = grid.core_mm[0] / 2
    bands = np.linspace(0, grid.core_mm[2], len(angles) + 1)
    bands[0], bands[-1] = -1e9, 1e9                                     # the outer bands extend into the margins
    sp = Splat(grid, tex, rng, device)
    tr = _blank_truth(grid)
    for (x0, x1), a in zip(zip(bands[:-1], bands[1:]), angles):
        sel = (X >= x0) & (X < x1)
        _add_population(sp, a, 1.0, region2d=sel)
        tr["dir1"][sel] = _dir(a)
    tr["npop"][:] = 1; tr["frac1"][:] = 1.0
    tr["dist"][:] = np.abs(Z - z_mid)
    tr["bnorm"][:] = (1.0, 0.0, 0.0)
    wm = tex.levels["wm"]
    base2d = np.where(Z >= z_mid, wm + step * tex.step_scale, wm).astype(np.float32)
    amp2d = np.full(Z.shape, tex.amp, np.float32)
    img = _finish(grid, tex, base2d, amp2d, sp.render(), rng)
    return img, tr, {"step": step, "angles": list(angles), "z_mid_mm": z_mid}


# ------------------------------------------------------------------------------------------------------------------
# P1 / P2: gyrus with a fan, with or without U-fibres
# ------------------------------------------------------------------------------------------------------------------

def _gyrus_tissue(grid: Grid, blade_hw=1.4, gm_mm=2.5):
    zc, xc = grid.coords(0), grid.coords(2)
    Z, X = np.meshgrid(zc, xc, indexing="ij")
    cz, cx = grid.core_mm[0], grid.core_mm[2]
    crown_z, base_z, bx = 0.62 * cz, 0.33 * cz, cx / 2
    wm = Z < base_z
    stalk = (np.abs(X - bx) <= blade_hw) & (Z <= crown_z - blade_hw)
    cap = (X - bx) ** 2 + (Z - (crown_z - blade_hw)) ** 2 <= blade_hw ** 2
    wm |= stalk | cap
    d_out = ndi.distance_transform_edt(~wm) * grid.vox
    gm = ~wm & (d_out <= gm_mm)
    sdf = ndi.distance_transform_edt(wm) * grid.vox - d_out              # + inside WM, mm
    focus = np.array([-0.75 * cz, 0.0, bx], np.float32)                 # (z, y, x) mm: the fan's origin, below
    return wm, gm, sdf, Z, X, focus


def _fan_dir(Z, X, focus):
    d = np.stack([Z - focus[0], np.zeros_like(Z), X - focus[2]], -1)
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)


def gyrus(grid: Grid, tex: TextureParams, rng, step: bool = True, ufibres: bool = False, device="cuda"):
    wm, gm, sdf, Z, X, focus = _gyrus_tissue(grid)
    sp = Splat(grid, tex, rng, device)
    tr = _blank_truth(grid)
    yc = grid.coords(1)
    fan = _fan_dir(Z, X, focus)
    # the fan: lines from a jittered focus over +-38 deg, through WM and (faintly) GM
    cz = grid.core_mm[0]
    n_fan = int(tex.density * 0.6 * (grid.core_mm[2] + 2 * grid.margin[2] * grid.vox) * (yc[-1] - yc[0]))
    tissue = wm | gm
    deep = sdf > 1.6 if ufibres else tissue
    for _ in range(n_fan):
        t = math.radians(rng.uniform(-38, 38))
        f0 = focus + np.array([0.0, 0.0, rng.uniform(-1.2, 1.2)], np.float32)
        d = np.array([math.cos(t), 0.0, math.sin(t)], np.float32)
        f0[1] = sp.random_y()
        sp.add_line(f0, sp.jitter(d), 2.5 * cz, mask2d=deep)
    tr["dir1"][tissue] = fan[tissue]
    if ufibres:
        import contourpy
        cgen = contourpy.contour_generator(z=sdf)
        band = (sdf >= 0.15) & (sdf <= 1.2)
        gx, gz = np.gradient(ndi.gaussian_filter(sdf, 5.0))
        nrm = np.stack([gx, gz], -1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
        tang = np.stack([-nrm[..., 1], np.zeros_like(nrm[..., 0]), nrm[..., 0]], -1).astype(np.float32)   # (z, y, x)
        zc, xc = grid.coords(0), grid.coords(2)
        n_u = int(tex.density * 0.8 * 1.05 * 2 * (grid.core_mm[2]) * (yc[-1] - yc[0]) / 4)
        for _ in range(n_u):
            dpt = rng.uniform(0.15, 1.2)
            cs = [c for c in cgen.lines(dpt) if len(c) > 50]            # vertices as (column = x index, row = z index)
            if not cs:
                continue
            c = cs[rng.integers(len(cs))]
            pz = np.interp(c[:, 1], np.arange(len(zc)), zc); px = np.interp(c[:, 0], np.arange(len(xc)), xc)
            L = rng.uniform(3.0, 6.0)
            s = np.r_[0, np.cumsum(np.hypot(np.diff(pz), np.diff(px)))]
            if s[-1] < 0.5:
                continue
            s0 = rng.uniform(0, max(1e-3, s[-1] - L))
            seg = (s >= s0) & (s <= s0 + L)
            y = sp.random_y()
            pts = np.stack([pz[seg], np.full(seg.sum(), y), px[seg]], 1)
            for q in range(len(pts) - 1):
                d = pts[q + 1] - pts[q]; ln = np.linalg.norm(d)
                if ln > 0:
                    sp.add_line(pts[q] + d / 2, d / ln, ln / 2 + grid.vox)
        tr["dir1"][:] = np.nan
        tr["dir1"][band] = tang[band]
        tr["dir1"][deep & tissue] = fan[deep & tissue]
    tr["npop"][tissue] = 1; tr["frac1"][tissue] = 1.0
    tr["dist"][:] = np.abs(sdf)
    gx, gz = np.gradient(ndi.gaussian_filter(sdf, 3.0))
    n = np.stack([gz, np.zeros_like(gx), gx], -1)                        # (z, y, x) normal of the GW boundary
    tr["bnorm"][:] = (n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)).astype(np.float32)
    tr["region"][:] = np.select([~tissue, gm, wm], [0, 1, 2]).astype(np.int8)
    lv = tex.levels
    if step:
        base2d = np.where(wm, lv["wm"], np.where(gm, lv["gm"], lv["csf"])).astype(np.float32)
    else:
        base2d = np.full(Z.shape, lv["wm"], np.float32)
    amp2d = np.where(wm, tex.amp, np.where(gm, tex.amp * tex.gm_amp_frac, 0.0)).astype(np.float32)
    img = _finish(grid, tex, base2d, amp2d, sp.render(), rng)
    return img, tr, {"step": step, "ufibres": ufibres, "focus_mm": focus.tolist()}


# ------------------------------------------------------------------------------------------------------------------
# P4: crossings; P5: fan x bundle
# ------------------------------------------------------------------------------------------------------------------

def crossing(grid: Grid, tex: TextureParams, rng, cross_deg: float = 60.0, ratio: float = 1.0, kind: str = "lines",
             single: bool = False, device="cuda"):
    """Two populations symmetric about the z axis (angles 90 +- cross/2), fractions ratio:1; ``single``: the first
    only, at 90 + cross/2 (oblique). ``kind='rods'``: dark rods on a lattice (the clean sanity check)."""
    a1, a2 = 90.0 + cross_deg / 2, 90.0 - cross_deg / 2
    f1 = ratio / (1 + ratio) if not single else 1.0
    f2 = 1 - f1
    tr = _blank_truth(grid)
    Z, X = grid.shape[0], grid.shape[2]
    tr["dir1"][:] = _dir(a1); tr["npop"][:] = 1; tr["frac1"][:] = f1
    if not single:
        tr["dir2"][:] = _dir(a2); tr["npop"][:] = 2; tr["cross_deg"][:] = cross_deg
    if kind == "lines":
        sp = Splat(grid, tex, rng, device)
        _add_population(sp, a1, 2 * f1)
        if not single:
            _add_population(sp, a2, 2 * f2)
        dens = sp.render()
        extra = None
    else:
        dens = torch.zeros(grid.shape, device=device)
        occ = _rods(grid, a1, rng, device, phase=0.0)
        if not single:
            occ = torch.maximum(occ, _rods(grid, a2, rng, device, phase=3.3))
        extra = -(tex.levels["wm"] - 80.0) * occ * (0.5 if not single else 1.0) * 0 - 60.0 * occ
    base2d = np.full((Z, X), tex.levels["wm"], np.float32)
    amp2d = np.full((Z, X), tex.amp if kind == "lines" else 0.0, np.float32)
    img = _finish(grid, tex, base2d, amp2d, dens, rng, extra=extra)
    return img, tr, {"cross_deg": cross_deg, "ratio": ratio, "kind": kind, "single": single, "angles": [a1, a2]}


def _rods(grid: Grid, angle: float, rng, device, spacing_vox: float = 10.0, radius_vox: float = 1.5, phase: float = 0.0):
    """Rod occupancy (0-1) of one population along ``angle``: axes on a lattice in y and in the in-plane normal."""
    Z, Y, X = grid.shape
    z = torch.arange(Z, device=device, dtype=torch.float32)[:, None, None] + 0.5
    y = torch.arange(Y, device=device, dtype=torch.float32)[None, :, None] + 0.5
    x = torch.arange(X, device=device, dtype=torch.float32)[None, None, :] + 0.5
    d = _dir(angle)
    nrm = (d[2], 0.0, -d[0])
    sn = z * nrm[0] + x * nrm[2] + phase
    dn = (sn % spacing_vox) - spacing_vox / 2
    dy = (y % spacing_vox) - spacing_vox / 2
    return (radius_vox + 0.5 - torch.sqrt(dn ** 2 + dy ** 2)).clamp(0, 1)


def fan_bundle(grid: Grid, tex: TextureParams, rng, device="cuda"):
    """A fan from a focus below the block crossed by a straight bundle along x, both at half density."""
    zc, xc = grid.coords(0), grid.coords(2)
    Z, X = np.meshgrid(zc, xc, indexing="ij")
    cz, _, cx = grid.core_mm
    focus = np.array([-0.75 * cz, 0.0, cx / 2], np.float32)
    sp = Splat(grid, tex, rng, device)
    yc = grid.coords(1)
    n_fan = int(tex.density * 0.5 * 1.4 * (cx + 2 * grid.margin[2] * grid.vox) * (yc[-1] - yc[0]))
    for _ in range(n_fan):
        t = math.radians(rng.uniform(-40, 40))
        f0 = focus + np.array([0.0, 0.0, rng.uniform(-1.2, 1.2)], np.float32); f0[1] = sp.random_y()
        sp.add_line(f0, sp.jitter(np.array([math.cos(t), 0.0, math.sin(t)], np.float32)), 2.5 * cz)
    _add_population(sp, 0.0, 1.0)                                        # the bundle along x (half of total density)
    fan = _fan_dir(Z, X, focus)
    tr = _blank_truth(grid)
    tr["dir1"][:] = fan; tr["dir2"][:] = (0.0, 0.0, 1.0); tr["npop"][:] = 2; tr["frac1"][:] = 0.5
    tr["cross_deg"][:] = np.degrees(np.arccos(np.clip(np.abs(fan[..., 2]), 0, 1)))
    base2d = np.full(Z.shape, tex.levels["wm"], np.float32)
    amp2d = np.full(Z.shape, tex.amp, np.float32)
    img = _finish(grid, tex, base2d, amp2d, sp.render(), rng)
    return img, tr, {"focus_mm": focus.tolist()}


# ------------------------------------------------------------------------------------------------------------------
# P6 vessels, P7 laminar slab, P8 homogeneous, P9 null
# ------------------------------------------------------------------------------------------------------------------

def homogeneous(grid: Grid, tex: TextureParams, rng, angle: float = 60.0, device="cuda"):
    sp = Splat(grid, tex, rng, device)
    _add_population(sp, angle, 1.0)
    tr = _blank_truth(grid)
    tr["dir1"][:] = _dir(angle); tr["npop"][:] = 1; tr["frac1"][:] = 1.0
    Z, X = grid.shape[0], grid.shape[2]
    img = _finish(grid, tex, np.full((Z, X), tex.levels["wm"], np.float32), np.full((Z, X), tex.amp, np.float32), sp.render(), rng)
    return img, tr, {"angle": angle}


def null(grid: Grid, tex: TextureParams, rng, device="cuda"):
    tr = _blank_truth(grid)
    Z, X = grid.shape[0], grid.shape[2]
    dens = torch.zeros(grid.shape, device=device)
    img = _finish(grid, tex, np.full((Z, X), tex.levels["wm"], np.float32), np.zeros((Z, X), np.float32), dens, rng)
    return img, tr, {}


def vessels(grid: Grid, tex: TextureParams, rng, n_vessels: int = 40, radius_um=(20.0, 100.0), drop: float = 70.0,
            angle: float = 60.0, with_vessels: bool = True, device="cuda"):
    """Homogeneous fibres at ``angle`` plus dark tubes (random 3-D segments through the block). Truth ``dist`` is the
    distance to the nearest tube surface on the mid-y plane (NaN inside a tube); ``region`` 3 inside tubes."""
    sp = Splat(grid, tex, rng, device)
    _add_population(sp, angle, 1.0)
    tr = _blank_truth(grid)
    tr["dir1"][:] = _dir(angle); tr["npop"][:] = 1; tr["frac1"][:] = 1.0
    Zs, Ys, Xs = grid.shape
    extra = None
    if with_vessels:
        zc, yc, xc = grid.coords(0), grid.coords(1), grid.coords(2)
        zz = torch.as_tensor(zc, device=device)[:, None, None]
        yy = torch.as_tensor(yc, device=device)[None, :, None]
        xx = torch.as_tensor(xc, device=device)[None, None, :]
        occ = torch.zeros(grid.shape, device=device)
        Zm, Xm = np.meshgrid(zc, xc, indexing="ij")
        dmin = np.full(Zm.shape, np.inf, np.float32)
        y_mid = yc[grid.y_mid]
        for _ in range(n_vessels):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            p0 = np.array([rng.uniform(zc[0], zc[-1]), rng.uniform(yc[0], yc[-1]), rng.uniform(xc[0], xc[-1])])
            r = rng.uniform(*radius_um) / 1000.0
            # distance from every voxel to the infinite line (segments through the whole block)
            dz, dy, dx = zz - p0[0], yy - p0[1], xx - p0[2]
            t = dz * d[0] + dy * d[1] + dx * d[2]
            dist = torch.sqrt((dz - t * d[0]) ** 2 + (dy - t * d[1]) ** 2 + (dx - t * d[2]) ** 2)
            occ = torch.maximum(occ, ((r + grid.vox / 2 - dist) / grid.vox).clamp(0, 1))
            # mid-y plane truth
            dz2, dx2 = Zm - p0[0], Xm - p0[2]; dy2 = y_mid - p0[1]
            t2 = dz2 * d[0] + dy2 * d[1] + dx2 * d[2]
            dist2 = np.sqrt((dz2 - t2 * d[0]) ** 2 + (dy2 - t2 * d[1]) ** 2 + (dx2 - t2 * d[2]) ** 2) - r
            dmin = np.minimum(dmin, dist2)
        extra = -drop * occ
        tr["dist"][:] = np.where(dmin > 0, dmin, np.nan)
        tr["region"][dmin <= 0] = 3
        tr["dir1"][dmin <= 0] = np.nan
    img = _finish(grid, tex, np.full((Zs, Xs), tex.levels["wm"], np.float32), np.full((Zs, Xs), tex.amp, np.float32),
                  sp.render(), rng, extra=extra)
    return img, tr, {"n_vessels": n_vessels if with_vessels else 0, "radius_um": list(radius_um), "drop": drop}


def laminar(grid: Grid, tex: TextureParams, rng, thickness_mm: float = 1.0, step: float = 15.0, device="cuda"):
    """A slab of fibres along x, ``thickness_mm`` thick at the core's mid-z, in tissue darker by ``step`` with no
    fibre texture. Truth inside: direction x, ``dist`` to the nearer face; outside NaN."""
    zc, xc = grid.coords(0), grid.coords(2)
    Z, X = np.meshgrid(zc, xc, indexing="ij")
    z_mid = grid.core_mm[0] / 2
    inside = np.abs(Z - z_mid) <= thickness_mm / 2
    sp = Splat(grid, tex, rng, device)
    _add_population(sp, 0.0, 1.0, region2d=inside)
    tr = _blank_truth(grid)
    tr["dir1"][inside] = (0.0, 0.0, 1.0); tr["npop"][inside] = 1; tr["frac1"][inside] = 1.0
    tr["dist"][:] = np.abs(thickness_mm / 2 - np.abs(Z - z_mid))
    tr["bnorm"][:] = (1.0, 0.0, 0.0)
    tr["region"][~inside] = 1
    wm = tex.levels["wm"]
    base2d = np.where(inside, wm, wm - step * tex.step_scale).astype(np.float32)
    amp2d = np.where(inside, tex.amp, 0.0).astype(np.float32)
    img = _finish(grid, tex, base2d, amp2d, sp.render(), rng)
    return img, tr, {"thickness_mm": thickness_mm, "step": step}


# ------------------------------------------------------------------------------------------------------------------
# registry, build, save, load
# ------------------------------------------------------------------------------------------------------------------

PHANTOMS = {
    "P1": (gyrus, {"step": True}), "P1t": (gyrus, {"step": False}),
    "P2": (gyrus, {"step": True, "ufibres": True}), "P2t": (gyrus, {"step": False, "ufibres": True}),
    "P3-0": (interface, {"step": 0.0}), "P3-8": (interface, {"step": 8.0}), "P3-15": (interface, {"step": 15.0}),
    "P3-30": (interface, {"step": 30.0}), "P3-40": (interface, {"step": 40.0}),
    "P5": (fan_bundle, {}),
    "P6": (vessels, {"with_vessels": True}), "P6t": (vessels, {"with_vessels": False}),
    "P7": (laminar, {}),
    "P8": (homogeneous, {}),
    "P9": (null, {}),
}
for _c in (90, 60, 45, 30):
    PHANTOMS[f"P4-lines-c{_c}-f1"] = (crossing, {"cross_deg": float(_c), "ratio": 1.0, "kind": "lines"})
for _r in (2, 4):
    PHANTOMS[f"P4-lines-c60-f{_r}"] = (crossing, {"cross_deg": 60.0, "ratio": float(_r), "kind": "lines"})
PHANTOMS["P4-lines-single"] = (crossing, {"cross_deg": 60.0, "single": True, "kind": "lines"})
PHANTOMS["P4-rods-c60-f1"] = (crossing, {"cross_deg": 60.0, "ratio": 1.0, "kind": "rods"})
PHANTOMS["P4-rods-c90-f1"] = (crossing, {"cross_deg": 90.0, "ratio": 1.0, "kind": "rods"})
PHANTOMS["P4-rods-single"] = (crossing, {"cross_deg": 60.0, "single": True, "kind": "rods"})

#: phantoms whose geometry needs the larger (gyrus) core
GYRUS_CORE = (448, 64, 448)


def grid_for(pid: str, vox: float = 0.02, rho_max: float = 32.0) -> Grid:
    pid = split_pid(pid)[0]
    core = GYRUS_CORE if pid.startswith(("P1", "P2", "P5")) else (512, 64, 384)
    return default_grid(vox=vox, rho_max=rho_max, core=core)


def split_pid(pid: str) -> tuple[str, float]:
    """``"P3-40-t0.25"`` -> ``("P3-40", 0.25)``: an optional ``-t<f>`` suffix scales the fibre texture amplitude
    (the step-to-texture ratio is the difficulty axis of the interface phantoms; 1 = the calibrated texture)."""
    if "-t" in pid:
        base, f = pid.rsplit("-t", 1)
        return base, float(f)
    return pid, 1.0


def build(pid: str, seed: int, tex: TextureParams | None = None, grid: Grid | None = None, device="cuda"):
    base, tfrac = split_pid(pid)
    fn, kw = PHANTOMS[base]
    tex = tex or TextureParams()
    if tfrac != 1.0:
        tex = TextureParams(**{**asdict(tex), "amp": tex.amp * tfrac})
    grid = grid or grid_for(base)
    rng = np.random.default_rng([seed, abs(hash(base)) % (2 ** 31)])
    img, tr, meta = fn(grid, tex, rng, device=device, **kw)
    meta = {"pid": pid, "seed": seed, "grid": asdict(grid), "texture": asdict(tex), **meta}
    return img, tr, meta


def path_for(pid: str, seed: int, root: Path = PHANTOM_DIR) -> Path:
    return Path(root) / f"{pid}_s{seed}.npz"


def save(pid: str, seed: int, img, tr, meta, root: Path = PHANTOM_DIR) -> Path:
    p = path_for(pid, seed, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, image=img, meta=json.dumps(meta), **tr)
    return p


def load(pid: str, seed: int, root: Path = PHANTOM_DIR):
    with np.load(path_for(pid, seed, root)) as z:
        img = z["image"]
        meta = json.loads(str(z["meta"]))
        tr = {k: z[k] for k in z.files if k not in ("image", "meta")}
    g = meta["grid"]
    grid = Grid(vox=g["vox"], core=tuple(g["core"]), margin=tuple(g["margin"]))
    return img, tr, meta, grid
