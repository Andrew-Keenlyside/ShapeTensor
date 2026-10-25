"""Compositional populations: a population is a random composition of operators, not a hand-written case.

    population = direction field  x  growth (seeds, length, stop)  x  modulation  x  packing  x  topology
                 x  cross-section  x  relation to the populations drawn before it

* **Direction fields** (:class:`Field`, any point, unit and axial): smooth random fields (random Fourier modes), sources
  and sinks (fanning, convergence), frames tied to a primitive -- a cylinder around a line or a curve (helical families
  at +-alpha, pitch varying with the radius), a stack rotating with depth (laminar: the angle turns along a normal), a
  closed boundary (radial or tangential to a star-shaped blob), a folded surface (parallel offsets or its normal) -- and
  blends of two fields (one region of one, another of the other).
* **Growth**: paths integrated through the field (midpoint rule, axial sign kept continuous) from seeds drawn in a
  region (a box, a disc across a direction, a cylinder shell, a slab, a band along a surface), both ways from each seed,
  stopping where a stop function says (leaving the canvas, reaching a surface: objects ending on sheets).
* **Modulations**: crimp (periodic lateral displacement, coherent across neighbours, any waveform from sine to zig-zag),
  independent smooth wiggle, twist of a bundle's packing about its axis (either hand).
* **Elements**: what the field carries -- lines (fibres, tubes, bundles along grown paths), sheets (the level sets of a
  phase the field defines: parallel planes, concentric cylinders or shells, folded offsets, warped stacks; finite or
  whole, perforated, undulating; each sheet an object) or blobs (ellipsoids, superquadrics or cells, their long axis --
  or a disc's normal -- along the field; scattered, packed, beaded along paths, sitting on a surface, inside a tube).
* **Packing**: independent fibres (each path an object), bundles (fibres packed around grown centrelines, the bundle and
  its fascicles in the tree), tubes (few thick paths with wall and lumen).
* **Topology**: split/merge cycles (fascicles part and rejoin, fibres changing company: a plexus), branching (children
  leaving parents at an angle, then following the field), loops (bridges between nearby ends).
* **Cross-section**: solid, tube (wall, lumen contents), ribbon (flattened), concentric lamellae, beading, striation.
* **Relations**: cross (the same region, another field), wrap (a cylinder field around an earlier population's centreline),
  embed (seeds inside an earlier tube), end on (stop at an earlier population's surface), avoid (yield to obstacles).

Tissue arrangements -- arterial walls (cylinder x +-alpha x crimp), heart walls (laminar rotation), cortex (surface-
following + normal fields), nerves (bundle x split/merge x crimp), pennate muscle (fibres ending on sheets), tumour
collagen (boundary radial / tangential) -- are points of this space, reached by sampling; most samples have no name.
All sizes are in canvas voxels, drawn through the scene's physical scale where one applies.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


def _unit(v: torch.Tensor) -> torch.Tensor:
    return Fn.normalize(v, dim=-1)


def _perp(a: torch.Tensor) -> torch.Tensor:
    """A unit vector perpendicular to each row of ``a (..., 3)``."""
    e = torch.zeros_like(a); e[..., 0] = 1.0
    alt = torch.zeros_like(a); alt[..., 1] = 1.0
    e = torch.where((a * e).sum(-1, keepdim=True).abs() > 0.9, alt, e)
    return _unit(torch.linalg.cross(a, e, dim=-1))


# ================================================================================================================ #
# direction fields
# ================================================================================================================ #

class Field:
    """A direction field: ``dirs(x (N, 3)) -> (N, 3)`` unit vectors (axial: the sign carries no meaning)."""
    name = "field"

    def dirs(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x):
        return self.dirs(x)


class Smooth(Field):
    """``d0`` perturbed by a smooth random vector field (``K`` Fourier modes of wavelength ~``L`` voxels, amplitude ``amp``)."""
    name = "smooth"

    def __init__(self, d0, amp: float, L: float, rng, K: int = 6):
        dev = d0.device
        self.d0 = _unit(d0)
        self.k = _unit(rng.randn(K, 3)) * (2 * math.pi / L) * (0.6 + 0.8 * rng.rand(K)[:, None])
        self.A = rng.randn(K, 3) / math.sqrt(K); self.ph = rng.rand(K) * 2 * math.pi; self.amp = float(amp)

    def dirs(self, x):
        s = torch.sin(x @ self.k.T + self.ph)                                                    # (N, K)
        return _unit(self.d0 + self.amp * s @ self.A)


class Source(Field):
    """Fanning from (``sign`` +1) or converging to (-1) a point ``p``, mixed with a base direction ``d0`` by ``w0``."""
    name = "source"

    def __init__(self, p, d0, w0: float, sign: float = 1.0):
        self.p, self.d0, self.w0, self.sign = p, _unit(d0), float(w0), float(sign)

    def dirs(self, x):
        return _unit(self.w0 * self.d0 + self.sign * _unit(x - self.p))


class Cylinder(Field):
    """Helical directions about an axis (a line ``c + t a``, or a curve ``axis_P (S, 3)``): ``cos(alpha) t + sin(alpha)
    e_theta``, alpha in degrees, varying linearly with the distance from the axis (``dalpha`` per ``r0`` voxels: layer-
    dependent pitch), ``hand`` +-1 the handedness."""
    name = "cylinder"

    def __init__(self, alpha: float, hand: float = 1.0, c=None, a=None, axis_P=None, dalpha: float = 0.0, r0: float = 10.0):
        self.alpha, self.hand, self.dalpha, self.r0 = math.radians(alpha), float(hand), math.radians(dalpha), float(r0)
        self.c, self.a = c, None if a is None else _unit(a)
        if axis_P is not None:                                                                   # the axis, subsampled for the nearest-point search
            step = max(1, axis_P.shape[0] // 64)
            self.P = axis_P[::step].contiguous(); self.T = _unit(torch.gradient(axis_P, dim=0)[0])[::step].contiguous()
        else:
            self.P = None

    def frame(self, x):
        if self.P is None:
            v = x - self.c; t = self.a.expand_as(v); q = self.c + (v @ self.a)[:, None] * self.a
        else:
            d = torch.cdist(x, self.P); j = d.argmin(1); q = self.P[j]; t = self.T[j]
        r = x - q; r = r - (r * t).sum(-1, keepdim=True) * t
        rho = r.norm(dim=-1, keepdim=True)
        er = torch.where(rho > 1e-6, r / rho.clamp_min(1e-6), _perp(t))
        return t, er, rho[:, 0]

    def dirs(self, x):
        t, er, rho = self.frame(x)
        et = torch.linalg.cross(t, er, dim=-1)
        al = (self.alpha + self.dalpha * (rho / self.r0 - 1.0))[:, None]
        return _unit(torch.cos(al) * t + self.hand * torch.sin(al) * et)


class Laminar(Field):
    """In-plane directions rotating with depth along a normal ``n``: theta = theta0 + omega ((x - c) . n) (radians per
    voxel), in the plane spanned by ``u`` and ``n x u``, tilted out of plane by ``tilt`` radians (a helicoid / heart wall)."""
    name = "laminar"

    def __init__(self, c, n, u, theta0: float, omega: float, tilt: float = 0.0):
        self.c, self.n = c, _unit(n)
        u = u - (u @ self.n) * self.n; self.u = _unit(u); self.v = torch.linalg.cross(self.n, self.u)
        self.th0, self.om, self.tilt = float(theta0), float(omega), float(tilt)

    def depth(self, x):
        return (x - self.c) @ self.n

    def dirs(self, x):
        th = self.th0 + self.om * self.depth(x)
        d = torch.cos(th)[:, None] * self.u + torch.sin(th)[:, None] * self.v
        return _unit(math.cos(self.tilt) * d + math.sin(self.tilt) * self.n)


class Boundary(Field):
    """Radial (``mode`` "radial") or tangential ("tangential": a global direction ``g`` projected on the tangent plane)
    to a star-shaped blob around ``c`` (radius ``R0`` (1 + a smooth bump of the direction))."""
    name = "boundary"

    def __init__(self, c, R0: float, mode: str, rng, g=None, bump: float = 0.25):
        self.c, self.R0, self.mode = c, float(R0), mode
        self.g = _unit(g) if g is not None else _unit(rng.randn(3))
        self.k = _unit(rng.randn(4, 3)) * rng.u(1.5, 4.0); self.ph = rng.rand(4) * 6.283; self.bump = float(bump)

    def radius(self, e):
        return self.R0 * (1 + self.bump * torch.sin(e @ self.k.T + self.ph).mean(-1))

    def sdf(self, x):
        v = x - self.c; r = v.norm(dim=-1).clamp_min(1e-6)
        return r - self.radius(v / r[:, None])

    def dirs(self, x):
        er = _unit(x - self.c)
        if self.mode == "radial":
            return er
        t = self.g - (er @ self.g)[:, None] * er
        return torch.where(t.norm(dim=-1, keepdim=True) > 1e-3, _unit(t), _perp(er))


class Surface(Field):
    """A folded surface through ``c`` with normal ``n`` (height ``h(u, v)`` = random folds of amplitude ``amp``, wavelength
    ``L``): ``mode`` "parallel" (a direction ``g`` projected on the local tangent plane: fibres following the surface, its
    offsets alike) or "normal" (the local normal: radial to the surface)."""
    name = "surface"

    def __init__(self, c, n, amp: float, L: float, mode: str, rng, g=None):
        self.c, self.n = c, _unit(n)
        self.u = _perp(self.n); self.v = torch.linalg.cross(self.n, self.u)
        self.k = _unit(rng.randn(3, 2)) * (2 * math.pi / L); self.A = rng.randn(3) * amp / math.sqrt(3); self.ph = rng.rand(3) * 6.283
        self.mode = mode; g = _unit(g) if g is not None else _unit(rng.randn(3)); self.g = _unit(g - (g @ self.n) * self.n)

    def height(self, x):
        w = x - self.c; uv = torch.stack([w @ self.u, w @ self.v], -1)
        return (torch.sin(uv @ self.k.T + self.ph) * self.A).sum(-1), uv

    def sdf(self, x):
        """Signed distance-like height above the surface (along n)."""
        h, _ = self.height(x)
        return (x - self.c) @ self.n - h

    def normal(self, x):
        _, uv = self.height(x)
        dh = (torch.cos(uv @ self.k.T + self.ph) * self.A)[:, :, None] * self.k[None]          # (N, 3 modes, 2)
        g = dh.sum(1)                                                                            # dh/du, dh/dv
        return _unit(self.n - g[:, :1] * self.u - g[:, 1:] * self.v)

    def dirs(self, x):
        nn_ = self.normal(x)
        if self.mode == "normal":
            return nn_
        t = self.g - (nn_ * self.g).sum(-1)[:, None] * nn_
        return torch.where(t.norm(dim=-1, keepdim=True) > 1e-3, _unit(t), _perp(nn_))


class Blend(Field):
    """Two fields, ``f1`` where the smooth weight ``w(x)`` (a ramp across a random plane) is high, ``f2`` elsewhere."""
    name = "blend"

    def __init__(self, f1: Field, f2: Field, c, m, width: float):
        self.f1, self.f2, self.c, self.m, self.w = f1, f2, c, _unit(m), float(width)

    def dirs(self, x):
        a, b = self.f1(x), self.f2(x)
        b = b * torch.sign((a * b).sum(-1, keepdim=True) + 1e-9)                                 # axial: align before mixing
        w = torch.sigmoid(((x - self.c) @ self.m) / self.w)[:, None]
        return _unit(w * a + (1 - w) * b)


# ================================================================================================================ #
# growth, modulations, packing, topology
# ================================================================================================================ #

def grow(field: Field, seeds: torch.Tensor, length: float, h: float, shape, both: bool = True, stop=None, d_init=None, turn: float = 10.0):
    """Paths ``(N, S, 3)`` through ``field`` from ``seeds (N, 3)``: ``length`` voxels in all (half each way with
    ``both``), step ``h`` (midpoint rule, the axial sign kept continuous). A path freezes (repeats its last point: no
    material) once it leaves the canvas by more than 2 voxels or where ``stop(x)`` is true. With ``d_init (N, 3)`` the paths
    leave in that direction and turn into the field over ~``turn`` voxels (a branch leaving its parent at an angle)."""
    dev = seeds.device; N = seeds.shape[0]
    lim = torch.tensor(shape, device=dev, dtype=torch.float32)
    d0 = field(seeds) if d_init is None else _unit(d_init)

    def guided(xq, k, ref):
        f = field(xq); f = f * torch.sign((f * ref).sum(-1, keepdim=True) + 1e-9)
        if d_init is None:
            return f
        w = math.exp(-k * h / max(turn, 1e-3))                                                   # the initial direction fades over ~turn voxels
        return _unit(w * sign_d * d0 + (1 - w) * f)

    def run(sign, n):
        nonlocal sign_d
        sign_d = sign
        x = seeds.clone(); prev = sign * d0; alive = torch.ones(N, dtype=torch.bool, device=dev); out = []
        for k in range(n):
            d = guided(x, k, prev)
            xm = x + 0.5 * h * d
            dm = guided(xm, k + 0.5, d)
            xn = x + h * dm
            ok = alive & ((xn > -2) & (xn < lim + 1)).all(-1)
            if stop is not None:
                ok = ok & ~stop(xn)
            x = torch.where(ok[:, None], xn, x); prev = torch.where(ok[:, None], dm, prev); alive = ok
            out.append(x.clone())
        return torch.stack(out, 1) if out else seeds[:, None]

    sign_d = 1.0
    n = max(2, int(round(length / h / (2 if both else 1))))
    fw = run(1.0, n)
    if not both:
        return torch.cat([seeds[:, None], fw], 1)
    bw = run(-1.0, n)
    return torch.cat([bw.flip(1), seeds[:, None], fw], 1)


def frames(P: torch.Tensor):
    """Tangents and a smooth normal frame along paths ``(N, S, 3)`` (rotation minimising)."""
    from .scene import rmf, tangents
    T = tangents(P); e1, e2 = rmf(P, T)
    return T, e1, e2


def arclength(P: torch.Tensor) -> torch.Tensor:
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    return torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)


def crimp(P: torch.Tensor, period: float, amp: float, sharp: float, phase: torch.Tensor, plane_angle: torch.Tensor | None = None):
    """Periodic lateral displacement: ``amp`` voxels, ``period`` voxels of arc length, waveform from sine (``sharp`` 1) to
    zig-zag-like (``sharp`` < 1), per-path ``phase (N,)`` (equal phases: coherent crimp across neighbours)."""
    T, e1, e2 = frames(P); s = arclength(P)
    w = torch.sin(2 * math.pi * s / period + phase[:, None]); w = torch.sign(w) * w.abs() ** sharp
    pa = plane_angle if plane_angle is not None else torch.zeros(P.shape[0], device=P.device)
    e = torch.cos(pa)[:, None, None] * e1 + torch.sin(pa)[:, None, None] * e2
    return P + amp * w[..., None] * e


def wiggle(P: torch.Tensor, amp: float, corr: float, rng):
    """Independent smooth lateral noise per path (``amp`` voxels, correlation ``corr`` voxels of arc length)."""
    from .scene import smooth1d
    T, e1, e2 = frames(P); N, S = P.shape[:2]
    n = smooth1d(rng.randn(N, S, 2), max(corr, 1.0)); n = n / n.std().clamp_min(1e-6)
    return P + amp * (n[..., :1] * e1 + n[..., 1:] * e2)


def pack(centre: torch.Tensor, Rb: float, rf: float, spacing: float, twist: float, rng, jitter: float = 0.15, nmax: int = 400):
    """Fibres packed on a jittered hexagonal section around a centreline ``(S, 3)``: radius ``Rb``, centre spacing
    ``spacing`` (voxels), the section rotating ``twist`` radians per voxel of arc length. Returns ``(N, S, 3)`` and the
    section offsets ``(N, 2)`` (bundle units)."""
    dev = centre.device; S = centre.shape[0]
    n_side = int(Rb / spacing) + 2
    ii, jj = torch.meshgrid(torch.arange(-n_side, n_side + 1, device=dev), torch.arange(-n_side, n_side + 1, device=dev), indexing="ij")
    o = torch.stack([ii.float() + 0.5 * (jj % 2).float(), jj.float() * math.sqrt(3) / 2], -1).reshape(-1, 2) * spacing
    o = o + jitter * spacing * (rng.rand(*o.shape) - 0.5) * 2
    o = o[o.norm(dim=1) <= max(Rb - rf, 0.0) + 1e-6]
    if o.shape[0] > nmax:
        o = o[torch.randperm(o.shape[0], generator=rng.g, device=dev)[:nmax]]
    if o.shape[0] == 0:
        o = torch.zeros(1, 2, device=dev)
    T, e1, e2 = frames(centre[None])
    s = arclength(centre[None])[0]; ps = twist * s                                              # (S,)
    ox = o[:, :1] * torch.cos(ps)[None] - o[:, 1:] * torch.sin(ps)[None]; oy = o[:, :1] * torch.sin(ps)[None] + o[:, 1:] * torch.cos(ps)[None]
    return centre[None] + ox[..., None] * e1 + oy[..., None] * e2, o


def split_merge(F: torch.Tensor, offsets: torch.Tensor, centre: torch.Tensor, cycles: int, sep: float, rng):
    """Split/merge cycles along a bundle (fibres ``(N, S, 3)`` packed with section ``offsets (N, 2)`` about ``centre``):
    each cycle parts the fibres by a random line across the section, pushes the two sides ``sep`` voxels apart over a
    smooth bump along the bundle, and lets them rejoin -- the next cycle parts them differently (fibres change company).
    Returns the fibres and each fibre's side in the first cycle (for fascicle parents)."""
    dev = F.device; N, S = F.shape[:2]
    T, e1, e2 = frames(centre[None]); s = torch.linspace(0, 1, S, device=dev)
    first = None
    edges = torch.linspace(0.1, 0.9, cycles + 1, device=dev)
    for c in range(cycles):
        a, b = float(edges[c]), float(edges[c + 1]); mid, half = (a + b) / 2, (b - a) / 2
        bump = torch.clamp(1 - ((s - mid) / half) ** 2, min=0) ** 2                              # (S,) smooth, zero at the ends
        th = rng.u(0, math.pi); v = torch.tensor([math.cos(th), math.sin(th)], device=dev)
        side = torch.where(offsets @ v >= 0, 1.0, -1.0)
        if first is None:
            first = side
        dirv = math.cos(th) * e1[0] + math.sin(th) * e2[0]                                       # (S, 3)
        F = F + (sep * side[:, None, None]) * bump[None, :, None] * dirv[None]
    return F, first


def branch(P: torch.Tensor, field: Field, n: int, angle_deg: float, length: float, h: float, shape, rng):
    """``n`` children leaving random parents at random interior points, ``angle_deg`` off the parent, then following the
    field one way. Returns child paths ``(n, S, 3)`` and their parents' indices."""
    dev = P.device; N, S = P.shape[:2]
    pi = torch.randint(0, N, (n,), generator=rng.g, device=dev); si = torch.randint(S // 6, max(S // 6 + 1, 5 * S // 6), (n,), generator=rng.g, device=dev)
    x0 = P[pi, si]; T, _, _ = frames(P); t0 = T[pi, si]
    q = _perp(t0); ang = math.radians(angle_deg)
    d0 = _unit(math.cos(ang) * t0 + math.sin(ang) * q)
    return grow(field, x0, length, h, shape, both=False, d_init=d0, turn=rng.u(5.0, 20.0)), pi


def loops(P: torch.Tensor, radius: float, S_out: int):
    """Bridges between path ends closer than ``radius`` voxels (cubic Hermite curves leaving and entering along the paths'
    end tangents): ``(M, S_out, 3)``."""
    dev = P.device
    ends = torch.cat([P[:, -1], P[:, 0]]); tans = torch.cat([P[:, -1] - P[:, -2], P[:, 0] - P[:, 1]])
    tans = _unit(tans); N2 = ends.shape[0]
    if N2 < 2:
        return torch.zeros(0, S_out, 3, device=dev)
    d = torch.cdist(ends, ends); d.fill_diagonal_(float("inf")); n = P.shape[0]
    idx = torch.arange(N2, device=dev); d[idx, (idx + n) % N2] = float("inf")                   # not a path to itself
    j = d.argmin(1); ok = (d[idx, j] < radius) & (idx < j)
    a, b = idx[ok], j[ok]
    if a.numel() == 0:
        return torch.zeros(0, S_out, 3, device=dev)
    t = torch.linspace(0, 1, S_out, device=dev)[None, :, None]
    p0, p1 = ends[a][:, None], ends[b][:, None]; L = (p1 - p0).norm(dim=-1, keepdim=True)
    m0, m1 = tans[a][:, None] * L, -tans[b][:, None] * L
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def seeds_in(region: str, n: int, shape, rng, c=None, d=None, R: float = 10.0, width: float = 6.0, surface=None):
    """``n`` seed points: "box" (the canvas's central part), "disc" (across direction ``d`` at ``c``, radius ``R``),
    "shell" (a cylinder surface of radius ``R`` about the axis through ``c`` along ``d``), "slab" (|(x - c) . d| < width),
    "band" (within ``width`` of a :class:`Surface` / :class:`Boundary` ``surface``)."""
    dev = rng.device
    S = torch.tensor(shape, device=dev, dtype=torch.float32)
    if region == "disc":
        q = _perp(d[None])[0]; w = torch.linalg.cross(d, q)
        rr = R * torch.sqrt(rng.rand(n)); th = rng.rand(n) * 2 * math.pi
        return c + rr[:, None] * (torch.cos(th)[:, None] * q + torch.sin(th)[:, None] * w)
    if region == "shell":
        q = _perp(d[None])[0]; w = torch.linalg.cross(d, q)
        th = rng.rand(n) * 2 * math.pi; z = (rng.rand(n) - 0.5) * float(S.max())
        return c + z[:, None] * d + R * (torch.cos(th)[:, None] * q + torch.sin(th)[:, None] * w)
    x = (0.1 + 0.8 * rng.rand(n, 3)) * S
    if region == "around" and surface is not None:                                               # near a curve (S, 3): within R of it
        j = torch.randint(0, surface.shape[0], (n,), generator=rng.g, device=dev)
        T = _unit(torch.gradient(surface, dim=0)[0])[j]; q = _perp(T); w = torch.linalg.cross(T, q)
        th = rng.rand(n) * 2 * math.pi; rr = R * (0.5 + 0.5 * rng.rand(n))
        return surface[j] + rr[:, None] * (torch.cos(th)[:, None] * q + torch.sin(th)[:, None] * w)
    if region == "slab":
        dd = (x - c) @ d; x = x - (dd - (rng.rand(n) - 0.5) * 2 * width)[:, None] * d
    elif region == "band" and surface is not None:
        for _ in range(3):                                                                       # pull the points onto the band
            sd = surface.sdf(x); g = surface.normal(x) if hasattr(surface, "normal") else _unit(x - surface.c)
            x = x - (sd - (rng.rand(n) - 0.5) * 2 * width)[:, None] * g
    return x


# ================================================================================================================ #
# the population composer
# ================================================================================================================ #

FIELD_P = {"smooth": 0.26, "source": 0.12, "cylinder": 0.16, "laminar": 0.14, "boundary": 0.12, "surface": 0.12, "blend": 0.08}
ARRANGE_P = {"fibres": 0.42, "bundle": 0.38, "tubes": 0.20}
ELEMENT_P = {"line": 0.55, "sheet": 0.20, "blob": 0.25}
BLOB_PLACE_P = {"scatter": 0.35, "packed": 0.25, "beads": 0.20, "on_surface": 0.20}
BLOB_SHAPE_P = {"ellipsoid": 0.45, "superquadric": 0.30, "cell": 0.25}
RELATION_P = {"none": 0.40, "cross": 0.22, "wrap": 0.14, "embed": 0.08, "end_on": 0.08, "avoid": 0.08}


def _pick(rng, table: dict) -> str:
    ks = list(table); w = torch.tensor([table[k] for k in ks]); w = w / w.sum()
    return ks[int(torch.multinomial(w, 1, generator=torch.Generator().manual_seed(int(rng.i(0, 2 ** 30)))))]


def sample_field(sc, rng, kind: str | None = None, around=None) -> Field:
    """A random field of ``kind`` (drawn from :data:`FIELD_P` if None); ``around`` an earlier population's centreline
    ``(S, 3)`` for a cylinder field wrapping it."""
    dev = sc.device; Z = float(max(sc.shape)); c = sc.centre_point(0.25)
    kind = kind or _pick(rng, FIELD_P)
    if kind == "smooth":
        return Smooth(rng.unit(), rng.lu(0.05, 1.0), rng.lu(0.3, 2.0) * Z, rng)
    if kind == "source":
        p = c - rng.unit() * rng.u(0.3, 1.0) * Z
        return Source(p, rng.unit(), rng.u(0.0, 2.0), 1.0 if rng.p(0.7) else -1.0)
    if kind == "cylinder":
        al = rng.u(5, 85) if rng.p(0.8) else rng.u(-5, 5)
        if around is not None:
            return Cylinder(al, 1.0 if rng.p(0.5) else -1.0, axis_P=around, dalpha=rng.u(-30, 30) if rng.p(0.4) else 0.0, r0=rng.u(3, 15))
        return Cylinder(al, 1.0 if rng.p(0.5) else -1.0, c=c, a=rng.unit(), dalpha=rng.u(-30, 30) if rng.p(0.4) else 0.0, r0=rng.u(0.05, 0.3) * Z)
    if kind == "laminar":
        rate = math.radians(rng.u(30, 180)) / (rng.u(0.2, 1.0) * Z)                              # turns 30-180 deg across 20-100 % of the canvas
        return Laminar(c, rng.unit(), rng.unit(), rng.u(0, math.pi), rate * (1 if rng.p(0.5) else -1), tilt=rng.u(-0.3, 0.3))
    if kind == "boundary":
        return Boundary(c, rng.u(0.1, 0.35) * Z, "radial" if rng.p(0.5) else "tangential", rng, bump=rng.u(0.0, 0.4))
    if kind == "surface":
        return Surface(c, rng.unit(), rng.u(0.0, 0.15) * Z, rng.u(0.3, 1.5) * Z, "parallel" if rng.p(0.6) else "normal", rng)
    a = sample_field(sc, rng, _pick(rng, {k: v for k, v in FIELD_P.items() if k != "blend"}))
    b = sample_field(sc, rng, _pick(rng, {k: v for k, v in FIELD_P.items() if k != "blend"}))
    return Blend(a, b, c, rng.unit(), rng.u(0.03, 0.2) * Z)


def population(sc, prev: list | None = None, dense: bool = False):
    """One compositional population on scene ``sc`` (its rng, sizes, ids, parents): returns a list of
    :class:`~shapetensor.learn.scene.Group` (one; two for a bundle split by a relation) and records the recipe in
    ``sc.record["compose"]``."""
    from .scene import H, Group, curves_like
    r = sc.r; dev = sc.device; Z = float(max(sc.shape)); shape = sc.shape
    prev = prev or []
    rel = _pick(r, RELATION_P) if prev else "none"
    base = prev[r.i(0, len(prev) - 1)] if prev else None
    around = getattr(base, "centre_P", None) if base is not None else None
    if rel == "wrap" and around is None:
        rel = "cross"
    if rel in ("wrap", "embed") and around is None:
        rel = "cross"
    if rel == "end_on" and getattr(base, "sdf", None) is None:
        rel = "cross"
    fkind = "cylinder" if rel == "wrap" else None
    field = sample_field(sc, r, fkind, around=around if rel == "wrap" else None)
    element = _pick(r, ELEMENT_P)
    if element == "sheet":
        return sheet_population(sc, field, rel, base, dense)
    if element == "blob":
        return blob_population(sc, field, rel, base, around, dense)
    arr = _pick(r, ARRANGE_P)
    recipe = {"element": "line", "field": field.name, "arrange": arr, "relation": rel}
    stop = None
    if rel == "end_on" and base is not None and getattr(base, "sdf", None) is not None:
        sdf = base.sdf; sgn = 1.0 if r.p(0.5) else -1.0
        stop = lambda x: sgn * sdf(x) < 0
    Lp = r.u(0.6, 2.2) * Z                                                                       # path length (voxels)
    contrast = sc.contrast()
    out_groups = []
    if arr == "fibres" or arr == "tubes":
        tubes = arr == "tubes"
        rf = sc.size(0.03, 400, 0.6 if tubes else 0.3, Z / (10 if tubes else 25)) or (r.u(1.0, 4.0) if tubes else r.u(0.4, 1.5))
        n = r.i(2, 12) if tubes else int(min(400, max(4, r.lu(8, 220) * (2.5 if dense else 1.0))))
        vol = Lp * math.pi * rf * rf                                                              # at most ~35 % of the canvas in lines: an
        n = int(min(n, max(2, 0.35 * math.prod(shape) / max(vol, 1.0))))                          # overpacked draw is occluded anyway, and slow
        region = {"cylinder": "shell", "laminar": "slab", "surface": "band", "boundary": "band"}.get(field.name, "box")
        if rel == "embed" and around is not None:
            region = "disc"
        if rel == "wrap" and around is not None:
            region = "around"
        kw = {}
        if region == "around":
            kw = {"surface": around, "R": getattr(field, "r0", 6.0)}
        elif region in ("shell", "disc"):
            cc = getattr(field, "c", None)
            if cc is None or region == "disc":
                cc = around[around.shape[0] // 2] if around is not None else sc.centre_point()
            d = getattr(field, "a", None) if getattr(field, "a", None) is not None else r.unit()
            kw = {"c": cc, "d": d, "R": getattr(field, "r0", rf * 4) if region == "shell" else rf * 3}
        elif region == "slab":
            kw = {"c": field.c, "d": field.n, "width": r.u(0.05, 0.3) * Z}
        elif region == "band":
            kw = {"surface": field, "width": r.u(0.02, 0.2) * Z}
        seeds = seeds_in(region, n, shape, r, **kw)
        P = grow(field, seeds, Lp, H, shape, stop=stop)
        if tubes and r.p(0.35):                                                                  # branching
            Ch, par = branch(P, field, r.i(1, 6), r.u(20, 75), Lp * r.u(0.2, 0.6), H, shape, r)
            S = P.shape[1]; Ch = _resample(Ch, S); P = torch.cat([P, Ch]); recipe["branch"] = int(Ch.shape[0])
        if tubes and r.p(0.25):                                                                  # loops
            Lo = loops(P, r.u(0.05, 0.2) * Z, P.shape[1])
            if Lo.shape[0]:
                P = torch.cat([P, Lo]); recipe["loops"] = int(Lo.shape[0])
        P, rec_m = _modulate(P, r, rf)
        recipe.update(rec_m)
        N, S = P.shape[:2]
        rad = torch.full((N, S), float(rf), device=dev) * torch.exp(0.15 * r.randn(N, 1)).clamp(0.7, 1.4)
        rad, con, kw2 = sc._fibre_extras(N, S, rad, r)
        if tubes:
            kw2["wall"] = rad * r.u(0.15, 0.4); kw2["lumen_c"] = torch.full_like(rad, r.u(-40, 40))
        if r.p(0.2):
            kw2["flat"] = torch.full_like(rad, r.u(0.25, 0.8))                                    # ribbons
        cmap = contrast * (con if con is not None else 1.0) * torch.ones(N, S, device=dev)
        pid = int(sc.ids(1, Z)[0])
        obj = sc.ids(N, rad.amin(1))
        for o_ in obj.tolist():
            sc.parents[o_] = pid
        cv = curves_like(P, rad, r, obj=obj, seg=torch.full((N,), pid, device=dev), contrast=cmap, **kw2)
        if rel == "avoid":
            cv = sc.deflect(cv)
        g = Group("compose", "line", float(rf), contrast, curves=[cv])
        g.centre_P = P[P.shape[0] // 2]; g.sdf = getattr(field, "sdf", None); g.centre_pt = seeds.mean(0)
        out_groups.append(g)
    else:                                                                                        # bundles: packed fibres around grown centrelines
        nb = r.i(1, 4 if dense else 2)
        Rb = sc.size(1.0, 3000, 2.0, Z / 6) or r.u(3.0, Z / 8)
        rf = sc.size(0.05, 30, 0.3, Rb / 4) or r.u(0.4, max(0.5, Rb / 4))                       # several fibres per bundle
        c0 = seeds_in("box", nb, shape, r) if around is None or rel != "embed" else around[[around.shape[0] // 2]].expand(nb, 3).clone()
        Cl = grow(field, c0, Lp, H, shape, stop=stop)
        for b in range(nb):
            twist = math.radians(r.u(5, 60)) / (r.u(0.3, 1.5) * Z) * (1 if r.p(0.5) else -1) if r.p(0.3) else 0.0
            sp = max(2 * rf * 1.15 * r.u(1.05, 1.5), 0.8)
            F, off = pack(Cl[b], Rb, rf, sp, twist, r)
            parents_side = None
            if r.p(0.3):                                                                         # split / merge: a plexus
                F, parents_side = split_merge(F, off, Cl[b], r.i(1, 4), r.u(0.6, 1.6) * Rb, r)
                recipe["split_merge"] = True
            if twist:
                recipe["twist"] = True
            F, rec_m = _modulate(F, r, rf, coherent=True)
            recipe.update(rec_m)
            N, S = F.shape[:2]
            rad = (rf * torch.exp(0.12 * r.randn(N)).clamp(0.8, 1.15))[:, None].expand(N, S).clone()
            rad, con, kw2 = sc._fibre_extras(N, S, rad, r)
            cmap = contrast * (con if con is not None else 1.0) * torch.ones(N, S, device=dev)
            bid = int(sc.ids(1, Rb)[0]); obj = sc.ids(N, rad.amin(1))
            if parents_side is not None and getattr(sc, "deep", False):                          # fascicles: the first split's sides
                fid = sc.ids(2, Rb / math.sqrt(2))
                for o_, sd_ in zip(obj.tolist(), parents_side.tolist()):
                    sc.parents[o_] = int(fid[0] if sd_ > 0 else fid[1])
                for f_ in fid.tolist():
                    sc.parents[f_] = bid
            else:
                for o_ in obj.tolist():
                    sc.parents[o_] = bid
            cv = curves_like(F, rad, r, obj=obj, seg=torch.full((N,), bid, device=dev), contrast=cmap, **kw2)
            if rel == "avoid":
                cv = sc.deflect(cv)
            g = Group("compose", "line", float(rf), contrast, curves=[cv])
            g.centre_P = Cl[b]; g.sdf = getattr(field, "sdf", None); g.centre_pt = Cl[b][Cl.shape[1] // 2]; g.Rb = Rb; g.offsets = off
            out_groups.append(g)
    sc.record.setdefault("compose", []).append(recipe)
    return out_groups


def _grid_eval(fn, shape, dev, chunk: int = 131072) -> torch.Tensor:
    """``fn (N, 3) -> (N, ...)`` over every voxel centre of ``shape``, in chunks: ``(*shape, ...)``."""
    Z, Y, X = shape
    out = []
    for z0 in range(0, Z, max(1, chunk // (Y * X))):
        z1 = min(Z, z0 + max(1, chunk // (Y * X)))
        g = torch.stack(torch.meshgrid(torch.arange(z0, z1, device=dev, dtype=torch.float32), torch.arange(Y, device=dev, dtype=torch.float32),
                                       torch.arange(X, device=dev, dtype=torch.float32), indexing="ij"), -1).reshape(-1, 3)
        out.append(fn(g))
    v = torch.cat(out)
    return v.reshape(*shape, *v.shape[1:])


def _smooth_noise(shape, scale: float, rng) -> torch.Tensor:
    """Smooth noise of unit sd with features ~``scale`` voxels on ``shape`` (drawn coarse, interpolated)."""
    f = max(1, int(scale // 3))
    small = tuple(max(2, n // f) for n in shape)
    from ..filters import gaussian
    x = gaussian(rng.randn(1, *small), max(0.5, scale / f / 2), truncate=3.0)
    x = Fn.interpolate(x[None], size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def phase_of(field: Field, rng, Z: float):
    """A scalar whose level sets are sheets the field defines, ``phi (N, 3) -> (N,)`` (distance-like), and its name:
    a laminar stack's depth, a cylinder's radius, a surface's or a boundary's signed height, a source's distance; a smooth
    or blended field a warped stack -- containing its mean direction (lamellae the fibres lie in) or across it (plates)."""
    if isinstance(field, Laminar):
        return field.depth, "planes"
    if isinstance(field, Cylinder):
        return (lambda x: field.frame(x)[2]), "cylinders"
    if isinstance(field, (Surface, Boundary)):
        return field.sdf, ("folds" if isinstance(field, Surface) else "shells")
    if isinstance(field, Source):
        return (lambda x: (x - field.p).norm(dim=-1)), "spheres"
    d0 = field.d0 if isinstance(field, Smooth) else _unit(rng.randn(3))
    across = rng.p(0.4)
    m = d0 if across else _perp(d0[None])[0]
    m = _unit(m + (0.0 if across else 0.3) * torch.linalg.cross(d0, m) * rng.u(-1, 1))
    k = _unit(rng.randn(3, 3)) * (2 * math.pi / (rng.u(0.3, 1.5) * Z)); A = rng.randn(3) * rng.u(0.0, 0.08) * Z; ph = rng.rand(3) * 6.283
    c = torch.zeros(3, device=d0.device) + Z / 2
    return (lambda x: (x - c) @ m + (torch.sin(x @ k.T + ph) * A).sum(-1)), ("plates" if across else "lamellae")


def sheet_population(sc, field: Field, rel: str, base, dense: bool = False):
    """Sheets carried by ``field``: the level sets ``phi = phi0 + (k + 1/2) T`` of :func:`phase_of`, thickness ``t``
    (antialiased), ``n`` of them or all that cross the canvas, possibly finite (a smooth lateral mask), perforated
    (holes of a smooth noise's low quantile) or undulating (phi warped); normals from phi's gradient; each sheet an
    object under the population."""
    from .scene import Group
    from .generator import sheet_tensor
    r = sc.r; dev = sc.device; shape = sc.shape; Z = float(max(shape))
    phi_fn, pname = phase_of(field, r, Z)
    T = sc.size(0.01, 3000, 2.5, Z / 3) or r.u(3.0, Z / 4)                                      # spacing
    t = min(T * r.u(0.15, 0.6), max(0.6, T - 1.5))                                              # thickness, below the spacing
    phi = _grid_eval(phi_fn, shape, dev)
    rec = {"element": "sheet", "field": field.name, "relation": rel, "phase": pname, "spacing": round(T, 1), "thick": round(t, 2)}
    if r.p(0.35):                                                                                # undulating sheets
        phi = phi + _smooth_noise(shape, r.u(0.15, 0.6) * Z, r) * r.u(0.15, 0.6) * T; rec["undulate"] = True
    gz, gy, gx = torch.gradient(phi)
    gr = torch.stack([gz, gy, gx], -1); gn = gr.norm(dim=-1)
    nrm = gr / gn.clamp_min(1e-6)[..., None]
    c = torch.tensor([n / 2 for n in shape], device=dev)
    phi0 = float(phi_fn(c[None])[0]) + r.u(0, T)
    u = (phi - phi0) / T; k = torch.floor(u); dist = (u - k - 0.5).abs() * T / gn.clamp_min(0.2)
    tf = t * (1 + r.u(0.2, 0.5) * _smooth_noise(shape, r.u(0.2, 0.6) * Z, r)).clamp(0.4, 1.6) if r.p(0.2) else torch.full_like(phi, t)   # thickness along the sheet
    occ = ((tf / 2 - dist) + 0.5).clamp(0, 1) * (gn > 0.2).float()
    wall = None
    if r.p(0.25) and t >= 2.0:                                                                   # hollow: two walls (a membrane pair, a lumen between)
        wall = max(0.5, t * r.u(0.15, 0.35))
        occ = occ * (((dist - (tf / 2 - wall)) + 0.5).clamp(0, 1)); rec["walls"] = round(wall, 2)
    if not r.p(0.4):                                                                             # a finite stack: n sheets about the middle
        n_s = r.i(1, 8 if not dense else 16); kc = int(r.i(-2, 2))
        keep = (k >= kc - n_s // 2) & (k < kc - n_s // 2 + n_s); occ = occ * keep.float(); rec["n"] = n_s
    if r.p(0.35):                                                                                # finite in extent: a smooth patch
        m = _smooth_noise(shape, r.u(0.15, 0.5) * Z, r); occ = occ * ((m - r.u(-0.8, 0.3)) / 0.3).clamp(0, 1); rec["patch"] = True
    if r.p(0.3):                                                                                 # perforated
        h = _smooth_noise(shape, r.u(0.6, 2.5) * max(T, 3.0), r); q = r.u(0.1, 0.4)
        thr = float(torch.quantile(h.flatten()[::97], q)); occ = occ * ((h - thr) / 0.25).clamp(0, 1); rec["perforated"] = round(q, 2)
    occ = occ * r.u(0.6, 1.0)
    if float(occ.sum()) < 1.0:
        return []
    kmin = int(k[occ > 0].min()); kmax = int(k[occ > 0].max()); ns = kmax - kmin + 1
    pid = int(sc.ids(1, Z)[0]); oid = sc.ids(ns, t / 2)
    for o_ in oid.tolist():
        sc.parents[o_] = pid
    obj_map = oid[(k - kmin).clamp(0, ns - 1).long()]
    tens = sheet_tensor(nrm).movedim(-1, 0).contiguous()
    desc = torch.zeros((5,) + tuple(shape), device=dev); desc[0] = tf / 2
    if wall is not None:
        desc[1] = wall
    contrast = sc.contrast(); con = contrast
    if r.p(0.3):                                                                                 # patchy stain along the sheets
        con = contrast * (1 + r.u(0.2, 0.6) * _smooth_noise(shape, r.u(0.1, 0.4) * Z, r)).clamp(0.2, 1.8); rec["patchy"] = True
    g = Group("compose", "sheet", float(t / 2), contrast, dense=[(occ, tens, desc, con, obj_map)])
    g.centre_P = None; g.sdf = (lambda x, f=phi_fn, p0=phi0 + 0.5 * T: f(x) - p0); g.centre_pt = c
    sc.record.setdefault("compose", []).append(rec)
    return [g]


def blob_population(sc, field: Field, rel: str, base, around, dense: bool = False):
    """Blobs carried by ``field``: semi-axes ``rc (1, lat, e)``, the third along the field at the centre (``e`` > 1
    spindles, < 1 discs across it), placed by scattering, packing (near-touching in a ball), beading along grown
    paths, on a surface's band, or inside an earlier tube (``embed``); greedy minimum spacing; filled ellipsoids,
    superquadrics or cells (membrane and nucleus)."""
    from .scene import H, _solids_budget
    r = sc.r; dev = sc.device; shape = sc.shape; Z = float(max(shape))
    rc = sc.size(0.2, 2000, 0.8, Z / 10) or r.u(1.0, 4.0)
    e = r.lu(0.25, 5.0); lat = r.u(0.6, 1.0)
    e = min(e, max(1.0, Z / 6 / rc))                                                             # objects, not regions: the long axis <= canvas / 6
    ax_ = torch.tensor([rc, rc * lat, rc * e], device=dev); amax = float(ax_.max())
    place = _pick(r, BLOB_PLACE_P); shp = _pick(r, BLOB_SHAPE_P)
    if rel == "embed" and around is not None:
        place = "embed"
    elif rel == "end_on" and getattr(base, "sdf", None) is not None:
        place = "on_surface"
    n = int(min(_solids_budget(amax, shp), max(3, r.lu(6, 300) * (2.0 if dense else 1.0))))
    rec = {"element": "blob", "field": field.name, "relation": rel, "place": place, "shape": shp, "elong": round(e, 2), "n": n}
    chains = None
    if place == "packed":
        cc = sc.centre_point(0.3); Rb = r.u(0.15, 0.45) * Z
        X = cc + _unit(r.randn(6 * n, 3)) * (Rb * r.rand(6 * n) ** (1 / 3))[:, None]; f = r.u(0.85, 1.2)
    elif place == "beads":
        nk = r.i(1, 6); P = grow(field, seeds_in("box", nk, shape, r), r.u(0.6, 2.0) * Z, H, shape)
        step = 2 * rc * max(1.0, e) * r.u(1.05, 1.8); S = P.shape[1]
        al = arclength(P); X, chains = [], []
        for j in range(P.shape[0]):
            ts = torch.arange(0.0, float(al[j, -1]), step, device=dev)
            if ts.numel():
                idx = torch.searchsorted(al[j].contiguous(), ts).clamp(max=S - 1); X.append(P[j, idx]); chains.append(torch.full((idx.numel(),), j, device=dev))
        if not X:
            return []
        X = torch.cat(X); chains = torch.cat(chains); f = 0.9
    elif place == "on_surface":
        surf = base if (rel == "end_on" and getattr(base, "sdf", None) is not None) else (field if hasattr(field, "sdf") else None)
        if surf is None:
            surf = Surface(sc.centre_point(0.3), r.unit(), r.u(0.0, 0.12) * Z, r.u(0.3, 1.5) * Z, "normal", r); rec["surface"] = "new"
        X = seeds_in("box", 6 * n, shape, r)
        for _ in range(4):                                                                       # onto the band: x - (sdf - offset) grad
            sd = surf.sdf(X)
            g = surf.normal(X) if hasattr(surf, "normal") else (_unit(X - surf.c) if hasattr(surf, "c") else None)
            if g is None:
                e_ = 1.0; gg = torch.zeros_like(X)
                for a_ in range(3):
                    dx = torch.zeros(3, device=dev); dx[a_] = e_; gg[:, a_] = (surf.sdf(X + dx) - surf.sdf(X - dx)) / (2 * e_)
                g = _unit(gg)
            X = X - (sd - amax * r.u(0.8, 1.2))[:, None] * g
        f = r.u(1.0, 1.6)
    elif place == "embed":
        Rr = float(getattr(base, "Rb", getattr(base, "radius", 4.0))) * 0.7
        X = seeds_in("around", 6 * n, shape, r, surface=around, R=max(Rr, 1.0)); f = r.u(1.0, 1.8)
    else:
        X = seeds_in("box", 4 * n, shape, r); f = r.u(1.2, 3.0)
    S_ = torch.tensor(shape, device=dev, dtype=torch.float32)
    ok = ((X > -amax) & (X < S_ + amax)).all(1); X = X[ok]; chains = chains[ok] if chains is not None else None
    dmin = 2 * amax * f if place != "beads" else 0.0
    keep = []
    for i in range(X.shape[0]):                                                                  # greedy minimum spacing
        if len(keep) >= n:
            break
        if not keep or dmin <= 0 or float((X[keep] - X[i]).norm(dim=1).min()) >= dmin:
            keep.append(i)
    if not keep:
        return []
    C = X[keep]; m = C.shape[0]; rec["n"] = m
    d = field(C); q = _perp(d); w = torch.linalg.cross(d, q); th = r.rand(m) * 6.283
    q2 = torch.cos(th)[:, None] * q + torch.sin(th)[:, None] * w; w2 = torch.linalg.cross(d, q2)
    Rm = torch.stack([q2, w2, d], -1)                                                            # columns: the semi-axes' directions
    ax = ax_[None].expand(m, 3) * torch.exp(0.12 * r.randn(m, 1)).clamp(0.75, 1.3)
    pid = int(sc.ids(1, Z)[0])
    if shp == "cell":
        gs = sc._cell_groups(C, Rm, ax, rc)
    else:
        oid = sc.ids(m, ax.min(1).values)
        A = Rm @ torch.diag_embed(ax)
        exps = (r.u(1.0, 4.0), r.u(1.0, 4.0)) if shp == "superquadric" else (2.0, 2.0)
        gs = [sc._solids(C, A, "compose", oid, shape="superquadric" if shp == "superquadric" else "ellipsoid", exps=exps)]
        if r.p(0.35):                                                                            # each blob its own stain
            Q_, W_, V_, o_ = gs[0].points[0]; fac = torch.exp(r.u(0.15, 0.5) * r.randn(m))
            V_ = V_.clone(); V_[:, -1] = V_[:, -1] * fac[torch.searchsorted(oid.contiguous(), o_.contiguous())]; gs[0].points[0] = (Q_, W_, V_, o_); rec["stain_var"] = True
        if chains is not None and getattr(sc, "deep", False):                                    # beads: each chain an object between
            cid = sc.ids(int(chains.max()) + 1, rc)                                              # the blobs and the population
            for o_, j in zip(oid.tolist(), chains[keep].long().tolist()):
                sc.parents[o_] = int(cid[j])
            for c_ in cid.tolist():
                sc.parents[c_] = pid
        else:
            for o_ in oid.tolist():
                sc.parents[o_] = pid
        sc.obstacles.append((C, ax.max(1).values))
    for g in gs:
        g.centre_P = None; g.sdf = None; g.centre_pt = C.mean(0)
    sc.record.setdefault("compose", []).append(rec)
    return gs


def _resample(P: torch.Tensor, S: int) -> torch.Tensor:
    if P.shape[1] == S:
        return P
    return Fn.interpolate(P.permute(0, 2, 1), size=S, mode="linear", align_corners=True).permute(0, 2, 1)


def _modulate(P: torch.Tensor, r, rf: float, coherent: bool = False):
    """Random modulations: crimp (coherent phases when ``coherent`` or by chance), wiggle."""
    rec = {}
    if r.p(0.25):
        per = r.lu(6, 40) * max(rf, 0.5); amp = per * math.tan(math.radians(r.u(5, 20))) / (2 * math.pi)   # crimp angle 5-20 deg
        ph = torch.zeros(P.shape[0], device=P.device) + r.u(0, 6.28) if (coherent or r.p(0.6)) else r.rand(P.shape[0]) * 6.28
        P = crimp(P, per, amp, r.u(0.3, 1.0), ph, plane_angle=torch.full((P.shape[0],), r.u(0, math.pi), device=P.device))
        rec["crimp"] = round(per, 1)
    if r.p(0.3):
        P = wiggle(P, r.u(0.1, 0.6) * max(rf, 0.5), r.lu(5, 40), r)
        rec["wiggle"] = True
    return P, rec


def compose_scene(sc, n_pop: int, dense: bool = False) -> list:
    """``n_pop`` compositional populations, each possibly related to the ones before (crossing, wrapping, embedded, ending
    on, avoiding). Returns the groups."""
    groups = []
    for _ in range(n_pop):
        gs = population(sc, groups, dense=dense)
        groups.extend(gs)
    return groups
