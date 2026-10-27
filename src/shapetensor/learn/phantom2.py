"""Composable phantoms for the separation ladder: populations of coherent or dispersed fibres (straight, curved, fanning
bundles), sheets (stacks of planes) and blobs (non-fibrous objects) on a background, each in its own region and
orientation, with a varied look. Each population has an occupancy and a per-voxel orientation tensor (the second moment
of its material's directions: a fibre ``c uu' + (1 - c)(I - uu')/2``, a sheet ``(I - nn')/2``, a blob ``I/3``); the
targets at a query are each population's window mean of occupancy (its mass) and of occupancy-weighted tensor (its
ODF's moment: a fan or a curve broadens with the window, as its directions spread), fitted as a Bingham -- population
specific ODFs, the isotropic remainder the background entity.

Crossings (``cross``, ``fine``) are **Monte Carlo random fibre bundles** (kind ``fibres``, :mod:`~shapetensor.learn.fibres`):
fibres placed by dart throwing with gamma radii, wandering, collision-relaxed jointly so crossing bundles interweave, and
rendered with the scene generator's renderer (the per-voxel tensor is the fibres' own tangents). The earlier crossings --
two lattices interleaved in layers -- stacked into columns along the crossing plane's normal, a third orientation the
targets do not contain (X0, 26 Oct); they remain as ``cross_lattice`` / ``fine_lattice``.

``make(spec, n, seed, device)`` draws a phantom from a list of population specs (dicts: ``kind`` and its parameters);
``random_spec(family, rng)`` draws a spec for a family ("cross", "fan", "curve", "sheet", "blob", "mix", "empty");
``targets`` and ``draw_queries`` as :mod:`~shapetensor.learn.simple_phantom`.
"""
import math

import torch

from . import bingham as BG
from .model import window_mean


def _frame(u: torch.Tensor):
    """Two unit vectors completing ``u`` to an orthonormal frame."""
    ref = torch.tensor([1.0, 0, 0], device=u.device) if abs(float(u[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=u.device)
    e1 = torch.nn.functional.normalize(torch.cross(u, ref, dim=0), dim=0); e2 = torch.cross(u, e1, dim=0)
    return e1, e2


def _t6(T: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(6, ...)`` (xx, yy, zz, xy, xz, yz)."""
    return torch.stack([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2], T[..., 0, 1], T[..., 0, 2], T[..., 1, 2]], 0)


def _fibre_tensor(u: torch.Tensor, c: float) -> torch.Tensor:
    """Per-voxel ``(..., 3, 3)`` moment of fibres along unit ``u (..., 3)`` with concentration ``c`` (1: all along u)."""
    I = torch.eye(3, device=u.device)
    uu = u[..., :, None] * u[..., None, :]
    return c * uu + (1 - c) / 2 * (I - uu)


def _ramp(d: torch.Tensor, r: float) -> torch.Tensor:
    return (r + 0.5 - d).clamp(0, 1)


def render(p: dict, X: torch.Tensor, n: int) -> tuple:
    """One population on the grid ``X (n, n, n, 3)`` (z, y, x): ``(occupancy, tensor6 (6, n, n, n), kind)``."""
    dev = X.device; kind = p["kind"]
    c0 = torch.tensor(p.get("centre", (n / 2,) * 3), device=dev, dtype=torch.float32)
    P = X - c0
    if kind in ("bundle", "curve", "fan"):
        nrm = torch.nn.functional.normalize(torch.tensor(p["normal"], device=dev, dtype=torch.float32), dim=0)   # the bundle's plane normal
        a = torch.nn.functional.normalize(torch.tensor(p["axis"], device=dev, dtype=torch.float32), dim=0)
        a = torch.nn.functional.normalize(a - (a @ nrm) * nrm, dim=0); b = torch.cross(nrm, a, dim=0)   # in-plane: along a, across b
        s, r, half, z0, conc = p["spacing"], p["radius"], p["half"], p.get("z0", 0.0), p.get("conc", 0.94)
        pa, pb, pz = P @ a, P @ b, P @ nrm
        dz = pz - z0 - s * torch.round((pz - z0) / s)
        inz = pz.abs() <= p.get("half_z", half)
        if kind == "bundle":
            db = pb - s * torch.round(pb / s)
            occ = _ramp(torch.sqrt(db ** 2 + dz ** 2), r) * (pb.abs() <= half) * inz * (pa.abs() <= p.get("length", 1e9))
            u = a.expand(*X.shape[:-1], 3)
        elif kind == "curve":                                                                       # arcs about a centre of curvature
            Rc = p["curv_radius"]; cc = -Rc                                                         # the arc passes through the centre
            rho = torch.sqrt(pa ** 2 + (pb - cc) ** 2); phi = torch.atan2(pa, pb - cc)
            dr = rho - Rc; drr = dr - s * torch.round(dr / s)
            occ = _ramp(torch.sqrt(drr ** 2 + dz ** 2), r) * (dr.abs() <= half) * inz * (phi.abs() <= p.get("arc", 1.2))
            t = torch.stack([torch.cos(phi), -torch.sin(phi)], -1)                                  # the tangent in (a, b)
            u = t[..., :1] * a + t[..., 1:] * b
        else:                                                                                        # fan: rays from an apex
            ap = p.get("apex", -n / 2.5); spread = p["spread"]
            ra = pa - ap; rho = torch.sqrt(ra ** 2 + pb ** 2).clamp_min(1e-3); phi = torch.atan2(pb, ra)
            dphi = s / max(p.get("rho_ref", n / 2), 1.0)                                             # rays spaced s apart at rho_ref
            dperp = rho * torch.sin(phi - dphi * torch.round(phi / dphi)).abs()
            occ = _ramp(torch.sqrt(dperp ** 2 + dz ** 2), r) * (phi.abs() <= spread / 2) * inz * (rho >= p.get("rho_min", 6.0))
            u = (ra / rho)[..., None] * a + (pb / rho)[..., None] * b
        T = _fibre_tensor(u, conc)
    elif kind == "sheet":
        nrm = torch.nn.functional.normalize(torch.tensor(p["normal"], device=dev, dtype=torch.float32), dim=0)
        s, th, half = p["spacing"], p["thickness"], p["half"]
        pn = P @ nrm; d = pn - s * torch.round(pn / s)
        e1, e2 = _frame(nrm)
        occ = _ramp(d.abs(), th / 2) * ((P @ e1).abs() <= half) * ((P @ e2).abs() <= half) * (pn.abs() <= p.get("half_n", half))
        cs = p.get("conc", 0.94)                                                                     # in-plane, a little out of it
        T = (cs * (torch.eye(3, device=dev) - torch.outer(nrm, nrm)) / 2 + (1 - cs) * torch.outer(nrm, nrm)).expand(*X.shape[:-1], 3, 3)
    elif kind == "blob":
        g = torch.Generator(device="cpu").manual_seed(int(p.get("seed", 0)))
        half, R, k = p["half"], p["blob_radius"], p["count"]
        cen = (torch.rand(k, 3, generator=g) * 2 - 1) * (half - R)
        occ = torch.zeros(X.shape[:-1], device=dev)
        for cc in cen.to(dev):
            occ = torch.maximum(occ, _ramp((P - cc).norm(dim=-1), R))
        T = (torch.eye(3, device=dev) / 3).expand(*X.shape[:-1], 3, 3)
    else:
        raise ValueError(kind)
    return occ.float(), _t6(T).float(), kind


def _fibre_populations(specs: list, n: int, ss: int, seed: int, device) -> list:
    """Monte Carlo fibre bundles (kind ``fibres``), relaxed jointly and rendered on the ``n * ss`` grid: per spec
    ``(occupancy, tensor6)``. A spec: ``axis``, ``half`` (bundle radius), ``radius`` (mean fibre radius), ``fvf`` (fibre
    volume fraction of the section), ``wander`` (lateral wander, in fibre radii), ``conc`` (the dispersion floor), optional
    ``centre`` and ``k_shape`` (gamma shape of the radii); sizes in coarse voxels. The geometry is drawn and relaxed in
    coarse voxels whatever ``ss`` -- supersampling changes the rendering only -- then scaled to the fine grid, each curve
    resampled ``ss`` times more densely (the renderer's 0.5-voxel step on the fine grid)."""
    if not specs:
        return []
    import numpy as np
    from . import fibres as FB
    from .scene import Canvas, Rng, curves_like, render_curves, smooth1d
    nf = n * ss; rng = np.random.default_rng(seed); R = Rng(seed, device); cur = []
    for i, p in enumerate(specs):
        A = torch.nn.functional.normalize(torch.tensor(p["axis"], dtype=torch.float32, device=device), dim=0); e1, e2 = _frame(A)
        c0 = torch.tensor(p.get("centre", (n / 2,) * 3), dtype=torch.float32, device=device)
        C, Rr = FB.random_section(p["fvf"], p["radius"], p.get("k_shape", 8.0), p["half"], rng, max_n=800)
        C = torch.tensor(C, dtype=torch.float32, device=device); N = C.shape[0]
        L = 1.8 * n; S = int(L / 0.5) + 1; sl = torch.linspace(-L / 2, L / 2, S, device=device)
        w = p.get("wander", 0.5) * p["radius"] * smooth1d(R.randn(N, S, 2), 12.0)                     # lateral wander along each fibre
        off = (C[:, None, :] + w)
        P = c0 + sl[None, :, None] * A + off[..., :1] * e1 + off[..., 1:] * e2
        rad = torch.tensor(Rr, dtype=torch.float32, device=device)[:, None].expand(N, S).contiguous()
        cur.append(curves_like(P, rad, R, obj=torch.arange(N, device=device) + 1000 * i, seg=torch.zeros(N, device=device), contrast=1.0))
    FB.relax(cur, box=((0, 0, 0), (n - 1,) * 3))
    if ss > 1:                                                                                       # to the fine grid: coordinates and radii
        for c in cur:                                                                                # scaled, samples ss times denser
            N, S, _ = c.P.shape; Sf = (S - 1) * ss + 1
            up = lambda x: torch.nn.functional.interpolate(x.movedim(1, -1), size=Sf, mode="linear", align_corners=True).movedim(-1, 1)
            Pf = (up(c.P) + 0.5) * ss - 0.5; rf = up(c.rad[..., None])[..., 0] * ss
            c.P, c.rad = Pf, rf
            for k in ("flat", "wall", "sexp", "contrast", "lumen_c"):
                setattr(c, k, up(getattr(c, k)[..., None])[..., 0])
            c.wall = c.wall * ss                                                                     # a length, like the radius
    out = []
    for p, c in zip(specs, cur):
        cv = Canvas((nf,) * 3, device); render_curves(cv, c); f = cv.finish()
        T6 = f["tensor"]; cs = p.get("conc", 0.94)                                                 # the dispersion floor, as the lattices':
        eye = torch.tensor([1.0, 1, 1, 0, 0, 0], device=device).view(6, 1, 1, 1)                   # c T + (1 - c) (I - T) / 2
        out.append((f["occupancy"].float(), (cs * T6 + (1 - cs) * (eye - T6) / 2).float()))
    return out


def make(spec: list, n: int = 64, seed: int = 0, look: dict | None = None, device="cuda", ss: int = 1) -> dict:
    """A phantom from population specs: ``image (n, n, n)`` 0-1 (8-bit levels), ``occ (K, n, n, n)``, ``tens (K, 6, ...)``,
    ``kinds``, ``bg`` (the background's occupancy). ``look``: ``bg`` level, per-population ``contrast`` (sign: bright or
    dark), ``noise``, ``blur`` (voxels). ``ss`` > 1: rendered on a grid ``ss`` times finer and averaged down -- true
    partial volume (no aliasing of sub-voxel fibres) -- with the fine volumes kept (``occ_f``, ``tens_f``, ``bg_f``) for
    sub-voxel targets."""
    look = {"bg": 0.1, "contrast": [0.8] * len(spec), "noise": 0.01, "blur": 0.0, **(look or {})}
    nf = n * ss
    ax = (torch.arange(nf, dtype=torch.float32, device=device) + 0.5) / ss - 0.5                    # fine cells in the coarse voxels' units
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1)
    occ, tens, kinds = [], [], []
    mc = iter(_fibre_populations([p for p in spec if p["kind"] == "fibres"], n, ss, seed, device))
    for p in spec:
        if p["kind"] == "fibres":
            (o, t), k = next(mc), "bundle"
        else:
            o, t, k = render(p, X, n)
        occ.append(o); tens.append(t); kinds.append(k)
    occ = torch.stack(occ) if occ else torch.zeros(0, nf, nf, nf, device=device)
    tens = torch.stack(tens) if tens else torch.zeros(0, 6, nf, nf, nf, device=device)
    fine = None
    if ss > 1:                                                                                       # the fine volumes, then the coarse ones
        fine = {"occ_f": occ, "tens_f": tens}
        pool = lambda v: torch.nn.functional.avg_pool3d(v.reshape(-1, 1, nf, nf, nf), ss).reshape(*v.shape[:-3], n, n, n)
        wt = pool(occ[:, None] * tens) if len(spec) else tens[:, :, ::ss, ::ss, ::ss]
        occ = pool(occ) if len(spec) else torch.zeros(0, n, n, n, device=device)
        tens = wt / occ[:, None].clamp_min(1e-6) if len(spec) else wt
    tot = occ.sum(0).clamp(max=1.0) if len(spec) else torch.zeros(n, n, n, device=device)
    con = torch.tensor(look["contrast"][:len(spec)], device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    scale = (tot / occ.sum(0).clamp_min(1e-6)) if len(spec) else tot                                # shared voxels: scaled back to one
    img = torch.full((n, n, n), float(look["bg"]), device=device) + ((con * occ).sum(0) * scale if len(spec) else 0)
    if look["blur"] > 0:
        from ..filters import gaussian
        img = gaussian(img[None, None], float(look["blur"]))[0, 0]
    g = torch.Generator(device=device).manual_seed(seed)
    img = (img + look["noise"] * torch.randn(n, n, n, generator=g, device=device)).clamp(0, 1)
    img = (img * 255).round() / 255
    out = {"image": img, "occ": occ, "tens": tens, "kinds": kinds, "bg": 1 - tot, "spec": spec, "n": n, "look": look, "ss": ss}
    if fine is not None:
        fine["bg_f"] = 1 - fine["occ_f"].sum(0).clamp(max=1.0) if len(spec) else torch.ones(nf, nf, nf, device=device)
        out.update(fine)
    return out


def targets(ph: dict, pts: torch.Tensor, sig: torch.Tensor, emax: int = 8) -> dict:
    """Entity targets at ``pts (Q, 3)``, ``sig (Q,)``: each population (and the background) its window mass and the
    Bingham of its window-averaged moment; the crop's tree: root 0, populations 1..K, background K + 1."""
    dev = pts.device; Q = pts.shape[0]; K = ph["occ"].shape[0]; n = ph["n"]
    ss = int(ph.get("ss", 1))
    if ss > 1:                                                                                       # sub-voxel truth: the fine volumes
        P_occ, P_tens, P_bg = ph["occ_f"], ph["tens_f"], ph["bg_f"]
        qp, qs = (pts + 0.5) * ss - 0.5, sig * ss; nn_ = n * ss
    else:
        P_occ, P_tens, P_bg = ph["occ"], ph["tens"], ph["bg"]; qp, qs = pts, sig; nn_ = n
    occ = torch.cat([P_occ, P_bg[None]])                                                             # (K + 1, n, n, n)
    m = window_mean(occ, qp[None].expand(K + 1, -1, -1).contiguous(), qs[None].expand(K + 1, -1).contiguous()).T   # (Q, K + 1)
    T = torch.zeros(Q, K + 1, 3, 3, device=dev); T[:, K] = torch.eye(3, device=dev) / 3
    if K:
        wt = (P_occ[:, None] * P_tens).reshape(K * 6, nn_, nn_, nn_)
        mt = window_mean(wt, qp[None].expand(K * 6, -1, -1).contiguous(), qs[None].expand(K * 6, -1).contiguous()).T.view(Q, K, 6)
        mt = mt / m[:, :K, None].clamp_min(1e-6)
        for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
            T[:, :K, i, j] = mt[..., c]; T[:, :K, j, i] = mt[..., c]
    live = m > 1e-4
    E = min(emax, K + 1)
    node = torch.full((Q, emax), -1, dtype=torch.long, device=dev)
    node[:, :E] = torch.where(live[:, :E], torch.arange(1, E + 1, device=dev), torch.full_like(node[:, :E], -1))
    mass = torch.zeros(Q, emax, device=dev); mass[:, :E] = m[:, :E] * live[:, :E]
    T6 = torch.zeros(Q, emax, 6, device=dev); R = torch.eye(3, device=dev).expand(Q, emax, 3, 3).clone(); Kp = torch.zeros(Q, emax, 2, device=dev)
    Tq = T[:, :E]
    tr = Tq.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-9)
    Tq = Tq / tr[..., None, None]
    T6[:, :E] = _t6(Tq).movedim(0, -1)
    ok = live[:, :E]
    if bool(ok.any()):
        Rf, Kf = BG.fit(Tq[ok].double()); R[:, :E][ok], Kp[:, :E][ok] = Rf.float(), Kf.float()
    ev = torch.linalg.eigvalsh(Tq.double())                                                          # (near) isotropic moments: kappa 0 exactly
    near_iso = (ev - 1.0 / 3).abs().amax(-1) < 0.02                                                  # (a fit there is noise in the window's tails)
    Kp[:, :E] = torch.where(near_iso[..., None], torch.zeros_like(Kp[:, :E]), Kp[:, :E])
    anc = torch.full((1536, 16), -1, dtype=torch.long, device=dev); anc[0, 0] = 0
    for i in range(1, E + 1):
        anc[i, 0], anc[i, 1] = i, 0
    return {"e_points": pts, "e_sigma": sig, "e_node": node, "e_mass": mass, "e_vis": (node >= 0).float(), "e_T6": T6,
            "e_desc": torch.zeros(Q, emax, 5, device=dev), "e_anc": anc, "e_n": torch.tensor(E + 1, device=dev), "e_R": R, "e_kappa": Kp,
            "e_gnode": node, "region_t": torch.cat([ph["occ"], ph["bg"][None]]).sum(0), "iface_t": torch.zeros(n, n, n, device=dev),
            "voxel": torch.tensor(float("nan"), device=dev), "bg_t": ph["bg"], "e_bg": node == K + 1,
            **_split_targets(ph), **_split_masses(ph, m[:, :K])}


def _split_masses(ph: dict, m: torch.Tensor) -> dict:
    """The split targets' window means at the queries, from the populations' masses ``m (Q, K)`` -- the same window
    operator and (supersampled phantoms) the same fine volumes as ``e_mass`` (the coarse maps' means differ by up to
    15 % at sigma 0.5 there: review 2, 27 Oct)."""
    iso = torch.tensor([k in ISO_KINDS for k in ph["kinds"]], dtype=torch.bool, device=m.device)
    return {"ori_m": (m * ~iso).sum(1), "iso_m": (m * iso).sum(1)}


ISO_KINDS = ("blob",)


def _split_targets(ph: dict) -> dict:
    """The objects' occupancy by kind at every voxel (densities: crossings count both): ``ori_t`` the oriented
    populations' (bundles, fans, curves, sheets), ``iso_t`` the isotropic objects' (blobs)."""
    occ = ph["occ"]; z = torch.zeros(occ.shape[1:], device=occ.device)
    iso = [k in ISO_KINDS for k in ph["kinds"]]
    ori_t = sum((occ[k] for k in range(occ.shape[0]) if not iso[k]), z.clone())
    iso_t = sum((occ[k] for k in range(occ.shape[0]) if iso[k]), z.clone())
    return {"ori_t": ori_t, "iso_t": iso_t}


def draw_queries(ph: dict, g: torch.Generator, n_q: int, sigma=(1.0, 6.0)) -> tuple:
    """Half on the material, a quarter where two populations overlap (if any), the rest anywhere (2 voxels from the
    faces); sigma log-uniform in ``sigma`` (below 1: sub-voxel windows)."""
    n = ph["n"]; dev = ph["image"].device
    pts = 2 + torch.rand(n_q, 3, generator=g, device=dev) * (n - 5)
    tot = ph["occ"].sum(0) if ph["occ"].shape[0] else torch.zeros(n, n, n, device=dev)
    two = (ph["occ"] > 0.05).sum(0) >= 2 if ph["occ"].shape[0] >= 2 else torch.zeros(n, n, n, dtype=torch.bool, device=dev)
    for frac, off, sel in ((0.5, 0.0, tot > 0.05), (0.25, 0.5, two)):
        sel = sel.clone(); sel[:2] = sel[-2:] = False; sel[:, :2] = sel[:, -2:] = False; sel[:, :, :2] = sel[:, :, -2:] = False
        cand = torch.nonzero(sel).float(); k = int(frac * n_q); o = int(off * n_q)
        if cand.shape[0]:
            pts[o:o + k] = (cand[torch.randint(0, cand.shape[0], (k,), generator=g, device=dev)] + torch.rand(k, 3, generator=g, device=dev) - 0.5).clamp(2, n - 3)
    lo, hi = sigma
    sig = torch.exp(math.log(lo) + torch.rand(n_q, generator=g, device=dev) * math.log(hi / lo))
    return pts, sig


# ---- families ------------------------------------------------------------------------------------------------------

def _rand_unit(g):
    return torch.nn.functional.normalize(torch.randn(3, generator=g), dim=0)


def random_spec(family: str, g: torch.Generator, n: int = 64) -> tuple:
    """A random spec and look for ``family``. Populations crossing share a plane normal (their fibres interleave in z)."""
    u = lambda lo, hi: lo + float(torch.rand(1, generator=g)) * (hi - lo)
    nrm = _rand_unit(g); e1, e2 = _frame(nrm)
    def inplane(theta):
        return (math.cos(theta) * e1 + math.sin(theta) * e2).tolist()
    # the within-voxel dispersion is fixed: it only enters the targets' tensor (the image does not show it), so a drawn
    # value was unobservable (review 2026-10-27); fans and curves still broaden with the window, as their directions spread
    s0 = u(3.0, 5.5); fib = lambda: {"spacing": u(3.0, 5.5), "radius": u(0.8, 1.5), "half": u(7, 13), "conc": 0.94}
    xfib = lambda: {"spacing": s0, "radius": u(0.6, s0 / 4), "half": u(7, 13), "conc": 0.94}       # crossings: one z lattice, thin enough
                                                                                                     # to interleave (layers s0/2 apart)
    spec = []
    if family in ("cross", "fine"):                                                                 # Monte Carlo fibre crossings
        A = _rand_unit(g); q = torch.nn.functional.normalize(torch.linalg.cross(A, _rand_unit(g)), dim=0)
        ang = math.radians(u(30, 90)); B = math.cos(ang) * A + math.sin(ang) * q; nn_ = torch.linalg.cross(A, B).tolist()
        fine = family == "fine"                                                                      # fine: fibres below the grid (partial volume)
        mc = lambda ax: {"kind": "fibres", "axis": ax.tolist(), "normal": nn_, "half": u(7, 13), "radius": u(0.35, 0.6) if fine else u(0.6, 1.4),
                         "fvf": u(0.2, 0.35), "wander": u(0.2, 1.0), "conc": 0.94}
        spec = [mc(A), mc(B)]
        if u(0, 1) < 0.25:                                                                           # a third population, any direction
            spec.append(mc(_rand_unit(g)))
    elif family == "cross_lattice":
        th = u(0, math.pi); ang = math.radians(u(30, 90)) * (1 if u(0, 1) < 0.5 else -1)
        spec = [{"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th), **xfib(), "z0": 0.0},
                {"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th + ang), **xfib(), "z0": s0 / 2}]
        if u(0, 1) < 0.25:                                                                           # a third population (between the layers)
            third = xfib(); third["radius"] = min(third["radius"], s0 / 8 + 0.3)
            spec.append({"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th + ang / 2 + math.pi / 2), **third, "z0": s0 / 4})
    elif family == "fine_lattice":                                                                  # a crossing of unresolved fibres: finer and
        sf = u(1.6, 2.4)                                                                             # closer than the grid (partial volume only)
        th = u(0, math.pi); ang = math.radians(u(30, 90)) * (1 if u(0, 1) < 0.5 else -1)
        ff = lambda: {"spacing": sf, "radius": u(0.4, min(0.7, sf / 4 + 0.15)), "half": u(7, 13), "conc": 0.94}
        spec = [{"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th), **ff(), "z0": 0.0},
                {"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th + ang), **ff(), "z0": sf / 2}]
    elif family == "fan":
        spec = [{"kind": "fan", "normal": nrm.tolist(), "axis": inplane(u(0, math.pi)), **fib(), "spread": math.radians(u(30, 80)), "half_z": u(6, 12)}]
    elif family == "curve":
        spec = [{"kind": "curve", "normal": nrm.tolist(), "axis": inplane(u(0, math.pi)), **fib(), "curv_radius": u(18, 45), "arc": u(0.6, 1.3), "half_z": u(6, 12)}]
    elif family == "sheet":
        spec = [{"kind": "sheet", "normal": _rand_unit(g).tolist(), "spacing": u(3.0, 6.0), "thickness": u(1.0, 2.2), "half": u(10, 18), "conc": 0.94}]
    elif family == "blob":
        spec = [{"kind": "blob", "half": u(12, 22), "blob_radius": u(2.0, 4.5), "count": int(u(6, 25)), "seed": int(u(0, 1e6))}]
    elif family == "mix":                                                                            # a crossing or bundle with blobs or a sheet
        th = u(0, math.pi)
        spec = [{"kind": "bundle", "normal": nrm.tolist(), "axis": inplane(th), **fib(), "z0": 0.0}]
        r_ = u(0, 1)
        if r_ < 0.35:
            spec.append({"kind": "blob", "half": u(10, 18), "blob_radius": u(2.0, 4.0), "count": int(u(5, 15)), "seed": int(u(0, 1e6))})
        elif r_ < 0.7:
            spec.append({"kind": "sheet", "normal": _rand_unit(g).tolist(), "spacing": u(3.5, 6.0), "thickness": u(1.0, 2.0), "half": u(8, 14), "conc": 0.94})
        else:
            fz = xfib(); fz["spacing"] = s0
            spec.append({"kind": "fan", "normal": nrm.tolist(), "axis": inplane(th + math.radians(u(40, 90))), **fz, "spread": math.radians(u(25, 60)), "z0": s0 / 2, "half_z": u(6, 10)})
            spec[0].update({"spacing": s0, "radius": min(spec[0]["radius"], s0 / 4)})
    elif family == "empty":
        spec = []
    con = []
    for _ in spec:
        cc = u(0.3, 0.8); con.append(cc)
    bright = u(0, 1) < 0.5
    bg = u(0.05, 0.9 - max(con, default=0.3)) if bright else u(max(con, default=0.3) + 0.05, 0.95)
    look = {"bg": bg, "contrast": [c if bright else -c for c in con], "noise": u(0.0, 0.05), "blur": u(0, 0.8)}
    return spec, look


def batch_worker(rank: int, n_workers: int, families, batch: int, n_points: int, sigma, q, start: int, stop: int, threads: int = 4):
    """A CPU process making the trainer's batches ``(step, images, targets)`` for steps ``start + rank, + n_workers, ...``
    into its own queue ``q`` (in order: the trainer reads worker ``step % n_workers``'s queue for step ``step``, so the
    order and the backpressure are the queues'). Family, spec, phantom and queries seeded by the step: the same batches
    whatever the number of workers."""
    torch.set_num_threads(threads)
    for step in range(start + rank, stop, n_workers):
        imgs, ts = [], []
        for b in range(batch):
            fam = families[int(torch.randint(0, len(families), (1,), generator=torch.Generator().manual_seed(7919 * step + b)))]
            spec, look = random_spec(fam, torch.Generator().manual_seed(step * 31 + b))
            ph = make(spec, seed=step * 31 + b, look=look, device="cpu", ss=2 if fam == "fine" else 1)
            g = torch.Generator().manual_seed(104729 * step + b)
            imgs.append(ph["image"]); ts.append(targets(ph, *draw_queries(ph, g, n_points, sigma=sigma)))
        q.put((step, imgs, ts))
