"""Simple crossing phantoms for checking the entity model one step at a time: two straight bundles of parallel fibres
crossing at ``angle`` (A along x, B in the x-y plane), each one population entity (nothing below it), their fibres in
alternating z layers (interleaved, as crossing fibres are), on a flat background; the rest of each voxel is the
isotropic background entity. :func:`make` draws one; :func:`targets` gives the entity targets at any queries (the
model's own window operator for the masses); :func:`draw_queries` places training queries."""
import math

import torch

from . import bingham as BG
from .model import window_mean


def bundle(grid, c: float, u: torch.Tensor, z0: float, half: float, spacing: float, radius: float) -> torch.Tensor:
    """Occupancy of a bundle along unit ``u`` in the x-y plane through the centre ``c``: fibres on a ``spacing`` grid
    across it (in-plane and in z, the z layers from ``z0``), within ``half`` of the centre line in both cross directions,
    partial volume at the fibres' edges."""
    zz, yy, xx = grid
    p = torch.stack([zz - c, yy - c, xx - c], -1)
    w = torch.stack([torch.zeros_like(u[0]), -u[2], u[1]])                                          # in-plane, across the bundle
    q1 = (p * w).sum(-1); q2 = p[..., 0]
    d1 = q1 - spacing * torch.round(q1 / spacing); d2 = q2 - z0 - spacing * torch.round((q2 - z0) / spacing)
    inside = (q1.abs() <= half) & (q2.abs() <= half)
    return ((radius + 0.5 - torch.sqrt(d1 ** 2 + d2 ** 2)).clamp(0, 1) * inside).float()


def frame(u: torch.Tensor | None, conc: float, device) -> tuple:
    """``(T6, R, kappa)`` of a line along ``u`` with second moment ``conc`` on its axis (``None``: isotropic)."""
    I = torch.eye(3, device=device)
    T = I / 3 if u is None else conc * torch.outer(u, u) + (1 - conc) / 2 * (I - torch.outer(u, u))
    R, k = BG.fit(T.double()); T6 = torch.stack([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])
    return T6.float(), R.float(), k.float()


def make(angle: float = 90.0, noise: float = 0.01, n: int = 64, half: float = 10.0, spacing: float = 4.0, radius: float = 1.2,
         contrast: float = 0.8, bg: float = 0.1, seed: int = 0, device="cuda") -> dict:
    """One phantom: ``image (n, n, n)`` (0-1, 8-bit levels), ``occ`` A / B / bg, ``axes`` A / B (z, y, x), ``ent`` the
    entities' frames (A, B, background), ``core`` the crossing's cube, and its parameters."""
    c = n / 2.0; th = math.radians(angle)
    axes = {"A": torch.tensor([0.0, 0.0, 1.0], device=device), "B": torch.tensor([0.0, math.sin(th), math.cos(th)], device=device)}
    grid = torch.meshgrid(*(torch.arange(n, dtype=torch.float32, device=device),) * 3, indexing="ij")
    occ = {"A": bundle(grid, c, axes["A"], 0.0, half, spacing, radius), "B": bundle(grid, c, axes["B"], spacing / 2, half, spacing, radius)}
    occ["bg"] = 1 - (occ["A"] + occ["B"]).clamp(max=1.0)
    g = torch.Generator(device=device).manual_seed(seed)
    img = (bg + contrast * (occ["A"] + occ["B"]).clamp(max=1.0) + noise * torch.randn(n, n, n, generator=g, device=device)).clamp(0, 1)
    img = (img * 255).round() / 255                                                                 # 8-bit, as the generator's
    zz, yy, xx = grid
    in_a = (((yy - c) * axes["A"][2] - (xx - c) * axes["A"][1]).abs() <= half)
    in_b = (((yy - c) * axes["B"][2] - (xx - c) * axes["B"][1]).abs() <= half)
    core = in_a & in_b & ((zz - c).abs() <= half)
    ent = [frame(axes["A"], 0.94, device), frame(axes["B"], 0.94, device), frame(None, 0, device)]   # nodes 1, 2, 3 under the root 0
    return {"image": img, "occ": occ, "axes": axes, "ent": ent, "core": core, "n": n, "c": c, "half": half, "spacing": spacing,
            "angle": angle, "noise": noise}


def targets(ph: dict, pts: torch.Tensor, sig: torch.Tensor) -> dict:
    """The entity targets at ``pts (Q, 3)``, ``sig (Q,)``: per query A, B, background (window means of their
    occupancy, all visible), padded to 8, with the crop's tree (root 0; A, B, background its children)."""
    dev = pts.device; Q = pts.shape[0]; E = 8; n = ph["n"]
    vol = torch.stack([ph["occ"]["A"], ph["occ"]["B"], ph["occ"]["bg"]])
    m = window_mean(vol, pts[None].expand(3, -1, -1).contiguous(), sig[None].expand(3, -1).contiguous()).T   # (Q, 3)
    live = m > 1e-4
    node = torch.full((Q, E), -1, dtype=torch.long, device=dev)
    node[:, :3] = torch.where(live, torch.arange(1, 4, device=dev), torch.full_like(node[:, :3], -1))
    mass = torch.zeros(Q, E, device=dev); mass[:, :3] = m * live
    T6 = torch.zeros(Q, E, 6, device=dev); R = torch.eye(3, device=dev).expand(Q, E, 3, 3).clone(); K = torch.zeros(Q, E, 2, device=dev)
    for j, (t6, r, k) in enumerate(ph["ent"]):
        T6[:, j], R[:, j], K[:, j] = t6, r, k
    anc = torch.full((1536, 16), -1, dtype=torch.long, device=dev); anc[0, 0] = 0
    for i in (1, 2, 3):
        anc[i, 0], anc[i, 1] = i, 0
    return {"e_points": pts, "e_sigma": sig, "e_node": node, "e_mass": mass, "e_vis": (node >= 0).float(), "e_T6": T6,
            "e_desc": torch.zeros(Q, E, 5, device=dev), "e_anc": anc, "e_n": torch.tensor(4, device=dev), "e_R": R, "e_kappa": K,
            "e_gnode": node, "region_t": ph["occ"]["A"] + ph["occ"]["B"] + ph["occ"]["bg"], "iface_t": torch.zeros(n, n, n, device=dev),
            "voxel": torch.tensor(float("nan"), device=dev)}


def draw_queries(ph: dict, g: torch.Generator, n_q: int) -> tuple:
    """Half on the fibres, a quarter in the crossing core, the rest anywhere (2 voxels from the faces); sigma 1-6."""
    n = ph["n"]; dev = ph["image"].device
    pts = 2 + torch.rand(n_q, 3, generator=g, device=dev) * (n - 5)
    for frac, off, sel in ((0.5, 0, (ph["occ"]["A"] + ph["occ"]["B"]) > 0.05), (0.25, 0.5, ph["core"])):
        cand = torch.nonzero(sel).float(); k = int(frac * n_q); o = int(off * n_q)
        if cand.shape[0]:
            pts[o:o + k] = (cand[torch.randint(0, cand.shape[0], (k,), generator=g, device=dev)] + torch.rand(k, 3, generator=g, device=dev) - 0.5).clamp(2, n - 3)
    sig = torch.exp(torch.rand(n_q, generator=g, device=dev) * math.log(6.0))
    return pts, sig
