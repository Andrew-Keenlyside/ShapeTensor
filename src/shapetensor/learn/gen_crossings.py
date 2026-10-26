"""Held-out crossings from the scene generator for the separation check: two object populations that each run as lines
through their overlap and cross at >= 45 deg, both visible where they cross; the slice normal to the axis both lines
lie most in, through the most overlap; on that slice's grid (sigma 2.5 windows) the target mass of A, of B, of the
other object populations and of the background, each population's local axis (its entities' moments, carried a few
voxels past its edge), and the zones the criteria use (:func:`~shapetensor.learn.crossing_acceptance.evaluate_generic`)."""
import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def tensor33(t6: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` (xx, yy, zz, xy, xz, yz) -> ``(..., 3, 3)``."""
    T = torch.zeros(*t6.shape[:-1], 3, 3, device=t6.device, dtype=torch.float32); t6 = t6.float()
    for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T[..., i, j] = t6[..., c]; T[..., j, i] = t6[..., c]
    return T


def principal(T: torch.Tensor):
    ev, V = torch.linalg.eigh(T)
    return ev[..., -1] / ev.sum(-1).clamp_min(1e-12), V[..., -1]


def line_crossing(s, d, dev):
    """``(A, B, slice axis, slice index, angle)`` for the two most-overlapping object populations if both run as lines
    through their overlap and cross at >= 45 deg; else None."""
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
    if len(keep) < 2:
        return None
    occ = d["occupancy"][keep].float()
    ov = (occ.unsqueeze(1) * occ.unsqueeze(0)).sum((2, 3, 4)) * (1 - torch.eye(len(keep), device=dev))
    i, j = divmod(int(ov.argmax()), len(keep))
    core = (occ[i] > 0.3) & (occ[j] > 0.3)
    if int(core.sum()) < 60:
        return None
    A, B = keep[i], keep[j]; ax = []
    for k in (A, B):
        lin, u = principal(tensor33(d["tensors"][k].float()[:, core].mean(1)))
        if float(lin) < 0.6:
            return None
        ax.append(u)
    ang = math.degrees(math.acos(min(1.0, abs(float(ax[0] @ ax[1])))))
    if ang < 45:
        return None
    sa = int(torch.minimum(1 - ax[0] ** 2, 1 - ax[1] ** 2).argmax())
    prof = core.float().sum(tuple(k for k in range(3) if k != sa))
    return A, B, sa, int(prof.argmax()), ang


def entity_layers(s, d, gn, not_entity):
    """The layer of each target entity (from its node in the scene's tree; -1: none)."""
    ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"])
    gl_ = [k for k in gl if k >= 0]; dense = [k for k, L_ in enumerate(s.layers) if L_.role not in not_entity and k not in gl_]
    gn = gn.long(); lay = torch.full_like(gn, -1)
    grp = tree.group.to(gn.device)[gn.clamp(0, tree.n - 1)]
    glt = torch.tensor(gl + [-1], device=gn.device)
    lay = torch.where((gn >= 0) & (gn < tree.n), glt[grp.clamp(-1, len(gl) - 1)], lay)
    if dense:
        lay = torch.where(gn >= tree.n, torch.tensor(dense, device=gn.device)[(gn - tree.n).clamp(0, len(dense) - 1)], lay)
    return lay


def slice_targets(TR, s, d, seed, lc, dev, n=64):
    """The slice's grid points, image, outlines and target maps (A, B, other objects, background), each population's
    local axis, and the distance of each grid point to the volume's faces."""
    A, B, sa, si, ang = lc
    oth = [k for k in range(3) if k != sa]
    gu, gv = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    pts = torch.zeros(n * n, 3); pts[:, sa] = float(si); pts[:, oth[0]] = gu.flatten().float(); pts[:, oth[1]] = gv.flatten().float()
    pts = pts.to(dev); sig = torch.full((n * n,), 2.5, device=dev)
    t = TR.entity_queries(s, d, n, seed, dev, points=pts, sigma=sig)
    lay = entity_layers(s, d, t["e_gnode"], TR.NOT_ENTITY); w = t["e_mass"].float() * t["e_vis"].float() * (t["e_node"] >= 0)
    iso_l = torch.tensor([L_.role == "isotropic" for L_ in s.layers] + [False], device=dev)[lay]
    Tt = tensor33(t["e_T6"])
    out = {"pts": pts, "sig": sig, "image": d["image"].float().select(sa, si).cpu().numpy() / 255.0, "angle": ang,
           "occA": d["occupancy"][A].float().select(sa, si).cpu().numpy(), "occB": d["occupancy"][B].float().select(sa, si).cpu().numpy(), "axes": {}}
    for nm, k in (("A", A), ("B", B)):
        wk = w * (lay == k)
        out["t" + nm] = wk.sum(1).view(n, n).cpu().numpy()
        Tk = (wk[..., None, None] * Tt).sum(1).view(n, n, 9).cpu().numpy()
        Tk = np.stack([gaussian_filter(Tk[..., q], 3.0) for q in range(9)], -1)
        wb = gaussian_filter(out["t" + nm], 3.0)
        _, u = principal(torch.as_tensor(Tk).view(-1, 3, 3))
        out["axes"][nm] = (u.to(dev), torch.as_tensor(wb.reshape(-1) > 0.02, device=dev))
    out["tO"] = (w * ((lay >= 0) & (lay != A) & (lay != B) & ~iso_l)).sum(1).view(n, n).cpu().numpy()
    out["ent_t"] = {"mass": (w * ~iso_l).cpu().numpy(), "kappa": t["e_kappa"].cpu().numpy(), "R": t["e_R"].cpu().numpy(),      # the target entities
                    "iso_mass": (w * iso_l).sum(1).cpu().numpy()}                                                               # (the ground-truth check)
    out["tI"] = (w * iso_l).sum(1).view(n, n).cpu().numpy()
    out["_face"] = torch.minimum(pts, (n - 1) - pts).amin(1).view(n, n).cpu().numpy()
    return out


def cases(TR, SS, gen, seed0, n_cases, dev, strength=0.3, max_draws=3000, min_zone=20):
    """``n_cases`` held-out generator crossings (line crossings, both visible where they cross, every zone with at
    least ``min_zone`` points on the slice): dicts with the scene, the draw, the crossing and its slice targets."""
    from .crossing_acceptance import generic_zones
    out, seed = [], seed0
    while len(out) < n_cases and seed < seed0 + max_draws:
        SS._CACHE.clear()
        s, d = SS.draw_scene((64,) * 3, seed=seed, device=dev, strength=strength, entity_ctx=True, **gen["scene"])
        seed += 1
        lc = line_crossing(s, d, dev)
        if lc is None:
            continue
        tm = slice_targets(TR, s, d, seed - 1, lc, dev)
        Z = generic_zones(tm)
        if min(int(Z[z].sum()) for z in ("armA", "armB", "core", "bg")) < min_zone:
            continue
        out.append({"seed": seed - 1, "s": s, "d": d, "lc": lc, "tm": tm})
    return out, seed - seed0
