"""The entity field at inference (notes/plan_entity_field_model.md, section 6).

* :func:`features_tiled` / :func:`query` -- the decoder at any points and scales of a volume of any size: the trunk run
  on tiles whose origins are multiples of 8 (the trunk's coarsest stride) with a halo of half the receptive field (plus
  the decoder's lattice corner), the decoder told the true distance to the volume's edge. The trunk is convolutional
  without normalisation, so the tiled result is the whole-volume result: no block artefacts by construction.
* :func:`cluster` -- per point, the directions with density grouped into entities: neighbouring directions whose
  embeddings agree are one entity (label propagation on the direction grid); lobes apart on the sphere with the same
  embedding (a bend, a branch in the window) are merged. No fixed number of entities: the density threshold and the
  embedding distance are the knobs. Each entity: mass, the Bingham of its votes, the non-parametric ODF (its density
  over its directions) and that ODF's moment-matched Bingham, its embedding and descriptors.
* :func:`hierarchy` -- entities grouped at any depth by average-linkage clustering of their hyperbolic distances.
* :func:`link` -- object identities across points (and so across windows): entities at neighbouring points joined
  when their embeddings and axes agree (union-find); :func:`follow` -- seeded following along an entity's axis.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import bingham as BG
from .model import poincare_dist


def sphere_dirs(n: int = 300, device="cpu") -> torch.Tensor:
    """``n`` near-uniform directions on the upper hemisphere (Fibonacci): the density is antipodally symmetric."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - i / n                                                                                 # cos(theta) uniform in (0, 1]
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = torch.sqrt(1 - z ** 2)
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)        # (z, y, x)


def dir_neighbours(dirs: torch.Tensor, k: int = 8) -> torch.Tensor:
    """``(D, k)`` the ``k`` nearest directions of each (antipodes identified)."""
    c = (dirs @ dirs.T).abs(); c.fill_diagonal_(-1)
    return c.topk(k, dim=1).indices


def halo_for(model) -> int:
    """Half the receptive field plus a stride-4 lattice cell, rounded up to a multiple of 8."""
    from .train import receptive_field
    if not hasattr(model, "_halo"):
        with torch.enable_grad():                                                                     # measured by a backward pass
            model._halo = int(math.ceil((receptive_field(model) / 2 + 4) / 8) * 8)
    return model._halo


def tiles(shape, core: int = 128, halo: int = 56):
    """Tiles over a volume: ``(core_lo, core_hi, in_lo, in_hi)`` per tile (voxels), cores partitioning the volume,
    inputs the core plus the halo (clipped to the volume), origins on multiples of 8."""
    assert core % 8 == 0 and halo % 8 == 0
    rng = [list(range(0, n, core)) for n in shape]
    for a in rng[0]:
        for b in rng[1]:
            for c in rng[2]:
                lo = np.array([a, b, c]); hi = np.minimum(lo + core, shape)
                yield lo, hi, np.maximum(lo - halo, 0), np.minimum(hi + halo, shape)


@torch.no_grad()
def features(model, x: torch.Tensor, voxel: float | None = None):
    """The trunk's feature levels of ``x (Z, Y, X)`` (0-1), conditioned on the voxel size (log2 µm; None: unknown)."""
    xb = x[None, None].float()
    h0, h1, h = model.features(xb)
    v = torch.tensor([float("nan") if voxel is None else float(voxel)], device=x.device)
    return h0, h1, model.condition(h, v)


@torch.no_grad()
def total_odf(model, feats, points: torch.Tensor, dirs: torch.Tensor, offset=(0, 0, 0)) -> torch.Tensor:
    """The total-ODF head (SH on the stride-4 lattice) at ``points (Q, 3)`` (trilinear), evaluated at ``dirs``: ``(Q, D)``."""
    from .query import _sample
    from .sh import real_sh
    off = torch.as_tensor(offset, device=points.device, dtype=torch.float32)
    c = model.total(feats[2])[0].float()                                                          # (n_coef, z, y, x)
    cq = _sample(c, ((points.float() - off) - 1.5) / 4).T                                         # (Q, n_coef): lattice cell i at voxel 4 i + 1.5
    lmax = int(round((math.sqrt(8 * c.shape[0] + 1) - 3) / 2))
    return cq @ real_sh(dirs.float(), lmax).T


@torch.no_grad()
def query(model, feats, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, offset=(0, 0, 0), vol_shape=None,
          chunk: int = 1024, t_rel: float | None = None, t_abs: float = 0.0) -> dict:
    """The decoder at ``points (Q, 3)`` (volume voxels), ``sigma (Q,)`` (voxels), every direction of ``dirs (D, 3)``, from
    ``feats`` of the region starting at ``offset``; the context is the distance to ``vol_shape``'s edge. With ``t_rel``,
    only where the total ODF (the cheap SH head) reaches ``t_rel`` of its peak at the point and ``t_abs`` absolute (the
    rest: no density) -- most directions and empty points skipped. Returns ``f (Q, D)``, ``z (Q, D, E)``, ``R (Q, D, 3,
    3)``, ``kappa (Q, D, 2)``, ``desc (Q, D, 5)`` and ``evaluated (Q, D)``."""
    from .sh import real_sh
    off = torch.as_tensor(offset, device=points.device, dtype=torch.float32)
    vs = torch.as_tensor(vol_shape if vol_shape is not None else feats[0].shape[-3:], device=points.device, dtype=torch.float32)
    dec = model.ent; Y = real_sh(dirs.float(), dec.lmax)
    out = {}
    for a in range(0, points.shape[0], chunk):
        p = points[a:a + chunk].float(); s = sigma[a:a + chunk].float(); n = p.shape[0]; D = dirs.shape[0]
        ctx = torch.minimum(p, vs - 1 - p).amin(-1)
        enc = dec.encode(feats, (p - off)[None], s[None], ctx[None])[0]                            # (n, hidden): once per point
        if t_rel is None:
            ev = torch.ones(n, D, dtype=torch.bool, device=p.device)
        else:
            odf = total_odf(model, feats, p, dirs, offset)
            ev = (odf >= t_rel * odf.amax(1, keepdim=True)) & (odf >= t_abs)
        qi, di = torch.nonzero(ev, as_tuple=True)
        r = dec.decode(enc[qi], Y[di].to(enc.dtype)) if qi.numel() else None
        full = {"f": torch.zeros(n, D, device=p.device), "z": torch.zeros(n, D, dec.emb, device=p.device),
                "R": torch.eye(3, device=p.device).expand(n, D, 3, 3).clone(), "kappa": torch.zeros(n, D, 2, device=p.device),
                "desc": torch.zeros(n, D, 5, device=p.device)}
        if r is not None:
            for k in full:
                full[k][qi, di] = r[k].float()
        full["evaluated"] = ev
        for k, v in full.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def query_volume(model, x: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, voxel: float | None = None,
                 core: int = 128, halo: int | None = None, chunk: int = 1024, t_rel: float | None = None, t_abs: float = 0.0) -> dict:
    """:func:`query` over a volume of any size, tiled (the result equals the whole-volume result)."""
    halo = halo_for(model) if halo is None else halo
    res = None
    for clo, chi, ilo, ihi in tiles(x.shape, core, halo):
        lo = torch.as_tensor(clo, device=points.device, dtype=torch.float32); hi = torch.as_tensor(chi, device=points.device, dtype=torch.float32)
        sel = ((points.round() >= lo) & (points.round() < hi)).all(1)                              # points whose nearest voxel lies in the core
        if not bool(sel.any()):
            continue
        sl = tuple(slice(int(a), int(b)) for a, b in zip(ilo, ihi))
        f = features(model, x[sl], voxel)
        r = query(model, f, points[sel], sigma[sel], dirs, offset=ilo.tolist(), vol_shape=x.shape, chunk=chunk, t_rel=t_rel, t_abs=t_abs)
        if res is None:
            res = {k: torch.zeros((points.shape[0],) + v.shape[1:], device=v.device, dtype=v.dtype) for k, v in r.items()}
        for k, v in r.items():
            res[k][sel] = v
    return res


def _propagate(lab: torch.Tensor, nb: torch.Tensor, ok: torch.Tensor, iters: int) -> torch.Tensor:
    """Connected components of the direction graph per point: ``lab (Q, D)`` initial labels, ``nb (D, k)`` neighbours,
    ``ok (Q, D, k)`` which edges hold. Labels become the smallest index of the component."""
    for _ in range(iters):
        cand = torch.where(ok, lab[:, nb], torch.full_like(lab[:, nb], 1 << 30)).amin(-1)
        new = torch.minimum(lab, cand)
        if bool((new == lab).all()):
            break
        lab = new
    return lab


@torch.no_grad()
def cluster(r: dict, dirs: torch.Tensor, f_rel: float = 0.15, f_abs: float = 0.02, z_link: float = 0.6, z_merge: float = 0.4,
            k: int = 8, cmax: int = 12) -> dict:
    """Entities per point from the decoder at the direction grid (:func:`query`): directions with density above
    ``max(f_abs, f_rel * the point's peak)``; neighbouring directions (``k`` nearest) joined when their embeddings lie
    within ``z_link`` (hyperbolic); components with mean embeddings within ``z_merge`` merged (lobes of one entity).
    Returns padded ``(Q, C)`` arrays (heaviest first): ``mass`` (the density's mean over the sphere), ``share`` (of the
    point's entity mass: what partial volume leaves identifiable), ``R``, ``kappa``
    (the votes' Bingham), ``R_odf``, ``kappa_odf`` (the entity's own density, moment matched), ``z``, ``desc``,
    ``live``, and ``label (Q, D)`` (the entity of each direction, -1 none)."""
    f, z = r["f"], r["z"]; Q, D = f.shape; dev = f.device
    nb = dir_neighbours(dirs, k).to(dev)
    alive = f >= torch.maximum(torch.full_like(f[:, :1], f_abs), f_rel * f.amax(1, keepdim=True))
    zl = poincare_dist(z[:, :, None, :], z[:, nb, :])                                              # (Q, D, k)
    ok = alive[:, :, None] & alive[:, nb] & (zl < z_link)
    lab = torch.where(alive, torch.arange(D, device=dev)[None].expand(Q, D), torch.full((Q, D), 1 << 30, device=dev, dtype=torch.long))
    lab = _propagate(lab, nb, ok, D)
    # compact labels per point, heaviest first; then merge components with the same embedding
    w = torch.where(alive, f, torch.zeros_like(f))
    key = torch.where(alive, lab, torch.full_like(lab, D))
    comp_mass = torch.zeros(Q, D + 1, device=dev).scatter_add_(1, key, w)[:, :D]
    order = comp_mass.argsort(1, descending=True)[:, :cmax]                                         # (Q, C) component labels by mass
    C = order.shape[1]
    rank = torch.full((Q, D + 1), -1, device=dev, dtype=torch.long)
    rank.scatter_(1, order, torch.arange(C, device=dev)[None].expand(Q, C))
    rank[:, D] = -1
    ent = rank.gather(1, key)                                                                        # (Q, D) entity per direction, -1 none
    ent = torch.where(comp_mass.gather(1, order).gather(1, ent.clamp_min(0)) > 0, ent, torch.full_like(ent, -1))
    zm = _comp_mean(z, w, ent, C)                                                                    # (Q, C, E)
    live_c = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, ent.clamp_min(-1) % (C + 1), w)[:, :C] > 0
    # merge: components whose mean embeddings agree (greedy to the heavier one)
    dzc = poincare_dist(zm[:, :, None], zm[:, None, :])
    same = (dzc < z_merge) & live_c[:, :, None] & live_c[:, None, :]
    tgt = torch.where(same, torch.arange(C, device=dev)[None, None].expand(Q, C, C), torch.full_like(same, C, dtype=torch.long)).amin(-1)
    tgt = torch.where(live_c, tgt, torch.arange(C, device=dev)[None].expand(Q, C))
    ent = torch.where(ent >= 0, tgt.gather(1, ent.clamp_min(0)), ent)
    return _entity_stats(r, dirs, w, ent, C) | {"label": ent}


def _comp_mean(v: torch.Tensor, w: torch.Tensor, ent: torch.Tensor, C: int) -> torch.Tensor:
    Q, D = w.shape; e = (ent % (C + 1)).clamp_min(0); e = torch.where(ent >= 0, ent, torch.full_like(ent, C))
    s = torch.zeros(Q, C + 1, v.shape[-1], device=v.device).scatter_add_(1, e[..., None].expand(-1, -1, v.shape[-1]), w[..., None] * v)
    n = torch.zeros(Q, C + 1, device=v.device).scatter_add_(1, e, w)
    return (s / n.clamp_min(1e-12)[..., None])[:, :C]


def _entity_stats(r: dict, dirs: torch.Tensor, w: torch.Tensor, ent: torch.Tensor, C: int) -> dict:
    """Per (point, entity): mass, the votes' Bingham (their moments averaged, refitted), the entity's density moment and
    its Bingham, mean embedding (projected into the ball) and descriptors."""
    Q, D = w.shape; dev = w.device
    wm = torch.where(ent >= 0, w, torch.zeros_like(w))
    mass = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, torch.where(ent >= 0, ent, torch.full_like(ent, C)), wm)[:, :C] / (2 * D)
    Mv = BG.moments(r["R"].double(), r["kappa"].double()).float().reshape(Q, D, 9)                   # each vote's second moment
    uu = (dirs[:, :, None] * dirs[:, None, :]).reshape(1, D, 9).expand(Q, -1, -1)
    mv = _comp_mean(Mv, wm, ent, C).reshape(Q, C, 3, 3); mo = _comp_mean(uu, wm, ent, C).reshape(Q, C, 3, 3)
    zm = _comp_mean(r["z"], wm, ent, C); n = zm.norm(dim=-1, keepdim=True); zm = torch.where(n >= 0.999, zm / n * 0.999, zm)
    dm = _comp_mean(r["desc"], wm, ent, C)
    live = mass > 0
    R = torch.eye(3, device=dev).expand(Q, C, 3, 3).clone(); K = torch.zeros(Q, C, 2, device=dev)
    Ro, Ko = R.clone(), K.clone()
    if bool(live.any()):
        a, b = BG.fit(mv[live].double()); R[live], K[live] = a.float(), b.float()
        a, b = BG.fit(mo[live].double()); Ro[live], Ko[live] = a.float(), b.float()
    share = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)                                       # identifiable under partial volume
    return {"mass": mass, "share": share, "R": R, "kappa": K, "R_odf": Ro, "kappa_odf": Ko, "z": zm, "desc": dm, "live": live}


def hierarchy(z: torch.Tensor):
    """Average-linkage clustering of entities ``z (N, E)`` by hyperbolic distance (scipy linkage matrix): cut at any
    distance for the groups at that depth (``scipy.cluster.hierarchy.fcluster(L, t, "distance")``)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    d = poincare_dist(z[:, None].double(), z[None].double()).cpu().numpy(); np.fill_diagonal(d, 0.0)
    return linkage(squareform(d, checks=False), "average")


def link(points: torch.Tensor, ents: dict, radius: float = 2.5, z_link: float = 0.6, ang_max: float = 25.0) -> torch.Tensor:
    """Object identities across points: entity ``c`` at point ``i`` and ``c'`` at a point within ``radius`` voxels are
    candidates when their embeddings lie within ``z_link`` and their mode axes within ``ang_max`` degrees, and each is
    the other's best match there (mutual best). Candidates are joined strongest first (union-find), never joining two
    objects that share a point -- entities seen apart at one point stay apart, so a chain of weak links through a
    crossing cannot merge the crossing objects. ``ents`` from :func:`cluster` at ``points (Q, 3)``. Returns ``(Q, C)``
    ids (-1 none)."""
    from scipy.spatial import cKDTree
    Q, C = ents["mass"].shape
    live = ents["live"].cpu().numpy(); z = ents["z"].cpu().double(); ax = ents["R"][..., :, 2].cpu()
    parent = np.arange(Q * C); members = {a: {a // C} for a in range(Q * C) if live.reshape(-1)[a]}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pairs = cKDTree(points.cpu().numpy()).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        i, j = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
        dz = poincare_dist(z[i][:, :, None], z[j][:, None, :])                                     # (P, C, C)
        ca = (ax[i][:, :, None] * ax[j][:, None, :]).sum(-1).abs()
        ok = (dz < z_link) & (ca > math.cos(math.radians(ang_max)))
        ok &= torch.as_tensor(live[pairs[:, 0]])[:, :, None] & torch.as_tensor(live[pairs[:, 1]])[:, None, :]
        sc = torch.where(ok, dz, torch.full_like(dz, math.inf))
        mutual = ok & (sc == sc.amin(2, keepdim=True)) & (sc == sc.amin(1, keepdim=True))
        p_, a_, b_ = torch.nonzero(mutual, as_tuple=True)
        for k in torch.argsort(sc[p_, a_, b_]).tolist():                                            # strongest first
            ra, rb = find(int(pairs[p_[k], 0]) * C + int(a_[k])), find(int(pairs[p_[k], 1]) * C + int(b_[k]))
            if ra == rb or (members[ra] & members[rb]):                                             # cannot-link: one point, two objects
                continue
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra; members[ra] |= members.pop(rb)
    ids = np.array([find(a) for a in range(Q * C)]).reshape(Q, C)
    ids = np.where(live, ids, -1)
    u, inv = np.unique(ids[ids >= 0], return_inverse=True)
    out = np.full((Q, C), -1, dtype=np.int64); out[ids >= 0] = inv
    return torch.as_tensor(out)


@torch.no_grad()
def follow(model, x: torch.Tensor, seed, direction, voxel: float | None = None, sigma: float = 2.0, step: float = 2.0, max_steps: int = 400,
           z_max: float = 0.6, ang_max: float = 35.0, n_dirs: int = 300, core: int = 64, halo: int | None = None, **cluster_kw) -> dict:
    """Seeded following: from ``seed`` (voxels), the entity nearest ``direction``; then step along its axis, re-query,
    keep the entity whose embedding lies within ``z_max`` of the last and whose axis is within ``ang_max`` degrees,
    until none does or the volume ends. The trunk is computed on a tile around the current point as needed. Returns
    ``points (n, 3)``, ``axes``, ``z``, ``mass``."""
    dev = x.device; dirs = sphere_dirs(n_dirs, dev); halo = halo_for(model) if halo is None else halo
    cache = {"lo": None}
    def ents_at(p):
        cl = (torch.div(p.round().long(), core, rounding_mode="floor") * core).cpu().numpy()
        if cache["lo"] is None or (cache["lo"] != cl).any():
            ilo = np.maximum(cl - halo, 0); ihi = np.minimum(cl + core + halo, x.shape)
            cache.update(lo=cl, ilo=ilo, f=features(model, x[tuple(slice(int(a), int(b)) for a, b in zip(ilo, ihi))], voxel))
        r = query(model, cache["f"], p[None].float(), torch.tensor([sigma], device=dev), dirs, offset=cache["ilo"].tolist(), vol_shape=x.shape)
        return cluster(r, dirs, **cluster_kw)
    p = torch.as_tensor(seed, device=dev, dtype=torch.float32); d = torch.nn.functional.normalize(torch.as_tensor(direction, device=dev, dtype=torch.float32), dim=0)
    e = ents_at(p)
    if not bool(e["live"][0].any()):
        return {"points": p[None], "axes": d[None], "z": torch.zeros(1, e["z"].shape[-1], device=dev), "mass": torch.zeros(1, device=dev)}
    c = int(torch.where(e["live"][0], (e["R"][0, :, :, 2] @ d).abs(), torch.full_like(e["mass"][0], -1)).argmax())
    pts, axes, zs, ms = [p], [], [], []
    zc, ac = e["z"][0, c], e["R"][0, c, :, 2]; ac = ac if float(ac @ d) >= 0 else -ac
    axes.append(ac); zs.append(zc); ms.append(e["mass"][0, c])
    shape = torch.tensor(x.shape, device=dev, dtype=torch.float32)
    for _ in range(max_steps):
        p = p + step * ac
        if bool((p < 0).any() or (p > shape - 1).any()):
            break
        e = ents_at(p)
        dz = poincare_dist(e["z"][0], zc[None]); ca = (e["R"][0, :, :, 2] @ ac).abs()
        ok = e["live"][0] & (dz < z_max) & (ca > math.cos(math.radians(ang_max)))
        if not bool(ok.any()):
            break
        c = int(torch.where(ok, -dz, torch.full_like(dz, -1e9)).argmax())
        a_ = e["R"][0, c, :, 2]; ac = a_ if float(a_ @ ac) >= 0 else -a_; zc = e["z"][0, c]
        pts.append(p); axes.append(ac); zs.append(zc); ms.append(e["mass"][0, c])
    return {"points": torch.stack(pts[:len(axes)]), "axes": torch.stack(axes), "z": torch.stack(zs), "mass": torch.stack(ms)}
