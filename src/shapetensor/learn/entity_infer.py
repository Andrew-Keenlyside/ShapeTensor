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
from .entity_train import TREE_EDGE
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


def tiles(shape, core: int = 128, halo: int = 80):
    """Tiles over a volume: ``(core_lo, core_hi, in_lo, in_hi)`` per tile (voxels), cores partitioning the volume,
    inputs the core plus the halo (clipped to the volume), origins on multiples of 8."""
    assert core % 8 == 0 and halo % 8 == 0
    rng = [list(range(0, n, core)) for n in shape]
    for a in rng[0]:
        for b in rng[1]:
            for c in rng[2]:
                lo = np.array([a, b, c]); hi = np.minimum(lo + core, shape)
                yield lo, hi, np.maximum(lo - halo, 0), np.minimum(hi + halo, shape)


def normalise(x: torch.Tensor, q=(0.001, 0.999), n: int = 2_000_000, seed: int = 0) -> torch.Tensor:
    """One intensity window for a whole volume (the quantiles of a random subsample), as training windows each crop:
    the same mapping for every tile -- a per-tile window would make blocks. Returns 0-1."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    flat = x.reshape(-1)
    sub = flat[torch.randint(0, flat.numel(), (min(n, flat.numel()),), generator=g, device=x.device)].float()
    lo, hi = torch.quantile(sub, torch.tensor(q, device=x.device))
    return ((x.float() - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)


def _voxel(voxel=None, voxel_um=None):
    """The model's voxel conditioning: log2 of the voxel size in µm (None: unknown)."""
    if voxel_um is not None:
        return math.log2(float(voxel_um))
    return voxel


@torch.no_grad()
def features(model, x: torch.Tensor, voxel: float | None = None, voxel_um: float | None = None):
    """The trunk's feature levels of ``x (Z, Y, X)`` (0-1: see :func:`normalise`), conditioned on the voxel size
    (``voxel``: log2 µm, or ``voxel_um``; None: unknown). The input is padded (replicate) to multiples of 8 at its far
    faces -- the trunk's coarsest stride, as in training; the padding's features are never queried."""
    pad = [(-n) % 8 for n in x.shape]
    xb = x[None, None].float()
    if any(pad):
        xb = torch.nn.functional.pad(xb, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
    h0, h1, h = model.features(xb)
    v = _voxel(voxel, voxel_um)
    v = torch.tensor([float("nan") if v is None else float(v)], device=x.device)
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
    reg = None
    if getattr(model, "mass_region", False) and getattr(model, "region", None) is not None:     # the mass: the region map's window mean
        from .model import window_mean
        tsz = [int(min(n, v - o)) for n, v, o in zip(feats[0].shape[-3:], vs.tolist(), off.tolist())]   # the tile (no far-face padding)
        reg = torch.sigmoid(model.region_map(*feats).float())[:, :tsz[0], :tsz[1], :tsz[2]]
    for a in range(0, points.shape[0], chunk):
        p = points[a:a + chunk].float(); s = sigma[a:a + chunk].float(); n = p.shape[0]; D = dirs.shape[0]
        ctx = torch.minimum(p, vs - 1 - p).amin(-1)
        enc = dec.encode(feats, (p - off)[None], s[None], ctx[None])[0]                            # (n, hidden): once per point
        mq = dec.mass_of(enc) if getattr(dec, "mass", None) is not None else None                  # the mass output (where), if the model has one
        if reg is not None:
            mq = window_mean(reg, (p - off)[None], s[None])[0]
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
        if mq is not None:
            full["m"] = mq.float()
        for k, v in full.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def query_volume(model, x: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, voxel: float | None = None,
                 core: int = 128, halo: int | None = None, chunk: int = 1024, t_rel: float | None = None, t_abs: float = 0.0,
                 voxel_um: float | None = None) -> dict:
    """:func:`query` over a volume of any size, tiled (the result equals the whole-volume result)."""
    halo = halo_for(model) if halo is None else halo
    voxel = _voxel(voxel, voxel_um)
    inside = ((points >= 0) & (points <= torch.tensor(x.shape, device=points.device) - 1)).all(1)
    if not bool(inside.all()):
        raise ValueError(f"{int((~inside).sum())} query points lie outside the volume")
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


def sym_neighbours(dirs: torch.Tensor, k: int = 8) -> torch.Tensor:
    """``(D, kk)`` neighbour lists of the direction grid made symmetric (k nearest, plus every direction that counts
    this one among its k nearest), padded with -1."""
    nb = dir_neighbours(dirs, k); D = dirs.shape[0]
    A = torch.zeros(D, D, dtype=torch.bool, device=dirs.device)
    A[torch.arange(D, device=dirs.device)[:, None].expand(-1, k), nb] = True; A = A | A.T
    kk = int(A.sum(1).max())
    out = torch.full((D, kk), -1, dtype=torch.long, device=dirs.device)
    for d in range(D):
        nz = torch.nonzero(A[d])[:, 0]; out[d, :nz.numel()] = nz
    return out


def _components(adj: torch.Tensor) -> torch.Tensor:
    """Connected components of dense symmetric adjacencies ``(Q, N, N)``: each node labelled with its component's
    smallest index (min-label propagation, to convergence)."""
    Q, N, _ = adj.shape
    lab = torch.arange(N, device=adj.device)[None].expand(Q, N).clone()
    big = 1 << 30
    for _ in range(N):
        new = torch.minimum(lab, torch.where(adj, lab[:, None, :].expand(-1, N, -1), torch.full_like(adj, big, dtype=torch.long)).amin(-1))
        if bool((new == lab).all()):
            break
        lab = new
    return lab


@torch.no_grad()
def cluster(r: dict, dirs: torch.Tensor, f_frac: float = 0.25, f_abs: float = 1e-4, vote_tau: float = 0.25, z_link: float = 0.5 * TREE_EDGE,
            core_frac: float = 0.6, share_min: float = 0.03, k: int = 8, na: int = 128, cmax: int = 12, chunk: int = 256) -> dict:
    """Entities per point from the decoder at the direction grid (:func:`query`).

    Directions with density at least ``f_frac`` of the point's mean density (and ``f_abs``) take part -- a cut relative
    to the mean, not the peak, keeps broad entities (fans, sheets, isotropic material) beside sharp ones. Each direction
    carries its entity's vote (the Bingham's second moment) and embedding. **Cores** are directions whose grid
    neighbours (symmetric k-nearest) mostly (``core_frac``) agree with them -- vote within ``vote_tau`` (Frobenius of the
    moments; ~0.25 = 10° between peaked lobes) and embedding within ``z_link``: the transitions between entities, where
    neither is settled, are not cores and cannot chain entities together. Cores agreeing in vote and embedding are one
    entity wherever they lie on the sphere (a branch's two arms vote for one entity), so entities separate by vote
    (crossings) or by embedding (parallel neighbours). The other living directions join the entity nearest in vote and
    embedding together; cores keep their own. Entities below ``share_min`` of the point's entity density are dropped. Returns padded ``(Q, C)`` arrays
    (heaviest first) as :func:`_entity_stats`, and ``label (Q, D)`` (each direction's entity, -1 none)."""
    f, z = r["f"], r["z"]; Q, D = f.shape; dev = f.device
    nb = sym_neighbours(dirs, k).to(dev); nbv = nb >= 0; nbc = nb.clamp_min(0)
    M = BG.moments(r["R"].double(), r["kappa"].double()).float().reshape(Q, D, 9)
    alive = (f >= f_frac * f.mean(1, keepdim=True)) & (f >= f_abs)
    labels = torch.full((Q, D), -1, dtype=torch.long, device=dev)
    for a0 in range(0, Q, chunk):
        sl = slice(a0, a0 + chunk); fq, zq, Mq, al = f[sl], z[sl], M[sl], alive[sl]; q = fq.shape[0]
        agree = ((Mq[:, :, None] - Mq[:, nbc]).norm(dim=-1) < vote_tau) & (poincare_dist(zq[:, :, None], zq[:, nbc]) < z_link)
        nal = al[:, nbc] & nbv[None]
        frac = (agree & nal).sum(-1).float() / nal.sum(-1).clamp_min(1).float()
        core = al & (frac >= core_frac)
        sc = torch.where(core, fq, torch.full_like(fq, -1.0)); top = sc.topk(min(na, D), dim=1).indices; tv = sc.gather(1, top) >= 0
        Mt = Mq.gather(1, top[..., None].expand(-1, -1, 9)); zt = zq.gather(1, top[..., None].expand(-1, -1, zq.shape[-1]))
        adj = ((Mt[:, :, None] - Mt[:, None]).norm(dim=-1) < vote_tau) & (poincare_dist(zt[:, :, None], zt[:, None]) < z_link)
        adj &= tv[:, :, None] & tv[:, None, :]
        comp = torch.where(tv, _components(adj), torch.full_like(top, -1))
        nc = top.shape[1]; key = torch.where(comp >= 0, comp, torch.full_like(comp, nc))
        w = torch.where(tv, fq.gather(1, top), torch.zeros_like(fq.gather(1, top)))
        Wc = torch.zeros(q, nc + 1, device=dev).scatter_add_(1, key, w)[:, :nc]
        Mc = torch.zeros(q, nc + 1, 9, device=dev).scatter_add_(1, key[..., None].expand(-1, -1, 9), w[..., None] * Mt)[:, :nc] / Wc.clamp_min(1e-12)[..., None]
        Zc = torch.zeros(q, nc + 1, zq.shape[-1], device=dev).scatter_add_(1, key[..., None].expand(-1, -1, zq.shape[-1]), w[..., None] * zt)[:, :nc] / Wc.clamp_min(1e-12)[..., None]
        has = Wc > 0
        dM = (Mq[:, :, None] - Mc[:, None]).norm(dim=-1) / vote_tau + poincare_dist(zq[:, :, None], Zc[:, None]) / z_link
        dM = torch.where(has[:, None], dM, torch.full_like(dM, math.inf))
        lab = torch.where(al & has.any(1, keepdim=True), dM.argmin(-1), torch.full((q, D), -1, device=dev, dtype=torch.long))
        lab.scatter_(1, top, torch.where(tv, comp, lab.gather(1, top)))                                # cores keep their own component
        lab = torch.where(al & ~has.any(1, keepdim=True), torch.zeros_like(lab), lab)               # no cores: one entity of what lives
        labels[sl] = lab
    key = torch.where(labels >= 0, labels, torch.full_like(labels, D))
    w = torch.where(labels >= 0, f, torch.zeros_like(f))
    cm = torch.zeros(Q, D + 1, device=dev).scatter_add_(1, key, w)[:, :D]
    order = cm.argsort(1, descending=True)[:, :cmax]; C = order.shape[1]
    keepc = cm.gather(1, order) >= share_min * cm.sum(1, keepdim=True).clamp_min(1e-12)
    keepc &= cm.gather(1, order) > 0
    rank = torch.full((Q, D + 1), -1, device=dev, dtype=torch.long)
    rank.scatter_(1, order, torch.where(keepc, torch.arange(C, device=dev)[None].expand(Q, C), torch.full((Q, C), -1, device=dev, dtype=torch.long)))
    rank[:, D] = -1
    ent = rank.gather(1, key)
    st = _entity_stats(r, dirs, f, ent, C)
    if "m" in r:                                                                                   # a mass output: the point's mass shared by the entities
        st["mass"] = r["m"].float()[:, None] * st["share"] * st["live"]
    return st | {"label": ent}


def cluster_lobes(r: dict, dirs: torch.Tensor, f_frac: float = 0.25, f_abs: float = 1e-4, merge_deg: float = 25.0,
                  share_min: float = 0.03, k: int = 8, cmax: int = 12) -> dict:
    """Entities per point from the density's lobes alone: each living direction (density >= ``f_frac`` of the mean)
    climbs the density over the direction grid (steepest ascent among its ``k`` nearest neighbours, pointer jumping) to
    a local maximum; a maximum's basin is an entity; maxima within ``merge_deg`` of a heavier one join it; the
    ``cmax`` heaviest kept, those below ``share_min`` dropped. Where the votes do not (yet) switch between the lobes
    of a crossing, the density's own peaks still separate its entities. Returns :func:`cluster`'s fields."""
    f = r["f"]; Q, D = f.shape; dev = f.device
    nb = sym_neighbours(dirs, k).to(dev); nbc = nb.clamp_min(0)
    alive = (f >= f_frac * f.mean(1, keepdim=True)) & (f >= f_abs)
    cand = torch.cat([torch.arange(D, device=dev)[:, None], nbc], 1)                                # (D, k + 1): itself and its neighbours
    fc = f[:, cand].masked_fill(~torch.cat([torch.ones(D, 1, dtype=torch.bool, device=dev), nb >= 0], 1)[None], -math.inf)
    up = cand[torch.arange(D, device=dev)[None], fc.argmax(-1)]                                     # (Q, D) the steepest step
    for _ in range(int(math.ceil(math.log2(max(D, 2)))) + 2):                                     # pointer jumping to the maxima
        up = up.gather(1, up)
    peak = torch.where(alive, up, torch.full_like(up, -1))
    w = torch.where(alive, f, torch.zeros_like(f)); key = peak.clamp_min(0)
    pm = torch.zeros(Q, D, device=dev).scatter_add_(1, key, w) * (peak >= 0).any(1, keepdim=True)  # each maximum's basin mass
    order = pm.argsort(1, descending=True)[:, :cmax]; C = order.shape[1]
    pk = pm.gather(1, order) > 0
    cosm = (dirs[order] @ dirs[order].transpose(1, 2)).abs()                                         # (Q, C, C) between the kept maxima
    close = cosm >= math.cos(math.radians(merge_deg))
    heavier = torch.arange(C, device=dev)[None, :, None] > torch.arange(C, device=dev)[None, None, :]   # j before i: heavier
    tgt = torch.where(close & heavier & pk[:, None, :] & pk[:, :, None], torch.arange(C, device=dev)[None, None].expand(Q, C, C), torch.full((Q, C, C), C, device=dev))
    root = tgt.min(-1).values; root = torch.where(root < C, root, torch.arange(C, device=dev)[None].expand(Q, C))
    for _ in range(C):                                                                             # chains of merges to their heaviest
        root = root.gather(1, root)
    slot = torch.full((Q, D + 1), -1, dtype=torch.long, device=dev)                                # maximum index -> kept slot
    slot.scatter_(1, order, torch.where(pk, root, torch.full_like(root, -1)))
    ent0 = torch.where(peak >= 0, slot.gather(1, peak.clamp_min(0)), torch.full_like(peak, -1))
    # far lobes not kept (beyond cmax): to the nearest kept maximum within merge_deg, else none
    lost = alive & (ent0 < 0)
    if bool(lost.any()):
        pd = dirs[peak.clamp_min(0)]                                                               # (Q, D, 3)
        cd = (pd @ dirs[order].transpose(1, 2)).abs().masked_fill(~pk[:, None, :], -1.0)          # (Q, D, C)
        bc, bi = cd.max(-1)
        ent0 = torch.where(lost & (bc >= math.cos(math.radians(merge_deg))), root.gather(1, bi), ent0)
    cm = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, torch.where(ent0 >= 0, ent0, torch.full_like(ent0, C)), w)[:, :C]
    o2 = cm.argsort(1, descending=True); keepc = (cm.gather(1, o2) >= share_min * cm.sum(1, keepdim=True).clamp_min(1e-12)) & (cm.gather(1, o2) > 0)
    rank = torch.full((Q, C + 1), -1, dtype=torch.long, device=dev)
    rank.scatter_(1, o2, torch.where(keepc, torch.arange(C, device=dev)[None].expand(Q, C), torch.full((Q, C), -1, dtype=torch.long, device=dev)))
    ent = rank.gather(1, torch.where(ent0 >= 0, ent0, torch.full_like(ent0, C)))
    st = _entity_stats(r, dirs, f, ent, C)
    if "m" in r:
        st["mass"] = r["m"].float()[:, None] * st["share"] * st["live"]
    return st | {"label": ent}


def _comp_mean(v: torch.Tensor, w: torch.Tensor, ent: torch.Tensor, C: int) -> torch.Tensor:
    Q, D = w.shape; e = torch.where(ent >= 0, ent, torch.full_like(ent, C))
    s = torch.zeros(Q, C + 1, v.shape[-1], device=v.device).scatter_add_(1, e[..., None].expand(-1, -1, v.shape[-1]), w[..., None] * v)
    n = torch.zeros(Q, C + 1, device=v.device).scatter_add_(1, e, w)
    return (s / n.clamp_min(1e-12)[..., None])[:, :C]


def _entity_stats(r: dict, dirs: torch.Tensor, f: torch.Tensor, ent: torch.Tensor, C: int) -> dict:
    """Per (point, entity): the votes' Bingham (their moments averaged, refitted); the mass -- the entity's density over
    its directions against its voted Bingham's (``sum f / sum p``: the grid's sampling cancels, so sharp entities are
    weighed right at any orientation); its share; the entity's own density moment and that moment's Bingham; the mean
    embedding (projected into the ball) and descriptors."""
    Q, D = f.shape; dev = f.device
    wm = torch.where(ent >= 0, f, torch.zeros_like(f)); e = torch.where(ent >= 0, ent, torch.full_like(ent, C))
    Mv = BG.moments(r["R"].double(), r["kappa"].double()).float().reshape(Q, D, 9)
    uu = (dirs[:, :, None] * dirs[:, None, :]).reshape(1, D, 9).expand(Q, -1, -1)
    mv = _comp_mean(Mv, wm, ent, C).reshape(Q, C, 3, 3); mo = _comp_mean(uu, wm, ent, C).reshape(Q, C, 3, 3)
    zm = _comp_mean(r["z"], wm, ent, C); n = zm.norm(dim=-1, keepdim=True); zm = torch.where(n >= 0.9999, zm / n * 0.9999, zm)
    dm = _comp_mean(r["desc"], wm, ent, C)
    fs = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, e, wm)[:, :C]
    live = fs > 0
    R = torch.eye(3, device=dev).expand(Q, C, 3, 3).clone(); K = torch.zeros(Q, C, 2, device=dev)
    Ro, Ko = R.clone(), K.clone()
    if bool(live.any()):
        a, b = BG.fit(mv[live].double()); R[live], K[live] = a.float(), b.float()
        a, b = BG.fit(mo[live].double()); Ro[live], Ko[live] = a.float(), b.float()
    ec = e.clamp(max=C - 1)
    Rd = R.gather(1, ec[..., None, None].expand(-1, -1, 3, 3)); Kd = K.gather(1, ec[..., None].expand(-1, -1, 2))
    p = BG.pdf(dirs[None].expand(Q, -1, -1), Rd, Kd) * (ent >= 0)                                   # unit mean over the sphere
    ps = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, e, p)[:, :C]
    mass = torch.where(ps > 1e-6, fs / ps.clamp_min(1e-12), fs / D) * live
    share = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)                                       # identifiable under partial volume
    return {"mass": mass, "share": share, "R": R, "kappa": K, "R_odf": Ro, "kappa_odf": Ko, "z": zm, "desc": dm, "live": live}


def hierarchy(z: torch.Tensor):
    """Average-linkage clustering of entities ``z (N, E)`` by hyperbolic distance (scipy linkage matrix): cut at any
    distance for the groups at that depth (``scipy.cluster.hierarchy.fcluster(L, t, "distance")``)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    d = poincare_dist(z[:, None].double(), z[None].double()).cpu().numpy(); np.fill_diagonal(d, 0.0)
    return linkage(squareform(d, checks=False), "average")


def link(points: torch.Tensor, ents: dict, radius: float = 2.5, z_link: float = 0.4 * TREE_EDGE, ang_max: float = 25.0,
         depth_tol: float = 0.5 * TREE_EDGE) -> torch.Tensor:
    """Object identities across points: entity ``c`` at point ``i`` and ``c'`` at a point within ``radius`` voxels are
    candidates when their embeddings lie within ``z_link`` and their mode axes within ``ang_max`` degrees, and each is
    the other's best match there (mutual best). Candidates are joined strongest first (union-find), never joining two
    objects that share a point -- entities seen apart at one point stay apart, so a chain of weak links through a
    crossing cannot merge the crossing objects -- and never joining entities at different depths of the tree (the
    distance of the embedding from the ball's centre, within ``depth_tol``): between two parallel objects their parent
    is the entity, one level up, and cannot chain them. ``z_link`` is below half a tree edge. ``ents`` from
    :func:`cluster` at ``points (Q, 3)``. Returns ``(Q, C)`` ids (-1 none)."""
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
        dep = 2 * torch.atanh(z.norm(dim=-1).clamp(max=1 - 1e-5))                                    # (Q, C) depth: distance from the centre
        ok = (dz < z_link) & (ca > math.cos(math.radians(ang_max))) & ((dep[i][:, :, None] - dep[j][:, None, :]).abs() < depth_tol)
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
           z_max: float = 0.4 * TREE_EDGE, z_drift: float = 0.8 * TREE_EDGE, ang_max: float = 35.0, n_dirs: int = 300, core: int = 64, halo: int | None = None,
           voxel_um: float | None = None, **cluster_kw) -> dict:
    """Seeded following: from ``seed`` (voxels), the entity nearest ``direction``; then step along its axis, re-query,
    keep the entity whose embedding lies within ``z_max`` of the last and within ``z_drift`` of the running mean along
    the path (no unbounded drift), and whose axis is within ``ang_max`` degrees, until none does or the volume ends. The trunk is computed on a tile around the current point as needed. Returns
    ``points (n, 3)``, ``axes``, ``z``, ``mass``."""
    dev = x.device; dirs = sphere_dirs(n_dirs, dev); halo = halo_for(model) if halo is None else halo; voxel = _voxel(voxel, voxel_um)
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
        zbar = torch.stack(zs).mean(0); zbar = zbar / zbar.norm().clamp_min(1.0 / 0.999) if float(zbar.norm()) >= 0.999 else zbar
        dz = poincare_dist(e["z"][0], zc[None]); ca = (e["R"][0, :, :, 2] @ ac).abs()
        ok = e["live"][0] & (dz < z_max) & (poincare_dist(e["z"][0], zbar[None]) < z_drift) & (ca > math.cos(math.radians(ang_max)))
        if not bool(ok.any()):
            break
        c = int(torch.where(ok, -dz, torch.full_like(dz, -1e9)).argmax())
        a_ = e["R"][0, c, :, 2]; ac = a_ if float(a_ @ ac) >= 0 else -a_; zc = e["z"][0, c]
        pts.append(p); axes.append(ac); zs.append(zc); ms.append(e["mass"][0, c])
    return {"points": torch.stack(pts[:len(axes)]), "axes": torch.stack(axes), "z": torch.stack(zs), "mass": torch.stack(ms)}
