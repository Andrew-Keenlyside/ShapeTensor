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
import re

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


def parse_options(s: str) -> dict:
    """Readout options from ``"saddle=0.9,lobe_merge_deg=15,sharpen_to=None"`` (numbers, None, True / False, words)."""
    kw = {}
    for kv in filter(None, s.split(",")):
        k, x = kv.split("=")
        kw[k] = None if x == "None" else (x == "True") if x in ("True", "False") else float(x) if re.fullmatch(r"-?[0-9.]+(e-?[0-9]+)?", x) else x
    return kw


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
    fe = (h0, h1, model.condition(h, v))
    if getattr(getattr(model, "ent", None), "oprobe", False):                                      # the oriented probe's scalar fields
        from .model import with_scal
        fe = with_scal(fe, model.probe_fields(xb, h0))
    if getattr(getattr(model, "ent", None), "odirect", 0):                                        # the direct oriented path's energies
        from .model import with_oe
        fe = with_oe(fe, model.direct_levels(xb))
    if getattr(getattr(model, "ent", None), "gate", None) is not None or getattr(getattr(model, "ent", None), "st_gain", None) is not None or getattr(getattr(model, "ent", None), "dens_ce", False):
        from .model import with_st6, st_products                                                  # the structure tensor (the gate's cue, the
        fe = with_st6(fe, st_products(xb))                                                        # direct path)
    return fe


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


_MAPS: dict = {}                                                                                 # the last features' maps (query)


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
    from .model import window_mean
    tsz = [int(min(n, v - o)) for n, v, o in zip(feats[0].shape[-3:], vs.tolist(), off.tolist())]       # the tile (no far-face padding)
    hit = _MAPS.get("key")
    if hit is not None and hit[0] is feats[0] and hit[1] is model and hit[2] == tsz:              # the maps of these features: once (readout
        reg, bgm, objm, spm = _MAPS["maps"]                                                         # queries a slice in chunks)
    else:
        if getattr(model, "mass_region", False) and getattr(model, "region", None) is not None:     # the mass: the region map's window mean
            reg = model.region_value(model.region_map(*feats))[:, :tsz[0], :tsz[1], :tsz[2]]
        bgm = model.bg_map(*feats)[:, :tsz[0], :tsz[1], :tsz[2]] if getattr(model, "bgh", None) is not None else None   # the background's
        objm = model.obj_map(*feats)[:, :tsz[0], :tsz[1], :tsz[2]] if getattr(model, "objh", None) is not None else None  # the objects'
        spm = model.split_map(*feats)[:, :, :tsz[0], :tsz[1], :tsz[2]] if getattr(model, "splith", None) is not None else None  # by kind
        if not torch.is_grad_enabled():
            _MAPS["key"] = (feats[0], model, tsz); _MAPS["maps"] = (reg, bgm, objm, spm)
    for a in range(0, points.shape[0], chunk):
        p = points[a:a + chunk].float(); s = sigma[a:a + chunk].float(); n = p.shape[0]; D = dirs.shape[0]
        ctx = torch.minimum(p, vs - 1 - p).amin(-1)
        enc = dec.encode(feats, (p - off)[None], s[None], ctx[None])[0]                            # (n, hidden): once per point
        mq = dec.mass_of(enc) if getattr(dec, "mass", None) is not None else None                  # the mass output (where), if the model has one
        if reg is not None:
            mq = window_mean(reg, (p - off)[None], s[None])[0]
        bq = window_mean(bgm, (p - off)[None], s[None])[0] if bgm is not None else None
        oq = window_mean(objm, (p - off)[None], s[None])[0] if objm is not None else None
        sq = [window_mean(spm[:, c], (p - off)[None], s[None])[0] for c in range(2)] if spm is not None else None
        if t_rel is None:
            ev = torch.ones(n, D, dtype=torch.bool, device=p.device)
        else:
            odf = total_odf(model, feats, p, dirs, offset)
            ev = (odf >= t_rel * odf.amax(1, keepdim=True)) & (odf >= t_abs)
        qi, di = torch.nonzero(ev, as_tuple=True)
        rr = dec.ray_pairs(feats, (p - off)[qi], dirs[di]) if (getattr(dec, "ray", False) and qi.numel()) else None
        oo = dec.oprobe_feats(feats, (p - off)[qi], dirs[di]) if (getattr(dec, "oprobe", False) and qi.numel()) else None
        od = None; ovp = dec.od_sample(feats, p - off) if getattr(dec, "odirect", 0) else None  # the bank sampled once per point
        if ovp is not None and qi.numel():
            od = dec.od_code(ovp, dec.od_weights(dirs), pairs=(qi, di))
        dc = dec.dens(enc)[qi] if (getattr(dec, "dens", None) is not None and qi.numel()) else None   # (per point, then gathered)
        st6 = getattr(feats, "st6", None); j6 = None; st_q = None
        if st6 is not None and (getattr(dec, "gate", None) is not None or getattr(dec, "st_gain", None) is not None or getattr(dec, "dens_ce", False)):
            j6 = window_mean(st6[0], (p - off)[None].expand(6, -1, -1).contiguous(), s[None].expand(6, -1).contiguous()).T   # (n, 6)
            if getattr(dec, "st_gain", None) is not None and qi.numel():
                from .model import st_matrix
                Mn = st_matrix(j6)[qi]; u = dirs[di].float(); st_q = torch.einsum("ni,nij,nj->n", u, Mn, u)
        r = dec.decode(enc[qi], Y[di].to(enc.dtype), rr, oo, od, dens_c=dc, st_q=st_q) if qi.numel() else None
        full = {"f": torch.zeros(n, D, device=p.device), "z": torch.zeros(n, D, dec.emb, device=p.device),
                "R": torch.eye(3, device=p.device).expand(n, D, 3, 3).clone(), "kappa": torch.zeros(n, D, 2, device=p.device),
                "desc": torch.zeros(n, D, 5, device=p.device)}
        if getattr(dec, "mvote", None) is not None:
            full["mvote"] = torch.zeros(n, D, device=p.device)
        if r is not None:
            for k in full:
                full[k][qi, di] = r[k].float()
        if getattr(dec, "gate", None) is not None:                                                 # gated density: g x the lobe + (1 - g) x flat
            from .model import st_features, st_features_ext
            stf = (st_features_ext(j6, s) if getattr(dec, "gate2", False) else st_features(j6)) if j6 is not None else None
            res = dec.gate_of(enc, ovp, stf, parts=True); g, gv = res[0], res[1]; fm = (full["f"] * ev).sum(1) / ev.sum(1).clamp_min(1); full["gv"] = gv
            full["f_lobe"] = full["f"]; full["f"] = g[:, None] * full["f"] + (1 - g[:, None]) * fm[:, None]; full["g"] = g
        if getattr(dec, "pop_k", 0):                                                                # the populations (pop_k)
            pu = None
            if getattr(dec, "pop_dir", 0):
                U = dec.pop_quadrature(1, p.device)[0]; cu = dec.od_code(ovp, dec.od_weights(U))
                if getattr(dec, "oprobe", False):
                    ou = dec.oprobe_feats(feats, (p - off)[:, None].expand(-1, U.shape[0], -1).reshape(-1, 3), U[None].expand(n, -1, -1).reshape(-1, 3))
                    cu = torch.cat([cu, ou.view(n, U.shape[0], -1).to(cu.dtype)], -1)
                pu = (U, cu)
            full.update(dec.populations(enc, pu))
        if getattr(dec, "dens_ce", False) and j6 is not None:                                       # the normalised density (dens_ce)
            ce = dec.ce_density(enc, ovp, j6, s, dirs=dirs)
            full["f_dir"] = full["f"]; full["f"] = torch.exp(ce["logq"]); full["logq"] = ce["logq"]
            full["logq_g"] = ce["logq_g"]; full["w0"] = ce["w0"]
        full["evaluated"] = ev
        if mq is not None:
            full["m"] = mq.float()
        if bq is not None:
            full["bgm"] = bq.float()
        if oq is not None:
            full["objm"] = oq.float()
        if sq is not None:
            full["orim"], full["isom"] = sq[0].float(), sq[1].float()
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


_SYMNB: dict = {}


def sym_neighbours(dirs: torch.Tensor, k: int = 8) -> torch.Tensor:
    """``(D, kk)`` neighbour lists of the direction grid made symmetric (k nearest, plus every direction that counts
    this one among its k nearest), padded with -1. Cached per grid (it took ~200 host syncs per readout chunk)."""
    key = (dirs.data_ptr(), tuple(dirs.shape), str(dirs.device), k, float(dirs.sum()))
    hit = _SYMNB.get(key)
    if hit is not None:
        return hit
    if len(_SYMNB) > 16:
        _SYMNB.clear()
    out = _SYMNB[key] = _sym_neighbours(dirs, k)
    return out


def _sym_neighbours(dirs: torch.Tensor, k: int) -> torch.Tensor:
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
            core_frac: float = 0.6, share_min: float = 0.03, k: int = 8, na: int = 128, cmax: int = 12, chunk: int = 256,
            floor: float | None = None) -> dict:
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
    (heaviest first) as :func:`_entity_stats`, and ``label (Q, D)`` (each direction's entity, -1 none).

    ``floor`` (a quantile, e.g. 0.05): the density's floor -- that quantile over the directions, the isotropic background
    by definition -- taken off first: entities form only from what rises above it (the background's directions vote
    nothing to trust) and their mass is the point's mass less the floor's; ``floor_mass (Q,)`` the floor's."""
    f, z = r["f"], r["z"]; Q, D = f.shape; dev = f.device
    floor_frac = None
    if floor is not None:
        f0 = f.float(); fl = torch.quantile(f0, float(floor), dim=1, keepdim=True)
        floor_frac = (fl[:, 0] / f0.mean(1).clamp_min(1e-12)).clamp(0, 1)
        f = (f0 - fl).clamp_min(0)
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
        st["mass"] = r["m"].float()[:, None] * st["share"] * st["live"] * (1 - floor_frac[:, None] if floor_frac is not None else 1)
    if floor_frac is not None:                                                                     # (less the floor's)
        st["floor_mass"] = (r["m"].float() if "m" in r else r["f"].float().mean(1)) * floor_frac
    return st | {"label": ent}


def cluster_lobes(r: dict, dirs: torch.Tensor, f_frac: float = 0.25, f_abs: float = 1e-4, merge_deg: float = 25.0,
                  share_min: float = 0.03, k: int = 8, cmax: int = 12, saddle: float | None = None, saddle_ring: float | None = None,
                  saddle_peaks: float | None = None) -> dict:
    """Entities per point from the density's lobes alone: each living direction (density >= ``f_frac`` of the mean)
    climbs the density over the direction grid (steepest ascent among its ``k`` nearest neighbours, pointer jumping) to
    a local maximum; a maximum's basin is an entity; maxima within ``merge_deg`` of a heavier one join it; the
    ``cmax`` heaviest kept, those below ``share_min`` dropped. Where the votes do not (yet) switch between the lobes
    of a crossing, the density's own peaks still separate its entities. Returns :func:`cluster`'s fields.

    ``saddle``: basins whose shared boundary's highest point is at least ``saddle`` of the lower peak are joined
    (single linkage). ``saddle_peaks``: and only if the lower peak is at least ``saddle_peaks`` of the higher -- the
    valley must be shallow seen from both peaks (a ridge of comparable peaks: a sheet's ring, a fan's arc), so a weak
    population standing on a strong one's flank (a shoulder: little prominence of its own, but a low peak) stays its own."""
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
    if saddle is not None:                                                                         # basins joined across a high saddle: a girdle's
        lab0 = torch.where(alive, ent0, torch.full_like(ent0, -1))                                  # ridge has many peaks (~4 basins per point on a
        la, lb = lab0[:, :, None].expand(-1, -1, nb.shape[1]), lab0[:, nbc]                         # sheet), a crossing's lobes a deep valley
        fa, fb = f[:, :, None].expand(-1, -1, nb.shape[1]), f[:, nbc]                               # between them (review 27 Oct: girdle saddles ~0.9
        edge = (la >= 0) & (lb >= 0) & (la != lb) & (nb >= 0)[None]                                 # of the lower peak; 45-deg crossings <= 0.75)
        sv = torch.where(edge, torch.minimum(fa, fb), torch.zeros_like(fa))
        S = torch.zeros(Q, C * C, device=dev).scatter_reduce_(1, (la.clamp_min(0) * C + lb.clamp_min(0)).reshape(Q, -1), sv.reshape(Q, -1), "amax")
        S = S.view(Q, C, C); S = torch.maximum(S, S.transpose(1, 2))
        pkv = torch.full((Q, C + 1), 0.0, device=dev).scatter_reduce_(1, torch.where(lab0 >= 0, lab0, torch.full_like(lab0, C)), torch.where(lab0 >= 0, f, torch.zeros_like(f)), "amax")[:, :C]
        lo, hi = torch.minimum(pkv[:, :, None], pkv[:, None, :]), torch.maximum(pkv[:, :, None], pkv[:, None, :])
        adj = (S >= saddle * lo) & (S > 0)
        if saddle_peaks is not None:                                                               # comparable peaks only: a weak lobe on a strong
            adj &= lo >= saddle_peaks * hi                                                         # one's flank is a shoulder, not a ridge's peak
        adj |= torch.eye(C, dtype=torch.bool, device=dev)[None]
        comp = _components(adj)                                                                     # (Q, C): the smallest slot of each group
        if saddle_ring is not None:                                                                 # a group is kept only if its directions make a
            g0 = torch.where(lab0 >= 0, comp.gather(1, lab0.clamp_min(0)), lab0)                   # ring (a sheet's ridge: in-plane eigenvalue ratio
            uu = (dirs[:, :, None] * dirs[:, None, :]).reshape(1, D, 9).expand(Q, -1, -1)         # >= saddle_ring); two bundles joined across a
            Mg = _comp_mean(uu, torch.where(g0 >= 0, f, torch.zeros_like(f)), g0, C).reshape(Q, C, 3, 3)   # saddle are not one (review 6: the
            ev = torch.linalg.eigvalsh(0.5 * (Mg + Mg.transpose(-1, -2)))                          # merge absorbed weak crossing populations)
            ring_ok = ev[..., 1] >= saddle_ring * ev[..., 2].clamp_min(1e-12)                      # (Q, C) per group slot
            many = (torch.zeros(Q, C, device=dev).scatter_add_(1, comp, torch.ones_like(comp, dtype=torch.float)) > 1)   # merged groups
            keep = ring_ok | ~many
            comp = torch.where(keep.gather(1, comp), comp, torch.arange(C, device=dev)[None].expand(Q, C))
        ent0 = torch.where(ent0 >= 0, comp.gather(1, ent0.clamp_min(0)), ent0)
    cm = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, torch.where(ent0 >= 0, ent0, torch.full_like(ent0, C)), w)[:, :C]
    o2 = cm.argsort(1, descending=True); keepc = (cm.gather(1, o2) >= share_min * cm.sum(1, keepdim=True).clamp_min(1e-12)) & (cm.gather(1, o2) > 0)
    rank = torch.full((Q, C + 1), -1, dtype=torch.long, device=dev)
    rank.scatter_(1, o2, torch.where(keepc, torch.arange(C, device=dev)[None].expand(Q, C), torch.full((Q, C), -1, dtype=torch.long, device=dev)))
    ent = rank.gather(1, torch.where(ent0 >= 0, ent0, torch.full_like(ent0, C)))
    st = _entity_stats(r, dirs, f, ent, C)
    if "m" in r:
        st["mass"] = r["m"].float()[:, None] * st["share"] * st["live"]
    return st | {"label": ent}


def cluster_lobes_floor(r: dict, dirs: torch.Tensor, floor: float | None = None, **kw) -> dict:
    """:func:`cluster_lobes` on the density less its floor (the ``floor`` quantile over the directions, as
    :func:`cluster`'s): the lobes form only from what rises above the isotropic background."""
    if floor is None:
        return cluster_lobes(r, dirs, **kw)
    f0 = r["f"].float(); fl = torch.quantile(f0, float(floor), dim=1, keepdim=True)
    return cluster_lobes({**r, "f": (f0 - fl).clamp_min(0)}, dirs, **kw)


def sharpen_density(f: torch.Tensor, dirs: torch.Tensor, kappa: float = 7.5, iters: int = 30, kappa_to: float | None = 17.0,
                    floor: float = 0.05) -> torch.Tensor:
    """The density ``f (Q, D)`` above its floor (the ``floor`` quantile) deconvolved by the model's own single-bundle
    lobe -- a Watson exp(-``kappa`` sin^2) -- in ``iters`` Richardson-Lucy steps (non-negative, mass kept), then redrawn
    with lobes of ``kappa_to`` (None: left as deconvolved). The overnight pilot draws one bundle ~30 deg wide at half
    height at any window (kappa ~7.5, readout_xing.py, 27 Oct) against its training targets' ~17 (the phantom
    families' conc 0.94; the scene generator's reach 40), so two bundles 25-30 deg apart sum to one lobe; deconvolved
    they are two."""
    fl = torch.quantile(f.float(), float(floor), dim=1, keepdim=True)
    h = (f.float() - fl).clamp_min(0); c2 = (dirs.float() @ dirs.float().T) ** 2                   # (antipodes identified)
    H = torch.exp(-kappa * (1 - c2)); H = H / H.sum(0, keepdim=True)                               # columns: a unit lobe at each direction
    w = h.clone() + 1e-12
    for _ in range(int(iters)):
        w = w * ((h / (w @ H.T).clamp_min(1e-12)) @ H)
    if kappa_to is not None:
        Ht = torch.exp(-kappa_to * (1 - c2)); w = w @ (Ht / Ht.sum(0, keepdim=True)).T
    return w


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
    lnC = BG.log_normaliser(K)                                                                      # (Q, C): once per entity, gathered
    a_, b_ = Rd[..., :, 0], Rd[..., :, 1]; u = dirs[None].expand(Q, -1, -1)                         # (= BG.pdf per direction: review 5)
    q = Kd[..., 0] * (u * a_).sum(-1) ** 2 + Kd[..., 1] * (u * b_).sum(-1) ** 2
    p = torch.exp(-q - lnC.gather(1, ec).to(q.dtype)) * (ent >= 0)                                  # unit mean over the sphere
    ps = torch.zeros(Q, C + 1, device=dev).scatter_add_(1, e, p)[:, :C]
    mass = torch.where(ps > 1e-6, fs / ps.clamp_min(1e-12), fs / D) * live
    share = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)                                       # identifiable under partial volume
    return {"mass": mass, "share": share, "R": R, "kappa": K, "R_odf": Ro, "kappa_odf": Ko, "z": zm, "desc": dm, "live": live}


def split_mass(r: dict, dirs: torch.Tensor, e: dict, kappa_max: float = 40.0, iters: int = 300) -> dict:
    """The point's mass split among its entities by their lobes, not by direction: the density (scaled to the point's
    mass: ``r["m"]``, else its sphere mean) fitted as a non-negative sum of each entity's voted Bingham (concentrations
    capped at ``kappa_max``, as the density's targets) and a uniform floor (multiplicative NNLS updates). Where a lobe
    stands on the isotropic floor, the floor under it stays the floor's -- assigning whole directions gives it to the
    lobe. Returns ``mass (Q, C)`` (the entities') and ``floor (Q,)`` (the uniform component's)."""
    f = r["f"].float(); Q, D = f.shape; dev = f.device
    m_tot = r["m"].float() if "m" in r else f.mean(1)
    y = f / f.mean(1, keepdim=True).clamp_min(1e-12) * m_tot[:, None]                             # mean over the sphere: the mass
    live = e["live"]; C = live.shape[1]
    Phi = BG.pdf(dirs[None, :, None], e["R"][:, None], e["kappa"][:, None].clamp(max=kappa_max)) * live[:, None]   # (Q, D, C)
    Phi = torch.cat([Phi, torch.ones(Q, D, 1, device=dev)], -1)                                     # + the uniform floor
    ok = torch.cat([live, torch.ones(Q, 1, dtype=torch.bool, device=dev)], 1)
    m = ok.float() * (m_tot / ok.sum(1).clamp_min(1))[:, None]
    PtY = (Phi * y[..., None]).sum(1); PtP = Phi.transpose(1, 2) @ Phi
    for _ in range(iters):
        m = m * PtY.clamp_min(0) / (PtP @ m[..., None])[..., 0].clamp_min(1e-12)
    m = m * ok
    return {"mass": m[:, :C], "floor": m[:, C]}


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


def merge_parallel(e: dict, mass: torch.Tensor, deg: float = 15.0, iso_k: float = 0.5) -> tuple:
    """Entities of one type (peaked: kappa2 >= 2; girdle: kappa2 < 2 <= kappa1, by its normal) within ``deg`` of a
    heavier one at the same point are one population: their masses join the heavier one's (its frame kept). At a point
    two lobes along one axis cannot be told apart by the density -- a split there is the readout's, not the material's
    (85 deg phantom crossing: ~4 entities, the heaviest 39 %). Returns ``(mass, live)``."""
    k = e["kappa"].float(); R = e["R"].float(); live = e["live"] & (mass > 0)
    typ = torch.where(k[..., 0] < iso_k, 2, torch.where(k[..., 1] < 2.0, 1, 0))
    ax = torch.where((typ == 1)[..., None], R[..., :, 0], R[..., :, 2])
    m = mass.clone(); lv = live.clone(); C = m.shape[1]
    order = torch.argsort(m, 1, descending=True)
    cosd = math.cos(math.radians(deg))
    for a_ in range(C):                                                                              # heaviest first: absorb the lighter
        i = order[:, a_]
        ai = ax.gather(1, i[:, None, None].expand(-1, 1, 3))[:, 0]; ti = typ.gather(1, i[:, None])[:, 0]
        li = lv.gather(1, i[:, None])[:, 0]
        near = ((ax * ai[:, None]).sum(-1).abs() >= cosd) & (typ == ti[:, None]) & (ti[:, None] < 2) & lv & li[:, None]
        near.scatter_(1, i[:, None], False)
        add = (m * near).sum(1)
        m.scatter_add_(1, i[:, None], add[:, None]); m = torch.where(near, torch.zeros_like(m), m); lv = lv & ~near
    return m, lv


@torch.no_grad()
def readout(model, image: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, voxel_um=None, floor: float = 0.05,
            iso_k: float = 0.5, merge_deg: float | None = 15.0, chunk: int = 1024, feats=None, tta: int = 0, **kw) -> dict:
    """The entity readout at ``points``: the decoder queried at every direction of ``dirs`` (:func:`query`, or
    :func:`query_tta` with ``tta`` rotations), read out by :func:`entities_from_query` (``floor``, ``iso_k``,
    ``merge_deg`` and its options ``kw``), in chunks of ``chunk`` points."""
    f = feats if (feats is not None or tta > 1) else features(model, image, voxel_um=voxel_um)
    parts = []
    for a0 in range(0, points.shape[0], chunk):
        sl = slice(a0, a0 + chunk)
        r = (query_tta(model, image, points[sl], sigma[sl], dirs, n_rot=tta, voxel_um=voxel_um) if tta > 1     # ``tta``: rotations averaged
             else query(model, f, points[sl], sigma[sl], dirs, vol_shape=image.shape))
        parts.append(entities_from_query(r, dirs, floor=floor, iso_k=iso_k, merge_deg=merge_deg, **kw))
    C = max(p_["mass"].shape[1] for p_ in parts)
    pad = lambda x: torch.cat([x, torch.zeros((x.shape[0], C - x.shape[1]) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)], 1) if x.shape[1] < C else x
    return {k: torch.cat([pad(p_[k]) if k in ("mass", "k", "R", "cover", "sheet") else p_[k] for p_ in parts]) for k in parts[0]}


@torch.no_grad()
def entities_from_query(r: dict, dirs: torch.Tensor, floor: float | None = 0.05, iso_k: float = 0.5, merge_deg: float | None = 15.0,
                        entities: str = "lobes", axes: str = "ring", kappa: str = "votes", prune: float = 0.0, mass: str = "basins",
                        saddle: float | None = 0.8, maps_cap: float | None = None, ring: float = 0.8,
                        saddle_ring: float | None = None, lobe_frac: float = 0.1, weak_lines: bool = True, saddle_peaks: float | None = None,
                        ring_soft: float = 0.1, lobe_merge_deg: float = 25.0, sharpen: int = 0, sharpen_kappa: float = 7.5,
                        sharpen_to: float | None = 17.0) -> dict:
    """The entities from the decoder's outputs ``r`` at the directions ``dirs`` (:func:`query`): the density's floor
    (``floor`` quantile) the background plus isotropic objects (told apart by the background head where the model has
    one), entities from what rises above it, their mass split by lobe fit (:func:`split_mass`), near-parallel same-type
    entities merged (``merge_deg``, None: off). Returns ``mass (P, C)`` (oriented entities; isotropic ones zeroed),
    ``k``, ``R``, ``iso`` (isotropic-object mass), ``bg`` (background mass), ``bgm`` (the head's window mean),
    ``iso_z`` (the floor directions' mean embedding: the isotropic material's place in the tree), ``cover`` (``axes="ring"``:
    each entity's in-plane eigenvalue ratio) and ``sheet`` (each entity's sheet weight in [0, 1], for maps).

    ``entities``: "votes" (:func:`cluster`: the directions grouped by their votes and embeddings) or "lobes"
    (:func:`cluster_lobes_floor`: the density's own lobes). ``axes``: "votes" (each entity's frame and type from its
    directions' averaged votes), "density" (from the entity's own density moment) -- used for the types, the mass
    split and the merging alike -- or "ring": density frames, and a girdle only where the entity's density covers its
    great circle -- the moment's in-plane eigenvalue ratio (middle / largest; 1 a full ring, ~0.4 a 120-deg sector) at
    least ``ring`` -- else peaked: a fan's or a curve's lobe is broader than its target and its moment often falls
    below the girdle line (kappa2 < 2) while covering only a sector; the mass split keeps the moment's own Bingham.
    ``kappa``: "moments" (the Bingham fitted to the averaged vote moments: broader than any single vote where the
    votes' frames scatter) or "votes" (the density-weighted mean of the directions' own
    concentration votes, in log1p). ``prune`` (> 0): oriented entities whose lobe stands less than ``prune`` times the
    point's floor above it (the peak of the floor-subtracted density over the floor) are dropped before the mass split
    -- ripples of the floor, not populations. ``mass``: "maps" (with a split head: the oriented material's mass from
    its map, shared among the oriented entities by their lobes; the isotropic objects' from theirs) or "density" (the
    lobe fit's floor less the background head's).

    Defaults (readout_lab.py, selection set): entities from the density's lobes (count 7 -> 10 of 16 on the direct-path
    pilot), the votes' own concentrations reported (ODF shape 4 -> 16 of 16), frames and types from the density's own
    moment with lobes joined across high saddles (``saddle``: a sheet's ridge stays one entity). The votes lag the density
    early in training: at step 2000 of the overnight pilot, density frames matched 0.80 / 0.83 / 1.06 of crossings' /
    curves' / sheets' mass against 0.20 / 0.01 / 0.00 with the votes' (count 5 -> 13, direction 11 -> 15 of 16); on the
    long-trained direct-path pilot they were neutral to better (count 10 -> 12, direction 12 -> 16). ``axes="ring"``
    (the default since 28 Oct): density frames, but a lobe the moment calls a girdle stays one only if its density
    covers most of the great circle (in-plane eigenvalue ratio >= ``ring``) -- a fan's spread lobe is otherwise a
    line: fans found 0.2-0.6 -> 0.7-1.0, sheets kept (1.0), all criteria 2 -> 3 (direct-path pilot) and 3 -> 4
    (overnight pilot, step 2000) of 16. ``mass="basins"`` (the default since 27 Oct): each oriented entity's share of the
    oriented-material map is its directions' density above the floor (a basin integral), so masses move continuously with
    the density -- a lobe is born with ~0 mass and no fitted shapes reassign mass when another appears (the "maps" split
    jumped where the population count or a lobe's fit changed); the criteria unchanged (7/16, mass level 8 -> 9).
    ``sheet``: the type as a weight for maps -- ``axes="ring"``: a smoothstep of the cover over ``ring`` +- ``ring_soft``
    (0.5 at the type boundary; 0: the hard type). The cover is a smooth field on real tissue while the types cut it at
    ``ring``, so a map drawing lines by their axis and sheets by their normal jumped 90 deg there (27 Oct); the types
    themselves (``k``) are unchanged. ``lobe_merge_deg``: density maxima within it of a heavier one join it (25; 15 with
    ``saddle_ring=0.6`` resolved more 30-60 deg crossings on the tuning set but split fans on the selection set, so not
    the default -- notes/readout_crossings_2026-10-27.md). ``sharpen``: entities from the density deconvolved by the
    model's single-bundle lobe (:func:`sharpen_density`, that many steps; off) -- narrower crossings, but fans split and
    sheets lost."""
    for k_, v_, ok in (("entities", entities, ("votes", "lobes")), ("axes", axes, ("votes", "density", "hybrid", "ring")), ("kappa", kappa, ("moments", "votes", "votes_all")),
                       ("mass", mass, ("maps", "density", "votes", "maps_votes", "basins"))):
        if v_ not in ok:
            raise ValueError(f"entities_from_query: {k_}={v_!r}, not one of {ok}")
    r_raw = r
    if sharpen > 0:                                                                               # entities, frames and basin shares from the
        r = {**r, "f": sharpen_density(r["f"], dirs, sharpen_kappa, sharpen, sharpen_to, floor if floor is not None else 0.05)}   # sharpened density
    e = cluster(r, dirs, floor=floor) if entities == "votes" else cluster_lobes_floor(r, dirs, floor=floor, saddle=saddle, saddle_ring=saddle_ring, f_frac=lobe_frac, saddle_peaks=saddle_peaks,
                                                                                         merge_deg=lobe_merge_deg)
    if axes == "hybrid":                                                                          # frames from the density, types (and shapes)
        lab_ = e["label"]; C_ = e["live"].shape[1]                                                 # from the votes' own concentrations
        w_ = torch.where(lab_ >= 0, r["f"].float(), torch.zeros_like(r["f"].float()))
        kv_h = torch.expm1(_comp_mean(torch.log1p(r["kappa"].float().clamp_min(0)), w_, lab_, C_))
        kv_h = torch.stack([kv_h.max(-1).values, kv_h.min(-1).values], -1)
        e = {**e, "R": e["R_odf"], "kappa": torch.where(e["live"][..., None], kv_h, e["kappa_odf"])}
    elif axes == "density":
        e = {**e, "R": e["R_odf"], "kappa": e["kappa_odf"]}
    elif axes == "ring":                                                                          # density frames; a girdle only where the density
        Ko = e["kappa_odf"]; _, ea, eb = BG._stats(Ko)                                            # covers its great circle: the in-plane eigenvalue
        cover = (eb / (1.0 - ea - eb).clamp_min(1e-12)).to(Ko.dtype)                              # ratio of its moment (its fitted Bingham's, exactly)
        line = (Ko[..., 0] >= iso_k) & (Ko[..., 1] < 2.0) & (cover < ring)                       # a sector of it: peaked (the votes' kappa reported)
        sw = ((cover - ring + ring_soft) / (2 * ring_soft)).clamp(0, 1) if ring_soft > 0 else (cover >= ring).to(Ko.dtype)
        sw = sw * sw * (3 - 2 * sw) * (Ko[..., 0] >= iso_k)                                       # (the sheet weight for maps: smooth in the cover)
        e = {**e, "R": e["R_odf"], "kappa": torch.where(line[..., None], Ko.clamp_min(2.0), Ko), "kappa_fit": Ko, "cover": cover, "sheet": sw}
    k_rep = None
    if kappa in ("votes", "votes_all"):                                                          # the reported shape: the votes' own concentrations
        lab = e["label"]; C = e["live"].shape[1]
        w = torch.where(lab >= 0, r["f"].float(), torch.zeros_like(r["f"].float()))
        lk = _comp_mean(torch.log1p(r["kappa"].float().clamp_min(0)), w, lab, C)                    # (Q, C, 2)
        kv_ = torch.expm1(lk); kv_ = torch.stack([kv_.max(-1).values, kv_.min(-1).values], -1)
        k_rep = torch.where(e["live"][..., None], kv_, e["kappa"])
        if kappa == "votes_all":                                                                 # ... also for the types and the mass split
            e = {**e, "kappa": k_rep}
    if prune > 0:
        f0 = r["f"].float(); fl = torch.quantile(f0, float(floor if floor is not None else 0.05), dim=1, keepdim=True)
        lab = e["label"]; C = e["live"].shape[1]
        h = torch.where(lab >= 0, (f0 - fl), torch.full_like(f0, -1.0))
        peak = torch.full((f0.shape[0], C + 1), -1.0, device=f0.device).scatter_reduce_(1, torch.where(lab >= 0, lab, torch.full_like(lab, C)), h, "amax")[:, :C]
        orie = e["kappa"].float()[..., 0] >= iso_k
        weak = orie & (peak < prune * fl)
        e = {**e, "live": e["live"] & ~weak}
    sp = split_mass(r_raw, dirs, {**e, "kappa": e["kappa_fit"]} if "kappa_fit" in e else e)       # (ring: the lobes fitted with the moment's shape)
    ms = (sp["mass"] * e["live"]).float(); kv = e["kappa"].float(); isoe = kv[..., 0] < iso_k
    if mass == "basins" and weak_lines:                                                           # lobes too broad to type where the map holds
        weak = e["live"] & isoe                                                                   # oriented material: lines along their mode axis
        kv = torch.where(weak[..., None], kv.clamp_min(2.0), kv); isoe = kv[..., 0] < iso_k        # (typed isotropic, they left the oriented mass
        e = {**e, "kappa": kv}                                                                    # unresolved: a hard edge at bundles' rims)
    if mass == "basins":                                                                          # each oriented entity's share: its directions'
        f0 = r["f"].float(); fl = torch.quantile(f0, float(floor if floor is not None else 0.05), dim=1, keepdim=True)   # density above the floor
        h = (f0 - fl).clamp_min(0); lab = e["label"]; C = e["live"].shape[1]                      # (a basin integral: it moves continuously with
        bs = torch.zeros(f0.shape[0], C + 1, device=f0.device).scatter_add_(1, torch.where(lab >= 0, lab, torch.full_like(lab, C)), h)[:, :C]   # the
        ok = e["live"] & ~isoe                                                                     # density, and a lobe is born with ~0 mass --
        share = bs * ok / (bs * ok).sum(1, keepdim=True).clamp_min(1e-12)                         # no fitted lobe shapes, no reassignment when
        tot_o = r["orim"].float() if "orim" in r else (ms * ~isoe).sum(1)                         # another lobe appears: the hard edges, 27 Oct)
        ms = torch.where(ok, tot_o[:, None] * share, torch.zeros_like(ms))
    if merge_deg is not None:
        ms, lv = merge_parallel({**e, "live": e["live"] & ~isoe}, ms * ~isoe, merge_deg, iso_k)
        ms = ms + (sp["mass"] * e["live"]).float() * isoe                                       # (isotropic entities kept apart)
    iso_tot = sp["floor"].float() + (ms * isoe).sum(1)
    bgm = r["bgm"].float().clamp(0, 1) if "bgm" in r else iso_tot
    bg = torch.minimum(iso_tot, bgm)
    unres = torch.zeros_like(iso_tot)
    if mass == "basins":                                                                          # isotropic objects and background from their
        if "isom" in r:                                                                           # maps, as the maps readout
            bg = bgm if "bgm" in r else (iso_tot - r["isom"].float()).clamp_min(0); iso_tot = bg + r["isom"].float()
            unres = torch.where(((ms * ~isoe).sum(1) > 0), torch.zeros_like(iso_tot), r["orim"].float())
    elif mass in ("votes", "maps_votes"):                                                           # the entities' own mass votes (density-weighted
        if "mvote" not in r:                                                                     # over their directions; merged pieces: the
            raise ValueError("entities_from_query: mass='votes' needs a model with mass votes")  # survivor's -- a population's pieces vote alike)
        lab = e["label"]; C = e["live"].shape[1]; f0 = r["f"].float()
        fl = torch.quantile(f0, float(floor if floor is not None else 0.05), dim=1, keepdim=True)
        live_o = (ms > 0) & ~isoe
        typ = torch.where(kv[..., 1] < 2.0, 1, 0)                                                # pieces of one population (same type, axes
        axv = torch.where((typ == 1)[..., None], e["R"].float()[..., :, 0], e["R"].float()[..., :, 2])   # within the scorer's 30 deg) share one
        near = ((axv @ axv.transpose(1, 2)).abs() >= math.cos(math.radians(30.0))) & (typ[:, :, None] == typ[:, None, :])   # vote: the mean
        near &= live_o[:, :, None] & live_o[:, None, :]; near |= torch.eye(C, dtype=torch.bool, device=near.device)[None]  # over their union
        grp = _components(near)                                                                   # (Q, C): each piece's group (its smallest slot)
        lab_g = torch.where(lab >= 0, grp.gather(1, lab.clamp_min(0)), lab)
        w = torch.where(lab >= 0, (f0 - fl).clamp_min(0), torch.zeros_like(f0))                  # (the lobe above the floor, as clustered)
        mv_g = _comp_mean(r["mvote"].float()[..., None], w, lab_g, C)[..., 0]                    # (Q, C) per group slot
        msk = torch.where(live_o, ms, torch.full_like(ms, -1.0))                                  # the group's vote on its heaviest piece (ties:
        ar = torch.arange(C, device=ms.device)                                                    # the lowest slot)
        beats = (grp[:, :, None] == grp[:, None, :]) & ((msk[:, None, :] > msk[:, :, None]) | ((msk[:, None, :] == msk[:, :, None]) & (ar[None, None, :] < ar[None, :, None])))
        head = live_o & ~beats.any(-1)
        ms = torch.where(head, mv_g.gather(1, grp), torch.zeros_like(ms))
        if mass == "maps_votes" and "orim" in r:                                                 # shares from the votes, the total from the map
            S = ms.sum(1); ms = ms * (r["orim"].float() / S.clamp_min(1e-6)).clamp(max=4.0)[:, None]
            unres = (r["orim"].float() - ms.sum(1)).clamp_min(0)
        if "isom" in r:
            bg = bgm if "bgm" in r else (iso_tot - r["isom"].float()).clamp_min(0); iso_tot = bg + r["isom"].float()
    elif "orim" in r and mass == "maps":                                                             # the oriented material's and the isotropic
        ori = ms * ~isoe; S = ori.sum(1); Si = (ms * isoe).sum(1); has = S > 1e-6                # objects' mass from their maps: the lobes only
        scale = r["orim"].float() / (S + Si).clamp_min(1e-6)                                      # share the oriented mass out (a partner lobe
        new_ms = ori * scale[:, None]                                                             # typed isotropic keeps its share: review 2);
        if maps_cap is not None:                                                                  # optionally no more than maps_cap x the density
            new_ms = torch.minimum(new_ms, maps_cap * ori)                                        # split's (2x starved weak lobes early in training)
        ms = torch.where(isoe | ~has[:, None], torch.zeros_like(ms), new_ms)
        unres = (r["orim"].float() - ms.sum(1)).clamp_min(0)                                     # unexplained oriented mass: reported apart,
        bg = bgm if "bgm" in r else (iso_tot - r["isom"].float()).clamp_min(0)                    # never as isotropic objects
        iso_tot = bg + r["isom"].float()
    elif "objm" in r:                                                                            # the objects' mass predicted: the oriented
        obj = r["objm"].float(); ori = ms * ~isoe; S = ori.sum(1); has = S > 1e-6                # populations and the isotropic objects share
        iso_o = torch.minimum((iso_tot - bgm).clamp_min(0), obj)                                 # it -- isotropic objects the floor beyond the
        scale = torch.where(has, (obj - iso_o).clamp_min(0) / S.clamp_min(1e-6), torch.zeros_like(S))   # background (all of it where no lobe
        ms = torch.where(isoe, ms, ori * scale[:, None])                                         # stands); the background is the head's
        bg = bgm; iso_tot = bg + torch.where(has, iso_o, obj)
    un = (e["label"] < 0).float() * r_raw["f"].float()                                              # the floor's directions
    iso_z = (un[..., None] * r["z"].float()).sum(1) / un.sum(1, keepdim=True).clamp_min(1e-9)
    if k_rep is not None:                                                                       # reported: the votes' concentrations where they
        kr = k_rep.float().clamp_min(0)                                                          # keep the entity's type (peaked / girdle /
        typ = lambda k_: torch.where(k_[..., 0] < iso_k, 2, torch.where(k_[..., 1] < 2.0, 1, 0))   # isotropic: its frame was fitted for that type)
        kv = torch.where((typ(kr) == typ(kv))[..., None], kr, kv)
    return {"mass": ms * ~isoe, "k": kv, "R": e["R"].float(), "iso": (iso_tot - bg).clamp_min(0), "bg": bg, "bgm": bgm, "iso_z": iso_z,
            "ori_unres": unres, "cover": e["cover"].float() if "cover" in e else torch.zeros_like(ms),
            "sheet": (e["sheet"].float() if "sheet" in e else (kv[..., 1] < 2.0).float()) * ~isoe}


def cube_rotations(device="cpu") -> list:
    """The 24 proper rotations of the cube as ``(perm, signs, M)``: ``M[a, perm[a]] = signs[a]`` (z, y, x), identity first,
    then the quarter turns, then the rest -- a short prefix already spans the axes."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = torch.zeros(3, 3)
            for a_ in range(3):
                M[a_, perm[a_]] = sg[a_]
            if torch.det(M) > 0:
                out.append((list(perm), list(sg), M.to(device)))
    first, rest, seen = [], [], set()                                                                 # the identity, then one per axis permutation,
    for r in out:                                                                                     # then the rest
        k_ = tuple(r[0])
        if r[0] == [0, 1, 2] and r[1] == [1, 1, 1]:
            first.insert(0, r)
        elif k_ not in seen and k_ != (0, 1, 2):
            seen.add(k_); first.append(r)
        else:
            rest.append(r)
    return first + rest


@torch.no_grad()
def query_tta(model, image: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, n_rot: int = 4, voxel_um=None) -> dict:
    """:func:`query` averaged over ``n_rot`` exact cube rotations of the volume (points and directions rotated with it):
    the densities averaged, the votes averaged as Bingham moments (rotated back) and refitted; embeddings, masses from
    the unrotated pass. A test-time rotation average: an orientation the model handles worse is pulled toward the
    others' answer."""
    c = (torch.tensor(image.shape, device=points.device, dtype=torch.float32) - 1) / 2
    rots = cube_rotations(points.device)[:max(1, n_rot)]
    base = None; F = 0; Mom = 0
    for perm, sg, M in rots:
        vol = image.permute(*perm)
        for a_ in range(3):
            if sg[a_] < 0:
                vol = vol.flip(a_)
        p_r = (points - c) @ M.T + c; u_r = dirs @ M.T
        f = features(model, vol.contiguous(), voxel_um=voxel_um)
        r = query(model, f, p_r, sigma, u_r, vol_shape=vol.shape)
        if base is None:
            base = r
        R_back = M.T @ r["R"].float()                                                                # (Q, D, 3, 3): frames back in the original axes
        Mom = Mom + BG.moments(R_back.double(), r["kappa"].double())
        F = F + r["f"].float()
    Rf, Kf = BG.fit((Mom / len(rots)).reshape(-1, 3, 3))
    out = dict(base); out["f"] = F / len(rots)
    out["R"] = Rf.float().reshape(base["R"].shape); out["kappa"] = Kf.float().reshape(base["kappa"].shape)
    return out
