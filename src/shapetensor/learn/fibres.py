"""Monte Carlo random fibre architecture: the concepts of the fibre-substrate simulators (fastPLI, NFG, CACTUS), native.

Those simulators build white-matter-like geometry in three moves, all random and all cheap:

1. **packing** -- a bundle's cross-section is filled by random sequential addition (dart throwing) of fibres with
   gamma-distributed radii up to a target fibre volume fraction: no lattice (a hexagonal one leaves a periodic cue);
2. **dispersion** -- each fibre wanders about its place in the bundle (a smooth random walk), so neighbours intertwine
   and the bundle's orientation spread is a drawn quantity;
3. **collision relaxation** -- every fibre is a chain of spheres; overlapping spheres of different fibres are pushed
   apart (each by its share of the overlap), obstacles (cells, vessels) push fibres out, and a Laplacian step along each
   fibre keeps it smooth (the curvature bound). Iterated until the overlap is gone. Crossing bundles then **interweave**
   -- their fibres pass between one another -- instead of sharing voxels that the compositor clamps to one: the crossing
   core keeps both directions' fibre-scale texture in the image, as real tissue does.

``RELAX`` and ``PACKING`` switch them on in :class:`~shapetensor.learn.scene.Scene` (off by default). Everything happens
to the skeletons (S2), so the renderer, physics, acquisition and targets are unchanged: the labels still come from the
geometry that makes the image.
"""
from __future__ import annotations

import math

import numpy as np
import torch

RELAX = False                     # relax the scene's fibres jointly after the scene is built (crossings interweave)
PACKING = "hex"                   # "hex": the jittered lattice; "random": dart throwing with gamma radii
MOVABLE = ("bundle", "compose", "fascicle_fibres", "neuropil")   # line populations whose fibres take part (trees, vessels: obstacles)
GAP = 0.1                         # clearance between fibres (voxels) the relaxation aims for


def random_section(fvf: float, r_mean: float, k_shape: float, Rb: float, rng: np.random.Generator, max_n: int = 400,
                   tries: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Fibre centres ``(n, 2)`` in a disc of radius ``Rb`` and radii ``(n,)`` (voxels), drawn by random sequential addition:
    radii gamma (shape ``k_shape``, mean ``r_mean``, clipped to 0.4-2.5x), each placed at a uniform random point where it
    clears every fibre already placed, until the area fraction reaches ``fvf``, ``max_n`` fibres or ``tries`` darts. Random
    addition jams near 0.55 in 2-D: the relaxation and the fibres' wander pack the rest."""
    area = math.pi * Rb ** 2; C = np.zeros((0, 2)); R = np.zeros(0); filled = 0.0
    batch = 256
    for _ in range(max(1, tries // batch)):
        if filled / area >= fvf or len(R) >= max_n:
            break
        r = np.clip(rng.gamma(k_shape, r_mean / k_shape, batch), 0.4 * r_mean, 2.5 * r_mean)
        rho = np.sqrt(rng.uniform(0, 1, batch)) * np.maximum(Rb - r, 0); th = rng.uniform(0, 2 * np.pi, batch)
        P = np.stack([rho * np.cos(th), rho * np.sin(th)], 1)
        for p, rr in zip(P, r):                                               # sequential within the batch (exact: no overlaps)
            if len(R) and (np.hypot(*(C - p).T) < R + rr + GAP).any():
                continue
            C = np.vstack([C, p]); R = np.append(R, rr); filled += math.pi * rr ** 2
            if filled / area >= fvf or len(R) >= max_n:
                break
    if len(R) == 0:
        C, R = np.zeros((1, 2)), np.array([min(r_mean, Rb)])
    return C, R


def _cell_pairs(Xa: torch.Tensor, Xb: torch.Tensor, cut: float, same: bool, chunk: int = 16384) -> torch.Tensor:
    """Index pairs ``(n, 2)`` of points of ``Xa (Na, 3)`` and ``Xb (Nb, 3)`` closer than ``cut``, on the GPU: ``Xb`` hashed into
    cells of side ``cut`` (sorted keys), each point of ``Xa`` looking in its 27 neighbouring cells. ``same``: ``Xa`` is
    ``Xb``, and each pair is returned once (i < j)."""
    dev = Xa.device; M = 1 << 20
    key = lambda c: ((c[..., 0] + 4096) * M + (c[..., 1] + 4096)) * M + (c[..., 2] + 4096)
    cb = torch.floor(Xb / cut).long(); kb = key(cb); kb_s, order = torch.sort(kb)
    offs = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1).view(27, 3)
    out = []
    for a0 in range(0, Xa.shape[0], chunk):
        xa = Xa[a0:a0 + chunk]; ca = torch.floor(xa / cut).long()
        q = key(ca[:, None, :] + offs[None])                                               # (n, 27)
        lo = torch.searchsorted(kb_s, q); hi = torch.searchsorted(kb_s, q, right=True); cnt = (hi - lo).reshape(-1)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        ia = torch.arange(xa.shape[0], device=dev).repeat_interleave(27)[cnt > 0].repeat_interleave(cnt[cnt > 0])
        start = torch.repeat_interleave(lo.reshape(-1)[cnt > 0], cnt[cnt > 0])
        within = torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(cnt[cnt > 0], 0) - cnt[cnt > 0], cnt[cnt > 0])
        jb = order[start + within]
        ok = (xa[ia] - Xb[jb]).norm(dim=1) <= cut
        ia, jb = ia[ok] + a0, jb[ok]
        if same:
            keep = ia < jb; ia, jb = ia[keep], jb[keep]
        out.append(torch.stack([ia, jb], 1))
    return torch.cat(out) if out else torch.zeros(0, 2, dtype=torch.long, device=dev)


def relax(curves: list, obstacles: list | None = None, box: tuple | None = None, iters: int = 100, step: float = 0.5,
          smooth: float = 0.25, spacing: float = 1.6, rebuild: int = 10, skin: float = 2.0, tol: float = 0.005) -> dict:
    """Resolve the collisions among the fibres of ``curves`` (a list of :class:`~shapetensor.learn.scene.Curves`, all
    relaxed jointly) and with ``obstacles`` (``[(centres (n, 3), radii (n,))]``, fixed). Each fibre becomes a chain of
    spheres every ``spacing`` x its smallest radius (touching spheres); per iteration the overlapping pairs of different
    fibres move apart by ``step`` of their overlap (the thinner fibre more), obstacle overlaps push fibres out, and a
    Laplacian step of ``smooth`` along each chain keeps the fibres smooth (the curvature bound). Candidate pairs come from
    hashed grid cells per radius octave (:func:`_cell_pairs`), rebuilt every ``rebuild`` iterations with a ``skin``; all
    of it runs on the curves' device. Only the part inside ``box`` (``(lo (3,), hi (3,))``, voxels, plus a margin) moves. Stops when fewer than
    ``tol`` of the moving spheres overlap another fibre by more than a tenth of their radius. The curves' points are moved
    in place (the chain's displacement interpolated to every sample). Returns statistics."""
    if not curves:
        return {"fibres": 0}
    dev = curves[0].P.device
    chains = []
    for c in curves:
        S = c.P.shape[1]; rmin = float(c.rad.clamp_min(0.05).amin()); st = max(1, int(round(spacing * rmin / 0.5)))   # samples H = 0.5 apart
        st = max(st, 2)                                                          # at least a voxel apart (thin fibres: a sparser chain)
        chains.append((c, torch.unique(torch.cat([torch.arange(0, S, st), torch.tensor([S - 1])])).to(dev)))
    X0 = torch.cat([c.P[:, idx].reshape(-1, 3).float() for c, idx in chains])
    rr = torch.cat([c.rad[:, idx].reshape(-1).float() for c, idx in chains]).clamp_min(0.05)
    fib, pos, length, off = [], [], [], 0
    for c, idx in chains:
        N, Sk = c.P.shape[0], idx.numel()
        fib.append(torch.arange(N, device=dev).repeat_interleave(Sk) + off); off += N
        pos.append(torch.arange(Sk, device=dev).repeat(N)); length.append(torch.full((N * Sk,), Sk, device=dev))
    fib, pos, length = torch.cat(fib), torch.cat(pos), torch.cat(length)
    ar = torch.arange(X0.shape[0], device=dev)
    is_end = (pos == 0) | (pos == length - 1)
    prv = torch.where(pos == 0, ar, ar - 1); nxt = torch.where(pos == length - 1, ar, ar + 1)
    active = torch.ones(X0.shape[0], dtype=torch.bool, device=dev)
    if box is not None:                                                       # only the part that can show (plus a margin)
        lo = torch.tensor(box[0], device=dev, dtype=torch.float32) - 8; hi = torch.tensor(box[1], device=dev, dtype=torch.float32) + 8
        active = ((X0 >= lo) & (X0 <= hi)).all(1)
    ai = torch.nonzero(active)[:, 0]
    cls = np.digitize(np.log2(rr.cpu().numpy()), np.array([0.0, 2.0]))           # radius classes (< 1, 1-4, > 4 voxels): a pair's search radius is its classes'
    oC = oR = None
    if obstacles:
        oC = torch.cat([o[0].float() for o in obstacles]).to(dev); oR = torch.cat([o[1].float() for o in obstacles]).to(dev)
    X = X0.clone(); g = torch.Generator(device=dev).manual_seed(0)

    cls_t = torch.as_tensor(cls, device=dev)
    o_rmax = float(oR.max()) if oC is not None else 0.0

    def neighbours(Xn):
        """Candidate pairs (different fibres) and sphere-obstacle pairs, per radius-octave pair of classes (GPU cells)."""
        out = []; groups = {}
        for c_ in torch.unique(cls_t[ai]).tolist():
            sel = ai[cls_t[ai] == c_]; groups[c_] = (sel, float(rr[sel].max()))
        keys = sorted(groups)
        for k_a, ka in enumerate(keys):
            sa, ra = groups[ka]
            for kb in keys[k_a:]:
                sb, rb = groups[kb]; q = _cell_pairs(Xn[sa], Xn[sb], ra + rb + GAP + skin, same=(ka == kb))
                if q.shape[0]:
                    out.append(torch.stack([sa[q[:, 0]], sb[q[:, 1]]], 1))
        p_ = torch.cat(out) if out else torch.zeros(0, 2, dtype=torch.long, device=dev)
        p_ = p_[fib[p_[:, 0]] != fib[p_[:, 1]]]                                     # different fibres only
        po = None
        if oC is not None:
            q = _cell_pairs(Xn[ai], oC, float(rr[ai].max()) + o_rmax + GAP + skin, same=False)
            po = torch.stack([ai[q[:, 0]], q[:, 1]], 1) if q.shape[0] else None
        return p_, po

    def overlapping(i, j, ov, rmin):
        bad = torch.zeros(X.shape[0], dtype=torch.bool, device=dev)
        big = ov > 0.1 * rmin
        bad[i[big]] = True; bad[j[big]] = True
        return float(bad[active].float().mean())

    hist = []
    for it in range(iters):
        if it % rebuild == 0:
            pairs, po = neighbours(X)
        disp = torch.zeros_like(X); check = it % rebuild == rebuild - 1 or it == 0; frac = 0.0
        if pairs.shape[0]:
            i, j = pairs[:, 0], pairs[:, 1]
            v = X[i] - X[j]; d = v.norm(dim=1)
            ov = rr[i] + rr[j] + GAP - d; hit = ov > 0
            i, j, v, d, ov = i[hit], j[hit], v[hit], d[hit], ov[hit]
            if i.numel():
                same = d < 1e-3                                               # coincident centres: a random direction
                v = torch.where(same[:, None], torch.randn(v.shape, generator=g, device=dev), v)
                u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
                wi = rr[j] / (rr[i] + rr[j])
                disp.index_add_(0, i, (step * ov * wi)[:, None] * u); disp.index_add_(0, j, -(step * ov * (1 - wi))[:, None] * u)
                if check:
                    frac = overlapping(i, j, ov, torch.minimum(rr[i], rr[j]))
        if po is not None and po.shape[0]:
            i, k = po[:, 0], po[:, 1]
            v = X[i] - oC[k]; d = v.norm(dim=1).clamp_min(1e-6); ov = rr[i] + oR[k] + GAP - d; hit = ov > 0
            if bool(hit.any()):
                disp.index_add_(0, i[hit], (step * ov[hit] / d[hit])[:, None] * v[hit])
        lap = 0.5 * (X[prv] + X[nxt]) - X; lap[is_end] = 0                      # smoothness along each fibre
        disp = disp + smooth * lap
        disp[is_end] = disp[torch.where(pos == 0, nxt, prv)[is_end]]             # the ends follow their neighbour (no kink)
        disp[~active] = 0
        X = X + disp
        if check:
            hist.append(frac)
            if frac < tol:
                break
    D = X - X0; o = 0                                                          # the chains' displacement to every sample
    for c, idx in chains:
        N, S, _ = c.P.shape; Sk = idx.numel(); Dc = D[o:o + N * Sk].view(N, Sk, 3); o += N * Sk
        if Sk == 1:
            c.P = c.P + Dc.expand(N, S, 3).to(c.P.dtype); continue
        s_ = torch.arange(S, device=dev)
        k = (torch.searchsorted(idx, s_, right=True) - 1).clamp(0, Sk - 2)
        t = ((s_ - idx[k]).float() / (idx[k + 1] - idx[k]).float()).clamp(0, 1)[None, :, None]
        c.P = c.P + (Dc[:, k] * (1 - t) + Dc[:, k + 1] * t).to(c.P.dtype)
    return {"fibres": int(fib.max()) + 1, "spheres": int(X.shape[0]), "active": int(active.sum()), "iters": it + 1,
            "overlapping_start": round(hist[0], 4) if hist else 0.0, "overlapping_end": round(hist[-1], 4) if hist else 0.0}


def overlap_fraction(curves: list, shape, device) -> float:
    """The fraction of the fibres' rendered volume that lies in voxels holding more than one voxel of material (what the
    compositor clamps): rendered on a scratch canvas, one population for all."""
    from .scene import Canvas, render_curves
    cv = Canvas(shape, device)
    for c in curves:
        render_curves(cv, c)
    occ = cv.occ
    tot = float(occ.sum())
    return float((occ - 1).clamp_min(0).sum()) / max(tot, 1e-9)


def relax_scene(sc, **kw) -> dict:
    """Relax every movable line population's fibres of a built scene jointly, the scene's obstacles fixed. Unresolved
    bundles (a partial-volume curve) and density sections take no part."""
    cur = [c for g_ in sc.groups if g_.kind == "line" and g_.name in MOVABLE for c in g_.curves
           if c.fibre_r is None and (c.density is None or bool((c.density >= 1).all()))]
    if not cur:
        return {"fibres": 0}
    st = relax(cur, sc.obstacles or None, box=((0, 0, 0), tuple(float(n - 1) for n in sc.shape)), **kw)
    sc.record["relax"] = st
    return st
