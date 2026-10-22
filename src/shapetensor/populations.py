"""Populations on the lifted field: carry-forward (momentum), lobes, sheets and identities. Classical throughout.

* :func:`momentum` -- each direction's share of the field smoothed *along that direction* and bridged where it holds on
  both sides, never past a free end (a directional closing on the position × orientation field);
* :func:`peaks` -- the lobes of each position's distribution: up to ``nmax`` directions at least ``min_sep_deg`` apart,
  refined off the grid, with their masses -- N per position from the data;
* :func:`ring_fraction` -- a sheet (a boundary, a vessel wall) correlates along a whole plane and lifts to a ring, which
  peak-picking would read as two crossing populations; the ring's share of the distribution tells it apart;
* :func:`label_populations` -- identities 1, 2, 3, ... across space: fixels linked to their best-aligned neighbours,
  taken best first, with the rule that one population never holds two fixels of one position.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn


# ---------------------------------------------------------------------------------------------------------------
# carry-forward along each direction
# ---------------------------------------------------------------------------------------------------------------

def _base_grid(cs, device):
    Z, Y, X = cs
    base = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32) for n in cs), indexing="ij"))
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=device).view(3, 1, 1, 1)
    return base, scale


def momentum(O: torch.Tensor, grid: torch.Tensor, *, sigma_along: float = 3.0, reach: int = 6, angular_deg: float = 0.0,
             dir_batch: int = 40) -> torch.Tensor:
    """Population-specific smoothing of distributions ``O (n_dirs, Z, Y, X)`` (lattice cells). For each direction ``u``
    (after an optional angular smoothing of ``angular_deg`` so bends are followed):

    * ``sm``: the field smoothed along ``u`` (Gaussian ``sigma_along`` cells, truncated at ``reach``);
    * ``bridge``: the weaker of the two sides' largest value within ``reach`` cells along ``u`` -- the level the
      population holds on *both* sides (zero at a free end);
    * the result ``min(max(sm, bridge), max(O, bridge))``: a dip between two holding sides (a crossing, a gap) is lifted
      to the bridge, a spike is smoothed, and beyond a free end the smoothing's tail is cut by the cell's own value, so
      nothing is extended.

    All directions of a batch are sampled in one call per lag (``dir_batch`` at a time)."""
    nd = O.shape[0]; cs = tuple(O.shape[1:]); dev = O.device
    if angular_deg > 0:
        k = 1.0 / (2 * math.sin(math.radians(angular_deg)) ** 2)
        Ka = torch.exp(k * ((grid @ grid.T) ** 2 - 1)); Ka = Ka / Ka.sum(1, keepdim=True)
        Oa = (Ka @ O.reshape(nd, -1)).reshape(O.shape)
    else:
        Oa = O
    base, scale = _base_grid(cs, dev)
    ts = list(range(1, reach + 1))
    w = torch.tensor([math.exp(-0.5 * (t / sigma_along) ** 2) for t in range(-reach, reach + 1)], device=dev); w = w / w.sum()
    w0, wt = w[reach], w[reach + 1:]
    out = torch.empty_like(O)
    for d0 in range(0, nd, dir_batch):
        u = grid[d0:d0 + dir_batch]; B = u.shape[0]
        f = Oa[d0:d0 + dir_batch][:, None]                                                    # (B, 1, Z, Y, X)
        sm = w0 * f[:, 0]
        fw = torch.zeros((B,) + cs, device=dev); bw = torch.zeros((B,) + cs, device=dev)
        for t in ts:
            for sgn, acc in ((1.0, fw), (-1.0, bw)):
                p = (base[None] + sgn * t * u.view(B, 3, 1, 1, 1)) * scale[None] - 1.0            # (B, 3, Z, Y, X)
                g = p.flip(1).movedim(1, -1)                                                  # grid_sample wants (x, y, z)
                s = Fn.grid_sample(f, g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
                sm = sm + wt[t - 1] * s
                torch.maximum(acc, s, out=acc)
        bridge = torch.minimum(fw, bw)
        out[d0:d0 + dir_batch] = torch.minimum(torch.maximum(sm, bridge), torch.maximum(O[d0:d0 + dir_batch], bridge))
    return out


# ---------------------------------------------------------------------------------------------------------------
# lobes
# ---------------------------------------------------------------------------------------------------------------

def peaks(f: torch.Tensor, grid: torch.Tensor, *, nmax: int = 2, rel: float = 0.3, min_mass: float = 0.0, min_sep_deg: float = 30.0,
          nbr_deg: float | None = None, lobe_deg: float = 20.0, valley: float = 0.0):
    """Lobes of distributions ``f (n_dirs, ...)``: grid local maxima (no neighbour within ``nbr_deg`` higher), taken
    strongest first at least ``min_sep_deg`` apart; each lobe's mass is ``f`` over the directions within ``lobe_deg``
    of its peak that are nearer it than any other kept peak, and its direction their weighted principal axis. Lobes
    with mass below ``rel`` of the strongest or ``min_mass`` are dropped. With ``valley`` > 0 a further lobe must also
    stand above the saddle between it and the strongest peak: the lowest value of ``f`` along the arc joining them
    must be below ``(1 - valley)`` of the lobe's own peak (a shoulder of a dispersed population is not a lobe).
    Returns ``dirs (nmax, 3, ...)`` and ``mass (nmax, ...)`` (largest first, zero where absent)."""
    from .st import _eig3_sym
    nd = f.shape[0]; shp = f.shape[1:]; Ff = f.reshape(nd, -1); V = Ff.shape[1]; dev = f.device
    cosg = (grid @ grid.T).abs()
    if nbr_deg is None:
        nbr_deg = 1.3 * math.degrees(math.sqrt(2 * math.pi / nd))                                   # ~1.3 grid spacings
    nbr = (cosg >= math.cos(math.radians(nbr_deg))) & ~torch.eye(nd, dtype=torch.bool, device=dev)
    is_max = torch.ones_like(Ff, dtype=torch.bool)
    for k in range(nd):
        nk = nbr[k].nonzero().flatten()
        if len(nk):
            is_max[k] = Ff[k] >= Ff[nk].max(0).values
    s = torch.where(is_max & (Ff > 0), Ff, torch.zeros_like(Ff))
    cos_sep = math.cos(math.radians(min_sep_deg)); pk = torch.full((nmax, V), -1, dtype=torch.long, device=dev)
    for r in range(nmax):
        val, idx = s.max(0); ok = val > 0
        if r > 0 and valley > 0:                                                                   # the saddle between this peak and the first
            u1 = grid[pk[0].clamp_min(0)]; u2 = grid[idx]                                           # (V, 3)
            u2 = torch.where((u1 * u2).sum(1, keepdim=True) < 0, -u2, u2)
            saddle = torch.full((V,), float("inf"), device=dev)
            for t in (0.25, 0.5, 0.75):
                m = (1 - t) * u1 + t * u2; m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
                j = (grid @ m.T).abs().argmax(0)                                                    # the nearest grid direction
                saddle = torch.minimum(saddle, Ff.gather(0, j[None])[0])
            ok &= saddle <= (1 - valley) * val
        pk[r] = torch.where(ok, idx, torch.full_like(idx, -1))
        s = torch.where((cosg[idx] >= cos_sep).T & ok[None], torch.zeros_like(s), s)
        if r > 0 and valley > 0:                                                                   # a rejected shoulder is gone for good
            s = torch.where((~ok)[None] & (torch.arange(nd, device=dev)[:, None] == idx[None]), torch.zeros_like(s), s)
    cp = torch.stack([torch.where(pk[r] >= 0, cosg[:, pk[r].clamp_min(0)], torch.full((nd, V), -1.0, device=dev)) for r in range(nmax)])
    owner = cp.argmax(0); cos_lobe = math.cos(math.radians(lobe_deg))
    dirs = torch.zeros(nmax, 3, V, device=dev); mass = torch.zeros(nmax, V, device=dev)
    gz, gy, gx = grid[:, 0], grid[:, 1], grid[:, 2]
    for r in range(nmax):
        w = Ff * ((owner == r) & (cp[r] >= cos_lobe)).float()
        mass[r] = w.sum(0)
        M = torch.stack([c @ w for c in (gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz)])     # (6, V) [xx, yy, zz, xy, xz, yz]
        _, vec = _eig3_sym(M, full=True)                                                            # ascending; vec[2] the principal axis, (x, y, z)
        dirs[r] = vec[2].flip(0)
    order = mass.argsort(0, descending=True)
    mass = mass.gather(0, order); dirs = dirs.gather(0, order[:, None, :].expand(-1, 3, -1))
    keep = (mass >= rel * mass[:1]) & (mass >= min_mass) & (mass > 0)
    keep[0] = mass[0] > 0                                                                          # the strongest lobe is always the position's direction; the rules bound the extras
    return (dirs * keep[:, None]).reshape(nmax, 3, *shp), (mass * keep).reshape(nmax, *shp)


def lobe_ring_ratio(O: torch.Tensor, grid: torch.Tensor, dirs: torch.Tensor, mass: torch.Tensor, *, plane_deg: float = 12.0,
                    away_deg: float = 25.0) -> torch.Tensor:
    """Lobe or ring point? For every further lobe ``r >= 1`` of ``dirs (K, 3, ...)`` the great circle through it and
    the strongest lobe is read on the raw lift ``O (n_dirs, ...)``: the level of the circle's directions away from both
    lobes (within ``plane_deg`` of the plane, farther than ``away_deg`` from either), above the median direction, over
    the lobe's own height above the median. Near 0 for a second population (the rest of the circle is at the base
    level), near 1 for the far point of a sheet's ring (a boundary that contains the first population's direction
    correlates along every direction of its plane). ``(K, ...)``, zero for lobe 0 and where a lobe is absent."""
    nd = O.shape[0]; shp = O.shape[1:]; Of = O.reshape(nd, -1); V = Of.shape[1]; K = dirs.shape[0]
    med = Of.median(0).values
    D = dirs.reshape(K, 3, -1); has = mass.reshape(K, -1) > 0
    out = torch.zeros(K, V, device=O.device)
    sin_p, cos_a = math.sin(math.radians(plane_deg)), math.cos(math.radians(away_deg))
    u1 = D[0]
    near1 = (grid @ u1).abs() >= cos_a                                                             # (nd, V)
    for r in range(1, K):
        u2 = D[r]
        n = torch.linalg.cross(u1, u2, dim=0); nn = n.norm(dim=0)
        n = n / nn.clamp_min(1e-12)
        inplane = (grid @ n).abs() <= sin_p
        rest = (inplane & ~near1 & ((grid @ u2).abs() < cos_a)).float()
        level = (Of * rest).sum(0) / rest.sum(0).clamp_min(1)
        j = (grid @ u2).abs().argmax(0)
        o2 = Of.gather(0, j[None])[0]
        ratio = (level - med).clamp_min(0) / (o2 - med).clamp_min(1e-9)
        out[r] = torch.where(has[r] & (nn > 1e-6), ratio, torch.zeros_like(ratio))
    return out.reshape(K, *shp)


def ring_fraction(f: torch.Tensor, grid: torch.Tensor, dirs: torch.Tensor, mass: torch.Tensor, *, plane_deg: float = 12.0,
                  away_deg: float = 25.0):
    """For distributions ``f (n_dirs, ...)`` and their kept peaks ``dirs (nmax, 3, ...)`` (``mass`` zero where absent):
    the plane of ``f`` (normal = least axis of its second moment), the mass within ``plane_deg`` of that plane, and the
    share of it farther than ``away_deg`` from every kept peak (about 0.6 for a ring, near 0 for one or two lines);
    and the prominence: the largest value of ``f`` over the mean of the ring's directions away from the peaks (a
    population lying in a sheet stands above it)."""
    from .st import _eig3_sym
    nd = f.shape[0]; shp = f.shape[1:]; Ff = f.reshape(nd, -1); V = Ff.shape[1]
    gz, gy, gx = grid[:, 0], grid[:, 1], grid[:, 2]
    M = torch.stack([c @ Ff for c in (gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz)])
    _, n = _eig3_sym(M, full=True); n = n[0].flip(0)                                               # (3, V) plane normal, (z, y, x)
    inplane = ((grid @ n).abs() <= math.sin(math.radians(plane_deg))).float()                     # (nd, V)
    D = dirs.reshape(dirs.shape[0], 3, -1); has = mass.reshape(mass.shape[0], -1) > 0
    near = torch.zeros(nd, V, dtype=torch.bool, device=f.device); cos_a = math.cos(math.radians(away_deg))
    for r in range(D.shape[0]):
        near |= ((grid @ D[r]).abs() >= cos_a) & has[r][None]
    tot = (Ff * inplane).sum(0); away = inplane * (~near).float()
    ring = ((Ff * away).sum(0) / tot.clamp_min(1e-12)).reshape(shp)
    level = (Ff * away).sum(0) / away.sum(0).clamp_min(1)
    return ring, (Ff.max(0).values / level.clamp_min(1e-12)).reshape(shp)


# ---------------------------------------------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------------------------------------------

def _half_offsets(connectivity: int = 26):
    offs = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) > (0, 0, 0)]
    if connectivity == 6:
        offs = [o for o in offs if sum(map(abs, o)) == 1]
    elif connectivity == 18:
        offs = [o for o in offs if sum(map(abs, o)) <= 2]
    return offs


_KRUSKAL = None


def _compile_kruskal():
    from numba import njit

    @njit(cache=True)
    def run(n, src, dst, order, pa, pb, tol, tol_min):
        parent = np.arange(n); size = np.ones(n, np.int64)
        m = 2 * len(pa)
        ent_partner = np.empty(m, np.int64); ent_next = np.full(m, -1, np.int64)
        head = np.full(n, -1, np.int64); tail = np.full(n, -1, np.int64); length = np.zeros(n, np.int64)
        for i in range(len(pa)):
            for e, a, b in ((2 * i, pa[i], pb[i]), (2 * i + 1, pb[i], pa[i])):
                ent_partner[e] = b
                if head[a] == -1:
                    head[a] = e
                else:
                    ent_next[tail[a]] = e
                tail[a] = e; length[a] += 1

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                nx = parent[x]; parent[x] = r; x = nx
            return r
        for j in range(len(order)):
            ra = find(src[order[j]]); rb = find(dst[order[j]])
            if ra == rb:
                continue
            small, other = (ra, rb) if length[ra] <= length[rb] else (rb, ra)
            allow = max(tol_min, int(tol * min(size[ra], size[rb])))
            e = head[small]; bad = 0
            while e != -1:
                if find(ent_partner[e]) == other:
                    bad += 1
                    if bad > allow:
                        break
                e = ent_next[e]
            if bad > allow:
                continue
            big, little = (ra, rb) if size[ra] >= size[rb] else (rb, ra)
            parent[little] = big; size[big] += size[little]
            if head[little] != -1:
                if head[big] == -1:
                    head[big] = head[little]
                else:
                    ent_next[tail[big]] = head[little]
                tail[big] = tail[little]; length[big] += length[little]
        roots = np.empty(n, np.int64)
        for x in range(n):
            roots[x] = find(x)
        return roots
    return run


def _constrained_components(n: int, src, dst, order, pa, pb, tol: float = 0.0, tol_min: int = 0):
    """Kruskal over edges ``src[order], dst[order]`` with cannot-link pairs ``(pa[i], pb[i])`` (fixels of one position):
    two components merge only if at most ``max(tol_min, tol * size of the smaller)`` cannot-link pairs would fall
    inside the merged one. Returns each node's root."""
    global _KRUSKAL
    if _KRUSKAL is None:
        _KRUSKAL = _compile_kruskal()
    return _KRUSKAL(n, src, dst, order, pa, pb, float(tol), int(tol_min))


def label_populations(dirs: torch.Tensor, mass: torch.Tensor, *, max_angle_deg: float = 20.0, min_fixels: int = 8,
                      connectivity: int = 26, conflict_tol: float = 0.02, conflict_min: int = 2):
    """Population identities for fixels ``dirs (K, 3, Z, Y, X)`` with ``mass (K, Z, Y, X)`` (zero = absent). Candidate
    links: fixel ``a`` at ``p`` and ``b`` at a neighbour ``p + o`` when each is the other's best-aligned fixel there and
    they are within ``max_angle_deg``. Links are taken best-aligned first (Kruskal), and two populations merge only if
    the merged one would hold two fixels of one position at more than ``max(conflict_min, conflict_tol x the smaller's
    fixels)`` positions: a population follows its bends, but chains of in-between directions (unresolved crossing
    positions, noise) cannot join two populations that cross, while a few noisy fixels cannot keep one population in
    pieces. Where a population still holds two fixels of a position the weaker loses the label. Populations are
    numbered 1, 2, ... by size; fewer than ``min_fixels`` fixels get 0. Returns ``labels (K, Z, Y, X)`` (int64, 0 =
    none) and a table (per label: fixels, mass, mean direction (z, y, x))."""
    K = dirs.shape[0]; shp = tuple(mass.shape[1:]); V = int(np.prod(shp)); dev = mass.device
    cos_max = math.cos(math.radians(max_angle_deg))
    idx = torch.arange(V, device=dev).reshape(shp)
    src, dst, cost = [], [], []
    for o in _half_offsets(connectivity):
        a_sl = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(o, shp))
        b_sl = tuple(slice(sa.start + d, sa.stop + d) for sa, d in zip(a_sl, o))
        A = dirs[(slice(None), slice(None)) + a_sl]; B = dirs[(slice(None), slice(None)) + b_sl]
        ma = mass[(slice(None),) + a_sl] > 0; mb = mass[(slice(None),) + b_sl] > 0
        C = torch.einsum("ic...,jc...->ij...", A, B).abs()                                         # (K, K, ...)
        C = torch.where(ma[:, None] & mb[None, :], C, torch.full_like(C, -1.0))
        best_b = C.argmax(1); best_a = C.argmax(0)
        ia = idx[a_sl]; ib = idx[b_sl]
        for k1 in range(K):
            k2 = best_b[k1]
            c = C[k1].gather(0, k2[None])[0]
            mutual = best_a.gather(0, k2[None])[0] == k1
            ok = mutual & (c >= cos_max)
            if ok.any():
                src.append((k1 * V + ia[ok]).cpu()); dst.append((k2[ok] * V + ib[ok]).cpu()); cost.append((1 - c[ok]).cpu())
    nodes = K * V
    if src:
        s_ = torch.cat(src).numpy().astype(np.int64); d_ = torch.cat(dst).numpy().astype(np.int64); c_ = torch.cat(cost).numpy()
    else:
        s_ = d_ = np.zeros(0, np.int64); c_ = np.zeros(0, np.float32)
    present = (mass.reshape(-1) > 0).cpu().numpy()
    pres = present.reshape(K, V)
    pa, pb = [], []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            both = np.nonzero(pres[k1] & pres[k2])[0]
            pa.append(k1 * V + both); pb.append(k2 * V + both)
    pa = np.concatenate(pa).astype(np.int64) if pa else np.zeros(0, np.int64)
    pb = np.concatenate(pb).astype(np.int64) if pb else np.zeros(0, np.int64)
    comp = _constrained_components(nodes, s_, d_, np.argsort(c_, kind="stable").astype(np.int64), pa, pb, conflict_tol, conflict_min)
    if len(pa):                                                                                   # one fixel per position and population
        clash = comp[pa] == comp[pb]
        wm = mass.reshape(-1).cpu().numpy()
        weaker = np.where(wm[pa[clash]] >= wm[pb[clash]], pb[clash], pa[clash])
        present = present.copy(); present[weaker] = False
    comp = np.where(present, comp, -1)
    ids, counts = np.unique(comp[present], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remap = np.zeros(comp.max() + 2 if comp.size else 1, np.int64)
    n_lab = 0
    for j in order:
        if counts[j] < min_fixels:
            break
        n_lab += 1; remap[ids[j] + 1] = n_lab
    lab = remap[comp + 1]
    labels = torch.as_tensor(lab, device=dev).reshape(K, *shp)
    D = dirs.movedim(1, -1).reshape(-1, 3)
    w = mass.reshape(-1); L = labels.reshape(-1)
    table = []
    if n_lab:
        M = torch.zeros(n_lab + 1, 3, 3, device=dev)
        M.index_add_(0, L, w[:, None, None] * D[:, :, None] * D[:, None, :])
        cnt = torch.bincount(L, minlength=n_lab + 1); msum = torch.zeros(n_lab + 1, device=dev).index_add_(0, L, w)
        ev = torch.linalg.eigh(M[1:].double())[1][:, :, -1].float()
        for j in range(n_lab):
            table.append({"label": j + 1, "fixels": int(cnt[j + 1]), "mass": round(float(msum[j + 1]), 2),
                          "direction": [round(float(x), 3) for x in ev[j]]})
    return labels, table


# ---------------------------------------------------------------------------------------------------------------
# lobe or ring point: the shape of a further lobe on the raw lift
# ---------------------------------------------------------------------------------------------------------------

def lobe_shape(O: torch.Tensor, grid: torch.Tensor, dirs: torch.Tensor, mass: torch.Tensor, R: torch.Tensor, bin_deg: float = 2.5,
               *, step_deg: float = 15.0, span_deg: float = 75.0, exclude_deg: float = 25.0):
    """For every further lobe ``r >= 1``: its profile on the raw lift ``O`` after the strongest lobe's predicted tail is
    removed (``O - median - h1 R(angle to u1)``), sampled on the great circle through both lobes (*along*, where a
    sheet's ring runs) and on the great circle through the lobe perpendicular to it (*across*), as the mean over
    ``±span_deg`` in steps of ``step_deg`` relative to the value at the lobe. Directions within ``exclude_deg`` of
    the strongest lobe are left out. A population lobe decays both ways (both near the response's mean); the far point
    of a sheet's ring is flat along and decays across. Returns ``along (K, ...)``, ``across (K, ...)`` (zero for lobe 0
    and absent lobes)."""
    from .lift import _profile_at
    nd = O.shape[0]; shp = O.shape[1:]; Of = O.reshape(nd, -1); V = Of.shape[1]; K = dirs.shape[0]; dev = O.device
    med = Of.median(0).values
    D = dirs.reshape(K, 3, -1); has = mass.reshape(K, -1) > 0
    u1 = D[0]
    j1 = (grid @ u1).abs().argmax(0); h1 = (Of.gather(0, j1[None])[0] - med).clamp_min(0)

    def resid(m):
        """``m (3, V)`` unit directions -> the residual lift there."""
        j = (grid @ m).abs().argmax(0)
        ang1 = torch.rad2deg(torch.acos((m * u1).sum(0).abs().clamp(max=1.0)))
        return Of.gather(0, j[None])[0] - med - h1 * _profile_at(R, ang1, bin_deg), ang1
    along = torch.zeros(K, V, device=dev); across = torch.zeros(K, V, device=dev)
    phis = [math.radians(p) for p in torch.arange(-span_deg, span_deg + 1e-6, step_deg).tolist() if abs(p) > 1e-6]
    for r in range(1, K):
        u2 = D[r]
        w = u1 - (u1 * u2).sum(0, keepdim=True) * u2; w = w / w.norm(dim=0).clamp_min(1e-12)        # in the plane of both, toward u1
        n = torch.linalg.cross(u2, w, dim=0); n = n / n.norm(dim=0).clamp_min(1e-12)                 # the plane's normal
        r0, _ = resid(u2); r0 = r0.clamp_min(1e-9)
        for axis, acc in ((w, along), (n, across)):
            s = torch.zeros(V, device=dev); c = torch.zeros(V, device=dev)
            for phi in phis:
                m = math.cos(phi) * u2 + math.sin(phi) * axis
                rv, ang1 = resid(m)
                ok = (ang1 >= exclude_deg).float()
                s += rv.clamp_min(0) * ok; c += ok
            acc[r] = torch.where(has[r], s / c.clamp_min(1) / r0, torch.zeros_like(s))
    return along.reshape(K, *shp), across.reshape(K, *shp)
