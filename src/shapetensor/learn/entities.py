"""Entities: a scene's objects, the populations they form at any depth, and their targets at query points.

The scene canvas stores, per (voxel, population, object), the object's volume there and the volume-weighted sums of its
shape tensor and descriptors (:meth:`~shapetensor.learn.scene.Canvas.entries`), so an entity's tensor is the exact sum
of its objects', with the populations' own trilinear weights.

* **Tree.** A leaf is an object within a population (one object can lie in two populations, e.g. a split tract's
  halves). Its ancestors follow the scene's parent links (fibre -> bundle, nucleus -> cell, fibre -> fascicle ->
  nerve, ...); the top of the chain joins its population node, and the populations join the root. An ancestor shared
  by several populations sits above them. Any depth; objects carry their smallest radius, populations and above none.
* **Resolution.** At a window of scale sigma an object is its own entity when its radius is at least
  ``max(r_min, c * sigma)`` (crop voxels), or when it is the only object of its merged entity in the window (an
  isolated thin fibre); otherwise it merges into its parent, repeatedly.
* **Targets** (:func:`entity_targets`): for each query point and sigma, the entries in a Gaussian window around the
  point the warp pulls it from, grouped by resolved entity: mass (volume fraction in the window times its
  population's visibility), shape tensor (rotated by the warp's Jacobian, trace 1; the Bingham target is
  moment-matched from it) and descriptors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

OBJ = 1 << 22                                       # the leaf key: group * OBJ + object (as the canvas's keys)
KIND_OBJECT, KIND_POPULATION, KIND_SHARED, KIND_ROOT = 0, 1, 2, 3


@dataclass
class Tree:
    """A scene's entity tree. Node 0 is the root. ``parent (N,)`` (-1 at the root), ``radius (N,)`` (canvas voxels;
    inf where the node is a population or above), ``kind (N,)``, ``group (N,)`` (its population, -1 above them),
    ``depth (N,)`` (root 0), ``leaf_keys`` sorted ``group * OBJ + object`` and ``leaf_nodes`` the node of each."""
    parent: torch.Tensor
    radius: torch.Tensor
    kind: torch.Tensor
    group: torch.Tensor
    depth: torch.Tensor
    leaf_keys: torch.Tensor
    leaf_nodes: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.parent.shape[0])

    def leaves(self, grp: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        """The leaf node of each (population, object) (-1 if unknown)."""
        k = grp.long() * OBJ + obj.long()
        i = torch.searchsorted(self.leaf_keys, k).clamp(max=self.leaf_keys.numel() - 1)
        return torch.where(self.leaf_keys[i] == k, self.leaf_nodes[i], torch.full_like(i, -1))

    def resolve(self, node: torch.Tensor, r_thr: torch.Tensor) -> torch.Tensor:
        """Each node lifted to its first ancestor-or-self whose radius reaches ``r_thr`` (canvas voxels, per item)."""
        for _ in range(int(self.depth.max()) + 1):
            up = (self.radius[node] < r_thr) & (self.parent[node] >= 0)
            if not bool(up.any()):
                break
            node = torch.where(up, self.parent[node], node)
        return node

    def ancestors(self, nodes: torch.Tensor) -> torch.Tensor:
        """``(n, D)`` the chain from each node up to the root (itself first), padded with -1."""
        D = int(self.depth.max()) + 1; out = torch.full((nodes.numel(), D), -1, dtype=torch.long, device=nodes.device)
        cur = nodes.long().clone()
        for k in range(D):
            ok = cur >= 0
            out[:, k] = torch.where(ok, cur, torch.full_like(cur, -1))
            cur = torch.where(ok, self.parent[cur.clamp_min(0)], cur)
        return out


def build_tree(sc, grp: torch.Tensor, obj: torch.Tensor) -> Tree:
    """The tree of a rendered scene ``sc`` over the (population, object) pairs present (``grp``, ``obj``: the entries)."""
    dev = grp.device
    keys = torch.unique(grp.long() * OBJ + obj.long())
    G = (keys // OBJ).cpu().numpy(); O = (keys % OBJ).cpu().numpy()
    parents, radius = sc.parents, sc.obj_r
    def chain(o):                                                                                # o, its parent, ... (cycle-safe)
        c = [int(o)]
        while c[-1] in parents and len(c) < 64:
            p = int(parents[c[-1]])
            if p in c:
                break
            c.append(p)
        return c
    has_parent = np.array([int(o) in parents for o in O], dtype=bool)
    tops = {}                                                                                    # top ancestor -> its populations
    chains = {}
    for o in np.unique(O[has_parent]):
        chains[int(o)] = chain(o)
    def top(o):
        return chains[int(o)][-1] if int(o) in chains else int(o)
    for g, o in zip(G, O):
        tops.setdefault(top(o), set()).add(int(g))
    shared = {t for t, gs in tops.items() if len(gs) > 1}
    # nodes: root, populations, shared ancestors, then objects within populations
    n_groups = int(G.max()) + 1 if G.size else 0
    par, rad, kind, grp_ = [-1], [math.inf], [KIND_ROOT], [-1]
    pop_node = {}
    for g in range(n_groups):
        pop_node[g] = len(par); par.append(0); rad.append(math.inf); kind.append(KIND_POPULATION); grp_.append(g)
    shared_node = {}
    for t in sorted(shared):
        shared_node[t] = len(par); par.append(0); rad.append(math.inf); kind.append(KIND_SHARED); grp_.append(-1)
    for g, t in ((g, t) for t in shared for g in tops[t]):                                         # populations under their shared ancestor
        par[pop_node[g]] = shared_node[t]
    obj_node = {}
    def node_of(x, g, above):
        k = (int(g), int(x))
        if k not in obj_node:
            obj_node[k] = len(par); par.append(above); rad.append(float(radius.get(int(x), math.inf))); kind.append(KIND_OBJECT); grp_.append(int(g))
        return obj_node[k]
    leaf_nodes = np.empty(len(O), dtype=np.int64)
    simple = ~has_parent & np.array([int(o) not in shared for o in O], dtype=bool)                 # most objects: leaf -> population
    base = len(par); idx = np.nonzero(simple)[0]
    par += [pop_node[int(g)] for g in G[idx]]; rad += [float(radius.get(int(o), math.inf)) for o in O[idx]]
    kind += [KIND_OBJECT] * idx.size; grp_ += [int(g) for g in G[idx]]
    leaf_nodes[idx] = base + np.arange(idx.size)
    for i in np.nonzero(~simple)[0]:
        g, o = int(G[i]), int(O[i]); c = chains.get(o, [o]); t = c[-1]
        if t in shared and len(c) > 1:
            c = c[:-1]                                                                           # the shared top sits above the populations
        above = pop_node[g]
        for x in reversed(c):                                                                    # build top-down: each node under the one above
            above = node_of(x, g, above)
        leaf_nodes[i] = obj_node[(g, o)]
    t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    parent = t(par); depth = torch.zeros_like(parent); cur = parent.clone()
    for _ in range(64):
        ok = cur >= 0
        if not bool(ok.any()):
            break
        depth += ok.long(); cur = torch.where(ok, parent[cur.clamp_min(0)], cur)
    return Tree(parent=parent, radius=t(rad, torch.float32), kind=t(kind, torch.int8), group=t(grp_), depth=depth,
                leaf_keys=keys.to(dev), leaf_nodes=t(leaf_nodes))


def _pulled(points: torch.Tensor, grid: torch.Tensor | None, shape):
    """Query points (output crop voxels, ``(Q, 3)``) -> the input positions the warp samples them from, and the warp's
    Jacobian there ``(Q, 3, 3)`` (identity without a warp)."""
    Q = points.shape[0]; dev = points.device
    if grid is None:
        return points, torch.eye(3, device=dev).expand(Q, 3, 3)
    Z, Y, X = shape
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=dev)
    pin = ((grid[0].flip(-1) + 1.0) / scale)                                                     # (Z, Y, X, 3): input voxel position of each output voxel
    def at(f, p):                                                                                # trilinear sample of f (Z, Y, X, C) at p (n, 3)
        g = (p * scale - 1.0).flip(-1).view(1, -1, 1, 1, 3)
        return torch.nn.functional.grid_sample(f.permute(3, 0, 1, 2)[None], g, align_corners=True, padding_mode="border")[0, :, :, 0, 0].T
    xp = at(pin, points)
    J = torch.stack(torch.gradient(pin, dim=(0, 1, 2)), -1)                                      # (Z, Y, X, 3 out, 3 in) d pin / d x
    Jq = at(J.reshape(Z, Y, X, 9), points).view(Q, 3, 3)
    return xp, Jq


def entity_targets(entries, tree: Tree, canvas_shape, origin, zoom: int, shape, points: torch.Tensor, sigma: torch.Tensor, grid=None,
                   vis: torch.Tensor | None = None, group_layer=None, r_min: float = 1.5, c_sig: float = 0.5, emax: int = 8,
                   bin_size: int = 2, mask: torch.Tensor | None = None, keep: torch.Tensor | None = None, m_min: float = 1e-3,
                   ovis: torch.Tensor | None = None, alone_is_object: bool = True, sig_acq: float = 0.0, share_keep: float = 0.2,
                   comp: torch.Tensor | None = None) -> dict:
    """The entities at query ``points (Q, 3)`` (output crop voxels) and scales ``sigma (Q,)`` (crop voxels).

    ``entries`` the canvas's ``(vox, group, obj, w, v)`` (canvas linear index; ``v`` the w-weighted tensor (6) and
    descriptors (5)); ``origin`` / ``zoom`` / ``shape`` the crop on the canvas (``shape`` the output crop's); ``grid`` the
    warp's sampling grid (None: no warp); ``vis (K, Z, Y, X)`` the layers' visibility in the output crop and
    ``group_layer`` the layer of each population (-1: none); ``mask (Z, Y, X)`` the crop's tissue (specimen edge, cut-outs:
    tissue removed takes its entities with it) and ``keep`` the labelled objects (sparse labelling: the others are gone);
    entities whose visible mass is below ``m_min`` are dropped; ``ovis (K, Z, Y, X)`` the layers' orientation visibility
    (each population's tensor blended toward isotropic where its orientation does not show, as the total density's
    target, :func:`~shapetensor.learn.query.voxel_density`); ``alone_is_object``: an entity the resolution merged that
    holds a single object in this window (no sibling with mass >= ``m_min`` there) is that object -- an isolated thin
    fibre, a sparsely labelled neurite is traceable as itself; objects merge only where they share the window;
    ``sig_acq`` the acquisition's own blur (crop voxels): what the image cannot resolve merges, the resolution rule's
    scale being ``sqrt(sigma^2 + (2 sig_acq)^2)``; an entity is kept below ``m_min`` when it holds ``share_keep`` of
    its window's visible mass (a thin bright line alone in a large window); ``comp (L, Z, Y, X)`` each population layer's
    composite factor (pre-warp crop space: occlusion and the volume budget, as the image and the total target). Only
    entries inside the crop count (the image shows nothing beyond it); visibility is taken where each part of the
    window lies (as the total target weighs each voxel). Returns, padded to ``emax`` entities per query (by mass):
    ``node (Q, E)`` (tree nodes, -1 pad), ``mass``, ``vis``, ``T (Q, E, 3, 3)`` (trace 1), ``desc (Q, E, 5)``."""
    vox, grp, obj, w, v = entries
    dev = points.device; Q = points.shape[0]; zf = float(zoom)
    canvas = torch.as_tensor(canvas_shape, device=dev)
    # entries -> crop coordinates (a crop voxel's centre sits at origin + zoom j + (zoom - 1) / 2 on the canvas)
    vox = vox.long(); cz = vox // (canvas[1] * canvas[2]); cy = (vox // canvas[2]) % canvas[1]; cx = vox % canvas[2]
    o = torch.as_tensor(origin, device=dev, dtype=torch.float32)
    c = (torch.stack([cz, cy, cx], 1).float() - o - (zf - 1) / 2) / zf
    xp, Jq = _pulled(points.float(), grid, shape)
    hi_ = torch.tensor(shape, device=dev, dtype=torch.float32) - 1
    xp = torch.minimum(torch.maximum(xp, torch.zeros_like(xp)), hi_)                              # pulled beyond a face: the face (the image's padding)
    smax = float(sigma.max()); m = 3 * smax + bin_size
    lo = xp.min(0).values - m; hi = xp.max(0).values + m
    sel_ = ((c >= lo) & (c <= hi)).all(1) & ((c >= -0.5) & (c <= hi_ + 0.5)).all(1)              # inside the crop only
    if keep is not None:                                                                        # sparse labelling: unlabelled objects are gone
        sel_ &= torch.as_tensor(keep, device=dev)[obj.long().clamp(max=int(keep.numel()) - 1)]
    c, grp, obj, w, v = c[sel_], grp[sel_].long(), obj[sel_].long(), w[sel_].float(), v[sel_].float()
    if mask is not None:                                                                        # tissue removed (edge, cut-outs): its entries too
        mw = _vis_at(mask.float()[None], c.clamp(min=torch.zeros(3, device=dev), max=torch.tensor(mask.shape, device=dev, dtype=torch.float32) - 1))[:, 0]
        w = w * mw; v = v * mw[:, None]
    leaf = tree.leaves(grp, obj)
    ok = leaf >= 0; c, grp, w, v, leaf = c[ok], grp[ok], w[ok], v[ok], leaf[ok]
    # bins of bin_size crop voxels: one row per (bin, leaf) with its volume, sums and centroid
    b3 = torch.floor((c - lo) / bin_size).long(); nb = torch.ceil((hi - lo) / bin_size).long() + 1
    blin = (b3[:, 0] * nb[1] + b3[:, 1]) * nb[2] + b3[:, 2]
    key = blin * (tree.n + 1) + leaf
    uk, inv = torch.unique(key, return_inverse=True)
    R = uk.shape[0]
    rw = torch.zeros(R, device=dev).index_add_(0, inv, w)
    rv = torch.zeros(R, v.shape[1], device=dev).index_add_(0, inv, v)
    rc = torch.zeros(R, 3, device=dev).index_add_(0, inv, w[:, None] * c) / rw.clamp_min(1e-12)[:, None]
    rbin = uk // (tree.n + 1); rleaf = uk % (tree.n + 1)
    rgrp = torch.zeros(R, dtype=torch.long, device=dev).index_put_((inv,), grp)
    if comp is not None and group_layer is not None:                                           # composite: occlusion, the volume budget
        gl_r = torch.as_tensor(list(group_layer) + [-1], device=dev, dtype=torch.long)[rgrp.clamp(max=len(group_layer))]
        fr = torch.ones(R, device=dev)
        for k in torch.unique(gl_r[gl_r >= 0]).tolist():
            mk = gl_r == k
            fr[mk] = _vis_at(comp[k:k + 1].float(), rc[mk])[:, 0]
        rw = rw * fr; rv = rv * fr[:, None]
    # (query, row) pairs: the bins within 3 sigma of each query's pulled point
    starts = torch.searchsorted(rbin, torch.arange(int(nb.prod()), device=dev))
    ends = torch.searchsorted(rbin, torch.arange(int(nb.prod()), device=dev), right=True)
    qi_all, ri_all = [], []
    for rad in torch.unique(torch.ceil((3 * sigma + 1) / bin_size).long()).tolist():
        qs = torch.nonzero(torch.ceil((3 * sigma + 1) / bin_size).long() == rad)[:, 0]
        off = torch.arange(-rad, rad + 1, device=dev)
        O3 = torch.stack(torch.meshgrid(off, off, off, indexing="ij"), -1).view(-1, 3)
        qb = torch.floor((xp[qs] - lo) / bin_size).long()[:, None] + O3[None]                       # (q, nO, 3)
        inside = ((qb >= 0) & (qb < nb)).all(-1)
        ql = ((qb[..., 0] * nb[1] + qb[..., 1]) * nb[2] + qb[..., 2]).clamp(0, int(nb.prod()) - 1)
        s_, e_ = starts[ql] * inside, ends[ql] * inside
        cnt = (e_ - s_).flatten()
        qrep = qs[:, None].expand_as(ql).flatten().repeat_interleave(cnt)
        first = s_.flatten().repeat_interleave(cnt)
        within = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        qi_all.append(qrep); ri_all.append(first + within)
    qi = torch.cat(qi_all); ri = torch.cat(ri_all)
    d2 = ((rc[ri] - xp[qi]) ** 2).sum(1); sg = sigma[qi]
    g = torch.exp(-0.5 * d2 / sg ** 2) * (d2 <= (3 * sg) ** 2)
    # the resolved entity of each pair, then sums per (query, entity)
    thr = zf * torch.clamp(c_sig * torch.sqrt(sg ** 2 + (2 * float(sig_acq)) ** 2), min=r_min)
    node = tree.resolve(rleaf[ri], thr)
    gw = g * rw[ri]
    if alone_is_object:                                                                         # a merged entity holding one object here is that object
        node = _single_object(qi, node, rleaf[ri], gw / (zf ** 3 * (2 * math.pi * sg ** 2) ** 1.5), m_min, tree.n)
    vl = torch.ones_like(gw); ol = None
    if group_layer is not None and (vis is not None or ovis is not None):
        gl = torch.as_tensor(list(group_layer) + [-1], device=dev, dtype=torch.long)[rgrp[ri].clamp(max=len(group_layer))]   # long even when empty
        Ji_q = torch.linalg.inv(Jq)                                                                # each part of the window where it lies in the output
        y = (points.float()[qi] + (Ji_q[qi] @ (rc[ri] - xp[qi])[..., None])[..., 0])
        y = torch.minimum(torch.maximum(y, torch.zeros_like(y)), hi_)
        vl = torch.zeros_like(gw) if vis is not None else vl                                       # a population absent from the crop: not visible
        ol = torch.ones_like(gw) if ovis is not None else None
        for k in torch.unique(gl[gl >= 0]).tolist():
            mk = gl == k
            if vis is not None:
                vl[mk] = _vis_at(vis[k:k + 1].float(), y[mk])[:, 0]
            if ovis is not None:
                ol[mk] = _vis_at(ovis[k:k + 1].float(), y[mk])[:, 0].clamp(0, 1)
    ek = qi * (tree.n + 1) + node
    ue, einv = torch.unique(ek, return_inverse=True)
    eq = ue // (tree.n + 1); en = ue % (tree.n + 1)
    Wm = torch.zeros(ue.shape[0], device=dev).index_add_(0, einv, gw)
    vr = g[:, None] * rv[ri]
    if ol is not None:                                                                             # orientation not visible: toward isotropic
        iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=dev) / 3
        vr = torch.cat([ol[:, None] * vr[:, :6] + (1 - ol)[:, None] * gw[:, None] * iso, vr[:, 6:]], 1)
    Vs = torch.zeros(ue.shape[0], v.shape[1], device=dev).index_add_(0, einv, vr)
    Vis = torch.zeros(ue.shape[0], device=dev).index_add_(0, einv, gw * vl) / Wm.clamp_min(1e-12)
    norm = zf ** 3 * (2 * math.pi * sigma[eq] ** 2) ** 1.5
    mass = Wm / norm
    T6 = Vs[:, :6] / Wm.clamp_min(1e-12)[:, None]
    T = torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], -1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], -1),
                     torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], -1)], 1)
    Ji = torch.linalg.inv(Jq[eq])
    T = Ji @ T @ Ji.transpose(1, 2); T = T / T.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-9)[:, None, None]
    desc = Vs[:, 6:] / Wm.clamp_min(1e-12)[:, None]; desc[:, :2] = desc[:, :2] / zf                 # sizes in the crop's voxels
    desc[:, :2] = desc[:, :2] * torch.linalg.det(Jq[eq]).abs().clamp_min(1e-6).pow(-1.0 / 3)[:, None]   # and in the output's (the warp's stretch)
    # visible mass; negligible entities (window edges, masked tissue) dropped; pad per query, heaviest first
    score = mass * Vis
    tot = torch.zeros(Q, device=dev).index_add_(0, eq, score)
    big = (score >= m_min) | ((score >= share_keep * tot[eq]) & (score > 1e-7))
    eq, en, mass, Vis, T, desc, score = eq[big], en[big], mass[big], Vis[big], T[big], desc[big], score[big]
    o1 = torch.argsort(-score); order = o1[torch.argsort(eq[o1], stable=True)]                  # by query, then visible mass descending
    eq, en, mass, Vis, T, desc = eq[order], en[order], mass[order], Vis[order], T[order], desc[order]
    cnt = torch.bincount(eq, minlength=Q); first = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(eq.numel(), device=dev) - first[eq]
    sel = rank < emax; e_, r_ = eq[sel], rank[sel]
    out = {"node": torch.full((Q, emax), -1, dtype=torch.long, device=dev), "mass": torch.zeros(Q, emax, device=dev),
           "vis": torch.zeros(Q, emax, device=dev), "T": torch.zeros(Q, emax, 3, 3, device=dev), "desc": torch.zeros(Q, emax, 5, device=dev)}
    out["node"][e_, r_] = en[sel]; out["mass"][e_, r_] = mass[sel]; out["vis"][e_, r_] = Vis[sel]
    out["T"][e_, r_] = T[sel]; out["desc"][e_, r_] = desc[sel]
    return out


def _single_object(qi: torch.Tensor, node: torch.Tensor, leaf: torch.Tensor, m: torch.Tensor, m_min: float, n_nodes: int) -> torch.Tensor:
    """Per (query, row) pair: where the (query, resolved node) group holds exactly one leaf with mass >= ``m_min``,
    that leaf instead of the node (its other rows, below ``m_min``, stay with the node and are dropped later)."""
    merged = node != leaf
    if not bool(merged.any()):
        return node
    gk = qi * (n_nodes + 1) + node                                                               # (query, node) groups
    # (group, leaf) pairs as one integer key: a 1-D unique instead of a unique over rows
    L = int(leaf[merged].max()) + 1
    u, inv = torch.unique(gk[merged].long() * L + leaf[merged].long(), return_inverse=True)
    trip = torch.stack([u // L, u % L], 1)
    tm = torch.zeros(trip.shape[0], device=m.device).index_add_(0, inv, m[merged])                # each leaf's mass in the window
    trip = trip[tm >= m_min]                                                                     # distinct significant leaves per group
    if trip.shape[0] == 0:
        return node
    ug, cnt = torch.unique(trip[:, 0], return_counts=True)
    one = ug[cnt == 1]
    if one.numel() == 0:
        return node
    lone = trip[torch.isin(trip[:, 0], one)]                                                     # (group, its single leaf)
    i = torch.searchsorted(lone[:, 0].contiguous(), gk).clamp(max=lone.shape[0] - 1)
    hit = merged & (lone[i, 0] == gk) & (lone[i, 1] == leaf)
    return torch.where(hit, leaf, node)


def dense_entities(occ: torch.Tensor, T6: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor, vis: torch.Tensor | None = None,
                   ovis: torch.Tensor | None = None, desc: torch.Tensor | None = None, m_min: float = 1e-3) -> dict:
    """Layers drawn without objects (a background's cell stain, the interfaces between regions): one population-level
    entity each, from the crop's own fields (output space, after the warp) -- so the entities together hold everything the
    total density holds. ``occ (K, Z, Y, X)``, ``T6 (K, 6, Z, Y, X)`` (trace 1), ``vis`` / ``ovis`` ``(K, Z, Y, X)`` or
    ``(K,)``, ``desc (K, 5)``; at ``points (Q, 3)`` and ``sigma (Q,)`` (voxels): ``mass``, ``vis (Q, K)``, ``T (Q, K, 3,
    3)``, ``desc (Q, K, 5)``, ``live (Q, K)``. The Gaussian window is evaluated at sigma steps of sqrt(2) and
    interpolated in log sigma."""
    from ..filters import gaussian
    K = occ.shape[0]; Q = points.shape[0]; dev = points.device
    full = lambda x: x.float().view(-1, 1, 1, 1).expand(K, *occ.shape[1:]) if x.dim() == 1 else x.float()
    o = occ.float()
    Tb = T6.float()
    if ovis is not None:
        ov = full(ovis).clamp(0, 1)[:, None]; iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=dev).view(1, 6, 1, 1, 1) / 3
        Tb = ov * Tb + (1 - ov) * iso
    F = torch.cat([o[:, None], o[:, None] * Tb], 1).reshape(K * 7, *occ.shape[1:])                  # (K * 7, Z, Y, X)
    s_lo, s_hi = float(sigma.min()), float(sigma.max())
    n_lv = max(2, int(math.ceil(math.log(max(s_hi / s_lo, 1.0001)) / math.log(math.sqrt(2)))) + 1)
    lv = torch.exp(torch.linspace(math.log(s_lo), math.log(max(s_hi, s_lo * 1.0001)), n_lv, device=dev))
    S = torch.stack([_vis_at(gaussian(F, float(sl_)), points.float()) for sl_ in lv])            # (n_lv, Q, K * 7)
    t = (torch.log(sigma.float()) - math.log(float(lv[0]))) / (math.log(float(lv[-1])) - math.log(float(lv[0]))) * (n_lv - 1)
    i0 = t.floor().long().clamp(0, n_lv - 2); a = (t - i0.float()).clamp(0, 1)[:, None]
    qi = torch.arange(Q, device=dev)
    Sq = ((1 - a) * S[i0, qi] + a * S[i0 + 1, qi]).view(Q, K, 7)
    mass = Sq[..., 0].clamp_min(0)
    T6q = Sq[..., 1:] / mass.clamp_min(1e-12)[..., None]
    T = torch.stack([torch.stack([T6q[..., 0], T6q[..., 3], T6q[..., 4]], -1), torch.stack([T6q[..., 3], T6q[..., 1], T6q[..., 5]], -1),
                     torch.stack([T6q[..., 4], T6q[..., 5], T6q[..., 2]], -1)], -2)
    T = T / T.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-9)[..., None, None]
    V = _vis_at(full(vis), points.float()) if vis is not None else torch.ones(Q, K, device=dev)
    dsc = (desc.float()[None].expand(Q, K, 5) if desc is not None else torch.zeros(Q, K, 5, device=dev)).clone()
    return {"mass": mass, "vis": V, "T": T, "desc": dsc, "live": mass * V >= m_min}


def _vis_at(vis: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """The layers' visibility ``(K, Z, Y, X)`` at points ``(Q, 3)`` (voxels): ``(Q, K)``."""
    K, Z, Y, X = vis.shape
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=points.device)
    g = (points * scale - 1.0).flip(-1).view(1, -1, 1, 1, 3)
    return torch.nn.functional.grid_sample(vis.float()[None], g, align_corners=True, padding_mode="border")[0, :, :, 0, 0].T


OVIS_ANGLE = (10.0, 40.0)           # orientation visibility: 1 within 10 deg of the axis, 0 beyond 40


def orientation_visibility(img: torch.Tensor, pts: torch.Tensor, sig: torch.Tensor, R: torch.Tensor, kappa: torch.Tensor,
                           mass: torch.Tensor, angle=None) -> torch.Tensor:
    """How far the acquired image ``img (Z, Y, X)`` shows each query's orientation target: ``(Q,)`` in 0-1.

    The structure tensor in the query's own window (gradients at sigma 1, their products averaged over a Gaussian
    window of ``sig``) is compared with the dominant line entity's axis (``R[..., 2]`` of the heaviest entity with
    ``kappa2 >= 2``), which is the tensor's weakest direction. At a crossing core (a second line entity with at least
    half the first's mass, at least ~17 degrees from it) the reference is the two axes' plane normal, the tensor's
    strongest direction. 1 within ``angle[0]`` degrees, 0 beyond ``angle[1]``; 1 where no entity is a line. The
    targets use it so that they do not ask for lobes the image does not show."""
    from ..filters import gaussian
    from .model import window_mean
    angle = angle or OVIS_ANGLE
    x = img.float(); dev = pts.device; Q = pts.shape[0]
    gz = gaussian(x, (1.0,) * 3, order=(1, 0, 0)); gy = gaussian(x, (1.0,) * 3, order=(0, 1, 0)); gx = gaussian(x, (1.0,) * 3, order=(0, 0, 1))
    pr = torch.stack([gz * gz, gy * gy, gx * gx, gz * gy, gz * gx, gy * gx])                      # (6, Z, Y, X)
    j6 = window_mean(pr, pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T   # (Q, 6)
    M = torch.stack([torch.stack([j6[:, 0], j6[:, 3], j6[:, 4]], -1), torch.stack([j6[:, 3], j6[:, 1], j6[:, 5]], -1),
                     torch.stack([j6[:, 4], j6[:, 5], j6[:, 2]], -1)], 1)
    _, V = torch.linalg.eigh(M.double() + 1e-12 * torch.eye(3, dtype=torch.float64, device=dev))
    weak = V[..., :, 0].float()                                                                    # the weakest direction: a line's axis
    strong = V[..., :, 2].float()                                                                  # the strongest direction: a crossing plane's normal
    mo = mass.float() * (kappa[..., 1] >= 2.0)                                                    # the line entities
    srt, o = mo.sort(1, descending=True); ar = torch.arange(Q, device=dev)
    ax1 = R[ar, o[:, 0]][:, :, 2]
    if mo.shape[1] > 1:
        ax2 = R[ar, o[:, 1]][:, :, 2]; nrm = torch.linalg.cross(ax1, ax2)
        core = (srt[:, 1] >= 0.5 * srt[:, 0]) & (srt[:, 1] > 0.02) & (nrm.norm(dim=-1) > 0.3)
        ref = torch.where(core[:, None], torch.nn.functional.normalize(nrm, dim=-1), ax1)
    else:
        ref = ax1
    ref_e = torch.where(core[:, None], strong, weak) if mo.shape[1] > 1 else weak
    ang = torch.rad2deg(torch.acos((ref_e * ref).sum(-1).abs().clamp(max=1)))
    v = (1 - (ang - angle[0]) / (angle[1] - angle[0])).clamp(0, 1)
    return torch.where(srt[:, 0] > 0.02, v, torch.ones_like(v))
