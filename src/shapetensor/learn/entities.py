"""Entities: a scene's objects, the populations they form at any depth, and the entity-field model's targets.

A scene's canvas keeps, per (voxel, population, object), the object's volume there and the volume-weighted sums of its
shape tensor and descriptors (:meth:`~shapetensor.learn.scene.Canvas.entries`) -- so an entity's tensor is the exact sum
of its objects', with the populations' own trilinear weights.

**The tree.** A leaf is an object within a population (the same object can lie in two populations: a vessel tree's two
branch sides, a split tract's halves). Its chain of ancestors follows the scene's parent links (fibre -> bundle,
nucleus -> cell, fibre -> fascicle -> nerve, ...); the chain's top joins its population node, and the populations join
the scene's root. An ancestor shared by several populations sits above them (the split tract: fibre -> half -> tract).
Any depth; each node carries the radius of what it is (objects: their smallest radius; populations and above: none).

**Resolution.** At a window of scale sigma an object is its own entity when its radius is at least
``max(r_min, c * sigma)`` (in the crop's voxels); otherwise it merges into its parent, repeatedly -- at fine sigma in
resolved data entities are single objects, at coarse sigma sub-populations and populations.

**Targets at query points** (:func:`entity_targets`): for each query point x (the output crop's voxels, after the warp)
and scale sigma, the entries in a Gaussian window around the point the warp pulls x from are grouped by their resolved
entity; each entity gets its mass (volume fraction in the window, times its population's visibility at x), its shape
tensor (rotated by the warp's Jacobian, trace 1 -- the Bingham target is moment-matched from it) and its descriptors.
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
                   bin_size: int = 2, mask: torch.Tensor | None = None, keep: torch.Tensor | None = None, m_min: float = 1e-3) -> dict:
    """The entities at query ``points (Q, 3)`` (output crop voxels) and scales ``sigma (Q,)`` (crop voxels).

    ``entries`` the canvas's ``(vox, group, obj, w, v)`` (canvas linear index; ``v`` the w-weighted tensor (6) and
    descriptors (5)); ``origin`` / ``zoom`` / ``shape`` the crop on the canvas (``shape`` the output crop's); ``grid`` the
    warp's sampling grid (None: no warp); ``vis (K, Z, Y, X)`` the layers' visibility in the output crop and
    ``group_layer`` the layer of each population (-1: none); ``mask (Z, Y, X)`` the crop's tissue (specimen edge, cut-outs:
    tissue removed takes its entities with it) and ``keep`` the labelled objects (sparse labelling: the others are gone);
    entities whose visible mass is below ``m_min`` are dropped. Returns, padded to ``emax`` entities per query (by mass):
    ``node (Q, E)`` (tree nodes, -1 pad), ``mass``, ``vis``, ``T (Q, E, 3, 3)`` (trace 1), ``desc (Q, E, 5)``."""
    vox, grp, obj, w, v = entries
    dev = points.device; Q = points.shape[0]; zf = float(zoom)
    canvas = torch.as_tensor(canvas_shape, device=dev)
    # entries -> crop coordinates (a crop voxel's centre sits at origin + zoom j + (zoom - 1) / 2 on the canvas)
    vox = vox.long(); cz = vox // (canvas[1] * canvas[2]); cy = (vox // canvas[2]) % canvas[1]; cx = vox % canvas[2]
    o = torch.as_tensor(origin, device=dev, dtype=torch.float32)
    c = (torch.stack([cz, cy, cx], 1).float() - o - (zf - 1) / 2) / zf
    xp, Jq = _pulled(points.float(), grid, shape)
    smax = float(sigma.max()); m = 3 * smax + bin_size
    lo = xp.min(0).values - m; hi = xp.max(0).values + m
    sel_ = ((c >= lo) & (c <= hi)).all(1)
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
    thr = zf * torch.clamp(c_sig * sg, min=r_min)
    node = tree.resolve(rleaf[ri], thr)
    gw = g * rw[ri]
    vl = torch.ones_like(gw)
    if vis is not None and group_layer is not None:
        gl = torch.as_tensor(group_layer, device=dev)[rgrp[ri]]
        vq = _vis_at(vis, points.float())                                                         # (Q, K)
        vl = torch.where(gl >= 0, vq[qi, gl.clamp_min(0)], torch.ones_like(gw))
    ek = qi * (tree.n + 1) + node
    ue, einv = torch.unique(ek, return_inverse=True)
    eq = ue // (tree.n + 1); en = ue % (tree.n + 1)
    Wm = torch.zeros(ue.shape[0], device=dev).index_add_(0, einv, gw)
    Vs = torch.zeros(ue.shape[0], v.shape[1], device=dev).index_add_(0, einv, g[:, None] * rv[ri])
    Vis = torch.zeros(ue.shape[0], device=dev).index_add_(0, einv, gw * vl) / Wm.clamp_min(1e-12)
    norm = zf ** 3 * (2 * math.pi * sigma[eq] ** 2) ** 1.5
    mass = Wm / norm
    T6 = Vs[:, :6] / Wm.clamp_min(1e-12)[:, None]
    T = torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], -1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], -1),
                     torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], -1)], 1)
    Ji = torch.linalg.inv(Jq[eq])
    T = Ji @ T @ Ji.transpose(1, 2); T = T / T.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-9)[:, None, None]
    desc = Vs[:, 6:] / Wm.clamp_min(1e-12)[:, None]; desc[:, :2] = desc[:, :2] / zf                 # sizes in the crop's voxels
    # visible mass; negligible entities (window edges, masked tissue) dropped; pad per query, heaviest first
    score = mass * Vis
    big = score >= m_min
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


def _vis_at(vis: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """The layers' visibility ``(K, Z, Y, X)`` at points ``(Q, 3)`` (voxels): ``(Q, K)``."""
    K, Z, Y, X = vis.shape
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=points.device)
    g = (points * scale - 1.0).flip(-1).view(1, -1, 1, 1, 3)
    return torch.nn.functional.grid_sample(vis.float()[None], g, align_corners=True, padding_mode="border")[0, :, :, 0, 0].T
