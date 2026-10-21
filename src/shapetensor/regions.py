"""Regions in the shape space: bounded, not partitioned. Classical statistics on anatomix-dev-vit features.

On a lattice of one cell per 4³ voxels (80 µm at 20 µm):

1. **Features** -- dev-vit's first decoder stage (216 channels, native at that resolution) and its output (32 channels,
   averaged over each cell), both blended across tiles and free of the patch grid (``DevVitEngine.decoder1`` and the
   engine's output); standardised, 50 principal components fitted on the calibration samples.
2. **Calibration samples** -- cells where the excised structure tensor is confident (its gap above one threshold for
   the whole pool, so a block of crossings, where the tensor reports a confident-looking bisector rarely, contributes
   few), labelled with its direction; and cells of synthetic fibre-free texture, labelled *texture*.
3. **Assignment by nearest neighbours** (k in the PCA space): a cell is *texture* when most of its neighbours are, else
   *oriented*, with the direction the principal axis of its (oriented) neighbours' mean dyad and their agreement as
   confidence, the neighbours for the direction looked up in a Fisher discriminant space of the direction (linear
   discriminant analysis over ``n_bins`` hemisphere bins of the samples' directions: the features' variation that
   tells directions apart, over the variation within a direction);
   but only if it lies within the region the calibration samples occupy -- its k-th neighbour no farther than the
   ``reject_q`` quantile of the samples' own k-th-neighbour distances, each measured among samples from *other*
   regions only -- other blocks, and other ``group_cells``-cell cubes of the same block (neighbouring cells of one
   image are correlated; a new region's cells are as far from the calibration as another region's are). Beyond that it
   is *unassigned*: fibre-like
   (most neighbours oriented, a likely crossing or anything the calibration never saw) or texture-like.
4. **Field** -- the cells' dyads, weighted by their oriented share and zero where unassigned, interpolated to every voxel.

No partition runs out to infinity: a voxel unlike any calibration sample gets no direction.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .pipeline import teacher
from .separate import sample_idx
from .tensors import direction_and_gap

TEXTURE, ORIENTED, UNASSIGNED_FIBRE, UNASSIGNED_TEXTURE = 0, 1, 2, 3


def lattice_features(eng, v: torch.Tensor) -> torch.Tensor:
    """``(248, z, y, x)``: decoder 1 and the output averaged over each 4³ cell (the output padded to whole cells)."""
    f1 = eng.decoder1(v)
    fo = eng(v).float()
    pad = [0, 0, 0, 0, 0, 0]
    for ax in range(3):
        r = (-fo.shape[1 + ax]) % 4; pad[2 * (2 - ax) + 1] = r
    fo = Fn.avg_pool3d(Fn.pad(fo[None], pad, mode="replicate"), 4)[0]
    return torch.cat([f1, fo[:, :f1.shape[1], :f1.shape[2], :f1.shape[3]]])


def _cell_dyads(d: torch.Tensor, w: torch.Tensor, cshape) -> tuple[torch.Tensor, torch.Tensor]:
    """The weighted mean dyad of directions ``d (3, Z, Y, X)`` over each 4³ cell: its principal direction and coherence."""
    M = torch.stack([d[i] * d[j] * w for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
    pad = [0, 0, 0, 0, 0, 0]
    for ax in range(3):
        pad[2 * (2 - ax) + 1] = (-d.shape[1 + ax]) % 4
    Mc = Fn.avg_pool3d(Fn.pad(M[None], pad), 4)[0][:, :cshape[0], :cshape[1], :cshape[2]]
    wc = Fn.avg_pool3d(Fn.pad(w[None, None], pad), 4)[0, 0][:cshape[0], :cshape[1], :cshape[2]]
    Mc = Mc / wc.clamp_min(1e-12)[None]
    full = torch.stack([torch.stack([Mc[0], Mc[3], Mc[4]]), torch.stack([Mc[3], Mc[1], Mc[5]]), torch.stack([Mc[4], Mc[5], Mc[2]])])   # (3, 3, ...)
    A = full.permute(*range(2, full.dim()), 0, 1)                                                     # (..., 3, 3)
    val, vec = torch.linalg.eigh(A.double())
    return vec[..., :, -1].float().permute(-1, *range(vec.dim() - 2)), ((val[..., -1] - val[..., -2]) / val[..., -1].clamp_min(1e-12)).float()


def _knn(Q: torch.Tensor, Y: torch.Tensor, k: int, chunk: int = 8192, exclude_self: bool = False, q_group=None, y_group=None):
    """Indices and distances of the ``k`` nearest rows of ``Y`` for every row of ``Q`` (Euclidean); with groups, rows
    of ``Y`` in the query's own group are excluded."""
    I, Dd = [], []
    for i in range(0, Q.shape[0], chunk):
        d = torch.cdist(Q[i:i + chunk], Y)
        if exclude_self:
            d[torch.arange(d.shape[0], device=d.device), torch.arange(i, i + d.shape[0], device=d.device)] = float("inf")
        if q_group is not None:
            d = d.masked_fill(q_group[i:i + chunk, None] == y_group[None, :], float("inf"))
        dist, idx = d.topk(k, dim=1, largest=False)
        I.append(idx); Dd.append(dist)
    return torch.cat(I), torch.cat(Dd)


class RegionModel:
    def __init__(self, k: int = 15, n_pcs: int = 50, reject_q: float = 0.99, gap_q: float = 0.7, per_block: int = 4000, seed: int = 0,
                 sigma: float = 1.0, rho: float = 10.0, truncate: float = 3.0, margin: int = 32, n_bins: int = 32, n_lda: int = 16,
                 group_cells: int = 32):
        self.k, self.n_pcs, self.reject_q, self.gap_q, self.per_block, self.seed = int(k), int(n_pcs), float(reject_q), float(gap_q), int(per_block), int(seed)
        self.n_bins, self.n_lda, self.group_cells = int(n_bins), int(n_lda), int(group_cells)
        self.sigma, self.rho, self.truncate, self.margin = sigma, rho, truncate, int(margin)

    def fit(self, eng, volumes, tissues=None, iso=()):
        """``volumes``: blocks ``(Z, Y, X)`` (any mix of tissue: their confident cells are the oriented samples);
        ``tissues``: bool masks (None: all); ``iso``: fibre-free texture blocks (the texture samples)."""
        dev = volumes[0].device; g = torch.Generator(device="cpu").manual_seed(self.seed)
        tissues = tissues or [None] * len(volumes)
        cells = []
        for v, t in zip(volumes, tissues):
            L = lattice_features(eng, v); cs = L.shape[1:]
            d, gap = teacher(v, self.sigma, self.rho, self.truncate)
            w = torch.ones_like(gap) if t is None else t.float()
            dc, coh = _cell_dyads(d, w * gap, cs)
            gc = Fn.avg_pool3d(Fn.pad((gap * w)[None, None], [0, (-gap.shape[2]) % 4, 0, (-gap.shape[1]) % 4, 0, (-gap.shape[0]) % 4]), 4)[0, 0][:cs[0], :cs[1], :cs[2]]
            m = self.margin // 4
            inner = torch.zeros(cs, dtype=torch.bool, device=dev); inner[m:cs[0] - m, m:cs[1] - m, m:cs[2] - m] = True
            if t is not None:
                tc = Fn.avg_pool3d(Fn.pad(t.float()[None, None], [0, (-t.shape[2]) % 4, 0, (-t.shape[1]) % 4, 0, (-t.shape[0]) % 4]), 4)[0, 0][:cs[0], :cs[1], :cs[2]]
                inner &= tc > 0.5
            cells.append((L, dc, gc, inner))
        allg = torch.cat([gc[inner] for _, _, gc, inner in cells])
        thr = float(torch.quantile(allg[torch.randperm(allg.numel(), generator=g)[:2_000_000].to(allg.device)], self.gap_q))
        X, Dl, B = [], [], []
        gcell = self.group_cells
        def groups(cs, base):                                                                      # a group id per cell: its block and its cube
            zz, yy, xx = (torch.arange(n, device=dev) // gcell for n in cs)
            ny, nx = -(-cs[1] // gcell), -(-cs[2] // gcell)
            return base + (zz[:, None, None] * ny + yy[None, :, None]) * nx + xx[None, None, :]
        base = 0
        for b, (L, dc, gc, inner) in enumerate(cells):
            ok = (inner & (gc >= thr)).flatten()
            idx = sample_idx(ok.float(), self.per_block, g)
            grp = groups(L.shape[1:], base).flatten(); base = int(grp.max()) + 1
            X.append(L.reshape(L.shape[0], -1)[:, idx].T); Dl.append(dc.reshape(3, -1)[:, idx].T); B.append(grp[idx])
        n_or = sum(len(x) for x in X)
        for j, t in enumerate(iso):
            L = lattice_features(eng, t); cs = L.shape[1:]; m = self.margin // 4
            Li = L[:, m:cs[0] - m, m:cs[1] - m, m:cs[2] - m].reshape(L.shape[0], -1)
            idx = torch.randperm(Li.shape[1], generator=g)[:self.per_block].to(dev)
            X.append(Li[:, idx].T); Dl.append(torch.zeros(len(idx), 3, device=dev))
            grp = groups(tuple(Li.shape[1:]) if Li.dim() == 4 else (cs[0] - 2 * m, cs[1] - 2 * m, cs[2] - 2 * m), base).flatten(); base = int(grp.max()) + 1
            B.append(grp[idx])
        X = torch.cat(X); D = torch.cat(Dl); self.block = torch.cat(B)
        self.is_texture = torch.zeros(len(X), dtype=torch.bool, device=dev); self.is_texture[n_or:] = True
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xs = (X - self.mean) / self.std
        U, S, V = torch.pca_lowrank(Xs, q=min(self.n_pcs, Xs.shape[1]), center=False, niter=4)
        self.V = V[:, :self.n_pcs]
        self.Y = Xs @ self.V; self.D = D
        _, dist = _knn(self.Y, self.Y, self.k, q_group=self.block, y_group=self.block)                 # leave one block out
        self.radius = float(torch.quantile(dist[:, -1], self.reject_q))
        # the direction metric: Fisher discriminant over hemisphere bins of the oriented samples' directions
        from .separate import hemisphere
        grid = hemisphere(self.n_bins, dev)
        Yo = self.Y[:n_or]; lab = (D[:n_or] @ grid.T).abs().argmax(1)
        mu = Yo.mean(0); Sw = torch.zeros(Yo.shape[1], Yo.shape[1], device=dev); Sb = torch.zeros_like(Sw)
        for c in lab.unique():
            Yc = Yo[lab == c]; mc = Yc.mean(0)
            Sw += (Yc - mc).T @ (Yc - mc); Sb += len(Yc) * torch.outer(mc - mu, mc - mu)
        Sw = Sw / len(Yo) + 1e-3 * torch.trace(Sw / len(Yo)) / Sw.shape[0] * torch.eye(Sw.shape[0], device=dev); Sb = Sb / len(Yo)
        w, Vw = torch.linalg.eigh(Sw.double()); Wi = (Vw @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ Vw.T).float()
        e, Ve = torch.linalg.eigh((Wi @ Sb @ Wi).double())
        self.L = (Wi @ Ve[:, -self.n_lda:].float())                                                       # (n_pcs, n_lda): whitened within, top between
        self.Yl = Yo @ self.L; self.n_or = n_or
        self.report = {"oriented samples": n_or, "texture samples": int(len(X) - n_or), "gap threshold": round(thr, 4),
                       "pca variance kept": round(float((S[:self.n_pcs] ** 2).sum() / (Xs ** 2).sum()), 3), "radius": round(self.radius, 3)}
        return self

    def assign(self, L: torch.Tensor) -> dict:
        """Lattice features ``(C, z, y, x)`` -> per cell: ``state`` (TEXTURE / ORIENTED / UNASSIGNED_FIBRE / UNASSIGNED_TEXTURE),
        ``direction (3, ...)``, ``coherence`` of the neighbours' directions, ``texture_share``, ``distance_ratio`` (k-th
        neighbour over the radius)."""
        cs = L.shape[1:]
        Q = ((L.reshape(L.shape[0], -1).T - self.mean) / self.std) @ self.V
        idx, dist = _knn(Q, self.Y, self.k)
        tex = self.is_texture[idx].float()
        tshare = tex.mean(1)
        idl, _ = _knn(Q @ self.L, self.Yl, self.k)                                                   # direction neighbours (oriented samples, Fisher space)
        Dn = self.D[:self.n_or][idl]                                                                   # (M, k, 3)
        M = torch.einsum("mki,mkj->mij", Dn, Dn) / self.k
        val, vec = torch.linalg.eigh(M.double())
        direction = vec[:, :, -1].float(); coh = ((val[:, -1] - val[:, -2]) / val[:, -1].clamp_min(1e-12)).float()
        ratio = dist[:, -1] / self.radius
        inside = ratio <= 1.0
        state = torch.where(inside, torch.where(tshare >= 0.5, TEXTURE, ORIENTED), torch.where(tshare >= 0.5, UNASSIGNED_TEXTURE, UNASSIGNED_FIBRE))
        r = lambda x: x.reshape(cs) if x.dim() == 1 else x.T.reshape((x.shape[1],) + tuple(cs))
        return {"state": r(state), "direction": r(direction), "coherence": r(coh), "texture_share": r(tshare), "distance_ratio": r(ratio)}

    def field(self, eng, v: torch.Tensor) -> dict:
        """The block's lattice assignment and the field at every voxel: ``vec (3, Z, Y, X)`` from the cells' dyads
        (weight: oriented share, zero where unassigned) interpolated trilinearly, ``weight (Z, Y, X)``."""
        L = lattice_features(eng, v); a = self.assign(L)
        d = a["direction"]; w = ((1 - a["texture_share"]) * (a["state"] == ORIENTED).float())
        M = torch.stack([d[2] * d[2], d[1] * d[1], d[0] * d[0], d[2] * d[1], d[2] * d[0], d[1] * d[0]]) * w[None]
        tr = M[0] + M[1] + M[2]
        S = -M; S[0] += tr; S[1] += tr; S[2] += tr
        size = tuple(int(n) for n in v.shape)
        up = lambda x: Fn.interpolate(x[None], scale_factor=4, mode="trilinear", align_corners=False)[0][..., :size[0], :size[1], :size[2]]
        Sf = up(S); wf = up(w[None])[0]
        vec, gap = direction_and_gap(Sf)
        return {"vec": vec, "gap": gap, "weight": wf, "lattice": a}
