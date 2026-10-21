"""Stage 2 prototypes: populations as sets of fibres, not as directions.

In the pipeline a population is one direction: a mode of the block's orientation density, with an activation layer
tuned to that direction's signature. A bundle that turns through more than the separation angle inside a block is
therefore split between populations (the bend phantom). Here a population is defined by *continuity in space*
instead, with no number of populations to choose:

* :func:`connected_sets` — the readout's per-voxel orientation distribution lifted to (position, direction): every
  significant peak of a voxel's distribution is a node; a node is linked to the nodes a short step along its own
  direction whose direction differs by no more than a turn of radius ``r_min_mm`` allows (plus the readout's noise);
  a population is a connected set of nodes. A bend is one set (its direction changes gradually along its path), a
  crossing two (where they meet the two directions differ by more than any allowed turn).
* :func:`smooth_fields` — a population is a direction field that may rotate across space, with a weight: an
  expectation–maximisation fit of the voxels' orientation distributions to Watson components whose mean directions
  are smoothed in space (the field), plus a uniform background. A new field is seeded from the mass no field explains
  and kept only where it coexists with another field ``split_deg`` away over a real volume, so the number of fields
  follows the data; each field's support is then cut into connected parts (sets of fibres).

Both return activation layers ``(N, Z, Y, X)`` in multiples of the null for the unchanged stages 3–4
(:func:`fields_from_layers`). The evidence is the orientation readout of :mod:`shapetensor.separate`, smoothed in
space (:func:`lift`); the null is the same readout on texture without fibres (:func:`null_level`).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from .. import st
from ..filters import gaussian
from ..merge import multi_field, single_field
from ..tensors import population_tensor


# ---------------------------------------------------------------------------------------------------------------
# the evidence: the readout's orientation distributions, lifted and smoothed
# ---------------------------------------------------------------------------------------------------------------

def lift(cal, F, sigma: float = 2.0, stride: int = 2, chunk: int = 16) -> torch.Tensor:
    """``U (n_dirs, Z', Y', X')``: the readout's per-voxel orientation distribution of features ``F``, smoothed over
    ``sigma`` voxels in space and sampled every ``stride`` voxels (each voxel's distribution sums to 1)."""
    P = cal.readout.predict_dist(F)
    out = []
    for a in range(0, P.shape[0], chunk):
        q = P[a:a + chunk]
        if sigma > 0:
            q = gaussian(q, sigma, truncate=3.0)
        out.append(q[:, ::stride, ::stride, ::stride].contiguous())
    del P
    return torch.cat(out)


def _near(u: torch.Tensor, half_angle: float) -> torch.Tensor:
    return ((u @ u.T).abs() >= math.cos(math.radians(half_angle))).float()


def peak_mass(U: torch.Tensor, u: torch.Tensor, half_angle: float = 20.0) -> torch.Tensor:
    """The distribution's mass within ``half_angle`` of every grid direction, ``(n_dirs, ...)``."""
    D = U.shape[0]
    return (_near(u, half_angle) @ U.reshape(D, -1)).reshape(U.shape)


def peaks(U: torch.Tensor, u: torch.Tensor, null: float, rel: float = 0.5, half_angle: float = 20.0, k: int = 6) -> torch.Tensor:
    """Each voxel's significant directions: local maxima of ``U`` on the direction grid (over its ``k`` nearest axial
    neighbours) whose mass within ``half_angle`` is at least the ``null`` and ``rel`` × the voxel's largest. Returns
    that mass where a direction is a peak and 0 elsewhere, ``(n_dirs, ...)``."""
    D = U.shape[0]
    S = (u @ u.T).abs(); S.fill_diagonal_(-1.0)
    nb = S.topk(k, dim=1).indices                                                          # (D, k)
    Uf = U.reshape(D, -1)
    M = (_near(u, half_angle) @ Uf)
    top = M.max(0).values
    out = torch.zeros_like(M)
    step = 200_000
    for a in range(0, Uf.shape[1], step):
        uf = Uf[:, a:a + step]
        ismax = (uf[:, None, :] >= uf[nb]).all(1)
        m = M[:, a:a + step]
        out[:, a:a + step] = torch.where(ismax & (m >= null) & (m >= rel * top[a:a + step]), m, torch.zeros_like(m))
    return out.reshape(U.shape)


def null_level(cal, null_volume: torch.Tensor, sigma: float = 2.0, stride: int = 2, q: float = 0.95, half_angle: float = 20.0,
               margin: int = 16) -> float:
    """The ``q`` quantile of the largest peak mass per voxel on texture without fibres (``null_volume``), away from its
    faces: what the readout reports where there is no fibre."""
    U = lift(cal, cal.features(null_volume), sigma, stride)
    m = peak_mass(U, cal.readout.grid, half_angle).max(0).values
    c = max(1, margin // stride)
    m = m[c:-c, c:-c, c:-c]
    return float(m.reshape(-1)[:: max(1, m.numel() // 2_000_000)].float().quantile(q))


def _upsample(A: torch.Tensor, shape, stride: int, sigma: float = 1.0) -> torch.Tensor:
    """Layers on the strided grid ``(N, Z', Y', X')`` -> the full grid ``(N, Z, Y, X)``: nearest, then a light blur."""
    if A.shape[0] == 0:
        return torch.zeros((0,) + tuple(shape), device=A.device)
    out = A
    for ax in (1, 2, 3):
        out = out.repeat_interleave(stride, dim=ax)
    out = out[:, : shape[0], : shape[1], : shape[2]]
    pad = [(0, s - o) for s, o in zip(shape, out.shape[1:])]
    if any(p for _, p in pad):
        out = Fn.pad(out, (0, pad[2][1], 0, pad[1][1], 0, pad[0][1]), mode="replicate")
    return gaussian(out.contiguous(), sigma, truncate=3.0) if sigma > 0 else out


# ---------------------------------------------------------------------------------------------------------------
# prototype 1: connected sets in (position, direction)
# ---------------------------------------------------------------------------------------------------------------

def connected_sets(U: torch.Tensor, u: torch.Tensor, tissue: torch.Tensor, null: float, *, vox_mm: float, stride: int = 2,
                   radius: int = 2, cone_deg: float = 35.0, r_min_mm: float = 0.25, noise_deg: float = 15.0,
                   min_volume_mm3: float = 0.005, rel: float = 0.5, half_angle: float = 20.0, max_slots: int = 4,
                   full_shape=None, layers: bool = True) -> tuple[torch.Tensor | None, dict]:
    """Populations as connected sets of (voxel, direction) nodes. ``U`` the lifted evidence on the strided grid,
    ``tissue`` on that grid. A node links to a node at offset ``o`` (within ``radius`` grid steps) when ``o`` lies within
    ``cone_deg`` of its own direction and their directions differ by at most ``|o| / r_min_mm`` (the turn a radius of
    ``r_min_mm`` makes over the step) plus ``noise_deg``. Sets smaller than ``min_volume_mm3`` are dropped. Returns the
    activation layers on the full grid (mass above the null, in nulls) and a report."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    dev = U.device
    D, Z, Y, X = U.shape
    Pk = peaks(U, u, null, rel=rel, half_angle=half_angle) * tissue[None].float()
    # at most ``max_slots`` peaks per voxel, strongest first (a technical bound; the report says how often it binds)
    vals, idx = Pk.reshape(D, -1).topk(max_slots, dim=0)                                    # (S, V)
    valid = vals > 0
    n_over = int(((Pk.reshape(D, -1) > 0).sum(0) > max_slots).sum())
    ids = torch.full(valid.shape, -1, dtype=torch.int64, device=dev)
    n_nodes = int(valid.sum())
    ids[valid] = torch.arange(n_nodes, device=dev)
    dirs = u[idx]                                                                          # (S, V, 3)
    S_ = max_slots
    ids = ids.reshape(S_, Z, Y, X); dirs = dirs.reshape(S_, Z, Y, X, 3); vals = vals.reshape(S_, Z, Y, X)
    cos_cone = math.cos(math.radians(cone_deg))
    rows, cols = [], []
    rng = range(-radius, radius + 1)
    for oz in rng:
        for oy in rng:
            for ox in rng:
                if (oz, oy, ox) == (0, 0, 0) or (oz, oy, ox) < (0, 0, 0):                # each unordered pair once
                    continue
                o = torch.tensor([oz, oy, ox], dtype=torch.float32, device=dev)
                dist = float(o.norm())
                oh = o / dist
                cos_turn = math.cos(min(math.pi / 2, dist * stride * vox_mm / r_min_mm + math.radians(noise_deg)))
                sa = tuple(slice(max(0, -c), min(n, n - c)) for c, n in zip((oz, oy, ox), (Z, Y, X)))
                sb = tuple(slice(max(0, c), min(n, n + c)) for c, n in zip((oz, oy, ox), (Z, Y, X)))
                for a in range(S_):
                    ia = ids[(a,) + sa]
                    da = dirs[(a,) + sa]
                    along = (da @ oh).abs() >= cos_cone                                   # the step lies along a's fibre
                    if not bool(along.any()):
                        continue
                    for b in range(S_):
                        ib = ids[(b,) + sb]
                        db = dirs[(b,) + sb]
                        ok = along & (ia >= 0) & (ib >= 0) & ((da * db).sum(-1).abs() >= cos_turn) & ((db @ oh).abs() >= cos_cone)
                        if bool(ok.any()):
                            rows.append(ia[ok]); cols.append(ib[ok])
    if rows:
        r = torch.cat(rows).cpu().numpy(); c = torch.cat(cols).cpu().numpy()
    else:
        r = c = np.zeros(0, np.int64)
    G = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n_nodes, n_nodes))
    ncomp, lab = connected_components(G, directed=False)
    lab_t = torch.as_tensor(lab, device=dev)
    sizes = torch.bincount(lab_t, minlength=ncomp)
    node_vol = (stride * vox_mm) ** 3
    keep = torch.nonzero(sizes * node_vol >= min_volume_mm3).flatten()
    keep = keep[torch.argsort(sizes[keep], descending=True)]
    node_vol_ = node_vol
    if not layers:                                                                         # the sets' sizes only
        return None, {"nodes": n_nodes, "edges": int(len(r)), "sets_all": int(ncomp), "sets_kept": int(len(keep)),
                      "kept_volumes_mm3": [round(float(sizes[k]) * node_vol_, 4) for k in keep.tolist()][:12],
                      "voxels_over_max_slots": n_over}
    remap = torch.full((ncomp,), -1, dtype=torch.int64, device=dev); remap[keep] = torch.arange(len(keep), device=dev)
    N = len(keep)
    A = torch.zeros((N, Z * Y * X), device=dev)
    node_lab = remap[lab_t]
    vflat = vals.reshape(S_, -1); iflat = ids.reshape(S_, -1)
    for a in range(S_):
        sel = iflat[a] >= 0
        lv = torch.full_like(iflat[a], -1); lv[sel] = node_lab[iflat[a][sel]]
        m = lv >= 0
        vox = torch.nonzero(m).flatten()
        A.index_put_((lv[m], vox), (vflat[a][m] - null).clamp_min(0) / null, accumulate=True)
    A = A.reshape(N, Z, Y, X)
    report = {"nodes": n_nodes, "edges": int(len(r)), "sets_all": int(ncomp), "sets_kept": N,
              "kept_volumes_mm3": [round(float(sizes[k]) * node_vol, 4) for k in keep.tolist()][:12],
              "voxels_over_max_slots": n_over}
    full = full_shape or (Z * stride, Y * stride, X * stride)
    return _upsample(A, full, stride), report


# ---------------------------------------------------------------------------------------------------------------
# prototype 2: smooth direction fields
# ---------------------------------------------------------------------------------------------------------------

def _principal(M: torch.Tensor, which: int = -1) -> torch.Tensor:
    """Eigenvector ``which`` (``-1`` the largest eigenvalue's, ``-2`` the second's) of moment tensors ``M (..., 3, 3)``."""
    shp = M.shape[:-2]
    flat = torch.nan_to_num(M.reshape(-1, 3, 3).double())
    out = torch.empty(flat.shape[0], 3, dtype=torch.float32, device=M.device)
    step = 1 << 19                                                                         # cusolver's batched eigh caps the batch
    for a in range(0, flat.shape[0], step):
        _, V = torch.linalg.eigh(flat[a:a + step])
        out[a:a + step] = V[..., which].float()
    return out.reshape(*shp, 3)


def _smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian over the last three axes, any leading axes."""
    if sigma <= 0:
        return x
    shp = x.shape
    return gaussian(x.reshape(-1, *shp[-3:]).contiguous(), sigma, truncate=3.0).reshape(shp)


def _moments(U: torch.Tensor, R: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
    """``(K, 3, 3, Z, Y, X)``: the orientation mass each field is responsible for, as second moments."""
    D = U.shape[0]
    w = (U[None] * R).reshape(R.shape[0], D, -1)                                             # (K, D, V)
    return torch.einsum("kdv,dij->kijv", w, uu).reshape(R.shape[0], 3, 3, *U.shape[1:])


def _em(U, u, uu, tissue, d, pi, kappa, sigma_f, iters):
    """``iters`` rounds of the field fit: responsibilities against Watson kernels about each field's direction (plus a
    uniform background), mixing weights and spatially smoothed moments. Returns the directions ``(K, 3, ...)`` (z, y,
    x), the weights ``(K + 1, ...)`` (background last), the fields' smoothed moments and the background's
    responsibilities ``(D, ...)`` (the orientation mass no field explains)."""
    D = U.shape[0]
    for _ in range(iters):
        cos2 = torch.einsum("dc,kc...->kd...", u, d) ** 2                                   # (K, D, ...)
        W = torch.exp(kappa * (cos2 - 1))
        W = W / W.sum(1, keepdim=True).clamp_min(1e-12)
        num = torch.cat([pi[:-1, None] * W, (pi[-1:, None] / D).expand(1, D, *U.shape[1:])], 0)   # (K + 1, D, ...)
        R = num / num.sum(0, keepdim=True).clamp_min(1e-12)
        pi = _smooth((U[None] * R).sum(1), sigma_f)
        pi = pi / pi.sum(0, keepdim=True).clamp_min(1e-12)
        M = _smooth(_moments(U, R[:-1], uu), sigma_f)
        d = _principal(M.permute(0, 3, 4, 5, 1, 2)).permute(0, 4, 1, 2, 3).contiguous()
    pi = pi * tissue[None].float() + torch.cat([torch.zeros_like(pi[:-1]), (~tissue)[None].float()], 0)
    return d, pi, M, R[-1]


def _start(U, u, uu, sigma_f):
    """One field along the principal direction of the smoothed evidence, half the mass."""
    M0 = _smooth(torch.einsum("dv,dij->ijv", U.reshape(U.shape[0], -1), uu).reshape(3, 3, *U.shape[1:])[None], sigma_f)
    d = _principal(M0.permute(0, 3, 4, 5, 1, 2)).permute(0, 4, 1, 2, 3).contiguous()
    return d, torch.stack([torch.full(U.shape[1:], 0.5, device=U.device)] * 2)


def smooth_fields(U: torch.Tensor, u: torch.Tensor, tissue: torch.Tensor, null_pi: float, *, vox_mm: float, stride: int = 2,
                  kappa: float = 8.0, sigma_f: float = 2.0, iters: int = 12, split_deg: float = 25.0,
                  min_volume_mm3: float = 0.005, max_rounds: int = 6, full_shape=None, log=None,
                  layers: bool = True) -> tuple[torch.Tensor | None, dict]:
    """Populations as smooth direction fields, divided into connected sets.

    One field to start (the principal direction of the smoothed evidence, ``sigma_f`` grid steps). Then, round by
    round, a new field is seeded from the orientation mass no field explains (the principal direction of the
    background's smoothed moments) and the fit rerun; it is kept only where it is present (weight at least ``null_pi``)
    *together with* another present field at least ``split_deg`` away, over at least ``min_volume_mm3`` — a second
    population only where two directions coexist. Each kept field's support (weight at least ``null_pi``) is then cut
    into its connected parts, each part of at least ``min_volume_mm3`` a population: a field may span bundles that do
    not touch. ``U`` the lifted evidence on the strided grid; ``kappa`` the Watson concentration. Returns the
    activation layers on the full grid (weight above the null, in nulls) and a report."""
    from scipy import ndimage as ndi

    say = log or (lambda *_: None)
    U = U * tissue[None].float()
    uu = u[:, :, None] * u[:, None, :]                                                      # (D, 3, 3)
    node_vol = (stride * vox_mm) ** 3
    d, pi = _start(U, u, uu, sigma_f)
    d, pi, M, R0 = _em(U, u, uu, tissue, d, pi, kappa, sigma_f, iters)
    history = []
    for rnd in range(max_rounds):
        res = _smooth(_moments(U, R0[None], uu), sigma_f)[0]                              # the unexplained mass, (3, 3, ...)
        e = _principal(res.permute(2, 3, 4, 0, 1)).permute(3, 0, 1, 2)
        share = _smooth((U * R0).sum(0)[None], sigma_f)[0]
        d2 = torch.cat([d, e[None]])
        pi2 = torch.cat([pi[:-1], (share * 0.5)[None], (pi[-1] - share * 0.5).clamp_min(1e-6)[None]])
        pi2 = pi2 / pi2.sum(0, keepdim=True)
        d2, pi2, M2, R2 = _em(U, u, uu, tissue, d2, pi2, kappa, sigma_f, iters)
        new = d2.shape[0] - 1
        present = (pi2[:-1] >= null_pi) & tissue[None]
        cos_sep = math.cos(math.radians(split_deg))
        apart = torch.zeros_like(tissue)
        for k in range(new):
            apart |= present[k] & ((d2[new] * d2[k]).sum(0).abs() < cos_sep)
        both = present[new] & apart
        vol = float(both.sum()) * node_vol
        history.append({"round": rnd, "fields_before": int(d.shape[0]), "colocated_volume_mm3": round(vol, 4)})
        say(f"round {rnd}: a new field co-located with another over {vol:.4f} mm3")
        if vol < min_volume_mm3:
            break
        d, pi, M, R0 = d2, pi2, M2, R2
    stack, parts = [], []
    for k in range(d.shape[0]):
        sup = ((pi[k] >= null_pi) & tissue).cpu().numpy()
        lab, n = ndi.label(sup, structure=np.ones((3, 3, 3)))
        if n == 0:
            continue
        sizes = np.bincount(lab.reshape(-1), minlength=n + 1)[1:]
        if not layers:                                                                     # the sets' sizes only
            parts += [(k, round(float(z) * node_vol, 4)) for z in sorted(sizes, reverse=True) if z * node_vol >= min_volume_mm3][:12]
            continue
        lab_t = torch.as_tensor(lab, device=U.device)
        a = ((pi[k] - null_pi).clamp_min(0) / null_pi) * tissue.float()
        for j in np.argsort(-sizes):
            if sizes[j] * node_vol < min_volume_mm3:
                break
            stack.append(a * (lab_t == j + 1).float()); parts.append((k, round(float(sizes[j]) * node_vol, 4)))
    report = {"fields": int(d.shape[0]), "sets_kept": len(parts), "sets": sorted(parts, key=lambda t: -t[1])[:12], "rounds": history}
    if not layers:
        return None, report
    A = torch.stack(stack) if stack else torch.zeros((0,) + tuple(U.shape[1:]), device=U.device)
    full = full_shape or tuple(n * stride for n in U.shape[1:])
    return _upsample(A, full, stride), report


def field_null(U_null: torch.Tensor, u: torch.Tensor, *, kappa: float = 8.0, sigma_f: float = 2.0, iters: int = 12, q: float = 0.95,
               margin: int = 8) -> float:
    """The ``q`` quantile of a single field's weight fitted to evidence from texture without fibres."""
    tissue = torch.ones(U_null.shape[1:], dtype=torch.bool, device=U_null.device)
    uu = u[:, :, None] * u[:, None, :]
    d, pi = _start(U_null, u, uu, sigma_f)
    _, pi, _, _ = _em(U_null, u, uu, tissue, d, pi, kappa, sigma_f, iters)
    m = pi[0][margin:-margin, margin:-margin, margin:-margin]
    return float(m.reshape(-1).float().quantile(q))


# ---------------------------------------------------------------------------------------------------------------
# stages 3-4, unchanged, from any set of activation layers
# ---------------------------------------------------------------------------------------------------------------

def fields_from_layers(cal, v: torch.Tensor, tissue, A: torch.Tensor, *, tensor_floor: float = 0.01, merge_floor: float = 0.1,
                       spacing=None, relative: bool = False) -> dict:
    """The pipeline's stages 3–4 on activation layers ``A (N, Z, Y, X)`` (in nulls): a population tensor per layer (the
    excised structure tensor weighted by the layer), the merged single field and the multi-vector field, as
    :func:`shapetensor.pipeline.apply` returns them. With no layer, one empty layer (the excised structure tensor).
    ``relative``: each layer divided by the largest layer at each voxel first."""
    if A.shape[0] == 0:
        A = torch.zeros((1,) + tuple(v.shape), device=v.device)
    if relative:
        # each layer over the largest at the voxel: 1 wherever a set is alone (its tensor is then the excised structure
        # tensor of its own voxels, free of the activation's pull towards the population's axis), shared where sets meet
        top = A.max(0).values.clamp_min(1.0)                                                  # never above 1, tapering below the null
        A = A / top[None]
    w_exc = st.excision_weights(v, mask=tissue if tissue is not None else None, spacing=st.spacing_ratio(spacing))
    prods = st.gradient_products(v, cal.sigma, cal.truncate, spacing=spacing)
    S, dirs, gaps = [], [], []
    for k in range(A.shape[0]):
        Sk, dk, gk = population_tensor(v, A[k], cal.sigma, cal.rho, cal.truncate, w_exc, tensor_floor, spacing, prods)
        S.append(Sk); dirs.append(dk); gaps.append(gk)
    S, dirs = torch.stack(S), torch.stack(dirs)
    vec, vec_gap, weight, T = single_field(S, A, floor=merge_floor, return_tensor=True)
    vecs, weights = multi_field(dirs, A)
    return {"activation": A, "dirs": dirs, "vec": vec, "vec_gap": vec_gap, "weight": weight, "vecs": vecs, "weights": weights,
            "n_populations": int(A.shape[0])}


# ---------------------------------------------------------------------------------------------------------------
# prototype 3 (hybrid) and 4 (fully local): orientation populations joined where they are one set of fibres
# ---------------------------------------------------------------------------------------------------------------

def agreement(Ai, Aj, di, dj, gi, gj, mask) -> tuple[float | None, int]:
    """Whether two populations are one set of fibres where both are present: the weighted median angle between their
    own layer directions over the voxels where both activations are at least the null (``A`` in nulls) inside
    ``mask``, each voxel weighted by ``min(A_i, A_j) · gap_i · gap_j`` (confident, jointly active voxels count). Returns
    the angle (degrees, None without such voxels) and the number of voxels."""
    both = (Ai >= 1) & (Aj >= 1) & mask
    n = int(both.sum())
    if n == 0:
        return None, 0
    c = (di[:, both] * dj[:, both]).sum(0).abs().clamp(max=1)
    ang = torch.rad2deg(torch.arccos(c))
    w = (torch.minimum(Ai[both], Aj[both]) * gi[both].clamp_min(0) * gj[both].clamp_min(0)).double()
    if float(w.sum()) <= 0:
        return float(ang.median()), n
    o = torch.argsort(ang)
    cw = torch.cumsum(w[o], 0)
    return float(ang[o][int(torch.searchsorted(cw, cw[-1] / 2))]), n


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def merge_by_agreement(A: torch.Tensor, dirs: torch.Tensor, gap: torch.Tensor, mask: torch.Tensor, *, sep_deg: float = 25.0,
                       share: float = 0.05):
    """Groups of populations that are one set of fibres: two are joined where both are present over at least ``share``
    of the voxels where either is, and there their own directions agree within ``sep_deg`` (:func:`agreement`); a
    crossing (co-located, directions apart) and two bundles that never meet stay apart. ``A (N, ...)`` in nulls,
    ``dirs (N, 3, ...)``, ``gap (N, ...)``. Returns the groups (lists of indices) and the pairs examined."""
    N = A.shape[0]
    uf = _UnionFind(N); pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            either = int((((A[i] >= 1) | (A[j] >= 1)) & mask).sum())
            ang, n = agreement(A[i], A[j], dirs[i], dirs[j], gap[i], gap[j], mask)
            joined = ang is not None and either > 0 and n >= share * either and ang < sep_deg
            pairs.append({"pair": [i, j], "angle_deg": None if ang is None else round(ang, 1), "shared_share": round(n / max(either, 1), 3),
                          "joined": bool(joined)})
            if joined:
                uf.union(i, j)
    groups = {}
    for i in range(N):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values()), pairs


def hybrid(out: dict, tissue: torch.Tensor, *, sep_deg: float = 25.0, share: float = 0.05, vox_mm: float = 0.02,
           min_volume_mm3: float = 0.0) -> tuple[torch.Tensor, dict]:
    """The pipeline's populations (``out`` from :func:`shapetensor.pipeline.apply`, activation in nulls) joined into
    sets of fibres by :func:`merge_by_agreement`; each set's layer is the sum of its members'. Sets whose layer is at
    least the null over less than ``min_volume_mm3`` are dropped."""
    A, dirs, gap = out["activation"].float(), out["dirs"].float(), out["gap"].float()
    groups, pairs = merge_by_agreement(A, dirs, gap, tissue, sep_deg=sep_deg, share=share)
    layers = [A[g].sum(0) for g in groups]
    vol = [float(((L >= 1) & tissue).sum()) * vox_mm ** 3 for L in layers]
    keep = [k for k in np.argsort(vol)[::-1] if vol[k] >= min_volume_mm3]
    L = torch.stack([layers[k] for k in keep]) if keep else torch.zeros((0,) + tuple(A.shape[1:]), device=A.device)
    return L, {"groups": [groups[k] for k in keep], "pairs": pairs, "volumes_mm3": [round(vol[k], 4) for k in keep]}


def local_populations(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, window: int = 32, stride: int = 16,
                      region=None, nmax: int = 3, min_sep_deg: float = 25.0, sep_deg: float = 25.0, share: float = 0.05,
                      vox_mm: float = 0.02, min_volume_mm3: float = 0.0, device: str = "cuda", log=None) -> tuple[torch.Tensor, dict]:
    """Populations found in small windows and joined by continuity. Windows of ``window``³ voxels every ``stride``
    over ``region`` (a tuple of slices; default the whole block) each run the pipeline's stage 2 on their own (the
    block's readers and features, the window's own orientation density, at most ``nmax`` populations
    ``min_sep_deg`` apart), with the filters' reach as context. Populations of the same or neighbouring windows are
    joined where they are one set of fibres (:func:`agreement` within ``sep_deg`` over their shared voxels), so a
    bundle whose direction changes across the block is one set if it is coherent locally, and a crossing (co-located,
    directions apart) stays two. Each set's layer is its members' activations (in nulls) blended over the windows with
    ramps that peak at each window's centre. Returns the layers ``(N, Z, Y, X)`` and a report."""
    from ..pipeline import apply

    say = log or (lambda *_: None)
    Z, Y, X = v.shape
    region = region or (slice(0, Z), slice(0, Y), slice(0, X))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    starts = []
    for ax, (sl, n) in enumerate(zip(region, (Z, Y, X))):
        lo, hi = sl.start, sl.stop
        w = min(window, hi - lo)
        s = list(range(lo, max(lo, hi - w) + 1, stride))
        if s[-1] + w < hi:
            s.append(hi - w)
        starts.append((s, w))
    wins, pops = [], []                                                                    # pops: (window, k) -> layer data on the window core
    tau = None
    for iz, z0 in enumerate(starts[0][0]):
        for iy, y0 in enumerate(starts[1][0]):
            for ix, x0 in enumerate(starts[2][0]):
                wz, wy, wx = starts[0][1], starts[1][1], starts[2][1]
                core = (slice(z0, z0 + wz), slice(y0, y0 + wy), slice(x0, x0 + wx))
                ctx = tuple(slice(max(0, c.start - reach), min(n, c.stop + reach)) for c, n in zip(core, (Z, Y, X)))
                inner = tuple(slice(c.start - k.start, c.stop - k.start) for c, k in zip(core, ctx))
                wid = len(wins); wins.append({"idx": (iz, iy, ix), "core": core})
                t = tissue[ctx]
                if float(t[inner].float().mean()) < 0.01:
                    continue
                try:
                    o = apply(cal, v[ctx].contiguous(), t.contiguous(), nmax=nmax, min_sep_deg=min_sep_deg, window=0,
                              features=F[(slice(None),) + ctx].contiguous(), device=device)
                except Exception as e:                                                     # too little to find a population on
                    say(f"window {wid}: {type(e).__name__}")
                    continue
                tau = o["tau"]
                for k in range(o["activation"].shape[0]):
                    pops.append({"win": wid, "A": (o["activation"][k][inner] / o["tau"]).float(), "dir": o["dirs"][k][(slice(None),) + inner].float(),
                                 "gap": o["gap"][k][inner].float()})
                del o
    say(f"{len(wins)} windows, {len(pops)} window populations")
    # join: populations of the same window or of overlapping windows, compared on their shared voxels
    uf = _UnionFind(len(pops)); joined = 0
    byw = {}
    for p_i, p in enumerate(pops):
        byw.setdefault(p["win"], []).append(p_i)
    for a in range(len(wins)):
        for b in range(a, len(wins)):
            if a not in byw or b not in byw:
                continue
            ca, cb = wins[a]["core"], wins[b]["core"]
            ov = tuple(slice(max(x.start, y.start), min(x.stop, y.stop)) for x, y in zip(ca, cb))
            if any(s.start >= s.stop for s in ov):
                continue
            ra = tuple(slice(s.start - c.start, s.stop - c.start) for s, c in zip(ov, ca))
            rb = tuple(slice(s.start - c.start, s.stop - c.start) for s, c in zip(ov, cb))
            m = tissue[ov]
            for i in byw[a]:
                for j in byw[b]:
                    if i >= j and a == b:
                        continue
                    Pi, Pj = pops[i], pops[j]
                    Ai, Aj = Pi["A"][ra], Pj["A"][rb]
                    either = int((((Ai >= 1) | (Aj >= 1)) & m).sum())
                    ang, n = agreement(Ai, Aj, Pi["dir"][(slice(None),) + ra], Pj["dir"][(slice(None),) + rb], Pi["gap"][ra], Pj["gap"][rb], m)
                    if ang is not None and either > 0 and n >= share * either and ang < sep_deg:
                        uf.union(i, j); joined += 1
    sets = {}
    for i in range(len(pops)):
        sets.setdefault(uf.find(i), []).append(i)
    sets = list(sets.values())
    # each set's layer, ramp-blended over the windows
    acc = torch.zeros((len(sets), Z, Y, X), device=v.device); wsum = torch.zeros((Z, Y, X), device=v.device)
    for wd in wins:
        c = wd["core"]
        ramp = 1.0
        for ax, s in enumerate(c):
            n = s.stop - s.start
            r = 1.0 - (torch.arange(n, device=v.device, dtype=torch.float32) + 0.5 - n / 2).abs() / (n / 2)
            shape = [1, 1, 1]; shape[ax] = n
            ramp = ramp * r.clamp_min(1e-3).reshape(shape)
        wsum[c] += ramp
        wd["ramp"] = ramp
    for si, members in enumerate(sets):
        for i in members:
            p = pops[i]; c = wins[p["win"]]["core"]
            acc[si][c] += p["A"] * wins[p["win"]]["ramp"]
    acc = acc / wsum.clamp_min(1e-6)[None]
    vol = [float(((acc[k] >= 1) & tissue).sum()) * vox_mm ** 3 for k in range(len(sets))]
    keep = [k for k in np.argsort(vol)[::-1] if vol[k] >= min_volume_mm3]
    L = acc[keep] if keep else torch.zeros((0, Z, Y, X), device=v.device)
    return L, {"windows": len(wins), "window_populations": len(pops), "joins": joined, "sets_all": len(sets),
               "sets_kept": len(keep), "volumes_mm3": [round(vol[k], 4) for k in keep][:12]}


# ---------------------------------------------------------------------------------------------------------------
# prototype 5: local populations joined by good continuation
# ---------------------------------------------------------------------------------------------------------------

def _window_grid(region, window, stride):
    out = []
    for sl in region:
        lo, hi = sl.start, sl.stop
        w = min(window, hi - lo)
        s = list(range(lo, max(lo, hi - w) + 1, stride))
        if s[-1] + w < hi:
            s.append(hi - w)
        out.append((s, w))
    return out


def local_continuation(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, gap_null: float, window: int = 32, stride: int = 16,
                       region=None, nmax: int = 3, min_sep_deg: float = 25.0, min_share: float = 0.05, r_min_mm: float = 0.3,
                       noise_deg: float = 10.0, vox_mm: float = 0.02, device: str = "cuda", log=None,
                       layers: bool = True) -> tuple[torch.Tensor | None, dict]:
    """Populations found in small windows, joined from window to window by good continuation.

    Every ``window``³ window (every ``stride`` voxels over ``region``) runs the pipeline's stage 2 on its own (the
    block's readers and features; its own orientation density; at most ``nmax`` populations ``min_sep_deg`` apart). A
    window population is kept where it is present — activation at least the null *and* its tensor more coherent than
    on fibre-free texture (gap at least ``gap_null``) — over at least ``min_share`` of the window's tissue (set it to
    the largest share any window population reaches on fibre-free texture: the report's ``shares`` of such a run), and is
    summarised by one direction (the principal axis of its present voxels' directions, weighted by activation × gap);
    populations of one window less than ``min_sep_deg`` apart are one. Populations of neighbouring windows are joined
    when each is the other's closest in direction and the two differ by no more than a turn of radius ``r_min_mm`` over
    the distance between the windows' centres, plus ``noise_deg``: in a bend every window holds the one local
    direction and the chain follows it; at a crossing whose window holds both directions each arm continues into its
    own. Each set's layer is its members' activations (in nulls) blended over the windows with ramps peaking at their
    centres. Returns the layers ``(N, Z, Y, X)`` and a report."""
    from ..pipeline import apply

    say = log or (lambda *_: None)
    Z, Y, X = v.shape
    region = region or (slice(0, Z), slice(0, Y), slice(0, X))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    grid = _window_grid(region, window, stride)
    wins, pops, shares = {}, [], []
    for iz, z0 in enumerate(grid[0][0]):
        for iy, y0 in enumerate(grid[1][0]):
            for ix, x0 in enumerate(grid[2][0]):
                core = (slice(z0, z0 + grid[0][1]), slice(y0, y0 + grid[1][1]), slice(x0, x0 + grid[2][1]))
                ctx = tuple(slice(max(0, c.start - reach), min(n, c.stop + reach)) for c, n in zip(core, (Z, Y, X)))
                inner = tuple(slice(c.start - k.start, c.stop - k.start) for c, k in zip(core, ctx))
                key = (iz, iy, ix); wins[key] = {"core": core, "pops": []}
                tc = tissue[core]
                if float(tc.float().mean()) < 0.01:
                    continue
                try:
                    o = apply(cal, v[ctx].contiguous(), tissue[ctx].contiguous(), nmax=nmax, min_sep_deg=min_sep_deg, window=0,
                              features=F[(slice(None),) + ctx].contiguous(), device=device)
                except Exception as e:                                                 # too little to find a population on
                    say(f"window {key}: {type(e).__name__}")
                    continue
                reps = []
                for k in range(o["activation"].shape[0]):
                    A = (o["activation"][k][inner] / o["tau"]).float()
                    d = o["dirs"][k][(slice(None),) + inner].float(); g = o["gap"][k][inner].float()
                    pres = (A >= 1) & (g >= gap_null) & tc
                    shares.append(float(pres.sum()) / max(float(tc.sum()), 1.0))
                    if float(pres.sum()) < min_share * float(tc.sum()):
                        continue
                    w = (A * g)[pres]
                    dv = d[:, pres]
                    M = (dv * w[None]) @ dv.T
                    _, V = torch.linalg.eigh(M.double())
                    r = V[:, -1].float()
                    # one population per distinct direction within the window
                    for e in reps:
                        if float((e["dir"] @ r).abs()) >= math.cos(math.radians(min_sep_deg)):
                            e["A"] = e["A"] + A; break
                    else:
                        reps.append({"dir": r, "A": A})
                for e in reps:
                    e["win"] = key; pops.append(e); wins[key]["pops"].append(len(pops) - 1)
                del o
    say(f"{len(wins)} windows, {len(pops)} window populations")
    if not layers:
        return None, {"windows": len(wins), "window_populations": len(pops), "shares": sorted(shares, reverse=True)}
    uf = _UnionFind(len(pops)); links = 0
    keys = list(wins)
    for a in keys:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    b = (a[0] + dz, a[1] + dy, a[2] + dx)
                    if b <= a or b not in wins or not wins[a]["pops"] or not wins[b]["pops"]:
                        continue
                    ca = [(s.start + s.stop) / 2 for s in wins[a]["core"]]; cb = [(s.start + s.stop) / 2 for s in wins[b]["core"]]
                    dist = math.dist(ca, cb) * vox_mm
                    cos_max = math.cos(min(math.pi / 2, dist / r_min_mm + math.radians(noise_deg)))
                    PA, PB = wins[a]["pops"], wins[b]["pops"]
                    C = torch.tensor([[float((pops[i]["dir"] @ pops[j]["dir"]).abs()) for j in PB] for i in PA])
                    for ii, i in enumerate(PA):
                        jj = int(C[ii].argmax())
                        if int(C[:, jj].argmax()) == ii and float(C[ii, jj]) >= cos_max:           # mutual closest, within the turn
                            uf.union(i, PB[jj]); links += 1
    sets = {}
    for i in range(len(pops)):
        sets.setdefault(uf.find(i), []).append(i)
    sets = list(sets.values())
    acc = torch.zeros((len(sets), Z, Y, X), device=v.device); wsum = torch.zeros((Z, Y, X), device=v.device)
    ramps = {}
    for key, wd in wins.items():
        c = wd["core"]; ramp = 1.0
        for ax, s in enumerate(c):
            n = s.stop - s.start
            r = (1.0 - (torch.arange(n, device=v.device, dtype=torch.float32) + 0.5 - n / 2).abs() / (n / 2)).clamp_min(1e-3)
            shp = [1, 1, 1]; shp[ax] = n; ramp = ramp * r.reshape(shp)
        wsum[c] += ramp; ramps[key] = ramp
    for si, members in enumerate(sets):
        for i in members:
            p = pops[i]; c = wins[p["win"]]["core"]
            acc[si][c] += p["A"] * ramps[p["win"]]
    acc = acc / wsum.clamp_min(1e-6)[None]
    order = sorted(range(len(sets)), key=lambda k: -float(acc[k].sum()))
    return acc[order] if sets else torch.zeros((0, Z, Y, X), device=v.device), \
        {"windows": len(wins), "window_populations": len(pops), "links": links, "sets": len(sets),
         "set_sizes_windows": sorted([len(s) for s in sets], reverse=True)[:12]}


# ---------------------------------------------------------------------------------------------------------------
# prototype 6: windowed density populations joined by good continuation
# ---------------------------------------------------------------------------------------------------------------

def window_densities(cal, Fp: torch.Tensor, region, window: int, stride: int, chunk: int = 16):
    """The orientation density of every ``window``³ window every ``stride`` voxels over ``region`` (the readout's
    per-voxel distributions summed, divided by the readers' direction gains): ``(n_dirs, nz, ny, nx)`` and the windows'
    start corners per axis."""
    P = cal.readout.predict_dist(Fp, prepped=True)
    sub = P[(slice(None),) + tuple(region)]
    w = tuple(min(window, n) for n in sub.shape[1:])
    out = []
    for a in range(0, sub.shape[0], chunk):
        out.append(Fn.avg_pool3d(sub[a:a + chunk][None], w, stride=stride)[0] * float(np.prod(w)))
    D = torch.cat(out); del P, sub
    if cal.density_gain is not None:
        D = D / cal.density_gain[:, None, None, None]
    starts = [[sl.start + k * stride for k in range(n)] for sl, n in zip(region, D.shape[1:])]
    return D, starts, w


def peak_null(cal, volumes, region, window: int = 32, stride: int = 16, nmax: int = 3, min_sep_deg: float = 25.0) -> float:
    """The tallest second peak (relative to the first) of any window's orientation density over ``volumes`` of a single
    population each (the calibration bundles): what one bundle's density shows besides its own direction. A window's
    second population must stand taller."""
    from ..separate import modes_from_density
    top = 0.0
    for vol in volumes:
        Fp = cal.dictionary.prep(cal.features(vol))
        D, _, _ = window_densities(cal, Fp, region, window, stride)
        prom = prominence(D)
        for idx in torch.nonzero(prom > 0).tolist():
            _, h, n = modes_from_density(D[(slice(None),) + tuple(idx)], cal.readout.grid, rel=0.0, min_sep_deg=min_sep_deg, nmax=nmax,
                                         mass_floor=0.1)
            if n >= 2:
                top = max(top, float(h[1]))
        del Fp, D
    return top


def prominence(D: torch.Tensor) -> torch.Tensor:
    """Each window's top density over its mean density, ``(nz, ny, nx)``: how far one direction stands out."""
    return D.max(0).values / D.mean(0).clamp_min(1e-12)


def windowed_continuation(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, prominence_null: float, second_null: float = 0.3,
                          window: int = 32,
                          stride: int = 16, region=None, nmax: int = 3, min_sep_deg: float = 25.0, r_min_mm: float = 0.3,
                          noise_deg: float = 10.0, vox_mm: float = 0.02, log=None) -> tuple[torch.Tensor, dict]:
    """Populations found per window from the window's own orientation density and joined by good continuation.

    Each ``window``³ window (every ``stride`` voxels over ``region``) holds fibres if its density's top stands out more
    than in any window of fibre-free texture (``prominence_null``, :func:`prominence`); its populations are the
    density's peaks (``min_sep_deg`` apart, each owning 0.1 of the density, at most ``nmax``), a second or third only
    where it stands taller, relative to the first, than any second peak of a single bundle (``second_null``,
    :func:`peak_null`), with activations from the pipeline's unmixing onto those directions. Neighbouring
    windows' populations are joined when each is the other's closest in direction and they differ by no more than a
    turn of radius ``r_min_mm`` over the windows' distance plus ``noise_deg``. Two populations of one window are joined
    end to end — the window holds a bend — when each continues out of the window on one side only and on opposite
    sides, and their directions differ by no more than the turn across the window; a population that continues on both
    sides passes through (a crossing) and is never joined. Each set's layer is its members' activations (in nulls)
    blended with ramps peaking at the windows' centres. Returns the layers ``(N, Z, Y, X)`` and a report."""
    from ..pipeline import _normalised
    from ..separate import modes_from_density, unmix

    say = log or (lambda *_: None)
    Z, Y, X = v.shape
    region = tuple(region or (slice(0, Z), slice(0, Y), slice(0, X)))
    Fp = cal.dictionary.prep(F)
    D, starts, w = window_densities(cal, Fp, region, window, stride)
    prom = prominence(D)
    grid = cal.readout.grid
    pops, wins = [], {}
    for iz in range(D.shape[1]):
        for iy in range(D.shape[2]):
            for ix in range(D.shape[3]):
                key = (iz, iy, ix)
                core = tuple(slice(s[k], s[k] + w[a]) for a, (s, k) in enumerate(zip(starts, key)))
                wins[key] = {"core": core, "pops": [], "centre": [(c.start + c.stop) / 2 for c in core]}
                if float(prom[key]) < prominence_null or float(tissue[core].float().mean()) < 0.01:
                    continue
                m, h, n = modes_from_density(D[(slice(None),) + key], grid, rel=0.0, min_sep_deg=min_sep_deg, nmax=nmax, mass_floor=0.1)
                if n == 0:
                    continue
                n = 1 + sum(1 for k in range(1, n) if float(h[k]) > second_null)
                modes = m[:n]
                A, _ = unmix(Fp[(slice(None),) + core], cal.dictionary, modes, prepped=True)
                A = (_normalised(cal, A, modes) / cal.tau).float()
                for k in range(n):
                    pops.append({"win": key, "dir": modes[k].float(), "A": A[k]}); wins[key]["pops"].append(len(pops) - 1)
    say(f"{len(wins)} windows, {len(pops)} window populations")
    uf = _UnionFind(len(pops)); links = 0
    linked = {i: [] for i in range(len(pops))}                                            # the neighbour windows each is linked to
    for a in wins:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    b = (a[0] + dz, a[1] + dy, a[2] + dx)
                    if b <= a or b not in wins or not wins[a]["pops"] or not wins[b]["pops"]:
                        continue
                    dist = math.dist(wins[a]["centre"], wins[b]["centre"]) * vox_mm
                    cos_max = math.cos(min(math.pi / 2, dist / r_min_mm + math.radians(noise_deg)))
                    PA, PB = wins[a]["pops"], wins[b]["pops"]
                    C = torch.tensor([[float((pops[i]["dir"] @ pops[j]["dir"]).abs()) for j in PB] for i in PA])
                    for ii, i in enumerate(PA):
                        jj = int(C[ii].argmax())
                        if int(C[:, jj].argmax()) == ii and float(C[ii, jj]) >= cos_max:
                            uf.union(i, PB[jj]); links += 1
                            linked[i].append(b); linked[PB[jj]].append(a)
    # end-to-end joins inside a window: a bend's two local directions, never a crossing's
    ends = 0
    span = (window * vox_mm) / r_min_mm + math.radians(noise_deg)
    for key, wd in wins.items():
        P_ = wd["pops"]
        if len(P_) < 2:
            continue
        side = {}
        for i in P_:
            d = pops[i]["dir"].cpu().numpy()
            offs = [np.array(wins[b]["centre"]) - np.array(wd["centre"]) for b in linked[i]]
            proj = [float(o @ d) for o in offs if abs(float(o @ d)) > 0.5 * np.linalg.norm(o) * 0.5]
            sgn = {1 if p > 0 else -1 for p in proj}
            side[i] = (sgn, np.mean([o / np.linalg.norm(o) for o in offs], axis=0) if offs else None)
        for x_, i in enumerate(P_):
            for j in P_[x_ + 1:]:
                (si, ui), (sj, uj) = side[i], side[j]
                if len(si) != 1 or len(sj) != 1 or ui is None or uj is None:
                    continue                                                              # one of them passes through, or ends here
                ang = math.acos(min(1.0, abs(float(pops[i]["dir"] @ pops[j]["dir"]))))
                if float(ui @ uj) < 0 and ang <= span:
                    uf.union(i, j); ends += 1
    sets = {}
    for i in range(len(pops)):
        sets.setdefault(uf.find(i), []).append(i)
    sets = list(sets.values())
    acc = torch.zeros((len(sets), Z, Y, X), device=v.device); wsum = torch.zeros((Z, Y, X), device=v.device)
    ramps = {}
    for key, wd in wins.items():
        c = wd["core"]; ramp = 1.0
        for ax, s in enumerate(c):
            n = s.stop - s.start
            r = (1.0 - (torch.arange(n, device=v.device, dtype=torch.float32) + 0.5 - n / 2).abs() / (n / 2)).clamp_min(1e-3)
            shp = [1, 1, 1]; shp[ax] = n; ramp = ramp * r.reshape(shp)
        wsum[c] += ramp; ramps[key] = ramp
    for si, members in enumerate(sets):
        for i in members:
            p = pops[i]; c = wins[p["win"]]["core"]
            acc[si][c] += p["A"] * ramps[p["win"]]
    acc = acc / wsum.clamp_min(1e-6)[None]
    order = sorted(range(len(sets)), key=lambda k: -float(acc[k].sum()))
    report = {"windows": len(wins), "windows_with_fibres": int(sum(1 for wd in wins.values() if wd["pops"])), "window_populations": len(pops),
              "windows_with_2plus": int(sum(1 for wd in wins.values() if len(wd["pops"]) >= 2)), "links": links, "end_joins": ends,
              "sets": len(sets), "set_sizes": sorted([len(s) for s in sets], reverse=True)[:12]}
    return (acc[order] if sets else torch.zeros((0, Z, Y, X), device=v.device)), report


# ---------------------------------------------------------------------------------------------------------------
# prototype 7: windowed stage 2 (the pipeline's own confirmation per window), joined by good continuation
# ---------------------------------------------------------------------------------------------------------------

def readout_response(cal, volumes, dirs, region, window: int = 32, stride: int = 8, bin_deg: float = 2.0):
    """The readout's response to one population: the window-averaged orientation density (each window's density
    divided by its sum, after the direction gains) against the angle to the population, over ``volumes`` of one
    population each at directions ``dirs (k, 3)`` (the calibration bundles). Returns ``(angles_deg, profile)``."""
    u = cal.readout.grid
    bins = np.arange(0.0, 90.0 + bin_deg, bin_deg); prof = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1)
    for vol, d in zip(volumes, dirs):
        D, _, _ = window_densities(cal, cal.dictionary.prep(cal.features(vol)), region, window, stride)
        D = D.reshape(D.shape[0], -1); D = D / D.sum(0, keepdim=True).clamp_min(1e-12)
        d = torch.as_tensor(np.asarray(d), device=u.device, dtype=u.dtype)
        th = torch.rad2deg(torch.arccos((u @ d).abs().clamp(max=1))).cpu().numpy()
        idx = np.clip(np.digitize(th, bins) - 1, 0, len(prof) - 1)
        np.add.at(prof, idx, D.mean(1).cpu().numpy()); np.add.at(cnt, idx, 1)
    keep = cnt > 0
    return 0.5 * (bins[1:] + bins[:-1])[keep], prof[keep] / cnt[keep]


class DeconvolvedReadout:
    """The readout with its block density deconvolved by its own response to one population (``readout_response``):
    a crossing's density is close to the sum of its populations' responses, whose overlap pulls the peaks towards each
    other; the non-negative deconvolution (the response's directional part plus its flat floor) puts them back. Every
    other use of the readout is unchanged."""

    def __init__(self, base, response, gain=None):
        self._base, self._gain = base, gain
        ang, prof = response; u = base.grid
        G = torch.rad2deg(torch.arccos((u @ u.T).abs().clamp(max=1))).cpu().numpy()
        floor = float(prof.min())
        K = np.interp(G, ang, prof - floor)
        self._K = np.concatenate([K, np.full((len(u), 1), floor)], 1).astype(np.float64)       # (n_dirs, n_dirs + 1), on the CPU

    def __getattr__(self, k):
        return getattr(self._base, k)

    def deconvolve(self, dens: torch.Tensor) -> torch.Tensor:
        """A gain-normalised density ``(n_dirs,)`` -> its deconvolved weights ``(n_dirs,)``, the same total."""
        from scipy.optimize import nnls
        tot = dens.sum().clamp_min(1e-12)
        w, _ = nnls(self._K, (dens / tot).double().cpu().numpy(), maxiter=2000)
        w = torch.as_tensor(w[:dens.shape[0]], device=dens.device, dtype=dens.dtype)
        return w / w.sum().clamp_min(1e-12) * tot

    def density(self, F, region=None, *a, **kw):
        D = self._base.density(F, region, *a, **kw)
        g = self._gain if self._gain is not None else torch.ones_like(D)
        return self.deconvolve(D / g) * g


def _window_stage2(cal, v, tissue, Fp_or_F, core, reach, nmax, min_sep_deg, device):
    """The pipeline's stage 2 on one window with the filters' reach as context: ``(modes, A in nulls on the core,
    mask shares on the core)`` sorted by share (largest first), or None."""
    from ..pipeline import apply
    Z, Y, X = v.shape
    ctx = tuple(slice(max(0, c.start - reach), min(n, c.stop + reach)) for c, n in zip(core, (Z, Y, X)))
    inner = tuple(slice(c.start - k.start, c.stop - k.start) for c, k in zip(core, ctx))
    try:
        o = apply(cal, v[ctx].contiguous(), tissue[ctx].contiguous(), nmax=nmax, min_sep_deg=min_sep_deg, window=0,
                  features=Fp_or_F[(slice(None),) + ctx].contiguous(), device=device)
    except Exception:
        return None
    tc = tissue[core]
    nt = max(float(tc.sum()), 1.0)
    shares = torch.tensor([float((o["masks"][k][inner] & tc).sum()) / nt for k in range(o["modes"].shape[0])])
    order = torch.argsort(shares, descending=True)
    A = (o["activation"][:, inner[0], inner[1], inner[2]] / o["tau"]).float()
    return o["modes"][order].float(), A[order], shares[order]


def second_share_null(cal, volumes, region, *, window: int = 32, stride: int = 16, nmax: int = 3, min_sep_deg: float = 25.0,
                      every: int = 2, device: str = "cuda") -> float:
    """The largest mask share any second population reaches in a window of a single-population volume (the calibration
    bundles), every ``every``-th window: a window's second population must cover more."""
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    top = 0.0
    for vol in volumes:
        F = cal.features(vol); t = torch.ones_like(vol, dtype=torch.bool)
        grid = _window_grid(region, window, stride)
        for iz, z0 in enumerate(grid[0][0][::every]):
            for y0 in grid[1][0][::every]:
                for x0 in grid[2][0][::every]:
                    core = (slice(z0, z0 + grid[0][1]), slice(y0, y0 + grid[1][1]), slice(x0, x0 + grid[2][1]))
                    r = _window_stage2(cal, vol, t, F, core, reach, nmax, min_sep_deg, device)
                    if r is not None and len(r[2]) >= 2:
                        top = max(top, float(r[2][1]))
        del F
    return top


def windowed_stage2(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, prominence_null: float, share_null: float,
                    window: int = 32, stride: int = 8, region=None, nmax: int = 3, min_sep_deg: float = 25.0, r_min_mm: float = 0.3,
                    noise_deg: float = 10.0, vox_mm: float = 0.02, device: str = "cuda", log=None) -> tuple[torch.Tensor, dict]:
    """Populations from the pipeline's own stage 2 run per window, joined by good continuation.

    A ``window``³ window (every ``stride`` voxels over ``region``) holds fibres if its orientation density stands out
    more than in any fibre-free window (``prominence_null``). Its populations are those the pipeline's stage 2 finds on
    it (candidates from the window's density, each confirmed by its own layer, at most ``nmax``, ``min_sep_deg`` apart):
    the largest always, another only where it covers more of the window than any second population of a single-bundle
    window does (``share_null``, :func:`second_share_null`). Populations of neighbouring windows are joined as mutual
    closest directions within the turn a radius of ``r_min_mm`` makes over the distance the step travels *along* the
    fibres (a sideways step to the rest of the same bundle allows only ``noise_deg``); two of one window end to end only
    (a bend), never one that continues on both sides (a crossing). Layers as :func:`windowed_continuation`."""
    say = log or (lambda *_: None)
    Z, Y, X = v.shape
    region = tuple(region or (slice(0, Z), slice(0, Y), slice(0, X)))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    D, starts, w = window_densities(cal, cal.dictionary.prep(F), region, window, stride)
    prom = prominence(D); del D
    pops, wins = [], {}
    for iz in range(prom.shape[0]):
        for iy in range(prom.shape[1]):
            for ix in range(prom.shape[2]):
                key = (iz, iy, ix)
                core = tuple(slice(s[k], s[k] + w[a]) for a, (s, k) in enumerate(zip(starts, key)))
                wins[key] = {"core": core, "pops": [], "centre": [(c.start + c.stop) / 2 for c in core]}
                if float(prom[key]) < prominence_null or float(tissue[core].float().mean()) < 0.01:
                    continue
                r = _window_stage2(cal, v, tissue, F, core, reach, nmax, min_sep_deg, device)
                if r is None:
                    continue
                modes, A, shares = r
                n = 1 + sum(1 for k in range(1, len(shares)) if float(shares[k]) > share_null)
                for k in range(n):
                    pops.append({"win": key, "dir": modes[k], "A": A[k]}); wins[key]["pops"].append(len(pops) - 1)
    say(f"{len(wins)} windows, {len(pops)} window populations")
    Fp = cal.dictionary.prep(F)

    def infill(pops, wins, concats):
        """For each concatenated pair of chain ends, every window whose centre lies on the segment between them (within
        half a window of it) gets the joined direction as a population: the window's features unmixed onto its own
        populations plus that direction, the joined direction's activation kept (in nulls)."""
        from ..pipeline import _normalised
        from ..separate import unmix
        added = 0
        for i, j in concats:
            ca = np.array(wins[pops[i]["win"]]["centre"]); cb = np.array(wins[pops[j]["win"]]["centre"])
            seg = cb - ca; L = np.linalg.norm(seg)
            if L == 0:
                continue
            u = seg / L
            d = pops[i]["dir"] + torch.sign(pops[i]["dir"] @ pops[j]["dir"]) * pops[j]["dir"]; d = d / d.norm().clamp_min(1e-9)
            for key, wd in wins.items():
                if key in (pops[i]["win"], pops[j]["win"]):
                    continue
                o = np.array(wd["centre"]) - ca; t = float(o @ u)
                if t <= 0 or t >= L or np.linalg.norm(o - t * u) > window / 2:
                    continue
                own = [pops[k]["dir"] for k in wd["pops"] if float((pops[k]["dir"] @ d).abs()) < math.cos(math.radians(min_sep_deg))]
                modes = torch.stack(own + [d])
                Aw, _ = unmix(Fp[(slice(None),) + wd["core"]], cal.dictionary, modes, prepped=True)
                Aw = (_normalised(cal, Aw, modes) / cal.tau).float()
                pops.append({"win": key, "dir": d, "A": Aw[-1], "joins": i}); wd["pops"].append(len(pops) - 1); added += 1
        return added

    L, rep = _join_and_blend(pops, wins, v, tissue, window, r_min_mm, noise_deg, vox_mm, infill=infill)
    rep.pop("labels", None)
    return L, rep


def _join_and_blend(pops, wins, v, tissue, window, r_min_mm, noise_deg, vox_mm, bridge_mm: float = 1.0, cone_deg: float = 25.0,
                    concat_deg: float = 20.0, infill=None):
    """Mutual-closest joins between neighbouring windows, end-to-end joins within a window, ramp-blended set layers."""
    Z, Y, X = v.shape
    uf = _UnionFind(len(pops)); links = 0
    linked = {i: [] for i in range(len(pops))}
    for a in wins:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    b = (a[0] + dz, a[1] + dy, a[2] + dx)
                    if b <= a or b not in wins or not wins[a]["pops"] or not wins[b]["pops"]:
                        continue
                    off = (np.array(wins[b]["centre"]) - np.array(wins[a]["centre"])) * vox_mm
                    PA, PB = wins[a]["pops"], wins[b]["pops"]
                    C = torch.full((len(PA), len(PB)), -1.0)
                    for ii, i in enumerate(PA):
                        for kk, j in enumerate(PB):
                            di, dj = pops[i]["dir"], pops[j]["dir"]
                            c = float((di @ dj).abs())
                            dm = (di + torch.sign(di @ dj) * dj); dm = (dm / dm.norm().clamp_min(1e-9)).cpu().numpy()
                            along = abs(float(off @ dm))                               # how far the step travels along the fibre
                            if c >= math.cos(min(math.pi / 2, along / r_min_mm + math.radians(noise_deg))):
                                C[ii, kk] = c
                    for ii, i in enumerate(PA):
                        jj = int(C[ii].argmax())
                        if float(C[ii, jj]) >= 0 and int(C[:, jj].argmax()) == ii:
                            uf.union(i, PB[jj]); links += 1
                            linked[i].append(b); linked[PB[jj]].append(a)
    # concatenation: a chain end joins the chain end that faces it along both their directions (within ``cone_deg``) across
    # a gap of at most ``bridge_mm``, if their directions agree within ``concat_deg`` (straight continuation; a crossing's
    # other band leaves the windows over the overlap without this direction, so a band's chain stops on each side)
    def sides(i):
        key = pops[i]["win"]; d = pops[i]["dir"].cpu().numpy()
        out = set()
        for b in linked[i]:
            o = np.array(wins[b]["centre"]) - np.array(wins[key]["centre"])
            pr = float(o @ d)
            if abs(pr) > 0.25 * np.linalg.norm(o):
                out.add(1 if pr > 0 else -1)
        return out
    bridges = 0
    cos_cone = math.cos(math.radians(cone_deg))
    endset = {i: {1, -1} - sides(i) for i in range(len(pops))}
    cand = {}
    steps = sorted({round(abs(wins[k]["centre"][ax] - wins[k2]["centre"][ax]), 3) for k in list(wins)[:2] for k2 in list(wins)[:2] for ax in range(3)} - {0})
    stride_vox = min([abs(np.array(wins[k]["centre"]) - np.array(wins[(k[0], k[1], k[2] + 1)]["centre"])).max()
                      for k in wins if (k[0], k[1], k[2] + 1) in wins][:1] or [8.0])
    bridge = max(2, int(math.ceil(bridge_mm / (stride_vox * vox_mm))))
    for i, missing in endset.items():
        if not missing:
            continue
        a = pops[i]["win"]; d = pops[i]["dir"].cpu().numpy(); ca = np.array(wins[a]["centre"])
        for dz in range(-bridge, bridge + 1):
            for dy in range(-bridge, bridge + 1):
                for dx in range(-bridge, bridge + 1):
                    b = (a[0] + dz, a[1] + dy, a[2] + dx)
                    if max(abs(dz), abs(dy), abs(dx)) < 2 or b not in wins:
                        continue
                    o = np.array(wins[b]["centre"]) - ca
                    no = np.linalg.norm(o); pr = float(o @ d)
                    if (1 if pr > 0 else -1) not in missing or abs(pr) < cos_cone * no:   # not along the fibre, or on a linked side
                        continue
                    for j in wins[b]["pops"]:
                        dj = pops[j]["dir"].cpu().numpy()
                        prj = float(-o @ dj)
                        if (1 if prj > 0 else -1) not in endset[j] or abs(prj) < cos_cone * no:   # j: an end facing back along its own axis
                            continue
                        c = abs(float(d @ dj))
                        if c >= math.cos(math.radians(concat_deg)) and uf.find(i) != uf.find(j):
                            cand.setdefault(i, []).append((c - 1e-3 * no, j))           # straightest first, then nearest
    best = {i: max(cs)[1] for i, cs in cand.items()}
    concats = []
    for i, j in best.items():
        if best.get(j) == i and i < j:
            uf.union(i, j); bridges += 1; concats.append((i, j))
    # in-fill: the windows along each concatenated gap get the joined direction, unmixed against what they already hold
    filled = 0
    if infill is not None and concats:
        n0 = len(pops)
        filled = infill(pops, wins, concats)
        uf.p.extend(range(n0, len(pops)))
        for k in range(n0, len(pops)):
            uf.union(k, pops[k]["joins"])
    ends = 0
    span = (window * vox_mm) / r_min_mm + math.radians(noise_deg)
    for key, wd in wins.items():
        P_ = wd["pops"]
        if len(P_) < 2:
            continue
        side = {}
        for i in P_:
            d = pops[i]["dir"].cpu().numpy()
            offs = [np.array(wins[b]["centre"]) - np.array(wd["centre"]) for b in linked.get(i, [])]
            proj = [float(o @ d) for o in offs if abs(float(o @ d)) > 0.25 * np.linalg.norm(o)]
            side[i] = ({1 if p > 0 else -1 for p in proj}, np.mean([o / np.linalg.norm(o) for o in offs], axis=0) if offs else None)
        for x_, i in enumerate(P_):
            for j in P_[x_ + 1:]:
                (si, ui), (sj, uj) = side[i], side[j]
                if len(si) != 1 or len(sj) != 1 or ui is None or uj is None:
                    continue
                if float(ui @ uj) < 0 and math.acos(min(1.0, abs(float(pops[i]["dir"] @ pops[j]["dir"])))) <= span:
                    uf.union(i, j); ends += 1
    sets = {}
    for i in range(len(pops)):
        sets.setdefault(uf.find(i), []).append(i)
    sets = list(sets.values())
    labels = [0] * len(pops)
    for si, members in enumerate(sets):
        for i in members:
            labels[i] = si
    acc = torch.zeros((len(sets), Z, Y, X), device=v.device); wsum = torch.zeros((Z, Y, X), device=v.device)
    ramps = {}
    for key, wd in wins.items():
        c = wd["core"]; ramp = 1.0
        for ax, s in enumerate(c):
            n = s.stop - s.start
            r = (1.0 - (torch.arange(n, device=v.device, dtype=torch.float32) + 0.5 - n / 2).abs() / (n / 2)).clamp_min(1e-3)
            shp = [1, 1, 1]; shp[ax] = n; ramp = ramp * r.reshape(shp)
        wsum[c] += ramp; ramps[key] = ramp
    for si, members in enumerate(sets):
        for i in members:
            p = pops[i]; c = wins[p["win"]]["core"]
            acc[si][c] += p["A"] * ramps[p["win"]]
    acc = acc / wsum.clamp_min(1e-6)[None]
    order = sorted(range(len(sets)), key=lambda k: -float(acc[k].sum()))
    report = {"windows": len(wins), "windows_with_fibres": int(sum(1 for wd in wins.values() if wd["pops"])), "window_populations": len(pops),
              "windows_with_2plus": int(sum(1 for wd in wins.values() if len(wd["pops"]) >= 2)), "links": links, "concatenations": bridges, "infilled_windows": filled, "end_joins": ends,
              "sets": len(sets), "set_sizes": sorted([len(s) for s in sets], reverse=True)[:12], "labels": labels}
    return (acc[order] if sets else torch.zeros((0, Z, Y, X), device=v.device)), report


# ---------------------------------------------------------------------------------------------------------------
# prototype 7b: windowed stage 2 with the activations re-estimated once the sets are known
# ---------------------------------------------------------------------------------------------------------------

def set_extents(pops, wins, labels) -> dict:
    """How far each set runs along its own fibres: the largest distance (voxels) between two of its windows measured
    along the direction of one of them."""
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    out = {}
    for l, mem in by.items():
        C = np.array([wins[pops[i]["win"]]["centre"] for i in mem], dtype=np.float64)
        Dm = np.stack([pops[i]["dir"].cpu().numpy() for i in mem]).astype(np.float64)
        diff = C[None, :, :] - C[:, None, :]                                            # (i, j, 3): c_j - c_i
        out[l] = float(np.abs((diff * Dm[:, None, :]).sum(-1)).max()) if len(mem) > 1 else 0.0
    return out


def _axial_kmeans2(D: np.ndarray, iters: int = 20):
    """Two axial clusters of unit directions ``D (n, 3)``: labels (n,) in {0, 1} and the two axes."""
    G = np.abs(D @ D.T); i, j = np.unravel_index(np.argmin(G), G.shape)               # the two most different members
    axes = np.stack([D[i], D[j]])
    lab = np.zeros(len(D), int)
    for _ in range(iters):
        new = np.argmax(np.abs(D @ axes.T), 1)
        for k in (0, 1):
            M = D[new == k]
            if len(M):
                _, V = np.linalg.eigh(M.T @ M); axes[k] = V[:, -1]
        if (new == lab).all():
            break
        lab = new
    return lab, axes


def split_crossings(pops, wins, labels, *, stride: float, min_angle_deg: float = 40.0, interleave: float = 0.5):
    """Fibre sets do not cross themselves. A set whose members fall into two direction groups (axial 2-means, axes at
    least ``min_angle_deg`` apart: more than consecutive windows turn) is two sets where the groups *cross* — each
    group has members beyond both ends of the other's extent along its own axis, as two fascicles crossing do and the
    two arms of a bend do not — or *interleave* (at least ``interleave`` of the smaller group's windows next to the
    other's, as fibres crossing throughout a region). Returns the labels and the number of splits."""
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    labels = list(labels); nxt = max(labels) + 1 if labels else 0; splits = 0
    cos_min = math.cos(math.radians(min_angle_deg))
    for l, mem in by.items():
        if len(mem) < 4:
            continue
        D = np.stack([pops[i]["dir"].cpu().numpy() for i in mem]).astype(np.float64)
        C = np.array([wins[pops[i]["win"]]["centre"] for i in mem], dtype=np.float64)
        lab, axes = _axial_kmeans2(D)
        if abs(float(axes[0] @ axes[1])) > cos_min or min((lab == 0).sum(), (lab == 1).sum()) < 2:
            continue

        def beyond(k):                                                                     # group k on both sides of the other along axis k
            t = C @ axes[k]; tk, to = t[lab == k], t[lab != k]
            lo, hi = np.quantile(to, 0.1), np.quantile(to, 0.9)
            return bool((tk < lo - stride).any() and (tk > hi + stride).any())
        crossing = beyond(0) and beyond(1)
        small = int(np.argmin([(lab == 0).sum(), (lab == 1).sum()]))
        Cs, Co = C[lab == small], C[lab != small]
        near = (np.abs(Cs[:, None, :] - Co[None, :, :]).max(-1) <= stride * 1.01).any(1).mean()
        if crossing or near >= interleave:
            for i, k in zip(mem, lab):
                if k == 1:
                    labels[i] = nxt
            nxt += 1; splits += 1
    return labels, splits


def prune_short(pops, wins, labels, min_extent: float):
    """Sets that do not run ``min_extent`` voxels along their own fibres are not fibre courses at the windows' scale
    (a window's mixture mode, a band edge, texture): their labels become -1. Returns the labels and a report."""
    ext = set_extents(pops, wins, labels)
    drop = {l for l, e in ext.items() if e < min_extent}
    sizes = {}
    for l in labels:
        sizes[l] = sizes.get(l, 0) + 1
    new = [(-1 if l in drop else l) for l in labels]
    return new, {"min_extent_vox": round(min_extent, 1), "sets_dropped": len(drop), "kept_extents_vox": sorted([round(e) for l, e in ext.items() if l not in drop], reverse=True)[:8],
                 "dropped_sizes": sorted([sizes[l] for l in drop], reverse=True)[:12]}


def _reunmix(cal, Fp, pops, wins, labels, *, reach_windows: int = 2, cone_deg: float = 25.0, min_sep_deg: float = 25.0,
             prefer_alone: bool = False):
    """Once the sets are known: every window re-unmixed jointly onto the local direction of every set that is present
    in it or runs into it along its own axis from at most ``reach_windows`` windows away (within ``cone_deg`` of that
    set's direction there). A set whose window-level detection lost it over a crossing (its windows saw mostly the other
    band) keeps its activation there, and the two directions compete for each voxel as in the pipeline's stage 2.
    ``prefer_alone``: a set's direction is taken from its nearest member in a window it holds alone (an arm, free of a
    second band's pull), the nearest member at all only where no such window is within reach.
    Returns new window populations ``{"win", "dir", "A", "set"}``."""
    from ..pipeline import _normalised
    from ..separate import unmix
    cos_cone = math.cos(math.radians(cone_deg))
    new = []
    for key, wd in wins.items():
        cands = {}                                                                         # set -> (distance, direction)
        for dz in range(-reach_windows, reach_windows + 1):
            for dy in range(-reach_windows, reach_windows + 1):
                for dx in range(-reach_windows, reach_windows + 1):
                    b = (key[0] + dz, key[1] + dy, key[2] + dx)
                    if b not in wins:
                        continue
                    o = np.array(wd["centre"]) - np.array(wins[b]["centre"]); no = np.linalg.norm(o)
                    for i in wins[b]["pops"]:
                        d = pops[i]["dir"]; dn = d.cpu().numpy()
                        if no > 0 and abs(float(o @ dn)) < cos_cone * no:                 # not along this set's axis
                            continue
                        s_ = labels[i]
                        if s_ < 0:                                                         # a dropped set
                            continue
                        rank = no if (not prefer_alone or len(wins[b]["pops"]) == 1) else no + 1e6   # windows it holds alone first
                        if s_ not in cands or rank < cands[s_][0]:
                            cands[s_] = (rank, d)
        if not cands:
            continue
        sets_, dirs = [], []
        for s_, (_, d) in sorted(cands.items(), key=lambda kv: kv[1][0]):
            if all(float((d @ e).abs()) < math.cos(math.radians(min_sep_deg)) for e in dirs):   # two sets along one axis: the nearer
                sets_.append(s_); dirs.append(d)
        modes = torch.stack(dirs)
        A, _ = unmix(Fp[(slice(None),) + wd["core"]], cal.dictionary, modes, prepped=True)
        A = (_normalised(cal, A, modes) / cal.tau).float()
        for k, s_ in enumerate(sets_):
            new.append({"win": key, "dir": dirs[k], "A": A[k], "set": s_})
    return new


def _blend(new, wins, v):
    """Each set's layer: its window activations blended with ramps that peak at the windows' centres."""
    Z, Y, X = v.shape
    ids = sorted({p["set"] for p in new}); idx = {s_: k for k, s_ in enumerate(ids)}
    acc = torch.zeros((len(ids), Z, Y, X), device=v.device); wsum = torch.zeros((Z, Y, X), device=v.device)
    ramps = {}
    for key, wd in wins.items():
        c = wd["core"]; ramp = 1.0
        for ax, s in enumerate(c):
            n = s.stop - s.start
            r = (1.0 - (torch.arange(n, device=v.device, dtype=torch.float32) + 0.5 - n / 2).abs() / (n / 2)).clamp_min(1e-3)
            shp = [1, 1, 1]; shp[ax] = n; ramp = ramp * r.reshape(shp)
        wsum[c] += ramp; ramps[key] = ramp
    for p in new:
        c = wins[p["win"]]["core"]
        acc[idx[p["set"]]][c] += p["A"] * ramps[p["win"]]
    acc = acc / wsum.clamp_min(1e-6)[None]
    order = sorted(range(len(ids)), key=lambda k: -float(acc[k].sum()))
    return acc[order] if ids else torch.zeros((0, Z, Y, X), device=v.device)


def windowed_sets(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, prominence_null: float, share_null: float,
                  window: int = 32, stride: int = 8, region=None, nmax: int = 3, min_sep_deg: float = 25.0, r_min_mm: float = 0.3,
                  noise_deg: float = 10.0, vox_mm: float = 0.02, device: str = "cuda", log=None, prefer_alone: bool = False,
                  reach_windows: int = 2, min_extent: float | None = None, split: bool = False, response=None) -> tuple[torch.Tensor, dict]:
    """:func:`windowed_stage2` for the sets (windows with fibres, the pipeline's stage 2 per window, good continuation,
    concatenation, end-to-end joins), then the activations re-estimated with the sets known (:func:`_reunmix`) and
    blended into one layer per set."""
    say = log or (lambda *_: None)
    Z, Y, X = v.shape
    region = tuple(region or (slice(0, Z), slice(0, Y), slice(0, X)))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    cal_w = cal
    if response is not None:                                                               # the windows' modes from the deconvolved density
        import copy
        cal_w = copy.copy(cal); cal_w.readout = DeconvolvedReadout(cal.readout, response, cal.density_gain)
    D, starts, w = window_densities(cal, cal.dictionary.prep(F), region, window, stride)
    prom = prominence(D); del D
    pops, wins = [], {}
    for iz in range(prom.shape[0]):
        for iy in range(prom.shape[1]):
            for ix in range(prom.shape[2]):
                key = (iz, iy, ix)
                core = tuple(slice(s[k], s[k] + w[a]) for a, (s, k) in enumerate(zip(starts, key)))
                wins[key] = {"core": core, "pops": [], "centre": [(c.start + c.stop) / 2 for c in core]}
                if float(prom[key]) < prominence_null or float(tissue[core].float().mean()) < 0.01:
                    continue
                r = _window_stage2(cal_w, v, tissue, F, core, reach, nmax, min_sep_deg, device)
                if r is None:
                    continue
                modes, A, shares = r
                n = 1 + sum(1 for k in range(1, len(shares)) if float(shares[k]) > share_null)
                for k in range(n):
                    pops.append({"win": key, "dir": modes[k], "A": A[k]}); wins[key]["pops"].append(len(pops) - 1)
    # the joins (as windowed_stage2, without the in-fill); the labels from the union-find
    _, rep = _join_and_blend(pops, wins, v, tissue, window, r_min_mm, noise_deg, vox_mm)
    labels = rep.pop("labels")
    if split:                                                                              # crossing groups within one set
        labels, rep["crossing_splits"] = split_crossings(pops, wins, labels, stride=float(stride))
    if min_extent is not None:                                                             # e.g. ``window``: one window length
        labels, rp = prune_short(pops, wins, labels, min_extent); rep.update(rp)
    new = _reunmix(cal, cal.dictionary.prep(F), pops, wins, labels, min_sep_deg=min_sep_deg, prefer_alone=prefer_alone,
                   reach_windows=reach_windows)
    L = _blend(new, wins, v)
    rep.update({"reunmixed_window_populations": len(new), "sets_after_reunmix": int(L.shape[0])})
    return L, rep


def _reunmix2(cal, v, tissue, Fp, pops, wins, labels, *, reach_windows: int = 3, cone_deg: float = 25.0, min_sep_deg: float = 25.0,
              iterations: int = 1, stride: int = 8):
    """:func:`_reunmix` with each set's local direction made robust — the principal axis of its members' directions
    along its axis within ``reach_windows``, weighted towards windows the set holds alone and towards nearer windows —
    and then ``iterations`` refinements by layer axes (each set's direction in a window taken from its own
    activation-weighted tensor there, then the window unmixed again), as the pipeline's confirmation does."""
    from ..pipeline import _dominant_axis, _normalised
    from ..separate import unmix
    from ..tensors import population_tensor
    cos_cone = math.cos(math.radians(cone_deg))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    Z, Y, X = v.shape
    out = []
    for key, wd in wins.items():
        cands = {}                                                                         # set -> list of (weight, direction)
        for dz in range(-reach_windows, reach_windows + 1):
            for dy in range(-reach_windows, reach_windows + 1):
                for dx in range(-reach_windows, reach_windows + 1):
                    b = (key[0] + dz, key[1] + dy, key[2] + dx)
                    if b not in wins or not wins[b]["pops"]:
                        continue
                    o = np.array(wd["centre"]) - np.array(wins[b]["centre"]); no = np.linalg.norm(o)
                    alone = 1.0 / len(wins[b]["pops"])
                    for i in wins[b]["pops"]:
                        d = pops[i]["dir"]
                        if no > 0 and abs(float(o @ d.cpu().numpy())) < cos_cone * no:
                            continue
                        cands.setdefault(labels[i], []).append((alone / (1.0 + no / stride), d, no))
        if not cands:
            continue
        sets_, dirs = [], []
        for s_, lst in sorted(cands.items(), key=lambda kv: min(t[2] for t in kv[1])):
            M = sum(w_ * torch.outer(d, d) for w_, d, _ in lst)
            _, V = torch.linalg.eigh(M.double()); d = V[:, -1].float()
            if all(float((d @ e).abs()) < math.cos(math.radians(min_sep_deg)) for e in dirs):
                sets_.append(s_); dirs.append(d)
        modes = torch.stack(dirs)
        core = wd["core"]
        ctx = tuple(slice(max(0, c.start - reach), min(n, c.stop + reach)) for c, n in zip(core, (Z, Y, X)))
        inner = tuple(slice(c.start - k.start, c.stop - k.start) for c, k in zip(core, ctx))
        for it in range(iterations + 1):
            A, _ = unmix(Fp[(slice(None),) + ctx], cal.dictionary, modes, prepped=True)
            A = (_normalised(cal, A, modes) / cal.tau).float()
            if it == iterations:
                break
            vb = v[ctx].contiguous(); tb = tissue[ctx]
            w_exc = st.excision_weights(vb, mask=tb)
            prods = st.gradient_products(vb, cal.sigma, cal.truncate)
            reg = torch.zeros_like(tb); reg[inner] = tb[inner]
            new_modes = []
            for k in range(modes.shape[0]):
                _, dk, gk = population_tensor(vb, A[k], cal.sigma, cal.rho, cal.truncate, w_exc, 0.0, None, prods)
                ax = _dominant_axis(dk, (A[k].clamp_min(0) * gk) * reg)
                new_modes.append(ax if float((ax @ modes[k]).abs()) >= math.cos(math.radians(min_sep_deg)) else modes[k])
            modes = torch.stack(new_modes)
        for k, s_ in enumerate(sets_):
            out.append({"win": key, "dir": modes[k], "A": A[k][inner], "set": s_})
    return out


def windowed_sets2(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, *, prominence_null: float, share_null: float,
                   window: int = 32, stride: int = 8, region=None, nmax: int = 3, min_sep_deg: float = 25.0, r_min_mm: float = 0.3,
                   noise_deg: float = 10.0, vox_mm: float = 0.02, iterations: int = 1, device: str = "cuda", log=None):
    """:func:`windowed_sets` with robust set directions and layer-axis refinement (:func:`_reunmix2`)."""
    Z, Y, X = v.shape
    region = tuple(region or (slice(0, Z), slice(0, Y), slice(0, X)))
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    D, starts, w = window_densities(cal, cal.dictionary.prep(F), region, window, stride)
    prom = prominence(D); del D
    pops, wins = [], {}
    for iz in range(prom.shape[0]):
        for iy in range(prom.shape[1]):
            for ix in range(prom.shape[2]):
                key = (iz, iy, ix)
                core = tuple(slice(s[k], s[k] + w[a]) for a, (s, k) in enumerate(zip(starts, key)))
                wins[key] = {"core": core, "pops": [], "centre": [(c.start + c.stop) / 2 for c in core]}
                if float(prom[key]) < prominence_null or float(tissue[core].float().mean()) < 0.01:
                    continue
                r = _window_stage2(cal, v, tissue, F, core, reach, nmax, min_sep_deg, device)
                if r is None:
                    continue
                modes, A, shares = r
                n = 1 + sum(1 for k in range(1, len(shares)) if float(shares[k]) > share_null)
                for k in range(n):
                    pops.append({"win": key, "dir": modes[k], "A": A[k]}); wins[key]["pops"].append(len(pops) - 1)
    _, rep = _join_and_blend(pops, wins, v, tissue, window, r_min_mm, noise_deg, vox_mm)
    labels = rep.pop("labels")
    new = _reunmix2(cal, v, tissue, cal.dictionary.prep(F), pops, wins, labels, min_sep_deg=min_sep_deg, iterations=iterations, stride=stride)
    L = _blend(new, wins, v)
    rep.update({"reunmixed_window_populations": len(new), "sets_after_reunmix": int(L.shape[0])})
    return L, rep
