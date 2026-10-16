"""Shared pieces of the bending / wrapping / structure study.

- extract():  a build's vertices, eigenvectors, keys and graph edges (from the graph checkpoint), cached as an npz;
              the walk replays reproduce the stored streamlines exactly.
- walks:      the pipeline's pointer walk (bridge.stages.junction) and the two-hop momentum prototype.
- Anatomy:    SynthSeg-derived fields in world mm: label, signed depth below the white-matter surface (the interface of
              cerebral/cerebellar WM + deep structures with cortex, CSF or outside) and its normal, the cortex parcel
              nearest each voxel, the depth below the brain's closed hull (gyral crown vs sulcal fundus).
- resample(): every streamline at 0.5 mm of arc, so bend and tangent measures have a physical scale.
"""
import glob, json, time
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit
from scipy import ndimage as ndi

STEP = 0.5                                                     # resampling step, mm
CORTEX = lambda S: (S == 3) | (S == 42) | ((S >= 1000) & (S < 3000))
CEREB_CTX = (8, 47)
DEEP = (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
WM = (2, 41)
INNER = WM + DEEP + (7, 46, 16, 4, 5, 43, 44, 14, 15)          # inside the "white-matter surface"
ZONES = ["cortex", "superficial WM (<3 mm)", "deep WM", "deep grey", "brainstem", "cerebellar WM", "cerebellar cortex",
         "ventricle/CSF", "outside"]
SS_S2146 = "/hdd/BRIDGE_tutorial/synthseg_500um.nii.gz"         # 1 mm, S2146 (MNI-warped)
SS_MNI = "/hdd/data/hcp_tracts/mni_synthseg.nii.gz"            # 1 mm, the MNI template (HCP-1065's own anatomy)
HCP = "/hdd/data/hcp_tracts/hcp1065_whole_brain.trk"
REPLAYS = Path("/home/andrew/bridge_wmcov/replays")


# --------------------------------------------------------------------------- builds
def extract(study_root, label, affine):
    """pos (store z,y,x voxel-centred), vec, keys, pairs of a build; world = affine @ pos[:, ::-1]."""
    REPLAYS.mkdir(parents=True, exist_ok=True)
    out = REPLAYS / f"{label}.npz"
    if out.exists():
        z = np.load(out)
        return {k: z[k] for k in z.files}
    from bridge.ids import pack
    from bridge.utils.store.vertices import _open_level_group, _read_chunk_columns, find_vertex_chunk_ids
    der = Path(glob.glob(str(Path(study_root) / "derivatives/BRIDGE/sub-*/micr/*desc-bridge*.ome.zarr"))[0])
    store = str(der / "graph.zarrvectors")
    cfg = json.loads((der / "state" / "project.json").read_text())["config"]["values"]
    lg, _ = _open_level_group(store)
    P_, V_, K_, G_, I_ = [], [], [], [], []
    for cid in find_vertex_chunk_ids(store):
        cc = tuple(int(c) for c in cid.split("-")); pos, at = _read_chunk_columns(lg, cc)
        if pos is None or len(pos) == 0:
            continue
        P_.append(pos); V_.append(at["principal_eigenvector"]); K_.append(at["voxel_key"]); I_.append(at["img"])
        G_.append(pack(np.asarray(cc), np.arange(len(pos))))
    pos = np.concatenate(P_).astype(np.float64); vec = np.concatenate(V_).astype(np.float64)
    keys = np.concatenate(K_).astype(np.int64); gid = np.concatenate(G_); img = np.concatenate(I_)
    o = np.argsort(gid); gs = gid[o]
    E = []
    for f in sorted((der / "graph.zarrvectors" / "graph").glob("*.npz")):
        e = np.load(f, allow_pickle=True); E.append(np.stack([e["src"], e["dst"]], 1))
    E = np.concatenate(E)

    def idx(gg):
        k = np.clip(np.searchsorted(gs, gg), 0, len(gs) - 1)
        return o[k], gs[k] == gg
    a, oka = idx(E[:, 0]); b, okb = idx(E[:, 1])
    pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1)[oka & okb & (a != b)], axis=0)
    rep = {"pos": pos, "vec": vec, "keys": keys, "pairs": pairs, "img": img, "affine": np.asarray(affine, float),
           "max_bend": float(cfg["graphs"]["max_bend_deg"]), "reach": float(cfg["graphs"]["max_edge_distance"]),
           "density": float(cfg["vertices"]["sampling_density"])}
    np.savez(out, **rep)
    return rep


def world_of(rep):
    return nib.affines.apply_affine(rep["affine"], rep["pos"][:, ::-1])


def vec_world(rep):
    """Eigenvectors in world x,y,z: stored vectors are in the store's z,y,x order, like the positions."""
    v = rep["vec"][:, ::-1] @ np.asarray(rep["affine"], float)[:3, :3].T
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def pointer_walk(rep, max_bend=None):
    from bridge.stages import junction as J
    flat, offs = J.walk_streamlines(rep["pos"], rep["vec"], rep["pairs"], rep["keys"],
                                    max_bend_deg=float(rep["max_bend"] if max_bend is None else max_bend), cost="lateral")
    return np.asarray(flat, np.int64), np.asarray(offs, np.int64)


def csr(rep):
    nV = len(rep["pos"]); p = rep["pairs"]
    s_ = np.concatenate([p[:, 0], p[:, 1]]); d_ = np.concatenate([p[:, 1], p[:, 0]])
    o_ = np.argsort(s_, kind="stable")
    return np.concatenate([[0], np.cumsum(np.bincount(s_, minlength=nV))]).astype(np.int64), d_[o_].astype(np.int64)


@njit(cache=True, nogil=True)
def momentum2_nb(world, ev, indptr, nbr, seeds, cone_deg, memory_mm, lam, max_steps, hops):
    """The two-hop momentum walk prototype (cosmo_fix2.py): from each seed both ways; at each vertex the candidate (a
    neighbour, or a neighbour's neighbour) best continuing the momentum within cone_deg, with lam x its own misalignment;
    momentum = exponential average of the steps over memory_mm; no revisits."""
    cmin = np.cos(np.radians(cone_deg))
    out = np.empty(seeds.shape[0] * 2 * max_steps, np.int64); offs = np.zeros(seeds.shape[0] + 1, np.int64)
    buf = np.empty((2, max_steps), np.int64); nb = np.zeros(2, np.int64); pos_out = 0
    for si in range(seeds.shape[0]):
        s0 = seeds[si]
        for half in range(2):
            m = ev[s0] * (1.0 if half == 0 else -1.0); u = s0; buf[half, 0] = u; k = 1
            while k < max_steps:
                best = -1; bscore = 1e9
                for p in range(indptr[u], indptr[u + 1]):
                    x1 = nbr[p]
                    nq = 1 if hops == 1 else 1 + indptr[x1 + 1] - indptr[x1]
                    for qq in range(nq):
                        x = x1 if qq == 0 else nbr[indptr[x1] + qq - 1]
                        if x == u:
                            continue
                        t = world[x] - world[u]; tn = np.sqrt((t * t).sum())
                        if tn <= 0:
                            continue
                        t = t / tn; c = (t * m).sum()
                        if c < cmin:
                            continue
                        sc = (1.0 - c) + lam * (1.0 - abs((t * ev[x]).sum())) - 0.02 * tn
                        if sc < bscore:
                            seen = False
                            for h in range(2):
                                lim = k if h == half else (nb[h] if half == 1 else 0)
                                for q in range(lim):
                                    if buf[h, q] == x:
                                        seen = True; break
                                if seen:
                                    break
                            if not seen:
                                best = x; bscore = sc
                if best < 0:
                    break
                t = world[best] - world[u]; tn = np.sqrt((t * t).sum()); t = t / tn
                beta = np.exp(-tn / memory_mm); m = beta * m + (1.0 - beta) * t; m = m / np.sqrt((m * m).sum())
                u = best; buf[half, k] = u; k += 1
            nb[half] = k
        for q in range(nb[1] - 1, 0, -1):
            out[pos_out] = buf[1, q]; pos_out += 1
        for q in range(nb[0]):
            out[pos_out] = buf[0, q]; pos_out += 1
        offs[si + 1] = pos_out
    return out[:pos_out], offs


@njit(cache=True, nogil=True)
def momentum2t_nb(world, ev, indptr, nbr, seeds, cone_deg, memory_mm, lam, max_steps, hops, terminal, nout, radial_cos):
    """The two-hop momentum walk prototype (cosmo_fix2.py): from each seed both ways; at each vertex the candidate (a
    neighbour, or a neighbour's neighbour) best continuing the momentum within cone_deg, with lam x its own misalignment;
    momentum = exponential average of the steps over memory_mm; no revisits."""
    cmin = np.cos(np.radians(cone_deg))
    out = np.empty(seeds.shape[0] * 2 * max_steps, np.int64); offs = np.zeros(seeds.shape[0] + 1, np.int64)
    buf = np.empty((2, max_steps), np.int64); nb = np.zeros(2, np.int64); pos_out = 0
    for si in range(seeds.shape[0]):
        s0 = seeds[si]
        for half in range(2):
            m = ev[s0] * (1.0 if half == 0 else -1.0); u = s0; buf[half, 0] = u; k = 1
            while k < max_steps:
                best = -1; bscore = 1e9
                for p in range(indptr[u], indptr[u + 1]):
                    x1 = nbr[p]
                    nq = 1 if hops == 1 else 1 + indptr[x1 + 1] - indptr[x1]
                    for qq in range(nq):
                        x = x1 if qq == 0 else nbr[indptr[x1] + qq - 1]
                        if x == u:
                            continue
                        t = world[x] - world[u]; tn = np.sqrt((t * t).sum())
                        if tn <= 0:
                            continue
                        t = t / tn; c = (t * m).sum()
                        if c < cmin:
                            continue
                        sc = (1.0 - c) + lam * (1.0 - abs((t * ev[x]).sum())) - 0.02 * tn
                        if sc < bscore:
                            seen = False
                            for h in range(2):
                                lim = k if h == half else (nb[h] if half == 1 else 0)
                                for q in range(lim):
                                    if buf[h, q] == x:
                                        seen = True; break
                                if seen:
                                    break
                            if not seen:
                                best = x; bscore = sc
                if best < 0:
                    break
                t = world[best] - world[u]; tn = np.sqrt((t * t).sum()); t = t / tn
                beta = np.exp(-tn / memory_mm); m = beta * m + (1.0 - beta) * t; m = m / np.sqrt((m * m).sum())
                u = best; buf[half, k] = u; k += 1
                if terminal[u] and (radial_cos <= -1.0 or (t * nout[u]).sum() > radial_cos):
                    break
            nb[half] = k
        for q in range(nb[1] - 1, 0, -1):
            out[pos_out] = buf[1, q]; pos_out += 1
        for q in range(nb[0]):
            out[pos_out] = buf[0, q]; pos_out += 1
        offs[si + 1] = pos_out
    return out[:pos_out], offs


def momentum_walk(rep, world, n_seeds=500_000, cone=45.0, memory=3.0, lam=0.3, hops=2, seed=1, terminal=None,
                  nout=None, radial_cos=-2.0, threads=8):
    """Seeds split over threads (the walk kernels release the GIL); output in seed order, as a single call gives."""
    from concurrent.futures import ThreadPoolExecutor
    indptr, nbr = csr(rep)
    ev = vec_world(rep)
    nV = len(world); seeds = np.sort(np.random.default_rng(seed).choice(nV, min(n_seeds, nV), replace=False)).astype(np.int64)
    world = np.ascontiguousarray(world, np.float64); ev = np.ascontiguousarray(ev, np.float64)
    if terminal is not None:
        term = terminal.astype(np.bool_); no = np.ascontiguousarray(nout if nout is not None else np.zeros((nV, 3)), np.float64)
        fn = lambda sd: momentum2t_nb(world, ev, indptr, nbr, sd, cone, memory, lam, 800, hops, term, no, float(radial_cos))
    else:
        fn = lambda sd: momentum2_nb(world, ev, indptr, nbr, sd, cone, memory, lam, 800, hops)
    parts = np.array_split(seeds, threads)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(fn, parts))
    flat = np.concatenate([r[0] for r in res])
    offs = [np.zeros(1, np.int64)]; base = 0
    for f, o in res:
        offs.append(o[1:] + base); base += len(f)
    return flat, np.concatenate(offs)


# --------------------------------------------------------------------------- resampling
@njit(cache=True)
def resample_nb(pts, off, step):
    """Each polyline at `step` of arc (linear interpolation, first point kept, last partial step dropped)."""
    n = off.shape[0] - 1
    cnt = np.zeros(n, np.int64)
    for s in range(n):
        a, b = off[s], off[s + 1]; arc = 0.0
        for i in range(a + 1, b):
            arc += np.sqrt(((pts[i] - pts[i - 1]) ** 2).sum())
        cnt[s] = int(arc / step) + 1 if b - a >= 2 else 0
    o2 = np.zeros(n + 1, np.int64)
    for s in range(n):
        o2[s + 1] = o2[s] + cnt[s]
    out = np.empty((o2[n], 3))
    for s in range(n):
        a, b = off[s], off[s + 1]
        if cnt[s] == 0:
            continue
        k = o2[s]; out[k] = pts[a]; k += 1; target = step; acc = 0.0; i = a + 1
        while k < o2[s + 1] and i < b:
            seg = np.sqrt(((pts[i] - pts[i - 1]) ** 2).sum())
            while k < o2[s + 1] and acc + seg >= target:
                f = (target - acc) / seg if seg > 0 else 0.0
                out[k] = pts[i - 1] + f * (pts[i] - pts[i - 1]); k += 1; target += step
            acc += seg; i += 1
    return out, o2


def subset(flat, offs, frac, seed=0):
    k = np.random.default_rng(seed).random(len(offs) - 1) < frac
    L = np.diff(offs)[k]
    idx = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in np.flatnonzero(k)]) if k.any() else np.zeros(0, np.int64)
    return flat[idx], np.concatenate([[0], np.cumsum(L)]).astype(np.int64)


def subset_fast(flat, offs, frac, seed=0):
    k = np.random.default_rng(seed).random(len(offs) - 1) < frac
    sid = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    return flat[k[sid]], np.concatenate([[0], np.cumsum(np.diff(offs)[k])]).astype(np.int64)


def load_hcp(frac=1.0, seed=0):
    sl = nib.streamlines.load(HCP).streamlines
    P = np.asarray(sl._data, np.float64); L = np.asarray(sl._lengths); off = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    if frac < 1:
        flat = np.arange(len(P)); f2, off = subset_fast(flat, off, frac, seed); P = P[f2]
    return P, off


# --------------------------------------------------------------------------- anatomy
class Anatomy:
    def __init__(self, path):
        im = nib.load(path); self.S = np.asarray(im.dataobj).astype(np.int32); self.A = im.affine; self.Ai = np.linalg.inv(im.affine)
        S = self.S; vox = float(abs(im.affine[0, 0]))
        inner = np.isin(S, INNER)
        self.sdf = (ndi.distance_transform_edt(inner) - ndi.distance_transform_edt(~inner)) * vox   # + inside WM, mm
        g = ndi.gaussian_filter(self.sdf, 1.0 / vox)
        self.grad = np.stack(np.gradient(g), 0)
        ctx = CORTEX(S) | np.isin(S, CEREB_CTX)
        _d, near = ndi.distance_transform_edt(~ctx, return_indices=True)
        self.near_ctx = S[tuple(near)]
        tissue = S > 0
        r = 8.0 / vox                                                          # closed hull, 8 mm
        dil = ndi.distance_transform_edt(~tissue) <= r
        closed = ndi.distance_transform_edt(dil) > r
        self.hull_depth = ndi.distance_transform_edt(closed) * vox
        zone = np.full(S.shape, 8, np.int8)
        zone[CORTEX(S)] = 0; zone[np.isin(S, WM) & (self.sdf < 3)] = 1; zone[np.isin(S, WM) & (self.sdf >= 3)] = 2
        zone[np.isin(S, DEEP)] = 3; zone[S == 16] = 4; zone[np.isin(S, (7, 46))] = 5; zone[np.isin(S, CEREB_CTX)] = 6
        zone[np.isin(S, (4, 5, 14, 15, 24, 43, 44, 72))] = 7
        self.zone = zone
        # interface voxels: inside-WM voxels with an outside neighbour that is cortex
        self.iface = inner & ndi.binary_dilation(ctx) & ~ctx

    def vox(self, w):
        return nib.affines.apply_affine(self.Ai, w)

    def sample(self, w):
        """Per world point: zone, label, signed WM depth (mm), unit normal (pointing into WM), nearest cortex label, hull depth."""
        q = self.vox(w); qi = tuple(np.clip(np.rint(q).astype(np.int64), 0, np.array(self.S.shape) - 1).T)
        sdf = ndi.map_coordinates(self.sdf, q.T, order=1, mode="nearest")
        n = np.stack([ndi.map_coordinates(self.grad[d], q.T, order=1, mode="nearest") for d in range(3)], 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        return {"zone": self.zone[qi], "label": self.S[qi], "sdf": sdf, "normal": n, "near_ctx": self.near_ctx[qi],
                "hull": self.hull_depth[qi]}


class MaskField:
    """A build's own mask: signed depth (mm, + inside) and normal on a 0.4 mm grid from f^3 block fractions of level 0
    (f = 0.4 mm / voxel)."""

    def __init__(self, mask_zarr, src_affine, cache=REPLAYS / "mask_sdf_0p4.npz"):
        if Path(cache).exists():
            z = np.load(cache); self.sdf, self.A = z["sdf"], z["A"]
        else:
            import zarr
            vox = float(abs(src_affine[0, 0])); f = max(1, int(round(0.4 / vox)))
            a = zarr.open(mask_zarr, mode="r")["0"]; sh = np.array(a.shape)
            out = np.zeros(tuple(-(-sh // f)), np.float32)
            C = 512
            for z0 in range(0, sh[0], C):
                for y0 in range(0, sh[1], C):
                    for x0 in range(0, sh[2], C):
                        b = np.asarray(a[z0:z0 + C, y0:y0 + C, x0:x0 + C], np.float32)
                        pad = [(0, (-s) % f) for s in b.shape]; b = np.pad(b, pad)
                        r = b.reshape(b.shape[0] // f, f, b.shape[1] // f, f, b.shape[2] // f, f).mean((1, 3, 5))
                        out[z0 // f:z0 // f + r.shape[0], y0 // f:y0 // f + r.shape[1], x0 // f:x0 // f + r.shape[2]] = r
            inside = out.transpose(2, 1, 0) >= 0.5                                      # x, y, z
            v = vox * f
            self.sdf = ((ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)) * v).astype(np.float32)
            self.A = src_affine.copy(); self.A[:3, :3] = src_affine[:3, :3] * f
            self.A[:3, 3] = src_affine[:3, 3] + vox * (f - 1) / 2.0
            np.savez(cache, sdf=self.sdf, A=self.A)
        self.Ai = np.linalg.inv(self.A)
        g = ndi.gaussian_filter(self.sdf, 1.0 / abs(self.A[0, 0]))                       # 1 mm
        self.grad = np.stack(np.gradient(g), 0)

    def sample(self, w):
        q = nib.affines.apply_affine(self.Ai, w)
        sdf = ndi.map_coordinates(self.sdf, q.T, order=1, mode="nearest")
        n = np.stack([ndi.map_coordinates(self.grad[d], q.T, order=1, mode="nearest") for d in range(3)], 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        return sdf, n


# --------------------------------------------------------------------------- measures
def tangents(R, o2, half=2):
    """Unit tangent at each resampled point from the chord over +-half steps (clipped at the ends)."""
    n = len(R); i = np.arange(n); sid = np.repeat(np.arange(len(o2) - 1), np.diff(o2))
    lo = np.maximum(i - half, o2[sid]); hi = np.minimum(i + half, o2[sid + 1] - 1)
    t = R[hi] - R[lo]
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9), sid


def chord_angles(R, o2, L_mm):
    """Angle (deg) between the chords p(s-L)->p(s) and p(s)->p(s+L) at every resampled point with both chords inside."""
    k = int(round(L_mm / STEP)); n = len(R); i = np.arange(n); sid = np.repeat(np.arange(len(o2) - 1), np.diff(o2))
    ok = (i - k >= o2[sid]) & (i + k <= o2[sid + 1] - 1)
    a = R[i[ok]] - R[i[ok] - k]; b = R[i[ok] + k] - R[i[ok]]
    c = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1))), i[ok], sid[ok]


def runs(flag, sid):
    """Lengths (in samples) and start indices of runs of True that do not cross streamlines."""
    f = flag.astype(np.int8); brk = np.r_[True, sid[1:] != sid[:-1]]
    start = f.astype(bool) & (brk | np.r_[True, f[:-1] == 0])
    st = np.flatnonzero(start)
    end = f.astype(bool) & (np.r_[sid[1:] != sid[:-1], True] | np.r_[f[1:] == 0, True])
    en = np.flatnonzero(end)
    return en - st + 1, st


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- walks with terminal vertices
def pointer_walk_terminal(rep, terminal, max_bend=None, step_ok=None):
    """The pipeline's walk with an ACT-like stop that stays a function on steps: a step whose TARGET vertex is
    terminal (e.g. within d of the mask edge) visits it and ends there. If *step_ok* is given (per step, 2n bool), only
    steps where it is False end at a terminal target (e.g. end only tangential arrivals). Sources, both-ways joins and
    dedup follow junction.walk_steps exactly."""
    from bridge.stages import junction as J
    mb = float(rep["max_bend"] if max_bend is None else max_bend)
    pos = rep["pos"]
    succ, to_v, to_s = J.step_successors(pos, rep["vec"], rep["pairs"], rep["keys"], max_bend_deg=mb, cost="lateral")
    have = to_v >= 0
    stop = have & terminal[np.where(have, to_v, 0)]
    if step_ok is not None:
        stop &= ~step_ok
    succ = np.where(stop & (succ >= 0), -1, succ)
    n2 = succ.shape[0]
    indeg = np.zeros(n2, dtype=np.int64); ok = succ >= 0
    np.add.at(indeg, succ[ok], 1)
    src = (succ != -2) & (indeg == 0)
    sources = np.flatnonzero(src); u = sources >> 1
    both = src[2 * u] & src[2 * u + 1]
    sources = sources[~(both & ((sources & 1) == 1))]
    both_ways = J._both_ways(np.asarray(pos, np.float64), to_v, sources, mb)
    # a terminal source vertex does not join its two halves (the walk would run through the surface band)
    both_ways &= ~terminal[sources >> 1]
    flat, off = J._walk_streamlines_nb(sources.astype(np.int64), both_ways, succ, to_v)
    flat, off = J.dedup_reversed(flat, off)
    return np.asarray(flat, np.int64), np.asarray(off, np.int64)


def step_directions(rep, world):
    """Unit direction of every vertex side's pointer step (2n x 3; zeros where none) and its target."""
    from bridge.stages import junction as J
    to_v, to_s = J.side_pointers(rep["pos"], rep["vec"], rep["pairs"], rep["keys"], cost="lateral")
    u = np.arange(len(to_v)) >> 1
    d = np.zeros((len(to_v), 3)); h = to_v >= 0
    d[h] = world[to_v[h]] - world[u[h]]
    d[h] /= np.maximum(np.linalg.norm(d[h], axis=1, keepdims=True), 1e-12)
    return d, to_v
