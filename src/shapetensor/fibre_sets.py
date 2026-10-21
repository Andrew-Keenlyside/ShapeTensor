"""Stage 2 as fibre sets: populations found per window and joined by continuity along the fibres (the dev pipeline).

In the released pipeline a population is one direction for the whole block, so a bundle that turns through more than the
separation angle inside a block is split between populations. Here a population is a set of fibres that continue
through space, and the number of populations is an output (no clustering, no number to choose):

1. **Windows.** The block is covered by ``window``³ windows every ``stride`` voxels. A window holds fibres when the
   prominence of its orientation density exceeds the level reached on fibre-free texture of the same size.
2. **Populations per window.** The released stage 2 runs inside each fibre window; a second population is kept only
   where its share exceeds the largest second share seen in single-bundle windows.
3. **Links.** A window population links to its mutually closest neighbour along its own fibre direction when their
   directions differ by no more than a turn of radius ``r_min_mm`` allows; sideways neighbours only by ``noise_deg``.
4. **Joins.** Facing chain ends are bridged across short gaps; collinear pieces are concatenated; end-to-end joins are
   made within a window (a bend), never for a population that continues on both sides (a crossing).
5. **Splitting and pruning.** A set whose members fall into two crossing or interleaving direction groups is split; a
   set must run ``min_extent`` voxels along its own fibres (shorter: a mixture mode, a band edge, texture).
6. **Re-unmixing.** Every window is unmixed again onto the surviving sets' local directions and the window layers are
   ramp-blended into one activation layer per set (in multiples of the null).

The thresholds come without labels from synthetic references read through the block's own readers (:class:`References`):
texture without fibres (the window-presence null) and straight bundles at four orientations (the second-population null
and the readout's one-population response, used to deconvolve window densities).

Promoted from ``shapetensor.experimental.fibre_sets`` (2026-10-20): the functions below are the prototype's
``windowed_sets`` path, unchanged; the experimental module keeps the other prototypes as the record.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as Fn

class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


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


def prominence(D: torch.Tensor) -> torch.Tensor:
    """Each window's top density over its mean density, ``(nz, ny, nx)``: how far one direction stands out."""
    return D.max(0).values / D.mean(0).clamp_min(1e-12)


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
    from .pipeline import apply
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
    from .pipeline import _normalised
    from .separate import unmix
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
    rep.update({"prominence_null": round(float(prominence_null), 3), "window_prominence_max": round(float(prom.max()), 3),
                "window_prominence_median": round(float(prom.median()), 3), "share_null": round(float(share_null), 4)})   # presence diagnostics
    new = _reunmix(cal, cal.dictionary.prep(F), pops, wins, labels, min_sep_deg=min_sep_deg, prefer_alone=prefer_alone,
                   reach_windows=reach_windows)
    L = _blend(new, wins, v)
    rep.update({"reunmixed_window_populations": len(new), "sets_after_reunmix": int(L.shape[0])})
    return L, rep



# ---------------------------------------------------------------------------------------------------------------
# configuration, label-free thresholds, the entry point
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class FibreSetConfig:
    """Stage 2's settings (the values the prototypes were validated with)."""
    window: int = 32
    stride: int = 8
    nmax: int = 3
    min_sep_deg: float = 25.0
    r_min_mm: float = 0.3
    noise_deg: float = 10.0
    min_extent: float | None = 32.0
    split: bool = True
    deconvolve: bool = True
    reach_windows: int = 2
    share_every: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class References:
    """Synthetic references for stage 2's thresholds, read through a calibration's readers: texture without fibres as
    large as the analysed region (the window-presence null: more windows reach a higher maximum, so the reference must
    hold as many) and straight bundles at 0, 45, 90 and 135° (the second-population null and the readout's
    one-population response), at the phantom texture of ``setting`` (``"calibrated"``: matched to S-21-46 at 20 µm).

    Their features depend only on the engine, so they are computed once (with the first calibration's engine) and
    reused for every calibration; the thresholds depend on the readers and are computed per calibration."""

    def __init__(self, region_shape, setting: str = "calibrated", device: str = "cuda", seed: int = 7):
        from .phantoms import Grid, TextureParams, build, direction, finish
        from .validate import SETTINGS
        p = SETTINGS[setting]
        tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
        t_ = TextureParams(**{**tex.__dict__, "amp": tex.amp * p["texture"]})
        gt = Grid(core=tuple(int(n) for n in region_shape))
        Zg, Xg = gt.shape[0], gt.shape[2]
        self.texture = torch.as_tensor(finish(gt, t_, np.full((Zg, Xg), t_.levels["gm"], np.float32), np.zeros((Zg, Xg), np.float32),
                                              torch.zeros(gt.shape, device=device), np.random.default_rng([seed, 4242])), device=device).float()
        self.texture_region = gt.inner
        gb = Grid()
        self.angles = (0.0, 45.0, 90.0, 135.0)
        self.bundles = [torch.as_tensor(build("bundle", 300 + k, gb, tex, texture=p["texture"], angle=a, device=device)[0], device=device).float()
                        for k, a in enumerate(self.angles)]
        self.dirs = [direction(a) for a in self.angles]
        self.bundle_region = gb.inner
        self.setting, self.device = setting, device
        self._feat = None

    def _features(self, cal):
        if self._feat is None:
            self._feat = {"texture": cal.features(self.texture), "bundles": [cal.features(b) for b in self.bundles]}
        return self._feat

    def nulls(self, cal, cfg: FibreSetConfig) -> dict:
        """``{"prominence", "share", "response"}`` for this calibration's readers."""
        f = self._features(cal)
        D, _, _ = window_densities(cal, cal.dictionary.prep(f["texture"]), self.texture_region, cfg.window, cfg.stride)
        prom = float(prominence(D).max()); del D
        share = _second_share(cal, self.bundles, f["bundles"], self.bundle_region, cfg)
        resp = _response(cal, f["bundles"], self.dirs, self.bundle_region, cfg) if cfg.deconvolve else None
        return {"prominence": prom, "share": share, "response": resp}


def _second_share(cal, vols, feats, region, cfg) -> float:
    """:func:`second_share_null` with the bundles' features precomputed."""
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    top = 0.0
    grid = _window_grid(region, cfg.window, cfg.stride)
    for vol, F in zip(vols, feats):
        t = torch.ones_like(vol, dtype=torch.bool)
        for z0 in grid[0][0][::cfg.share_every]:
            for y0 in grid[1][0][::cfg.share_every]:
                for x0 in grid[2][0][::cfg.share_every]:
                    core = (slice(z0, z0 + grid[0][1]), slice(y0, y0 + grid[1][1]), slice(x0, x0 + grid[2][1]))
                    r = _window_stage2(cal, vol, t, F, core, reach, cfg.nmax, cfg.min_sep_deg, str(vol.device))
                    if r is not None and len(r[2]) >= 2:
                        top = max(top, float(r[2][1]))
    return top


def _response(cal, feats, dirs, region, cfg, bin_deg: float = 2.0):
    """:func:`readout_response` with the bundles' features precomputed."""
    u = cal.readout.grid
    bins = np.arange(0.0, 90.0 + bin_deg, bin_deg); prof = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1)
    for F, d in zip(feats, dirs):
        D, _, _ = window_densities(cal, cal.dictionary.prep(F), region, cfg.window, cfg.stride)
        D = D.reshape(D.shape[0], -1); D = D / D.sum(0, keepdim=True).clamp_min(1e-12)
        d = torch.as_tensor(np.asarray(d), device=u.device, dtype=u.dtype)
        th = torch.rad2deg(torch.arccos((u @ d).abs().clamp(max=1))).cpu().numpy()
        idx = np.clip(np.digitize(th, bins) - 1, 0, len(prof) - 1)
        np.add.at(prof, idx, D.mean(1).cpu().numpy()); np.add.at(cnt, idx, 1)
    keep = cnt > 0
    return 0.5 * (bins[1:] + bins[:-1])[keep], prof[keep] / cnt[keep]


def fibre_sets(cal, v: torch.Tensor, tissue: torch.Tensor, F: torch.Tensor, nulls: dict, cfg: FibreSetConfig | None = None, *,
               region=None, vox_mm: float = 0.02, device: str = "cuda", log=None) -> tuple[torch.Tensor, dict]:
    """Stage 2 as fibre sets on a block: the activation layers ``(N, Z, Y, X)`` in multiples of the null, largest set
    first, and a report. ``nulls``: :meth:`References.nulls` for this calibration."""
    cfg = cfg or FibreSetConfig()
    return windowed_sets(cal, v, tissue, F, prominence_null=nulls["prominence"], share_null=nulls["share"], window=cfg.window,
                         stride=cfg.stride, region=region, nmax=cfg.nmax, min_sep_deg=cfg.min_sep_deg, r_min_mm=cfg.r_min_mm,
                         noise_deg=cfg.noise_deg, vox_mm=vox_mm, device=device, log=log, reach_windows=cfg.reach_windows,
                         min_extent=cfg.min_extent, split=cfg.split, response=nulls.get("response") if cfg.deconvolve else None)
