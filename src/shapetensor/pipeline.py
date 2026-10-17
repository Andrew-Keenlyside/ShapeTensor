"""The pipeline: shape features → populations in the shape space → one tensor per population → the merged field.

:func:`calibrate` fits the stage-2 readers on a block (or a set of blocks) with the excised structure tensor at its
confident sites as the teacher — no labels. :func:`apply` runs stages 2–4 on a block with a calibration.
:func:`population_field` does both on one block. :func:`run_volume` runs a whole zarr volume chunk by chunk with
halos, calibrated once on a central block.

One pair of scales governs everything, like for like: ``sigma`` (the derivative scale) and ``rho`` (the integration
window) are used by the teacher, by every population's tensor and by the baseline structure tensor the shape tensor
is compared with.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from . import st
from .features import make_engine
from .merge import multi_field, single_field
from .separate import (OrientationReadout, SignatureDictionary, activation_null, direction_gains, gain_at, hemisphere, modes_from_density, population_masks, unmix)
from .tensors import population_tensor


# ---------------------------------------------------------------------------------------------------------------
# the teacher
# ---------------------------------------------------------------------------------------------------------------

def teacher(v: torch.Tensor, sigma: float = 1.0, rho: float = 10.0, truncate: float = 4.0, spacing=None):
    """The excised structure tensor's direction ``(3, Z, Y, X)`` in (z, y, x) and its orientation gap: the
    label-free teacher of the stage-2 readers."""
    S = st.excised_structure_tensor_3d(v, sigma, rho, truncate, spacing=spacing)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def _quantile(x: torch.Tensor, q: float, n: int = 2_000_000) -> torch.Tensor:
    x = x.reshape(-1)
    if x.numel() > n:
        x = x[torch.randint(0, x.numel(), (n,), device=x.device)]
    return torch.quantile(x.float(), q)


def _calibration_weights(d, gap, tissue, q: float = 0.7, n_bins: int = 24):
    """Sampling weights of the confident voxels (gap above its ``q`` quantile in the tissue), stratified so that
    every orientation bin is drawn equally; the background voxels (the least oriented 30% of the tissue); and the
    number of orientation bins occupied."""
    u = hemisphere(n_bins, d.device)
    conf = tissue & (gap > _quantile(gap[tissue], q))
    idx = conf.nonzero()
    dv = d[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    b = ((dv @ u.T) ** 2).argmax(1)
    counts = torch.bincount(b, minlength=len(u)).float()
    w = torch.zeros_like(gap); w[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0 / counts[b].clamp_min(1)
    wb = ((gap < _quantile(gap[tissue], 0.3)) & tissue).float()
    return w, wb, int((counts > 0).sum())


# ---------------------------------------------------------------------------------------------------------------
# calibrate / apply
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Calibration:
    """What stage 2 learned on the calibration block(s): the signature dictionary, the orientation readout and the
    activation null, with the scales they were fitted at."""
    engine: str
    engine_kwargs: dict
    sigma: float
    rho: float
    truncate: float
    smooth: int
    dictionary: SignatureDictionary
    readout: OrientationReadout
    tau: float
    margin: int = 24
    density_gain: torch.Tensor = None
    activation_gain: torch.Tensor = None
    orientation_bins_occupied: int = 0
    readout_overlap: float = 0.0
    _eng: object = field(default=None, repr=False)

    def features(self, v: torch.Tensor) -> torch.Tensor:
        if self._eng is None:
            self._eng = make_engine(self.engine, **({"nyquist": "none"} | self.engine_kwargs))
        return self._eng(v)


def calibrate(volumes, tissue=None, *, engine: str = "anatomix_enc2", engine_kwargs: dict | None = None, sigma: float = 1.0,
              rho: float = 10.0, truncate: float = 4.0, smooth: int = 3, margin: int = 24, n_sample: int = 240_000,
              features=None, spacing=None, device: str = "cuda") -> Calibration:
    """Fit the stage-2 readers. ``volumes``: one block ``(Z, Y, X)`` or a list of blocks (for example phantoms at
    several orientations); ``tissue``: a bool mask per block (None: all voxels); ``features``: the blocks' feature
    fields if already computed. The teacher is the excised structure tensor at ``sigma`` / ``rho``; its confident
    sites (gap above the 70th percentile), stratified over orientation, are the samples; the least oriented 30%
    give the background signature; ``margin`` voxels at the faces are not sampled."""
    vols = volumes if isinstance(volumes, (list, tuple)) else [volumes]
    tiss = tissue if isinstance(tissue, (list, tuple)) else [tissue] * len(vols)
    feats = features if isinstance(features, (list, tuple)) else [features] * len(vols)
    cal = Calibration(engine, dict(engine_kwargs or {}), sigma, rho, truncate, smooth, SignatureDictionary(smooth), OrientationReadout(smooth), 0.0, margin)
    blocks, bg, bins = [], [], 0
    for v, t, F in zip(vols, tiss, feats):
        v = st._to_torch(v, device, torch.float32)
        t = torch.ones_like(v, dtype=torch.bool) if t is None else st._to_torch(t, device) > 0.5
        F = cal.features(v) if F is None else F
        Fp = cal.dictionary.prep(F); del F                                              # the averaged features, once for every reader
        d, gap = teacher(v, sigma, rho, truncate, spacing)
        inner = tuple(slice(margin, n - margin) for n in v.shape)
        w = torch.zeros_like(gap); wb = torch.zeros_like(gap)
        w[inner], wb[inner], nb = _calibration_weights(d[(slice(None),) + inner], gap[inner], t[inner])
        bins = max(bins, nb)
        blocks.append((Fp, d, w)); bg.append((Fp, wb))
    cal.dictionary.fit(blocks, n_sample, background=bg, prepped=True)
    cal.readout.fit(blocks, n_sample, prepped=True)
    cal.density_gain, cal.activation_gain = direction_gains(cal.dictionary, cal.readout, blocks, prepped=True)
    cal.tau = activation_null(cal.dictionary, blocks, smooth=smooth, activation_gain=cal.activation_gain, prepped=True)
    cal.orientation_bins_occupied, cal.readout_overlap = bins, cal.readout.fit_overlap
    return cal


def _dominant_axis(d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The principal axis of the directions ``d (3, Z, Y, X)`` weighted by ``w (Z, Y, X)``."""
    dd = d.reshape(3, -1); ww = w.reshape(-1).clamp_min(0)
    T = (dd * ww) @ dd.T
    _, V = torch.linalg.eigh(T)
    return V[:, -1]


def _candidates(cal: Calibration, Fp: torch.Tensor, reg: torch.Tensor, n: int, min_sep_deg: float, window: int = 64):
    """Candidate populations: the peaks of the block's orientation density and of every ``window``³ sub-window's (a
    block may hold regions of different orientation), gain-normalised, merged ``min_sep_deg`` apart, the block's
    first and then the sub-windows' by their mass, at most ``n``. ``Fp``: the averaged features; one pass of the
    readout gives the sub-window densities and, as their sum, the block's."""
    grid = cal.readout.grid; gain = cal.density_gain if cal.density_gain is not None else 1.0
    if window and window < max(reg.shape):
        W = cal.readout.window_densities(Fp, reg, window, prepped=True) / gain
        dens = [W.sum(0)] + [W[i] for i in torch.argsort(W.sum(1), descending=True) if float(W[i].sum()) > 0]
    else:
        dens = [cal.readout.density(Fp, reg, prepped=True) / gain]
    cos_sep = math.cos(math.radians(min_sep_deg)); cands = []
    for d in dens:
        m, h, k = modes_from_density(d, grid, rel=0.3, min_sep_deg=min_sep_deg, nmax=4, mass_floor=0.1)
        for j in range(k):
            if len(cands) < n and all(float((m[j] @ c).abs()) < cos_sep for c in cands):
                cands.append(m[j])
    return torch.stack(cands)


def _normalised(cal, A, modes):
    """Activations divided by the gain of their mode's direction (every direction counts alike)."""
    if cal.activation_gain is None:
        return A
    return A / gain_at(cal.dictionary, cal.activation_gain, modes)[:, None, None, None]


def _axes(cal, v, Fp, reg, modes, w_exc, spacing, products=None):
    """Each mode's layer axis under a joint unmixing of ``modes``: the activation- and gap-weighted direction of its
    population tensor (no activation floor: the pure activation decides). ``Fp``: the averaged features;
    ``products``: the image's gradient products."""
    A, _ = unmix(Fp, cal.dictionary, modes, prepped=True); A = _normalised(cal, A, modes); out = []
    for k in range(modes.shape[0]):
        _, dk, gk = population_tensor(v, A[k], cal.sigma, cal.rho, cal.truncate, w_exc, 0.0, spacing, products)
        out.append(_dominant_axis(dk, (A[k].clamp_min(0) * gk) * reg))
    return out


def _confirm(cal: Calibration, v, Fp, reg, cand, w_exc, nmax: int, min_sep_deg: float, iterations: int, spacing, products=None):
    """The populations among the candidates, confirmed by their own layers. A candidate's population tensor — the
    image's gradients weighted by its activation — points where its fibres lie. First every candidate is unmixed
    *alone* against the background (well conditioned whatever their number, and fair to every orientation), and
    the distinct axes (``min_sep_deg`` apart) are kept; then the kept axes are unmixed *together* and the distinct
    axes kept again (``iterations`` passes): under a joint unmixing the shoulders of one population lose their
    shared voxels to it and collapse onto its axis, while a real second population keeps its own. The joint pass
    stays within the rank of the orientation code (at most ``nmax + 1`` signatures)."""
    cos_sep = math.cos(math.radians(min_sep_deg))

    def distinct(axes, cap):
        keep = []
        for k in range(len(axes)):
            if all(float((axes[k] @ axes[j]).abs()) < cos_sep for j in keep):
                keep.append(k)
        return torch.stack([axes[k] for k in keep])[:cap]

    if cand.shape[0] <= 1:
        return cand
    axes = []
    for k in range(cand.shape[0]):
        axes += _axes(cal, v, Fp, reg, cand[k:k + 1], w_exc, spacing, products)
    modes = distinct(axes, nmax + 1)
    for _ in range(iterations):
        if modes.shape[0] <= 1:
            break
        modes = distinct(_axes(cal, v, Fp, reg, modes, w_exc, spacing, products), nmax)
    return modes[:nmax]


def apply(cal: Calibration, volume, tissue=None, *, nmax: int = 3, min_sep_deg: float = 25.0, min_mask_share: float = 0.05, iterations: int = 2,
          tensor_floor: float = 0.01, window: int = 64, features=None, spacing=None, timings: dict | None = None, device: str = "cuda") -> dict:
    """Stages 2–4 on ``volume (Z, Y, X)`` with a calibration. Returns a dict (vectors in (z, y, x), no masks applied):

    ``modes (N, 3)`` the populations' directions; ``activation (N, Z, Y, X)`` their layers; ``dirs (N, 3, Z, Y, X)`` and
    ``gap (N, Z, Y, X)`` each layer's own direction and its uniqueness; ``vec (3, Z, Y, X)``, ``vec_gap`` and ``weight``
    the single field (the merged population tensors, the total activation as weight) and ``tensor (6, Z, Y, X)`` the
    merged tensor itself; ``vecs (N, 3, Z, Y, X)`` and
    ``weights`` the multi-vector field (every population everywhere, ordered by activation); ``confidence`` (the
    readout's fit, the share of tissue above the null, the populations' mask shares); ``masks (N, Z, Y, X)``
    (``iterations``: joint refinements of the populations from their own layers; ``nmax``: at most this many;
    ``tensor_floor``: the weight floor of every population tensor, as a fraction of the null; ``window``: the sub-window
    size for candidate populations, 0 = the block only);
    where each population's smoothed activation exceeds the null ``tau``, ``tissue_mask`` where the smoothed *total*
    activation does, ``strongest`` (int8, 1 + index inside the tissue mask); ``residual``, ``tau``, ``n_candidates``.
    ``timings``: a dict that receives the seconds per stage."""
    marks = []

    def mark(name):
        if timings is not None:
            torch.cuda.synchronize(); marks.append((name, time.perf_counter()))
    mark("start")
    v = st._to_torch(volume, device, torch.float32)
    tis = torch.ones_like(v, dtype=torch.bool) if tissue is None else st._to_torch(tissue, device) > 0.5
    F = cal.features(v) if features is None else features; mark("features")
    Fp = cal.dictionary.prep(F); del F                                                  # the averaged features, once for every reader
    reg = torch.zeros_like(tis); inner = tuple(slice(cal.margin, max(cal.margin + 1, n - cal.margin)) for n in v.shape); reg[inner] = tis[inner]
    reg = reg if bool(reg.any()) else tis
    w_exc = st.excision_weights(v, mask=tis if tissue is not None else None, spacing=st.spacing_ratio(spacing))
    prods = st.gradient_products(v, cal.sigma, cal.truncate, spacing=spacing)        # the image's gradient products, once for every tensor
    # candidates from the block's orientation density, confirmed by their own layers: a
    # candidate's population tensor points where its fibres lie; the shoulders of one population all point at it,
    # so the populations are the distinct layer axes (no activation floor here: the pure activation decides)
    cand = _candidates(cal, Fp, reg, nmax + 3, min_sep_deg, window); n = cand.shape[0]
    modes = _confirm(cal, v, Fp, reg, cand, w_exc, nmax, min_sep_deg, iterations, spacing, prods)
    keep = list(range(modes.shape[0]))
    while keep:                                                                         # a population must own a mask
        A, R = unmix(Fp, cal.dictionary, modes[keep], prepped=True); A = _normalised(cal, A, modes[keep])
        masks, _, _ = population_masks(A, cal.tau, cal.smooth)
        share = torch.stack([(m & tis).float().sum() / tis.float().sum().clamp_min(1) for m in masks])
        worst = int(share.argmin())
        if float(share[worst]) >= min_mask_share or len(keep) == 1:
            break
        keep.pop(worst)
    modes = modes[keep]; mark("populations")
    S, dirs, gaps = [], [], []
    for k in range(len(keep)):
        Sk, dk, gk = population_tensor(v, A[k], cal.sigma, cal.rho, cal.truncate, w_exc, tensor_floor * cal.tau, spacing, prods); S.append(Sk); dirs.append(dk); gaps.append(gk)
    S, dirs, gaps = torch.stack(S), torch.stack(dirs), torch.stack(gaps); del prods; mark("tensors")
    vec, vec_gap, weight, T = single_field(S, A, floor=0.1 * cal.tau, return_tensor=True); del S
    vecs, weights = multi_field(dirs, A)
    total = Fn.avg_pool3d(weight[None, None], cal.smooth, stride=1, padding=cal.smooth // 2)[0, 0] if cal.smooth > 1 else weight
    tissue_mask = total > cal.tau
    strongest = (A.argmax(0) + 1).to(torch.int8) * tissue_mask.to(torch.int8); mark("merge")
    if timings is not None:
        timings.update({b: round(tb - ta, 3) for (_, ta), (b, tb) in zip(marks[:-1], marks[1:])}); timings["total"] = round(marks[-1][1] - marks[0][1], 3)
    confidence = {"readout_overlap": round(float(cal.readout_overlap), 3), "tissue_above_null": round(float((tissue_mask & tis).float().sum() / tis.float().sum().clamp_min(1)), 3),
                  "mask_shares": [round(float(x), 3) for x in share]}
    return {"modes": modes, "activation": A, "dirs": dirs, "gap": gaps, "vec": vec, "vec_gap": vec_gap, "weight": weight, "tensor": T, "vecs": vecs, "weights": weights,
            "masks": masks, "tissue_mask": tissue_mask, "strongest": strongest, "residual": R, "tau": cal.tau, "n_candidates": int(n), "confidence": confidence}


def population_field(volume, tissue=None, *, nmax: int = 3, timings: dict | None = None, device: str = "cuda", **calibration) -> dict:
    """Calibrate on ``volume`` and apply to it: the whole pipeline on one block (see :func:`calibrate` for the
    keywords — ``sigma``, ``rho``, ``engine``, ``smooth`` — and :func:`apply` for the result)."""
    t0 = time.perf_counter()
    v = st._to_torch(volume, device, torch.float32)
    cal = calibrate(v, tissue, device=device, **calibration)
    F = cal.features(v)
    if timings is not None:
        torch.cuda.synchronize(); timings["calibration"] = round(time.perf_counter() - t0, 3)
    out = apply(cal, v, tissue, nmax=nmax, features=F, timings=timings, device=device)
    out["calibration"] = cal
    return out


# ---------------------------------------------------------------------------------------------------------------
# a whole volume
# ---------------------------------------------------------------------------------------------------------------

def _ramp(lo: int, hi: int, c0: int, c1: int, blend: int, n_total: int, device) -> torch.Tensor:
    """The blending weight along one axis for the output span ``[lo, hi)`` of a chunk whose core is ``[c0, c1)``: 1
    deep inside, a linear ramp over ``2 * blend`` voxels centred on each core face that has a neighbour (so the two
    chunks' ramps sum to 1 across the seam), 1 up to the volume's own boundary."""
    u = torch.arange(lo, hi, device=device, dtype=torch.float32) + 0.5
    w = torch.ones_like(u)
    if c0 > 0:
        w = w * ((u - (c0 - blend)) / (2 * blend)).clamp(0, 1)
    if c1 < n_total:
        w = w * (((c1 + blend) - u) / (2 * blend)).clamp(0, 1)
    return w


def run_volume(src: str, dst: str, *, level: str | None = None, chunk=(192, 192, 192), nmax: int = 3, tissue_threshold: float | None = None,
               calibration_size: int = 256, calibration_origin=None, calibrate_chunks: bool = True, min_bins: int = 12, blend: int = 32, multi: bool = True,
               spacing=None, rank: int = 0, world: int = 1, device: str = "cuda", log=print, **calibration) -> str:
    """The pipeline over a whole zarr volume, chunk by chunk, one process per GPU (``rank`` / ``world`` split the
    chunk list; blending needs ``world == 1``).

    *Calibration.* A volume calibration is fitted once on a ``calibration_size``³ block at ``calibration_origin``
    (default: the centre); with ``calibrate_chunks`` (the default) every chunk is also calibrated on itself — the
    readers are specific to the local texture, and a calibration from elsewhere in the brain reads a region's
    populations poorly — and the volume calibration is the fallback for a chunk whose teacher covers fewer than
    ``min_bins`` of 24 orientation bins (one bundle, where self-calibration degenerates).

    *Seams.* Each chunk finds its own populations, so its merged tensor differs from its neighbour's near their
    common face even though every voxel's tensors are exact. With ``blend`` > 0 the chunks overlap by ``blend``
    voxels on each side of every face, each chunk's merged tensor (unit trace) and its weight are accumulated with
    linear ramps that sum to 1 across the seam, and the field is the eigensolution of the blended tensor: the
    transition from one chunk's reading to the next is spread over ``2 * blend`` voxels. The network sees the
    overlap plus the filters' reach plus its own margin; stages 2–4 run on the core plus the overlap plus the reach,
    so every blended tensor is exact. ``blend = 0`` writes each chunk's own field (seams at the faces).

    *Output.* ``dst`` receives ``vec (3, Z, Y, X)`` float16 in (x, y, z) component order (as the ``structure-tensor``
    package), ``gap`` and ``weight`` (Z, Y, X), the weight in multiples of the chunk's null ``tau`` (so that chunks
    calibrated on themselves are comparable; ``weight`` ≥ 1: a population present), zero where the volume has no
    data (below ``tissue_threshold``); with ``multi`` also ``vecs (nmax, 3, Z, Y, X)`` and ``weights`` per chunk
    without blending (zero rows where a chunk found fewer populations; the first vector means the same on both sides
    of a seam, the set of populations may not). ``tissue_threshold``: the grey level above which a voxel is tissue
    (None: all); chunks whose core holds less than 1% tissue are skipped and stay zero; a chunk that fails is logged,
    left zero and listed in the attributes. The vectors are never masked."""
    import zarr
    if blend > 0 and world != 1:
        raise ValueError("blending across chunks needs a single process (world == 1); use blend=0 for a split run")
    a = zarr.open(src, mode="r")
    if level is not None:
        a = a[level]
    Z, Y, X = a.shape
    cz, cy, cx = chunk
    sigma, rho, truncate = calibration.get("sigma", 1.0), calibration.get("rho", 10.0), calibration.get("truncate", 4.0)
    reach = int(truncate * (sigma + rho) + 0.5); ext = reach + blend; halo = ext + 24                    # the filters' reach, the overlap, the engine's margin
    g = zarr.open_group(dst, mode="a")
    specs = {"vec": ((3, Z, Y, X), np.float16), "gap": ((Z, Y, X), np.float16), "weight": ((Z, Y, X), np.float16)}
    if multi:
        specs |= {"vecs": ((nmax, 3, Z, Y, X), np.float16), "weights": ((nmax, Z, Y, X), np.float16)}
    if blend > 0:
        specs |= {"_tensor": ((6, Z, Y, X), np.float16), "_wsum": ((Z, Y, X), np.float16)}
    for name, (shape, dt) in specs.items():
        if name not in g:
            g.create_array(name, shape=shape, dtype=dt, chunks=(shape[:-3] + (cz, cy, cx)))
    n = min(calibration_size, Z, Y, X)
    o = tuple((s - n) // 2 for s in (Z, Y, X)) if calibration_origin is None else tuple(calibration_origin)
    vc = torch.as_tensor(np.asarray(a[o[0]:o[0] + n, o[1]:o[1] + n, o[2]:o[2] + n]).astype(np.float32), device=device)
    cal = calibrate(vc, None if tissue_threshold is None else vc >= tissue_threshold, spacing=spacing, device=device, **calibration); del vc
    g.attrs.update({"sigma": sigma, "rho": rho, "nmax": nmax, "halo": halo, "blend": blend, "tau": cal.tau, "engine": cal.engine, "calibration_origin": list(o),
                    "calibration_size": n, "calibrate_chunks": calibrate_chunks, "min_bins": min_bins,
                    "layout": "vec in (x, y, z); weight in multiples of the chunk's null; vecs ordered by activation, zero where absent"})
    log(f"[{rank}/{world}] calibrated on {n}^3 at {o}: orientation bins {cal.orientation_bins_occupied}/24, readout overlap {cal.readout_overlap:.2f}, tau {cal.tau:.3f}")
    tiles = [(z0, y0, x0) for z0 in range(0, Z, cz) for y0 in range(0, Y, cy) for x0 in range(0, X, cx)]
    skipped, failed, chunk_cal = 0, [], {}
    for i, (z0, y0, x0) in enumerate(tiles):
        if i % world != rank:
            continue
        z1, y1, x1 = min(Z, z0 + cz), min(Y, y0 + cy), min(X, x0 + cx)
        lo = (max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)); hi = (min(Z, z1 + halo), min(Y, y1 + halo), min(X, x1 + halo))
        raw = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        core = (slice(z0 - lo[0], z1 - lo[0]), slice(y0 - lo[1], y1 - lo[1]), slice(x0 - lo[2], x1 - lo[2]))
        if tissue_threshold is not None and (raw[core] >= tissue_threshold).mean() < 0.01:     # no tissue in the core: stays zero
            skipped += 1; continue
        blk = torch.as_tensor(raw.astype(np.float32), device=device); del raw
        try:
            F = cal.features(blk)                                                              # the network needs the whole halo ...
            cut = tuple(slice(max(0, c.start - ext), min(n_, c.stop + ext)) for c, n_ in zip(core, blk.shape))   # ... stages 2-4 the core, the overlap and the filters' reach
            blk = blk[cut].contiguous(); F = F[(slice(None),) + cut].contiguous()
            core = tuple(slice(c.start - k.start, c.stop - k.start) for c, k in zip(core, cut))
            tis = None if tissue_threshold is None else blk >= tissue_threshold
            use, which = cal, 2
            if calibrate_chunks:                                                               # the chunk's own readers, unless its teacher holds too few orientations
                own = calibrate(blk, tis, features=F, spacing=spacing, device=device, **calibration)
                if own.orientation_bins_occupied >= min_bins:
                    use, which = own, 1
            chunk_cal[f"{z0},{y0},{x0}"] = which
            out = apply(use, blk, tis, nmax=nmax, spacing=spacing, features=F, device=device); del F
        except Exception as e:                                                                 # a chunk must not end a whole-volume run
            failed.append((z0, y0, x0)); log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}) FAILED: {e!r}"); del blk; torch.cuda.empty_cache(); continue
        data = torch.ones_like(blk) if tissue_threshold is None else (blk >= tissue_threshold).float()    # the weights are zero where the volume has no data
        wt = out["weight"] / max(use.tau, 1e-12) * data
        if blend > 0:
            # the output span: the core widened by the overlap (within the volume); the ramp is the product of the three axes'
            b = blend
            span = tuple(slice(max(c.start - b, 0), min(c.stop + b, n_)) for c, n_ in zip(core, blk.shape))
            out_sl = tuple(slice(lo_ + sp.start, lo_ + sp.stop) for lo_, sp in zip([k.start + l_ for k, l_ in zip(cut, lo)], span))
            rz = _ramp(out_sl[0].start, out_sl[0].stop, z0, z1, b, Z, device); ry = _ramp(out_sl[1].start, out_sl[1].stop, y0, y1, b, Y, device); rx = _ramp(out_sl[2].start, out_sl[2].stop, x0, x1, b, X, device)
            ramp = rz[:, None, None] * ry[None, :, None] * rx[None, None, :]
            T = out["tensor"][(slice(None),) + span]; T = T / T[:3].sum(0).clamp_min(1e-12)[None]        # unit trace: the blend is of orientations, the weight is blended apart
            acc = torch.as_tensor(np.asarray(g["_tensor"][(slice(None),) + out_sl]).astype(np.float32), device=device)
            g["_tensor"][(slice(None),) + out_sl] = (acc + T * ramp[None]).half().cpu().numpy(); del acc, T
            acc = torch.as_tensor(np.asarray(g["_wsum"][out_sl]).astype(np.float32), device=device)
            g["_wsum"][out_sl] = (acc + ramp).half().cpu().numpy()
            acc = torch.as_tensor(np.asarray(g["weight"][out_sl]).astype(np.float32), device=device)
            g["weight"][out_sl] = (acc + wt[span] * ramp).half().cpu().numpy(); del acc, ramp
        else:
            out_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            g["vec"][(slice(None),) + out_sl] = out["vec"].flip(0)[(slice(None),) + core].half().cpu().numpy()
            g["gap"][out_sl] = out["vec_gap"][core].half().cpu().numpy(); g["weight"][out_sl] = wt[core].half().cpu().numpy()
        if multi:
            out_core = (slice(z0, z1), slice(y0, y1), slice(x0, x1)); N = out["vecs"].shape[0]
            vecs = torch.zeros((nmax, 3) + tuple(blk.shape), device=device); vecs[:N] = out["vecs"].flip(1)
            wts = torch.zeros((nmax,) + tuple(blk.shape), device=device); wts[:N] = out["weights"] / max(use.tau, 1e-12) * data
            g["vecs"][(slice(None), slice(None)) + out_core] = vecs[(slice(None), slice(None)) + core].half().cpu().numpy()
            g["weights"][(slice(None),) + out_core] = wts[(slice(None),) + core].half().cpu().numpy()
        npop = int(out["modes"].shape[0]); del out, blk; torch.cuda.empty_cache()
        log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}): {npop} populations" + (" (volume calibration)" if which == 2 and calibrate_chunks else ""))
    if skipped or failed:
        log(f"[{rank}/{world}] {skipped} chunks without tissue skipped; {len(failed)} failed: {failed}")
    g.attrs.update({"failed_chunks": [list(f) for f in failed], "chunk_calibration": chunk_cal})
    if blend > 0:                                                                              # the field of the blended tensor, chunk by chunk; the accumulators removed
        for i, (z0, y0, x0) in enumerate(tiles):
            z1, y1, x1 = min(Z, z0 + cz), min(Y, y0 + cy), min(X, x0 + cx); out_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            ws = torch.as_tensor(np.asarray(g["_wsum"][out_sl]).astype(np.float32), device=device)
            if not bool((ws > 0).any()):
                continue
            T = torch.as_tensor(np.asarray(g["_tensor"][(slice(None),) + out_sl]).astype(np.float32), device=device) / ws.clamp_min(1e-6)[None]
            val, vec = st.eig_special_3d(T, device=device)
            gap = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1) * (ws > 0).float()
            g["vec"][(slice(None),) + out_sl] = (vec * (ws > 0).float()[None]).half().cpu().numpy(); g["gap"][out_sl] = gap.half().cpu().numpy()
            g["weight"][out_sl] = (torch.as_tensor(np.asarray(g["weight"][out_sl]).astype(np.float32), device=device) / ws.clamp_min(1e-6)).half().cpu().numpy()
            del T, val, vec, gap, ws
        for name in ("_tensor", "_wsum"):
            del g[name]
        log(f"[{rank}/{world}] blended field written")
    return dst
