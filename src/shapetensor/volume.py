"""The chain over a whole array, chunk by chunk, on one or several GPUs.

Every stage of :class:`shapetensor.field.Orientations` is local: the lift reaches ``r_max + 3 scale`` voxels, the
carry-forward ``reach`` cells further, the field's lobes are read per voxel. So a chunk computed with a halo of that
size gives, on its core, exactly what one pass over the whole array would give -- provided the two block-level
quantities, the presence floor and the response, are shared. :func:`run_volume` measures both once on a calibration
region (the centre by default, or any region given) and applies them to every chunk; no blending, no seams. Population
identities are labelled per chunk and made unique across chunks by an offset; :func:`stitch_labels` joins them across
chunk faces afterwards.

Output (a zarr group): ``vecs (nmax, 3, Z, Y, X)`` float16 in (x, y, z) component order (as the ``structure-tensor``
package), ``weights (nmax, Z, Y, X)`` float16, ``labels (nmax, Z, Y, X)`` int32, ``vec (3, Z, Y, X)`` and ``weight`` for
the single field, ``N`` uint8, ``sheet`` uint8; parameters, floor and response in the attributes. Nothing is masked;
chunks whose core holds no tissue (``tissue_threshold``) are skipped and stay zero.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from .field import Orientations


def run_volume(src: str, dst: str, *, level: str | None = None, chunk=(256, 256, 256), tissue_threshold: float | None = None,
               calibration_size: int = 256, calibration_origin=None, floor: float | None = None, response=None, bias=None,
               null: str = "isotropic", rank: int = 0, world: int = 1, device: str = "cuda", log=print, labels: bool = True, **kwargs) -> str:
    """The chain over the zarr array ``src`` (or its ``level``) into the group ``dst``. ``kwargs`` are
    :class:`Orientations` parameters (``scale`` above all). ``floor`` / ``response`` / ``bias``: given, or measured on the
    ``calibration_size``³ region at ``calibration_origin`` (the centre) -- with ``null="spectrum"`` the region's
    full-spectrum surrogate gives a directional bias (the acquisition's anisotropy, but also the region's dominant
    texture orientation) that is subtracted from every chunk's lift; ``"isotropic"`` (the default) measures the floor only. ``rank`` / ``world`` split the chunk list between processes (one per GPU)."""
    import zarr
    t0 = time.time()
    a = zarr.open(src, mode="r")
    if level is not None:
        a = a[level]
    Z, Y, X = a.shape
    model = Orientations(device=device, null=null, **kwargs)
    nmax = model.nmax
    halo = int(math.ceil((model.edge + model.reach * model.step + model.step) / model.step)) * model.step   # a multiple of the lattice step
    cz, cy, cx = chunk
    if any(c % model.step for c in chunk):
        raise ValueError(f"chunk sizes must be multiples of the lattice step ({model.step}) so that every chunk's lattice is the one-pass lattice")
    g = zarr.open_group(dst, mode="a")
    specs = {"vecs": ((nmax, 3, Z, Y, X), np.float16), "weights": ((nmax, Z, Y, X), np.float16), "vec": ((3, Z, Y, X), np.float16),
             "weight": ((Z, Y, X), np.float16), "N": ((Z, Y, X), np.uint8), "sheet": ((Z, Y, X), np.uint8)}
    if labels:
        specs["labels"] = ((nmax, Z, Y, X), np.int32)
    for name, (shape, dt) in specs.items():
        if name not in g:
            g.create_array(name, shape=shape, dtype=dt, chunks=(shape[:-3] + (cz, cy, cx)))
    # the volume's null (bias and floor) and response, once
    if floor is None or response is None or (bias is None and null == "spectrum"):
        n = min(calibration_size, Z, Y, X)
        o = tuple((s - n) // 2 for s in (Z, Y, X)) if calibration_origin is None else tuple(calibration_origin)
        vc = torch.as_tensor(np.asarray(a[o[0]:o[0] + n, o[1]:o[1] + n, o[2]:o[2] + n]).astype(np.float32), device=device)
        mask = None if tissue_threshold is None else vc >= tissue_threshold
        if bias is None and null == "spectrum":
            bias = model.null_profile(vc, mask)
            gz = (model.grid @ torch.tensor([1.0, 0.0, 0.0], device=device)).abs()
            log(f"[{rank}/{world}] spectrum null: directional bias along z {float(bias[gz > 0.94].mean()):+.4f}, in-plane {float(bias[gz < 0.34].mean()):+.4f}")
        model.bias = None if bias is None else torch.as_tensor(bias, device=device).float()
        O = model.lift(vc)
        if model.bias is not None:
            O = O - model.bias.view(-1, 1, 1, 1)
        if response is None:
            from .lift import self_response
            response, n_cells = self_response(O, model.grid, bin_deg=model.bin_deg)
            log(f"[{rank}/{world}] response from {n_cells} single-lobe cells of the {n}^3 region at {o}")
        if floor is None:
            floor = model.null_floor(vc, mask)
            log(f"[{rank}/{world}] presence floor {floor:.4f} ({null} surrogate of the region)")
        del vc, O
    model.response = response
    model.bias = None if bias is None else torch.as_tensor(bias, device=device).float()
    g.attrs.update({"pipeline": "shapetensor.field.Orientations", "scale": model.scale, "step": model.step, "lags": [model.r_min, model.r_max],
                    "n_dirs": int(model.grid.shape[0]), "nmax": nmax, "min_sep_deg": model.min_sep_deg, "momentum": model.momentum,
                    "reach_cells": model.reach, "sigma_along_cells": model.sigma_along, "mean_scale_voxels": model.sigma_m,
                    "excise": model.excise, "halo": halo, "floor": float(floor), "response": [float(x) for x in response],
                    "null": null, "bias": None if model.bias is None else [float(x) for x in model.bias],
                    "tissue_threshold": tissue_threshold, "chunk": list(chunk),
                    "layout": "vecs/vec in (x, y, z); weights in [0, 1] (0 = absent); labels unique per chunk (stitch_labels joins them)"})
    tiles = [(z0, y0, x0) for z0 in range(0, Z, cz) for y0 in range(0, Y, cy) for x0 in range(0, X, cx)]
    skipped, failed, times = 0, [], []
    for i, (z0, y0, x0) in enumerate(tiles):
        if i % world != rank:
            continue
        z1, y1, x1 = min(Z, z0 + cz), min(Y, y0 + cy), min(X, x0 + cx)
        lo = (max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)); hi = (min(Z, z1 + halo), min(Y, y1 + halo), min(X, x1 + halo))
        raw = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        core = (slice(z0 - lo[0], z1 - lo[0]), slice(y0 - lo[1], y1 - lo[1]), slice(x0 - lo[2], x1 - lo[2]))
        if tissue_threshold is not None and (raw[core] >= tissue_threshold).mean() < 0.01:
            skipped += 1; continue
        t1 = time.time()
        blk = torch.as_tensor(raw.astype(np.float32), device=device); del raw
        try:
            out = model.run(blk, floor=float(floor), response=response, labels=labels)
        except Exception as e:                                                                 # a chunk must not end a whole-volume run
            failed.append((z0, y0, x0)); log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}) FAILED: {e!r}"); del blk
            torch.cuda.empty_cache(); continue
        data = 1.0 if tissue_threshold is None else (blk >= tissue_threshold).float()[core]
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        g["vecs"][(slice(None), slice(None)) + sl] = out["vecs"].flip(1)[(slice(None), slice(None)) + core].half().cpu().numpy()
        g["weights"][(slice(None),) + sl] = (out["weights"][(slice(None),) + core] * data).half().cpu().numpy()
        g["vec"][(slice(None),) + sl] = out["vec"].flip(0)[(slice(None),) + core].half().cpu().numpy()
        g["weight"][sl] = (out["weight"][core] * data).half().cpu().numpy()
        g["N"][sl] = out["N"][core].to(torch.uint8).cpu().numpy()
        g["sheet"][sl] = out["sheet"][core].to(torch.uint8).cpu().numpy()
        if labels:
            lab = out["labels"][(slice(None),) + core]
            lab = torch.where(lab > 0, lab + (i + 1) * 100_000, torch.zeros_like(lab))         # unique across chunks
            g["labels"][(slice(None),) + sl] = lab.to(torch.int32).cpu().numpy()
        n_lab = len(out.get("table", [])); del out, blk
        torch.cuda.empty_cache()
        times.append(time.time() - t1)
        log(f"[{rank}/{world}] chunk {i + 1}/{len(tiles)} ({z0}, {y0}, {x0}): {n_lab} populations, {times[-1]:.1f} s")
    g.attrs.update({"failed_chunks": [list(f) for f in failed], "skipped_chunks": skipped,
                    "seconds_per_chunk_median": round(float(np.median(times)), 1) if times else None, "seconds_total": round(time.time() - t0, 1)})
    log(f"[{rank}/{world}] done: {len(times)} chunks, {skipped} skipped, {len(failed)} failed, {time.time() - t0:.0f} s")
    return dst


def stitch_labels(dst: str, *, max_angle_deg: float = 20.0, log=print) -> int:
    """Join the per-chunk population labels of ``dst`` across chunk faces: on every face, the fixels of the two
    neighbouring voxels that are each other's best-aligned within ``max_angle_deg`` and both labelled are linked, and
    linked labels become one (union-find). Rewrites ``labels``; returns the number of populations."""
    import zarr
    g = zarr.open_group(dst, mode="a")
    L = g["labels"]; V = g["vecs"]
    K, Z, Y, X = L.shape[0], *L.shape[1:]
    cz, cy, cx = L.chunks[-3:]
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    cos_max = math.cos(math.radians(max_angle_deg))
    for axis, (n, c) in enumerate(zip((Z, Y, X), (cz, cy, cx))):
        for f in range(c, n, c):                                                                 # the face between slab f-1 and f
            sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3; sl_a[axis] = slice(f - 1, f); sl_b[axis] = slice(f, f + 1)
            la = np.asarray(L[(slice(None),) + tuple(sl_a)]); lb = np.asarray(L[(slice(None),) + tuple(sl_b)])
            if not (la > 0).any() or not (lb > 0).any():
                continue
            va = np.asarray(V[(slice(None), slice(None)) + tuple(sl_a)]).astype(np.float32); vb = np.asarray(V[(slice(None), slice(None)) + tuple(sl_b)]).astype(np.float32)
            C = np.abs(np.einsum("ic...,jc...->ij...", va, vb))                                   # (K, K, ...)
            C = np.where((la[:, None] > 0) & (lb[None, :] > 0), C, -1.0)
            best_b = C.argmax(1); best_a = C.argmax(0)
            for k1 in range(K):
                k2 = best_b[k1]
                c = np.take_along_axis(C[k1], k2[None], 0)[0]
                mutual = np.take_along_axis(best_a, k2[None], 0)[0] == k1
                ok = mutual & (c >= cos_max)
                A = la[k1][ok]; B = np.take_along_axis(lb, k2[None], 0)[0][ok]
                for x, y in set(zip(A.tolist(), B.tolist())):
                    union(x, y)
    # relabel: roots -> 1..n by size (vectorised: unique labels per slab, a lookup table, one gather)
    counts = {}
    for z0 in range(0, Z, cz):
        blk = np.asarray(L[:, z0:z0 + cz])
        u, cnt = np.unique(blk[blk > 0], return_counts=True)
        for x, n_ in zip(u.tolist(), cnt.tolist()):
            r = find(x); counts[r] = counts.get(r, 0) + n_
    order = sorted(counts, key=lambda r: -counts[r]); remap = {r: i + 1 for i, r in enumerate(order)}
    for z0 in range(0, Z, cz):
        blk = np.asarray(L[:, z0:z0 + cz])
        if not (blk > 0).any():
            continue
        u, inv = np.unique(blk, return_inverse=True)
        lut = np.array([remap[find(int(x))] if x > 0 else 0 for x in u], dtype=blk.dtype)
        L[:, z0:z0 + cz] = lut[inv].reshape(blk.shape)
    g.attrs.update({"populations": len(order), "labels_stitched": True})
    log(f"labels stitched: {len(order)} populations")
    return len(order)
