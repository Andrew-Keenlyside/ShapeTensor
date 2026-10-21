"""Held-out phantom test of lag-TCA (``shapetensor.lagtca``, SOBI per window, no calibration) with the same cases and
scores as ``scripts/test_populations.py`` (the populations model, the comparison): per lattice cell of the core, by the
truth's number of populations, the share with N = 0, 1, 2, 3, direction errors and the T-junction's false extension.

    python scripts/test_lagtca_phantoms.py [OUT_DIR] [--keep K] [--pcs K] [--window W] [--stride S]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from shapetensor.features import make_engine
from shapetensor.fibre_sets import References
from shapetensor.lagtca import lagtca_field, reduce_features
from shapetensor.phantoms import BEND_GRID, Grid, build

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("tp", str(HERE / "test_populations.py"))
TP = importlib.util.module_from_spec(spec); spec.loader.exec_module(TP)
spec = importlib.util.spec_from_file_location("proto", str(HERE / "prototype_fibre_sets.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
DEV = "cuda"


def field_cells(F, grid, pad=16, **kw):
    """lag-TCA over the core +- ``pad`` voxels, placed on the full block's 4-voxel cell lattice (cell i at voxel 4i + 2)."""
    cs = tuple(-(-n // 4) for n in F.shape[1:])
    lo = [max(0, s.start - pad) // 4 * 4 for s in grid.inner]; hi = [min(n, s.stop + pad) for s, n in zip(grid.inner, F.shape[1:])]
    dirs, w, rep = lagtca_field(F[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].contiguous(), **kw)
    nmax = w.shape[0]; D = torch.zeros(nmax, 3, *cs, device=F.device); W = torch.zeros(nmax, *cs, device=F.device)
    c0 = [l // 4 for l in lo]; sl = tuple(slice(c, c + n) for c, n in zip(c0, w.shape[1:]))
    D[(slice(None), slice(None)) + sl] = dirs; W[(slice(None),) + sl] = w
    return {"N": (W > 0).sum(0), "mass": W, "dirs_cells": D}, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/lagtca")
    ap.add_argument("--keep", type=int, default=8); ap.add_argument("--pcs", type=int, default=0)
    ap.add_argument("--window", type=int, default=32); ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--min-aniso", type=float, default=0.35); ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    tex, texture = P.tex_for("calibrated")
    eng = make_engine("anatomix_devvit", device=DEV)
    kw = dict(keep=a.keep, window=a.window, stride=a.stride, min_aniso=a.min_aniso, floor=a.floor)
    tag = f"keep{a.keep}_pcs{a.pcs}_w{a.window}_s{a.stride}_a{a.min_aniso}_f{a.floor}"
    res = {"settings": {**kw, "pcs": a.pcs}, "cases": {}}
    refs = References(Grid().core, "calibrated", DEV)
    cases = list(TP.TEST) + [("texture", None, {})]
    if a.cases:
        cases = [c for c in cases if any(k in c[0] for k in a.cases.split(","))]
    for name, kind, kwc in cases:
        if kind is None:
            v, tr, meta, grid = refs.texture, None, {}, Grid(core=tuple(int(s.stop - s.start) for s in refs.texture_region), margin=tuple(int(s.start) for s in refs.texture_region))
        else:
            img, tr, meta = build(kind, 1, tex=tex, texture=texture, device=DEV, **kwc)
            v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        F = eng(v).float()
        if a.pcs:
            F = reduce_features(F, a.pcs)
        t1 = time.time(); o, rep = field_cells(F, grid, **kw); rep["seconds"] = round(time.time() - t1, 1)
        r = TP.score(o, tr, grid, meta); r["report"] = rep
        res["cases"][name] = r; log(name, json.dumps(r))
        json.dump(res, open(out / f"lagtca_{tag}.json", "w"), indent=1)
        del F, o; torch.cuda.empty_cache()
    log("wrote", out / f"lagtca_{tag}.json")


if __name__ == "__main__":
    main()
