"""Held-out phantom test of the lag-ODF pathway (``shapetensor.lagodf``: classical, intensity only), with the cases
and scores of ``scripts/test_populations.py`` (the calibrated populations model, the comparison).

Responses: ``synthetic`` (fitted on the calibration bundles, seed 0, at the 12 angles, cells of the core) and
``self`` (estimated per block from its own most anisotropic single-lobe cells). Each with and without momentum.

    python scripts/test_lagodf.py [OUT_DIR] [--cases a,b] [--n-dirs N] [--sigma-w S] [--radii 3,6]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from shapetensor.fibre_sets import References
from shapetensor.lagodf import LagODF, self_response
from shapetensor.phantoms import BEND_GRID, Grid, build, direction
from shapetensor.validate import ANGLES

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("tp", str(HERE / "test_populations.py"))
TP = importlib.util.module_from_spec(spec); spec.loader.exec_module(TP)
P = TP.P
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/lagodf")
    ap.add_argument("--cases", default=""); ap.add_argument("--n-dirs", type=int, default=160)
    ap.add_argument("--sigma-w", type=float, default=10.0); ap.add_argument("--radii", default="3,6")
    ap.add_argument("--along", type=float, default=0.0); ap.add_argument("--min-weight", type=float, default=0.2)
    ap.add_argument("--floor", type=float, default=0.06); ap.add_argument("--full", type=float, default=0.12)
    ap.add_argument("--min-share", type=float, default=0.15); ap.add_argument("--rel", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    tex, texture = P.tex_for("calibrated")
    model = LagODF(n_dirs=a.n_dirs, sigma_w=a.sigma_w, radii=tuple(float(r) for r in a.radii.split(",")), floor=a.floor, full=a.full,
                   min_share=a.min_share, rel=a.rel, window_along=a.along or None, min_weight=a.min_weight, device=DEV)
    cg = Grid(); vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(ang), device=DEV)[0], device=DEV).float()
                         for k, ang in enumerate(ANGLES)]

    def truth(i, O):
        cs = O.shape[1:]; d = torch.as_tensor(direction(ANGLES[i]), device=DEV).float().view(3, 1, 1, 1).expand(3, *cs)
        m = torch.zeros(cs, dtype=torch.bool, device=DEV)
        zs, ys, xs = cg.inner; m[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
        return d, m
    model.fit_response(vols, truth); del vols
    R_syn = model.response
    log("synthetic response", [round(float(x), 2) for x in R_syn[::4]])
    tag = f"n{a.n_dirs}_sw{a.sigma_w}_r{a.radii.replace(',', '-')}_f{a.floor}-{a.full}_s{a.min_share}_rel{a.rel}_al{a.along}_mw{a.min_weight}"
    res = {"settings": vars(a), "synthetic response": R_syn.cpu().numpy().round(3).tolist(), "cases": {}}
    refs = References(Grid().core, "calibrated", DEV)
    cases = list(TP.TEST) + [("texture", None, {})]
    if a.cases:
        cases = [c for c in cases if any(k in c[0] for k in a.cases.split(","))]
    for name, kind, kw in cases:
        if kind is None:
            v, tr, meta = refs.texture, None, {}
            grid = Grid(core=tuple(int(s.stop - s.start) for s in refs.texture_region), margin=tuple(int(s.start) for s in refs.texture_region))
        else:
            img, tr, meta = build(kind, 1, tex=tex, texture=texture, device=DEV, **kw)
            v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        t1 = time.time(); O = model.odf(v); R_self, n_self = self_response(O, model.grid, bin_deg=model.bin_deg)
        rr = {"seconds lag odf": round(time.time() - t1, 1), "self response cells": n_self,
              "self response": [round(float(x), 2) for x in R_self[::4]]}
        for rname, R in (("synthetic", R_syn), ("self", R_self)):
            for mom in (False, True):
                o = model.populations(v, O=O, response=R, use_momentum=mom)
                rr[f"{rname} response, {'momentum' if mom else 'no momentum'}"] = TP.score(o, tr, grid, meta); del o
                rr["presence floor"] = round(model.last_floor, 4)
        res["cases"][name] = rr; log(name, json.dumps(rr))
        json.dump(res, open(out / f"lagodf_{tag}.json", "w"), indent=1)
        del O; torch.cuda.empty_cache()
    log("wrote", out / f"lagodf_{tag}.json")


if __name__ == "__main__":
    main()
