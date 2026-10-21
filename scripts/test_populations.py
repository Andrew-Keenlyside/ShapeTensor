"""Held-out test of the flexible-N population model (``shapetensor.populations``), with and without momentum.

Calibration (seed 0): single-population samples from the 12 straight bundles, the bend, its control and the two band
crossings (confident cells of the excised structure tensor; tissue = intensity mask on the bend and bands); synthetic
crossings at 90, 60 and 40° with their two directions; synthetic fibre-free texture.
Test (seed 1): a bundle at 60°, the bend, the straight fascicles, the band crossings at 90 and 60°, the T-junction at
90°, crossings at 90° and 60° (angles calibrated, new seed) and 75° and 50° (angles never calibrated), texture.

Per lattice cell of the core, by the truth's number of populations: the share with N = 0, 1, 2, 3 found; for one
population the top lobe's direction error; for two, the error of the best pairing of the two strongest lobes with the
two truths (mean) and the share with both within 15°; for none, the share with N = 0. T-junction: the share of cells in
the second band's would-be continuation beyond the first band that hold a lobe within 20° of the second band's
direction (false extension).

    python scripts/test_populations.py [OUT_DIR]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.features import make_engine
from shapetensor.fibre_sets import References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, build, direction
from shapetensor.populations import PopulationModel
from shapetensor.validate import ANGLES, angle_deg

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/populations")
DEV = "cuda"
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
TEST = (("bundle 60", "bundle", {"angle": 60.0}), ("bend", "bend", {}), ("straight", "bend", {"straight": True}),
        ("band crossing 90", "band_crossing", {"cross_deg": 90.0}), ("band crossing 60", "band_crossing", {"cross_deg": 60.0}),
        ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}),
        ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}),
        ("crossing 75 (new angle)", "crossing", {"cross_deg": 75.0}), ("crossing 50 (new angle)", "crossing", {"cross_deg": 50.0}))


def cell_truth(tr, cs):
    zc = np.clip(np.arange(cs[0]) * 4 + 2, 0, tr["npop"].shape[0] - 1); xc = np.clip(np.arange(cs[2]) * 4 + 2, 0, tr["npop"].shape[1] - 1)
    npop = torch.as_tensor(tr["npop"][np.ix_(zc, xc)], device=DEV).long()[:, None, :].expand(cs)
    d1 = torch.as_tensor(tr["dir1"][np.ix_(zc, xc)], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, *cs).nan_to_num(0)
    d2 = torch.as_tensor(tr["dir2"][np.ix_(zc, xc)], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, *cs).nan_to_num(0)
    extra = {}
    if "corridor2" in tr:
        extra["beyond"] = torch.as_tensor(tr["corridor2"][np.ix_(zc, xc)] & ~tr["band1"][np.ix_(zc, xc)] & ~tr["band2"][np.ix_(zc, xc)], device=DEV)[:, None, :].expand(cs)
    return npop, d1, d2, extra


def score(o, tr, grid, meta):
    N, mass, D = o["N"], o["mass"], o["dirs_cells"]; cs = N.shape
    zs, ys, xs = grid.inner
    inner = torch.zeros(cs, dtype=torch.bool, device=DEV)
    inner[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
    r = {}
    if tr is None:
        sel = inner; r["none (texture)"] = {"cells": int(sel.sum()), **{f"N={k}": round(float((N[sel] == k).float().mean()), 3) for k in range(4)}}
        return r
    npop, d1, d2, extra = cell_truth(tr, cs)
    for cat, k in (("none (texture)", 0), ("one population", 1), ("two populations", 2)):
        sel = inner & (npop == k)
        if int(sel.sum()) == 0:
            continue
        rr = {"cells": int(sel.sum()), **{f"N={j}": round(float((N[sel] == j).float().mean()), 3) for j in range(4)}}
        if k == 1:
            s1 = sel & (N >= 1)
            if int(s1.sum()):
                rr["top lobe error median"] = round(float(angle_deg(D[0], d1)[s1].median()), 1)
        if k == 2:
            s2 = sel & (N >= 2)
            if int(s2.sum()):
                e11, e12, e21, e22 = angle_deg(D[0], d1), angle_deg(D[0], d2), angle_deg(D[1], d1), angle_deg(D[1], d2)
                sw = (e12 + e21) < (e11 + e22); ea, eb = torch.where(sw, e12, e11), torch.where(sw, e21, e22)
                rr["pair error median (N>=2)"] = round(float(((ea + eb) / 2)[s2].median()), 1)
                rr["both within 15 (of all two-population cells)"] = round(float(((ea <= 15) & (eb <= 15) & (N >= 2))[sel].float().mean()), 3)
        r[cat] = rr
    if "beyond" in extra:
        sel = inner & extra["beyond"]
        if int(sel.sum()):
            b2 = torch.as_tensor(direction(meta["angles"][1]), device=DEV).view(3, 1, 1, 1).expand(3, *cs)
            hit = torch.zeros(cs, dtype=torch.bool, device=DEV)
            for j in range(D.shape[0]):
                hit |= (mass[j] > 0) & (angle_deg(D[j], b2) <= 20)
            r["false extension beyond the T"] = round(float(hit[sel].float().mean()), 3)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    tex, texture = P.tex_for("calibrated"); cg = Grid()
    eng = make_engine("anatomix_devvit", device=DEV)
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES)]
    tis = [None] * len(vols)
    for kind, kw in (("bend", {}), ("bend", {"straight": True}), ("band_crossing", {"cross_deg": 90.0}), ("band_crossing", {"cross_deg": 60.0})):
        img = build(kind, 0, tex=tex, texture=texture, device=DEV, **kw)[0]
        v = torch.as_tensor(img, device=DEV).float(); vols.append(v); tis.append(v > P.otsu(img))
    crossings = []
    for c in (90.0, 60.0, 40.0):
        img = build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=c)[0]
        crossings.append((torch.as_tensor(img, device=DEV).float(), direction(45 + c / 2), direction(45 - c / 2)))
    model = PopulationModel().fit(eng, vols, tis, crossings=crossings, iso=iso_textures("calibrated", DEV)); del vols, tis, crossings
    log("fitted", json.dumps(model.report))
    res = {"model": model.report, "cases": {}}
    refs = References(Grid().core, "calibrated", DEV)
    for name, kind, kw in list(TEST) + [("texture", None, {})]:
        if kind is None:
            v, tr, meta, grid = refs.texture, None, {}, Grid(core=tuple(int(s.stop - s.start) for s in refs.texture_region))
        else:
            img, tr, meta = build(kind, 1, tex=tex, texture=texture, device=DEV, **kw)
            v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        rr = {}
        for mom in (False, True):
            o = model.populations(eng, v, use_momentum=mom)
            rr["momentum" if mom else "no momentum"] = score(o, tr, grid, meta); del o
        res["cases"][name] = rr; log(name, json.dumps(rr))
        json.dump(res, open(OUT / "test_populations.json", "w"), indent=1)
        torch.cuda.empty_cache()
    log("wrote", OUT / "test_populations.json")


if __name__ == "__main__":
    main()
