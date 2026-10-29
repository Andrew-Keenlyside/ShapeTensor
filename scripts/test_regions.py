"""Held-out test of the regions model (``shapetensor.regions``): calibrated on seed-0 phantoms and bundles, scored on seed 1.

Calibration: the 12 straight bundles, the bend, its straight control and the two band crossings (seed 0; tissue = the
intensity mask on the bend and bands) for the oriented samples; synthetic fibre-free texture for the texture samples.
Whole-block crossings are left out: they are what the reject option must catch.

Test (seed 1): a bundle at 60°, the bend, the straight fascicles, the band crossings at 90 and 60°, the T-junction at
90°, the crossings at 90 and 60°, and fibre-free texture. Per lattice cell (4³ voxels) of the core, by the truth: one
population -- the share given each state and the direction error of the oriented ones; two populations (crossings,
overlaps) -- the share unassigned-fibre (flagged), oriented (and its error to the nearer population), texture; none
(grey-matter texture) -- the share texture, oriented (a false fibre), unassigned. At every voxel with one population:
the field's median error where it has a weight, and its coverage, against the standard and the excised structure tensor.

    python scripts/test_regions.py [OUT_DIR]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor import st
from shapetensor.features import make_engine
from shapetensor.fibre_sets import References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, build
from shapetensor.regions import ORIENTED, TEXTURE, UNASSIGNED_FIBRE, UNASSIGNED_TEXTURE, RegionModel
from shapetensor.validate import ANGLES, angle_deg

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/regions")
DEV = "cuda"
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
STATES = {TEXTURE: "texture", ORIENTED: "oriented", UNASSIGNED_FIBRE: "unassigned fibre-like", UNASSIGNED_TEXTURE: "unassigned texture-like"}
TEST = (("bundle 60", "bundle", {"angle": 60.0}), ("bend", "bend", {}), ("straight", "bend", {"straight": True}),
        ("band crossing 90", "band_crossing", {"cross_deg": 90.0}), ("band crossing 60", "band_crossing", {"cross_deg": 60.0}),
        ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}), ("crossing 90", "crossing", {"cross_deg": 90.0}),
        ("crossing 60", "crossing", {"cross_deg": 60.0}))


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
    model = RegionModel().fit(eng, vols, tis, iso=iso_textures("calibrated", DEV)); del vols, tis
    log("fitted", json.dumps(model.report))
    res = {"model": model.report, "cases": {}}
    cases = list(TEST) + [("texture", None, {})]
    for name, kind, kw in cases:
        if kind is None:
            v = References(Grid().core, "calibrated", DEV).texture; grid = Grid(core=tuple(int(s.stop - s.start) for s in References(Grid().core, "calibrated", DEV).texture_region))
            tr = None
        else:
            img, tr, _ = build(kind, 1, tex=tex, texture=texture, device=DEV, **kw)
            v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        o = model.field(eng, v); a = o["lattice"]
        state = a["state"]; cs = state.shape
        zc = torch.arange(cs[0], device=DEV) * 4 + 2; xc = torch.arange(cs[2], device=DEV) * 4 + 2
        zs, ys, xs = grid.inner
        inner_c = torch.zeros(cs, dtype=torch.bool, device=DEV)
        inner_c[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
        if tr is None:
            npop = torch.zeros(cs, device=DEV, dtype=torch.long); d1 = d2 = None
        else:
            zcc, xcc = zc.clamp(max=tr["npop"].shape[0] - 1).cpu().numpy(), xc.clamp(max=tr["npop"].shape[1] - 1).cpu().numpy()
            npop = torch.as_tensor(tr["npop"][np.ix_(zcc, xcc)], device=DEV).long()[:, None, :].expand(cs)
            d1 = torch.as_tensor(tr["dir1"][np.ix_(zcc, xcc)], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, *cs)
            d2 = torch.as_tensor(tr["dir2"][np.ix_(zcc, xcc)], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, *cs)
        r = {}
        for cat, sel in (("one population", (npop == 1) & inner_c), ("two populations", (npop == 2) & inner_c), ("none (texture)", (npop == 0) & inner_c)):
            n = int(sel.sum())
            if n == 0:
                continue
            rr = {"cells": n, **{STATES[k]: round(float((state[sel] == k).float().mean()), 3) for k in STATES}}
            orient = sel & (state == ORIENTED)
            if int(orient.sum()) and d1 is not None:
                e = angle_deg(a["direction"], d1.nan_to_num(0))
                if cat == "two populations":
                    e = torch.minimum(e, angle_deg(a["direction"], d2.nan_to_num(0)))
                rr["oriented: direction error median"] = round(float(e[orient].median()), 1)
            r[cat] = rr
        if tr is not None:                                                                        # the field at every voxel with one population
            from shapetensor.phantoms import truth_field
            t1 = truth_field(tr, "dir1", v.shape[1]).to(DEV)
            one = torch.as_tensor(tr["npop"] == 1, device=DEV)[:, None, :].expand(v.shape).clone()
            m3 = torch.zeros_like(one); m3[grid.inner] = True; one &= m3
            cover = one & (o["weight"] > 0.05)
            e_reg = angle_deg(o["vec"], t1)
            d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=False, order="zyx", device=DEV)
            d_ex = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=True, order="zyx", device=DEV)
            r["field, one population"] = {"coverage": round(float(cover.sum() / one.sum().clamp_min(1)), 3),
                                          "regions median error (covered)": round(float(e_reg[cover].median()), 1) if int(cover.sum()) else None,
                                          "standard structure tensor median error": round(float(angle_deg(d_st, t1)[one].median()), 1),
                                          "excised structure tensor median error": round(float(angle_deg(d_ex, t1)[one].median()), 1)}
        res["cases"][name] = r; log(name, json.dumps(r))
        json.dump(res, open(OUT / "test_regions.json", "w"), indent=1)
        del o, v; torch.cuda.empty_cache()
    log("wrote", OUT / "test_regions.json")


if __name__ == "__main__":
    main()
