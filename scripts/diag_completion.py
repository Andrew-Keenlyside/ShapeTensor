"""Inside a band crossing's overlap, with and without completion: each set's steering direction against the truth, each
set's activation (relative to its arms), and the multi-vector field's two vectors' errors separately (which one fails).

    python scripts/diag_completion.py [OUT_DIR]       CASE="band crossing 90" | "band crossing 60"
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.completion import steering
from shapetensor.dev import DevConfig, apply_dev
from shapetensor.fibre_sets import FibreSetConfig, References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES, angle_deg

torch.use_deterministic_algorithms(True, warn_only=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/completion")
DEV = "cuda"; CASE = os.environ.get("CASE", "band crossing 90")
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def main():
    t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    tex, texture = P.tex_for("calibrated"); cg = Grid(); grid = BEND_GRID
    cross = float(CASE.split()[-1])
    img, tr, _ = build("band_crossing", 0, tex=tex, texture=texture, device=DEV, cross_deg=cross)
    v = torch.as_tensor(img, device=DEV).float(); tissue = v > P.otsu(img)
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES)]
    cal = calibrate(vols + [v], [None] * len(vols) + [tissue], sigma=1.0, rho=10.0, truncate=3.0, margin=cg.margin[0], engine="anatomix_devvit",
                    engine_kwargs={}, smooth=3, seed=0, device=DEV, iso=iso_textures("calibrated", DEV)); del vols
    refs = References(tuple(s.stop - s.start for s in grid.inner), "calibrated", DEV); F = cal.features(v)
    zs, ys, xs = grid.inner
    b1 = torch.as_tensor(tr["band1"], device=DEV); b2 = torch.as_tensor(tr["band2"], device=DEV)
    core = torch.zeros_like(b1); core[zs, xs] = True
    ov = (b1 & b2 & core)[:, None, :].expand(v.shape).clone(); ov[:, :ys.start] = False; ov[:, ys.stop:] = False
    arm1 = (b1 & ~b2 & core)[:, None, :].expand(v.shape).clone(); arm1[:, :ys.start] = False; arm1[:, ys.stop:] = False
    arm2 = (b2 & ~b1 & core)[:, None, :].expand(v.shape).clone(); arm2[:, :ys.start] = False; arm2[:, ys.stop:] = False
    d1 = torch.as_tensor(tr["dir1"][np.asarray(tr["band1"] & ~tr["band2"])][0], device=DEV).view(3, 1, 1, 1).expand(3, *v.shape)
    d2 = torch.as_tensor(tr["dir1"][np.asarray(tr["band2"] & ~tr["band1"])][0], device=DEV).view(3, 1, 1, 1).expand(3, *v.shape)
    res = {"case": CASE}; s2 = None
    for mode in (None, "image_cont", "shape"):
        cfg = DevConfig(engine="anatomix_devvit", engine_kwargs={}, smooth=3, direction="readout", fibre_sets=FibreSetConfig(presence="distance"),
                        vox_mm=grid.vox, completion=mode)
        o = apply_dev(cal, v, tissue, cfg=cfg, references=refs, features=F, region=grid.inner, device=DEV, stage2=s2); s2 = o["stage2"]
        L, D, V = o["activation"], o["dirs"], o["vecs"]
        r = {"n_sets": int(L.shape[0])}
        for k in range(L.shape[0]):
            e_to1, e_to2 = angle_deg(D[k], d1), angle_deg(D[k], d2)
            r[f"set {k}"] = {"activation overlap / arm1 / arm2": [round(float(L[k][m].mean()), 2) for m in (ov, arm1, arm2)],
                             "direction error in overlap to band 1 / band 2": [round(float(e_to1[ov].median()), 1), round(float(e_to2[ov].median()), 1)]}
            if mode is None:
                st_ = steering(L, D, k)
                r[f"set {k}"]["steering error in overlap to band 1 / band 2"] = [round(float(angle_deg(st_, d1)[ov].median()), 1),
                                                                                  round(float(angle_deg(st_, d2)[ov].median()), 1)]
        if V.shape[0] >= 2:
            e11, e12, e21, e22 = angle_deg(V[0], d1), angle_deg(V[0], d2), angle_deg(V[1], d1), angle_deg(V[1], d2)
            sw = (e12 + e21) < (e11 + e22); ea, eb = torch.where(sw, e12, e11), torch.where(sw, e21, e22)
            lo, hi = torch.minimum(ea, eb), torch.maximum(ea, eb)
            r["overlap: better vector error p25/50/75"] = [round(float(torch.quantile(lo[ov], q)), 1) for q in (0.25, 0.5, 0.75)]
            r["overlap: worse vector error p25/50/75"] = [round(float(torch.quantile(hi[ov], q)), 1) for q in (0.25, 0.5, 0.75)]
            r["overlap: second vector weight / first"] = round(float((o["weights"][1][ov] / o["weights"][0][ov].clamp_min(1e-9)).median()), 3)
        res[str(mode)] = r; log(mode, json.dumps(r))
    json.dump(res, open(OUT / f"diag_completion_{CASE.replace(' ', '_')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
