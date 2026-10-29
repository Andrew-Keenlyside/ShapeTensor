"""Stage 3 alone: each population's direction from the image (the released population tensor) against from the shape
features (the dev pipeline's shape tensor), with stage 2 replaced by the truth (each fascicle's mask as its layer).

On the bend phantom (three fascicles turning through 90° in grey-matter-level tissue; the tissue is extruded along y,
so a fascicle is a slab in y while its fibres are lines): per fascicle, the median 3D error to the truth tangent and
the median |y| component of the direction (a fibre direction has none; a slab's least-changing axes are the fibre
and y). Feature engines: encoder 2 (3³ box), dev-vit, dev-vit + sliceTCA; feature sigma 1 (as the pipeline) and 0.5.

    python scripts/diag_stage3.py [OUT_DIR]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor import st
from shapetensor.features import make_engine
from shapetensor.phantoms import BEND_GRID, TextureParams, build
from shapetensor.phantoms import Grid
from shapetensor.pipeline import calibrate
from shapetensor.tensors import feature_products, population_tensor, readout_products, shape_tensor
from shapetensor.validate import ANGLES, SETTINGS, angle_deg

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"; RHO, TRUNC = 10.0, 3.0
import os
READOUT_ONLY = os.environ.get("READOUT_ONLY") == "1"
SELF = os.environ.get("SELF") == "1"


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    img, tr, _ = build("bend", 0, BEND_GRID, tex, texture=p["texture"], device=DEV)
    v = torch.as_tensor(img, device=DEV).float(); Z, Y, X = v.shape
    fid = torch.as_tensor(tr["fid"], device=DEV) if "fid" in tr else None
    t1 = torch.as_tensor(tr["dir1"], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, Z, Y, X)
    inside = torch.isfinite(t1[0])
    sel = torch.zeros_like(inside); sel[BEND_GRID.inner] = True
    ids = sorted(int(k) for k in torch.unique(fid[torch.isfinite(torch.as_tensor(tr["dir1"][..., 0], device=DEV))]).tolist()) if fid is not None else [None]
    layers = []
    for k in ids:
        m2 = (fid == k) if fid is not None else torch.isfinite(torch.as_tensor(tr["dir1"][..., 0], device=DEV))
        layers.append(m2[:, None, :].expand(Z, Y, X).float())
    w_exc = st.excision_weights(v)
    res = {"fascicles": ids}

    def score(dirs, name):
        out = {}
        for i, (k, a) in enumerate(zip(ids, layers)):
            s = (a > 0) & sel
            e = angle_deg(dirs[i], t1)[s]; yc = dirs[i][1].abs()[s]
            out[f"fascicle {k}"] = {"median_deg": round(float(e.median()), 1), "p90_deg": round(float(torch.quantile(e, 0.9)), 1),
                                    "median_abs_y": round(float(yc.median()), 3)}
        res[name] = out; log(name, json.dumps(out))

    P = st.gradient_products(v, 1.0, TRUNC, device=DEV)
    score([population_tensor(v, a, 1.0, RHO, TRUNC, w_exc, 0.01, None, P)[1] for a in layers], "image structure tensor (released stage 3)")
    del P
    for eng, kw in (("anatomix_enc2", {}), ("anatomix_devvit", {}), ("devvit_slicetca", {"ranks": (2, 2, 2), "spacing": 3})):
        F = make_engine(eng, device=DEV, **kw)(v)
        if eng == "anatomix_enc2":                                                    # the readers' 3^3 box, as the pipeline averages encoder features
            F = torch.nn.functional.avg_pool3d(F.float()[None], 3, stride=1, padding=1)[0]
        for sig in ((1.0, 0.5) if not READOUT_ONLY else ()):
            Pf = feature_products(F, sig, TRUNC)
            score([shape_tensor(Pf, a, RHO, TRUNC, w_exc, 0.01)[1] for a in layers], f"shape tensor, {eng}, sigma {sig}")
            del Pf
        del F; torch.cuda.empty_cache()
        # the readout tensor: the readers calibrated as the validation calibrates them, their distributions on the bend
        cg = Grid(); smooth = 3 if eng != "devvit_slicetca" else 1
        bundles = [] if SELF else [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=p["texture"], angle=float(a_), device=DEV)[0], device=DEV).float()
                                   for k, a_ in enumerate(ANGLES)]
        if SELF:                                                                      # the block calibrates on itself, as run_volume does
            from importlib.util import module_from_spec, spec_from_file_location
            sp = spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py"))); PR = module_from_spec(sp); sp.loader.exec_module(PR)
            cal = calibrate(v, v > PR.otsu(img), sigma=1.0, rho=RHO, truncate=TRUNC, margin=BEND_GRID.margin[0], engine=eng, engine_kwargs=kw, smooth=smooth,
                            seed=0, device=DEV)
        else:
            cal = calibrate(bundles, sigma=1.0, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=eng, engine_kwargs=kw, smooth=smooth, seed=0, device=DEV)
        del bundles
        Pd = cal.readout.predict_dist(cal.dictionary.prep(cal.features(v)), prepped=True)
        R = readout_products(Pd, cal.readout.grid); del Pd
        score([shape_tensor(R, a, RHO, TRUNC, w_exc, 0.01)[1] for a in layers], f"readout tensor, {eng}, {'self-calibrated' if SELF else 'bundle-calibrated'} (readout fit {cal.readout_overlap:.2f})")
        del R, cal; torch.cuda.empty_cache()
    json.dump(res, open(OUT / (("diag_stage3_readout_self.json" if SELF else "diag_stage3_readout.json") if READOUT_ONLY else "diag_stage3.json"), "w"), indent=1)
    log("wrote", OUT / "diag_stage3.json")


if __name__ == "__main__":
    main()
