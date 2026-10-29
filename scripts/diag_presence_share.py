"""A presence test that does not read the readout's peak sharpness: per window, the share of tissue where the window's
largest population's activation exceeds the activation null (the released stage 2 run on the window, as fibre sets
already run it), on fibre-free texture against fibre windows.

For each engine (calibrated as validate_dev.py: the twelve straight bundles at the calibrated texture), every
``every``-th 32³ window (stride 8) of: the texture reference; a bundle at 60°; the bend phantom, its windows labelled
by the truth as holding a fascicle (> 20% of the window inside a fascicle) or not (< 2%). Reported: the top share's
quantiles per kind, the texture's maximum (the would-be null) and the share of fibre windows above it; the same for
the prominence, for comparison.

    python scripts/diag_presence_share.py [OUT_DIR]       ENGINES=name;name
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.fibre_sets import References, _window_grid, _window_stage2, prominence, window_densities
from shapetensor.phantoms import BEND_GRID, Grid, TextureParams, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES, SETTINGS

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0; EVERY = 2
ENGINES = {"anatomix_enc2": ({}, 3), "devvit_slicetca": ({"ranks": (2, 2, 2), "spacing": 3}, 1), "anatomix_devvit": ({}, 3)}
if os.environ.get("ENGINES"):
    ENGINES = {k: v for k, v in ENGINES.items() if k in os.environ["ENGINES"].split(";")}


def windows(cal, v, tissue, region, every):
    F = cal.features(v); Fp = cal.dictionary.prep(F)
    D, starts, w = window_densities(cal, Fp, region, 32, 8); prom = prominence(D); del D
    reach = int(cal.truncate * (cal.sigma + cal.rho) + 0.5)
    rows = []
    for iz in range(0, prom.shape[0], every):
        for iy in range(0, prom.shape[1], every):
            for ix in range(0, prom.shape[2], every):
                core = tuple(slice(s[k], s[k] + w[a]) for a, (s, k) in enumerate(zip(starts, (iz, iy, ix))))
                if float(tissue[core].float().mean()) < 0.01:
                    continue
                r = _window_stage2(cal, v, tissue, F, core, reach, 3, 25.0, DEV)
                top = float(r[2][0]) if r is not None and len(r[2]) else 0.0
                amax = float(r[1][0].max()) if r is not None and len(r[2]) else 0.0
                rows.append({"core": [[c.start, c.stop] for c in core], "share": top, "prominence": float(prom[iz, iy, ix]), "a_max": amax})
    del F, Fp
    return rows


def q(x):
    x = np.asarray(x, float)
    return {k: round(float(np.quantile(x, p)), 3) for k, p in (("min", 0), ("p10", 0.1), ("median", 0.5), ("p90", 0.9), ("max", 1))} if len(x) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    cg = Grid()
    bundles = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=p["texture"], angle=float(a), device=DEV)[0], device=DEV).float()
               for k, a in enumerate(ANGLES)]
    refs = References(cg.core, "calibrated", DEV)
    b60 = torch.as_tensor(build("bundle", 7, cg, tex, texture=p["texture"], angle=60.0, device=DEV)[0], device=DEV).float()
    bimg, btr, _ = build("bend", 0, BEND_GRID, tex, texture=p["texture"], device=DEV)
    bend = torch.as_tensor(bimg, device=DEV).float()
    th = float(np.argmax(np.cumsum(np.bincount(bimg.reshape(-1), minlength=256)) > 0)) if False else None
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py"))); P = module_from_spec(spec); spec.loader.exec_module(P)
    btis = bend > P.otsu(bimg)
    fasc = torch.as_tensor(np.isfinite(btr["dir1"][..., 0]), device=DEV)[:, None, :].expand(bend.shape)   # inside a fascicle
    res = {}
    for name, (kw, smooth) in ENGINES.items():
        cal = calibrate(bundles, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=name, engine_kwargs=kw, smooth=smooth, seed=0, device=DEV)
        tex_rows = windows(cal, refs.texture, torch.ones_like(refs.texture, dtype=torch.bool), refs.texture_region, EVERY)
        b_rows = windows(cal, b60, torch.ones_like(b60, dtype=torch.bool), cg.inner, EVERY)
        bend_rows = windows(cal, bend, btis, BEND_GRID.inner, EVERY)
        fib, non = [], []
        for r in bend_rows:
            c = tuple(slice(a, b) for a, b in r["core"]); f = float(fasc[c].float().mean())
            (fib if f > 0.2 else non if f < 0.02 else []).append(r)
        out = {"readout_fit": round(float(cal.readout_overlap), 3), "tau": round(float(cal.tau), 4)}
        for stat in ("share", "prominence"):
            null = max(r[stat] for r in tex_rows)
            out[stat] = {"texture": q([r[stat] for r in tex_rows]), "bundle 60": q([r[stat] for r in b_rows]),
                         "bend, fascicle windows": q([r[stat] for r in fib]), "bend, no-fascicle windows": q([r[stat] for r in non]),
                         "texture max (null)": round(null, 3),
                         "bundle windows above null": round(float(np.mean([r[stat] > null for r in b_rows])), 3),
                         "bend fascicle windows above null": round(float(np.mean([r[stat] > null for r in fib])), 3) if fib else None,
                         "bend no-fascicle windows above null": round(float(np.mean([r[stat] > null for r in non])), 3) if non else None}
        out["n_windows"] = {"texture": len(tex_rows), "bundle": len(b_rows), "bend fascicle": len(fib), "bend no fascicle": len(non)}
        res[name] = out; log(name, json.dumps(out))
        json.dump(res, open(OUT / "diag_presence_share.json", "w"), indent=1)
        del cal; torch.cuda.empty_cache()
    log("wrote", OUT / "diag_presence_share.json")


if __name__ == "__main__":
    main()
