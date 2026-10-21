"""Validation of the dev pipeline against the released one, one change at a time, on the phantoms the fibre-set
prototypes were scored on (the scores are the prototype harness's, so the numbers compare with the results log).

Methods (features | stage 2 | stage 3):
    released           anatomix encoder 2, 3^3 box | block populations | image structure tensor
    released+devvit    dev-vit, 3^3 box            | block populations | image structure tensor   (the plan's test 2)
    fibre sets         anatomix encoder 2, 3^3 box | fibre sets        | image structure tensor   (stage 2 alone)
    dev (intensity)    dev-vit + sliceTCA          | fibre sets        | image structure tensor   (all but stage 3)
    dev                dev-vit + sliceTCA          | fibre sets        | shape tensor
Every method is calibrated on the validation's straight bundles of each texture setting (as the prototypes were).
Also reported per method: how linear the mean feature pattern is in the orientation dyad d d^T (the plan's test 1:
the share of the signature dictionary's variance a linear map of d d^T explains).

    QUICK=1 python scripts/validate_dev.py [OUT_DIR]      (QUICK: seed 0 only)
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

from shapetensor.dev import DevConfig, apply_dev
from shapetensor.fibre_sets import FibreSetConfig, References
from shapetensor.phantoms import BEND_GRID, Grid, build
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import ANGLES

torch.use_deterministic_algorithms(True, warn_only=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/validate")
DEV = "cuda"; VOX = 0.02; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
SEEDS = (0,) if os.environ.get("QUICK") == "1" else (0, 1, 2)
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)          # scoring, textures, Otsu: the prototypes' own

METHODS = {
    "released":        {"engine": "anatomix_enc2", "engine_kwargs": {}, "smooth": 3, "stage2": "block", "direction": "intensity"},
    "released+devvit": {"engine": "anatomix_devvit", "engine_kwargs": {}, "smooth": 3, "stage2": "block", "direction": "intensity"},
    "fibre sets":      {"engine": "anatomix_enc2", "engine_kwargs": {}, "smooth": 3, "stage2": "sets", "direction": "intensity"},
    "dev (intensity)": {"engine": "devvit_slicetca", "engine_kwargs": {"ranks": (2, 2, 2), "spacing": 3}, "smooth": 1, "stage2": "sets", "direction": "intensity"},
    "dev":             {"engine": "devvit_slicetca", "engine_kwargs": {"ranks": (2, 2, 2), "spacing": 3}, "smooth": 1, "stage2": "sets", "direction": "shape"},
}
CASES = (("bend", {}), ("straight", {"straight": True}), ("band crossing 90", {"cross_deg": 90.0}), ("band crossing 60", {"cross_deg": 60.0}),
         ("crossing 90", {"cross_deg": 90.0}), ("crossing 60", {"cross_deg": 60.0}))


def dyad_r2(cal) -> float:
    """Share of the signature dictionary's variance (directions x channels) explained by a linear map of d d^T."""
    D = cal.dictionary.D.double(); u = cal.dictionary.grid.double()
    X = torch.stack([u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2, u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2], torch.ones_like(u[:, 0])], 1)
    W = torch.linalg.lstsq(X, D).solution
    Dc = D - D.mean(0)
    return float(1 - ((D - X @ W) ** 2).sum() / (Dc ** 2).sum().clamp_min(1e-12))


def run(method, m, cal, refs, v, tissue, F, grid):
    if m["stage2"] == "block":
        o = apply(cal, v, tissue, features=F, device=DEV)
        o["activation"] = o["activation"] / max(cal.tau, 1e-12)
        return o
    cfg = DevConfig(engine=m["engine"], engine_kwargs=dict(m["engine_kwargs"]), sigma=SIGMA, rho=RHO, truncate=TRUNC, smooth=m["smooth"],
                    direction=m["direction"], fibre_sets=FibreSetConfig(), vox_mm=VOX)
    return apply_dev(cal, v, tissue, cfg=cfg, references=refs, features=F, region=grid.inner, device=DEV)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    results = {"settings": {"sigma": SIGMA, "rho": RHO, "truncate": TRUNC, "seeds": SEEDS}, "dyad_r2": {}, "runs": []}
    for setting in ("calibrated", "coherent"):
        tex, texture = P.tex_for(setting); cg = Grid()
        bundles = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float()
                   for k, a in enumerate(ANGLES)]
        for method, m in METHODS.items():
            cal = calibrate(bundles, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=m["engine"],
                            engine_kwargs=m["engine_kwargs"], smooth=m["smooth"], seed=0, device=DEV)
            results["dyad_r2"][f"{method} ({setting})"] = round(dyad_r2(cal), 4)
            log(f"{setting} | {method}: calibrated, readout fit {cal.readout_overlap:.2f}, dyad R2 {results['dyad_r2'][f'{method} ({setting})']}")
            refs = {}
            for case, kw in CASES:
                name = "bend" if case in ("bend", "straight") else ("band_crossing" if case.startswith("band") else "crossing")
                grid = BEND_GRID if name in ("bend", "band_crossing") else Grid()
                if m["stage2"] == "sets" and id(grid) not in refs:
                    refs[id(grid)] = References(tuple(s.stop - s.start for s in grid.inner), setting, DEV)
                for seed in SEEDS:
                    img, tr, meta = build(name, seed, tex=tex, texture=texture, device=DEV, **kw)
                    v = torch.as_tensor(img, device=DEV).float()
                    tissue = (v > P.otsu(img)) if name in ("bend", "band_crossing") else torch.ones_like(v, dtype=torch.bool)
                    t1 = time.time()
                    F = cal.features(v)
                    o = run(method, m, cal, refs.get(id(grid)), v, tissue, F, grid)
                    sc = (P.score_bend(o, tr, grid) if name == "bend" else P.score_band(o, tr, grid) if name == "band_crossing"
                          else P.score_crossing(o, v, tr, grid, setting))
                    row = {"setting": setting, "method": method, "case": case, "seed": seed, "seconds": round(time.time() - t1, 1),
                           "n_populations": int(o.get("n_populations", o["activation"].shape[0])), "scores": sc}
                    if "report" in o:                                                   # fibre sets: presence and second-population diagnostics
                        row["stage2"] = {k: o["report"].get(k) for k in ("windows", "windows_with_fibres", "windows_with_2plus", "sets_after_reunmix",
                                                                           "prominence_null", "window_prominence_max", "window_prominence_median", "share_null")}
                    results["runs"].append(row); log(f"  {case} seed {seed}: {json.dumps(sc, default=str)[:300]}")
                    json.dump(results, open(OUT / "validate_dev.json", "w"), indent=1, default=str)
                    del F, o, v; torch.cuda.empty_cache()
    log(f"wrote {OUT / 'validate_dev.json'}")


if __name__ == "__main__":
    main()
