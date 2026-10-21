"""Why the fibre-set presence test finds no fibre windows with dev-vit features: the texture reference's window
prominence against the fibre windows', per engine, and what the texture's peak directions are.

For each engine (calibrated as validate_dev.py does: the twelve straight bundles at the calibrated texture):
  * the window prominence (max / mean of the gain-normalised orientation density) on fibre-free texture and on a bundle
    at 60°: quantiles;
  * where the texture windows peak: the angle of each window's top direction to the nearest coordinate axis
    (the dev-vit patch grid would put them on the axes), and the share within 10° of an axis vs the share a uniform
    direction would have;
  * how peaked the readout is per voxel (the mean top probability) and how far the readout's predictions stay alike
    (the correlation length of the per-voxel top direction along each axis);
  * the patch-grid residual of the features: the variance over position-in-patch (mod 8) of the texture's mean feature,
    over the total feature variance.

    python scripts/diag_presence.py [OUT_DIR]       ENGINES=name,name (default: anatomix_enc2, anatomix_devvit, devvit_slicetca)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.fibre_sets import References, prominence, window_densities
from shapetensor.phantoms import Grid, TextureParams, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES, SETTINGS

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
ENGINES = {"anatomix_enc2": ({}, 3), "anatomix_devvit": ({}, 3), "devvit_slicetca": ({"ranks": (2, 2, 2), "spacing": 3}, 1),
           "anatomix_devvit, shifts 0 only": ({"shifts": [(0, 0, 0)]}, 3),
           "anatomix_devvit, 4 shifts": ({"shifts": [(0, 0, 0), (4, 4, 4), (0, 4, 4), (4, 0, 0)]}, 3)}
if os.environ.get("ENGINES"):
    ENGINES = {k: v for k, v in ENGINES.items() if k in os.environ["ENGINES"].split(";")}


def q(x):
    x = x.flatten().float()
    return {k: round(float(torch.quantile(x, p)), 2) for k, p in (("min", 0.0), ("median", 0.5), ("p90", 0.9), ("max", 1.0))}


def corr_length(idx_dir, axis):
    """How many voxels along ``axis`` the per-voxel top direction stays the same (mean run length)."""
    a = idx_dir.movedim(axis, -1)
    same = (a[..., 1:] == a[..., :-1]).float().mean()
    return round(float(1.0 / max(1e-6, 1.0 - float(same))), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    cg = Grid()
    bundles = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=p["texture"], angle=float(a), device=DEV)[0], device=DEV).float()
               for k, a in enumerate(ANGLES)]
    refs = References(cg.core, "calibrated", DEV)
    b60 = torch.as_tensor(build("bundle", 7, cg, tex, texture=p["texture"], angle=60.0, device=DEV)[0], device=DEV).float()
    res = {}
    for name, (kw, smooth) in ENGINES.items():
        engine = name.split(",")[0]
        cal = calibrate(bundles, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=engine, engine_kwargs=kw, smooth=smooth,
                        seed=0, device=DEV)
        u = cal.readout.grid                                                                # (n_dirs, 3)
        r = {"readout_fit": round(float(cal.readout_overlap), 3), "n_dirs": int(u.shape[0])}
        for what, vol, region in (("texture", refs.texture, refs.texture_region), ("bundle 60", b60, cg.inner)):
            F = cal.features(vol); Fp = cal.dictionary.prep(F)
            D, _, _ = window_densities(cal, Fp, region, 32, 8)
            pr = prominence(D)
            top = u[D.argmax(0).flatten()]                                                 # (n_win, 3)
            ax = torch.rad2deg(torch.arccos(top.abs().max(1).values.clamp(max=1)))         # angle to the nearest axis
            Pv = cal.readout.predict_dist(Fp, prepped=True)[(slice(None),) + tuple(region)]
            topv = Pv.argmax(0)
            r[what] = {"prominence": q(pr), "peak_to_axis_deg": q(ax), "peaks_within_10deg_of_axis": round(float((ax <= 10).float().mean()), 3),
                       "voxel_top_probability_mean": round(float(Pv.max(0).values.mean()), 3),
                       "top_direction_run_length_zyx": [corr_length(topv, a) for a in range(3)]}
            if what == "texture":
                Fi = F[(slice(None),) + tuple(region)].float()
                pos = [torch.arange(n, device=DEV) % 8 for n in Fi.shape[1:]]
                cell = (pos[0][:, None, None] * 64 + pos[1][None, :, None] * 8 + pos[2][None, None, :]).flatten()
                Ff = Fi.reshape(Fi.shape[0], -1)
                means = torch.zeros(Fi.shape[0], 512, device=DEV).index_add_(1, cell, Ff) / torch.bincount(cell, minlength=512).clamp_min(1)[None]
                r[what]["patch_grid_variance_share"] = round(float(means.var(1).sum() / Ff.var(1).sum().clamp_min(1e-12)), 4)
            del F, Fp, D, Pv
        # a uniform direction's chance of lying within 10 deg of an axis, on this grid
        axg = torch.rad2deg(torch.arccos(u.abs().max(1).values.clamp(max=1)))
        r["grid_share_within_10deg_of_axis"] = round(float((axg <= 10).float().mean()), 3)
        res[name] = r; log(name, json.dumps(r))
        json.dump(res, open(OUT / "diag_presence.json", "w"), indent=1)
        del cal; torch.cuda.empty_cache()
    log("wrote", OUT / "diag_presence.json")


if __name__ == "__main__":
    main()
