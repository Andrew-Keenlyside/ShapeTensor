"""Presence without positives: how far a window's features lie from fibre-free texture (a Gaussian fitted to synthetic
texture, the Mahalanobis distance per voxel, averaged per 32³ window), against the linear probe's needs for fibre
examples. Is a crossing (which the probe, fitted on single-direction fibres, partly takes for texture) far from texture?

Features: dev-vit with the readers' 3³ box (as the dev pipeline reads them). Texture model: iso_textures (grey- and
white-matter level); held out: the References presence texture. Reported per kind: window distances (quantiles), the
held-out texture's maximum (the null) and the share of windows above it.

    python scripts/diag_oneclass.py [OUT_DIR]       ENGINE=anatomix_devvit (default) | devvit_slicetca | anatomix_enc2
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor.features import make_engine
from shapetensor.fibre_sets import References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, TextureParams, build
from shapetensor.validate import SETTINGS

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"; ENGINE = os.environ.get("ENGINE", "anatomix_devvit")
KW = {"devvit_slicetca": {"ranks": (2, 2, 2), "spacing": 3}}.get(ENGINE, {})
SMOOTH = 1 if ENGINE == "devvit_slicetca" else 3


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    eng = make_engine(ENGINE, device=DEV, **KW)

    def feats(v):
        F = eng(v).float()
        return Fn.avg_pool3d(F[None], SMOOTH, stride=1, padding=SMOOTH // 2)[0] if SMOOTH > 1 else F

    tr = [feats(t) for t in iso_textures("calibrated", DEV)]
    X = torch.cat([f[:, 32:-32, 32:-32, 32:-32].reshape(f.shape[0], -1).T for f in tr])
    X = X[torch.randperm(len(X), device=X.device)[:400_000]]
    mu = X.mean(0); C = torch.cov(X.T); C = C + 1e-3 * torch.trace(C) / C.shape[0] * torch.eye(C.shape[0], device=DEV)
    Ci = torch.linalg.inv(C); del X, tr

    def dist(v, region):
        f = feats(v)[(slice(None),) + tuple(region)]; c = f.shape[0]
        d = f.reshape(c, -1).T - mu
        m = ((d @ Ci) * d).sum(1).clamp_min(0).sqrt().reshape(f.shape[1:])
        w = tuple(min(32, n) for n in m.shape)
        return Fn.avg_pool3d(m[None, None], w, stride=8)[0, 0].flatten().cpu().numpy()

    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]); cg = Grid()
    refs = References(cg.core, "calibrated", DEV)
    kinds = {"texture (held out)": dist(refs.texture, refs.texture_region)}
    for nm, kind, kw, g in (("bundle 60", "bundle", {"angle": 60.0}, cg), ("crossing 90", "crossing", {"cross_deg": 90.0}, cg),
                            ("crossing 60", "crossing", {"cross_deg": 60.0}, cg), ("crossing 40", "crossing", {"cross_deg": 40.0}, cg),
                            ("band crossing 90", "band_crossing", {"cross_deg": 90.0}, BEND_GRID), ("bend", "bend", {}, BEND_GRID)):
        kinds[nm] = dist(torch.as_tensor(build(kind, 0, g, tex, texture=p["texture"], device=DEV, **kw)[0], device=DEV).float(), g.inner)
    null = float(kinds["texture (held out)"].max())
    q = lambda x: {k: round(float(np.quantile(x, pq)), 2) for k, pq in (("min", 0), ("median", 0.5), ("max", 1))}
    res = {"engine": ENGINE, "null (held-out texture window max)": round(null, 2),
           "kinds": {k: {**q(x), "share above null": round(float((x > null).mean()), 3)} for k, x in kinds.items()}}
    log(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f"diag_oneclass_{ENGINE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
