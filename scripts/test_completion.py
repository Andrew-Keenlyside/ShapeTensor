"""Population completion across crossings (``shapetensor.completion``): each fibre set carried along its own direction
from its arms on both sides, in image space or in shape space, against no completion.

One pooled calibration (dev-vit, 3³ box, fibre-free texture for presence: the 12 bundles + every case, label-free),
stage 2 once per case, then stage 3 four times: no completion, ``image``, ``image_cont``, ``shape``. Cases: the bend
and its straight control (must not change), the band crossings at 90 and 60° (overlap pair error, both within 15°;
each band whole), the whole-block crossings at 90 and 60°, and T-junctions at 90 and 60° (the second band stops at the
first's centre line: its set must not be carried on past the junction -- ``false extension``: its mean activation in
the strip it would have continued along, beyond the first band, over its mean on its own band).

    python scripts/test_completion.py [OUT_DIR]       HALF=16 (voxels per side), CASES="a;b"
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
from shapetensor.fibre_sets import FibreSetConfig, References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES

torch.use_deterministic_algorithms(True, warn_only=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/completion")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0; HALF = int(os.environ.get("HALF", "24")); STEER = os.environ.get("STEER", "1") == "1"; REDUCE = os.environ.get("REDUCE", "max")
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
CASES = (("bend", "bend", {}), ("straight", "bend", {"straight": True}),
         ("band crossing 90", "band_crossing", {"cross_deg": 90.0}), ("band crossing 60", "band_crossing", {"cross_deg": 60.0}),
         ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}), ("T-junction 60", "band_crossing", {"cross_deg": 60.0, "t_junction": True}),
         ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}))
if os.environ.get("CASES"):
    CASES = tuple(c for c in CASES if c[0] in os.environ["CASES"].split(";"))
MODES = (None, "image", "image_cont", "shape")


def false_extension(out, tr, grid):
    """The second band's set beyond a T-junction: its mean activation in the continuation strip (outside both bands) over
    its mean on its own band (outside the first), and the same for the strongest other set (any set carried there)."""
    zs, ys, xs = grid.inner
    core = np.zeros(tr["band1"].shape, bool); core[zs, xs] = True
    b1, b2 = tr["band1"] & core, tr["band2"] & core
    beyond = tr["corridor2"] & core & ~tr["band1"] & ~tr["band2"]
    A = out["activation"].float()[:, :, ys, :].mean(2).cpu().numpy()                     # (N, Z, X)
    if A.shape[0] == 0 or not beyond.any():
        return {"false extension": None}
    own = b2 & ~b1
    k2 = int(A[:, own].mean(1).argmax())
    ratio = float(A[k2][beyond].mean() / max(A[k2][own].mean(), 1e-9))
    return {"false extension": round(ratio, 3), "beyond activation (max over sets)": round(float(A[:, beyond].mean(1).max()), 3),
            "own band activation": round(float(A[k2][own].mean()), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    tex, texture = P.tex_for("calibrated"); cg = Grid()
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES)]
    tis = [None] * len(vols); imgs = {}
    for nm, kind, kw in CASES:
        img, tr, _ = build(kind, 0, tex=tex, texture=texture, device=DEV, **kw); imgs[nm] = (img, tr)
        v = torch.as_tensor(img, device=DEV).float(); vols.append(v)
        tis.append((v > P.otsu(img)) if kind in ("bend", "band_crossing") else None)
    cal = calibrate(vols, tis, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine="anatomix_devvit", engine_kwargs={}, smooth=3, seed=0,
                    device=DEV, iso=iso_textures("calibrated", DEV)); del vols, tis
    log(f"calibrated on the pool: readout fit {cal.readout_overlap:.2f}")
    refs, res = {}, {"settings": {"half": HALF, "steer": STEER, "reduce": REDUCE, "engine": "anatomix_devvit", "presence": "distance", "direction": "readout"}, "runs": []}
    for nm, kind, kw in CASES:
        img, tr = imgs[nm]; grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        if id(grid) not in refs:
            refs[id(grid)] = References(tuple(s.stop - s.start for s in grid.inner), "calibrated", DEV)
        v = torch.as_tensor(img, device=DEV).float()
        tissue = (v > P.otsu(img)) if kind in ("bend", "band_crossing") else torch.ones_like(v, dtype=torch.bool)
        F = cal.features(v); s2 = None
        for mode in MODES:
            cfg = DevConfig(engine="anatomix_devvit", engine_kwargs={}, sigma=SIGMA, rho=RHO, truncate=TRUNC, smooth=3, direction="readout",
                            fibre_sets=FibreSetConfig(presence="distance"), vox_mm=grid.vox, completion=mode, completion_half=HALF,
                            completion_steer=STEER, completion_reduce=REDUCE)
            t1 = time.time()
            o = apply_dev(cal, v, tissue, cfg=cfg, references=refs[id(grid)], features=F, region=grid.inner, device=DEV, stage2=s2)
            s2 = o["stage2"]
            if kind == "bend":
                sc = P.score_bend(o, tr, grid)[0]
            elif kind == "band_crossing":
                sc = P.score_band(o, tr, grid)[0]
                if kw.get("t_junction"):
                    sc.update(false_extension(o, tr, grid))
            else:
                sc = P.score_crossing(o, v, tr, grid, "calibrated")
            row = {"case": nm, "completion": mode or "none", "seconds": round(time.time() - t1, 1), "n_sets": o["n_populations"], "scores": sc,
                   "completion_report": o["report"].get("completion")}
            res["runs"].append(row)
            log(f"{nm:17s} {str(mode):10s} {json.dumps(sc, default=str)[:420]}")
            json.dump(res, open(OUT / f"completion_half{HALF}{'_steer' if STEER else ''}_{REDUCE}.json", "w"), indent=1, default=str)
            del o; torch.cuda.empty_cache()
        del F, v
    log("wrote", OUT / f"completion_half{HALF}{'_steer' if STEER else ''}_{REDUCE}.json")


if __name__ == "__main__":
    main()
