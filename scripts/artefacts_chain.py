"""Resilience of the chain to reconstruction artefacts: a bundle at 60° with streaks (along x: in the fibre plane; along y:
across it; any direction) and rings at 0 / 50 / 100 / 200 grey levels (the fibre darkening is 50). The chain's single
field and its population count against the excised structure tensor.

    python scripts/artefacts_chain.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shapetensor import st
from shapetensor.artefacts import add
from shapetensor.field import Orientations
from shapetensor.phantoms import Grid, build, direction
from shapetensor.validate import angle_deg, tex_for

DEV = "cuda"
KINDS = {"streaks along x": ("streaks", {"angle_deg": 0.0}), "streaks along y": ("streaks", {"angle_deg": 90.0}),
         "streaks, any": ("streaks", {"spread_deg": 90.0}), "rings": ("rings", {})}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/artefacts_chain")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tex, texture = tex_for("calibrated"); grid = Grid(core=(128, 96, 128))
    img0, tr, meta = build("bundle", 1, grid, tex=tex, texture=texture, device=DEV, angle=60.0)
    model = Orientations(device=DEV)
    core = torch.zeros(img0.shape, dtype=torch.bool, device=DEV); zs, ys, xs = grid.inner; core[zs, ys, xs] = True
    d1 = torch.as_tensor(direction(60.0), device=DEV).view(3, 1, 1, 1).expand(3, *img0.shape)
    res = {}
    for name, (kind, kw) in KINDS.items():
        res[name] = {}
        for amp in (0.0, 50.0, 100.0, 200.0):
            img = add(kind, img0, amp, device=DEV, **kw)[0] if amp > 0 else img0
            v = torch.as_tensor(img, device=DEV).float()
            fld = model.run(v, labels=False)
            d_st = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=True, order="zyx", device=DEV)
            e_sh = angle_deg(fld["vec"], d1)[core]; e_st = angle_deg(d_st, d1)[core]
            Nc = fld["N_cells"]; cs = Nc.shape
            inner = torch.zeros(cs, dtype=torch.bool, device=DEV); inner[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
            r = {"shape median deg": round(float(e_sh.median()), 1), "structure tensor median deg": round(float(e_st.median()), 1),
                 "shape within 15 deg": round(float((e_sh <= 15).float().mean()), 3), "structure tensor within 15 deg": round(float((e_st <= 15).float().mean()), 3),
                 "cells N=2 (an artefact population)": round(float((Nc[inner] == 2).float().mean()), 3), "cells N=0": round(float((Nc[inner] == 0).float().mean()), 3),
                 "sheet cells": round(float(fld["sheet_cells"][inner].float().mean()), 3)}
            res[name][f"amp {amp:g}"] = r
            print(f"[{time.time() - t0:5.0f}s] {name:16s} amp {amp:5.0f}: {json.dumps(r)}", flush=True)
            json.dump(res, open(out / "artefacts_chain.json", "w"), indent=1)
            del fld, d_st; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
