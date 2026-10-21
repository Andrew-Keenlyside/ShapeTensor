"""How well can a *linear* model of the features find a direction? On single-population phantoms (bundle at 60°, the
straight fascicles), the integrated, whitened features at fibre voxels against each grid direction's signature:

  * NNLS deconvolution (the classical pathway's) -- the largest coefficient's direction;
  * a matched filter: the correlation with each direction's signature (centred on the mean signature) -- its peak;
  * the orientation signal-to-noise: the variance of the signatures across directions over the residual variance, in
    whitened units (the Fisher criterion of direction), and the effective rank of the signatures' variation.

Median angular error of the found direction to the truth, per engine (pooled calibration as the validation's).

    python scripts/diag_linear.py [OUT_DIR]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.classical import _design, nnls_fista, whitener
from shapetensor.filters import gaussian
from shapetensor.phantoms import BEND_GRID, Grid, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def main():
    t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    tex, texture = P.tex_for("calibrated"); cg = Grid()
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES)]
    straight, trs, _ = build("bend", 0, tex=tex, texture=texture, device=DEV, straight=True)
    b60, trb, _ = build("bundle", 5, cg, tex, texture=texture, device=DEV, angle=60.0)
    vs = torch.as_tensor(straight, device=DEV).float()
    res = {}
    for eng in ("anatomix_enc2", "anatomix_devvit"):
        cal = calibrate(vols + [vs], [None] * len(vols) + [vs > P.otsu(straight)], sigma=1.0, rho=10.0, truncate=3.0, margin=cg.margin[0], engine=eng,
                        engine_kwargs={}, smooth=3, seed=0, device=DEV)
        D = cal.dictionary; grid = D.grid; Wh = whitener(D)
        S = D.D.float(); mu = S.mean(0)
        Sw = (S - mu) @ Wh.T                                                                             # centred, whitened signatures (nd, C)
        U, sv, _ = torch.linalg.svd(Sw, full_matrices=False)
        fisher = float((Sw ** 2).sum(1).mean())                                                         # whitened residual variance is 1 per channel
        eff_rank = float((sv.sum() ** 2) / (sv ** 2).sum())
        r = {"orientation SNR (mean ||whitened signature deviation||²)": round(fisher, 3), "effective rank of the signatures' variation": round(eff_rank, 2)}
        for name, img, tr, grd in (("bundle 60", b60, trb, cg), ("straight", straight, trs, BEND_GRID)):
            v = torch.as_tensor(img, device=DEV).float()
            Fp = D.prep(cal.features(v)).float(); C = Fp.shape[0]
            Xf = (Fp - D.mean.view(C, 1, 1, 1)) / D.std.view(C, 1, 1, 1)
            Xf = torch.stack([gaussian(Xf[k], 10.0, truncate=3.0) for k in range(C)])
            t1 = torch.as_tensor(tr["dir1"], device=DEV).permute(2, 0, 1)[:, :, None, :].expand(3, *v.shape)
            sel = torch.isfinite(t1[0]); inner = torch.zeros_like(sel); inner[grd.inner] = True; sel &= inner
            idx = sel.nonzero()[torch.randperm(int(sel.sum()), device=DEV)[:4000]]
            X = Xf[:, idx[:, 0], idx[:, 1], idx[:, 2]].T                                                  # (n, C)
            T = t1[:, idx[:, 0], idx[:, 1], idx[:, 2]].T                                                  # (n, 3)
            Xw = (X - mu) @ Wh.T
            mf = (Xw @ Sw.T)                                                                              # (n, nd) matched filter
            d_mf = grid[mf.argmax(1)]
            A = Wh @ _design(D); a = nnls_fista(A, X @ Wh.T, iters=400)[:, :grid.shape[0]]
            d_nn = grid[a.argmax(1)]
            err = lambda d: torch.rad2deg(torch.arccos((d * T).sum(1).abs().clamp(max=1)))
            r[name] = {"matched filter median error": round(float(err(d_mf).median()), 1), "NNLS largest coefficient median error": round(float(err(d_nn).median()), 1),
                       "NNLS coefficients above 10% of the largest (median count)": float(((a >= 0.1 * a.max(1, keepdim=True).values).sum(1)).float().median())}
            del Fp, Xf
        res[eng] = r; log(eng, json.dumps(r))
        json.dump(res, open(OUT / "diag_linear.json", "w"), indent=1)
        del cal; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
