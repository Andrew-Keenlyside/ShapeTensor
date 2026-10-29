"""Can the features tell fibre-free texture from fibres? A linear probe (logistic regression on the readers' prepared,
standardised features) trained on texture (seed 7) against the twelve calibration bundles, tested on held-out texture
(seed 8, grey-matter level and white-matter level) against a bundle at 60° and the bend phantom's fascicles; AUC per
voxel and per 32³ window (the window's mean probe output).

    python scripts/diag_probe.py [OUT_DIR]       ENGINES=name;name
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor.fibre_sets import References
from shapetensor.phantoms import BEND_GRID, Grid, TextureParams, build, finish
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES, SETTINGS

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/diag")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
ENGINES = {"anatomix_enc2": ({}, 3), "devvit_slicetca": ({"ranks": (2, 2, 2), "spacing": 3}, 1)}
if os.environ.get("ENGINES"):
    ENGINES = {k: v for k, v in ENGINES.items() if k in os.environ["ENGINES"].split(";")}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def texture(grid, level, seed, tex, texture_amp):
    t_ = TextureParams(**{**tex.__dict__, "amp": tex.amp * texture_amp})
    Z, X = grid.shape[0], grid.shape[2]
    return torch.as_tensor(finish(grid, t_, np.full((Z, X), level, np.float32), np.zeros((Z, X), np.float32), torch.zeros(grid.shape, device=DEV),
                                  np.random.default_rng([seed, 4242])), device=DEV).float()


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    cg = Grid()
    bundles = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=p["texture"], angle=float(a), device=DEV)[0], device=DEV).float()
               for k, a in enumerate(ANGLES)]
    t_train = texture(cg, tex.levels["gm"], 7, tex, p["texture"])
    t_gm, t_wm = texture(cg, tex.levels["gm"], 8, tex, p["texture"]), texture(cg, tex.levels["wm"], 9, tex, p["texture"])
    b60 = torch.as_tensor(build("bundle", 7, cg, tex, texture=p["texture"], angle=60.0, device=DEV)[0], device=DEV).float()
    bimg, btr, _ = build("bend", 0, BEND_GRID, tex, texture=p["texture"], device=DEV); bend = torch.as_tensor(bimg, device=DEV).float()
    fasc = torch.as_tensor(np.isfinite(btr["dir1"][..., 0]), device=DEV)[:, None, :].expand(bend.shape)
    res = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    _r = References(cg.core, "calibrated", DEV); refs_tex, refs_reg = _r.texture, _r.texture_region
    for name, (kw, smooth) in ENGINES.items():
        cal = calibrate(bundles, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=name, engine_kwargs=kw, smooth=smooth, seed=0, device=DEV)
        prep = lambda v: cal.dictionary.prep(cal.features(v))
        def samples(v, region, n):
            f = prep(v)[(slice(None),) + tuple(region)].float(); f = f.reshape(f.shape[0], -1).T
            return f[torch.randint(0, f.shape[0], (n,), generator=g).to(f.device)]
        Xp = torch.cat([samples(b, cg.inner, 4000) for b in bundles]); Xn = samples(t_train, cg.inner, 48000)
        X = torch.cat([Xp, Xn]); y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(DEV)
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6); X = (X - mu) / sd
        w = torch.zeros(X.shape[1], device=DEV, requires_grad=True); b = torch.zeros(1, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * (w ** 2).sum(); l.backward(); return l
        opt.step(closure)
        def score(v, region):
            f = prep(v)[(slice(None),) + tuple(region)].float(); C = f.shape[0]
            s = torch.sigmoid(((f.reshape(C, -1).T - mu) / sd) @ w + b).reshape(f.shape[1:])
            return s.detach()
        def wins(s, every=8, w_=32):
            out = []
            for z in range(0, s.shape[0] - w_ + 1, every):
                for yy in range(0, max(1, s.shape[1] - w_ + 1), every):
                    for x in range(0, s.shape[2] - w_ + 1, every):
                        out.append(float(s[z:z + w_, yy:yy + w_, x:x + w_].mean()))
            return out
        s_gm, s_wm, s_b = score(t_gm, cg.inner), score(t_wm, cg.inner), score(b60, cg.inner)
        s_bend = score(bend, BEND_GRID.inner); f_in = fasc[BEND_GRID.inner]
        extra = {}
        for nm, kind, kw_, gr in (("crossing 90", "crossing", {"cross_deg": 90.0}, cg), ("crossing 60", "crossing", {"cross_deg": 60.0}, cg),
                                  ("crossing 40", "crossing", {"cross_deg": 40.0}, cg), ("band crossing 90", "band_crossing", {"cross_deg": 90.0}, BEND_GRID)):
            im = torch.as_tensor(build(kind, 0, gr, tex, texture=p["texture"], device=DEV, **kw_)[0], device=DEV).float()
            sc_ = score(im, gr.inner)
            extra[nm] = {"voxel mean": round(float(sc_.mean()), 3), "window min": round(float(np.min(wins(sc_))), 3), "window median": round(float(np.median(wins(sc_))), 3)}
            del im, sc_
        tw = wins(score(refs_tex, refs_reg))
        extra["texture reference (presence null grid)"] = {"window max": round(float(np.max(tw)), 3), "window median": round(float(np.median(tw)), 3)}
        sub = lambda s: s.flatten()[torch.randperm(s.numel(), generator=g)[:50000].to(s.device)].cpu().numpy()
        out = {"voxel AUC, bundle 60 vs texture gm": round(auc(sub(s_b), sub(s_gm)), 3), "voxel AUC, bundle 60 vs texture wm": round(auc(sub(s_b), sub(s_wm)), 3),
               "voxel AUC, bend fascicle vs bend rest": round(auc(sub(s_bend[f_in]), sub(s_bend[~f_in])), 3),
               "window AUC, bundle 60 vs texture gm": round(auc(wins(s_b), wins(s_gm)), 3),
               "window AUC, bundle 60 vs texture wm": round(auc(wins(s_b), wins(s_wm)), 3),
               "window means": {"texture gm": round(float(np.mean(wins(s_gm))), 3), "texture wm": round(float(np.mean(wins(s_wm))), 3),
                                "bundle 60": round(float(np.mean(wins(s_b))), 3)},
               "window max, texture gm / wm": [round(float(np.max(wins(s_gm))), 3), round(float(np.max(wins(s_wm))), 3)],
               "window min, bundle 60": round(float(np.min(wins(s_b))), 3), **extra}
        res[name] = out; log(name, json.dumps(out))
        json.dump(res, open(OUT / os.environ.get("PROBE_OUT", "diag_probe.json"), "w"), indent=1)
        del cal; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
