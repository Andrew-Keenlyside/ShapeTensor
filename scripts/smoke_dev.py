"""End-to-end smoke run of the dev pipeline (shapetensor.dev) with the real anatomix-dev-vit network on one phantom: does
every stage run, and what does it find? Not a validation (scripts/validate_dev.py is): one phantom, one seed.

    python scripts/smoke_dev.py [crossing|bundle] [DEVICE] [OUT_JSON]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from shapetensor.dev import DevConfig, population_field_dev
from shapetensor.phantoms import Grid, TextureParams, build
from shapetensor.validate import SETTINGS

NAME = sys.argv[1] if len(sys.argv) > 1 else "crossing"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/smoke_{NAME}_{DEV}.json")


def main():
    torch.manual_seed(0); t0 = time.time()
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    g = Grid()
    kw = {"cross_deg": 90.0} if NAME == "crossing" else {"angle": 60.0}
    img, tr, meta = build(NAME, 0, g, tex, texture=p["texture"], device=DEV, **kw)
    v = torch.as_tensor(img, device=DEV).float()
    cfg = DevConfig()
    cfg.engine_kwargs = {**cfg.engine_kwargs, "device": DEV, "batch": 2 if DEV == "cpu" else 4}
    timings = {}
    out = population_field_dev(v, None, cfg, device=DEV, timings=timings, log=lambda *a: print(*a, flush=True))
    truth = [torch.as_tensor(tr["dir1"][g.margin[0], g.margin[2]], device=DEV).float()]
    if NAME == "crossing":
        truth.append(torch.as_tensor(tr["dir2"][g.margin[0], g.margin[2]], device=DEV).float())
    core = (slice(None),) + g.inner
    ang = lambda f, t: torch.rad2deg(torch.arccos((f * t.view(3, 1, 1, 1)).sum(0).abs().clamp(max=1)))
    vec = out["vec"][core]; v1 = out["vecs"][0][core]
    res = {"phantom": NAME, "device": DEV, "n_populations": out["n_populations"], "timings_s": timings, "seconds": round(time.time() - t0, 1),
           "nulls": {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out["nulls"].items()}, "report": out["report"],
           "confidence": out["confidence"], "config": out["config"]}
    for i, t in enumerate(truth):
        res[f"merged_field_median_deg_to_dir{i + 1}"] = round(float(ang(vec, t).median()), 1)
        res[f"first_vector_median_deg_to_dir{i + 1}"] = round(float(ang(v1, t).median()), 1)
    if NAME == "crossing" and out["vecs"].shape[0] >= 2:
        v2 = out["vecs"][1][core]
        a = torch.maximum(torch.minimum(ang(v1, truth[0]), ang(v1, truth[1])), torch.minimum(ang(v2, truth[0]), ang(v2, truth[1])))
        res["both_vectors_within_15deg_share"] = round(float((a <= 15).float().mean()), 3)
    OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in res if k not in ("config", "report")}, indent=1, default=str))


if __name__ == "__main__":
    main()
