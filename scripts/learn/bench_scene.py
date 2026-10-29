"""Timing benchmarks: the layer generator vs the object-scene generator (GPU idle, one process).

    python scripts/learn/bench_scene.py [--out FILE.json]

* old: ``generator.draw`` + ``targets.target`` per 64³ training pair;
* new: ``scene.draw_scene`` + ``target`` per 64³ pair -- the first crop of a scene (renders the 128³ scene) and the
  following crops of the same scene (cached), and the amortised cost per pair at 8 crops per scene;
* scene render alone at 96³ / 128³ / 192³ / 256³ (median of several scenes), and per family at 128³;
* the acquisition model with GPU-drawn scalars (old default) vs CPU-drawn scalars (``fast=True``) on the same sample.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from shapetensor.learn.generator import draw, nuisance, sample
from shapetensor.learn.scene import Scene, _CACHE, draw_scene
from shapetensor.learn.targets import target

DEV = "cuda"


def timed(fn, *a, **k):
    torch.cuda.synchronize(); t = time.perf_counter(); out = fn(*a, **k); torch.cuda.synchronize()
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_generator/bench.json")
    a = ap.parse_args(); res = {}
    draw((64,) * 3, seed=1, device=DEV, strength=0.3); draw_scene((64,) * 3, seed=1, device=DEV, strength=0.3)    # warm up
    # old generator
    t_old = []
    for i in range(12):
        t, (s, d) = timed(draw, (64,) * 3, seed=100 + i, device=DEV, strength=0.3)
        t2, _ = timed(target, s, d); t_old.append(t + t2)
    res["old_pair_s"] = float(np.median(t_old))
    # new generator: first crop (scene render) vs cached crops
    first, cached = [], []
    for sc in range(6):
        _CACHE.clear()
        for c in range(8):
            t, (s, d) = timed(draw_scene, (64,) * 3, seed=8 * (500 + sc) + c, device=DEV, strength=0.3)
            t2, _ = timed(target, s, d)
            (first if c == 0 else cached).append(t + t2)
    res["new_first_crop_s"] = float(np.median(first)); res["new_cached_crop_s"] = float(np.median(cached))
    res["new_amortised_pair_s_8crops"] = float((np.median(first) + 7 * np.median(cached)) / 8)
    # scene render alone by canvas size
    for n in (96, 128, 192, 256):
        ts = []
        for sd in range(5):
            t, _ = timed(lambda: Scene((n,) * 3, 900 + sd, DEV).build(None, (2, 5), 0.3).render())
            ts.append(t)
        res[f"render_{n}_s"] = float(np.median(ts)); print(n, res[f"render_{n}_s"], flush=True)
    # per family at 128³
    fam_t = {}
    for name in Scene.FAMILIES:
        ts = []
        for sd in range(4):
            fam = {k: (1.0 if k == name else 0.0) for k in Scene.FAMILIES}
            t, sc_ = timed(lambda: Scene((128,) * 3, 950 + sd, DEV, 5.0).build(fam, (1, 1), 1.0 if name == "radial" else 0.0).render())
            ts.append(t)
        fam_t[name] = float(np.median(ts))
    res["family_render_128_s"] = fam_t
    # acquisition: GPU-drawn vs CPU-drawn scalars
    s = sample((64,) * 3, seed=3, device=DEV)
    ts_slow = [timed(nuisance, s, seed=10 + i, device=DEV, strength=0.5)[0] for i in range(8)]
    ts_fast = [timed(nuisance, s, seed=10 + i, device=DEV, strength=0.5, fast=True)[0] for i in range(8)]
    res["nuisance_gpu_scalars_s"] = float(np.median(ts_slow)); res["nuisance_cpu_scalars_s"] = float(np.median(ts_fast))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
