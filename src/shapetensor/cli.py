"""``shapetensor`` command line: ``stats``, ``phantoms``, ``run``, ``report``.

    shapetensor stats  <omezarr> [--level 0] [--threshold T] -o stats.json
    shapetensor phantoms --pids P3-15 P8 --seeds 0 1 2 [--tex tex.json]
    shapetensor run --spec spec.json            # Spec fields as JSON (validate/suite.py)
    shapetensor report --tag main               # tables + figures from the results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _spec_from(path: str):
    from .validate.suite import Spec
    d = json.loads(Path(path).read_text())
    for k in ("pids", "seeds", "engines", "rhos"):
        if k in d:
            d[k] = tuple(d[k])
    return Spec(**d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shapetensor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("path"); s.add_argument("--level", type=int, default=0)
    s.add_argument("--threshold", type=float, default=None); s.add_argument("-o", "--out", default="stats.json")
    p = sub.add_parser("phantoms"); p.add_argument("--pids", nargs="+", required=True); p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--tex", default=None); p.add_argument("--rho-max", type=float, default=32.0); p.add_argument("--vox", type=float, default=0.02)
    r = sub.add_parser("run"); r.add_argument("--spec", required=True); r.add_argument("--only", choices=("caches", "single", "multi", "hybrid", "all"), default="all")
    q = sub.add_parser("report"); q.add_argument("--tag", default="main"); q.add_argument("--out", default=None)
    v = sub.add_parser("volume", help="structure-tensor-compatible outputs for a whole zarr volume, chunked, on one GPU")
    v.add_argument("src"); v.add_argument("dst"); v.add_argument("--level", default=None)
    v.add_argument("--sigma", type=float, default=1.0); v.add_argument("--rho", type=float, default=16.0)
    v.add_argument("--truncate", type=float, default=4.0); v.add_argument("--chunk", type=int, nargs=3, default=[256, 256, 256])
    v.add_argument("--outputs", nargs="+", default=["S", "eig", "excised", "fixels"], choices=["S", "eig", "excised", "fixels"])
    v.add_argument("--excise-factor", type=float, default=3.0); v.add_argument("--rank", type=int, default=0)
    v.add_argument("--world", type=int, default=1); v.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        from .phantoms import stats
        out = stats.real_crop_profile(a.path, a.level, a.threshold)
        Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)
    elif a.cmd == "phantoms":
        from .phantoms import make
        tex = make.TextureParams(**json.loads(Path(a.tex).read_text())) if a.tex else make.TextureParams()
        for pid in a.pids:
            for seed in a.seeds:
                grid = make.grid_for(pid, vox=a.vox, rho_max=a.rho_max)
                img, tr, meta = make.build(pid, seed, tex, grid)
                print("wrote", make.save(pid, seed, img, tr, meta), img.shape)
    elif a.cmd == "run":
        from .validate import suite
        spec = _spec_from(a.spec)
        if a.only == "all":
            suite.run_all(spec)
        else:
            for pid in spec.pids:
                for seed in spec.seeds:
                    if a.only == "caches":
                        suite.build_caches(spec, pid, seed)
                        continue
                    for engine in spec.engines:
                        for rho in spec.rhos:
                            if a.only == "single":
                                suite.run_single(spec, pid, seed, engine, rho)
                            elif a.only == "multi" and suite._is_multi(pid):
                                suite.run_multi(spec, pid, seed, engine, rho)
                            elif a.only == "hybrid" and engine != "intensity":
                                suite.run_hybrid(spec, pid, seed, engine, rho)
    elif a.cmd == "report":
        from .validate import report
        report.write(a.tag, a.out)
    elif a.cmd == "volume":
        from .st import run_volume
        run_volume(a.src, a.dst, sigma=a.sigma, rho=a.rho, truncate=a.truncate, chunk=tuple(a.chunk), outputs=tuple(a.outputs),
                   excise_factor=a.excise_factor, level=a.level, rank=a.rank, world=a.world, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
