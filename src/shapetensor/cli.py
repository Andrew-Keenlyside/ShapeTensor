"""``shapetensor`` command line.

    shapetensor run <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--chunk 192 192 192] [--tissue-threshold T]
    shapetensor block <in.npy> <out.zarr> [--rho 10] [--sigma 1] [--voxel-mm 0.02]
    shapetensor structure-tensor <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--no-excise]
    shapetensor validate [--out validation] [--seeds 0 1 2] [--rho 10]
    shapetensor phantom <bundle|crossing|interface> <out.npz> [--seed 0] [--cross-deg 60] [--step 15] [--angle 60]
    shapetensor stats <in.ome.zarr> [--level 0] [--threshold T] -o stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _scales(p):
    p.add_argument("--rho", type=float, default=10.0, help="integration window, voxels (the same for every tensor)")
    p.add_argument("--sigma", type=float, default=1.0, help="derivative scale, voxels")
    p.add_argument("--device", default="cuda")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shapetensor", description="fibre orientation fields from shape features, on the GPU")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the pipeline over a whole zarr volume, chunked with halos")
    r.add_argument("src"); r.add_argument("dst"); r.add_argument("--level", default=None); _scales(r)
    r.add_argument("--chunk", type=int, nargs=3, default=[192, 192, 192]); r.add_argument("--nmax", type=int, default=3)
    r.add_argument("--tissue-threshold", type=float, default=None); r.add_argument("--calibration-size", type=int, default=256)
    r.add_argument("--calibration-origin", type=int, nargs=3, default=None); r.add_argument("--engine", default="anatomix_enc2")
    r.add_argument("--no-multi", action="store_true"); r.add_argument("--rank", type=int, default=0); r.add_argument("--world", type=int, default=1)

    b = sub.add_parser("block", help="the pipeline on one block (.npy or a zarr array), exported with its layers")
    b.add_argument("src"); b.add_argument("dst"); b.add_argument("--level", default=None); _scales(b)
    b.add_argument("--nmax", type=int, default=3); b.add_argument("--tissue-threshold", type=float, default=None)
    b.add_argument("--engine", default="anatomix_enc2"); b.add_argument("--voxel-mm", type=float, default=0.02)

    s = sub.add_parser("structure-tensor", help="the (excised) structure tensor over a whole zarr volume")
    s.add_argument("src"); s.add_argument("dst"); s.add_argument("--level", default=None); _scales(s)
    s.add_argument("--chunk", type=int, nargs=3, default=[256, 256, 256]); s.add_argument("--no-excise", action="store_true")
    s.add_argument("--rank", type=int, default=0); s.add_argument("--world", type=int, default=1)

    v = sub.add_parser("validate", help="the three phantoms: structure tensor against shape tensor")
    v.add_argument("--out", default="validation"); v.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); _scales(v)
    v.add_argument("--engine", default="anatomix_enc2"); v.add_argument("--settings", nargs="+", default=["coherent", "calibrated"], choices=["coherent", "calibrated"])

    p = sub.add_parser("phantom", help="build one phantom with its truth")
    p.add_argument("name", choices=["bundle", "crossing", "interface"]); p.add_argument("out"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--angle", type=float, default=60.0); p.add_argument("--cross-deg", type=float, default=60.0); p.add_argument("--step", type=float, default=15.0)
    p.add_argument("--device", default="cuda")

    t = sub.add_parser("stats", help="image statistics of a real crop (to match the phantom texture)")
    t.add_argument("path"); t.add_argument("--level", type=int, default=0); t.add_argument("--threshold", type=float, default=None); t.add_argument("-o", "--out", default="stats.json")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run_volume
        run_volume(a.src, a.dst, level=a.level, chunk=tuple(a.chunk), nmax=a.nmax, tissue_threshold=a.tissue_threshold, calibration_size=a.calibration_size,
                   calibration_origin=a.calibration_origin, multi=not a.no_multi, rank=a.rank, world=a.world, device=a.device, sigma=a.sigma, rho=a.rho, engine=a.engine)
    elif a.cmd == "block":
        import numpy as np, torch
        from .merge import export
        from .pipeline import population_field
        if a.src.endswith(".npy"):
            img = np.load(a.src)
        else:
            import zarr
            z = zarr.open(a.src, mode="r"); img = np.asarray(z[a.level] if a.level is not None else z)
        v = torch.as_tensor(img.astype(np.float32), device=a.device)
        out = population_field(v, None if a.tissue_threshold is None else v >= a.tissue_threshold, nmax=a.nmax, device=a.device, sigma=a.sigma, rho=a.rho, engine=a.engine)
        export(a.dst, out, a.voxel_mm, {"source": a.src, "sigma": a.sigma, "rho": a.rho, "engine": a.engine})
        print(f"wrote {a.dst}: {int(out['modes'].shape[0])} populations, tau {out['tau']:.3f}")
    elif a.cmd == "structure-tensor":
        from .st import run_volume
        run_volume(a.src, a.dst, sigma=a.sigma, rho=a.rho, chunk=tuple(a.chunk), excise=not a.no_excise, level=a.level, rank=a.rank, world=a.world, device=a.device)
    elif a.cmd == "validate":
        from .validate import run
        run(a.out, seeds=tuple(a.seeds), sigma=a.sigma, rho=a.rho, engine=a.engine, settings=tuple(a.settings), device=a.device)
    elif a.cmd == "phantom":
        from .phantoms import build, save
        kw = {"bundle": {"angle": a.angle}, "crossing": {"cross_deg": a.cross_deg}, "interface": {"step": a.step}}[a.name]
        img, tr, meta = build(a.name, a.seed, device=a.device, **kw)
        print("wrote", save(a.out, img, tr, meta), img.shape)
    elif a.cmd == "stats":
        from .stats import real_crop_profile
        out = real_crop_profile(a.path, a.level, a.threshold)
        Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
