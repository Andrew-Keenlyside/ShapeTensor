"""``shapetensor`` command line.

    shapetensor run <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--chunk 192 192 192] [--tissue-threshold T]
    shapetensor block <in.npy> <out.zarr> [--rho 10] [--sigma 1] [--voxel-mm 0.02]
    shapetensor structure-tensor <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--no-excise] [--arrays vec gap] [--float16]
    shapetensor validate [--out validation] [--seeds 0 1 2] [--rho 10] [--settings calibrated]
    shapetensor phantom <bundle|crossing|interface> <out.npz> [--seed 0] [--cross-deg 60] [--step 40] [--angle 60]
    shapetensor demo [--out field.zarr]              # a crossing phantom through the whole pipeline, no data needed
    shapetensor bench [--size 192 256]               # the structure tensor and the pipeline timed on this GPU
    shapetensor stats <in.ome.zarr> [--level 0] [--threshold T] -o stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_volume(path: str, level: str | None = None):
    """A volume ``(Z, Y, X)`` float32 from ``.npy``, a zarr array or group (``level``), a TIFF stack (``tifffile``) or a
    NIfTI file (``nibabel``; axes reversed to (Z, Y, X))."""
    import numpy as np
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if low.endswith((".tif", ".tiff")):
        import tifffile
        return np.asarray(tifffile.imread(path)).astype(np.float32)
    if low.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(path).dataobj).astype(np.float32).transpose(2, 1, 0)
    import zarr
    z = zarr.open(path, mode="r")
    return np.asarray(z[level] if level is not None else z).astype(np.float32)


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
    r.add_argument("--volume-calibration", action="store_true", help="one calibration for the whole volume (default: every chunk on itself, the volume's as fallback)")
    r.add_argument("--min-bins", type=int, default=12, help="orientation bins (of 24) a chunk's teacher must cover to use its own calibration")
    r.add_argument("--blend", type=int, default=32, help="overlap band (voxels) over which neighbouring chunks' tensors are ramp-blended; 0 = none")
    r.add_argument("--spacing", type=float, nargs=3, default=None, help="voxel size per axis (dz dy dx) for anisotropic data")
    r.add_argument("--pipeline", choices=["released", "dev"], default="released",
                   help="released: anatomix encoder 2, block populations, intensity tensors; dev: dev-vit + sliceTCA, fibre sets, shape tensors")
    r.add_argument("--dev-config", default=None, help="a JSON DevConfig (shapetensor.dev) for --pipeline dev; default: the built-in settings")

    b = sub.add_parser("block", help="the pipeline on one block (.npy or a zarr array), exported with its layers")
    b.add_argument("src"); b.add_argument("dst"); b.add_argument("--level", default=None); _scales(b)
    b.add_argument("--nmax", type=int, default=3); b.add_argument("--tissue-threshold", type=float, default=None)
    b.add_argument("--engine", default="anatomix_enc2"); b.add_argument("--voxel-mm", type=float, default=0.02); b.add_argument("--spacing", type=float, nargs=3, default=None)

    s = sub.add_parser("structure-tensor", help="the (excised) structure tensor over a whole zarr volume")
    s.add_argument("src"); s.add_argument("dst"); s.add_argument("--level", default=None); _scales(s)
    s.add_argument("--chunk", type=int, nargs=3, default=[256, 256, 256]); s.add_argument("--no-excise", action="store_true")
    s.add_argument("--arrays", nargs="+", default=["S", "val", "vec"], choices=["S", "val", "vec", "gap"], help="what to write (vec gap for a compact field)")
    s.add_argument("--float16", action="store_true"); s.add_argument("--tissue-threshold", type=float, default=None, help="skip chunks with no voxel at or above it")
    s.add_argument("--rank", type=int, default=0); s.add_argument("--world", type=int, default=1)

    v = sub.add_parser("validate", help="the three phantoms: structure tensor against shape tensor")
    v.add_argument("--out", default="validation"); v.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); _scales(v)
    v.add_argument("--engine", default="anatomix_enc2"); v.add_argument("--settings", nargs="+", default=["calibrated"], choices=["calibrated", "coherent"])

    p = sub.add_parser("phantom", help="build one phantom with its truth")
    p.add_argument("name", choices=["bundle", "crossing", "interface"]); p.add_argument("out"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--angle", type=float, default=60.0); p.add_argument("--cross-deg", type=float, default=60.0); p.add_argument("--step", type=float, default=15.0)
    p.add_argument("--device", default="cuda")

    d = sub.add_parser("demo", help="build a crossing phantom and run the whole pipeline on it (no data needed)")
    d.add_argument("--out", default=None, help="write the field as a zarr group"); d.add_argument("--cross-deg", type=float, default=90.0); _scales(d)

    e = sub.add_parser("bench", help="time the structure tensor and the pipeline on this GPU")
    e.add_argument("--size", type=int, nargs="+", default=[192]); _scales(e)

    t = sub.add_parser("stats", help="image statistics of a real crop (to match the phantom texture)")
    t.add_argument("path"); t.add_argument("--level", type=int, default=0); t.add_argument("--threshold", type=float, default=None); t.add_argument("-o", "--out", default="stats.json")

    a = ap.parse_args(argv)
    if a.cmd == "run" and a.pipeline == "dev":
        import json
        from .dev import DevConfig, run_volume_dev
        cfg = DevConfig.from_dict(json.load(open(a.dev_config))) if a.dev_config else DevConfig()
        cfg.sigma, cfg.rho = a.sigma, a.rho
        run_volume_dev(a.src, a.dst, cfg, level=a.level, chunk=tuple(a.chunk), nmax=a.nmax, tissue_threshold=a.tissue_threshold,
                       calibration_size=a.calibration_size, calibration_origin=a.calibration_origin, calibrate_chunks=not a.volume_calibration,
                       min_bins=a.min_bins, blend=a.blend, multi=not a.no_multi, spacing=a.spacing, rank=a.rank, world=a.world, device=a.device)
    elif a.cmd == "run":
        from .pipeline import run_volume
        run_volume(a.src, a.dst, level=a.level, chunk=tuple(a.chunk), nmax=a.nmax, tissue_threshold=a.tissue_threshold, calibration_size=a.calibration_size,
                   calibration_origin=a.calibration_origin, calibrate_chunks=not a.volume_calibration, min_bins=a.min_bins, blend=a.blend, multi=not a.no_multi, spacing=a.spacing,
                   rank=a.rank, world=a.world, device=a.device, sigma=a.sigma, rho=a.rho, engine=a.engine)
    elif a.cmd == "block":
        import numpy as np, torch
        from .merge import export
        from .pipeline import population_field
        v = torch.as_tensor(read_volume(a.src, a.level), device=a.device)
        out = population_field(v, None if a.tissue_threshold is None else v >= a.tissue_threshold, nmax=a.nmax, device=a.device, sigma=a.sigma, rho=a.rho, engine=a.engine, spacing=a.spacing)
        export(a.dst, out, a.voxel_mm, {"source": a.src, "sigma": a.sigma, "rho": a.rho, "engine": a.engine})
        print(f"wrote {a.dst}: {int(out['modes'].shape[0])} populations, tau {out['tau']:.3f}")
    elif a.cmd == "structure-tensor":
        from .st import run_volume
        import numpy as np
        run_volume(a.src, a.dst, sigma=a.sigma, rho=a.rho, chunk=tuple(a.chunk), excise=not a.no_excise, level=a.level, arrays=tuple(a.arrays),
                   dtype=np.float16 if a.float16 else np.float32, tissue_threshold=a.tissue_threshold, rank=a.rank, world=a.world, device=a.device)
    elif a.cmd == "validate":
        from .validate import run
        run(a.out, seeds=tuple(a.seeds), sigma=a.sigma, rho=a.rho, engine=a.engine, settings=tuple(a.settings), device=a.device)
    elif a.cmd == "phantom":
        from .phantoms import build, save
        kw = {"bundle": {"angle": a.angle}, "crossing": {"cross_deg": a.cross_deg}, "interface": {"step": a.step}}[a.name]
        img, tr, meta = build(a.name, a.seed, device=a.device, **kw)
        print("wrote", save(a.out, img, tr, meta), img.shape)
    elif a.cmd == "demo":
        import time, torch
        from .phantoms import Grid, TextureParams, build, truth_field
        from .pipeline import apply, calibrate
        from .validate import ANGLES, angle_deg
        g = Grid(); tex = TextureParams(dispersion_deg=8.0, pitch_deg=4.0); t0 = time.time()
        cal = calibrate([torch.as_tensor(build("bundle", 100 + k, g, tex, angle=float(x))[0], device=a.device).float() for k, x in enumerate(ANGLES)], margin=32, sigma=a.sigma, rho=a.rho, device=a.device)
        img, tr, meta = build("crossing", 0, g, tex, cross_deg=a.cross_deg, device=a.device)
        out = apply(cal, torch.as_tensor(img, device=a.device).float(), device=a.device)
        ny = g.shape[1]; sel = torch.zeros(g.shape, dtype=torch.bool, device=a.device); sel[g.inner] = True
        t1, t2 = truth_field(tr, "dir1", ny).to(a.device), truth_field(tr, "dir2", ny).to(a.device)
        e = torch.minimum(angle_deg(out["vec"], t1), angle_deg(out["vec"], t2))[sel]
        print(f"demo: a {a.cross_deg:g}° crossing phantom ({'x'.join(map(str, g.shape))}); calibrated on {len(ANGLES)} bundles; {out['modes'].shape[0]} populations found; "
              f"single field within 15° of the nearer population at {float((e <= 15).float().mean()):.0%} of voxels; {time.time() - t0:.0f} s")
        if a.out:
            from .merge import export
            export(a.out, out, g.vox, {"source": "demo crossing phantom", **{k: v for k, v in meta.items() if k != "grid"}}); print("wrote", a.out)
    elif a.cmd == "bench":
        import json, time, torch
        from . import st
        from .phantoms import Grid, build
        from .pipeline import population_field
        for n in a.size:
            g = Grid(core=(n - 64, n - 64, n - 64), margin=(32, 32, 32)); v = torch.as_tensor(build("crossing", 0, g, cross_deg=90.0, device=a.device)[0], device=a.device).float()
            st.structure_tensor(v, sigma=a.sigma, rho=a.rho); torch.cuda.synchronize(); t0 = time.perf_counter()
            st.structure_tensor(v, sigma=a.sigma, rho=a.rho); torch.cuda.synchronize(); t_st = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats(); tm = {}; population_field(v, timings=tm, sigma=a.sigma, rho=a.rho, device=a.device)
            print(json.dumps({"size": n, "device": torch.cuda.get_device_name(0), "structure_tensor_s": round(t_st, 2), "pipeline_s": tm, "pipeline_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}))
    elif a.cmd == "stats":
        from .stats import real_crop_profile
        out = real_crop_profile(a.path, a.level, a.threshold)
        Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
