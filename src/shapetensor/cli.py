"""``shapetensor`` command line.

    shapetensor run <in.zarr> <out.zarr> [--level 0] [--scale 10] [--chunk 384 384 384] [--tissue-threshold T] [--rank 0 --world 1]
    shapetensor block <in.npy|zarr|tif|nii> <out.zarr> [--scale 10] [--spacing dz dy dx]
    shapetensor structure-tensor <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--no-excise] [--arrays vec gap] [--float16]
    shapetensor validate [--out DIR] [--seeds 1] [--cases band,bend] [--scale 10]
    shapetensor phantom <bundle|crossing|interface|bend|band_crossing|texture> <out.npz> [--seed 0] [--cross-deg 60] ...
    shapetensor demo                                  # a 90° band crossing through the chain, tracked, no data needed
    shapetensor bench [--size 192 256]                # the structure tensor and the chain timed on this GPU
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


def export_block(path, out: dict, attrs: dict | None = None) -> Path:
    """Write a block's field (:func:`shapetensor.orientations`) as a zarr group, vectors in (x, y, z)."""
    import numpy as np
    import zarr
    g = zarr.open_group(str(path), mode="w")
    g.create_array("vecs", data=out["vecs"].flip(1).half().cpu().numpy()); g.create_array("weights", data=out["weights"].half().cpu().numpy())
    g.create_array("vec", data=out["vec"].flip(0).half().cpu().numpy()); g.create_array("weight", data=out["weight"].half().cpu().numpy())
    g.create_array("N", data=out["N"].cpu().numpy().astype(np.uint8)); g.create_array("sheet", data=out["sheet"].cpu().numpy().astype(np.uint8))
    if "labels" in out:
        g.create_array("labels", data=out["labels"].cpu().numpy().astype(np.int32))
    g.attrs.update({"component_order": "xyz", "floor": out["floor"], "response": [float(x) for x in out["response"]], **(attrs or {})})
    return Path(path)


def _chain_args(p):
    p.add_argument("--scale", type=float, default=10.0, help="the window (voxels) every other length is tied to")
    p.add_argument("--nmax", type=int, default=2); p.add_argument("--min-sep-deg", type=float, default=30.0)
    p.add_argument("--no-momentum", action="store_true"); p.add_argument("--no-labels", action="store_true")
    p.add_argument("--device", default="cuda")


def _chain_kwargs(a):
    return {"scale": a.scale, "nmax": a.nmax, "min_sep_deg": a.min_sep_deg, "momentum": not a.no_momentum}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shapetensor", description="orientation fields of any array: one or several directions per voxel, on the GPU")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="the chain over a whole zarr volume, chunked with halos (no seams)")
    r.add_argument("src"); r.add_argument("dst"); r.add_argument("--level", default=None); _chain_args(r)
    r.add_argument("--chunk", type=int, nargs=3, default=[384, 384, 384], help="core voxels per chunk (multiples of the lattice step)")
    r.add_argument("--tissue-threshold", type=float, default=None, help="grey level above which a voxel is tissue (chunks without any are skipped)")
    r.add_argument("--calibration-size", type=int, default=256); r.add_argument("--calibration-origin", type=int, nargs=3, default=None)
    r.add_argument("--rank", type=int, default=0); r.add_argument("--world", type=int, default=1)
    r.add_argument("--no-stitch", action="store_true", help="leave the population labels per chunk")

    b = sub.add_parser("block", help="the chain on one block, written as a zarr group")
    b.add_argument("src"); b.add_argument("dst"); b.add_argument("--level", default=None); _chain_args(b)
    b.add_argument("--spacing", type=float, nargs=3, default=None, help="voxel size per axis (dz dy dx) for anisotropic data")

    s = sub.add_parser("structure-tensor", help="the (excised) structure tensor over a whole zarr volume")
    s.add_argument("src"); s.add_argument("dst"); s.add_argument("--level", default=None)
    s.add_argument("--rho", type=float, default=10.0); s.add_argument("--sigma", type=float, default=1.0); s.add_argument("--device", default="cuda")
    s.add_argument("--chunk", type=int, nargs=3, default=[256, 256, 256]); s.add_argument("--no-excise", action="store_true")
    s.add_argument("--arrays", nargs="+", default=["S", "val", "vec"], choices=["S", "val", "vec", "gap"], help="what to write (vec gap for a compact field)")
    s.add_argument("--float16", action="store_true"); s.add_argument("--tissue-threshold", type=float, default=None)
    s.add_argument("--rank", type=int, default=0); s.add_argument("--world", type=int, default=1)

    v = sub.add_parser("validate", help="the acceptance suite: phantoms with truth, the chain against the structure tensor, tracking")
    v.add_argument("--out", default="validation"); v.add_argument("--seeds", type=int, default=1); v.add_argument("--cases", default="")
    v.add_argument("--scale", type=float, default=10.0); v.add_argument("--device", default="cuda")

    p = sub.add_parser("phantom", help="build one phantom with its truth")
    p.add_argument("name", choices=["bundle", "crossing", "interface", "bend", "band_crossing", "texture"]); p.add_argument("out"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--angle", type=float, default=60.0); p.add_argument("--cross-deg", type=float, default=60.0); p.add_argument("--step", type=float, default=40.0)
    p.add_argument("--t-junction", action="store_true"); p.add_argument("--straight", action="store_true"); p.add_argument("--device", default="cuda")

    d = sub.add_parser("demo", help="a 90° band crossing through the chain, tracked against the structure tensor (no data needed)")
    d.add_argument("--cross-deg", type=float, default=90.0); d.add_argument("--device", default="cuda")

    e = sub.add_parser("bench", help="time the structure tensor and the chain on this GPU")
    e.add_argument("--size", type=int, nargs="+", default=[256]); e.add_argument("--scale", type=float, default=10.0); e.add_argument("--device", default="cuda")

    t = sub.add_parser("stats", help="image statistics of a real crop (to match the phantom texture)")
    t.add_argument("path"); t.add_argument("--level", type=int, default=0); t.add_argument("--threshold", type=float, default=None); t.add_argument("-o", "--out", default="stats.json")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .volume import run_volume, stitch_labels
        run_volume(a.src, a.dst, level=a.level, chunk=tuple(a.chunk), tissue_threshold=a.tissue_threshold, calibration_size=a.calibration_size,
                   calibration_origin=a.calibration_origin, rank=a.rank, world=a.world, device=a.device, labels=not a.no_labels, **_chain_kwargs(a))
        if not a.no_labels and not a.no_stitch and a.world == 1:
            stitch_labels(a.dst)
    elif a.cmd == "block":
        import torch
        from .field import orientations
        v = torch.as_tensor(read_volume(a.src, a.level), device=a.device)
        out = orientations(v, spacing=a.spacing, device=a.device, **_chain_kwargs(a))
        export_block(a.dst, out, {"source": a.src, **_chain_kwargs(a)})
        print(f"wrote {a.dst}: {len(out.get('table', []))} populations, presence floor {out['floor']:.4f}")
    elif a.cmd == "structure-tensor":
        import numpy as np
        from .st import run_volume
        run_volume(a.src, a.dst, sigma=a.sigma, rho=a.rho, chunk=tuple(a.chunk), excise=not a.no_excise, level=a.level, arrays=tuple(a.arrays),
                   dtype=np.float16 if a.float16 else np.float32, tissue_threshold=a.tissue_threshold, rank=a.rank, world=a.world, device=a.device)
    elif a.cmd == "validate":
        from .validate import run
        run(a.out, seeds=a.seeds, cases=a.cases, scale=a.scale, device=a.device)
    elif a.cmd == "phantom":
        from .phantoms import build, save
        kw = {"bundle": {"angle": a.angle}, "crossing": {"cross_deg": a.cross_deg}, "interface": {"step": a.step}, "bend": {"straight": a.straight},
              "band_crossing": {"cross_deg": a.cross_deg, "t_junction": a.t_junction}, "texture": {}}[a.name]
        img, tr, meta = build(a.name, a.seed, device=a.device, **kw)
        print("wrote", save(a.out, img, tr, meta), img.shape)
    elif a.cmd == "demo":
        from .validate import run
        res = run(None, cases="band crossing 90" if a.cross_deg == 90.0 else f"band crossing {a.cross_deg:g}", device=a.device)
        for k, r in res["cases"].items():
            print(k); print(json.dumps(r["tracking"], indent=1))
    elif a.cmd == "bench":
        import time, torch
        from . import st
        from .field import Orientations
        for n in a.size:
            v = torch.randn(n, n, n, device=a.device) * 20 + 128
            st.structure_tensor(v, rho=a.scale); torch.cuda.synchronize(); t0 = time.perf_counter()
            st.structure_tensor(v, rho=a.scale); torch.cuda.synchronize(); t_st = time.perf_counter() - t0
            m = Orientations(scale=a.scale, device=a.device); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
            pop = m.populations(v); torch.cuda.synchronize(); t_lat = time.perf_counter() - t0
            m.field(pop, v.shape); torch.cuda.synchronize(); t_all = time.perf_counter() - t0
            print(json.dumps({"size": n, "device": torch.cuda.get_device_name(0), "structure_tensor_s": round(t_st, 2), "lattice_s": round(t_lat, 1), "chain_s": round(t_all, 1),
                              "Mvox_per_s": round(n ** 3 / t_all / 1e6, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}))
    elif a.cmd == "stats":
        from .stats import real_crop_profile
        out = real_crop_profile(a.path, a.level, a.threshold)
        Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
