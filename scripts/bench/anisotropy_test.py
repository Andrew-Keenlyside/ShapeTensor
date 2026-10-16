"""Robustness to anisotropic voxels, on the phantoms (truth known).

Each phantom (20 um isotropic) is resampled to 20 x 20 x 50 um -- 2.5x coarser along one axis, by area averaging --
and the directions are read back three ways: (a) *naive*: the voxel-unit filters, the anisotropy ignored;
(b) *spacing*: per-axis filters and physical gradients (``spacing=``); (c) *resampled*: the volume interpolated
back to isotropic first, then the plain filters. Arms: the plain and the excised structure tensor and the anatomix48
shape tensor; the isotropic run is the reference. Coarse axis y (through the scoring slab: the fibres lie in z-x, so
the lost resolution is across them -- the realistic section-stack case) and z (along the main fibre axis).

    python scripts/bench/anisotropy_test.py [pid ...]      -> results/anisotropy_test.json
"""
import json, math, sys, time
import numpy as np, torch, torch.nn.functional as Fn
from shapetensor import DATA_ROOT, st
from shapetensor.filters import gaussian
from shapetensor.phantoms import make
from shapetensor.validate import score

PIDS = sys.argv[1:] or ["P3-40-t0.5", "P1-t0.5", "P4-lines-single-t0.5", "P4-lines-c60-f1", "P6-t0.5"]
RHOS = (10.0, 16.0)
SIGMA, TR, RATIO, STRIDE = 1.0, 4.0, 2.5, 4
EK = {"nyquist": "none"}
BANDS = ("0-0.1mm", "0.25-0.5mm", ">2mm")


def coarsen(v, axis, ratio):
    size = list(v.shape); size[axis] = int(round(size[axis] / ratio))
    return Fn.interpolate(v[None, None], size=size, mode="area")[0, 0]


def refine(v, axis, n):
    size = list(v.shape); size[axis] = n
    return Fn.interpolate(v[None, None], size=size, mode="trilinear", align_corners=False)[0, 0]


def site_index(idx, ratio):
    """Fine-grid voxel index -> nearest index on the grid coarsened by ``ratio``."""
    return np.clip(np.round((np.asarray(idx) + 0.5) / ratio - 0.5).astype(int), 0, None)


def dirs_at(S, zs, row, xs):
    """``S (6, Z, Y, X)`` -> unit directions ``(nz * nx, 3)`` in (z, y, x) order at the sites."""
    Ss = S[:, zs][:, :, row][:, :, xs].contiguous()
    _, vec = st.eig_special_3d(Ss)
    return vec.reshape(3, -1).T.flip(1).cpu().numpy()


def bands_of(d, t):
    """Medians by distance band; phantoms without distances (the line crossings) report the ``all`` band, scored
    against the nearer of the two true directions where two exist."""
    err = score.angle_err(d, t["dir1"])
    if "dir2" in t and np.isfinite(t["dir2"][:, 0]).any():
        err = np.fmin(err, score.angle_err(d, t["dir2"]))
    if not np.isfinite(t["dist"]).any():
        return {"all": round(float(np.nanmedian(err[t["region"] == 2] if "region" in t else err)), 2)}
    b = score.band_summary(err, t["dist"], t["region"])
    return {k: round(float(b[k]["median"]), 2) for k in BANDS if k in b and "median" in b[k]}


def main():
    t0 = time.time(); out = {}
    for pid in PIDS:
        img, tr, meta, grid = make.load(pid, 0)
        v = torch.as_tensor(img, device="cuda").float()
        Z, Y, X = v.shape
        (z0, z1), (x0, x1) = grid.core_zx
        zs, xs = np.arange(z0, z1, STRIDE), np.arange(x0, x1, STRIDE)
        t = score.sample_truth(tr, zs, xs)
        out[pid] = {}
        for rho in RHOS:
            res = {}
            # the isotropic reference
            S = st.structure_tensor_3d(v, SIGMA, rho, TR); res["iso plain"] = bands_of(dirs_at(S, zs, grid.y_mid, xs), t); del S
            S = st.excised_structure_tensor_3d(v, SIGMA, rho, TR); res["iso excised"] = bands_of(dirs_at(S, zs, grid.y_mid, xs), t); del S
            S = st.shape_structure_tensor_3d(v, SIGMA, rho, TR, engine="anatomix48", n_pcs=8, engine_kwargs=EK)
            res["iso anatomix48"] = bands_of(dirs_at(S, zs, grid.y_mid, xs), t); del S; torch.cuda.empty_cache()
            for axis, name in ((1, "y"), (0, "z")):
                va = coarsen(v, axis, RATIO)
                sp = [1.0, 1.0, 1.0]; sp[axis] = RATIO
                zs_a = site_index(zs, RATIO) if axis == 0 else zs
                row_a = site_index([grid.y_mid], RATIO)[0] if axis == 1 else grid.y_mid
                # PSF matching: the area average over 2.5 voxels is a box of sd 2.5 / sqrt(12) fine voxels; blur the other
                # two axes by the same amount so all three share one resolution, then use physical spacing
                psf = tuple(0.0 if k == axis else RATIO / math.sqrt(12) for k in range(3))
                vm = gaussian(va, psf, truncate=TR)
                arms = {
                    "naive plain": lambda: st.structure_tensor_3d(va, SIGMA, rho, TR),
                    "psf-matched plain": lambda: st.structure_tensor_3d(vm, SIGMA, rho, TR, spacing=sp),
                    "psf-matched excised": lambda: st.excised_structure_tensor_3d(vm, SIGMA, rho, TR, spacing=sp),
                    "spacing plain": lambda: st.structure_tensor_3d(va, SIGMA, rho, TR, spacing=sp),
                    "naive excised": lambda: st.excised_structure_tensor_3d(va, SIGMA, rho, TR),
                    "spacing excised": lambda: st.excised_structure_tensor_3d(va, SIGMA, rho, TR, spacing=sp),
                    "naive anatomix48": lambda: st.shape_structure_tensor_3d(va, SIGMA, rho, TR, engine="anatomix48", n_pcs=8, engine_kwargs=EK),
                    "spacing anatomix48": lambda: st.shape_structure_tensor_3d(va, SIGMA, rho, TR, engine="anatomix48", n_pcs=8, engine_kwargs=EK, spacing=sp),
                }
                for arm, fn in arms.items():
                    S = fn(); res[f"{name}: {arm}"] = bands_of(dirs_at(S, zs_a, row_a, xs), t); del S; torch.cuda.empty_cache()
                vr = refine(va, axis, v.shape[axis])
                S = st.structure_tensor_3d(vr, SIGMA, rho, TR); res[f"{name}: resampled plain"] = bands_of(dirs_at(S, zs, grid.y_mid, xs), t); del S
                S = st.excised_structure_tensor_3d(vr, SIGMA, rho, TR); res[f"{name}: resampled excised"] = bands_of(dirs_at(S, zs, grid.y_mid, xs), t)
                del S, vr, va, vm; torch.cuda.empty_cache()
            out[pid][f"rho{rho:g}"] = res
            print(f"[{time.time() - t0:5.0f}s] {pid} rho {rho:g}", flush=True)
            for arm, b in res.items():
                print(f"    {arm:28s} " + "  ".join(f"{k} {v:5.1f}" for k, v in b.items()), flush=True)
        del v; torch.cuda.empty_cache()
        path = f"{DATA_ROOT}/results/anisotropy_test.json"
        try:
            prev = json.load(open(path))
        except Exception:
            prev = {}
        prev.update(out); json.dump(prev, open(path, "w"), indent=1)
    print("ANISOTROPY TEST DONE", flush=True)


if __name__ == "__main__":
    main()
