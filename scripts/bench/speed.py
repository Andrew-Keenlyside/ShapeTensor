"""Speed of the shapetensor paths relative to the structure tensor, on one GPU, for 128^3 and 256^3 blocks.

Arms: the structure-tensor package's algorithm on the CPU (scipy, as that package computes it), shapetensor's GPU
structure tensor + eigen, the excised tensor + eigen, fixels (order-8 ODF + CSD + peaks on a rho/4 site grid), the
shape tensor from anatomix48 (no de-checkerboarding) + eigen, and K + TCA rank 3 on the rho/4 site grid.
Median of 3 timings after a warm-up; voxels per second and the factor relative to the GPU structure tensor.

    python scripts/bench/speed.py [--sizes 128 256] [--rho 16] -> results/speed.json and a markdown table on stdout
"""
import argparse, json, time
import numpy as np, torch
from scipy import ndimage as ndi
from shapetensor import st, DATA_ROOT
from shapetensor.features import make_engine
from shapetensor.pca import PCA
from shapetensor.tensor import gradients, k_dense
from shapetensor.readout import common, single


def scipy_structure_tensor(vol, sigma, rho, truncate):
    Vx = ndi.gaussian_filter(vol, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)
    Vy = ndi.gaussian_filter(vol, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)
    Vz = ndi.gaussian_filter(vol, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)
    S = np.empty((6,) + vol.shape, np.float32)
    for k, (a, b) in enumerate(((Vx, Vx), (Vy, Vy), (Vz, Vz), (Vx, Vy), (Vx, Vz), (Vy, Vz))):
        S[k] = ndi.gaussian_filter(a * b, rho, mode="nearest", truncate=truncate)
    return S


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256]); ap.add_argument("--rho", type=float, default=16.0)
    ap.add_argument("--skip-cpu-above", type=int, default=256); a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {"device": torch.cuda.get_device_name(0), "rho": a.rho, "sizes": {}}
    for n in a.sizes:
        vol = (150 + 10 * rng.normal(size=(n, n, n))).astype(np.float32)
        v = torch.as_tensor(vol, device="cuda")
        nvox = n ** 3
        rows = {}
        if n <= a.skip_cpu_above:
            rows["structure-tensor algorithm, CPU (scipy)"] = timed(lambda: scipy_structure_tensor(vol, 1.0, a.rho, 4.0), n=1)
        rows["structure tensor + eigen, GPU"] = timed(lambda: st.eig_special_3d(st.structure_tensor_3d(v, 1.0, a.rho)))
        rows["excised tensor + eigen, GPU"] = timed(lambda: st.eig_special_3d(st.excised_structure_tensor_3d(v, 1.0, a.rho)))
        rows["fixels (ODF + CSD + peaks), GPU"] = timed(lambda: st.fixels_3d(v, 1.0, a.rho), n=2)
        b = max(1, int(a.rho // 4)); m = int(np.ceil((4 * (1 + a.rho) + 64) / b) * b)
        core = ((m, (n - m) // b * b),) * 3
        if core[0][1] - core[0][0] >= 4 * b:
            eng = make_engine("anatomix48", nyquist="none")
            rows["shape tensor (anatomix48) + eigen, GPU"] = timed(lambda: st.eig_special_3d(st.shape_structure_tensor_3d(v, 1.0, a.rho, engine="anatomix48", engine_kwargs={"nyquist": "none"})), n=2)

            def k_tca():
                F = eng(v); F = PCA.fit(F, 16)(F) if F.shape[0] > 16 else F
                G = gradients(F, 1.0, 4.0); K, _ = k_dense(G, a.rho, 4.0, b, core)
                single.tca(common.unpack(K.reshape(-1, K.shape[-1]), 48), rank=3)
            rows["K + TCA rank 3 on the rho/4 site grid (anatomix48), GPU"] = timed(k_tca, n=2)
        ref = rows["structure tensor + eigen, GPU"]
        out["sizes"][str(n)] = {k: {"seconds": t, "Mvox_per_s": nvox / t / 1e6, "x_structure_tensor_gpu": t / ref} for k, t in rows.items()}
        print(f"\n### {n}^3 block ({nvox / 1e6:.1f} Mvox), rho {a.rho:g}, {out['device']}\n")
        print("| arm | seconds | Mvoxel/s | relative to the GPU structure tensor |\n|---|---|---|---|")
        for k, t in rows.items():
            print(f"| {k} | {t:.2f} | {nvox / t / 1e6:.1f} | {t / ref:.1f}x |")
        del v; torch.cuda.empty_cache()
    json.dump(out, open(f"{DATA_ROOT}/results/speed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
