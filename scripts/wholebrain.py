"""A whole brain at 100 µm (S-21-46, level 1 of its 50 µm zarr): the shape tensor's single field and the excised
structure tensor over the volume, and a figure of three orthogonal planes of each.

    python scripts/wholebrain.py run         -> <OUT_ROOT>/shape.zarr (vec, gap, weight), <OUT_ROOT>/st.zarr (vec, gap)
    python scripts/wholebrain.py figure      -> docs/img/wholebrain_100um.png, results/wholebrain_100um.json

Both at σ 1, ρ 10 voxels (1 mm integration at 100 µm); the pipeline calibrated once on the central 256³ (25.6 mm);
chunks without tissue skipped (the volume is masked, air = 0).
"""
import json, sys, time
from pathlib import Path

import numpy as np

SRC = "/hdd/andrew/BRIDGE/50um_scale_S_21_46/sub-test/micr/sub-test_sample-brain_acq-50um_XPCT.ome.zarr"
LEVEL = "1"
OUT_ROOT = Path("/hdd/andrew/shapetensor_runs/S-21-46_100um")
DOC = Path(__file__).resolve().parents[1] / "docs" / "img"
RES = Path(__file__).resolve().parents[1] / "results"
SIGMA, RHO, THRESH = 1.0, 10.0, 1.0


def run():
    from shapetensor import st
    from shapetensor.pipeline import run_volume
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    run_volume(SRC, str(OUT_ROOT / "shape.zarr"), level=LEVEL, chunk=(192, 192, 192), nmax=3, tissue_threshold=THRESH, calibration_size=256, multi=False, sigma=SIGMA, rho=RHO, log=log)
    log("shape tensor done")
    st.run_volume(SRC, str(OUT_ROOT / "st.zarr"), sigma=SIGMA, rho=RHO, level=LEVEL, chunk=(256, 256, 256), arrays=("vec", "gap"), dtype=np.float16, tissue_threshold=THRESH, log=log)
    log("structure tensor done")


def figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch, zarr
    from shapetensor.validate import angle_deg
    from shapetensor.viz import dec_rgb
    img = zarr.open_group(SRC, mode="r")[LEVEL]; sh = zarr.open_group(str(OUT_ROOT / "shape.zarr"), mode="r"); stz = zarr.open_group(str(OUT_ROOT / "st.zarr"), mode="r")
    Z, Y, X = img.shape; c = (Z // 2, Y // 2, X // 2)
    cuts = [("axial (z)", lambda a, k: a[:, c[0], :, :] if k else a[c[0], :, :]), ("coronal (y)", lambda a, k: a[:, :, c[1], :] if k else a[:, c[1], :]),
            ("sagittal (x)", lambda a, k: a[:, :, :, c[2]] if k else a[:, :, c[2]])]
    fig, axs = plt.subplots(3, 3, figsize=(16.5, 16.5), dpi=100, constrained_layout=True)
    stats = {"shape": dict(sh.attrs), "structure_tensor": dict(stz.attrs), "planes": {}}
    for j, (name, cut) in enumerate(cuts):
        im = np.asarray(cut(img, False)).astype(np.float32); tissue = im >= THRESH
        v_sh = np.asarray(cut(sh["vec"], True)).astype(np.float32)[::-1]; w = np.asarray(cut(sh["weight"], False)).astype(np.float32)   # vec stored (x, y, z) -> (z, y, x)
        v_st = np.asarray(cut(stz["vec"], True)).astype(np.float32)[::-1]; gap = np.asarray(cut(stz["gap"], False)).astype(np.float32)
        wref = max(float(np.percentile(w[tissue], 99)), 1e-6)
        e = angle_deg(torch.as_tensor(v_sh), torch.as_tensor(v_st)).numpy(); conf = tissue & (gap > np.percentile(gap[tissue], 75))
        stats["planes"][name] = {"agreement_median_deg": round(float(np.nanmedian(e[tissue])), 1), "agreement_where_tensor_confident_deg": round(float(np.nanmedian(e[conf])), 1),
                                 "weight_p50": round(float(np.percentile(w[tissue], 50)), 3), "tau": sh.attrs.get("tau")}
        axs[0, j].imshow(im, cmap="gray", origin="lower"); axs[0, j].set_title(f"S-21-46 at 100 µm, {name}", fontsize=11)
        axs[1, j].imshow(dec_rgb(v_st, tissue.astype(np.float32)), origin="lower", interpolation="nearest"); axs[1, j].set_title("excised structure tensor (ρ 1 mm)", fontsize=11)
        axs[2, j].imshow(dec_rgb(v_sh, np.clip(w / wref, 0, 1)), origin="lower", interpolation="nearest")
        axs[2, j].set_title(f"shape tensor, brightness = activation ({stats['planes'][name]['agreement_median_deg']}° from the tensor, {stats['planes'][name]['agreement_where_tensor_confident_deg']}° where it is confident)", fontsize=10)
    for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("A whole brain at 100 µm: the excised structure tensor and the shape tensor's single field, σ 1, ρ 10 voxels (DEC: R = |x|, G = |y|, B = |z|)", x=0.01, ha="left", fontsize=12)
    fig.savefig(DOC / "wholebrain_100um.png"); plt.close(fig)
    json.dump(stats, open(RES / "wholebrain_100um.json", "w"), indent=1, default=str); print(json.dumps(stats["planes"], indent=1)); print("wrote wholebrain_100um.png")


if __name__ == "__main__":
    {"run": run, "figure": figure}[sys.argv[1]]()
