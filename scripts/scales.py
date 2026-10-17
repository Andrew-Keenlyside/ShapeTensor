"""HiP-CT across image scales and brain regions: the standard row on one 192³ block per scale (2.2, 10, 20, 50, 100
and 200 µm voxels; I74 for the two finest, the S-21-46 deep brain for the rest), and the image with the shape
tensor alone for four regions of S-21-46 at 20 µm (deep brain, cortex, brainstem, cerebellum), each block calibrated
on itself and all with the deep-brain block's calibration (one calibration per volume, as ``run_volume`` does).

    python scripts/scales.py                 -> docs/img/scales.png, docs/img/regions_20um.png, results/scales.json
    python scripts/scales.py --regions-only  -> the regions figure alone (the scales' entries of the JSON kept)

Every block is calibrated on the 384³ around it (tissue = grey level above zero) and run at σ 1, ρ 10 voxels, as
is the excised structure tensor it is compared with. The crop centres are zarr (z, y, x) indices at the level read;
the S-21-46 20 µm sites come from its 1 mm SynthSeg (deep brain: the thalamus crop of the other figures; cortex: a
gyral white-matter site half white matter, 40% cortex in 15 mm; brainstem and cerebellar white matter: the deepest
point of the label).
"""
import json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import RHO, SIGMA, dev                                                            # noqa: E402
from shapetensor import st                                                                     # noqa: E402
from shapetensor.pipeline import apply, calibrate                                             # noqa: E402
from shapetensor.validate import angle_deg                                                     # noqa: E402
from shapetensor.viz import dec_rgb, pipeline_row, shape_space                                 # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
RES = Path(__file__).resolve().parents[1] / "results"
S2146_20 = "/hdd/andrew/brain_atlas_20um_masked/S-21-46_brain_20um.ome.zarr"
S2146_50 = "/hdd/andrew/BRIDGE/50um_scale_S_21_46/sub-test/micr/sub-test_sample-brain_acq-50um_XPCT.ome.zarr"
I74_2 = "/hdd/andrew/I74_ChunkReg/omezarr/voi2_2.2um.ome.zarr"
I74_10 = "/hdd/andrew/I74_ChunkReg/omezarr/chunk_10um.ome.zarr"
# name: (path, level, centre zyx, voxel µm, row label)
SCALES = {
    "2.2um": (I74_2, "0", (6030, 4542, 4542), 2.2, "2.2 µm, I74 VOI"),
    "10um": (I74_10, "0", (2500, 1760, 1500), 10.0, "10 µm, I74 chunk"),
    "20um": (S2146_20, "0", (3833, 3865, 3022), 20.0, "20 µm, S-21-46 deep brain"),
    "50um": (S2146_50, "0", (1533, 1546, 1209), 50.0, "50 µm, S-21-46 deep brain"),
    "100um": (S2146_50, "1", (767, 773, 604), 100.0, "100 µm, S-21-46 deep brain"),
    "200um": (S2146_50, "2", (383, 387, 302), 200.0, "200 µm, S-21-46 deep brain"),
}
REGIONS = {
    "deep brain": (S2146_20, "0", (3833, 3865, 3022)),
    "cortex": (S2146_20, "0", (3300, 2600, 3950)),
    "brainstem": (S2146_20, "0", (3500, 5250, 3650)),
    "cerebellum": (S2146_20, "0", (2400, 5450, 2800)),
}
N, M = 192, 24


def load(path, level, centre, size=384):
    a = zarr.open_group(path, mode="r")[level]
    lo = [int(c - size // 2) for c in centre]
    assert all(l >= 0 and l + size <= s for l, s in zip(lo, a.shape)), (path, level, centre)
    return torch.as_tensor(np.asarray(a[lo[0]:lo[0] + size, lo[1]:lo[1] + size, lo[2]:lo[2] + size]).astype(np.float32), device=dev)


def block(path, level, centre, cal=None):
    """The standard measurements on one block: calibration on the 384³ (or ``cal``, a calibration from elsewhere),
    the pipeline and the excised tensor on the central 192³, the shape-space view, the agreement, the slices."""
    t0 = time.perf_counter()
    if cal is None:
        vc = load(path, level, centre); tc = vc > 0
        cal = calibrate(vc, tc, sigma=SIGMA, rho=RHO); o = (vc.shape[0] - N) // 2; v = vc[o:o + N, o:o + N, o:o + N].contiguous(); t = tc[o:o + N, o:o + N, o:o + N].contiguous(); del vc, tc
    else:
        v = load(path, level, centre, N); t = v > 0
    F = cal.features(v); out = apply(cal, v, t, features=F)
    d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, mask=t, order="zyx")
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO, mask=t); val, _ = st.eig_special_3d(S); gap_st = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S, val
    sel = torch.zeros_like(t); sel[M:-M, M:-M, M:-M] = t[M:-M, M:-M, M:-M]
    e = angle_deg(d_st, out["vec"]); conf = sel & (gap_st > gap_st[sel].quantile(0.75))
    space = shape_space(F, cal.dictionary, out["activation"], out["tau"], sel)[:3]; del F
    sl = (slice(M, N - M), N // 2, slice(M, N - M)); dsl = (slice(None),) + sl; zs, _, xs = sl
    A = out["activation"]; n = min(out["modes"].shape[0], 3)
    aref = float(A[:, zs, :, xs].float().reshape(-1)[::53].quantile(0.99).clamp_min(1e-6))
    torch.cuda.synchronize()
    stats = {"n_populations": int(out["modes"].shape[0]), "tau": round(out["tau"], 3), "readout_bins": cal.orientation_bins_occupied, "tissue_above_null": out["confidence"]["tissue_above_null"],
             "agreement_median_deg": round(float(e[sel].median()), 1), "agreement_where_tensor_confident_deg": round(float(e[conf].median()), 1),
             "structure_tensor_gap_median": round(float(gap_st[sel].median()), 3), "seconds": round(time.perf_counter() - t0, 1)}
    panels = {"image": v[sl].cpu().numpy(), "st": d_st[dsl].cpu().numpy(), "space": space, "acts": [A[k][sl].cpu().numpy() for k in range(n)], "aref": aref,
              "vec": out["vec"][dsl].cpu().numpy(), "vec1": out["vecs"][0][dsl].cpu().numpy()}
    del out, v, t, d_st, gap_st, A; torch.cuda.empty_cache()
    return stats, panels, cal


def main(regions_only=False):
    OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    rec, rows = {}, {}
    if regions_only and (RES / "scales.json").exists():
        rec = json.load(open(RES / "scales.json"))
    cal_deep = None
    for key, (path, level, centre, um, label) in SCALES.items():
        if regions_only and key != "20um":
            continue
        stats, panels, cal = block(path, level, centre); rec[key] = {"voxel_um": um, "source": path, "level": level, "centre_zyx": list(centre), **stats}; rows[key] = (label, panels)
        if key == "20um":
            cal_deep = cal
        print(key, json.dumps(stats), flush=True)
    # the regions: each block calibrated on itself, and every block with the deep-brain block's calibration (one calibration per volume, as run_volume)
    for key, (path, level, centre) in REGIONS.items():
        for mode, c in (("self", None), ("deep-brain calibration", cal_deep)):
            k = f"region: {key} ({mode})"
            if key == "deep brain":
                rec[k] = dict(rec["20um"]); rows[k] = ("deep brain", rows["20um"][1]); continue
            stats, panels, _ = block(path, level, centre, c); rec[k] = {"voxel_um": 20.0, "source": path, "level": level, "centre_zyx": list(centre), "calibration": mode, **stats}; rows[k] = (key, panels)
            print(k, json.dumps(stats), flush=True)
    json.dump(rec, open(RES / "scales.json", "w"), indent=1)
    # the regions at 20 µm: the image, the shape tensor calibrated on the block, the shape tensor with the deep-brain calibration
    fig, axs = plt.subplots(3, len(REGIONS), figsize=(4.4 * len(REGIONS), 13.6), dpi=100, constrained_layout=True)
    for j, key in enumerate(REGIONS):
        for i, mode in enumerate(("self", "deep-brain calibration")):
            label, p = rows[f"region: {key} ({mode})"]; r = rec[f"region: {key} ({mode})"]
            if i == 0:
                axs[0, j].imshow(p["image"], cmap="gray", origin="lower"); axs[0, j].set_title(f"{label}: S-21-46, 20 µm, 192³ block", fontsize=11)
            axs[1 + i, j].imshow(dec_rgb(p["vec"]), origin="lower", interpolation="nearest")
            axs[1 + i, j].set_title(f"{'calibrated on the block' if i == 0 else 'deep-brain calibration'}: {r['n_populations']} population{'s' if r['n_populations'] != 1 else ''}, "
                                    f"{r['agreement_median_deg']:g}° from the tensor,\n{r['readout_bins']}/24 orientation bins, {100 * r['tissue_above_null']:.0f}% of tissue above the null", fontsize=9.5)
    for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Four regions at 20 µm: the image and the shape tensor's single field (DEC), calibrated on the block itself or once on the deep brain", x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "regions_20um.png"); plt.close(fig); print("wrote regions_20um.png")
    if regions_only:
        return
    # the scales: one standard row per voxel size
    keys = list(SCALES); n_act = min(max(len(rows[k][1]["acts"]) for k in keys), 3); cols = 5 + n_act
    fig, axs = plt.subplots(len(keys), cols, figsize=(4.2 * cols, 4.7 * len(keys) + 0.4), dpi=100, squeeze=False, constrained_layout=True)
    for i, k in enumerate(keys):
        label, p = rows[k]; r = rec[k]
        pipeline_row(axs[i], p["image"], p["st"], p["space"], p["acts"], p["vec"], p["vec1"], aref=p["aref"], label=f"{label}: 192³ block, {0.192 * r['voxel_um']:g} mm",
                     extra={"single": f"({r['agreement_median_deg']:g}° from the tensor, {r['agreement_where_tensor_confident_deg']:g}° where it is confident)"})
    fig.suptitle("HiP-CT across voxel sizes, σ 1, ρ 10 voxels at every scale (every field as DEC: R = |x|, G = |y|, B = |z|)", x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "scales.png"); plt.close(fig); print("wrote scales.png")


if __name__ == "__main__":
    main(regions_only="--regions-only" in sys.argv)
