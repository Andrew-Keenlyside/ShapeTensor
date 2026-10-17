"""The single-vector comparison, parameter-free on both sides: the structure tensor (σ 1, ρ 10, one eigenvector per
voxel) against the shape tensor's single field (the activation-weighted average of the population layers, no mask),
and both against the multi-vector field of the layers (both populations where their masks intersect, the strongest
activated population elsewhere) — on the 60° and 90° crossing phantoms (truth) and on the real HiP-CT block
(S-21-46, 192³; agreement), with DEC images of each field, and the speed of the two pipelines on 192³ and 256³ blocks.

    python scripts/bench/single_vs_multi.py [--no-speed]   -> figures/tca_field/single_vs_multi.png, results/single_vs_multi.json
"""
import argparse, json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as Fn
import zarr

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.phantoms import make
from shapetensor.populations import population_field
from shapetensor.readout.layers import (SignatureDictionary, activation_null, combined_field, combined_tensor_field, layer_tensor_direction,
                                        multi_vector_field, population_masks, unmix)
from shapetensor.readout.orientation import OrientationReadoutDist, modes_from_density
from shapetensor.viz import dec_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fig_population_layers import ANG, SIGMA, TR, build, teacher, unit  # noqa: E402

FIGS = Path(__file__).resolve().parents[2] / "figures" / "tca_field"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
RHO, BRIGHT = 10.0, 142.5


def ang(d, t):
    """Angle (deg) between direction fields ``d (3, ...)`` and ``t (3, ...)`` (axial)."""
    return torch.rad2deg(torch.arccos((d * t).sum(0).abs().clamp(max=1)))


def st_dirs(v):
    S = st.structure_tensor_3d(v, SIGMA, RHO, TR); val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def summary(e, sel):
    e = e[sel]
    return {"median_deg": round(float(e.median()), 1), "within_15": round(float((e <= 15).float().mean()), 3)}


def singles(A, masks):
    """The layers' tensors and the three single fields from them (the tensor merge, the direction average, the
    coherence-weighted average) and the multi-vector field."""
    lt = [layer_tensor_direction(A[k], SIGMA, RHO, TR, return_tensor=True) for k in range(A.shape[0])]
    dirs = torch.stack([l[0] for l in lt]); gaps = torch.stack([l[1] for l in lt]); tens = torch.stack([l[2] for l in lt]); del lt
    vec_c, _, _ = combined_tensor_field(tens, A); vec_a, _ = combined_field(dirs, A); vec_g, _ = combined_field(dirs, A, gaps=gaps)
    vec_m, _, cnt = multi_vector_field(dirs, A, masks)
    return dirs, gaps, tens, vec_c, vec_a, vec_g, vec_m, cnt


def timed(fn, n=3):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-speed", action="store_true"); a = ap.parse_args(); t0 = time.time()
    res = {"rho": RHO, "sigma": SIGMA, "phantom": {}, "real": {}, "speed": {}}
    panels = []                                                               # (title, image, ST dirs, single dirs, multi dirs, count) slices
    # ------------------------------------------------------------------ phantoms: dictionary, readout and null from single-population blocks
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36)); row = (slice(36, Z - 36), y_mid, slice(36, X - 36))
    eng = make_engine("anatomix_enc2", nyquist="none")
    blocks, bg = [], []
    for k, angd in enumerate(ANG):
        v, fib, _ = build([(angd, 1.0)], 100 + k, tex, g); F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float(); blocks.append((F, d, w))
        wb = torch.zeros_like(gap); wb[inner] = (gap[inner] < torch.quantile(gap[inner], 0.3)).float() * (~fib[inner]).float(); bg.append((F, wb))
    dic = SignatureDictionary(smooth=3).fit(blocks, background=bg); ro = OrientationReadoutDist(smooth=3).fit(blocks, n_sample=240_000)
    tau = activation_null(dic, blocks); del blocks, bg; torch.cuda.empty_cache()
    print(f"[{time.time() - t0:4.0f}s] phantom calibration done (tau {tau:.3f})", flush=True)
    for cross in (60.0, 90.0):
        a1, a2 = 30.0, 30.0 + cross
        v, fib, tmasks = build([(a1, 0.5), (a2, 0.5)], 300, tex, g); F = cached_features(eng, v)
        P = ro.predict_dist(F); sl = (slice(36, Z - 36), slice(y_mid - 4, y_mid + 5), slice(36, X - 36))
        dens = (P[(slice(None),) + sl] * fib[sl].float()[None]).reshape(P.shape[0], -1).sum(1); del P
        modes, h, n = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=3, mass_floor=0.1)
        t1, t2 = unit(a1), unit(a2)
        em = [[float(ang(modes[r][:, None], t[:, None])[0]) for t in (t1, t2)] for r in range(n)]
        order = sorted(range(n), key=lambda r: min(em[r]))[:2]
        A, R = unmix(F, dic, modes[order]); As = Fn.avg_pool3d(A[None], 3, stride=1, padding=1)[0]
        masks, union, strongest = population_masks(As, tau)
        dirs, gaps, tens, vec_c, vec_a, vec_g, vec_m, cnt = singles(A, masks)
        d_st, gap_st = st_dirs(v)
        # the truth per voxel: the population that drew the voxel; the two directions
        own = torch.where(tmasks[0][None], t1[:, None, None, None], t2[:, None, None, None])
        sel = torch.zeros_like(fib); sel[inner] = fib[inner]
        singles_d = {"shape single (tensor merge)": vec_c, "shape single (direction average)": vec_a, "shape single (coherence-weighted average)": vec_g}
        e_own = {"structure tensor": ang(d_st, own), **{k: ang(dd, own) for k, dd in singles_d.items()},
                 "multi (nearest of its vectors)": torch.minimum(ang(vec_m[0], own), torch.where(cnt >= 2, ang(vec_m[1], own), torch.full_like(own[0], 180.0)))}
        near = lambda dd: torch.minimum(ang(dd, t1[:, None, None, None]), ang(dd, t2[:, None, None, None]))
        e_near = {"structure tensor": near(d_st), **{k: near(dd) for k, dd in singles_d.items()}}
        two = (cnt >= 2) & sel
        pair = torch.minimum(ang(vec_m[0], t1[:, None, None, None]) + ang(vec_m[1], t2[:, None, None, None]), ang(vec_m[0], t2[:, None, None, None]) + ang(vec_m[1], t1[:, None, None, None])) / 2
        r = {"n_modes": int(n), "mode_err": [round(min(e), 1) for e in em], "two_vector_share_of_fibre_voxels": round(float(two.float().sum() / sel.float().sum()), 3),
             "error_to_owning_population": {k: summary(e, sel) for k, e in e_own.items()}, "error_to_nearest_population": {k: summary(e, sel) for k, e in e_near.items()},
             "multi_pair_error_where_two_vectors": summary(pair, two) if two.any() else None,
             "multi_single_vector_error_to_owning_where_one": summary(ang(vec_m[0], own), (cnt == 1) & sel),
             "mask_shares": [round(float((m & sel).float().sum() / sel.float().sum()), 3) for m in masks], "mask_intersection_share": round(float((masks[0] & masks[1] & sel).float().sum() / sel.float().sum()), 3)}
        res["phantom"][f"{cross:.0f}"] = r
        print(f"[{time.time() - t0:4.0f}s] {cross:.0f}°: {json.dumps(r)}", flush=True)
        panels.append((f"{cross:.0f}° crossing phantom", v[row].cpu().numpy(), d_st[(slice(None),) + row].cpu().numpy(), vec_c[(slice(None),) + row].cpu().numpy(),
                       vec_m[(slice(None), slice(None)) + row].cpu().numpy(), cnt[row].cpu().numpy(), (tmasks[0][row] | tmasks[1][row]).cpu().numpy()))
        del F, A, As, dirs, tens, vec_c, vec_a, vec_g, vec_m, d_st, singles_d; torch.cuda.empty_cache()
    # ------------------------------------------------------------------ the real block: the exported field (calibrated on the 384^3 crop) and the structure tensor
    z0 = zarr.open(CROP, mode="r")["0"]; n0 = z0.shape[0]; lo = (n0 - 384) // 2
    vc = torch.as_tensor(np.asarray(z0[lo:lo + 384, lo:lo + 384, lo:lo + 384]).astype(np.float32), device="cuda")
    o = (384 - 192) // 2; v = vc[o:o + 192, o:o + 192, o:o + 192].contiguous(); tis = v >= BRIGHT
    gz = zarr.open_group(str(Path(DATA_ROOT) / "fields" / "S2146_population_field_block192_nmax2_s3.zarr"), mode="r")
    A = torch.as_tensor(np.asarray(gz["activation"]).astype(np.float32), device="cuda"); masks = torch.as_tensor(np.asarray(gz["masks"]), device="cuda")
    dirs, gaps, tens, vec_c, vec_a, vec_g, vec_m, cnt = singles(A, masks); d_st, gap_st = st_dirs(v)
    m = 24; inner_r = (slice(m, 192 - m),) * 3; sel = torch.zeros_like(tis); sel[inner_r] = tis[inner_r]; two = (cnt >= 2) & sel
    r = {"structure_tensor_vs_shape_single_tensor_merge": summary(ang(d_st, vec_c), sel), "structure_tensor_vs_shape_single_direction_average": summary(ang(d_st, vec_a), sel),
         "structure_tensor_vs_shape_single_coherence_weighted": summary(ang(d_st, vec_g), sel), "tensor_merge_vs_direction_average": summary(ang(vec_c, vec_a), sel),
         "structure_tensor_vs_multi_strongest": summary(ang(d_st, vec_m[0]), sel),
         "structure_tensor_vs_nearest_multi_vector": summary(torch.minimum(ang(d_st, vec_m[0]), torch.where(cnt >= 2, ang(d_st, vec_m[1]), torch.full_like(gap_st, 180.0))), sel),
         "two_vector_share_of_tissue": round(float(two.float().sum() / sel.float().sum()), 3), "angle_between_the_two_vectors_where_two": summary(ang(vec_m[0], vec_m[1]), two),
         "where_structure_tensor_confident_gap_top_quarter": {"structure_tensor_vs_shape_single_tensor_merge": summary(ang(d_st, vec_c), sel & (gap_st > gap_st[sel].quantile(0.75))),
                                                              "structure_tensor_vs_nearest_multi_vector": summary(torch.minimum(ang(d_st, vec_m[0]), torch.where(cnt >= 2, ang(d_st, vec_m[1]), torch.full_like(gap_st, 180.0))), sel & (gap_st > gap_st[sel].quantile(0.75)))},
         "where_bundle_mask_population_2": {"structure_tensor_vs_shape_single_tensor_merge": summary(ang(d_st, vec_c), sel & masks[1]), "structure_tensor_vs_layer_2": summary(ang(d_st, dirs[1]), sel & masks[1])},
         "mask_shares": [round(float((mk & sel).float().sum() / sel.float().sum()), 3) for mk in masks], "structure_tensor_gap_median": round(float(gap_st[sel].median()), 3)}
    res["real"] = r; print(f"[{time.time() - t0:4.0f}s] real: {json.dumps(r)}", flush=True)
    rowr = (slice(m, 192 - m), 96, slice(m, 192 - m))
    panels.append(("S-21-46 block (192³)", v[rowr].cpu().numpy(), d_st[(slice(None),) + rowr].cpu().numpy(), vec_c[(slice(None),) + rowr].cpu().numpy(),
                   vec_m[(slice(None), slice(None)) + rowr].cpu().numpy(), cnt[rowr].cpu().numpy(), tis[rowr].cpu().numpy()))
    del A, dirs, tens, masks, vec_c, vec_a, vec_g, vec_m, d_st, gap_st; torch.cuda.empty_cache()
    # ------------------------------------------------------------------ figure
    fig, axs = plt.subplots(len(panels), 6, figsize=(34, 5.8 * len(panels)), dpi=90)
    for i, (title, im, dst, dsi, dm, c, tm) in enumerate(panels):
        axs[i, 0].imshow(im, cmap="gray", origin="lower"); axs[i, 0].set_title(f"{title}: image", fontsize=11)
        axs[i, 1].imshow(dec_rgb(dst), origin="lower", interpolation="nearest"); axs[i, 1].set_title("structure tensor: DEC of the eigenvector (σ 1, ρ 10)", fontsize=11)
        axs[i, 2].imshow(dec_rgb(dsi), origin="lower", interpolation="nearest"); axs[i, 2].set_title("shape tensor, single field: DEC of the merged population tensors' direction", fontsize=11)
        axs[i, 3].imshow(dec_rgb(dm[0]), origin="lower", interpolation="nearest"); axs[i, 3].set_title("multi-vector field: DEC of the strongest population's vector", fontsize=11)
        axs[i, 4].imshow(dec_rgb(dm[1]), origin="lower", interpolation="nearest"); axs[i, 4].set_title("multi-vector field: DEC of the second vector (where the masks intersect; black = none)", fontsize=11)
        axs[i, 5].imshow(im, cmap="gray", origin="lower", alpha=0.5); axs[i, 5].imshow(np.ma.masked_where(c < 2, c), cmap=ListedColormap(["#eda100"]), origin="lower", alpha=0.75, interpolation="nearest")
        axs[i, 5].set_title(f"voxels with two vectors (orange): {float((c >= 2)[tm].mean()):.0%} of the fibre / tissue voxels in this slice", fontsize=11)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("One vector per voxel, no parameters on either side: the structure tensor against the shape tensor's single field; and the layers' multi-vector field "
                 "(both populations where their masks intersect, the strongest elsewhere). DEC: R = |x|, G = |y|, B = |z|.", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGS / "single_vs_multi.png"); plt.close(fig); print("wrote", FIGS / "single_vs_multi.png", flush=True)
    # ------------------------------------------------------------------ speed: the structure tensor against the whole shape pipeline on 192^3 and 256^3 blocks
    if not a.no_speed:
        del eng, dic, ro; torch.cuda.empty_cache()
        for nb in (192, 256):
            o = (384 - nb) // 2; vb = vc[o:o + nb, o:o + nb, o:o + nb].contiguous(); tb = vb >= BRIGHT
            t_st = timed(lambda: st.eig_special_3d(st.structure_tensor_3d(vb, SIGMA, RHO, TR)))
            torch.cuda.reset_peak_memory_stats(); tm = {}
            out = population_field(vb, tissue=tb, timings=tm); peak = torch.cuda.max_memory_allocated() / 2 ** 30
            res["speed"][str(nb)] = {"structure_tensor_s": round(t_st, 3), "shape_pipeline_s": tm, "shape_pipeline_peak_gb": round(peak, 1), "n_populations": int(out["modes"].shape[0]),
                                     "ratio": round(tm["total"] / t_st, 1), "Mvox_per_s_structure": round(nb ** 3 / t_st / 1e6, 2), "Mvox_per_s_shape": round(nb ** 3 / tm["total"] / 1e6, 3)}
            print(f"[{time.time() - t0:4.0f}s] speed {nb}^3: {json.dumps(res['speed'][str(nb)])}", flush=True)
            del out; torch.cuda.empty_cache()
    json.dump(res, open(f"{DATA_ROOT}/results/single_vs_multi.json", "w"), indent=1)
    print("wrote", f"{DATA_ROOT}/results/single_vs_multi.json")


if __name__ == "__main__":
    main()
