"""Real 20 um data (S-21-46 crop, internal capsule / thalamus): direction-encoded colour (DEC) maps of the structure
tensor, the excised tensor and the anatomix48 TCA direction; the shape-space feature RGB (first three PCs of the
layer-58 channels); and the angular difference of each alternative to the structure tensor, on three mid planes.
No ground truth: a consistency demonstration. GPU.

    python scripts/demo_real_dec.py [--core 256] [--rho 16] [--reduce 4] [--nyquist none]

Writes figures/tca_field/real_S2146_dec_<plane>.png, figures/tca_field/real_S2146_dec.json and the field store.
"""
import argparse, json, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as Fn
import zarr

from shapetensor import DATA_ROOT
from shapetensor.features import make_engine
from shapetensor.filters import gaussian
from shapetensor.readout import common, single
from shapetensor.tensor import excision_weights, gradients, k_dense

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def dec(d, w):
    """DEC RGB from (3, ...) directions in (z, y, x) order and a weight (...) in [0, 1]: R = |x|, G = |y|, B = |z|."""
    rgb = np.stack([np.abs(d[2]), np.abs(d[1]), np.abs(d[0])], -1)
    return np.clip(rgb * w[..., None], 0, 1)


def eig_dense(S6):
    """(6, ...) packed [zz, yy, xx, zy, zx, yx] tensor (k_dense order for one channel) -> e3 (3, ...) and gap."""
    Kp = S6.reshape(6, -1).T
    K = common.unpack(Kp, 3)
    e = single.eigen(K)
    return e["dir"].T.reshape((3,) + S6.shape[1:]), e["gap"].reshape(S6.shape[1:]), e["cl"].reshape(S6.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=256); ap.add_argument("--rho", type=float, default=16.0)
    ap.add_argument("--reduce", type=int, default=4); ap.add_argument("--nyquist", default="none"); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=256); ap.add_argument("--margin", type=int, default=64)   # tile 128 / margin 64 for a 12 GB card
    ap.add_argument("--n-pcs", type=int, default=8)   # 8 PCs read the same as 16 (results log §10); 16 at 520^3 needs 27 GB of channel gradients
    a = ap.parse_args()
    t0 = time.time()
    b, rho, tr = a.reduce, a.rho, 4.0
    m = int(np.ceil((tr * (a.sigma + rho) + 64) / b) * b)
    z0 = zarr.open(CROP, mode="r")["0"]; n = z0.shape[0]
    lo = (n - a.core) // 2 - m; hi = lo + a.core + 2 * m
    vol = np.asarray(z0[lo:hi, lo:hi, lo:hi]).astype(np.float32)
    v = torch.as_tensor(vol, device="cuda")
    Z, Y, X = v.shape
    core = ((m, Z - m), (m, Y - m), (m, X - m))
    print(f"block {vol.shape}, core {a.core}^3, margin {m}, read {time.time() - t0:.0f} s", flush=True)
    # 1. intensity: structure tensor and excised tensor on the site grid b (dense, 6 maps each)
    Gi = gradients(v[None], a.sigma, tr)
    S, sites = k_dense(Gi, rho, tr, b, core)
    d_st, gap_st, cl = eig_dense(S.permute(3, 0, 1, 2).contiguous()); del S
    w = excision_weights(v, 3.0, 3.0, tr)
    Sx, _ = k_dense(Gi, rho, tr, b, core, weights=w.half())
    d_ex, gap_ex, _ = eig_dense(Sx.permute(3, 0, 1, 2).contiguous()); del Sx, Gi
    gm = sum(gaussian(v, 3.0, order=o, truncate=tr) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    Kg, _ = k_dense(gm[None].half(), rho, tr, b, core); gm_s = Kg[..., 0].clamp_min(0).sqrt(); del Kg, gm, w
    print(f"intensity tensors done {time.time() - t0:.0f} s", flush=True)
    # 2. shape features: anatomix48 -> 16 PCs; feature RGB = PCs 1-3; TCA direction from K
    torch.cuda.empty_cache()
    eng = make_engine("anatomix48", nyquist=a.nyquist, tile=a.tile, margin=a.margin, n_pcs=a.n_pcs)
    F = eng(v)                                                           # (n_pcs, Z, Y, X) fp16, PCs in order
    rgb_full = F[:3].float()
    rgb_core = rgb_full[:, m:Z - m, m:Y - m, m:X - m].cpu().numpy()
    lo_p, hi_p = np.percentile(rgb_core, 1, axis=(1, 2, 3)), np.percentile(rgb_core, 99, axis=(1, 2, 3))
    feat_rgb = np.clip((rgb_core - lo_p[:, None, None, None]) / (hi_p - lo_p)[:, None, None, None], 0, 1).transpose(1, 2, 3, 0)
    del rgb_full
    G = gradients(F, a.sigma, tr); del F
    K, _ = k_dense(G, rho, tr, b, core); del G
    Kt = common.unpack(K.reshape(-1, K.shape[-1]), common.nc3_from_packed(K.shape[-1])); del K
    tc = single.tca(Kt, rank=3); del Kt
    shp = sites.shape
    d_tca = tc["dir"].T.reshape((3,) + shp); gap_tca = tc["gap"].reshape(shp)
    print(f"shape features and TCA done {time.time() - t0:.0f} s", flush=True)
    # 3. differences (site grid)
    ang = lambda u, w_: torch.rad2deg(torch.arccos((u * w_).sum(0).abs().clamp(max=1)))
    diff_ex, diff_tca = ang(d_st, d_ex), ang(d_st, d_tca)
    tissue = torch.as_tensor(vol[m:Z - m:b, m:Y - m:b, m:X - m:b] >= 142.5, device="cuda")       # bright tissue only
    summ = {"rho": rho, "reduce": b, "core": a.core, "block": [lo, hi],
            "median_angle_ST_vs_excised_bright": float(diff_ex[tissue].median()), "share_gt30_ST_vs_excised": float((diff_ex[tissue] > 30).float().mean()),
            "median_angle_ST_vs_tca_bright": float(diff_tca[tissue].median()), "share_gt30_ST_vs_tca": float((diff_tca[tissue] > 30).float().mean()),
            "median_gap_ST": float(gap_st[tissue].median()), "median_gap_excised": float(gap_ex[tissue].median()), "median_gap_tca": float(gap_tca[tissue].median()),
            "share_cl_gt_0.3": float((cl[tissue] > 0.3).float().mean())}
    # where do they disagree? against the boundary strength
    q = torch.quantile(gm_s[tissue], 0.9)
    summ["median_angle_ST_vs_excised_at_high_gm"] = float(diff_ex[tissue & (gm_s > q)].median())
    summ["median_angle_ST_vs_excised_at_low_gm"] = float(diff_ex[tissue & (gm_s <= q)].median())
    print(json.dumps(summ, indent=1), flush=True)
    # store
    store = zarr.open_group(str(Path(DATA_ROOT) / "fields" / f"S2146_dec_core{a.core}_rho{rho:g}.zarr"), mode="w")
    for name, arr in (("dir_st", d_st), ("dir_excised", d_ex), ("dir_tca48", d_tca), ("gap_st", gap_st), ("gap_excised", gap_ex), ("gap_tca48", gap_tca),
                      ("cl", cl), ("gm", gm_s), ("diff_st_excised_deg", diff_ex), ("diff_st_tca48_deg", diff_tca)):
        store.create_array(name, data=arr.cpu().numpy().astype(np.float16))
    store.create_array("feature_rgb", data=(feat_rgb * 255).astype(np.uint8))
    store.attrs.update({**summ, "site_origin_vox": [lo + m + b / 2] * 3, "voxel_mm": 0.02})
    # 4. figures: three mid planes
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    to_np = lambda t: t.cpu().numpy()
    D = {"st": (to_np(d_st), to_np(gap_st)), "ex": (to_np(d_ex), to_np(gap_ex)), "tca": (to_np(d_tca), to_np(gap_tca))}
    diffs = {"ex": to_np(diff_ex), "tca": to_np(diff_tca)}; gm_np = to_np(gm_s)
    img_core = vol[m:Z - m, m:Y - m, m:X - m]
    FIGS.mkdir(parents=True, exist_ok=True)
    for pname, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        mid = shp[axis] // 2
        sl = [slice(None)] * 3; sl[axis] = mid; sl = tuple(sl)
        isl = [slice(None)] * 3; isl[axis] = mid * b + b // 2; isl = tuple(isl)
        img = img_core[isl]
        fr = feat_rgb[isl]
        ext = (0, img.shape[1] * 0.02, 0, img.shape[0] * 0.02)
        fig, axs = plt.subplots(2, 4, figsize=(19, 10), dpi=100)
        axs[0, 0].imshow(img, cmap="gray", origin="lower", extent=ext); axs[0, 0].set_title("image (20 µm)", fontsize=11)
        axs[0, 1].imshow(fr, origin="lower", extent=ext, interpolation="nearest"); axs[0, 1].set_title("shape-space feature RGB (anatomix48 PCs 1–3)", fontsize=11)
        for ax, key, title in ((axs[0, 2], "st", "DEC: structure tensor (weight = orientation gap)"), (axs[0, 3], "ex", "DEC: excised tensor"),
                               (axs[1, 0], "tca", "DEC: anatomix48 TCA")):
            d, g = D[key]
            ax.imshow(dec(d[(slice(None),) + sl], np.clip(g[sl], 0, 1) ** 0.5), origin="lower", extent=ext, interpolation="nearest")
            ax.set_title(title, fontsize=11)
        for ax, key, title in ((axs[1, 1], "ex", "angle: excised vs structure tensor (deg)"), (axs[1, 2], "tca", "angle: anatomix48 TCA vs structure tensor (deg)")):
            im = ax.imshow(diffs[key][sl], cmap=cmap, vmin=0, vmax=90, origin="lower", extent=ext, interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); ax.set_title(title, fontsize=11)
        im = axs[1, 3].imshow(gm_np[sl], cmap=cmap, origin="lower", extent=ext, interpolation="nearest"); fig.colorbar(im, ax=axs[1, 3], fraction=0.04, pad=0.02)
        axs[1, 3].set_title("boundary strength: coarse gradient magnitude", fontsize=11)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"PROTOTYPE. S-21-46 20 µm crop, {a.core}³ core, {pname} mid plane, ρ {rho:g} vox ({rho * 0.02:.2f} mm), sites every {b} vox. "
                     f"DEC: R = |x|, G = |y|, B = |z|. Bright tissue: excised vs ST median {summ['median_angle_ST_vs_excised_bright']:.1f}°, TCA vs ST {summ['median_angle_ST_vs_tca_bright']:.1f}°. No ground truth.",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIGS / f"real_S2146_dec_{pname}.png"; fig.savefig(out); plt.close(fig); print("wrote", out, flush=True)
    json.dump(summ, open(FIGS / "real_S2146_dec.json", "w"), indent=1)
    print(f"DEC DEMO DONE {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
