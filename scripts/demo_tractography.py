"""Streamline tractography on the HiP-CT S-21-46 crop: the same deterministic tracker and the same seeds on four
fields -- the structure tensor (one vector), the excised tensor (one vector), the fixels of the intensity ODF and the
fixels of the anatomix48 shape-space ODF (up to three vectors) -- and the differences between the tractograms:
lengths, terminations, how often a secondary fixel was followed, streamline-density maps (Dice, correlation) and
the per-seed divergence from the structure-tensor track. No ground truth.

    python scripts/demo_tractography.py [--rho 10] [--core 256] [--reduce 2] [--seeds 20000]

Writes figures/tca_field/real_S2146_tracks_*.png and real_S2146_tracks.json; the fields and tracks to the data root.
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import zarr

from shapetensor import DATA_ROOT, tracking
from shapetensor.features import make_engine
from shapetensor.filters import gaussian
from shapetensor.odf import csd, peaks, ring_profile, zonal_kernel
from shapetensor.pca import PCA
from shapetensor.readout import common, single
from shapetensor.tensor import excision_weights, gradients, k_dense, odf_dense

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
VOX = 0.02
# CSD settings at the crop's texture level (results log §14): response widths from the t0.5 calibration, thresholds
# that give <= 2% false second populations on dispersed single-fibre texture
CSD = {"intensity": {"width": 29.0, "rel": 0.7}, "anatomix48": {"width": 30.0, "rel": 0.9}}


def eig_field(S6):
    """``(6, nz, ny, nx)`` packed tensor on the site grid -> unit directions ``(1, 3, nz, ny, nx)`` (z, y, x) and the gap."""
    Kp = S6.reshape(6, -1).T
    e = single.eigen(common.unpack(Kp, 3))
    shp = S6.shape[1:]
    return e["dir"].T.reshape((1, 3) + shp).contiguous(), e["gap"].reshape(shp)


def fixel_field(O, width_deg, rel, gate, nmax=3):
    """ODF coefficients ``(nz, ny, nx, ncoef)`` -> fixels ``(nmax, 3, nz, ny, nx)`` (z, y, x), zero where absent or gated."""
    shp = O.shape[:3]; lmax = 8
    kern = zonal_kernel(lmax, ring_profile(math.radians(width_deg)))
    f = csd(O.reshape(-1, O.shape[-1]), lmax, kern)
    d, a, cnt = peaks(f, lmax, nmax=nmax, rel=rel)
    dirs = d.reshape(shp + (nmax, 3)).permute(3, 4, 0, 1, 2).contiguous()
    dirs[:, :, gate] = 0.0
    npop = cnt.reshape(shp).clone(); npop[gate] = 0
    return dirs, npop


def dec_segments(ax, P, zsel, ext_scale, lw=0.6, alpha=0.8, max_lines=60000):
    """Streamline polylines projected on the (x, y) plane, coloured by the segment direction (R = |x|, G = |y|, B = |z|)."""
    segs, cols = [], []
    for half in (0, 1):
        pts = P[:, half]                                                     # (N, T, 3) with NaN past the end
        for i in range(pts.shape[0]):
            q = pts[i]; q = q[np.isfinite(q[:, 0])]
            if len(q) < 2 or not zsel(q):
                continue
            seg = np.stack([q[:-1], q[1:]], 1)                                # (n, 2, 3)
            d = seg[:, 1] - seg[:, 0]; d /= np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
            segs.append(seg[:, :, [2, 1]] * ext_scale)                       # (x, y) in mm
            cols.append(np.clip(np.stack([np.abs(d[:, 2]), np.abs(d[:, 1]), np.abs(d[:, 0])], 1), 0, 1))
            if sum(len(s) for s in segs) > max_lines:
                break
    if segs:
        ax.add_collection(LineCollection(np.concatenate(segs), colors=np.concatenate(cols), linewidths=lw, alpha=alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=10.0); ap.add_argument("--core", type=int, default=256); ap.add_argument("--reduce", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=1.0); ap.add_argument("--seeds", type=int, default=20000); ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--max-angle", type=float, default=15.0); ap.add_argument("--min-mm", type=float, default=0.2); ap.add_argument("--n-pcs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time(); rho, tr, b = a.rho, 4.0, a.reduce
    m = int(math.ceil((tr * (a.sigma + rho) + 64) / b) * b)
    z0 = zarr.open(CROP, mode="r")["0"]; n = z0.shape[0]
    lo = (n - a.core) // 2 - m; hi = lo + a.core + 2 * m
    vol = np.asarray(z0[lo:hi, lo:hi, lo:hi]).astype(np.float32)
    v = torch.as_tensor(vol, device="cuda"); Z, Y, X = v.shape
    core = ((m, Z - m), (m, Y - m), (m, X - m))
    print(f"block {vol.shape}, core {a.core}^3, margin {m}, sites every {b} vox; read {time.time() - t0:.0f} s", flush=True)
    # --- intensity: tensors and ODF on the site grid
    Gi = gradients(v[None], a.sigma, tr)
    S, sites = k_dense(Gi, rho, tr, b, core); shp = sites.shape
    f_st, gap_st = eig_field(S.permute(3, 0, 1, 2).contiguous()); del S
    w = excision_weights(v, 3.0, 3.0, tr)
    Sx, _ = k_dense(Gi, rho, tr, b, core, weights=w.half()); f_ex, gap_ex = eig_field(Sx.permute(3, 0, 1, 2).contiguous()); del Sx, w
    O, _ = odf_dense(Gi, rho, tr, b, core, lmax=8); del Gi
    gm = sum(gaussian(v, 3.0, order=o, truncate=tr) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    Kg, _ = k_dense(gm[None].half(), rho, tr, b, core); gm_s = Kg[..., 0].clamp_min(0).sqrt(); del Kg, gm
    tissue = torch.as_tensor(vol[m:Z - m:b, m:Y - m:b, m:X - m:b] >= 142.5, device="cuda")
    gate = gm_s > 3.0 * gm_s[tissue].median()                                 # the interface exclusion of the fixels
    f_ifx, n_ifx = fixel_field(O, CSD["intensity"]["width"], CSD["intensity"]["rel"], gate); del O
    print(f"intensity fields {time.time() - t0:.0f} s", flush=True)
    # --- anatomix48 (8 PCs): ODF of the feature gradients
    torch.cuda.empty_cache()
    F = make_engine("anatomix48", nyquist="none", n_pcs=a.n_pcs)(v)
    G = gradients(F, a.sigma, tr); del F
    O, _ = odf_dense(G, rho, tr, b, core, lmax=8); del G
    f_afx, n_afx = fixel_field(O, CSD["anatomix48"]["width"], CSD["anatomix48"]["rel"], gate); del O
    torch.cuda.empty_cache()
    print(f"anatomix48 fixels {time.time() - t0:.0f} s", flush=True)
    fields = {"structure tensor": f_st, "excised tensor": f_ex, "intensity fixels": f_ifx, "anatomix48 fixels": f_afx}
    # --- seeds: tissue sites away from the border, the same for every field
    g = torch.Generator(device="cuda").manual_seed(0)
    cand = tissue.clone(); cand[:4] = cand[-4:] = False; cand[:, :4] = cand[:, -4:] = False; cand[:, :, :4] = cand[:, :, -4:] = False
    idx = cand.nonzero()
    sel = idx[torch.randperm(idx.shape[0], generator=g, device="cuda")[:a.seeds]].float()
    seeds = sel + torch.rand(sel.shape, generator=g, device="cuda") - 0.5
    site_mm = b * VOX; min_steps = int(round(a.min_mm / (a.step * site_mm)))
    tracks, summ = {}, {"rho": rho, "core": a.core, "reduce": b, "site_mm": site_mm, "seeds": int(seeds.shape[0]), "step_sites": a.step, "max_angle_deg": a.max_angle,
                        "min_mm": a.min_mm, "csd": CSD, "share_gated_sites": float(gate[tissue].float().mean()),
                        "npop_share_intensity": [float((n_ifx[tissue] == k).float().mean()) for k in range(4)],
                        "npop_share_anatomix48": [float((n_afx[tissue] == k).float().mean()) for k in range(4)]}
    for name, D in fields.items():
        r = tracking.track(D, tissue, seeds, step=a.step, max_angle_deg=a.max_angle, max_steps=1000)
        L = r["length"] * site_mm
        keep = L >= a.min_mm
        reasons = {tracking.REASONS[k]: float((r["reason"] == k).float().mean()) for k in range(len(tracking.REASONS))}
        sec = (r["used_secondary"].sum(1).float() / r["n_steps"].sum(1).clamp_min(1).float())
        den = tracking.density(r["points"][keep], shp)
        tracks[name] = {"r": r, "keep": keep, "den": den}
        summ[name] = {"kept_share": float(keep.float().mean()), "length_mm_median": float(L[keep].median()) if keep.any() else 0.0,
                      "length_mm_p90": float(L[keep].quantile(0.9)) if keep.any() else 0.0, "terminations": reasons,
                      "secondary_fixel_step_share": float(sec[keep].mean()) if keep.any() else 0.0,
                      "streamlines_with_secondary": float((r["used_secondary"].sum(1)[keep] > 0).float().mean()) if keep.any() else 0.0,
                      "coverage_share_of_tissue": float((den[tissue] > 0).float().mean())}
        print(f"[{time.time() - t0:5.0f}s] {name:18s}: kept {summ[name]['kept_share']:.0%}, median {summ[name]['length_mm_median']:.2f} mm, p90 {summ[name]['length_mm_p90']:.2f} mm, "
              f"coverage {summ[name]['coverage_share_of_tissue']:.0%}, secondary {summ[name]['streamlines_with_secondary']:.0%}, stops {reasons}", flush=True)
    ref = tracks["structure tensor"]
    for name, tk in tracks.items():
        if name == "structure tensor":
            continue
        both = ref["keep"] & tk["keep"]
        div = tracking.endpoint_divergence(ref["r"], tk["r"], a.step) * site_mm
        la, lb = torch.log1p(ref["den"]), torch.log1p(tk["den"])
        corr = float(torch.corrcoef(torch.stack([la[tissue], lb[tissue]]))[0, 1])
        summ[f"{name} vs structure tensor"] = {"density_dice": tracking.dice(ref["den"], tk["den"]), "density_corr_log": corr,
                                               "divergence_mm_median": float(div[both].median()) if both.any() else 0.0,
                                               "divergence_gt_0.2mm_share": float((div[both] > 0.2).float().mean()) if both.any() else 0.0,
                                               "kept_in_both": float(both.float().mean())}
        print(f"   {name} vs ST: Dice {summ[f'{name} vs structure tensor']['density_dice']:.2f}, corr {corr:.2f}, divergence median {summ[f'{name} vs structure tensor']['divergence_mm_median']:.3f} mm, "
              f"> 0.2 mm {summ[f'{name} vs structure tensor']['divergence_gt_0.2mm_share']:.0%}", flush=True)
    print(json.dumps(summ, indent=1), flush=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(FIGS / f"real_S2146_tracks_rho{rho:g}.json", "w"), indent=1)
    # store the fields and the tracks
    store = zarr.open_group(str(Path(DATA_ROOT) / "fields" / f"S2146_tracks_core{a.core}_rho{rho:g}.zarr"), mode="w")
    for name, D in fields.items():
        store.create_array(name.replace(" ", "_"), data=D.cpu().numpy().astype(np.float16))
    store.create_array("tissue", data=tissue.cpu().numpy()); store.create_array("gate", data=gate.cpu().numpy()); store.create_array("seeds", data=seeds.cpu().numpy())
    for name, tk in tracks.items():
        store.create_array("tracks_" + name.replace(" ", "_"), data=tk["r"]["points"].cpu().numpy().astype(np.float16))
        store.create_array("density_" + name.replace(" ", "_"), data=tk["den"].cpu().numpy().astype(np.float32))
    store.attrs.update({k: v for k, v in summ.items() if not isinstance(v, dict)})
    # --- figures: the tracks through an axial slab, density maps, lengths and terminations
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    zmid = shp[0] // 2; img = vol[m + zmid * b + b // 2, m:Y - m, m:X - m]
    ext = (0, img.shape[1] * VOX, 0, img.shape[0] * VOX)
    fig, axs = plt.subplots(2, 4, figsize=(22, 11), dpi=110)
    for ax, (name, tk) in zip(axs[0], tracks.items()):
        ax.imshow(img, cmap="gray", origin="lower", extent=ext, alpha=0.7)
        P = tk["r"]["points"][tk["keep"]].cpu().numpy()
        dec_segments(ax, P, lambda q: np.abs(q[:, 0] - zmid).min() <= 2.0, site_mm)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        s = summ[name]; ax.set_title(f"{name}: {s['kept_share']:.0%} seeds kept, median {s['length_mm_median']:.2f} mm, p90 {s['length_mm_p90']:.2f} mm", fontsize=10)
    for ax, name in zip(axs[1, :3], ("structure tensor", "intensity fixels", "anatomix48 fixels")):
        im = ax.imshow(np.log1p(tracks[name]["den"][zmid].cpu().numpy()), cmap=cmap, origin="lower", extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); ax.set_title(f"log streamline density: {name}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1, 3]
    for (name, tk), col in zip(tracks.items(), CAT):
        L = (tk["r"]["length"] * site_mm)[tk["keep"]].cpu().numpy()
        ax.hist(L, bins=40, range=(0, max(1.0, float(np.percentile(L, 99)) if len(L) else 1.0)), histtype="step", color=col, label=name, lw=1.5)
    ax.set_xlabel("streamline length (mm)"); ax.set_ylabel("streamlines"); ax.legend(fontsize=9); ax.set_title("lengths (same seeds)", fontsize=10)
    fig.suptitle(f"PROTOTYPE. S-21-46 20 µm crop, {a.core}³ core, ρ {rho:g} vox, sites every {b} vox ({site_mm * 1000:.0f} µm); {seeds.shape[0]} seeds, step {a.step * site_mm * 1000:.0f} µm, "
                 f"max {a.max_angle:g}° per step, tracks through the axial mid-slab (±{2 * site_mm:.2f} mm), coloured by direction. No ground truth.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / f"real_S2146_tracks_rho{rho:g}.png"); plt.close(fig)
    # the comparison figure: where the fixel tractograms depart from the structure tensor's
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=110)
    for ax, name in zip(axs, ("excised tensor", "intensity fixels", "anatomix48 fixels")):
        d = (np.log1p(tracks[name]["den"][zmid].cpu().numpy()) - np.log1p(ref["den"][zmid].cpu().numpy()))
        im = ax.imshow(d, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower", extent=ext, interpolation="nearest"); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        s = summ[f"{name} vs structure tensor"]
        ax.set_title(f"{name} − structure tensor (log density)\nDice {s['density_dice']:.2f}, corr {s['density_corr_log']:.2f}, divergence median {s['divergence_mm_median']:.2f} mm", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGS / f"real_S2146_tracks_rho{rho:g}_diff.png"); plt.close(fig)
    print(f"TRACTOGRAPHY DONE {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
