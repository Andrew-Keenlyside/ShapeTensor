"""The BigBrain 20 um VOI (cell-body stain; white matter pale, grey matter dark, background 255): the structure
tensor, the excised tensor and the anatomix48 shape tensor at 20 um isotropic, and the same block resampled to
20 x 20 x 50 um along the section axis (y) and read naively (voxel-unit filters), with ``spacing`` and after
resampling back to isotropic -- each compared with the isotropic reference. No ground truth: a consistency test.

    python scripts/demo_bigbrain.py [--rho 10] [--block 640] [--shape-core 256]

Writes figures/bigbrain/bigbrain_rho<rho>_{iso,aniso}_<plane>.png and figures/bigbrain/bigbrain_rho<rho>.json.
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch, torch.nn.functional as Fn

from shapetensor import DATA_ROOT, st
from shapetensor.filters import gaussian

VOI = Path(DATA_ROOT) / "bigbrain" / "bigbrain_20um_x2336_y3936_z2464_768.npy"
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"            # the HiP-CT crop, as a second real-data case
FIGS_ROOT = Path(__file__).resolve().parents[1] / "figures"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
EK = {"nyquist": "none"}
RATIO = 2.5


def dec(vec, w):
    """DEC RGB from ``vec (3, ...)`` in (x, y, z) order and a weight in [0, 1]."""
    rgb = np.stack([np.abs(vec[0]), np.abs(vec[1]), np.abs(vec[2])], -1)
    return np.clip(rgb * w[..., None], 0, 1)


def angle(u, w):
    return torch.rad2deg(torch.arccos((u * w).sum(0).abs().clamp(max=1)))


def gap_of(val):
    return ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def coarsen(v, axis, ratio):
    size = list(v.shape); size[axis] = int(round(size[axis] / ratio))
    return Fn.interpolate(v[None, None], size=size, mode="area")[0, 0]


def refine(v, axis, n):
    size = list(v.shape); size[axis] = n
    return Fn.interpolate(v[None, None], size=size, mode="trilinear", align_corners=False)[0, 0]


def to_coarse_grid(t, axis, n):
    """Nearest sampling of ``t (..., Z, Y, X)`` from the fine grid onto the grid coarsened along ``axis`` to ``n``."""
    size = list(t.shape[-3:]); size[axis] = n
    return Fn.interpolate(t.reshape(-1, 1, *t.shape[-3:]).float(), size=size, mode="nearest").reshape(*t.shape[:-3], *size)


def stats(diff, tissue, gm_s, name):
    q = torch.quantile(gm_s[tissue], 0.9)
    return {f"{name}: median_deg": float(diff[tissue].median()), f"{name}: share_gt30": float((diff[tissue] > 30).float().mean()),
            f"{name}: median_deg_low_gm": float(diff[tissue & (gm_s <= q)].median()), f"{name}: median_deg_high_gm": float(diff[tissue & (gm_s > q)].median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=10.0); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=640); ap.add_argument("--shape-core", type=int, default=256); ap.add_argument("--reduce", type=int, default=4)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--no-shape", action="store_true", help="skip the anatomix48 arms (a dry run of the tensor parts)")
    ap.add_argument("--source", default="bigbrain", choices=["bigbrain", "s2146"], help="the BigBrain VOI (cell stain, background 255) or the HiP-CT S-21-46 crop")
    a = ap.parse_args()
    t0 = time.time(); rho, tr, b = a.rho, 4.0, a.reduce
    if a.source == "bigbrain":
        full = np.load(VOI, mmap_mode="r"); n = full.shape[0]
    else:
        import zarr
        full = zarr.open(CROP, mode="r")["0"]; n = full.shape[0]
    lo = (n - a.block) // 2; hi = lo + a.block
    dev = a.device
    v = torch.as_tensor(np.asarray(full[lo:hi, lo:hi, lo:hi]).astype(np.float32), device=dev)
    Z, Y, X = v.shape
    tissue_full = (v < 250) if a.source == "bigbrain" else (v >= 142.5)                # BigBrain: background 255; HiP-CT: bright tissue
    FIGS = FIGS_ROOT / ("bigbrain" if a.source == "bigbrain" else "tca_field"); tag = "bigbrain" if a.source == "bigbrain" else "real_S2146"
    vmin, vmax = (100, 255) if a.source == "bigbrain" else (None, None)
    m = int(tr * (a.sigma + rho) + 0.5) + 16                                       # the filters' own margin
    core = (slice(m, Z - m), slice(m, Y - m), slice(m, X - m))
    print(f"block {tuple(v.shape)}, rho {rho:g}, core margin {m}, read {time.time() - t0:.0f} s", flush=True)

    def site(t):                                                                   # core, every b voxels
        return t[(slice(None),) * (t.ndim - 3) + tuple(slice(c.start, c.stop, b) for c in core)]
    summ = {"rho": rho, "block": a.block, "reduce": b, "voxel_um": [21.17, 20.0, 21.17], "aniso_ratio": RATIO, "aniso_axis": "y (section axis)"}
    # ---- isotropic reference: plain, excised, shape (shape on a central core with the network's margin)
    S = st.structure_tensor_3d(v, a.sigma, rho, tr, device=dev); val, vec = st.eig_special_3d(site(S), device=dev); del S
    gap_st = gap_of(val); vec_st = vec
    S = st.excised_structure_tensor_3d(v, a.sigma, rho, tr, mask=tissue_full, device=dev); val, vec = st.eig_special_3d(site(S), device=dev); del S
    gap_ex = gap_of(val); vec_ex = vec
    gm = sum(gaussian(v, 3.0, order=o, truncate=tr) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    gm_s = site(gaussian(gm, rho, truncate=tr)); del gm
    tissue = site(tissue_full)
    print(f"intensity tensors {time.time() - t0:.0f} s", flush=True)
    # shape tensor on the central shape-core (+ 64 network margin + filter margin), read on the same site grid
    cs = a.shape_core; ms = m + 64
    c0 = (Z - cs) // 2 - ms; c1 = c0 + cs + 2 * ms
    vs = v[c0:c1, c0:c1, c0:c1]
    vec_sh = (torch.zeros((3,) + tuple(vs.shape), device=dev) if a.no_shape else
              st.shapetensor(vs, sigma=a.sigma, rho=rho, truncate=tr, engine="anatomix48", n_pcs=8, engine_kwargs=EK, excise=False, device=dev))
    # the shape-core's site grid in the block's site coordinates
    sh_core = slice(c0 + ms, c1 - ms)
    sh_sites = [slice(((c0 + ms) - c.start + b - 1) // b, ((c1 - ms) - c.start + b - 1) // b) for c in core]
    vec_sh_s = vec_sh[:, ms:-ms:b, ms:-ms:b, ms:-ms:b]
    vec_sh_s = vec_sh_s[:, :sh_sites[0].stop - sh_sites[0].start, :sh_sites[1].stop - sh_sites[1].start, :sh_sites[2].stop - sh_sites[2].start]
    sl_sh = (slice(None),) + tuple(sh_sites)
    sl_sh2 = tuple(sh_sites)
    diff_ex = angle(vec_st, vec_ex)
    diff_sh = angle(vec_st[sl_sh][:, :vec_sh_s.shape[1], :vec_sh_s.shape[2], :vec_sh_s.shape[3]], vec_sh_s)
    tis_sh = tissue[sl_sh2][:vec_sh_s.shape[1], :vec_sh_s.shape[2], :vec_sh_s.shape[3]]
    gm_sh = gm_s[sl_sh2][:vec_sh_s.shape[1], :vec_sh_s.shape[2], :vec_sh_s.shape[3]]
    summ.update(stats(diff_ex, tissue, gm_s, "iso excised vs ST")); summ.update(stats(diff_sh, tis_sh, gm_sh, "iso anatomix48 vs ST"))
    summ["iso median_gap_ST"] = float(gap_st[tissue].median()); summ["iso median_gap_excised"] = float(gap_ex[tissue].median())
    del vs, vec_sh; torch.cuda.empty_cache()
    print(f"shape tensor {time.time() - t0:.0f} s", flush=True)
    # ---- anisotropic: 20 x 20 x 50 um along y
    va = coarsen(v, 1, RATIO); Ya = va.shape[1]
    sp = (1.0, RATIO, 1.0)
    core_a = (core[0], slice(int(round(m / RATIO)), Ya - int(round(m / RATIO))), core[2])
    def site_a(t):
        return t[(slice(None),) * (t.ndim - 3) + (slice(core_a[0].start, core_a[0].stop, b), slice(core_a[1].start, core_a[1].stop, max(1, int(round(b / RATIO)))), slice(core_a[2].start, core_a[2].stop, b))]
    # the isotropic reference directions and masks sampled onto the anisotropic site grid
    def ref_on(t_fine):
        ta = to_coarse_grid(t_fine if t_fine.ndim == 4 else t_fine[None], 1, Ya)
        return site_a(ta) if t_fine.ndim == 4 else site_a(ta)[0]
    S = st.structure_tensor_3d(v, a.sigma, rho, tr, device=dev); vec_ref = st.eig_special_3d(S, device=dev)[1]; del S
    vec_ref_a = ref_on(vec_ref); del vec_ref
    tissue_a = ref_on(tissue_full.float()) > 0.5
    gm_a = ref_on(gaussian(sum(gaussian(v, 3.0, order=o, truncate=tr) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt(), rho, truncate=tr))
    torch.cuda.empty_cache()
    vm = gaussian(va, (RATIO / math.sqrt(12), 0.0, RATIO / math.sqrt(12)), truncate=tr)      # PSF matched to the y box average
    arms = {
        "naive ST": lambda: st.structure_tensor_3d(va, a.sigma, rho, tr, device=dev),
        "psf-matched ST": lambda: st.structure_tensor_3d(vm, a.sigma, rho, tr, spacing=sp, device=dev),
        "spacing ST": lambda: st.structure_tensor_3d(va, a.sigma, rho, tr, spacing=sp, device=dev),
        "spacing excised": lambda: st.excised_structure_tensor_3d(va, a.sigma, rho, tr, mask=(va < 250) if a.source == "bigbrain" else (va >= 142.5), spacing=sp, device=dev),
    }
    vec_a = {}
    for name, fn in arms.items():
        S = fn(); vec_a[name] = st.eig_special_3d(site_a(S), device=dev)[1]; del S; torch.cuda.empty_cache()
    vr = refine(va, 1, Y)
    S = st.structure_tensor_3d(vr, a.sigma, rho, tr, device=dev); vec_r = st.eig_special_3d(S, device=dev)[1]; del S, vr
    vec_a["resampled ST"] = ref_on(vec_r); del vec_r; torch.cuda.empty_cache()
    # the shape tensor on the anisotropic shape-core, naive and with spacing (resampled to isotropic inside)
    ca0 = int(round(c0 / RATIO)); ca1 = int(round(c1 / RATIO))
    vsa = va[c0:c1, ca0:ca1, c0:c1]
    for name, kw in (("naive anatomix48", {}), ("spacing anatomix48", {"spacing": sp})):
        vec_sa = torch.zeros((3,) + tuple(vsa.shape), device=dev) if a.no_shape else st.shapetensor(vsa, sigma=a.sigma, rho=rho, truncate=tr, engine="anatomix48", n_pcs=8, engine_kwargs=EK, excise=False, device=dev, **kw)
        msa = int(round(ms / RATIO)); bsa = max(1, int(round(b / RATIO)))
        vec_sa_s = vec_sa[:, ms:-ms:b, msa:-msa:bsa, ms:-ms:b]
        ref_s = vec_ref_a[(slice(None),) + tuple(sh_sites[0:1]) + (slice(None),) + tuple(sh_sites[2:3])]
        # align the y range: the shape-core's y sites inside the block's anisotropic site grid
        ya0 = (ca0 + msa - core_a[1].start + bsa - 1) // bsa
        ref_s = ref_s[:, :, ya0:ya0 + vec_sa_s.shape[2]]
        nz, ny, nx = [min(p, q) for p, q in zip(ref_s.shape[1:], vec_sa_s.shape[1:])]
        d = angle(ref_s[:, :nz, :ny, :nx], vec_sa_s[:, :nz, :ny, :nx])
        tis = tissue_a[sh_sites[0], ya0:ya0 + ny, sh_sites[2]][:nz, :ny, :nx]; g = gm_a[sh_sites[0], ya0:ya0 + ny, sh_sites[2]][:nz, :ny, :nx]
        summ.update(stats(d, tis, g, f"aniso {name} vs iso ST")); del vec_sa; torch.cuda.empty_cache()
    diffs = {name: angle(vec_ref_a, vec_a[name]) for name in vec_a}
    for name, d in diffs.items():
        summ.update(stats(d, tissue_a, gm_a, f"aniso {name} vs iso ST"))
    print(json.dumps(summ, indent=1), flush=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    summ["source"] = a.source
    json.dump(summ, open(FIGS / f"{tag}_rho{rho:g}_aniso.json", "w"), indent=1)
    # ---- figures: isotropic (axial z-const and coronal y-const planes), anisotropic (the same planes on the coarse grid)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    npx = lambda t: t.cpu().numpy()
    img_core = npx(v[core])
    w_st, w_ex = npx(gap_st) ** 0.5, npx(gap_ex) ** 0.5
    for pname, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        mid = vec_st.shape[axis + 1] // 2
        sl = [slice(None)] * 3; sl[axis] = mid; sl = tuple(sl)
        isl = [slice(None)] * 3; isl[axis] = mid * b + b // 2; isl = tuple(isl)
        ext = (0, img_core[isl].shape[1] * 0.02, 0, img_core[isl].shape[0] * 0.02)
        fig, axs = plt.subplots(2, 3, figsize=(16, 10), dpi=100)
        axs[0, 0].imshow(img_core[isl], cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=ext); axs[0, 0].set_title("BigBrain 20 µm (cell-body stain)" if a.source == "bigbrain" else "S-21-46 HiP-CT 20 µm")
        axs[0, 1].imshow(dec(npx(vec_st)[(slice(None),) + sl], w_st[sl]), origin="lower", extent=ext, interpolation="nearest"); axs[0, 1].set_title("DEC: structure tensor (weight = gap)")
        axs[0, 2].imshow(dec(npx(vec_ex)[(slice(None),) + sl], w_ex[sl]), origin="lower", extent=ext, interpolation="nearest"); axs[0, 2].set_title("DEC: excised tensor")
        im = axs[1, 0].imshow(npx(diff_ex)[sl], cmap=cmap, vmin=0, vmax=90, origin="lower", extent=ext, interpolation="nearest"); fig.colorbar(im, ax=axs[1, 0], fraction=0.04); axs[1, 0].set_title("angle: excised vs structure tensor (deg)")
        im = axs[1, 1].imshow(npx(gm_s)[sl], cmap=cmap, origin="lower", extent=ext, interpolation="nearest"); fig.colorbar(im, ax=axs[1, 1], fraction=0.04); axs[1, 1].set_title("boundary strength: coarse gradient magnitude")
        # the shape tensor's DEC on its own (smaller) core, placed on the same axes
        d_sh = npx(vec_sh_s); sh_mid = d_sh.shape[axis + 1] // 2
        ssl = [slice(None)] * 3; ssl[axis] = sh_mid; ssl = tuple(ssl)
        off = [(s.start) * b * 0.02 for s in sh_sites]; keep = [k for k in range(3) if k != axis]
        ext_sh = (off[keep[1]], off[keep[1]] + d_sh.shape[keep[1] + 1] * b * 0.02, off[keep[0]], off[keep[0]] + d_sh.shape[keep[0] + 1] * b * 0.02)
        axs[1, 2].imshow(img_core[isl], cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=ext, alpha=0.4)
        axs[1, 2].imshow(dec(d_sh[(slice(None),) + ssl], np.ones(d_sh[0][ssl].shape)), origin="lower", extent=ext_sh, interpolation="nearest"); axs[1, 2].set_title("DEC: anatomix48 shape tensor (central core)")
        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{tag} 20 µm, {a.block}³ block, {pname} mid plane, ρ {rho:g} vox: excised vs ST median {summ['iso excised vs ST: median_deg']:.1f}°, "
                     f"anatomix48 vs ST {summ['iso anatomix48 vs ST: median_deg']:.1f}°. No ground truth.", x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / f"{tag}_rho{rho:g}_iso_{pname}.png"); plt.close(fig)
        # anisotropic comparison on the coarse grid (axial plane through the middle; coronal is the coarse axis)
        if axis == 1:
            continue
        mid_a = vec_ref_a.shape[axis + 1] // 2
        sla = [slice(None)] * 3; sla[axis] = mid_a; sla = tuple(sla)
        keep = [k for k in range(3) if k != axis]
        # extent in mm of the anisotropic core along the two kept axes (y counts 2.5 fine voxels per coarse voxel)
        span = lambda k: (core_a[k].stop - core_a[k].start) * (RATIO if k == 1 else 1.0) * 0.02
        exta = (0, span(keep[1]), 0, span(keep[0]))
        fig, axs = plt.subplots(2, 5, figsize=(25, 10), dpi=100)
        panels = [("isotropic reference (ST)", vec_ref_a)] + [(k, vec_a[k]) for k in ("naive ST", "spacing ST", "psf-matched ST", "resampled ST")]
        for ax, (name, vv) in zip(axs[0], panels):
            ax.imshow(dec(npx(vv)[(slice(None),) + sla], np.ones(vv[0][sla].shape)), origin="lower", extent=exta, interpolation="nearest", aspect="auto"); ax.set_title(f"DEC: {name}")
        img_a = npx(va)[core_a]
        pl = [slice(None)] * 3; pl[axis] = mid_a * b + b // 2
        axs[1, 0].imshow(img_a[tuple(pl)], cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=exta, aspect="auto"); axs[1, 0].set_title("20 × 20 × 50 µm volume (area-resampled along y)")
        for ax, name in zip(axs[1, 1:], ("naive ST", "spacing ST", "psf-matched ST", "resampled ST")):
            im = ax.imshow(npx(diffs[name])[sla], cmap=cmap, vmin=0, vmax=90, origin="lower", extent=exta, interpolation="nearest", aspect="auto"); fig.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title(f"angle to the isotropic reference: {name} ({summ[f'aniso {name} vs iso ST: median_deg']:.1f}° median)")
        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{tag} resampled to 20 × 20 × 50 µm (coarse axis y = the section axis), {pname} plane, ρ {rho:g}: naive {summ['aniso naive ST vs iso ST: median_deg']:.1f}°, "
                     f"spacing {summ['aniso spacing ST vs iso ST: median_deg']:.1f}°, PSF-matched {summ['aniso psf-matched ST vs iso ST: median_deg']:.1f}°, resampled {summ['aniso resampled ST vs iso ST: median_deg']:.1f}°, anatomix48 spacing {summ['aniso spacing anatomix48 vs iso ST: median_deg']:.1f}° (median angle to the isotropic ST over tissue)",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / f"{tag}_rho{rho:g}_aniso_{pname}.png"); plt.close(fig)
    print(f"BIGBRAIN DEMO DONE {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
