"""The stage-2 prototype on real data: a slab of the S-21-46 20 µm cortical VOI (gyral white matter folded inside
cortex), the current populations against the windowed fibre sets, with the excised structure tensor as the reference.

    python scripts/prototype_fibre_sets_real.py [METHOD] [OUT_DIR]
        METHOD: windowed_sets (default: sets shorter than one window along their fibres dropped, crossing sets split,
                the windows' modes from the deconvolved density) | windowed_sets2 | windowed_stage2

The block: BRIDGE's image store of the VOI (768³ at 20 µm), a 256 × 64 × 256 voxel slab (z, y, x) at its centre with
the filters' reach as context; the readers calibrated on the block itself (as run_volume calibrates each chunk), σ 1,
ρ 10, truncate 3; tissue = the intensity mask (global Otsu, BRIDGE's rule). The thresholds the prototype needs come
from synthetic references read through the *real* readers: texture without fibres (the window-presence null) and
straight bundles (the second-population null), both at the texture calibrated to S-21-46 — no real-data labels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

from shapetensor import st
from shapetensor.experimental import fibre_sets as FS
from shapetensor.phantoms import Grid, TextureParams, build, finish
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import SETTINGS

METHOD = sys.argv[1] if len(sys.argv) > 1 else "windowed_sets"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/andrew/bridge_wmcov/shapetensor/prototypes/fibre_sets/real_cortex")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"
SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
CORE = (256, 64, 256)                                     # z, y, x voxels: 5.1 x 1.3 x 5.1 mm


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def main():
    import zarr
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

    a = zarr.open_group(SRC, mode="r")["0"]
    reach = int(TRUNC * (SIGMA + RHO) + 0.5) + 24
    c = [n // 2 for n in a.shape]
    lo = [ci - k // 2 - reach for ci, k in zip(c, CORE)]
    hi = [ci + k // 2 + reach for ci, k in zip(c, CORE)]
    img = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    v = torch.as_tensor(img, device=DEV).float()
    thr = otsu(np.asarray(a[::4, ::4, ::4]))                                           # the VOI's own Otsu split (BRIDGE: 131.5)
    tissue = v > thr
    region = tuple(slice(reach, reach + k) for k in CORE)
    log(f"block {img.shape} at {lo}, Otsu {thr}, tissue {float(tissue.float().mean()):.2f}")
    torch.manual_seed(0)                                                              # the calibration's sampling draws
    cal = calibrate(v, tissue, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=24, device=DEV)
    log(f"calibrated on the block: tau {cal.tau:.3f}, overlap {cal.readout_overlap:.2f}, bins {cal.orientation_bins_occupied}/24")
    # thresholds from synthetic references at the S-21-46-matched texture, read through the real readers
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"])
    g = Grid(); t_ = TextureParams(**{**tex.__dict__, "amp": tex.amp * p["texture"]})
    gb = Grid(core=CORE)                                                                # texture as large as the slab: as many windows
    Zg, Xg = gb.shape[0], gb.shape[2]
    nv = torch.as_tensor(finish(gb, t_, np.full((Zg, Xg), t_.levels["gm"], np.float32), np.zeros((Zg, Xg), np.float32),
                                torch.zeros(gb.shape, device=DEV), np.random.default_rng([7, 4242])), device=DEV).float()
    Dn, _, _ = FS.window_densities(cal, cal.dictionary.prep(cal.features(nv)), gb.inner, 32, 8)
    prom_null = float(FS.prominence(Dn).max()); del Dn, nv
    bundles = [torch.as_tensor(build("bundle", 300 + k, g, tex, texture=p["texture"], angle=float(ang), device=DEV)[0], device=DEV).float()
               for k, ang in enumerate((0.0, 45.0, 90.0, 135.0))]
    share_null = FS.second_share_null(cal, bundles, g.inner)
    from shapetensor.phantoms import direction
    response = FS.readout_response(cal, bundles, [direction(a_) for a_ in (0.0, 45.0, 90.0, 135.0)], g.inner)   # the readers' one-population response
    del bundles
    log(f"nulls: window prominence {prom_null:.3f}, second-population share {share_null:.3f}")

    F = cal.features(v)
    cur = apply(cal, v, tissue, features=F, device=DEV)
    cur["activation"] = cur["activation"] / cal.tau
    log(f"current: {cur['modes'].shape[0]} populations")
    fn = {"windowed_sets2": FS.windowed_sets2, "windowed_sets": FS.windowed_sets, "windowed_stage2": FS.windowed_stage2}[METHOD]
    t1 = time.time()
    kw = {"min_extent": 32.0, "split": True, "response": response} if METHOD == "windowed_sets" else {}
    L, rep = fn(cal, v, tissue, F, prominence_null=prom_null, share_null=share_null, region=region, vox_mm=0.02, device=DEV, **kw)
    rep.pop("labels", None)
    rep["seconds"] = round(time.time() - t1, 1)
    log(f"{METHOD}: {json.dumps(rep)}")
    new = FS.fields_from_layers(cal, v, tissue, L, relative=True)
    exc = st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, mask=tissue, order="zyx", device=DEV)
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO, truncate=TRUNC, mask=tissue); val, _ = st.eig_special_3d(S)
    gap = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S, val

    # numbers: agreement with the excised structure tensor in the white matter of the slab, overall and where it is confident
    sel = torch.zeros_like(tissue); sel[region] = tissue[region]
    conf = sel & (gap > gap[sel].quantile(0.75))
    ang = lambda d1, d2: torch.rad2deg(torch.arccos((d1 * d2).sum(0).abs().clamp(max=1)))
    Lr = L[(slice(None),) + region]
    sizes = [round(float(((Lr[k] >= 1) & tissue[region]).sum()) * 0.02 ** 3, 4) for k in range(L.shape[0])]
    big = [k for k, s_ in enumerate(sizes) if s_ >= 0.05]                                 # sets of at least 0.05 mm³ in the slab
    nums = {"block_start_zyx": lo, "core": CORE, "otsu": thr, "tau": cal.tau, "nulls": {"prominence": prom_null, "second_share": share_null},
            "current": {"n_populations": int(cur["modes"].shape[0]),
                        "agreement_median_deg": round(float(ang(cur["vec"], exc)[sel].median()), 1),
                        "agreement_where_tensor_confident_deg": round(float(ang(cur["vec"], exc)[conf].median()), 1)},
            METHOD: {"n_sets": int(L.shape[0]), "sets_of_0.05mm3_or_more": len(big), "set_volumes_mm3": sorted(sizes, reverse=True)[:12],
                     "agreement_median_deg": round(float(ang(new["vec"], exc)[sel].median()), 1),
                     "agreement_where_tensor_confident_deg": round(float(ang(new["vec"], exc)[conf].median()), 1),
                     "report": rep}}
    (OUT / f"numbers_{METHOD}.json").write_text(json.dumps(nums, indent=1, default=float) + "\n")
    log(json.dumps({k: v_ for k, v_ in nums.items() if k in ("current", METHOD)}, default=float)[:900])

    # images: the mid-y plane of the slab
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from shapetensor.viz import dec_rgb
    y = region[1].start + CORE[1] // 2
    zs, xs = region[0], region[2]
    sl = (zs, y, xs); dsl = (slice(None),) + sl
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100", "#e87ba4", "#008300"]

    def strongest(A, n_show=8):
        Ac = A[(slice(None),) + sl].cpu().numpy()
        if Ac.shape[0] == 0:
            return np.zeros(Ac.shape[1:], int), 0
        tot = Ac.reshape(Ac.shape[0], -1).sum(1); order = np.argsort(-tot)
        rank = np.full(Ac.shape[0], n_show); rank[order[:n_show]] = np.arange(min(n_show, Ac.shape[0]))
        lab = rank[Ac.argmax(0)] + 1
        return np.where(Ac.max(0) >= 1.0, lab, 0), min(n_show, Ac.shape[0])

    im = img[sl[0], y, sl[2]] if False else v[sl].cpu().numpy()
    tis = tissue[sl].cpu().numpy()
    s_cur, n_cur = strongest(cur["activation"]); s_new, n_new = strongest(L)
    cmap = ListedColormap([(0.12, 0.12, 0.12)] + CAT + ["#9a9890"])
    ext = (0, CORE[2] * 0.02, 0, CORE[0] * 0.02)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10.5), dpi=120, constrained_layout=True)
    axs[0, 0].imshow(im, cmap="gray", origin="lower", extent=ext); axs[0, 0].set_title("S-21-46 cortex, 20 µm (mid-y plane of the slab)", fontsize=10)
    axs[0, 1].imshow(dec_rgb(exc[dsl].cpu().numpy()) * tis[..., None], origin="lower", extent=ext, interpolation="nearest")
    axs[0, 1].set_title(f"excised structure tensor (DEC), in the intensity mask", fontsize=10)
    axs[0, 2].imshow(s_cur, cmap=cmap, vmin=-0.5, vmax=len(CAT) + 1.5, origin="lower", extent=ext, interpolation="nearest")
    axs[0, 2].set_title(f"current: strongest population ({cur['modes'].shape[0]} in the block)", fontsize=10)
    axs[0, 3].imshow(dec_rgb(cur["vec"][dsl].cpu().numpy()) * tis[..., None], origin="lower", extent=ext, interpolation="nearest")
    axs[0, 3].set_title(f"current: merged field ({nums['current']['agreement_median_deg']}° from the tensor)", fontsize=10)
    axs[1, 0].imshow(tis, cmap="gray", origin="lower", extent=ext); axs[1, 0].set_title(f"intensity mask (Otsu {thr:g})", fontsize=10)
    axs[1, 1].imshow(np.clip(L[(slice(None),) + sl].sum(0).cpu().numpy() / 4, 0, 1), cmap="magma", origin="lower", extent=ext)
    axs[1, 1].set_title("fibre sets: total activation (nulls / 4)", fontsize=10)
    axs[1, 2].imshow(s_new, cmap=cmap, vmin=-0.5, vmax=len(CAT) + 1.5, origin="lower", extent=ext, interpolation="nearest")
    axs[1, 2].set_title(f"fibre sets: strongest set ({len(big)} of ≥ 0.05 mm³ in the slab)", fontsize=10)
    axs[1, 3].imshow(dec_rgb(new["vec"][dsl].cpu().numpy()) * tis[..., None], origin="lower", extent=ext, interpolation="nearest")
    axs[1, 3].set_title(f"fibre sets: merged field ({nums[METHOD]['agreement_median_deg']}° from the tensor)", fontsize=10)
    for ax in axs.ravel():
        ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("z (mm)", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(f"Real data: populations per chunk (current) against windowed fibre sets ({METHOD}); colours in the set maps are the 8 largest", x=0.01,
                 ha="left", fontsize=12)
    fig.savefig(OUT / f"real_cortex_{METHOD}.png"); plt.close(fig)
    np.savez_compressed(OUT / f"real_cortex_{METHOD}_slices.npz", image=im, tissue=tis, strongest_current=s_cur, strongest_sets=s_new,
                        dec_tensor=exc[dsl].cpu().numpy(), dec_current=cur["vec"][dsl].cpu().numpy(), dec_sets=new["vec"][dsl].cpu().numpy())
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
