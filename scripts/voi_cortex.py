"""The chain on the HiP-CT cortex VOI (S-21-46, 20 µm, 768³) beside the excised structure tensor, with tractography.

    python scripts/voi_cortex.py [--out DIR] [--chunk 384] [--track-size 384] [--seeds 20000] [--skip-field]

Writes the field (``field.zarr``: vecs / weights / labels / vec / weight / N / sheet) and the structure tensor's
(``st.zarr``), tracks both from the same seeds in a central sub-volume, and reports: streamline length, termination
reasons, the share of steps taken on a secondary vector, folded streamlines (long tracks whose ends lie close), the
share of streamlines passing cells with two populations that keep their heading. A figure of the mid-slice: image,
structure tensor DEC, shape DEC, N, identities, streamline density.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from shapetensor import st
from shapetensor.tracking import REASONS, track
from shapetensor.volume import run_volume, stitch_labels

SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/hdd/andrew/shapetensor_runs/S-21-46_20um_cortex_v3")
    ap.add_argument("--chunk", type=int, default=384); ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--track-size", type=int, default=384); ap.add_argument("--seeds", type=int, default=20000)
    ap.add_argument("--skip-field", action="store_true"); ap.add_argument("--skip-st", action="store_true"); ap.add_argument("--skip-stitch", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    z = zarr.open_group(SRC, mode="r")["0"]
    from skimage.filters import threshold_multiotsu
    sub = np.asarray(z[::4, ::4, ::4]); thr = threshold_multiotsu(sub, classes=3); tissue_thr = float(thr[0]); wm_thr = float(thr[1])
    log(f"thresholds: tissue {tissue_thr:.0f}, white matter {wm_thr:.0f}")
    rep = {"thresholds": {"tissue": tissue_thr, "wm": wm_thr}, "scale": a.scale, "chunk": a.chunk}
    if not a.skip_field:
        t1 = time.time()
        run_volume(SRC, str(out / "field.zarr"), level="0", chunk=(a.chunk,) * 3, tissue_threshold=tissue_thr, scale=a.scale, log=log)
        rep["seconds field"] = round(time.time() - t1, 1)
    if not a.skip_stitch:
        t1 = time.time(); n_pop = stitch_labels(str(out / "field.zarr"), log=log); rep["populations"] = n_pop; rep["seconds stitch"] = round(time.time() - t1, 1)
    else:
        rep["populations"] = zarr.open_group(str(out / "field.zarr"), mode="r").attrs.get("populations")
    if not a.skip_st:
        t1 = time.time()
        st.run_volume(SRC, str(out / "st.zarr"), sigma=1.0, rho=a.scale, truncate=3.0, chunk=(384, 384, 384), level="0", arrays=("vec", "gap"), dtype=np.float16,
                      tissue_threshold=tissue_thr, log=log)
        rep["seconds structure tensor"] = round(time.time() - t1, 1)
    g = zarr.open_group(str(out / "field.zarr"), mode="r"); gs = zarr.open_group(str(out / "st.zarr"), mode="r")
    rep["field attrs"] = {k: g.attrs[k] for k in ("floor", "seconds_per_chunk_median", "seconds_total", "populations") if k in g.attrs}
    # --- tractography in a central sub-volume ---------------------------------------------------------------------
    n = a.track_size; Z, Y, X = z.shape; o = [(s - n) // 2 for s in (Z, Y, X)]
    sl = tuple(slice(oo, oo + n) for oo in o)
    img = torch.as_tensor(np.asarray(z[sl]).astype(np.float32), device=DEV)
    vecs = torch.as_tensor(np.asarray(g["vecs"][(slice(None), slice(None)) + sl]).astype(np.float32), device=DEV).flip(1)        # (K, 3, ...) -> (z, y, x)
    weights = torch.as_tensor(np.asarray(g["weights"][(slice(None),) + sl]).astype(np.float32), device=DEV)
    labels = np.asarray(g["labels"][(slice(None),) + sl]); Nmap = np.asarray(g["N"][sl]); weight = np.asarray(g["weight"][sl]).astype(np.float32)
    d_st = torch.as_tensor(np.asarray(gs["vec"][(slice(None),) + sl]).astype(np.float32), device=DEV).flip(0)
    tissue = img >= tissue_thr; wm = img >= wm_thr
    gen = torch.Generator().manual_seed(0)
    idx = wm.nonzero(); idx = idx[torch.randperm(idx.shape[0], generator=gen)[:a.seeds].to(DEV)].float()
    log(f"tracking {idx.shape[0]} seeds in the central {n}^3 (white matter {float(wm.float().mean()):.2f} of it)")
    res = {}
    for name, dirs, w in (("shape", vecs, weights), ("structure tensor", d_st[None], None)):
        r = track(dirs, tissue, idx, step=0.5, max_angle_deg=30.0, max_steps=4000, weights=w, min_weight=1e-6 if w is not None else 0.0)
        L = r["length"] * 0.02
        pts = r["points"]; e0 = pts[torch.arange(pts.shape[0]), 0, r["n_steps"][:, 0]]; e1 = pts[torch.arange(pts.shape[0]), 1, r["n_steps"][:, 1]]
        ends = (e0 - e1).norm(dim=1) * 0.02
        long = L >= 1.0
        folded = float(((ends < 0.5 * L) & long).float().sum() / long.float().sum().clamp_min(1))
        why = {REASONS[k]: round(float((r["reason"] == k).float().mean()), 3) for k in range(len(REASONS)) if bool((r["reason"] == k).any())}
        res[name] = {"length mm median": round(float(L.median()), 3), "length mm p90": round(float(L.quantile(0.9)), 3), ">= 1 mm share": round(float(long.float().mean()), 3),
                     "folded share (of >= 1 mm)": round(folded, 3), "secondary-vector steps share": round(float(r["used_secondary"].sum() / r["n_steps"].sum().clamp_min(1)), 4),
                     "stop reasons": why}
        if name == "shape":
            from shapetensor.tracking import density
            dens = density(pts, (n, n, n), r["n_steps"]); res[name]["density saved"] = True
            np.save(out / "density_shape.npy", dens.cpu().numpy().astype(np.float32))
        else:
            from shapetensor.tracking import density
            np.save(out / "density_st.npy", density(pts, (n, n, n), r["n_steps"]).cpu().numpy().astype(np.float32))
        log(f"{name}: {json.dumps(res[name])}")
    rep["tracking"] = res
    # how much of the tissue holds two populations, and where
    Nt = Nmap[tissue.cpu().numpy()]; rep["N in tissue (central)"] = {f"N={k}": round(float((Nt == k).mean()), 3) for k in range(3)}
    Nw = Nmap[wm.cpu().numpy()]; rep["N in white matter (central)"] = {f"N={k}": round(float((Nw == k).mean()), 3) for k in range(3)}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    # --- figure --------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapetensor.viz import dec_rgb, identity_palette, identity_rgb
    k = n // 2
    gimg = img[k].cpu().numpy(); gimg = np.clip((gimg - np.percentile(gimg, 1)) / max(np.ptp(np.percentile(gimg, [1, 99])), 1e-6), 0, 1)
    tis = tissue[k].cpu().numpy()
    fig, axs = plt.subplots(2, 3, figsize=(19, 12.6), facecolor="white")
    axs[0, 0].imshow(np.stack([gimg] * 3, -1)); axs[0, 0].set_title("HiP-CT (S-21-46, 20 µm), axial mid-slice of the central sub-volume")
    axs[0, 1].imshow(dec_rgb(d_st[:, k].cpu().numpy()) * tis[..., None]); axs[0, 1].set_title("excised structure tensor, DEC")
    axs[0, 2].imshow(dec_rgb(vecs[0, :, k].cpu().numpy()) * (weight[k] >= 0.2)[..., None] * tis[..., None]); axs[0, 2].set_title("shape: strongest population, DEC (weight >= 0.2)")
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#222222", "#2a78d6", "#e34948"])
    h = axs[1, 0].imshow(np.where(tis, Nmap[k], np.nan), cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest"); plt.colorbar(h, ax=axs[1, 0], fraction=0.046, ticks=[0, 1, 2])
    axs[1, 0].set_title("populations per voxel")
    lab = labels[:, k]; npal = max(int(lab.max()), 1); pal = identity_palette(min(npal, 4000))
    rgb = np.zeros(lab.shape[1:] + (3,), np.float32)
    for r_ in range(lab.shape[0]):
        m = lab[r_] > 0; rgb[m] = np.asarray([pal[int(x) % len(pal)] if int(x) < len(pal) else pal[int(x) % len(pal)] for x in lab[r_][m]])[:, :3] if m.any() else rgb[m]
    axs[1, 1].imshow(rgb * tis[..., None]); axs[1, 1].set_title(f"identities ({rep.get('populations', '?')} populations stitched)")
    ds = np.load(out / "density_shape.npy")[k]; dst = np.load(out / "density_st.npy")[k]
    axs[1, 2].imshow(np.stack([np.clip(dst / max(np.percentile(dst, 99), 1), 0, 1), np.clip(ds / max(np.percentile(ds, 99), 1), 0, 1), np.zeros_like(ds)], -1))
    axs[1, 2].set_title("streamline density: structure tensor (red), shape (green)")
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"S-21-46 cortex VOI, 20 µm: the chain (scale {a.scale:g}) beside the excised structure tensor", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "voi_cortex.png", dpi=90); log("figure", out / "voi_cortex.png")
    log(json.dumps(rep))


if __name__ == "__main__":
    main()
