"""QC figures of the dev pipeline against the released one, on the validation phantoms, a real slab and two artefact
examples. Every panel is the mid-y (x-z) plane of the scored core; directions in plain DEC (R |x|, G |y|, B |z|).

Per phantom: the image, the truth, the excised structure tensor, the released field, the dev field; the dev pipeline's
presence (distance from fibre-free texture, 32³-window mean, the null as a contour), its first two population layers
(direction as DEC, activation as opacity over the image), and the error of the released and dev fields to the truth.
Real slab (S-21-46 20 µm cortex): the same without truth, readers calibrated on the slab. Artefacts: the 60° bundle with
streaks along x and with rings at strength 50.

Readers as the validation's pooled calibration (12 bundles + the six cases); dev = dev-vit with the 3³ box,
distance presence, fibre sets, readout-tensor directions.

    python scripts/figures_dev.py [OUT_DIR]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
import zarr

from shapetensor import artefacts, st
from shapetensor.dev import DevConfig, apply_dev
from shapetensor.fibre_sets import FibreSetConfig, References, iso_textures
from shapetensor.phantoms import BEND_GRID, Grid, build, truth_field
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import ANGLES, angle_deg
from shapetensor.viz import dec_activation_overlay, dec_rgb

torch.use_deterministic_algorithms(True, warn_only=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/figures")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import os
CASES = (("bend", "bend", {}), ("straight", "bend", {"straight": True}), ("band crossing 90", "band_crossing", {"cross_deg": 90.0}),
         ("band crossing 60", "band_crossing", {"cross_deg": 60.0}), ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}))
DEVM = {"engine": "anatomix_devvit", "engine_kwargs": {}, "smooth": 3}
ONLY = [c for c in os.environ.get("ONLY", "").split(";") if c]                   # redraw only these cases (and skip the artefacts and the slab)
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"


def mid(a, grid):
    """The mid-y plane of the core: ``(..., Z, Y, X)`` -> ``(..., Zc, Xc)`` numpy."""
    z, y, x = grid.inner; yc = (y.start + y.stop) // 2
    a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return a[..., z, yc, x]


def run_dev(cal, v, tissue, refs, grid):
    cfg = DevConfig(engine=DEVM["engine"], engine_kwargs={}, sigma=SIGMA, rho=RHO, truncate=TRUNC, smooth=DEVM["smooth"],
                    fibre_sets=FibreSetConfig(presence="distance"), vox_mm=grid.vox, direction="readout")
    F = cal.features(v)
    o = apply_dev(cal, v, tissue, cfg=cfg, references=refs, features=F, region=grid.inner, device=DEV)
    dmap = cal.texture_model.predict(cal.dictionary.prep(F), prepped=True)
    dwin = Fn.avg_pool3d(dmap[None, None], 31, stride=1, padding=15)[0, 0]
    o["distance_window"] = dwin; del F
    return o


def panels(name, img, grid, st_vec, rel, dev, tr=None, extra_title=""):
    """One figure: two rows of panels on the mid-y plane."""
    g = mid(img, grid); null = dev["nulls"].get("distance")
    t1 = truth_field(tr, "dir1", img.shape[1]).to(DEV) if tr is not None else None
    t2 = truth_field(tr, "dir2", img.shape[1]).to(DEV) if tr is not None and np.isfinite(tr["dir2"]).any() else None

    def err(vec):                                                                   # to the nearest population present (NaN-aware: a band's arm has one)
        e = angle_deg(vec, t1)
        if t2 is not None:
            e2 = angle_deg(vec, t2)
            e = torch.where(torch.isnan(e), e2, torch.where(torch.isnan(e2), e, torch.minimum(e, e2)))
        return mid(e, grid)

    top = [("image", np.stack([(g - g.min()) / max(np.ptp(g), 1e-6)] * 3, -1))]
    if tr is not None:
        tt = mid(t1.nan_to_num(0), grid)
        if t2 is not None:
            tt = 0.5 * (dec_rgb(tt) + dec_rgb(mid(t2.nan_to_num(0), grid))); top.append(("truth (both populations)", tt))
        else:
            top.append(("truth", dec_rgb(tt)))
    top += [("excised structure tensor", dec_rgb(mid(st_vec, grid))), (f"released ({rel['n_populations']} populations)", dec_rgb(mid(rel["vec"], grid))),
            (f"dev ({dev['n_populations']} sets)", dec_rgb(mid(dev["vec"], grid)))]
    L, D = dev["activation"], dev["dirs"]
    gi = mid(img, grid)
    bot = [("dev presence (distance from texture)", None)]
    for k in range(2):
        if k < L.shape[0] and float(L[k].abs().sum()) > 0:
            bot.append((f"dev set {k + 1}: direction x activation", dec_activation_overlay(mid(D[k], grid), mid(L[k], grid), gi)))
        else:
            bot.append((f"dev set {k + 1}: none", np.stack([(gi - gi.min()) / max(np.ptp(gi), 1e-6)] * 3, -1) * 0.4))
    if tr is not None:
        er, ed = err(rel["vec"]), err(dev["vec"])
        bot += [(f"released error (median {np.nanmedian(er):.1f}°)", er), (f"dev error (median {np.nanmedian(ed):.1f}°)", ed)]
    n = max(len(top), len(bot))
    fig, axs = plt.subplots(2, n, figsize=(3.3 * n, 7.4), facecolor="white")
    for c in range(n):
        for r, row in enumerate((top, bot)):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if c >= len(row):
                ax.axis("off"); continue
            title, im = row[c]
            if title.startswith("dev presence"):
                dw = mid(dev["distance_window"], grid)
                h = ax.imshow(dw, origin="lower", cmap="magma", vmin=float(np.percentile(dw, 1)), vmax=float(np.percentile(dw, 99)))
                if null is not None:
                    ax.contour(dw, levels=[null], colors="cyan", linewidths=1.0)
                title += f"\nnull {null:.2f} (cyan); windows {dev['report'].get('windows_with_fibres')}/{dev['report'].get('windows')}"
                plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
            elif "error" in title:
                h = ax.imshow(im, origin="lower", cmap="viridis", vmin=0, vmax=45); plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02, label="deg")
            else:
                ax.imshow(im, origin="lower", interpolation="nearest")
            ax.set_title(title, fontsize=9)
    fig.suptitle(f"{name}{extra_title}   (mid-y plane, z up, x right; DEC: R |x|, G |y|, B |z|)", fontsize=11)
    plt.tight_layout()
    path = OUT / f"dev_qc_{name.replace(' ', '_').replace(',', '')}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    tex, texture = P.tex_for("calibrated"); cg = Grid()
    bundles = [build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0] for k, a in enumerate(ANGLES)]
    imgs = {nm: build(kind, 0, tex=tex, texture=texture, device=DEV, **kw)[:2] for nm, kind, kw in CASES}
    vols = [torch.as_tensor(b, device=DEV).float() for b in bundles]; tis = [None] * len(vols)
    for nm, kind, _ in CASES:
        v = torch.as_tensor(imgs[nm][0], device=DEV).float(); vols.append(v)
        tis.append((v > P.otsu(imgs[nm][0])) if kind in ("bend", "band_crossing") else None)
    cal_r = calibrate(vols, tis, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine="anatomix_enc2", smooth=3, seed=0, device=DEV)
    cal_d = calibrate(vols, tis, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=DEVM["engine"], engine_kwargs={}, smooth=DEVM["smooth"],
                      seed=0, device=DEV, iso=iso_textures("calibrated", DEV)); del vols, tis
    log(f"calibrated: released fit {cal_r.readout_overlap:.2f}, dev fit {cal_d.readout_overlap:.2f}")
    refs = {}
    paths = []
    for nm, kind, kw in CASES:
        if ONLY and nm not in ONLY:
            continue
        img, tr = imgs[nm]; grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        v = torch.as_tensor(img, device=DEV).float()
        tissue = (v > P.otsu(img)) if kind in ("bend", "band_crossing") else torch.ones_like(v, dtype=torch.bool)
        if id(grid) not in refs:
            refs[id(grid)] = References(tuple(s.stop - s.start for s in grid.inner), "calibrated", DEV)
        stv = st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, order="zyx", device=DEV)
        rel = apply(cal_r, v, tissue, features=cal_r.features(v), device=DEV); rel["n_populations"] = int(rel["modes"].shape[0])
        dev = run_dev(cal_d, v, tissue, refs[id(grid)], grid)
        paths.append(panels(nm, v, grid, stv, rel, dev, tr)); log(nm, "drawn")
        del rel, dev, stv, v; torch.cuda.empty_cache()
    if ONLY:
        log("redrew", ONLY); return
    # artefact examples on the 60° bundle, the readers as above (clean)
    grid = Grid(); b60, trb = build("bundle", 0, grid, tex, texture=texture, device=DEV, angle=60.0)[:2]
    for kind, kw, label in (("streaks", {"angle_deg": 0.0, "spread_deg": 5.0}, "streaks along x"), ("rings", {}, "rings")):
        im2, info = artefacts.add(kind, b60, 50.0, seed=0, device=DEV, **kw)
        v = torch.as_tensor(im2, device=DEV).float(); tissue = torch.ones_like(v, dtype=torch.bool)
        if id(grid) not in refs:
            refs[id(grid)] = References(tuple(s.stop - s.start for s in grid.inner), "calibrated", DEV)
        stv = st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, order="zyx", device=DEV)
        rel = apply(cal_r, v, tissue, features=cal_r.features(v), device=DEV); rel["n_populations"] = int(rel["modes"].shape[0])
        dev = run_dev(cal_d, v, tissue, refs[id(grid)], grid)
        paths.append(panels(f"bundle 60, {label} 50", v, grid, stv, rel, dev, trb)); log(label, "drawn")
        del rel, dev, stv, v; torch.cuda.empty_cache()
    del cal_r, cal_d; torch.cuda.empty_cache()
    # the real slab: readers calibrated on it (as run_volume calibrates a chunk)
    a = zarr.open_group(SRC, mode="r")["0"]; c = [n // 2 for n in a.shape]
    sl = tuple(slice(ci - s // 2, ci - s // 2 + s) for ci, s in zip(c, grid.shape))
    img = np.asarray(a[sl]).astype(np.uint8); v = torch.as_tensor(img, device=DEV).float(); tissue = v > P.otsu(img)
    stv = st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, mask=tissue, order="zyx", device=DEV)
    cr = calibrate(v, tissue, sigma=SIGMA, rho=RHO, truncate=TRUNC, engine="anatomix_enc2", smooth=3, seed=0, device=DEV)
    rel = apply(cr, v, tissue, features=cr.features(v), device=DEV); rel["n_populations"] = int(rel["modes"].shape[0]); del cr
    cd = calibrate(v, tissue, sigma=SIGMA, rho=RHO, truncate=TRUNC, engine=DEVM["engine"], engine_kwargs={}, smooth=DEVM["smooth"], seed=0, device=DEV,
                   iso=iso_textures("calibrated", DEV))
    dev = run_dev(cd, v, tissue, refs[id(grid)], grid)
    paths.append(panels("real slab S-21-46 20um cortex", v, grid, stv, rel, dev, None, extra_title=f"  (slab {[[s.start, s.stop] for s in sl]})"))
    log("real slab drawn")
    json.dump([str(p) for p in paths], open(OUT / "figures.json", "w"), indent=1)
    log("wrote", len(paths), "figures to", OUT)


if __name__ == "__main__":
    main()
