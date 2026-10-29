"""Resilience to reconstruction artefacts on real data: streaks and rings (``shapetensor.artefacts``) added to a slab of the
S-21-46 20 µm cortical VOI, each method's field compared with its own field on the unaltered slab (no truth here).

The slab: 128 x 32 x 128 voxels (z, y, x) at the centre of BRIDGE's image store of the VOI (768³), with 32 voxels of
context on every face (the phantoms' grid); tissue = the intensity mask (global Otsu, BRIDGE's rule). Methods as
``scripts/artefacts.py``; the readers are calibrated on the slab itself as ``run_volume`` calibrates a chunk, either on
the unaltered slab (``clean`` readers) or on the slab as altered (``self``: what a run on artefacted data does).

Scores on the core's tissue: ``change`` the median angle to the method's own field on the unaltered slab and the share
of voxels turned by more than 20°; ``captured`` (near-parallel streaks) the share turned onto the streaks: within 20° of
their direction where the unaltered field was more than 20° from it; populations found.

    python scripts/artefacts_real.py [OUT_DIR]        QUICK=1: strengths 0/50/200, streaks along x and rings
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from shapetensor import st
from shapetensor.dev import DevConfig, apply_dev
from shapetensor.fibre_sets import FibreSetConfig, References, iso_textures
from shapetensor.phantoms import Grid
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import angle_deg

torch.use_deterministic_algorithms(True, warn_only=True)
spec = importlib.util.spec_from_file_location("art", str(Path(__file__).with_name("artefacts.py")))
A = importlib.util.module_from_spec(spec); sys.argv = sys.argv[:1] + sys.argv[1:]; spec.loader.exec_module(A)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/artefacts")
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"
DEV = A.DEV; SIGMA, RHO, TRUNC = A.SIGMA, A.RHO, A.TRUNC
QUICK = os.environ.get("QUICK") == "1"
AMPS = (0.0, 50.0, 200.0) if QUICK else (0.0, 25.0, 50.0, 100.0, 200.0)
KINDS = {k: A.KINDS[k] for k in (("streaks along x", "rings") if QUICK else A.KINDS)}
READERS = ("self", "clean")


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def corrupt(img, kind, amp, seed):
    if amp <= 0:
        return img, None
    k, kw = KINDS[kind]
    from shapetensor import artefacts
    return artefacts.add(k, img, amp, seed=seed, device=DEV, **kw)


def field(method, img, tissue, cal_img, refs, grid):
    v = torch.as_tensor(img, device=DEV).float()
    if method == "structure tensor":
        return {"vec": st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, mask=tissue, order="zyx", device=DEV), "n_populations": None}
    m = A.METHODS[method]
    cal = calibrate(torch.as_tensor(cal_img, device=DEV).float(), tissue, sigma=SIGMA, rho=RHO, truncate=TRUNC, engine=m["engine"],
                    engine_kwargs=m["engine_kwargs"], smooth=m["smooth"], seed=0, device=DEV, iso=iso_textures("calibrated", DEV) if m.get("iso") else None)
    if method == "released":
        o = apply(cal, v, tissue, features=cal.features(v), device=DEV)
        return {"vec": o["vec"], "n_populations": int(o["modes"].shape[0]), "readout_fit": round(float(cal.readout_overlap), 2)}
    cfg = DevConfig(engine=m["engine"], engine_kwargs=dict(m["engine_kwargs"]), sigma=SIGMA, rho=RHO, truncate=TRUNC, smooth=m["smooth"],
                    fibre_sets=FibreSetConfig(), vox_mm=grid.vox)
    o = apply_dev(cal, v, tissue, cfg=cfg, references=refs, features=cal.features(v), region=grid.inner, device=DEV)
    rep = o["report"]
    return {"vec": o["vec"], "n_populations": o["n_populations"], "readout_fit": round(float(cal.readout_overlap), 2),
            "stage2": {k: rep.get(k) for k in ("windows", "windows_with_fibres", "sets_after_reunmix", "presence", "probe_null", "window_probe_max",
                                               "window_probe_median", "share_null")}}


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    grid = Grid()
    a = zarr.open_group(SRC, mode="r")["0"]
    c = [n // 2 for n in a.shape]
    sl = tuple(slice(ci - s // 2, ci - s // 2 + s) for ci, s in zip(c, grid.shape))
    img = np.asarray(a[sl]).astype(np.uint8)
    tissue = torch.as_tensor(img > otsu(img), device=DEV)
    sel = torch.zeros(img.shape, dtype=torch.bool, device=DEV); sel[grid.inner] = True; sel &= tissue
    refs = References(tuple(s.stop - s.start for s in grid.inner), "calibrated", DEV)
    res = {"settings": {"src": SRC, "slab": [[s.start, s.stop] for s in sl], "tissue_share_core": round(float(sel.sum() / tissue[grid.inner].numel()), 3),
                        "amps": AMPS, "kinds": KINDS}, "runs": []}
    path = OUT / ("artefacts_real_quick.json" if QUICK else "artefacts_real.json")
    for method in ("structure tensor", "released", "dev"):
        ref = field(method, img, tissue, img, refs, grid)
        log(f"{method}: unaltered slab, populations {ref['n_populations']}" + (f", stage 2 {ref['stage2']}" if "stage2" in ref else ""))
        res["runs"].append({"method": method, "readers": "self", "artefact": "none", "amp": 0.0, "n_populations": ref["n_populations"],
                            "stage2": ref.get("stage2"), "readout_fit": ref.get("readout_fit")})
        for kind in KINDS:
            for amp in AMPS[1:]:
                im2, info = corrupt(img, kind, amp, 0)
                for readers in (READERS if method != "structure tensor" else ("none",)):
                    t1 = time.time()
                    o = field(method, im2, tissue, im2 if readers == "self" else img, refs, grid)
                    e = angle_deg(o["vec"], ref["vec"])
                    r = {"change median": round(float(e[sel].median()), 1), "turned over 20": round(float((e[sel] > 20).float().mean()), 3),
                         "n_populations": o["n_populations"]}
                    if info["kind"] == "streaks" and info["spread_deg"] <= 10:
                        s = torch.as_tensor(info["direction"], device=DEV).view(3, 1, 1, 1).expand_as(o["vec"])
                        cap = (angle_deg(o["vec"], s) <= 20) & (angle_deg(ref["vec"], s) > 20)
                        r["captured"] = round(float(cap[sel].float().mean()), 3)
                    row = {"method": method, "readers": readers, "artefact": kind, "amp": amp, "artefact_rms": round(info["rms"], 2),
                           "seconds": round(time.time() - t1, 1), "scores": r, "stage2": o.get("stage2"), "readout_fit": o.get("readout_fit")}
                    res["runs"].append(row); log(f"  {method} [{readers}] {kind} {amp:g}: {json.dumps(r)}")
                    json.dump(res, open(path, "w"), indent=1, default=str)
                    del o; torch.cuda.empty_cache()
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
