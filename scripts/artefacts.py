"""Resilience to reconstruction artefacts: streaks and rings (``shapetensor.artefacts``) added at increasing strength to
the validation's phantoms, the field of each method scored against the truth and against its own artefact-free field.

Methods: the excised structure tensor (the baseline), the released pipeline (anatomix encoder 2, block populations,
image structure tensor) and the dev pipeline (dev-vit, fibre sets with the presence probe, readout-tensor directions).
The readers are calibrated
on the validation's twelve straight bundles at the calibrated texture and the phantoms themselves (the pooled, in-domain
calibration: dev-vit's readers do not carry over from bundles alone), either clean (``clean`` readers: the artefact is
met only at application) or carrying the same artefact (``artefacted`` readers: a block that calibrates on itself, as
``run_volume`` does, learns from contaminated data). The dev pipeline's references are synthetic and clean by design.

Phantoms (validation grid, 20 µm, whole block white matter): a bundle at 60° and the 90° crossing (and the 60°). Artefacts, in
z slices (the rotation axis along z): streaks near-parallel along x (inside the fibre planes), near-parallel along y
(across them), at any direction; rings about an axis half a block beyond the x = 0 face. Strength in grey levels
against the fibre darkening (50 at the calibrated texture): 0, 25, 50, 100, 200.

Scores on the core: the merged field's median error to the nearest population and its share within 15°; the
multi-vector field's first vector and its best (each true population's nearest vector, the worse of the two: is the
truth still in the field when an artefact adds a population?); on the crossings the pair (median, both within 15°);
``captured``: the share of voxels whose merged vector lies
within 20° of the streaks' direction and more than 20° from every population (near-parallel streaks only); ``change``:
the median angle between the field and the same method's field without the artefact; populations found; the dev
pipeline's stage-2 presence numbers.

    python scripts/artefacts.py [OUT_DIR]            QUICK=1: strengths 0/50/200, clean readers; CROSSING60=1 adds the 60° crossing
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from shapetensor import artefacts, st
from shapetensor.dev import DevConfig, apply_dev
from shapetensor.fibre_sets import FibreSetConfig, References, iso_textures
from shapetensor.phantoms import Grid, TextureParams, build, truth_field
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import ANGLES, SETTINGS, angle_deg

torch.use_deterministic_algorithms(True, warn_only=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/artefacts")
DEV = "cuda"; SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0; SETTING = "calibrated"
QUICK = os.environ.get("QUICK") == "1"
AMPS = (0.0, 50.0, 200.0) if QUICK else (0.0, 25.0, 50.0, 100.0, 200.0)
READERS = ("clean",) if QUICK else tuple(os.environ.get("READERS", "clean,artefacted").split(","))
PHANTOMS = (("bundle 60", "bundle", {"angle": 60.0}), ("crossing 90", "crossing", {"cross_deg": 90.0}))
if os.environ.get("CROSSING60") == "1":
    PHANTOMS += (("crossing 60", "crossing", {"cross_deg": 60.0}),)
KINDS = {"streaks along x": ("streaks", {"angle_deg": 0.0, "spread_deg": 5.0}),
         "streaks along y": ("streaks", {"angle_deg": 90.0, "spread_deg": 5.0}),
         "streaks, any direction": ("streaks", {"angle_deg": 0.0, "spread_deg": 90.0}),
         "rings": ("rings", {})}
if os.environ.get("KINDS"):
    KINDS = {k: v for k, v in KINDS.items() if k in os.environ["KINDS"].split(";")}
METHODS = {"released": {"engine": "anatomix_enc2", "engine_kwargs": {}, "smooth": 3},
           "dev": {"engine": "anatomix_devvit", "engine_kwargs": {}, "smooth": 3, "iso": True, "direction": "readout"}}


def corrupt(img, kind, amp, seed):
    if amp <= 0:
        return img, None
    k, kw = KINDS[kind]
    return artefacts.add(k, img, amp, seed=seed, device=DEV, **kw)


def run_method(method, cal, refs, v, grid):
    if method == "structure tensor":
        return {"vec": st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, order="zyx", device=DEV), "n_populations": None}
    if method == "released":
        o = apply(cal, v, features=cal.features(v), device=DEV)
        return {"vec": o["vec"], "vecs": o["vecs"], "n_populations": int(o["modes"].shape[0])}
    m = METHODS[method]
    cfg = DevConfig(engine=m["engine"], engine_kwargs=dict(m["engine_kwargs"]), sigma=SIGMA, rho=RHO, truncate=TRUNC, smooth=m["smooth"],
                    fibre_sets=FibreSetConfig(), vox_mm=grid.vox, direction=m.get("direction", "readout"))
    o = apply_dev(cal, v, None, cfg=cfg, references=refs, features=cal.features(v), region=grid.inner, device=DEV)
    rep = o["report"]
    return {"vec": o["vec"], "vecs": o["vecs"], "n_populations": o["n_populations"],
            "stage2": {k: rep.get(k) for k in ("windows", "windows_with_fibres", "windows_with_2plus", "sets_after_reunmix", "presence",
                                               "probe_null", "window_probe_max", "window_probe_median", "mass_null", "window_mass_max", "window_mass_median", "share_null")}}


def scores(o, tr, grid, ny, info, clean_vec):
    sel = torch.zeros((grid.shape[0], ny, grid.shape[2]), dtype=torch.bool, device=DEV); sel[grid.inner] = True
    t1 = truth_field(tr, "dir1", ny).to(DEV)
    t2 = truth_field(tr, "dir2", ny).to(DEV) if np.isfinite(tr["dir2"]).any() else None
    e = angle_deg(o["vec"], t1) if t2 is None else torch.minimum(angle_deg(o["vec"], t1), angle_deg(o["vec"], t2))
    r = {"merged median": round(float(e[sel].median()), 1), "merged within 15": round(float((e[sel] <= 15).float().mean()), 3),
         "n_populations": o["n_populations"]}
    if "vecs" in o:                                                                  # is the truth still in the multi-vector field?
        V = o["vecs"]; T = [t1] + ([t2] if t2 is not None else [])
        best = torch.stack([torch.stack([angle_deg(V[k], t) for k in range(V.shape[0])]).nan_to_num(90.0).min(0).values for t in T]).max(0).values
        r["first vector median"] = round(float((angle_deg(V[0], t1) if t2 is None else torch.minimum(angle_deg(V[0], t1), angle_deg(V[0], t2)))[sel].median()), 1)
        r["best vector median"] = round(float(best[sel].median()), 1)                     # every population matched by some vector: the worst of their best
    if t2 is not None and "vecs" in o:
        V = o["vecs"]
        if V.shape[0] >= 2:
            e11, e12, e21, e22 = angle_deg(V[0], t1), angle_deg(V[0], t2), angle_deg(V[1], t1), angle_deg(V[1], t2)
            swap = (e12 + e21) < (e11 + e22)
            ea, eb = torch.where(swap, e12, e11), torch.where(swap, e21, e22)
            r["pair median"] = round(float(((ea + eb) / 2)[sel].median()), 1)
            r["pair both within 15"] = round(float(((ea <= 15) & (eb <= 15))[sel].float().mean()), 3)
        else:
            r["pair median"], r["pair both within 15"] = None, 0.0
    if info is not None and info["kind"] == "streaks" and info["spread_deg"] <= 10:
        s = torch.as_tensor(info["direction"], device=DEV).view(3, 1, 1, 1).expand_as(o["vec"])
        cap = (angle_deg(o["vec"], s) <= 20) & (e > 20)
        r["captured"] = round(float(cap[sel].float().mean()), 3)
    if clean_vec is not None:
        r["change median"] = round(float(angle_deg(o["vec"], clean_vec)[sel].median()), 1)
    if "stage2" in o:
        r["stage2"] = o["stage2"]
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    p = SETTINGS[SETTING]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]); texture = p["texture"]
    cg = Grid(); grid = Grid()
    bundles = [build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0] for k, a in enumerate(ANGLES)]
    images = {name: build(kind, 0, grid, tex, texture=texture, device=DEV, **kw)[:2] for name, kind, kw in PHANTOMS}
    refs = References(tuple(s.stop - s.start for s in grid.inner), SETTING, DEV)
    res = {"settings": {"sigma": SIGMA, "rho": RHO, "truncate": TRUNC, "texture": SETTING, "fibre_darkening": tex.amp * texture,
                        "amps": AMPS, "readers": READERS, "kinds": {k: v for k, v in KINDS.items()}}, "runs": []}
    path = OUT / ("artefacts_quick.json" if QUICK else "artefacts.json")
    clean = {}                                                                      # (method, readers, phantom) -> the artefact-free field
    todo = [("structure tensor", "none")] + [(m, r) for m in METHODS for r in READERS]
    if os.environ.get("ART_METHODS"):                                                # e.g. "structure tensor;released"
        todo = [t for t in todo if t[0] in os.environ["ART_METHODS"].split(";")]
    if os.environ.get("ART_TAG"):
        path = path.with_name(path.stem + "_" + os.environ["ART_TAG"] + ".json")
    cals = {}
    for method, readers in todo:
        for kind in KINDS:
            for amp in AMPS:
                if amp == 0 and kind != next(iter(KINDS)):
                    continue                                                        # one artefact-free run per method and readers
                cal = None
                if method != "structure tensor":
                    key = (method, readers if amp > 0 else "clean", kind if (readers == "artefacted" and amp > 0) else "", amp if readers == "artefacted" else 0)
                    if key not in cals:
                        cals.clear()                                                # one calibration on the GPU at a time
                        m = METHODS[method]; tc = time.time()
                        pool = bundles + [images[n][0] for n, _, _ in PHANTOMS]                 # in-domain: the bundles and the phantoms themselves
                        vols = [torch.as_tensor(corrupt(b, kind, amp, 900 + k)[0] if key[2] else b, device=DEV).float() for k, b in enumerate(pool)]
                        cals[key] = calibrate(vols, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], engine=m["engine"],
                                              engine_kwargs=m["engine_kwargs"], smooth=m["smooth"], seed=0, device=DEV,
                                              iso=iso_textures(SETTING, DEV) if m.get("iso") else None)
                        log(f"{method} | {readers} readers, {kind if key[2] else 'no artefact'} {amp:g}: calibrated in {time.time() - tc:.0f} s, "
                            f"readout fit {cals[key].readout_overlap:.2f}")
                        del vols
                    cal = cals[key]
                for name, kind_, kw in PHANTOMS:
                    img, tr = images[name]
                    im2, info = corrupt(img, kind, amp, 0)
                    v = torch.as_tensor(im2, device=DEV).float(); t1 = time.time()
                    o = run_method(method, cal, refs, v, grid)
                    ck = (method, readers, name)
                    if amp == 0:
                        clean[ck] = o["vec"]
                    r = scores(o, tr, grid, v.shape[1], info, None if amp == 0 else clean.get(ck))
                    row = {"method": method, "readers": readers, "phantom": name, "artefact": kind if amp > 0 else "none", "amp": amp,
                           "artefact_rms": None if info is None else round(info["rms"], 2), "seconds": round(time.time() - t1, 1), "scores": r}
                    res["runs"].append(row)
                    log(f"  {method} [{readers}] {name} | {row['artefact']} {amp:g}: {json.dumps({k: v_ for k, v_ in r.items() if k != 'stage2'})}"
                        + (f" | windows {r['stage2']['windows_with_fibres']}/{r['stage2']['windows']}, sets {r['stage2']['sets_after_reunmix']}" if "stage2" in r else ""))
                    json.dump(res, open(path, "w"), indent=1, default=str)
                    del o, v; torch.cuda.empty_cache()
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
