"""Iterate on the chain: the lattice scores of :class:`shapetensor.field.Orientations` variants on the phantom set.

    python scripts/dev_chain.py --variants base,syn,valley --cases bundle,crossing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from shapetensor.field import Orientations
from shapetensor.lift import response_profile
from shapetensor.phantoms import BEND_GRID, Grid, build, direction
from shapetensor.validate import ANGLES

spec = importlib.util.spec_from_file_location("TP", str(Path(__file__).with_name("test_populations.py")))
TP = importlib.util.module_from_spec(spec); spec.loader.exec_module(TP)
DEV = "cuda"

CASES = (("bundle 60", "bundle", {"angle": 60.0}), ("bend", "bend", {}), ("straight", "bend", {"straight": True}),
         ("band crossing 90", "band_crossing", {"cross_deg": 90.0}), ("band crossing 60", "band_crossing", {"cross_deg": 60.0}),
         ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}),
         ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}),
         ("crossing 45", "crossing", {"cross_deg": 45.0}), ("texture", None, {}))

VARIANTS = {
    "base": {},
    "peakpres": {"presence_mode": "peak"},
    "noring": {"ring_rel": 1.0},
    "ring7": {"ring_rel": 0.7},
    "ringlift": {"ring_on": "lift"},
    "norings": {"rings": False},
    "noexc": {"excise": 0.0},
    "m4": {"mean_scale": 0.4},
    "m4noexc": {"mean_scale": 0.4, "excise": 0.0},
    "exc2": {"excise": 2.0},
    "m3": {"mean_scale": 0.3},
    "m5": {"mean_scale": 0.5},
    "mw1": {"min_weight": 0.1},
    "rel2": {"rel": 0.2},
    "share1": {"min_share": 0.1},
    "rel2share1": {"rel": 0.2, "min_share": 0.1},
    "short_uni": {"lags": (0.2, 0.45), "radial": "uniform"},
    "short_cnt": {"lags": (0.2, 0.45), "radial": "count"},
    "def_uni": {"radial": "uniform"},
    "short_uni_acq": {"lags": (0.2, 0.45), "radial": "uniform", "acquisition_null": True},
    "acq": {"acquisition_null": True},
    "short_cnt_acq": {"lags": (0.2, 0.45), "acquisition_null": True},
    "sheet7": {"max_sheet_share": 0.7},
    "ring3": {"ring_rel": 0.3},
    "valley": {"valley": 0.5},
    "valley3": {"valley": 0.3},
    "ang10": {"angular_deg": 10.0},
    "share2": {"min_share": 0.2},
    "k12": {"kernel_deg": 12.0},
    "nomom": {"momentum": False},
}


def synthetic_response(model, tex, texture):
    cg = Grid(); num = 0; cnt = 0
    for k, a in enumerate(ANGLES):
        v = torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float()
        O = model.lift(v); cs = O.shape[1:]
        d = torch.as_tensor(direction(a), device=DEV).float().view(3, 1, 1, 1).expand(3, *cs)
        m = torch.zeros(cs, dtype=torch.bool, device=DEV); zs, ys, xs = cg.inner
        m[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
        n = int(m.sum()); num = num + n * response_profile(O, d, model.grid, m, model.bin_deg); cnt += n
    return num / cnt


def summary(sc):
    """One line per case: the headline numbers."""
    out = []
    for cat, k in (("none (texture)", "N=0"), ("one population", "N=1"), ("two populations", "N=2")):
        if cat in sc:
            r = sc[cat]; s = f"{cat.split()[0]}:{k} {r[k]:.2f}"
            if "top lobe error median" in r:
                s += f" err {r['top lobe error median']}"
            if k == "N=2":
                s += f" (N0 {r['N=0']:.2f} N1 {r['N=1']:.2f})"
            if "pair error median (N>=2)" in r:
                s += f" pair {r['pair error median (N>=2)']} both15 {r['both within 15 (of all two-population cells)']:.2f}"
            out.append(s)
    if "false extension beyond the T" in sc:
        out.append(f"falseT {sc['false extension beyond the T']:.3f}")
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base"); ap.add_argument("--cases", default="")
    ap.add_argument("--response", default="self", choices=("self", "syn"))
    ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/chain")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    tex, texture = TP.P.tex_for("calibrated")
    cases = [c for c in CASES if not a.cases or any(k in c[0] for k in a.cases.split(","))]
    from shapetensor.fibre_sets import References
    refs = References(Grid().core, "calibrated", DEV)
    data = []
    for name, kind, kw in cases:
        if kind is None:
            v, tr, meta = refs.texture, None, {}
            grid = Grid(core=tuple(int(s.stop - s.start) for s in refs.texture_region), margin=tuple(int(s.start) for s in refs.texture_region))
        else:
            img, tr, meta = build(kind, 1, tex=tex, texture=texture, device=DEV, **kw)
            v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        data.append((name, v, tr, meta, grid))
    log("phantoms built")
    results = {}
    for vname in a.variants.split(","):
        model = Orientations(device=DEV, **VARIANTS[vname])
        R = synthetic_response(model, tex, texture) if a.response == "syn" else None
        results[vname] = {}
        for name, v, tr, meta, grid in data:
            t1 = time.time(); pop = model.populations(v, response=R, labels=False); dt = time.time() - t1
            sc = TP.score(pop, tr, grid, meta); results[vname][name] = sc
            log(f"{vname:8s} {name:18s} {dt:4.1f}s floor {pop['floor']:.3f}  {summary(sc)}")
        json.dump(results, open(out / f"chain_{a.response}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
