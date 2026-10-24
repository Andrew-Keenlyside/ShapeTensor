"""Where the generator, the targets and the models still fail: targeted probes drawn with the current generator, scored
per lattice cell against the training targets (analysis only; nothing is trained here).

    python scripts/learn/gap_probe.py --out DIR --models A.pt B.pt [--n 24] [--groups morph,res,cross,artefact,iso,limits]

Per condition (n crops of 64^3 from a 128^3 canvas, one scene each) and model: the single-population axis error and the
share of cells with no predicted peak; at two-population cells (target peaks > 30 deg apart) the detection of the second
peak and the pair's error; at three-population cells the detection of a third; at isotropic cells (live, no target
peak) the false-peak rate; the unexplained energy of order 2. Per condition also the targets' own orientation
visibility (occupancy-weighted, from the draw's record) -- low there means the generator hides it, not the model.
"limits" (no model): the smallest crossing angle the targets themselves resolve (two line populations in a cell, the
true angle between their axes from the drawn tensors, against the target's peak count).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).parent))
import capability_bank as CB  # noqa: E402
from shapetensor.learn import generator as G  # noqa: E402
from shapetensor.learn import scene as SC  # noqa: E402
from shapetensor.learn import train as TR  # noqa: E402
from shapetensor.learn.lobe_train import predict_sh, sh_peaks  # noqa: E402
from shapetensor.learn.losses import _cells  # noqa: E402
from shapetensor.learn.sh import n_coef  # noqa: E402
from shapetensor.learn.targets import _pool, target  # noqa: E402

DEV = "cuda"; LMAX = 8; NSH = n_coef(LMAX); MARGIN = 2; t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
ONLY = lambda *names: {k: (1.0 if k in names else 0.0) for k in SC.Scene.FAMILIES}


def make(seed, strength=0.5, canvas=128, **scene):
    """One pair as training makes it (64^3, exact SH lattice target), plus the drawn sample for the probes."""
    kw = {"canvas": canvas, "crops_per_scene": 1, **scene}
    s, d = SC.draw_scene((64,) * 3, seed=seed, device=DEV, strength=strength, **kw)
    from shapetensor.learn.query import voxel_density
    F = voxel_density(s, d, LMAX, desc_lmax=4, desc_key="descriptors5")
    t = {"sh_t": _pool(F, 5.0, 4).float(), "voxel": torch.tensor(math.log2(d["voxel_um"]) if d.get("voxel_um") else float("nan"), device=DEV)}
    rec = d["record"]; ov = rec.get("orientation_visibility") or []
    occ = [float(d["occupancy"][k].mean()) for k in range(len(ov))]
    vis = float(np.average(ov, weights=np.array(occ) + 1e-9)) if ov else float("nan")
    return d["image"].float() / 255, t, s, d, vis


def cells_eval(m, x, t):
    v = t["voxel"].view(1) if torch.isfinite(t["voxel"]) else None
    f = predict_sh(m, x[None, None], voxel=v, lmax=LMAX)[0]
    f = _cells(f, MARGIN)[0]
    if f.shape[0] < NSH:
        f = torch.cat([f, torch.zeros(NSH - f.shape[0], *f.shape[1:], device=DEV)])
    ct = _cells(t["sh_t"][None, :NSH], MARGIN)[0]
    return f.flatten(1), ct.flatten(1)


def score(P, T):
    """Per-cell scores from predicted and target SH cells (concatenated over crops)."""
    ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))
    Dt, Vt = sh_peaks(T, LMAX); Dp, Vp = sh_peaks(P, LMAX)
    mass = T[0] / math.sqrt(4 * math.pi); live = mass > 0.05
    nt, npd = (Vt > 0).sum(0), (Vp > 0).sum(0); out = {"cells_live": int(live.sum())}
    one = live & (nt == 1)
    if bool(one.any()):
        hit = one & (Vp[0] > 0); out["single_deg"] = float(ang(Dp[0][:, hit], Dt[0][:, hit]).median()) if bool(hit.any()) else float("nan")
        out["single_miss"] = float((~(Vp[0] > 0))[one].float().mean()); out["false_second"] = float((npd[one] >= 2).float().mean()); out["n_single"] = int(one.sum())
    sep = ang(Dt[0], Dt[1]); two = live & (nt == 2) & (sep > 30)
    if bool(two.any()):
        det = two & (npd >= 2); out["second_detected"] = float(det.sum() / two.sum()); out["n_two"] = int(two.sum())
        if bool(det.any()):
            e = torch.minimum((ang(Dp[0][:, det], Dt[0][:, det]) + ang(Dp[1][:, det], Dt[1][:, det])) / 2, (ang(Dp[0][:, det], Dt[1][:, det]) + ang(Dp[1][:, det], Dt[0][:, det])) / 2)
            out["pair_deg"] = float(e.median())
        for lo, hi in ((30, 45), (45, 60), (60, 90.1)):
            b = two & (sep >= lo) & (sep < hi)
            if int(b.sum()) >= 20:
                out[f"second_detected_{lo}_{int(hi)}"] = float((b & (npd >= 2)).sum() / b.sum())
    three = live & (nt == 3)
    if int(three.sum()) >= 10:
        out["third_detected"] = float((three & (npd >= 3)).sum() / three.sum()); out["n_three"] = int(three.sum())
    iso = live & (nt == 0)
    if int(iso.sum()) >= 20:
        out["false_peak_iso"] = float((iso & (npd >= 1)).float().sum() / iso.sum()); out["n_iso"] = int(iso.sum())
    c = slice(1, 6); out["unexplained_l2"] = float(((P[c] - T[c]) ** 2).sum() / (T[c] ** 2).sum().clamp_min(1e-9))
    out["target_peaks_share"] = {str(k): round(float(((nt == k) & live).sum() / live.sum().clamp_min(1)), 3) for k in range(4)}
    return out


def run_condition(models, n, seed0, strength=0.5, transform=None, **scene):
    data, vis, recs, fails = [], [], [], 0
    sd = seed0
    while len(data) < n and sd < seed0 + 4 * n:
        try:
            x, t, s, d, v = make(sd, strength, **scene)
        except Exception as e:                                                                   # a generator failure is a finding too
            fails += 1; recs.append(repr(e)[:160]); sd += 1; continue
        sd += 1
        if float(t["sh_t"][0].mean()) < 0.01 * math.sqrt(4 * math.pi):
            continue
        if transform is not None:
            x = transform(x, np.random.default_rng(sd))
        data.append((x, t)); vis.append(v)
    res = {"n": len(data), "target_orientation_visibility": float(np.nanmean(vis)) if vis else float("nan"), "generator_failures": fails}
    if fails:
        res["failure_examples"] = recs[:3]
    for name, m in models.items():
        Ps, Ts = [], []
        for x, t in data:
            with torch.no_grad():
                p, ct = cells_eval(m, x, t)
            Ps.append(p); Ts.append(ct)
        if Ps:
            res[name] = score(torch.cat(Ps, 1), torch.cat(Ts, 1))
    return res


def limits(n=24):
    """The smallest crossing angle the targets resolve: per lattice cell with two line populations (occupancy > 0.1 each
    in the cell), the true angle between their mean axes against the number of target peaks."""
    rows = []
    for i in range(20 * n):
        try:
            x, t, s, d, v = make(77_000_000 + i, 0.2, families=ONLY("bundle"), n_groups=(2, 3), underlay_p=0.0, voxel_um=3.0, p_grain=0.0, p_sparse=0.0, p_cutout=0.0,
                                 canvas=64)                                                      # crop = canvas: the crossing points are in view
        except Exception:
            continue
        ct = t["sh_t"][:NSH]; Dt, Vt = sh_peaks(ct.flatten(1), LMAX); nt = (Vt > 0).sum(0)
        lines = [k for k, L in enumerate(s.layers) if getattr(L, "kind", "") == "line" and hasattr(L, "desc5")]
        if len(lines) < 2:
            continue
        occ = Fn.avg_pool3d(d["occupancy"][lines][None].float(), 4)[0]                                  # (K, 16, 16, 16)
        T6 = torch.stack([Fn.avg_pool3d((d["tensors"][k] * d["occupancy"][k][None])[None].float(), 4)[0] for k in lines])
        M = torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], 1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], 1), torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], 1)], 1)
        M = M.permute(0, 3, 4, 5, 1, 2); ax = torch.linalg.eigh(M)[1][..., :, -1]                       # (K, 16, 16, 16, 3) top axis
        both = (occ > 0.1).sum(0) == 2
        if not bool(both.any()):
            continue
        idx = torch.nonzero(both)
        for z, y, x_ in idx.tolist():
            ks = [k for k in range(len(lines)) if float(occ[k, z, y, x_]) > 0.1][:2]
            a = float(torch.rad2deg(torch.acos((ax[ks[0], z, y, x_] * ax[ks[1], z, y, x_]).sum().abs().clamp(max=1))))
            rows.append((a, int(nt.view(16, 16, 16)[z, y, x_])))
        if len(rows) > 6000:
            break
    rows = np.array(rows); out = {"cells": int(len(rows))}
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 75), (75, 90.1)):
        b = (rows[:, 0] >= lo) & (rows[:, 0] < hi) if len(rows) else np.zeros(0, bool)
        if b.sum() >= 15:
            out[f"{lo}-{int(hi)}deg"] = {"n": int(b.sum()), "target_two_peaks": round(float((rows[b, 1] >= 2).mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--groups", default="morph,res,cross,artefact,iso,limits")
    a = ap.parse_args(); od = Path(a.out); od.mkdir(parents=True, exist_ok=True); groups = a.groups.split(",")
    CB._set_lobe_targets()
    models = {}
    for p in a.models:
        m, _ = CB.load_model(p); models[Path(p).parent.name] = m
    res_path = od / "probe_results.json"; R = json.load(open(res_path)) if res_path.exists() else {}
    def put(key, val):
        R[key] = val; json.dump(R, open(res_path, "w"), indent=1)
        line = " | ".join(f"{mn}: " + ", ".join(f"{k} {v:.2f}" for k, v in val.get(mn, {}).items() if isinstance(v, float)) for mn in models)
        log(f"{key}: n {val['n']} vis {val['target_orientation_visibility']:.2f} fails {val['generator_failures']} || {line}")
    n = a.n
    if "morph" in groups:
        FAM_V = {"bundle": 3, "tree": 20, "network": 2, "helix": 3, "sheet": 5, "foam": 20, "porous": 20, "cells": 2, "crystals": 3, "radial": 4,
                 "plywood": 1, "fascicles": 10, "trabecular": 30, "tubules": 3, "epithelium": 1.5, "dendrite": 0.5, "gland": 5, "neuron": 1,
                 "neuropil": 0.5, "organelles": 0.02, "compartments": 0.02, "myelin": 0.08, "nuclei": 0.1, "myofibrils": 0.05, "alveoli": 8,
                 "jigsaw": 1, "honeycomb": 2, "xylem": 4, "lattice": 1, "organs": 800, "granules": 1}
        for k, (f, v) in enumerate(FAM_V.items()):
            if f"morph:{f}" in R:
                continue
            put(f"morph:{f}", run_condition(models, max(12, n // 2), 80_000_000 + 1000 * k, 0.5, families=ONLY(f), n_groups=(1, 1), underlay_p=0.0, voxel_um=v,
                                            layout_p=1.0 if f == "radial" else 0.3))
    if "res" in groups:
        for k, v in enumerate((0.005, 0.05, 0.5, 2.0, 8.0, 32.0, 128.0, 500.0)):
            if f"res:voxel_{v}" not in R:
                put(f"res:voxel_{v}", run_condition(models, n, 81_000_000 + 1000 * k, 0.5, voxel_um=v))
        for k, v in enumerate((0.5, 2.0, 8.0, 32.0)):                                                  # bundles alone: resolved -> unresolved fibres
            if f"res:bundle_{v}" not in R:
                put(f"res:bundle_{v}", run_condition(models, n, 81_500_000 + 1000 * k, 0.5, families=ONLY("bundle"), n_groups=(1, 2), underlay_p=0.0, voxel_um=v))
    if "cross" in groups:
        for k, (ng, dens) in enumerate((((2, 2), "dense"), ((3, 3), "dense"), ((2, 3), "sparse"), ((4, 5), "mixed"))):
            key = f"cross:{ng[0]}-{ng[1]}pop_{dens}"
            if key in R:
                continue
            fam = ONLY("bundle") if dens != "mixed" else {**ONLY("bundle", "tree", "sheet", "cells"), "bundle": 3.0}
            put(key, run_condition(models, n, 82_000_000 + 1000 * k, 0.4, families=fam, n_groups=ng, underlay_p=0.0, voxel_um=3.0,
                                   p_sparse=1.0 if dens == "sparse" else 0.0, canvas=64))         # crop = canvas: crossings in view
    if "artefact" in groups:
        G.CROSS_MIX = 0.0; G.CROSS_KEY = {}
        keys = ["none", "slabs", "missing", "drift", "chatter", "lsstripes", "depth", "lowdose", "wedge", "ctf", "saturate", "misalign", "rings",
                "streaks", "thick", "aniso_psf", "poisson", "rician", "gibbs", "ghosting", "stain"]
        base = dict(families=None, n_groups=(1, 3), voxel_um=3.0, physics="additive", p_sparse=0.0, p_cutout=0.0, p_boundary=0.0)
        for k, key in enumerate(keys):
            if f"artefact:{key}" in R:
                continue
            prof = {kk: 0.0 for kk in G.MODALITIES["hipct"]}
            if key != "none":
                prof[key] = 1.0
            G.MODALITIES["_probe"] = prof
            put(f"artefact:{key}", run_condition(models, n, 83_000_000, 1.0, modality="_probe", **base))
        del G.MODALITIES["_probe"]
        # the CT scan of the content, cut-out tissue, sparse labelling (draw-time routes), batch-time k-space routes
        orig = SC._extra_acquisition
        def forced(img, layers, rng, modality, rec, strength=1.0):
            from shapetensor.learn.ctsim import scan
            return orig(scan(img, rng, rec, 1.0), layers, rng, modality, rec, strength)
        SC._extra_acquisition = forced
        if "artefact:ct_scan" not in R:
            put("artefact:ct_scan", run_condition(models, n, 83_000_000, 1.0, modality="ct", **base))
        SC._extra_acquisition = orig
        if "artefact:cutout" not in R:
            put("artefact:cutout", run_condition(models, n, 83_000_000, 0.5, **{**base, "p_cutout": 1.0}))
        if "artefact:sparse" not in R:
            put("artefact:sparse", run_condition(models, n, 83_000_000, 0.5, **{**base, "p_sparse": 1.0}))
        from shapetensor.learn import augment as AU
        for key, fn in (("aug_motion", AU.kspace_motion), ("aug_ghosting", AU.kspace_ghosting), ("aug_resolution", AU.per_axis_resolution),
                        ("aug_slices", getattr(AU, "missing_slices", None)), ("aug_lines", AU.missing_lines), ("aug_blocks", AU.missing_blocks)):
            if fn is None or f"artefact:{key}" in R:
                continue
            tf = lambda x, rng, fn=fn: ((lambda y: (y - y.min()) / (y.max() - y.min()).clamp_min(1e-6))(fn(x, rng, {}).float()))
            put(f"artefact:{key}", run_condition(models, n, 83_000_000, 0.5, transform=tf, **base))
    if "iso" in groups:
        G.CROSS_MIX = 0.08
        for key, kw in (("granules", dict(families=ONLY("granules"), n_groups=(1, 2), voxel_um=1.0)),
                        ("granules_wedge", None), ("granules_thick", None),
                        ("texture_only", dict(families=ONLY("granules"), n_groups=(1, 1), voxel_um=200.0, p_grain=1.0)),
                        ("cells_round", dict(families=ONLY("cells"), n_groups=(1, 1), voxel_um=1.5))):
            if f"iso:{key}" in R:
                continue
            if kw is None:                                                                           # anisotropic blur on isotropic content
                G.CROSS_MIX = 0.0; G.CROSS_KEY = {}
                prof = {kk: 0.0 for kk in G.MODALITIES["hipct"]}; prof["wedge" if key.endswith("wedge") else "thick"] = 1.0; G.MODALITIES["_probe"] = prof
                put(f"iso:{key}", run_condition(models, n, 84_000_000, 1.0, families=ONLY("granules"), n_groups=(1, 2), voxel_um=1.0, modality="_probe", physics="additive"))
                del G.MODALITIES["_probe"]; G.CROSS_MIX = 0.08
            else:
                put(f"iso:{key}", run_condition(models, n, 84_000_000, 0.5, **kw))
    if "limits" in groups and "limits:target_crossing_resolution" not in R:
        R["limits:target_crossing_resolution"] = limits(n); json.dump(R, open(res_path, "w"), indent=1)
        log(f"limits: {json.dumps(R['limits:target_crossing_resolution'])}")
    log(f"DONE {res_path}")


if __name__ == "__main__":
    main()
