"""Null-calibrated CSD peak threshold: per (engine, rho, texture level) the smallest relative threshold whose false
second-population rate on the single-population phantom is <= 2%, then the crossings scored at that threshold.
Reads the cached ODFs of tag `main` (both seeds where present)."""
import json, time
import numpy as np, torch
from shapetensor.validate import suite, score
from shapetensor.odf import csd, peaks, fit_ring_width, ring_profile, zonal_kernel
from shapetensor.readout import common

tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
spec = suite.Spec(pids=(), tex=tex, tag="main")
LMAX, RELS, STRIDE = 8, (0.3, 0.4, 0.5, 0.6, 0.7, 0.8), 2
t0 = time.time()


def load_odf(pid, seed, engine, rho):
    p = suite.cache_path(spec, pid, seed, engine, rho)
    if not p.exists():
        return None, None
    c = suite.load_cache(spec, pid, seed, engine, rho)
    O = c["odf"][::STRIDE, ::STRIDE]
    _, tr, _, _ = suite.make.load(pid, seed)
    t = score.sample_truth(tr, c["z"][::STRIDE], c["x"][::STRIDE])
    return torch.as_tensor(O.reshape(-1, O.shape[-1]), device="cuda"), t


out = {}
for tsuf, single, crossings in (("", "P4-lines-single", ["P4-lines-c90-f1", "P4-lines-c60-f1", "P4-lines-c45-f1", "P4-lines-c60-f4"]),
                                ("-t0.5", "P4-lines-single-t0.5", ["P4-lines-c90-f1-t0.5", "P4-lines-c60-f1-t0.5"])):
    for engine in ("intensity", "anatomix16", "anatomix48"):
        for rho in (16.0, 32.0):
            fs = {}
            for seed in (0, 1):
                O, t = load_odf(single, seed, engine, rho)
                if O is None:
                    continue
                width = fit_ring_width(O, torch.as_tensor(t["dir1"], device="cuda"), LMAX)
                kern = zonal_kernel(LMAX, ring_profile(width))
                f = csd(O, LMAX, kern)
                for rel in RELS:
                    _, _, cnt = peaks(f, LMAX, nmax=3, rel=rel, min_sep_deg=25.0)
                    fs.setdefault(rel, []).append(float((cnt.cpu().numpy() >= 2).mean()))
            if not fs:
                continue
            false2 = {rel: float(np.mean(v)) for rel, v in fs.items()}
            rel_star = next((rel for rel in RELS if false2[rel] <= 0.02), RELS[-1])
            key = f"{engine}|rho{rho:g}|t{tsuf or '1'}"
            out[key] = {"ring_width_deg": float(np.degrees(width)), "false2_by_rel": false2, "rel_star": rel_star, "crossings": {}}
            for pid in crossings:
                res = []
                for seed in (0, 1):
                    O, t = load_odf(pid, seed, engine, rho)
                    if O is None:
                        continue
                    f = csd(O, LMAX, kern)
                    d, a, cnt = peaks(f, LMAX, nmax=3, rel=rel_star, min_sep_deg=25.0)
                    frac = a / a.sum(1, keepdim=True).clamp_min(1e-30)
                    sc = score.multi_scores(common.canonical_sign(d).cpu().numpy(), frac.cpu().numpy(), cnt.cpu().numpy(),
                                            t["dir1"], t["dir2"], t["npop"], t["frac1"], t.get("cross_deg"), t["region"])
                    res.append(sc["two"])
                if res:
                    m = {k: float(np.mean([r[k] for r in res])) for k in ("found2_share", "both_within_15_share")}
                    m["err_median"] = float(np.nanmean([(r["err1"].get("median", np.nan) + r["err2"].get("median", np.nan)) / 2 for r in res]))
                    m["frac_abs_err"] = float(np.nanmean([r["frac_abs_err"].get("median", np.nan) for r in res]))
                    out[key]["crossings"][pid] = m
            print(f"[{time.time() - t0:4.0f}s] {key:26s} width {np.degrees(width):4.1f} rel* {rel_star} (false2 @0.3 {false2[0.3]:.0%}) | "
                  + " ".join(f"{pid.replace('P4-lines-', '')}: {m['found2_share']:.0%}/{m['both_within_15_share']:.0%} e{m['err_median']:.0f}" for pid, m in out[key]["crossings"].items()), flush=True)
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/csd_calibrated.json", "w"), indent=1)
print("CSD CALIB DONE", flush=True)
