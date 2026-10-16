"""Stage 2 of the pipeline on the crossing phantoms: separate the populations in the shape space, label-free, then
one tensor per population (stage 3), scored against the truth.

Shape spaces: encoder block 2 (`anatomix_enc2`, 32 channels, the orientation-selective layer) and layer 58
(`anatomix48`, 16 PCs) for comparison. Separations (no k-means, no labels):
  * signatures -- the principal signatures of the window's channel-energy cloud, per-voxel memberships by
    projection, a membership-weighted structure tensor per population (readout.populations); N by a null-calibrated
    eigenvalue ratio;
  * TCA -- the channel-resolved tensor's CP decomposition with over-complete grouping (readout.multi) and with BIC;
  * oracle -- the truth's voxel split, as the ceiling (labels used for evaluation only).
Phantoms: single fibres at 12 angles (the null for N and the calibration), crossings at 90 / 60 / 45 degrees at
1:1 and 60 degrees at 4:1. Small blocks (160 x 96 x 160), sites every 8 voxels on the mid slab, rho 10 and 16. GPU.

    python scripts/bench/separate_shape_space.py      -> results/separate_shape_space.json, figures/tca_field/separate_*.png
"""
import json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor import DATA_ROOT
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.phantoms import make
from shapetensor.readout import common, multi, populations
from shapetensor.tensor import gradients, k_sites
from shapetensor.validate import score

FIGS = Path(__file__).resolve().parents[2] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
CROSS = {"c90": (30.0, 120.0, 0.5), "c60": (30.0, 90.0, 0.5), "c45": (30.0, 75.0, 0.5), "c60-f4": (30.0, 90.0, 0.8)}
RHOS = (10.0, 16.0); SIGMA, TR, STRIDE = 1.0, 4.0, 8
ENGINES = {"anatomix_enc2": {}, "anatomix48": {"n_pcs": 16}}


def build(angles, seed, tex, g):
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cuda")
    masks = []
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    for a, f in angles:                       # each population alone, for the oracle split and the evaluation
        rng2 = np.random.default_rng(seed); sp2 = make.Splat(g, tex, rng2, "cuda")
        for a2, f2 in angles:
            make._add_population(sp2, a2, f2 if a2 == a else 0.0)
        masks.append((sp2.render() > 0.25))
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return img, masks


def unit(a):
    return np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))])           # (z, y, x) in-plane


def angle(d, t):
    return np.degrees(np.arccos(np.clip(np.abs((d * t).sum(-1)), 0, 1)))


def main():
    t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32))
    Z, Y, X = g.shape; y_mid = Y // 2
    zs = np.arange(40, Z - 40, STRIDE); xs = np.arange(40, X - 40, STRIDE)
    centers = torch.as_tensor([(z, y_mid, x) for z in zs for x in xs], device="cuda")
    res = {}
    for ename, ekw in ENGINES.items():
        eng = make_engine(ename, nyquist="none", **ekw)
        res[ename] = {}
        for rho in RHOS:
            r = {}
            # --- the null: single fibres at 12 angles -> eigenvalue ratios, and the single-direction error
            ratios, errs = [], []
            for k, a in enumerate(ANG):
                img, masks = build([(a, 1.0)], 100 + k, tex, g)
                v = torch.as_tensor(img, device="cuda").float()
                F = cached_features(eng, v); G = gradients(v[None], SIGMA, TR)
                s = populations.separate_sites(F, G, centers, rho)
                ratios.append((s["lam"][:, 1] / s["lam"][:, 0].clamp_min(1e-12)).cpu().numpy())
                errs.append(angle(s["dirs"][:, 0].cpu().numpy(), unit(a)))
                del F, G
            ratios = np.concatenate(ratios); errs = np.concatenate(errs)
            thr = float(np.percentile(ratios, 99))
            r["null"] = {"ratio_thr_99": round(thr, 4), "ratio_median": round(float(np.median(ratios)), 4), "single_dir_err_median": round(float(np.median(errs)), 2)}
            print(f"[{time.time() - t0:4.0f}s] {ename} rho {rho:g} null: ratio thr {thr:.3f} (median {np.median(ratios):.3f}), single error {np.median(errs):.1f} deg", flush=True)
            # --- the crossings
            for cname, (a1, a2, f1) in CROSS.items():
                img, masks = build([(a1, f1), (a2, 1 - f1)], 300, tex, g)
                v = torch.as_tensor(img, device="cuda").float()
                F = cached_features(eng, v); G = gradients(v[None], SIGMA, TR)
                t1, t2 = unit(a1), unit(a2)
                out = {}
                # signatures
                s = populations.separate_sites(F, G, centers, rho)
                n = populations.count_from_eigenvalues(s["lam"], thr).cpu().numpy()
                d = s["dirs"].cpu().numpy()
                e11, e12 = angle(d[:, 0], t1), angle(d[:, 0], t2); e21, e22 = angle(d[:, 1], t1), angle(d[:, 1], t2)
                both = np.minimum(np.maximum(e11, e22), np.maximum(e12, e21))              # best assignment of the two directions
                out["signatures"] = {"found2_share": round(float((n >= 2).mean()), 3), "both_within_15_share": round(float(((n >= 2) & (both <= 15)).mean()), 3),
                                     "dir_err_median_best_pair": round(float(np.median(both)), 1), "first_dir_err_to_nearest": round(float(np.median(np.minimum(e11, e12))), 1)}
                # TCA on the channel-resolved tensor at the same sites
                b = max(1, int(rho // 4)); K, sites = k_sites(gradients(F.float(), SIGMA, TR), rho, TR, b, [y_mid], ((40, Z - 40), (40, X - 40)))
                step = max(1, STRIDE // b)
                Kp = K[0, ::step, ::step].reshape(-1, K.shape[-1]); Kt = common.unpack(Kp, common.nc3_from_packed(Kp.shape[-1]))
                for mname, fn in (("tca_overcomplete", lambda: multi.overcomplete_group(Kt)), ("tca_bic", lambda: multi.n_by_bic(Kt))):
                    try:
                        m = fn(); fib = m["fibres"].cpu().numpy(); cnt = m["count"].cpu().numpy()
                        tt = score.multi_scores(fib, m["fracs"].cpu().numpy() if "fracs" in m else np.ones(fib.shape[:2]) / fib.shape[1], cnt,
                                                np.tile(t1, (len(fib), 1)), np.tile(t2, (len(fib), 1)), np.full(len(fib), 2), np.full(len(fib), f1))
                        two = tt.get("two") or {}
                        out[mname] = {"found2_share": round(float(two.get("found2_share", float("nan"))), 3), "both_within_15_share": round(float(two.get("both_within_15_share", float("nan"))), 3)}
                    except Exception as ex:
                        out[mname] = {"error": str(ex)[:80]}
                # oracle: the truth split
                m1, m2 = masks[0] & ~masks[1], masks[1] & ~masks[0]
                errs_o = []
                for c in centers:
                    half = int(TR * rho + 0.5); zz, yy, xx = populations.window_indices(c, half, F.shape[1:])
                    gg = G[:, zz][:, :, yy][:, :, :, xx].reshape(3, -1)
                    ds = []
                    for mk in (m1, m2):
                        mm = mk[zz][:, yy][:, :, xx].reshape(-1).float()
                        S = (gg * mm) @ gg.T; _, V = common.eigh(S[None] / mm.sum().clamp_min(1)); ds.append(V[0][:, 0].cpu().numpy())
                    errs_o.append([angle(ds[0], t1), angle(ds[1], t2)])
                errs_o = np.array(errs_o)
                out["oracle_split"] = {"both_within_15_share": round(float((errs_o.max(1) <= 15).mean()), 3), "dir_err_median": round(float(np.median(errs_o)), 1)}
                r[cname] = out
                print(f"[{time.time() - t0:4.0f}s] {ename} rho {rho:g} {cname}: " + " | ".join(f"{k}: {v}" for k, v in out.items()), flush=True)
                del F, G, K, Kp, Kt; torch.cuda.empty_cache()
            res[ename][f"rho{rho:g}"] = r
            json.dump(res, open(f"{DATA_ROOT}/results/separate_shape_space.json", "w"), indent=1)
    print("SEPARATE DONE", flush=True)


if __name__ == "__main__":
    main()
